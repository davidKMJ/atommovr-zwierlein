#!/usr/bin/env python3
"""
atommovr Production Controller
================================
Orchestrates the complete atom-rearrangement feedback loop:

  1. Sync camera → occupancy (``Camera.sync`` / ``detect_occupancy``)
  2. Check whether target is already filled  →  done
  3. Compute rearrangement moves (configurable algorithm)
  4. Convert moves to RF ramps  (RFConverter / AWGBatch)
  5. Write ramps to the Spectrum Instrumentation AWG card
  6. Offline: advance physics on the controller ``AtomArray``
  7. Repeat from step 1

Two interchangeable hardware backends (``backend=`` on ``atommovrController``):
  - ``"scapp"`` (default) — GPU-direct RDMA generation
    (``awg_controller.src.scapp.ScappFeeder``). No fixed tone-count
    ceiling; ``strategy=`` is not applicable.
  - ``"legacy_dds"`` — the original DDS core-register approach
    (``awg_controller.src.dds_strategies``), selectable via ``strategy=``.

Both backends open a single card (multi-card support has been removed) and
share the same 40 % total-amplitude-per-channel safety budget.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

#  repo root on sys.path
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

#  optional hardware driver.
#  Broadened beyond ImportError: an installed-but-driverless `spcm` package
#  raises a bare Exception from spcm_core/pyspcm.py when the vendor driver
#  .so isn't found, not ImportError.
try:
    import spcm
    from spcm import SpcmException

    _HW_AVAILABLE = True
except Exception:
    spcm = None  # type: ignore[assignment]
    SpcmException = Exception  # type: ignore[assignment,misc]
    _HW_AVAILABLE = False

from atommovr.algorithms.single_species import (
    BCv2,
    BalanceAndCompact,
    GeneralizedBalance,
    Hungarian,
    ParallelHungarian,
    ParallelLBAP,
    PCFA,
    Tetris,
)
from atommovr.utils.AtomArray import AtomArray
from atommovr.utils.ErrorModel import ErrorModel
from atommovr.utils.core import Configurations, PhysicalParams
from atommovr.utils.errormodels import ZeroNoise
from awg_controller.src.awg_control import (
    AODSettings,
    AWGBatch,
    RFConverter,
    validate_hardware_limits,
)
from awg_controller.src.camera import Camera, OfflineArrayCamera, RealArrayCamera
from awg_controller.src.dds_strategies import (
    DDSStrategy,
    DDSStreamingStrategy,
    STRATEGY_REGISTRY,
    get_strategy,
)
from awg_controller.src.scapp import (
    ScappFeeder,
    ScappFeederConfig,
    _GPU_AVAILABLE as _SCAPP_GPU_AVAILABLE,
)
from awg_controller.src.session_recorder import SessionRecorder, moves_to_records

#  logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("atommovr_controller.log"),
    ],
)
log = logging.getLogger(__name__)

@dataclass
class HardwareConfig:
    """Spectrum Instrumentation card configuration (mirrors cli.py defaults).

    A single card is opened regardless of backend — multi-card support has
    been removed (it only ever broadcast identical commands to every card,
    with no real per-card partitioning).
    """

    #: Device path, e.g. "/dev/spcm0"
    card_path: str = "/dev/spcm0"

    #: Output amplitude - manufacturer maximum is 1.6 V into 50 Ω. Hard
    #: safety ceiling (both backends): must never exceed 2.0 V.
    max_amplitude_v: float = 1.6

    #: Output impedance
    output_load_ohms: float = 50.0

    #: legacy_dds only: idle/holding TIMER interval (s). Set to the expected
    #: move duration so the card fires once per rearrangement batch. cli.py
    #: default was 0.2 s. Unused by the scapp backend (continuous streaming
    #: has no discrete TIMER trigger concept).
    trigger_timer_s: float = 0.2

    #: scapp only: GPU buffer fill block size (samples). See
    #: ``ScappFeederConfig.notify_samples``.
    notify_samples: int = 512 * 1024

    #: scapp only: total RDMA-pinned DMA buffer size (samples). See
    #: ``ScappFeederConfig.dma_buffer_samples``.
    dma_buffer_samples: int = 32 * 1024 * 1024

    #: scapp only: card.start() fires once the on-board buffer fill level
    #: crosses this (per-mille). See
    #: ``ScappFeederConfig.fill_start_threshold_promille``.
    fill_start_threshold_promille: int = 800

    #: legacy_dds only: number of pre-fill writes during initialisation
    #: (prevents underruns). cli.py uses floor(10 / trg_timer); we keep
    #: that formula.
    @property
    def prefill_count(self) -> int:
        return max(1, math.floor(10.0 / self.trigger_timer_s))


@dataclass
class SoftwareConfig:
    """Algorithm and imaging configuration."""

    #  Imaging
    #: Optional OpenCV SimpleBlobDetector_Params (or legacy dict — ignored by Camera).
    blob_params: Any = None

    #  Trap geometry (legacy square helpers; AODSettings is authoritative for RF)
    grid_size: int = 10
    target_size: int = 6

    #: AOD frequency-range and geometry.
    aod_settings: AODSettings = field(default_factory=AODSettings)

    #: Physical parameters (AOD speed, site spacing, loading probability …).
    physical_params: PhysicalParams = field(default_factory=PhysicalParams)

    #: Error model applied on the controller-owned ``AtomArray`` (offline physics).
    error_model: Optional[ErrorModel] = None

    #  Control loop
    max_rounds: int = 10
    algorithm_name: str = "PCFA"
    target_type: Configurations = Configurations.MIDDLE_FILL


_ALGORITHM_REGISTRY = {
    "PCFA": PCFA,
    "Hungarian": Hungarian,
    "Tetris": Tetris,
    "BalanceAndCompact": BalanceAndCompact,
    "BCv2": BCv2,
    "ParallelLBAP": ParallelLBAP,
    "ParallelHungarian": ParallelHungarian,
    "GeneralizedBalance": GeneralizedBalance,
}


class atommovrController:
    """End-to-end atom rearrangement controller.

    Parameters
    ----------
    sw_config : SoftwareConfig
        Algorithm, imaging and geometry settings.
    hw_config : HardwareConfig
        Spectrum Instrumentation card settings.
    camera : Camera, optional
        Shared ``Camera`` interface (``OfflineArrayCamera`` or
        ``RealArrayCamera``). Preferred over ``camera_fn``.
    camera_fn : callable, optional
        Legacy ``() -> np.ndarray`` grabber; wrapped in ``RealArrayCamera``.
        Mutually exclusive with ``camera``.
    recorder : SessionRecorder, optional
        Stage dumps + round JSONL logger. Owned by the controller and passed
        into ``Camera.sync`` / ``log_round`` each loop.
    backend : str, optional
        AWG generation backend: ``"scapp"`` (default, GPU-direct RDMA) or
        ``"legacy_dds"`` (DDS core registers).
    strategy : DDSStrategy or str, optional
        DDS execution strategy (``"streaming"``, ``"ramp"``, ``"pattern"``,
        ``"camera_triggered"``). Only valid when ``backend="legacy_dds"``;
        defaults to ``DDSStreamingStrategy`` in that case.
    scapp_config : ScappFeederConfig, optional
        Tuning knobs for the SCAPP feeder. Only used when
        ``backend="scapp"``.
    """

    def __init__(
        self,
        sw_config: SoftwareConfig,
        hw_config: HardwareConfig,
        camera: Optional[Camera] = None,
        camera_fn: Optional[Callable[[], np.ndarray]] = None,
        recorder: Optional[SessionRecorder] = None,
        backend: str = "scapp",
        strategy: Optional[Union[DDSStrategy, str]] = None,
        scapp_config: Optional[ScappFeederConfig] = None,
    ) -> None:
        if camera is not None and camera_fn is not None:
            raise ValueError("Pass camera= or camera_fn=, not both.")
        if backend not in ("scapp", "legacy_dds"):
            raise ValueError(
                f"Unknown backend {backend!r}; expected 'scapp' or 'legacy_dds'."
            )
        if backend == "scapp" and strategy is not None:
            raise ValueError("strategy= is only valid when backend='legacy_dds'.")

        self.backend = backend
        self.sw = sw_config
        self.hw = hw_config
        self.recorder = recorder

        aod = sw_config.aod_settings
        self.grid_shape: Tuple[int, int] = (int(aod.grid_rows), int(aod.grid_cols))

        if camera is not None:
            if tuple(camera.grid_shape) != self.grid_shape:
                raise ValueError(
                    f"camera.grid_shape {camera.grid_shape} != "
                    f"AODSettings lattice {self.grid_shape}"
                )
            self.camera: Optional[Camera] = camera
        elif camera_fn is not None:
            self.camera = RealArrayCamera(
                self.grid_shape,
                camera_fn=camera_fn,
                blob_params=sw_config.blob_params,
            )
        else:
            # Default closed-loop sim camera when nothing is attached.
            self.camera = OfflineArrayCamera(
                self.grid_shape,
                physical_params=sw_config.physical_params,
                blob_params=sw_config.blob_params,
            )

        if self.backend == "legacy_dds":
            if strategy is None:
                self.strategy: Optional[DDSStrategy] = DDSStreamingStrategy()
            elif isinstance(strategy, str):
                self.strategy = get_strategy(strategy)
            else:
                self.strategy = strategy
            self._scapp_config: Optional[ScappFeederConfig] = None
        else:
            self.strategy = None
            self._scapp_config = scapp_config or ScappFeederConfig(
                ramp_shape=aod.ramp_shape,
                notify_samples=hw_config.notify_samples,
                dma_buffer_samples=hw_config.dma_buffer_samples,
                fill_start_threshold_promille=hw_config.fill_start_threshold_promille,
            )

        self.algorithm = _ALGORITHM_REGISTRY[sw_config.algorithm_name]()
        self.rf_converter = RFConverter(
            aod, sw_config.physical_params, backend=self.backend
        )

        err = (
            sw_config.error_model if sw_config.error_model is not None else ZeroNoise()
        )
        self.array = AtomArray(
            list(self.grid_shape),
            n_species=1,
            params=sw_config.physical_params,
            error_model=err,
        )

        self._card: Any = None
        self._dds: Dict[int, Any] = {}
        self._feeder: Optional[ScappFeeder] = None
        self._grid_rotation: float = 0.0
        self._target_mask: Optional[np.ndarray] = None
        self._apply_target()

        self._initialize_hardware()

    # ------------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------------

    def _initialize_hardware(self) -> None:
        """Open the card and start the active backend (SCAPP feeder or
        legacy DDS strategy)."""
        if self.backend == "legacy_dds":
            self._initialize_legacy_dds()
        else:
            self._initialize_scapp()

    def _initialize_legacy_dds(self) -> None:
        """Open the card, configure DDS, pre-fill buffer, enable trigger."""
        if not _HW_AVAILABLE:
            log.info(
                f"Simulation mode: hardware init skipped "
                f"(backend=legacy_dds, strategy={self.strategy.name})."
            )
            return

        validate_hardware_limits(
            self.sw.aod_settings.grid_rows,
            self.sw.aod_settings.grid_cols,
        )

        try:
            self._card = spcm.Card(self.hw.card_path)
            holding = self.rf_converter.holding_config()
            core_map = self.rf_converter.core_map

            sn = self._card.sn()
            self._card.card_mode(spcm.SPC_REP_STD_DDS)
            channels = spcm.Channels(
                card=self._card,
                card_enable=spcm.CHANNEL0 | spcm.CHANNEL1,
            )
            channels.enable(True)
            channels.amp(self.hw.max_amplitude_v * spcm.units.V)
            channels.output_load(self.hw.output_load_ohms * spcm.units.ohm)
            self._card.write_setup()

            dds = self.strategy.create_dds(self._card, channels)
            self.strategy.configure(dds, self._card, self.hw, core_map)
            self.strategy.prefill(dds, holding, self.hw)
            self._dds[sn] = dds

            self.strategy.start(self._card)
            log.info(
                f"Hardware initialised (backend=legacy_dds, strategy={self.strategy.name})."
            )

        except SpcmException as exc:
            log.error(f"spcm error during init: {exc}")
            self._close_hardware()
            raise
        except Exception as exc:
            log.error(f"Unexpected error during init: {exc}")
            self._close_hardware()
            raise

    def _initialize_scapp(self) -> None:
        """Open the card and start the SCAPP GPU feeder thread."""
        if not _HW_AVAILABLE or not _SCAPP_GPU_AVAILABLE:
            log.info(
                f"Simulation mode: hardware init skipped (backend=scapp, "
                f"hw_available={_HW_AVAILABLE}, gpu_available={_SCAPP_GPU_AVAILABLE})."
            )
            return

        assert (
            self.hw.max_amplitude_v <= 2.0
        ), f"max_amplitude_v={self.hw.max_amplitude_v} exceeds 2.0 V hard safety ceiling"

        try:
            self._card = spcm.Card(self.hw.card_path)
            self._feeder = ScappFeeder(
                self._card, self.hw, self.sw.aod_settings, self._scapp_config
            )
            self._feeder.start(self.rf_converter.holding_config())
            log.info(
                f"Hardware initialised (backend=scapp, "
                f"sample_rate={self._feeder.sample_rate_hz / 1e6:.1f} MHz)."
            )

        except SpcmException as exc:
            log.error(f"spcm error during init: {exc}")
            self._close_hardware()
            raise
        except Exception as exc:
            log.error(f"Unexpected error during init: {exc}")
            self._close_hardware()
            raise

    def _output_batch(self, batch: AWGBatch) -> None:
        """Send one ``AWGBatch`` to the card via the active backend."""
        if not _HW_AVAILABLE or self._card is None:
            label = self.strategy.name if self.backend == "legacy_dds" else "scapp"
            log.info(
                f"[SIM:{label}] batch: {len(batch.ramps)} ramps, "
                f"duration={batch.total_duration_s*1e6:.1f} µs"
            )
            if batch.total_duration_s > 0:
                time.sleep(batch.total_duration_s)
            return

        if self.backend == "legacy_dds":
            dds = self._dds[self._card.sn()]
            self.strategy.execute_batch(dds, batch)
        else:
            self._feeder.submit_batch(batch)

    def _send_holding(self) -> None:
        """Restore static holding configuration (atoms held in place)."""
        holding = self.rf_converter.holding_config()

        if not _HW_AVAILABLE or self._card is None:
            label = self.strategy.name if self.backend == "legacy_dds" else "scapp"
            log.info(f"[SIM:{label}] holding: " f"{len(holding.ramps)} ramps")
            return

        if self.backend == "legacy_dds":
            dds = self._dds[self._card.sn()]
            self.strategy.send_holding(dds, holding)
        else:
            self._feeder.submit_holding(holding)

    # ------------------------------------------------------------------
    # Imaging / targets
    # ------------------------------------------------------------------

    def _ensure_camera(self) -> Camera:
        """Return the active camera, creating an offline default if needed."""
        if self.camera is not None:
            return self.camera
        self.camera = OfflineArrayCamera(
            self.grid_shape,
            physical_params=self.sw.physical_params,
            blob_params=self.sw.blob_params,
        )
        log.info(
            "No camera provided — using OfflineArrayCamera "
            f"on {self.grid_shape[0]}×{self.grid_shape[1]}."
        )
        return self.camera

    def _acquire(self) -> np.ndarray:
        """Return a grayscale frame (Camera.acquire / legacy dummy)."""
        if self.camera is not None:
            return self.camera.acquire()

        # Legacy dummy for unit tests that call _acquire() with no camera.
        rng = np.random.default_rng()
        grid = self.sw.grid_size
        img = np.zeros((512, 512), dtype=np.uint8)
        step = max(512 // max(grid, 1), 1)
        for r in range(grid):
            for c in range(grid):
                if rng.random() < self.sw.physical_params.loading_prob:
                    cy, cx = r * step + step // 2, c * step + step // 2
                    img[max(0, cy - 2) : cy + 3, max(0, cx - 2) : cx + 3] = 200
        log.debug("Dummy image generated (simulation mode).")
        return img

    def _build_target_mask(self, grid_shape: Tuple[int, int]) -> np.ndarray:
        """Centred rectangular target from middle_size / AOD / target_size."""
        rows, cols = grid_shape
        ms = self.sw.physical_params.middle_size
        if ms is not None and len(ms) >= 2:
            tr, tc = int(ms[0]), int(ms[1])
        else:
            tr = int(self.sw.aod_settings.target_rows or self.sw.target_size)
            tc = int(self.sw.aod_settings.target_cols or self.sw.target_size)

        tr = min(max(tr, 1), rows)
        tc = min(max(tc, 1), cols)
        mask = np.zeros((rows, cols), dtype=int)
        r0 = max((rows - tr) // 2, 0)
        c0 = max((cols - tc) // 2, 0)
        mask[r0 : r0 + tr, c0 : c0 + tc] = 1
        return mask

    def _apply_target(self) -> np.ndarray:
        """Build / cache target mask and copy it onto ``self.array.target``."""
        if self._target_mask is None:
            ms = self.sw.physical_params.middle_size
            if ms is not None and len(ms) >= 2:
                self.array.generate_target(self.sw.target_type, middle_size=list(ms))
                self._target_mask = (self.array.target[:, :, 0] > 0).astype(int)
            else:
                self._target_mask = self._build_target_mask(self.grid_shape)
                self.array.target[:, :, 0] = self._target_mask
        else:
            self.array.target[:, :, 0] = self._target_mask
        return self._target_mask

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """Execute the rearrangement feedback loop.

        Returns
        -------
        bool
            ``True`` if the target is successfully filled within ``max_rounds``.
        """
        cam = self._ensure_camera()
        recorder = self.recorder

        log.info(
            f"Loop start — algorithm={self.sw.algorithm_name}, "
            f"grid={self.grid_shape[0]}×{self.grid_shape[1]}, "
            f"max_rounds={self.sw.max_rounds}, camera={type(cam).__name__}"
        )

        try:
            for r in range(self.sw.max_rounds + 1):
                t_loop = time.perf_counter()
                if recorder is not None:
                    recorder.begin_round(r)

                # 1. Acquire + detect into the global array
                try:
                    cam.sync(self.array, recorder=recorder)
                except Exception as exc:
                    log.error(f"Round {r}: acquisition failed - {exc}")
                    return False

                self._grid_rotation = float(getattr(cam, "grid_rotation", 0.0) or 0.0)
                state = (self.array.matrix[:, :, 0] > 0).astype(int)
                target = self._apply_target()

                filled = int((state * target).sum())
                need = int(target.sum())
                atoms = int(state.sum())

                # 2. Success?
                if filled == need:
                    log.info(f"SUCCESS - target filled after {r} round(s).")
                    if recorder is not None:
                        recorder.log_round(
                            atoms=atoms,
                            filled=filled,
                            need=need,
                            n_moves=0,
                            success=True,
                        )
                    return True

                if r == self.sw.max_rounds:
                    log.warning(
                        f"Max rounds ({self.sw.max_rounds}) reached; "
                        f"{need - filled} target sites remain empty."
                    )
                    if recorder is not None:
                        recorder.log_round(
                            atoms=atoms,
                            filled=filled,
                            need=need,
                            n_moves=0,
                            success=False,
                        )
                    break

                if atoms < need:
                    log.error(
                        f"Round {r}: insufficient atoms "
                        f"(have {atoms}, need {need}). Aborting."
                    )
                    if recorder is not None:
                        recorder.log_round(
                            atoms=atoms,
                            filled=filled,
                            need=need,
                            n_moves=0,
                            aborted="insufficient_atoms",
                        )
                    return False

                # 3. Algorithm
                try:
                    _, move_batches, algo_ok = self.algorithm.get_moves(self.array)
                except Exception as exc:
                    log.exception(f"Round {r}: algorithm raised - {exc}")
                    return False

                if not algo_ok:
                    log.error(f"Round {r}: algorithm reported failure.")
                    if recorder is not None:
                        recorder.log_round(
                            atoms=atoms,
                            filled=filled,
                            need=need,
                            n_moves=0,
                            aborted="algo_fail",
                        )
                    return False

                # 4. RF + hardware
                rf_batches = self.rf_converter.convert_sequence(move_batches)
                n_moves = sum(len(b) for b in move_batches)
                log.info(
                    f"Round {r}: {n_moves} moves → {len(rf_batches)} hardware batches."
                )

                for batch in rf_batches:
                    self._output_batch(batch)
                self._send_holding()

                # 5. Offline physics on the controller-owned array
                if isinstance(cam, OfflineArrayCamera):
                    self.array.evaluate_moves(move_batches)

                if recorder is not None:
                    recorder.log_round(
                        atoms=atoms,
                        filled=filled,
                        need=need,
                        n_moves=n_moves,
                        n_parallel_batches=len(move_batches),
                        n_rf_batches=len(rf_batches),
                        rf_duration_s=float(
                            sum(b.total_duration_s for b in rf_batches)
                        ),
                        moves=moves_to_records(move_batches),
                    )

                elapsed_ms = (time.perf_counter() - t_loop) * 1e3
                log.info(f"Round {r} done in {elapsed_ms:.1f} ms.")

            return False
        finally:
            if recorder is not None:
                written = recorder.finalize()
                if written:
                    log.info(
                        "Recorder GIFs: "
                        + ", ".join(f"{k}→{p.name}" for k, p in written.items())
                    )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _close_hardware(self) -> None:
        if self.backend == "legacy_dds":
            if self._card is not None:
                dds = self._dds.get(self._card.sn())
                if dds is not None:
                    self.strategy.shutdown(dds, self._card)
                try:
                    self._card.stop(spcm.M2CMD_CARD_STOP)
                except Exception:
                    pass
                try:
                    self._card.close()
                except Exception:
                    pass
                self._card = None
                self._dds.clear()
        else:
            if self._feeder is not None:
                self._feeder.stop()
                self._feeder = None
            if self._card is not None:
                try:
                    self._card.close()
                except Exception:
                    pass
                self._card = None

    def shutdown(self) -> None:
        """Gracefully stop the card and release all resources."""
        self._close_hardware()
        log.info(f"Controller shut down (backend={self.backend}).")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="atommovr production controller — image → AOD feedback loop."
    )
    p.add_argument(
        "--algorithm",
        default="PCFA",
        choices=list(_ALGORITHM_REGISTRY),
        help="Rearrangement algorithm",
    )
    p.add_argument(
        "--grid-rows",
        type=int,
        default=10,
        help="Grid rows (V-AOD tones, max 16 w/ 2ch)",
    )
    p.add_argument(
        "--grid-cols", type=int, default=5, help="Grid cols (H-AOD tones, max 5)"
    )
    p.add_argument("--target-rows", type=int, default=6, help="Target sub-array rows")
    p.add_argument("--target-cols", type=int, default=5, help="Target sub-array cols")
    p.add_argument(
        "--max-rounds", type=int, default=10, help="Max rearrangement rounds"
    )
    p.add_argument(
        "--card",
        type=str,
        default="/dev/spcm0",
        help="Card device path (single card; multi-card support removed)",
    )
    p.add_argument(
        "--backend",
        default="scapp",
        choices=["scapp", "legacy_dds"],
        help="AWG generation backend",
    )
    p.add_argument(
        "--trg-timer",
        type=float,
        default=0.2,
        help="Idle/holding TIMER interval (s), legacy_dds backend only",
    )
    p.add_argument("--f-min-v", type=float, default=60e6, help="V-AOD f_min (Hz)")
    p.add_argument("--f-max-v", type=float, default=100e6, help="V-AOD f_max (Hz)")
    p.add_argument("--f-min-h", type=float, default=60e6, help="H-AOD f_min (Hz)")
    p.add_argument("--f-max-h", type=float, default=100e6, help="H-AOD f_max (Hz)")
    p.add_argument(
        "--strategy",
        default="streaming",
        choices=list(STRATEGY_REGISTRY),
        help="DDS execution strategy, legacy_dds backend only",
    )
    p.add_argument(
        "--ramp-shape",
        default="linear",
        choices=["linear", "scurve"],
        help="SCAPP frequency-ramp shape, scapp backend only",
    )
    p.add_argument(
        "--notify-samples",
        type=int,
        default=512 * 1024,
        help="SCAPP GPU buffer notify block size (samples), scapp backend only",
    )
    args = p.parse_args()

    if args.backend == "legacy_dds":
        validate_hardware_limits(args.grid_rows, args.grid_cols)

    hw = HardwareConfig(
        card_path=args.card,
        trigger_timer_s=args.trg_timer,
        notify_samples=args.notify_samples,
    )
    sw = SoftwareConfig(
        grid_size=max(args.grid_rows, args.grid_cols),
        target_size=min(args.target_rows, args.target_cols),
        algorithm_name=args.algorithm,
        max_rounds=args.max_rounds,
        physical_params=PhysicalParams(
            middle_size=[args.target_rows, args.target_cols],
        ),
        aod_settings=AODSettings(
            f_min_v=args.f_min_v,
            f_max_v=args.f_max_v,
            f_min_h=args.f_min_h,
            f_max_h=args.f_max_h,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            target_rows=args.target_rows,
            target_cols=args.target_cols,
            ramp_shape=args.ramp_shape,
        ),
    )

    strategy_arg = args.strategy if args.backend == "legacy_dds" else None

    with atommovrController(
        sw, hw, backend=args.backend, strategy=strategy_arg
    ) as ctrl:
        try:
            success = ctrl.run()
            sys.exit(0 if success else 1)
        except KeyboardInterrupt:
            log.info("Interrupted by user.")
            sys.exit(130)


if __name__ == "__main__":
    main()
