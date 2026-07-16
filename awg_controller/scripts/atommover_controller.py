#!/usr/bin/env python3
"""
atommovr Production Controller
================================
Orchestrates the complete atom-rearrangement feedback loop:

  1. Sync camera → occupancy (``Camera.sync`` / ``detect_occupancy``)
  2. Check whether target is already filled  →  done
  3. Compute rearrangement moves (configurable algorithm)
  4. Convert moves to RF ramps  (RFConverter / AWGBatch)
  5. Write ramps to Spectrum Instrumentation DDS card  (spcm)
  6. Offline: advance physics on the controller ``AtomArray``
  7. Repeat from step 1

Hardware integration is copied verbatim from cli.py (source of truth):
  - DDSCommandQueue for all core updates
  - dds[core].freq() / .phase() / .amp()  in spcm units
  - exec_at_trg() + write_to_card() per trigger event
  - spcm.SPCM_DDS_TRG_SRC_TIMER trigger mode
  - 40 % total amplitude budget per channel
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

#  optional hardware driver
try:
    import spcm
    from spcm import SpcmException

    _HW_AVAILABLE = True
except ImportError:
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

if not _HW_AVAILABLE:
    log.warning("spcm not found - running in SIMULATION mode (no hardware I/O).")


@dataclass
class HardwareConfig:
    """Spectrum Instrumentation card configuration (mirrors cli.py defaults)."""

    #: Device paths, e.g. ["/dev/spcm0"]  or  ["/dev/spcm0", "/dev/spcm1"]
    card_paths: List[str] = field(default_factory=lambda: ["/dev/spcm0"])

    #: Output amplitude - manufacturer maximum is 1.6 V into 50 Ω
    max_amplitude_v: float = 1.6

    #: Output impedance
    output_load_ohms: float = 50.0

    #: Timer-trigger interval (s).  Set to the expected move duration so the
    #: card fires once per rearrangement batch.  cli.py default was 0.2 s.
    trigger_timer_s: float = 0.2

    #: Number of pre-fill writes during initialisation (prevents underruns).
    #: cli.py uses  floor(10 / trg_timer); we keep that formula.
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
    strategy : DDSStrategy or str, optional
        DDS execution strategy (``"streaming"``, ``"ramp"``, ``"pattern"``,
        ``"camera_triggered"``). Defaults to ``DDSStreamingStrategy``.
    """

    def __init__(
        self,
        sw_config: SoftwareConfig,
        hw_config: HardwareConfig,
        camera: Optional[Camera] = None,
        camera_fn: Optional[Callable[[], np.ndarray]] = None,
        recorder: Optional[SessionRecorder] = None,
        strategy: Optional[Union[DDSStrategy, str]] = None,
    ) -> None:
        if camera is not None and camera_fn is not None:
            raise ValueError("Pass camera= or camera_fn=, not both.")

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

        if strategy is None:
            self.strategy: DDSStrategy = DDSStreamingStrategy()
        elif isinstance(strategy, str):
            self.strategy = get_strategy(strategy)
        else:
            self.strategy = strategy

        self.algorithm = _ALGORITHM_REGISTRY[sw_config.algorithm_name]()
        self.rf_converter = RFConverter(aod, sw_config.physical_params)

        err = (
            sw_config.error_model if sw_config.error_model is not None else ZeroNoise()
        )
        self.array = AtomArray(
            list(self.grid_shape),
            n_species=1,
            params=sw_config.physical_params,
            error_model=err,
        )

        self._dds: Dict[int, Any] = {}
        self._stack: Any = None
        self._grid_rotation: float = 0.0
        self._target_mask: Optional[np.ndarray] = None
        self._apply_target()

        self._initialize_hardware()

    # ------------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------------

    def _initialize_hardware(self) -> None:
        """Open cards, configure DDS, pre-fill buffer, enable trigger."""
        if not _HW_AVAILABLE:
            log.info(
                f"Simulation mode: hardware init skipped "
                f"(strategy={self.strategy.name})."
            )
            return

        validate_hardware_limits(
            self.sw.aod_settings.grid_rows,
            self.sw.aod_settings.grid_cols,
        )

        try:
            self._stack = spcm.CardStack(self.hw.card_paths)
            holding = self.rf_converter.holding_config()
            core_map = self.rf_converter.core_map

            for card in self._stack.cards:
                sn = card.sn()
                card.card_mode(spcm.SPC_REP_STD_DDS)
                channels = spcm.Channels(
                    card=card,
                    card_enable=spcm.CHANNEL0 | spcm.CHANNEL1,
                )
                channels.enable(True)
                channels.amp(self.hw.max_amplitude_v * spcm.units.V)
                channels.output_load(self.hw.output_load_ohms * spcm.units.ohm)
                card.write_setup()

                dds = self.strategy.create_dds(card, channels)
                self.strategy.configure(dds, card, self.hw, core_map)
                self.strategy.prefill(dds, holding, self.hw)
                self._dds[sn] = dds

            self.strategy.start(self._stack)
            log.info(f"Hardware initialised (strategy={self.strategy.name}).")

        except SpcmException as exc:
            log.error(f"spcm error during init: {exc}")
            self._close_stack()
            raise
        except Exception as exc:
            log.error(f"Unexpected error during init: {exc}")
            self._close_stack()
            raise

    def _output_batch(self, batch: AWGBatch) -> None:
        """Send one ``AWGBatch`` to all open cards via the active strategy."""
        if not _HW_AVAILABLE or self._stack is None:
            log.info(
                f"[SIM:{self.strategy.name}] batch: {len(batch.ramps)} ramps, "
                f"duration={batch.total_duration_s*1e6:.1f} µs"
            )
            if batch.total_duration_s > 0:
                time.sleep(batch.total_duration_s)
            return

        for card in self._stack.cards:
            sn = card.sn()
            dds = self._dds[sn]
            self.strategy.execute_batch(dds, batch)

    def _send_holding(self) -> None:
        """Restore static holding configuration (atoms held in place)."""
        holding = self.rf_converter.holding_config()

        if not _HW_AVAILABLE or self._stack is None:
            log.info(
                f"[SIM:{self.strategy.name}] holding: " f"{len(holding.ramps)} ramps"
            )
            return

        for card in self._stack.cards:
            sn = card.sn()
            dds = self._dds[sn]
            self.strategy.send_holding(dds, holding)

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

    def _close_stack(self) -> None:
        if self._stack is not None:
            for card in self._stack.cards:
                sn = card.sn()
                dds = self._dds.get(sn)
                if dds is not None:
                    self.strategy.shutdown(dds, card)
            try:
                self._stack.close()
            except Exception:
                pass
            self._stack = None
            self._dds.clear()

    def shutdown(self) -> None:
        """Gracefully stop the card and release all resources."""
        self._close_stack()
        log.info(f"Controller shut down (strategy={self.strategy.name}).")

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
        action="append",
        default=["/dev/spcm0"],
        help="Card device path (repeat for multiple cards)",
    )
    p.add_argument(
        "--trg-timer",
        type=float,
        default=0.2,
        help="Trigger timer interval (s), sets move window",
    )
    p.add_argument("--f-min-v", type=float, default=60e6, help="V-AOD f_min (Hz)")
    p.add_argument("--f-max-v", type=float, default=100e6, help="V-AOD f_max (Hz)")
    p.add_argument("--f-min-h", type=float, default=60e6, help="H-AOD f_min (Hz)")
    p.add_argument("--f-max-h", type=float, default=100e6, help="H-AOD f_max (Hz)")
    p.add_argument(
        "--strategy",
        default="streaming",
        choices=list(STRATEGY_REGISTRY),
        help="DDS execution strategy",
    )
    args = p.parse_args()

    validate_hardware_limits(args.grid_rows, args.grid_cols)

    hw = HardwareConfig(
        card_paths=args.card,
        trigger_timer_s=args.trg_timer,
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
        ),
    )

    with atommovrController(sw, hw, strategy=args.strategy) as ctrl:
        try:
            success = ctrl.run()
            sys.exit(0 if success else 1)
        except KeyboardInterrupt:
            log.info("Interrupted by user.")
            sys.exit(130)


if __name__ == "__main__":
    main()
