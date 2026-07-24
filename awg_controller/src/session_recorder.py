"""Optional session recorder for camera stage dumps and round stats.

Off by default — attach a ``SessionRecorder`` to ``atommovrController``
(``recorder=``) and the controller passes it into ``Camera.sync`` /
``log_round`` each loop. When ``enabled=False`` or unused, no I/O.

Writes:

* per-stage folders: ``round_{rr:02d}_{stage}/frame.png``, ``occupancy.npy``, …
* append-only ``rounds.jsonl`` for move / RF statistics
* optional GIFs: ``frames.gif`` / ``occupancy.gif`` (see :class:`GifOptions`)
* optional per-round AWG output spectrograms (see :class:`SpectrogramOptions`)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

_STAGE_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_stage(stage: str) -> str:
    cleaned = _STAGE_SAFE.sub("_", str(stage).strip()) or "stage"
    return cleaned[:64]


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize an array to uint8 grayscale (or pass through HxWx3 uint8)."""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.dtype == np.uint8:
        return arr
    amin = float(np.min(arr)) if arr.size else 0.0
    amax = float(np.max(arr)) if arr.size else 1.0
    if amax <= amin:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    scaled = (arr.astype(np.float64) - amin) / (amax - amin)
    return (scaled * 255.0).astype(np.uint8)


def _write_png(path: Path, image: np.ndarray) -> None:
    """Write a 2-D grayscale (or HxWx{1,3}) array as PNG via OpenCV if present."""
    arr = _to_uint8(image)
    try:
        import cv2  # soft dependency

        ok = cv2.imwrite(str(path), arr)
        if not ok:
            raise OSError(f"cv2.imwrite failed for {path}")
    except ImportError:
        # Fallback: raw dump already kept as .npy; skip PNG.
        pass


def _occupancy_heatmap(occ: np.ndarray, cell_px: int = 16) -> np.ndarray:
    """Upsample a binary occupancy grid to a small uint8 image."""
    binary = (np.asarray(occ) > 0).astype(np.uint8) * 255
    if cell_px <= 1:
        return binary
    return np.kron(binary, np.ones((cell_px, cell_px), dtype=np.uint8))


def _resize_max_side(image: np.ndarray, max_side: Optional[int]) -> np.ndarray:
    """Downscale so max(H, W) <= max_side (nearest-neighbour)."""
    if max_side is None or max_side <= 0:
        return image
    h, w = image.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return image
    scale = max_side / float(m)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    try:
        import cv2

        return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_NEAREST)
    except ImportError:
        ys = (np.linspace(0, h - 1, nh)).astype(int)
        xs = (np.linspace(0, w - 1, nw)).astype(int)
        return image[ys][:, xs]


def _write_gif(
    path: Path,
    frames: Sequence[np.ndarray],
    *,
    duration_s: float,
    loop: int,
) -> bool:
    """Write a GIF from uint8 frames. Returns True on success."""
    if not frames:
        return False
    duration_s = max(float(duration_s), 1e-3)
    # Ensure shared shape (crop/pad to first frame).
    h0, w0 = frames[0].shape[:2]
    normed = []
    for fr in frames:
        fr = _to_uint8(fr)
        if fr.ndim == 2:
            canvas = np.zeros((h0, w0), dtype=np.uint8)
        else:
            canvas = np.zeros((h0, w0, fr.shape[2]), dtype=np.uint8)
        hh, ww = min(h0, fr.shape[0]), min(w0, fr.shape[1])
        canvas[:hh, :ww] = fr[:hh, :ww]
        normed.append(canvas)

    try:
        import imageio.v2 as imageio

        with imageio.get_writer(
            str(path),
            mode="I",
            duration=duration_s,
            loop=int(loop),
        ) as writer:
            for fr in normed:
                writer.append_data(fr)
        return True
    except Exception:
        pass

    try:
        from PIL import Image

        imgs = [Image.fromarray(fr) for fr in normed]
        imgs[0].save(
            str(path),
            save_all=True,
            append_images=imgs[1:],
            duration=int(round(duration_s * 1000)),
            loop=int(loop),
        )
        return True
    except Exception:
        return False


@dataclass
class GifOptions:
    """Customization for optional rearrangement GIFs.

    Parameters
    ----------
    enabled
        When ``True``, accumulate stage images and write GIF(s) under ``run_dir``.
    sources
        Which payloads to animate: ``"frame"``, ``"occupancy"``, or both.
    stages
        Only append images from these ``save_stage`` names (e.g. ``("detect",)``).
        Empty tuple → include every stage.
    duration_s
        Seconds per GIF frame.
    loop
        GIF loop count (``0`` = infinite).
    max_side
        Optional downscale so ``max(H, W) <= max_side`` (keeps GIFs small).
    occupancy_cell_px
        Pixel size of each site in the occupancy heatmap used for GIFs/PNGs.
    auto_write
        Rewrite GIF files after every matching ``save_stage`` (always up to date).
        If ``False``, call :meth:`SessionRecorder.finalize` (or ``write_gifs``) once.
    """

    enabled: bool = True
    sources: Tuple[str, ...] = ("frame", "occupancy")
    stages: Tuple[str, ...] = ("detect",)
    duration_s: float = 0.4
    loop: int = 0
    max_side: Optional[int] = 512
    occupancy_cell_px: int = 16
    auto_write: bool = True


def _ramp_freq_bounds(
    rf_batches: Sequence[Any],
    channel: Optional[int] = None,
    *,
    moving_only: bool = False,
) -> Optional[Tuple[float, float]]:
    """Min/max of ``f_start``/``f_end`` over ramps (optionally one channel)."""
    vals: list[float] = []
    for batch in rf_batches:
        for ramp in getattr(batch, "ramps", ()) or ():
            if channel is not None and int(ramp.channel) != int(channel):
                continue
            if moving_only and not _is_moving_ramp(ramp):
                continue
            vals.append(float(ramp.f_start))
            vals.append(float(ramp.f_end))
    if not vals:
        return None
    return min(vals), max(vals)


def _is_moving_ramp(ramp: Any, *, tol_hz: float = 1.0) -> bool:
    """True when a ramp actually sweeps frequency (not a static hold)."""
    return abs(float(ramp.f_end) - float(ramp.f_start)) > float(tol_hz)


def _rf_batches_moves_only(rf_batches: Sequence[Any]) -> list[Any]:
    """Copy RF batches with holding-tone amplitudes zeroed for viz-only synthesis.

    Keeps every tone in the schedule (so scapp phase-continuity bookkeeping
    stays valid) but silences ``f_start == f_end`` holds so the spectrogram
    shows only moving chirps.
    """
    from copy import copy
    from dataclasses import replace

    from awg_controller.src.awg_control import AWGBatch

    out: list[Any] = []
    for batch in rf_batches:
        ramps = []
        for ramp in getattr(batch, "ramps", ()) or ():
            if _is_moving_ramp(ramp):
                ramps.append(ramp)
            else:
                ramps.append(replace(ramp, amplitude_pct=0.0))
        # Preserve duration even if every tone is silenced this batch.
        if isinstance(batch, AWGBatch):
            out.append(AWGBatch(ramps=ramps, total_duration_s=batch.total_duration_s))
        else:
            cloned = copy(batch)
            cloned.ramps = ramps
            out.append(cloned)
    return out


def _time_display_scale(duration_s: float) -> Tuple[float, str]:
    """Pick a human scale for spectrogram time axes."""
    if duration_s < 1e-3:
        return 1e6, "µs"
    if duration_s < 1.0:
        return 1e3, "ms"
    return 1.0, "s"


def _choose_nperseg(
    sample_rate_hz: float,
    n_samples: int,
    *,
    nperseg: Optional[int],
    target_df_hz: float,
    max_window_samples: Optional[int] = None,
) -> int:
    """STFT window length: explicit ``nperseg``, else ~``target_df_hz`` resolution
    capped by ``max_window_samples``.

    A window sized purely from ``target_df_hz`` can end up *longer than a
    single ramp/hold segment* in the round (short moves are only a few µs at
    a several-hundred-MHz synthesis rate). When that happens the STFT can't
    resolve the sweep within one window — adjacent segments blur together
    and the chirp reads as blocky/discrete instead of a smooth diagonal line.
    ``max_window_samples`` (derived from the shortest ramp duration in the
    round, see :func:`_min_batch_samples`) keeps the window meaningfully
    shorter than that, trading some frequency resolution for the time
    resolution needed to actually see the sweep.
    """
    n_samples = max(int(n_samples), 1)
    if nperseg is not None:
        return min(max(int(nperseg), 1), n_samples)
    hi = n_samples
    if max_window_samples is not None:
        hi = min(hi, max(int(max_window_samples), 1))
    hi = min(hi, 16384)
    target_df = max(float(target_df_hz), 1.0)
    # Prefer a power-of-two window near fs / Δf (better FFT performance / bins).
    ideal = float(sample_rate_hz) / target_df
    if ideal <= 1:
        return max(1, hi)
    pow2 = 1 << int(round(np.log2(min(ideal, hi))))
    return int(min(max(pow2, 16), hi))


def _min_batch_samples(
    rf_batches: Sequence[Any], sample_rate_hz: float
) -> Optional[int]:
    """Sample count of the shortest nonzero-duration batch in the round.

    Zero-duration batches (pure holds, ``convert_moves([])``) don't
    constrain the STFT window — they contribute no samples of their own.
    """
    durations = [float(getattr(b, "total_duration_s", 0.0) or 0.0) for b in rf_batches]
    durations = [d for d in durations if d > 0]
    if not durations:
        return None
    return max(int(round(min(durations) * float(sample_rate_hz))), 1)


@dataclass
class SpectrogramOptions:
    """Customization for optional per-round AWG frequency/spectrogram plots.

    Off by default — unlike ``GifOptions``, :meth:`SessionRecorder.save_spectrogram`
    re-synthesizes the full per-channel time-domain waveform for every ramp
    in a round *twice* (with and without holding tones — see
    ``awg_controller.src.scapp.synthesize_round_waveform``) plus the pure
    analytical frequency trajectory
    (``synthesize_round_frequency_trajectory``), and needs ``scipy``/
    ``matplotlib``, so it is heavier than the other :class:`SessionRecorder`
    outputs.

    Parameters
    ----------
    enabled
        When ``True``, :meth:`SessionRecorder.save_spectrogram` writes a PNG
        (and raw ``.npy`` waveforms) per round; otherwise it no-ops.
    include_stft
        When ``True`` (default), rows 2-3 (the STFT spectrograms) are
        computed and rendered alongside row 1. When ``False``, only the
        pure, FFT-free "commanded frequency f(t)" panel is rendered — no
        waveform synthesis, no STFT, no ``.npy`` waveform files — saved as
        ``spectrogram_f_t.png`` regardless of ``separate_holding_stft``.
        ``sample_rate_hz`` becomes unnecessary in this mode (the f(t) panel
        needs only ``rf_batches``' own ramp timings, not a synthesis rate).
    sample_rate_hz
        Synthesis sample rate (Hz). Must exceed 2x the highest AOD tone
        frequency in play (Nyquist) or the spectrogram will alias. ``None``
        requires callers to pass ``sample_rate_hz=`` explicitly to
        ``save_spectrogram`` instead. Unused when ``include_stft=False``.
    ramp_shape
        Frequency-ramp shape used to synthesize non-static moves
        (``"linear"`` | ``"scurve"``) — should match the AWG backend's
        actual ramp shape (``AODSettings.ramp_shape`` for scapp).
    nperseg
        Explicit ``scipy.signal.spectrogram`` window length (samples).
        ``None`` (default) chooses a power-of-two window targeting
        ``target_df_hz`` frequency resolution, additionally capped so at
        least ``min_windows_per_ramp`` windows fit inside the *shortest*
        ramp/hold-transition batch in the round (see ``min_windows_per_ramp``)
        — without this cap, short moves (a few µs, common for small-Chebyshev
        rearrangement steps) can be shorter than the frequency-resolution
        window, so the STFT can't resolve the sweep and the chirp reads as
        blocky/discrete rather than a smooth diagonal line.
    target_df_hz
        Desired STFT frequency bin width when ``nperseg`` is ``None``
        (default 50 kHz — enough to resolve typical AOD site spacings).
        Acts as an upper bound on the window; ``min_windows_per_ramp`` can
        shrink it further for short batches.
    min_windows_per_ramp
        Minimum number of STFT windows that must fit inside the shortest
        nonzero-duration batch in the round, when ``nperseg`` is ``None``
        (default 8). Directly trades frequency resolution for time
        resolution — lower if tones are closely spaced and holds/ramps are
        long; raise if chirps still look blocky.
    noverlap
        Spectrogram window overlap (samples). ``None`` →
        ``int(nperseg * noverlap_frac)``.
    noverlap_frac
        Fraction of ``nperseg`` overlapped when ``noverlap`` is ``None``
        (high overlap → smoother time axis).
    freq_min_hz / freq_max_hz
        Y-axis limits (AOD RF band). Prefer the lab ``AODSettings`` range
        (e.g. 82–118 MHz). ``None`` falls back to ramp frequencies in the
        round (plus ``freq_pad_hz``), then full Nyquist.
    freq_pad_hz
        Extra padding applied around auto / configured frequency limits.
    db_range
        Colorbar span in dB below the in-band peak (``vmax - db_range`` …
        ``vmax``). Keeps noise floor from washing out tone contrast.
    separate_holding_stft
        When ``True``, the "with holding" STFT (row 3) uses its own window
        driven purely by ``target_df_hz`` — *not* capped by the shortest
        ramp's duration like row 2's ``min_windows_per_ramp`` cap — since
        static holds don't need the fine time resolution a short chirp
        does, and sharing that small window (row 2's) otherwise smears
        closely-spaced holding tones together (frequency resolution
        ``= sample_rate_hz / nperseg`` can end up comparable to or coarser
        than the AOD's site spacing). Also switches output from one
        combined 3-row PNG to three separate files under the same
        ``round_{rr:02d}_spectrogram/`` directory — ``spectrogram_f_t.png``,
        ``spectrogram_moves_only.png``, ``spectrogram_with_holding.png`` —
        each with its own color scale, since the two STFT rows are no
        longer time/frequency-comparable on one shared axis. Default
        ``False`` preserves the single combined-figure, shared-window
        behavior.
    cmap
        Matplotlib colormap for power (dB).
    channel_labels
        Optional ``{channel: label}`` for subplot titles (e.g.
        ``{0: "V/row", 1: "H/col"}``); defaults to ``"ch{n}"``.
    """

    enabled: bool = False
    include_stft: bool = True
    sample_rate_hz: Optional[float] = None
    ramp_shape: str = "linear"
    nperseg: Optional[int] = None
    target_df_hz: float = 50e3
    min_windows_per_ramp: int = 8
    noverlap: Optional[int] = None
    noverlap_frac: float = 0.9375
    freq_min_hz: Optional[float] = None
    freq_max_hz: Optional[float] = None
    freq_pad_hz: float = 2e6
    db_range: float = 20.0
    separate_holding_stft: bool = False
    cmap: str = "magma"
    channel_labels: Optional[Mapping[int, str]] = None


@dataclass
class VisualizationOptions:
    """Customization for optional per-round move-batch visualizations
    (``atommovr.utils.imaging.visualization`` — schematic lattice view
    only; no camera-image rendering).

    Off by default — unlike the raw stage/occupancy dumps,
    :meth:`SessionRecorder.save_move_visualization` re-simulates every move
    batch in the round through a scratch ``AtomArray`` copy (collision/
    failure-aware, via ``AtomArray.move_atoms``) to render one panel per
    batch with move arrows and failure markers, so it's comparable in cost
    to :class:`SpectrogramOptions`. ``matplotlib`` is a soft dependency of
    this method only.

    Parameters
    ----------
    enabled
        When ``True``, :meth:`SessionRecorder.save_move_visualization`
        writes figure(s) per round under
        ``round_{rr:02d}_visualization/``; otherwise it no-ops.
    max_batches
        Cap on how many move batches to render (one panel/frame each, plus
        one "Initial" panel/frame). A round with many small parallel-move
        batches (e.g. dozens of single-move batches) produces one panel per
        batch — uncapped, the static figure can render many megapixels
        tall. ``None`` disables the cap (renders every batch). When capped,
        only the first ``max_batches`` batches are shown — an honest
        partial view of the round, not resampled or reordered, since each
        panel's state depends on every batch simulated before it.
    max_cols
        Maximum subplot columns in the static multi-panel figure
        (``grid.<grid_format>``).
    grid_format
        File extension/format for the static figure (matplotlib infers the
        writer from the extension).
    gif
        When ``True`` (default), also render an animated ``grid.gif``
        cycling through each snapshot (one second-or-so-of-cost-each frame
        via :func:`atommovr.utils.imaging.visualization.render_move_batch_frames`)
        — often easier to read than the static multi-panel figure for
        rounds with many batches. Respects ``max_batches``.
    gif_duration_s
        Seconds each GIF frame is displayed.
    gif_loop
        GIF loop count (``0`` = infinite).
    """

    enabled: bool = False
    max_batches: Optional[int] = 15
    max_cols: int = 3
    grid_format: str = "svg"
    gif: bool = True
    gif_duration_s: float = 0.5
    gif_loop: int = 0


class SessionRecorder:
    """Write stage frames/occupancy and append per-round JSONL stats.

    Parameters
    ----------
    run_root
        Parent directory; a timestamped ``run_YYYYMMDD_HHMMSS`` folder is created
        underneath (unless ``run_dir`` is passed explicitly).
    enabled
        When ``False``, all methods no-op.
    run_dir
        Optional fixed run directory (tests); skips timestamp folder creation.
    meta
        Optional dict merged into ``meta.json`` at construction.
    gif
        GIF customization. Pass ``GifOptions(enabled=False)`` to skip GIFs, or
        ``None`` for defaults (GIF on, ``detect`` stages, frame+occupancy).
    spectrogram
        AWG output spectrogram customization. ``None`` for defaults (off —
        pass ``SpectrogramOptions(enabled=True, ...)`` to opt in).
    visualization
        Move-batch visualization customization. ``None`` for defaults (off —
        pass ``VisualizationOptions(enabled=True, ...)`` to opt in).
    """

    def __init__(
        self,
        run_root: PathLike = "runs",
        *,
        enabled: bool = True,
        run_dir: Optional[PathLike] = None,
        meta: Optional[Mapping[str, Any]] = None,
        gif: Optional[GifOptions] = None,
        spectrogram: Optional[SpectrogramOptions] = None,
        visualization: Optional[VisualizationOptions] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.gif = gif if gif is not None else GifOptions()
        self.spectrogram = (
            spectrogram if spectrogram is not None else SpectrogramOptions()
        )
        self.visualization = (
            visualization if visualization is not None else VisualizationOptions()
        )
        self._round_idx: int = 0
        self.run_dir: Optional[Path] = None
        self._rounds_path: Optional[Path] = None
        self._gif_frames: dict[str, list[np.ndarray]] = {
            "frame": [],
            "occupancy": [],
        }

        if not self.enabled:
            return

        if run_dir is not None:
            self.run_dir = Path(run_dir)
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path(run_root) / f"run_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._rounds_path = self.run_dir / "rounds.jsonl"

        payload: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(self.run_dir.resolve()),
            "gif": asdict(self.gif),
            "spectrogram": asdict(self.spectrogram),
            "visualization": asdict(self.visualization),
        }
        if meta:
            payload.update(dict(meta))
        (self.run_dir / "meta.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        self._rounds_path.touch(exist_ok=True)

    def begin_round(self, round_idx: int) -> None:
        """Set the active round index used by subsequent ``save_stage`` calls."""
        if not self.enabled:
            return
        self._round_idx = int(round_idx)

    def _stage_matches_gif(self, stage: str) -> bool:
        if not self.gif.enabled:
            return False
        wanted = tuple(self.gif.stages)
        if not wanted:
            return True
        return _safe_stage(stage) in {_safe_stage(s) for s in wanted}

    def save_stage(
        self,
        stage: str,
        *,
        frame: Optional[np.ndarray] = None,
        occupancy: Optional[np.ndarray] = None,
    ) -> Optional[Path]:
        """Dump frame / occupancy under ``round_{rr:02d}_{stage}/``.

        Returns the stage directory, or ``None`` if disabled / nothing to write.
        """
        if not self.enabled or self.run_dir is None:
            return None
        if frame is None and occupancy is None:
            return None

        stage_dir = self.run_dir / f"round_{self._round_idx:02d}_{_safe_stage(stage)}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        if frame is not None:
            arr = np.asarray(frame)
            np.save(stage_dir / "frame.npy", arr)
            _write_png(stage_dir / "frame.png", arr)
            if self._stage_matches_gif(stage) and "frame" in self.gif.sources:
                self._gif_frames["frame"].append(
                    _resize_max_side(_to_uint8(arr), self.gif.max_side)
                )

        if occupancy is not None:
            occ = np.asarray(occupancy)
            np.save(stage_dir / "occupancy.npy", occ)
            heat = _occupancy_heatmap(occ, cell_px=self.gif.occupancy_cell_px)
            _write_png(stage_dir / "occupancy.png", heat)
            if self._stage_matches_gif(stage) and "occupancy" in self.gif.sources:
                self._gif_frames["occupancy"].append(
                    _resize_max_side(heat, self.gif.max_side)
                )

        if self.gif.enabled and self.gif.auto_write:
            self.write_gifs()

        return stage_dir

    def save_spectrogram(
        self,
        rf_batches: Sequence[Any],
        *,
        sample_rate_hz: Optional[float] = None,
        freq_min_hz: Optional[float] = None,
        freq_max_hz: Optional[float] = None,
    ) -> Optional[Path]:
        """Save AWG frequency/spectrogram PNG(s) (plus raw ``.npy``
        waveforms) under ``round_{rr:02d}_spectrogram/``, analyzing this
        round's AWG output from three angles:

        1. **Commanded frequency f(t)** — the pure, analytical ramp shape
           (``"linear"``/``"scurve"``, per ``SpectrogramOptions.ramp_shape``)
           each moving tone is programmed to follow. No waveform synthesis
           or FFT involved — this is the design intent
           (:func:`awg_controller.src.scapp.synthesize_round_frequency_trajectory`).
           A tone whose ``f_start`` doesn't match its own previous ``f_end``
           (e.g. ``RFConverter`` rebuilding a non-targeted tone's ramp from
           its nominal resting frequency rather than where an earlier batch
           actually left it) is drawn with a gap at that boundary instead of
           a connecting line, so a real commanded frequency jump isn't
           mistaken for a continuous ramp.
        2. **AWG output spectrogram (moves only)** — STFT of the synthesized
           waveform with holding tones silenced, so moving chirps aren't
           lost among ~dozens of stationary tones.
        3. **AWG output spectrogram (with holding)** — STFT of the full
           synthesized waveform, holds included — what the card actually
           streams to every tone, not just the moving ones.

        ``rf_batches`` is the ``List[AWGBatch]`` for one round (e.g. from
        ``RFConverter.convert_sequence``). No-ops unless both the recorder and
        ``self.spectrogram.enabled`` are true, or ``rf_batches`` is empty.
        ``matplotlib``/``scipy`` are soft dependencies of this method only.

        Frequency axis defaults to the AOD band via ``freq_min_hz`` /
        ``freq_max_hz`` (kwargs or :class:`SpectrogramOptions`), then ramp
        frequencies in ``rf_batches`` (moving-tone bounds for rows 1-2,
        all-tone bounds for row 3), then full Nyquist. Time axis spans the
        full synthesized waveform with units chosen from duration (µs/ms/s).

        By default (``SpectrogramOptions.separate_holding_stft=False``) all
        three views share one combined ``spectrogram.png``. When ``True``,
        row 3 gets its own frequency-optimized STFT window and all three
        views are written as separate files instead — see
        ``SpectrogramOptions.separate_holding_stft``. Set
        ``SpectrogramOptions.include_stft=False`` to skip rows 2-3 (and all
        waveform synthesis/STFT work) entirely, keeping only row 1.

        Returns the stage directory, or ``None`` if disabled / skipped.
        """
        if not self.enabled or self.run_dir is None or not self.spectrogram.enabled:
            return None
        if not rf_batches:
            return None

        opts = self.spectrogram
        rate = sample_rate_hz if sample_rate_hz is not None else opts.sample_rate_hz
        if opts.include_stft and rate is None:
            raise ValueError(
                "sample_rate_hz must be given (pass it to save_spectrogram(...) "
                "or set SpectrogramOptions.sample_rate_hz) when "
                "SpectrogramOptions.include_stft is True."
            )

        f_lo_opt = freq_min_hz if freq_min_hz is not None else opts.freq_min_hz
        f_hi_opt = freq_max_hz if freq_max_hz is not None else opts.freq_max_hz

        from awg_controller.src.scapp import synthesize_round_frequency_trajectory

        waveforms_no_hold: Dict[int, np.ndarray] = {}
        waveforms_with_hold: Dict[int, np.ndarray] = {}
        if opts.include_stft:
            from awg_controller.src.scapp import synthesize_round_waveform

            synth_no_hold = _rf_batches_moves_only(rf_batches)
            waveforms_no_hold = synthesize_round_waveform(
                synth_no_hold, rate, ramp_shape=opts.ramp_shape
            )
            waveforms_with_hold = synthesize_round_waveform(
                rf_batches, rate, ramp_shape=opts.ramp_shape
            )
            if not waveforms_with_hold:
                return None
            channels = sorted(waveforms_with_hold)
            duration_s = max(
                float(waveforms_with_hold[ch].size) / float(rate) for ch in channels
            )
        else:
            channels = sorted(
                {
                    int(ramp.channel)
                    for batch in rf_batches
                    for ramp in getattr(batch, "ramps", ()) or ()
                }
            )
            if not channels:
                return None
            duration_s = sum(
                max(float(getattr(b, "total_duration_s", 0.0) or 0.0), 0.0)
                for b in rf_batches
            )

        # Use an Agg canvas on standalone Figures so we never call
        # matplotlib.use("Agg") — that would permanently switch the process
        # backend and break interactive plt.show() in notebooks.
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            if opts.include_stft:
                from scipy.signal import spectrogram as _spectrogram
        except ImportError:
            log.warning("matplotlib/scipy not available; skipping spectrogram.")
            return None

        stage_dir = self.run_dir / f"round_{self._round_idx:02d}_spectrogram"
        stage_dir.mkdir(parents=True, exist_ok=True)

        labels = opts.channel_labels or {}
        n_ch = len(channels)

        t_scale, t_unit = _time_display_scale(duration_s)
        pad = float(opts.freq_pad_hz)
        # Wider for longer/busier rounds (many short chirps packed into one
        # axis are hard to tell apart at a fixed width) — scales mildly with
        # the displayed duration, clamped to a sane range.
        fig_width = min(max(14.0, 5.5 * n_ch, duration_s * t_scale * 0.12), 36.0)

        def _band_for(
            ch: int, moving_only: bool
        ) -> Tuple[Optional[float], Optional[float]]:
            lo, hi = f_lo_opt, f_hi_opt
            if lo is None or hi is None:
                bounds = _ramp_freq_bounds(
                    rf_batches, channel=ch, moving_only=moving_only
                )
                if bounds is not None:
                    lo = bounds[0] if lo is None else lo
                    hi = bounds[1] if hi is None else hi
            return lo, hi

        # Pure (FFT-free) commanded-frequency trajectory for row 1 — every
        # tone seen in the round is drawn, not just moving ones (a
        # never-moved row/col is real information, not clutter to hide);
        # moving tones are drawn last, bolder and in color, so they still
        # visually stand out against the dimmer non-moving ones.
        trajectories = synthesize_round_frequency_trajectory(
            rf_batches, ramp_shape=opts.ramp_shape
        )
        moving_keys = {
            (int(ramp.channel), int(ramp.tone_index))
            for batch in rf_batches
            for ramp in getattr(batch, "ramps", ()) or ()
            if _is_moving_ramp(ramp)
        }

        def _draw_traj(ax, ch: int, lo, hi) -> None:
            keys = sorted(k for k in trajectories if k[0] == ch)
            for key in keys:
                if key in moving_keys:
                    continue
                times, freqs = trajectories[key]
                ax.plot(
                    times * t_scale,
                    freqs / 1e6,
                    linewidth=0.6,
                    alpha=0.35,
                    color="0.6",
                    zorder=1,
                )
            for key in keys:
                if key not in moving_keys:
                    continue
                times, freqs = trajectories[key]
                ax.plot(
                    times * t_scale, freqs / 1e6, linewidth=1.2, alpha=0.85, zorder=3
                )
            if lo is not None and hi is not None and hi > lo:
                ax.set_ylim((lo - pad) / 1e6, (hi + pad) / 1e6)
            ax.set_xlim(0.0, duration_s * t_scale)
            ax.grid(True, alpha=0.3)

        if not opts.include_stft:
            fig = Figure(figsize=(fig_width, 3.6 + 1.0 * n_ch), layout="constrained")
            FigureCanvasAgg(fig)
            axes = fig.subplots(1, n_ch, squeeze=False)
            for col, ch in enumerate(channels):
                label = labels.get(ch, f"ch{ch}")
                lo_f, hi_f = _band_for(ch, False)
                _draw_traj(axes[0, col], ch, lo_f, hi_f)
                axes[0, col].set_title(f"Commanded frequency f(t) ({opts.ramp_shape})")
                axes[0, col].set_ylabel(f"{label}\nfreq (MHz)")
                axes[0, col].set_xlabel(f"time ({t_unit})")
            fig.suptitle(f"Round {self._round_idx} commanded frequency f(t)")
            fig.savefig(stage_dir / "spectrogram_f_t.png", dpi=200)
            return stage_dir

        def _stft(samples: np.ndarray, nperseg: int, noverlap: int):
            if samples.size == 0 or nperseg < 1:
                return None, None, None
            freqs, times, sxx = _spectrogram(
                samples,
                fs=rate,
                nperseg=nperseg,
                noverlap=noverlap,
                window="hann",
                scaling="spectrum",
            )
            return freqs, times, 10.0 * np.log10(sxx + 1e-300)

        def _nperseg_noverlap(n_samples: int, max_window: Optional[int]):
            nperseg = _choose_nperseg(
                rate,
                n_samples,
                nperseg=opts.nperseg,
                target_df_hz=opts.target_df_hz,
                max_window_samples=max_window,
            )
            if opts.noverlap is not None:
                noverlap = int(opts.noverlap)
            else:
                frac = min(max(float(opts.noverlap_frac), 0.0), 0.99)
                noverlap = int(round(nperseg * frac))
            return nperseg, min(noverlap, max(nperseg - 1, 0))

        # Cap the "moves only" STFT window so it doesn't span more than one
        # ramp/hold transition — otherwise a window sized purely for
        # target_df_hz resolution can be *longer than a short move itself*
        # (a single-site move can be a few µs at typical synthesis rates),
        # smearing the sweep into blocky/discrete-looking patches instead of
        # a smooth diagonal chirp. See _choose_nperseg docstring.
        min_batch_samples = _min_batch_samples(rf_batches, rate)
        max_window_samples = (
            max(int(min_batch_samples // max(int(opts.min_windows_per_ramp), 1)), 1)
            if min_batch_samples is not None
            else None
        )

        # First pass: STFT both variants per channel + dB clim(s) from
        # in-band peaks only (moves-only band for row 2, full band for row
        # 3). separate_holding_stft gives row 3 its own uncapped, purely
        # target_df_hz-driven window — static holds don't need row 2's fine
        # time resolution, and sharing that small window otherwise smears
        # closely-spaced holding tones together (frequency resolution
        # sample_rate_hz/nperseg can end up comparable to the AOD site
        # spacing). Not sharing the window means the two rows are no longer
        # directly time/frequency-comparable, so their color scales (and,
        # when separate_holding_stft, output files) are kept separate too.
        specs_no_hold: Dict[int, Tuple[Any, Any, Any]] = {}
        specs_with_hold: Dict[int, Tuple[Any, Any, Any]] = {}
        db_band_moves: list[np.ndarray] = []
        db_band_holding: list[np.ndarray] = []
        for ch in channels:
            samples_no_hold = waveforms_no_hold.get(ch, np.zeros(0, dtype=np.float64))
            samples_with_hold = waveforms_with_hold[ch]
            np.save(stage_dir / f"waveform_ch{ch}_no_holding.npy", samples_no_hold)
            np.save(stage_dir / f"waveform_ch{ch}_with_holding.npy", samples_with_hold)

            nperseg_m, noverlap_m = _nperseg_noverlap(
                samples_with_hold.size, max_window_samples
            )
            if opts.separate_holding_stft:
                nperseg_h, noverlap_h = _nperseg_noverlap(samples_with_hold.size, None)
            else:
                nperseg_h, noverlap_h = nperseg_m, noverlap_m

            freqs_nh, times_nh, db_nh = _stft(samples_no_hold, nperseg_m, noverlap_m)
            freqs_wh, times_wh, db_wh = _stft(samples_with_hold, nperseg_h, noverlap_h)
            specs_no_hold[ch] = (freqs_nh, times_nh, db_nh)
            specs_with_hold[ch] = (freqs_wh, times_wh, db_wh)

            for freqs, db, moving_only, pool in (
                (freqs_nh, db_nh, True, db_band_moves),
                (freqs_wh, db_wh, False, db_band_holding),
            ):
                if freqs is None or db is None:
                    continue
                lo, hi = _band_for(ch, moving_only)
                if lo is not None and hi is not None and hi > lo:
                    band = (freqs >= (lo - pad)) & (freqs <= (hi + pad))
                else:
                    band = np.ones(freqs.shape, dtype=bool)
                if np.any(band):
                    finite = db[band, :]
                    finite = finite[np.isfinite(finite)]
                    if finite.size:
                        pool.append(finite)

        db_range = max(float(opts.db_range), 1.0)

        def _clim(pool: list[np.ndarray]) -> Tuple[float, float]:
            if not pool:
                return -db_range, 0.0
            # Percentile peak (not absolute max) so a single hot bin doesn't
            # stretch the colorbar; floor is a fixed dynamic range below that.
            vmax = float(np.percentile(np.concatenate(pool), 99.0))
            return vmax - db_range, vmax

        if opts.separate_holding_stft:
            vmin_m, vmax_m = _clim(db_band_moves)
            vmin_h, vmax_h = _clim(db_band_holding)
        else:
            vmin_m, vmax_m = _clim(db_band_moves + db_band_holding)
            vmin_h, vmax_h = vmin_m, vmax_m

        def _draw_image(ax, freqs, times, db, lo, hi, vmin: float, vmax: float):
            if freqs is None or times is None or db is None or times.size == 0:
                return None
            if lo is not None and hi is not None and hi > lo:
                y0, y1 = (lo - pad) / 1e6, (hi + pad) / 1e6
                band = (freqs >= (lo - pad)) & (freqs <= (hi + pad))
            else:
                y0, y1 = float(freqs[0]) / 1e6, float(freqs[-1]) / 1e6
                band = np.ones(freqs.shape, dtype=bool)
            if not np.any(band):
                return None
            # scipy's `times` are STFT segment *centers*, offset from the true
            # signal edges by nperseg/(2*fs) — using them directly as the plot
            # extent silently crops up to half a window off each end. Use the
            # true [0, duration] span instead; imshow stretches the edge
            # columns to fill it, a good approximation since nperseg is
            # capped to stay well inside a single ramp (see max_window_samples).
            mesh = ax.imshow(
                db[band, :],
                origin="lower",
                aspect="auto",
                extent=(0.0, duration_s * t_scale, y0, y1),
                cmap=opts.cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="bilinear",
                rasterized=True,
            )
            ax.set_ylim(y0, y1)
            ax.set_xlim(0.0, duration_s * t_scale)
            ax.grid(False)
            return mesh

        if not opts.separate_holding_stft:
            fig = Figure(figsize=(fig_width, 10.5), layout="constrained")
            FigureCanvasAgg(fig)
            axes = fig.subplots(3, n_ch, squeeze=False, sharex=True)

            mesh = None
            for col, ch in enumerate(channels):
                label = labels.get(ch, f"ch{ch}")
                lo_m, hi_m = _band_for(ch, True)
                lo_f, hi_f = _band_for(ch, False)

                # Full band, not moving-only: row 1 now draws every tone, so
                # its y-limits must fit non-moving tones too.
                _draw_traj(axes[0, col], ch, lo_f, hi_f)
                axes[0, col].set_title(f"Commanded frequency f(t) ({opts.ramp_shape})")
                axes[0, col].set_ylabel(f"{label}\nfreq (MHz)")

                freqs_nh, times_nh, db_nh = specs_no_hold[ch]
                m = _draw_image(
                    axes[1, col], freqs_nh, times_nh, db_nh, lo_m, hi_m, vmin_m, vmax_m
                )
                mesh = m if m is not None else mesh
                axes[1, col].set_title("AWG output spectrogram (moves only)")
                axes[1, col].set_ylabel(f"{label}\nfreq (MHz)")

                freqs_wh, times_wh, db_wh = specs_with_hold[ch]
                m = _draw_image(
                    axes[2, col], freqs_wh, times_wh, db_wh, lo_f, hi_f, vmin_h, vmax_h
                )
                mesh = m if m is not None else mesh
                axes[2, col].set_title("AWG output spectrogram (with holding)")
                axes[2, col].set_ylabel(f"{label}\nfreq (MHz)")

            for col in range(n_ch):
                axes[-1, col].set_xlabel(f"time ({t_unit})")

            fig.suptitle(f"Round {self._round_idx} AWG frequency / spectrogram")
            if mesh is not None:
                cbar = fig.colorbar(
                    mesh, ax=axes[1:, :].ravel().tolist(), pad=0.02, fraction=0.03
                )
                cbar.set_label("power (dB)")
            fig.savefig(stage_dir / "spectrogram.png", dpi=200)
            return stage_dir

        # separate_holding_stft: three independent figures/files, each with
        # its own color scale (row 3's STFT uses a different window, so it's
        # no longer directly comparable to row 2 on one shared axis).
        def _new_row_figure():
            f = Figure(figsize=(fig_width, 3.6 + 1.0 * n_ch), layout="constrained")
            FigureCanvasAgg(f)
            return f, f.subplots(1, n_ch, squeeze=False)

        fig_f, axes_f = _new_row_figure()
        for col, ch in enumerate(channels):
            label = labels.get(ch, f"ch{ch}")
            # Full band, not moving-only: this panel draws every tone.
            lo_f, hi_f = _band_for(ch, False)
            _draw_traj(axes_f[0, col], ch, lo_f, hi_f)
            axes_f[0, col].set_title(f"Commanded frequency f(t) ({opts.ramp_shape})")
            axes_f[0, col].set_ylabel(f"{label}\nfreq (MHz)")
            axes_f[0, col].set_xlabel(f"time ({t_unit})")
        fig_f.suptitle(f"Round {self._round_idx} commanded frequency f(t)")
        fig_f.savefig(stage_dir / "spectrogram_f_t.png", dpi=200)

        fig_m, axes_m = _new_row_figure()
        mesh_m = None
        for col, ch in enumerate(channels):
            label = labels.get(ch, f"ch{ch}")
            lo_m, hi_m = _band_for(ch, True)
            freqs_nh, times_nh, db_nh = specs_no_hold[ch]
            m = _draw_image(
                axes_m[0, col], freqs_nh, times_nh, db_nh, lo_m, hi_m, vmin_m, vmax_m
            )
            mesh_m = m if m is not None else mesh_m
            axes_m[0, col].set_title("AWG output spectrogram (moves only)")
            axes_m[0, col].set_ylabel(f"{label}\nfreq (MHz)")
            axes_m[0, col].set_xlabel(f"time ({t_unit})")
        fig_m.suptitle(f"Round {self._round_idx} AWG output spectrogram (moves only)")
        if mesh_m is not None:
            cbar = fig_m.colorbar(
                mesh_m, ax=axes_m.ravel().tolist(), pad=0.02, fraction=0.03
            )
            cbar.set_label("power (dB)")
        fig_m.savefig(stage_dir / "spectrogram_moves_only.png", dpi=200)

        fig_h, axes_h = _new_row_figure()
        mesh_h = None
        for col, ch in enumerate(channels):
            label = labels.get(ch, f"ch{ch}")
            lo_f, hi_f = _band_for(ch, False)
            freqs_wh, times_wh, db_wh = specs_with_hold[ch]
            m = _draw_image(
                axes_h[0, col], freqs_wh, times_wh, db_wh, lo_f, hi_f, vmin_h, vmax_h
            )
            mesh_h = m if m is not None else mesh_h
            axes_h[0, col].set_title("AWG output spectrogram (with holding)")
            axes_h[0, col].set_ylabel(f"{label}\nfreq (MHz)")
            axes_h[0, col].set_xlabel(f"time ({t_unit})")
        fig_h.suptitle(f"Round {self._round_idx} AWG output spectrogram (with holding)")
        if mesh_h is not None:
            cbar = fig_h.colorbar(
                mesh_h, ax=axes_h.ravel().tolist(), pad=0.02, fraction=0.03
            )
            cbar.set_label("power (dB)")
        fig_h.savefig(stage_dir / "spectrogram_with_holding.png", dpi=200)

        return stage_dir

    def save_move_visualization(
        self,
        atom_array: Any,
        move_batches: Sequence[Sequence[Any]],
    ) -> Optional[Path]:
        """Render this round's move batches as step-by-step lattice
        snapshots (initial state + one panel per batch, with move arrows
        and failure markers) via ``atommovr.utils.imaging.visualization``,
        saved under ``round_{rr:02d}_visualization/`` — schematic lattice
        view only (no camera-image rendering).

        ``atom_array`` must be the array state as of the *start* of this
        round (before ``move_batches`` have been applied) — the underlying
        ``visualize_move_batches``/``render_move_batch_frames`` functions
        re-simulate the batches themselves through a scratch copy, so
        passing an already-mutated array would double-apply the moves and
        desync the snapshots from what actually happened.

        Always writes the static multi-panel ``grid.<grid_format>``; also
        writes an animated ``grid.gif`` cycling through the same snapshots
        when ``VisualizationOptions.gif`` is ``True`` (default).
        ``VisualizationOptions.max_batches`` caps how many batches get a
        panel/frame — a round with dozens of small parallel-move batches
        otherwise produces one enormous, unreadable figure (see that
        option's docstring).

        No-ops unless both the recorder and ``self.visualization.enabled``
        are true, or ``move_batches`` is empty. ``matplotlib`` is a soft
        dependency of this method only. Returns the stage directory, or
        ``None`` if disabled/skipped.
        """
        if not self.enabled or self.run_dir is None or not self.visualization.enabled:
            return None
        if not move_batches:
            return None

        try:
            from atommovr.utils.imaging.visualization import (
                render_move_batch_frames,
                visualize_move_batches,
            )
        except ImportError:
            log.warning("matplotlib not available; skipping move visualization.")
            return None

        opts = self.visualization
        batches = list(move_batches)
        if opts.max_batches is not None and len(batches) > opts.max_batches:
            log.info(
                f"Round {self._round_idx}: move visualization truncated to "
                f"the first {opts.max_batches} of {len(batches)} batches "
                "(VisualizationOptions.max_batches)."
            )
            batches = batches[: opts.max_batches]

        stage_dir = self.run_dir / f"round_{self._round_idx:02d}_visualization"
        stage_dir.mkdir(parents=True, exist_ok=True)

        visualize_move_batches(
            atom_array,
            batches,
            save_path=str(stage_dir / f"grid.{opts.grid_format}"),
            title_suffix=f"round_{self._round_idx:02d}",
            max_cols=opts.max_cols,
        )

        if opts.gif:
            frames = render_move_batch_frames(atom_array, batches)
            _write_gif(
                stage_dir / "grid.gif",
                frames,
                duration_s=opts.gif_duration_s,
                loop=opts.gif_loop,
            )

        return stage_dir

    def log_round(self, **stats: Any) -> None:
        """Append one JSON object to ``rounds.jsonl``."""
        if not self.enabled or self._rounds_path is None:
            return
        record = dict(stats)
        record.setdefault("round", self._round_idx)
        with self._rounds_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def write_gifs(self) -> dict[str, Path]:
        """Write accumulated GIF(s) under ``run_dir``. Returns written paths."""
        written: dict[str, Path] = {}
        if not self.enabled or self.run_dir is None or not self.gif.enabled:
            return written

        for source in self.gif.sources:
            frames = self._gif_frames.get(source) or []
            if len(frames) < 1:
                continue
            # frame → frames.gif; occupancy → occupancy.gif
            if source == "frame":
                out = self.run_dir / "frames.gif"
            elif source == "occupancy":
                out = self.run_dir / "occupancy.gif"
            else:
                out = self.run_dir / f"{source}.gif"

            if _write_gif(
                out,
                frames,
                duration_s=self.gif.duration_s,
                loop=self.gif.loop,
            ):
                written[source] = out
        return written

    def finalize(self) -> dict[str, Path]:
        """Flush GIFs (call at end of a controller run)."""
        return self.write_gifs()


def moves_to_records(move_batches: Any) -> list[dict[str, int]]:
    """Flatten parallel move batches into compact JSON-serializable dicts."""
    out: list[dict[str, int]] = []
    if not move_batches:
        return out
    for batch in move_batches:
        for m in batch:
            out.append(
                {
                    "fr": int(m.from_row),
                    "fc": int(m.from_col),
                    "tr": int(m.to_row),
                    "tc": int(m.to_col),
                }
            )
    return out
