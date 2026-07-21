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
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

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


@dataclass
class SpectrogramOptions:
    """Customization for optional per-round AWG output spectrograms.

    Off by default — unlike ``GifOptions``, this re-synthesizes the full
    per-channel time-domain waveform for every ramp in a round
    (``awg_controller.src.scapp.synthesize_round_waveform``) and needs
    ``scipy``/``matplotlib``, so it is heavier than the other
    :class:`SessionRecorder` outputs.

    Parameters
    ----------
    enabled
        When ``True``, :meth:`SessionRecorder.save_spectrogram` writes a PNG
        (and raw ``.npy`` waveforms) per round; otherwise it no-ops.
    sample_rate_hz
        Synthesis sample rate (Hz). Must exceed 2x the highest AOD tone
        frequency in play (Nyquist) or the spectrogram will alias. ``None``
        requires callers to pass ``sample_rate_hz=`` explicitly to
        ``save_spectrogram`` instead.
    ramp_shape
        Frequency-ramp shape used to synthesize non-static moves
        (``"linear"`` | ``"scurve"``) — should match the AWG backend's
        actual ramp shape (``AODSettings.ramp_shape`` for scapp).
    nperseg
        ``scipy.signal.spectrogram`` window length (samples), clamped to the
        waveform length if shorter.
    channel_labels
        Optional ``{channel: label}`` for subplot titles (e.g.
        ``{0: "V/row", 1: "H/col"}``); defaults to ``"ch{n}"``.
    """

    enabled: bool = False
    sample_rate_hz: Optional[float] = None
    ramp_shape: str = "linear"
    nperseg: int = 256
    channel_labels: Optional[Mapping[int, str]] = None


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
    ) -> None:
        self.enabled = bool(enabled)
        self.gif = gif if gif is not None else GifOptions()
        self.spectrogram = (
            spectrogram if spectrogram is not None else SpectrogramOptions()
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
    ) -> Optional[Path]:
        """Synthesize this round's per-channel AWG output waveform and save a
        spectrogram PNG (plus raw ``.npy`` waveforms) under
        ``round_{rr:02d}_spectrogram/``.

        ``rf_batches`` is the ``List[AWGBatch]`` for one round (e.g. from
        ``RFConverter.convert_sequence``). No-ops unless both the recorder and
        ``self.spectrogram.enabled`` are true, or ``rf_batches`` is empty.
        ``matplotlib``/``scipy`` are soft dependencies of this method only.

        Returns the stage directory, or ``None`` if disabled / skipped.
        """
        if not self.enabled or self.run_dir is None or not self.spectrogram.enabled:
            return None
        if not rf_batches:
            return None

        rate = (
            sample_rate_hz
            if sample_rate_hz is not None
            else self.spectrogram.sample_rate_hz
        )
        if rate is None:
            raise ValueError(
                "sample_rate_hz must be given (pass it to save_spectrogram(...) "
                "or set SpectrogramOptions.sample_rate_hz)."
            )

        from awg_controller.src.scapp import synthesize_round_waveform

        waveforms = synthesize_round_waveform(
            rf_batches, rate, ramp_shape=self.spectrogram.ramp_shape
        )
        if not waveforms:
            return None

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from scipy.signal import spectrogram as _spectrogram
        except ImportError:
            log.warning("matplotlib/scipy not available; skipping spectrogram.")
            return None

        stage_dir = self.run_dir / f"round_{self._round_idx:02d}_spectrogram"
        stage_dir.mkdir(parents=True, exist_ok=True)

        channels = sorted(waveforms)
        labels = self.spectrogram.channel_labels or {}
        fig, axes = plt.subplots(
            len(channels), 1, figsize=(8, 3 * len(channels)), squeeze=False, sharex=True
        )
        for ax, ch in zip(axes[:, 0], channels):
            samples = waveforms[ch]
            np.save(stage_dir / f"waveform_ch{ch}.npy", samples)
            nperseg = min(self.spectrogram.nperseg, max(int(samples.size), 1))
            if samples.size == 0 or nperseg < 1:
                continue
            freqs, times, sxx = _spectrogram(samples, fs=rate, nperseg=nperseg)
            ax.pcolormesh(
                times * 1e6, freqs / 1e6, 10 * np.log10(sxx + 1e-300), shading="auto"
            )
            ax.set_ylabel(f"{labels.get(ch, f'ch{ch}')}\nfreq (MHz)")
        axes[-1, 0].set_xlabel("time (µs)")
        fig.suptitle(f"Round {self._round_idx} AWG output spectrogram")
        fig.tight_layout()
        out_path = stage_dir / "spectrogram.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

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
