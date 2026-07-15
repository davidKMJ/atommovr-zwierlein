"""Optional session recorder for camera stage dumps and round stats.

Off by default — attach a ``SessionRecorder`` to ``atommovrController``
(``recorder=``) and the controller passes it into ``Camera.sync`` /
``log_round`` each loop. When ``enabled=False`` or unused, no I/O.

Writes:

* per-stage folders: ``round_{rr:02d}_{stage}/frame.png``, ``occupancy.npy``, …
* append-only ``rounds.jsonl`` for move / RF statistics
* optional GIFs: ``frames.gif`` / ``occupancy.gif`` (see :class:`GifOptions`)
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

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
    """

    def __init__(
        self,
        run_root: PathLike = "runs",
        *,
        enabled: bool = True,
        run_dir: Optional[PathLike] = None,
        meta: Optional[Mapping[str, Any]] = None,
        gif: Optional[GifOptions] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.gif = gif if gif is not None else GifOptions()
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
