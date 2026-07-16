"""Camera interfaces for real and offline atom-array imaging.

A ``Camera`` never owns an ``AtomArray`` — the controller holds the single
authoritative array and passes it into ``sync()`` each round. ``sync()``
reconciles the camera with that array; the direction of the reconciliation
depends on the camera type (see each subclass's docstring).

Optional stage dumps are driven by a ``SessionRecorder`` passed into
``sync`` / ``_measure_into`` (owned by the controller, not the camera).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

import numpy as np

from atommovr.utils.AtomArray import AtomArray
from atommovr.utils.core import PhysicalParams, random_loading
from atommovr.utils.imaging.extraction import (
    BlobDetection,
    estimate_grid_rotation_fit_rect,
    fit_grid_and_assign,
    inverse_rotate_centroids,
)
from atommovr.utils.imaging.generation import (
    compute_scaled_image_shape,
    generate_gaussian_image_from_binary_grid,
    generate_gaussian_image_from_binary_grid_with_spacing,
)

if TYPE_CHECKING:
    from awg_controller.src.session_recorder import SessionRecorder

#: occupancy (rows, cols) -> grayscale camera frame
ImageGenerator = Callable[[np.ndarray], np.ndarray]


class Camera(abc.ABC):
    """Shared interface for image acquisition + occupancy detection.

    Subclasses implement ``acquire()`` (a raw grayscale frame) and
    ``sync(array, recorder=...)`` (reconcile with the controller's global
    ``AtomArray``). ``detect_occupancy`` is a concrete, shared
    blob-detect -> rotation-correct -> grid-assign pipeline used by both.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        blob_params: Optional[Any] = None,
    ) -> None:
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.blob_params = blob_params
        self.grid_rotation: float = 0.0

    @abc.abstractmethod
    def acquire(self) -> np.ndarray:
        """Return a raw grayscale frame."""

    @abc.abstractmethod
    def sync(
        self,
        array: AtomArray,
        recorder: Optional["SessionRecorder"] = None,
    ) -> None:
        """Reconcile this camera's occupancy with ``array``."""

    @staticmethod
    def _maybe_record(
        recorder: Optional["SessionRecorder"],
        stage: str,
        *,
        frame: Optional[np.ndarray] = None,
        occupancy: Optional[np.ndarray] = None,
    ) -> None:
        """No-op unless a ``SessionRecorder`` is passed."""
        if recorder is not None:
            recorder.save_stage(stage, frame=frame, occupancy=occupancy)

    def detect_occupancy(self, image: np.ndarray) -> np.ndarray:
        """Blob detect -> rotation correct -> grid assign.

        Returns a binary occupancy matrix, shape ``grid_shape``, dtype int.
        Updates ``self.grid_rotation`` with the last estimated angle.
        """
        blob_params = self.blob_params
        if isinstance(blob_params, dict):
            blob_params = None  # OpenCV expects a SimpleBlobDetector_Params object

        detector = BlobDetection(shape=self.grid_shape, blob_params=blob_params)
        centroids, _ = detector.extract(image)
        if len(centroids) == 0:
            return np.zeros(self.grid_shape, dtype=int)

        rotation = estimate_grid_rotation_fit_rect(centroids)
        self.grid_rotation = rotation

        if abs(rotation) > 0.01:
            centroids = inverse_rotate_centroids(
                centroids, image_shape=image.shape[:2], angle_deg=rotation
            )

        return fit_grid_and_assign(
            centroids,
            self.grid_shape,
            image_shape=image.shape[:2],
        )

    def _measure_into(
        self,
        array: AtomArray,
        recorder: Optional["SessionRecorder"] = None,
    ) -> np.ndarray:
        """Acquire + detect, then write the reading into ``array.matrix``."""
        image = self.acquire()
        self._maybe_record(recorder, "acquire", frame=image)
        occ = self.detect_occupancy(image)
        self._maybe_record(recorder, "detect", frame=image, occupancy=occ)
        array.matrix[:, :, 0] = occ
        return occ


class RealArrayCamera(Camera):
    """Adapter wrapping a raw hardware callback in the ``Camera`` interface.

    ``sync(array)``: takes a fresh measurement (acquire + detect) and
    writes it into ``array`` — the array is always updated *from* the
    camera, since the physical world is the source of truth.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        camera_fn,
        blob_params: Optional[Any] = None,
    ) -> None:
        super().__init__(grid_shape, blob_params)
        self._camera_fn = camera_fn

    def acquire(self) -> np.ndarray:
        return self._camera_fn()

    def sync(
        self,
        array: AtomArray,
        recorder: Optional["SessionRecorder"] = None,
    ) -> None:
        self._measure_into(array, recorder=recorder)


@dataclass
class GaussianCameraConfig:
    """Camera-like knobs for synthetic fluorescence images.

    By default, pixel lattice spacing is controlled by ``image_shape`` and
    ``min_spacing_px`` (via ``compute_scaled_image_shape``). Set
    ``spacing_x`` (and optionally ``spacing_y``, which defaults to
    ``spacing_x``) to instead pin the lattice pitch directly in pixels,
    independent of ``image_shape`` — see
    ``generate_gaussian_image_from_binary_grid_with_spacing``. Either way,
    ``image_shape`` is the sensor frame the lattice gets centered onto, and
    may be considerably larger than the atom-bearing region. Physical trap
    spacing (µm/m) lives on ``PhysicalParams`` and is independent of both.
    """

    image_shape: Tuple[int, int] = (624, 816)
    sigma_px: float = 2.0
    peak_counts: float = 200.0
    background: float = 10.0
    noise_level: float = 0.02
    min_spacing_px: float = 24.0
    spacing_x: Optional[float] = None
    spacing_y: Optional[float] = None
    angle: float = 0.0
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.uint8))
    stripe_intensity: float = 0.001

    def resolve_shape(self, grid_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Return image shape large enough for ``grid_shape`` at ``min_spacing_px``.

        Only used on the legacy (``spacing_x is None``) path."""
        n = max(int(grid_shape[0]), int(grid_shape[1]))
        return compute_scaled_image_shape(
            self.image_shape, n, min_spacing_px=self.min_spacing_px
        )

    def __call__(self, occupancy: np.ndarray) -> np.ndarray:
        """Render a grayscale frame from a binary occupancy grid."""
        binary = np.asarray(occupancy, dtype=int)
        if binary.ndim != 2:
            raise ValueError(f"occupancy must be 2-D; got shape {binary.shape}")

        if self.spacing_x is not None:
            img = generate_gaussian_image_from_binary_grid_with_spacing(
                binary,
                spacing_x=self.spacing_x,
                spacing_y=self.spacing_y,
                sigma=self.sigma_px,
                brightness_factor=float(self.peak_counts),
                image_shape=self.image_shape,
                noise_level=self.noise_level,
                stripe_intensity=self.stripe_intensity,
                angle=self.angle,
            )
            out = np.asarray(img, dtype=float) + float(self.background)
        else:
            shape = self.resolve_shape(binary.shape)
            img = generate_gaussian_image_from_binary_grid(
                binary,
                sigma=self.sigma_px,
                brightness_factor=float(self.peak_counts),
                image_shape=shape,
                noise_level=self.noise_level,
                stripe_intensity=self.stripe_intensity,
                angle=self.angle,
            )
            img = np.asarray(img, dtype=float)
            h, w = shape
            out = np.full(shape, float(self.background), dtype=float)
            hh, ww = img.shape[:2]
            y0 = max(0, (hh - h) // 2)
            x0 = max(0, (ww - w) // 2)
            crop = img[y0 : y0 + h, x0 : x0 + w]
            out[: crop.shape[0], : crop.shape[1]] = crop + float(self.background)

        info = np.iinfo(self.dtype) if np.issubdtype(self.dtype, np.integer) else None
        if info is not None:
            out = np.clip(out, info.min, info.max)
            return out.astype(self.dtype)
        return out.astype(self.dtype)


class OfflineArrayCamera(Camera):
    """Stateful offline camera: occupancy -> image, array truth -> occupancy.

    Round 0 renders from ``initial_occupancy`` or a Bernoulli load. Physics
    simulation (move transport + error model) happens on the controller's
    own ``AtomArray``; call ``sync(array)`` after each rearrangement round
    so the next ``acquire()`` reflects the controller's post-move ground
    truth. ``sync()`` also writes a fresh (detected) reading back into
    ``array``, so offline mode exercises the same imaging pipeline a real
    camera would.

    Defaults to ``GaussianCameraConfig()``; pass e.g.
    ``image_generator=GaussianCameraConfig(spacing_x=..., spacing_y=...)``
    to pin the lattice's pixel spacing directly instead of deriving it from
    ``image_shape``/``min_spacing_px``.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        *,
        image_generator: Optional[ImageGenerator] = None,
        physical_params: Optional[PhysicalParams] = None,
        initial_occupancy: Optional[np.ndarray] = None,
        loading_prob: Optional[float] = None,
        seed: Optional[int] = None,
        blob_params: Optional[Any] = None,
    ) -> None:
        super().__init__(grid_shape, blob_params)
        self.params = physical_params or PhysicalParams()
        self.image_generator: ImageGenerator = image_generator or GaussianCameraConfig()
        self._rng = np.random.default_rng(seed)
        self._loading_prob = (
            float(loading_prob)
            if loading_prob is not None
            else float(self.params.loading_prob)
        )

        if initial_occupancy is not None:
            occ = np.asarray(initial_occupancy, dtype=int)
            if occ.shape != self.grid_shape:
                raise ValueError(
                    f"initial_occupancy shape {occ.shape} != grid_shape {self.grid_shape}"
                )
            self._occupancy = (occ > 0).astype(int)
        else:
            self._occupancy = None  # lazy-init on first acquire

        self._synced = False

    @property
    def occupancy(self) -> Optional[np.ndarray]:
        """Current binary occupancy, or ``None`` before the first acquire."""
        return None if self._occupancy is None else self._occupancy.copy()

    def _ensure_occupancy(self) -> np.ndarray:
        if self._occupancy is None:
            self._occupancy = random_loading(
                list(self.grid_shape), self._loading_prob, rng=self._rng
            ).astype(int)
        return self._occupancy

    def acquire(self) -> np.ndarray:
        """Return a fluorescence frame for the current occupancy."""
        occ = self._ensure_occupancy()
        return self.image_generator(occ)

    def sync(
        self,
        array: AtomArray,
        recorder: Optional["SessionRecorder"] = None,
    ) -> None:
        """Reconcile with the controller's global array.

        After the first call, absorbs ``array``'s ground truth (as
        advanced by ``array.evaluate_moves``) into this camera's occupancy
        before rendering, so the next frame reflects the real post-move
        state. Always takes a fresh (detected) reading and writes it back
        into ``array``, matching what a real camera would provide.
        """
        if self._synced:
            self.set_occupancy(array.matrix[:, :, 0])
        self._measure_into(array, recorder=recorder)
        self._synced = True

    def set_occupancy(self, occupancy: np.ndarray) -> None:
        """Overwrite the simulated occupancy (e.g. after external processing)."""
        occ = np.asarray(occupancy, dtype=int)
        if occ.shape != self.grid_shape:
            raise ValueError(
                f"occupancy shape {occ.shape} != grid_shape {self.grid_shape}"
            )
        self._occupancy = (occ > 0).astype(int)
