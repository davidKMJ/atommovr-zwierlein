from typing import Tuple, List, Optional
import math
import os
import numpy as np
import matplotlib.pyplot as plt

from .geometry import rotate_points_ccw


def compute_scaled_image_shape(
    image_shape: Tuple[int, int],
    grid_size: int,
    min_spacing_px: float = 24.0,
) -> Tuple[int, int]:
    """Ensure each lattice site has at least ``min_spacing_px`` separation.

    The default spacing of 24 pixels keeps the synthetic imagery within the
    response range of the blob detector even for 100x100+ grids, which need a
    higher pixel density than the legacy 12 px baseline."""
    min_spacing_px = max(2.0, float(min_spacing_px))
    required_extent = int(math.ceil((grid_size + 1) * min_spacing_px))
    height = int(max(required_extent, int(image_shape[0])))
    width = int(max(required_extent, int(image_shape[1])))
    return height, width


def gaussian_2d(x: np.ndarray, y: np.ndarray, x0: float, y0: float, sigma: float):
    """
    Generate a 2D Gaussian centered at (x0, y0) with standard deviation sigma.
    Returns a 2D array.
    """
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


def generate_gaussian_image_from_binary_grid(
    binary_grid: np.ndarray,
    sigma: float = 1.0,
    brightness_factor: float = 1.0,
    image_shape: Tuple[int, int] = (64, 128),
    noise_level: float = 0.01,
    stripe_intensity: float = 0.001,
    angle: float = 0.0,
):
    """
    Generate a 2D Gaussian image from a binary grid of atom occupancy.
    """
    grid_ones = np.argwhere(binary_grid == 1)
    n_points = len(grid_ones)
    grid_shape = binary_grid.shape

    if n_points == 0:
        return np.zeros(image_shape, dtype=float)

    points = []
    start_x = image_shape[0] // grid_shape[0]
    start_y = image_shape[1] // grid_shape[1]
    spacing_x = image_shape[0] // (grid_shape[0] + 1)
    spacing_y = image_shape[1] // (grid_shape[1] + 1)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if binary_grid[i, j] == 1:
                px = start_x + i * spacing_x
                py = start_y + j * spacing_y
                points.append((px, py))

    # handle scalar or per-point sigma/brightness
    if np.ndim(sigma) == 0:
        sigmas = [float(sigma)] * n_points
    else:
        s = np.asarray(sigma, dtype=float)
        if s.size != n_points:
            raise ValueError("sigma length must equal number of ones in binary_grid")
        sigmas = list(s)

    if np.ndim(brightness_factor) == 0:
        brightness_factors = [float(brightness_factor)] * n_points
    else:
        b = np.asarray(brightness_factor, dtype=float)
        if b.size != n_points:
            raise ValueError(
                "brightness_factor length must equal number of ones in binary_grid"
            )
        brightness_factors = list(b)

    return generate_gaussian_image(
        points,
        sigmas,
        brightness_factors,
        image_shape,
        noise_level,
        stripe_intensity,
        angle=angle,
    )


def generate_gaussian_image(
    points: List[Tuple[int, int]],
    sigmas: List[float],
    brightness_factors: List[float],
    image_shape: Tuple[int, int] = (64, 128),
    noise_level: float = 0.04,
    stripe_intensity: float = 0.005,
    angle: float = 0.0,
):
    """
    Generate a 2D Gaussian image with specified parameters, planting Gaussians at
    positions rotated by `angle` (counter-clockwise positive) around the image centre. Uses padded canvas to
    avoid clipping when rotating.
    Render Gaussian spots using local patches to avoid O(N_px · N_spot) cost.
    """
    img = render_gaussian_spots(
        points, sigmas, brightness_factors, image_shape, angle=angle
    )
    return add_sensor_noise(
        img, noise_level=noise_level, stripe_intensity=stripe_intensity
    )


def generate_rot_img(
    image_shape: Tuple[int, int],
    grid_size: int,
    true_angle: float,
    suffix: Optional[str],
    directory: str = "output",
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Generate a rotated Gaussian grid image and save it.

    Parameters
    ----------
    image_shape : Tuple[int, int]
        Size of the generated image (height, width).
    grid_size : int
        Size of the grid (grid_size x grid_size).
    true_angle : float
        Rotation angle in degrees (counter-clockwise positive).
    suffix : Optional[str]
        Suffix for the saved image filenames.
    directory : str
        Directory to save the images.

    Returns
    -------
    points : List[Tuple[int, int]]
        List of (x, y) coordinates of the Gaussian centers.
    true_binary : np.ndarray
        Binary grid indicating presence (1) or absence (0) of Gaussians.
    """
    image_shape = compute_scaled_image_shape(image_shape, grid_size)
    grid_shape = (grid_size, grid_size)
    points: List[Tuple[int, int]] = []
    start_x = image_shape[0] // grid_shape[0]
    start_y = image_shape[1] // grid_shape[1]
    spacing_x = image_shape[0] // (grid_shape[0] + 1)
    spacing_y = image_shape[1] // (grid_shape[1] + 1)
    true_binary = np.zeros(grid_shape, dtype=int)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if np.random.rand() > 0.4:
                points.append((start_x + i * spacing_x, start_y + j * spacing_y))
                true_binary[i, j] = 1

    sigmas = np.random.normal(1.5, 0.01, len(points))
    brightness_factors = np.random.uniform(0.8, 1.0, len(points))

    # Create the unrotated image (for reference) and the rotated image
    gaussian_img = generate_gaussian_image(
        points, sigmas, brightness_factors, image_shape, angle=0.0
    )
    rot_image = generate_gaussian_image(
        points, sigmas, brightness_factors, image_shape, angle=true_angle
    )

    os.makedirs(directory, exist_ok=True)

    plt.imsave(f"{directory}/{suffix}_image.png", gaussian_img, cmap="Blues")
    plt.imsave(f"{directory}/{suffix}_rot_image.png", rot_image, cmap="Blues")
    print(
        f"Saved images to {directory}/{suffix}_image.png and {directory}/{suffix}_rot_image.png"
    )
    return points, true_binary


def compute_spacing_image_shape(
    grid_shape: Tuple[int, int],
    spacing_x: float,
    spacing_y: Optional[float] = None,
) -> Tuple[int, int]:
    """Return the (height, width) needed to hold ``grid_shape`` lattice sites
    at explicit per-axis pixel spacing.

    This is the inverse of :func:`compute_scaled_image_shape`: instead of
    deriving spacing from a target image size, it derives the minimal
    "atom-bearing" image extent from a requested spacing, mirroring the
    ``start == spacing`` placement convention used by
    ``generate_gaussian_image_from_binary_grid`` (site ``i`` sits at
    ``spacing_x * (i + 1)``, leaving one spacing of margin on each edge).
    """
    if spacing_y is None:
        spacing_y = spacing_x
    spacing_x = max(1.0, float(spacing_x))
    spacing_y = max(1.0, float(spacing_y))
    height = int(math.ceil(spacing_x * (grid_shape[0] + 1)))
    width = int(math.ceil(spacing_y * (grid_shape[1] + 1)))
    return height, width


def render_gaussian_spots(
    points: List[Tuple[float, float]],
    sigmas: List[float],
    brightness_factors: List[float],
    image_shape: Tuple[int, int],
    angle: float = 0.0,
) -> np.ndarray:
    """Noiseless counterpart of :func:`generate_gaussian_image`.

    Rotates and renders Gaussian spots onto a padded canvas exactly as
    ``generate_gaussian_image`` does (the canvas may be a little larger
    than ``image_shape`` — a few pixels of rotation-safety margin — as it
    is not cropped here), but stops before the sensor-noise /
    interference-stripe step, so callers can add noise separately (see
    :func:`add_sensor_noise`) and/or re-center the render onto a larger
    sensor canvas (see :func:`pad_image_to_shape`) — decoupling the
    lattice's pixel spacing from the final image size.
    """
    h, w = image_shape
    angle_rad = np.deg2rad(angle)
    cos_a = abs(np.cos(angle_rad))
    sin_a = abs(np.sin(angle_rad))
    new_w = h * sin_a + w * cos_a
    new_h = h * cos_a + w * sin_a
    base_pad_w = max(int(math.ceil((new_w - w) / 2.0)) + 2, 0)
    base_pad_h = max(int(math.ceil((new_h - h) / 2.0)) + 2, 0)

    n_points = len(points)
    if np.ndim(sigmas) == 0:
        sigmas = [float(sigmas)] * n_points
    else:
        sigmas = list(np.asarray(sigmas, dtype=float))
    if np.ndim(brightness_factors) == 0:
        brightness_factors = [float(brightness_factors)] * n_points
    else:
        brightness_factors = list(np.asarray(brightness_factors, dtype=float))
    if n_points:
        base_points = np.asarray(points, dtype=float).reshape(n_points, 2)
        rotated_points = rotate_points_ccw(base_points, image_shape, angle)

        safety_margin = max(2.0, 0.002 * max(h, w))
        min_y = float(np.min(rotated_points[:, 0])) if len(rotated_points) else 0.0
        max_y = float(np.max(rotated_points[:, 0])) if len(rotated_points) else 0.0
        min_x = float(np.min(rotated_points[:, 1])) if len(rotated_points) else 0.0
        max_x = float(np.max(rotated_points[:, 1])) if len(rotated_points) else 0.0

        pad_top = int(math.ceil(max(base_pad_h, -min_y + safety_margin)))
        pad_bottom = int(math.ceil(max(base_pad_h, max_y - h + safety_margin)))
        pad_left = int(math.ceil(max(base_pad_w, -min_x + safety_margin)))
        pad_right = int(math.ceil(max(base_pad_w, max_x - w + safety_margin)))

        canvas_h = h + pad_top + pad_bottom
        canvas_w = w + pad_left + pad_right
        rotated_points = rotated_points + np.array([pad_top, pad_left], dtype=float)
    else:
        pad_top = pad_bottom = base_pad_h
        pad_left = pad_right = base_pad_w
        canvas_h = h + pad_top + pad_bottom
        canvas_w = w + pad_left + pad_right
        rotated_points = np.empty((0, 2), dtype=float)

    x = np.arange(canvas_w)
    y = np.arange(canvas_h)
    X, Y = np.meshgrid(x, y)
    canvas = np.zeros((canvas_h, canvas_w), dtype=float)

    for (y_canvas, x_canvas), sigma, brightness in zip(
        rotated_points, sigmas, brightness_factors
    ):
        canvas += float(brightness) * gaussian_2d(
            X, Y, float(x_canvas), float(y_canvas), float(sigma)
        )

    return canvas


def pad_image_to_shape(
    img: np.ndarray,
    image_shape: Tuple[int, int],
    fill_value: float = 0.0,
) -> np.ndarray:
    """Center ``img`` onto a canvas of ``image_shape``.

    ``image_shape`` may be larger than ``img`` on either axis (the common
    case here: a small atom-bearing region padded onto a bigger sensor
    frame) or smaller (in which case ``img`` is center-cropped)."""
    h, w = int(image_shape[0]), int(image_shape[1])
    ih, iw = img.shape[:2]
    out = np.full((h, w), float(fill_value), dtype=float)

    copy_h = min(h, ih)
    copy_w = min(w, iw)
    y_dst = max(0, (h - ih) // 2)
    x_dst = max(0, (w - iw) // 2)
    y_src = max(0, (ih - h) // 2)
    x_src = max(0, (iw - w) // 2)

    out[y_dst : y_dst + copy_h, x_dst : x_dst + copy_w] = img[
        y_src : y_src + copy_h, x_src : x_src + copy_w
    ]
    return out


def add_sensor_noise(
    img: np.ndarray,
    noise_level: float = 0.04,
    stripe_intensity: float = 0.005,
) -> np.ndarray:
    """Add Gaussian sensor noise and horizontal interference stripes.

    Matches the noise/stripe tail of :func:`generate_gaussian_image`,
    exposed separately so it can be applied after :func:`pad_image_to_shape`
    puts a noiseless render onto the full sensor canvas."""
    noise = np.random.normal(0, noise_level, img.shape)
    out = img + noise
    xvec = np.arange(out.shape[1])
    stripe_pattern = stripe_intensity * np.sin(2 * np.pi * xvec / 20.0)
    out += stripe_pattern[np.newaxis, :]
    return out


def generate_gaussian_image_with_padding(
    points: List[Tuple[float, float]],
    sigmas: List[float],
    brightness_factors: List[float],
    core_shape: Tuple[int, int],
    image_shape: Optional[Tuple[int, int]] = None,
    noise_level: float = 0.04,
    stripe_intensity: float = 0.005,
    angle: float = 0.0,
) -> np.ndarray:
    """Points-based counterpart to :func:`generate_gaussian_image` that
    decouples the lattice's rendering size from the final image size.

    Gaussians are rendered (rotated, noiseless) onto a ``core_shape``
    canvas — sized to just fit the lattice, e.g. via
    :func:`compute_spacing_image_shape` — which is then centered onto a
    (typically larger) ``image_shape`` sensor canvas before noise and
    interference stripes are added. ``image_shape`` defaults to
    ``core_shape`` and is expanded (never shrunk) to fit it if smaller.
    """
    if image_shape is None:
        image_shape = core_shape
    else:
        image_shape = (
            max(int(image_shape[0]), int(core_shape[0])),
            max(int(image_shape[1]), int(core_shape[1])),
        )

    core_img = render_gaussian_spots(
        points, sigmas, brightness_factors, core_shape, angle=angle
    )
    padded = pad_image_to_shape(core_img, image_shape)
    return add_sensor_noise(
        padded, noise_level=noise_level, stripe_intensity=stripe_intensity
    )


def generate_gaussian_image_from_binary_grid_with_spacing(
    binary_grid: np.ndarray,
    spacing_x: float,
    spacing_y: Optional[float] = None,
    sigma: float = 1.0,
    brightness_factor: float = 1.0,
    image_shape: Optional[Tuple[int, int]] = None,
    noise_level: float = 0.01,
    stripe_intensity: float = 0.001,
    angle: float = 0.0,
) -> np.ndarray:
    """Like :func:`generate_gaussian_image_from_binary_grid`, but takes
    explicit per-axis pixel spacing instead of deriving it from
    ``image_shape``.

    The lattice is rendered onto a canvas sized just large enough to hold
    it at the requested ``spacing_x``/``spacing_y`` (see
    :func:`compute_spacing_image_shape`), then centered onto ``image_shape``
    — a sensor frame that may be considerably bigger than the atom-bearing
    region — before noise and interference stripes are added. If
    ``image_shape`` is omitted (or smaller than the lattice needs), it
    defaults to (grows to fit) the lattice's own extent.
    """
    if spacing_y is None:
        spacing_y = spacing_x
    grid_shape = binary_grid.shape
    core_shape = compute_spacing_image_shape(grid_shape, spacing_x, spacing_y)

    grid_ones = np.argwhere(binary_grid == 1)
    n_points = len(grid_ones)

    if n_points == 0:
        target_shape = (
            (int(image_shape[0]), int(image_shape[1]))
            if image_shape is not None
            else core_shape
        )
        target_shape = (
            max(target_shape[0], core_shape[0]),
            max(target_shape[1], core_shape[1]),
        )
        return np.zeros(target_shape, dtype=float)

    points = []
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if binary_grid[i, j] == 1:
                px = spacing_x * (i + 1)
                py = spacing_y * (j + 1)
                points.append((px, py))

    # handle scalar or per-point sigma/brightness
    if np.ndim(sigma) == 0:
        sigmas = [float(sigma)] * n_points
    else:
        s = np.asarray(sigma, dtype=float)
        if s.size != n_points:
            raise ValueError("sigma length must equal number of ones in binary_grid")
        sigmas = list(s)

    if np.ndim(brightness_factor) == 0:
        brightness_factors = [float(brightness_factor)] * n_points
    else:
        b = np.asarray(brightness_factor, dtype=float)
        if b.size != n_points:
            raise ValueError(
                "brightness_factor length must equal number of ones in binary_grid"
            )
        brightness_factors = list(b)

    return generate_gaussian_image_with_padding(
        points,
        sigmas,
        brightness_factors,
        core_shape,
        image_shape,
        noise_level,
        stripe_intensity,
        angle=angle,
    )
