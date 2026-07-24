from __future__ import annotations
from typing import Optional, Tuple, List, Dict
import os
import numpy as np
import matplotlib.pyplot as plt

from atommovr.utils.imaging.generation import generate_gaussian_image_from_binary_grid
from atommovr.utils.imaging.extraction import BlobDetection
from atommovr.utils.Move import Move
from atommovr.utils.move_utils import MoveType

#: (color, marker size, linewidth, zorder, label) per failure-marker kind,
#: shared by _draw_grid_panel. zorder matches the original per-marker
#: values (crossed/collision drawn above the others).
_FAILURE_MARKER_STYLE: Dict[str, Tuple[str, int, float, int, str]] = {
    "pickup": ("gold", 40, 1.8, 12, "pickup fail"),
    "putdown": ("magenta", 40, 1.8, 12, "putdown fail"),
    "noatom": ("gray", 36, 1.6, 12, "no atom"),
    "crossed": ("red", 44, 2.0, 13, "crossed"),
    "collision": ("black", 44, 2.0, 13, "collision"),
    "eject": ("lime", 40, 1.8, 12, "eject"),
}


def _resolve_grid_dims(n_plots: int, max_cols: Optional[int]) -> tuple[int, int]:
    """Determine subplot grid while enforcing a maximum column count."""
    if n_plots <= 0:
        return 0, 0
    if n_plots == 4:
        ncols = 2
    elif n_plots > 4:
        ncols = 3
    else:
        ncols = n_plots
    ncols = max(1, ncols)
    if max_cols is not None and max_cols > 0:
        ncols = min(ncols, max_cols)
    nrows = int(np.ceil(n_plots / ncols))
    return nrows, ncols


def _init_failure_markers() -> Dict[str, List[Tuple[int, int]]]:
    return {key: [] for key in _FAILURE_MARKER_STYLE}


def _simulate_move_batches(atom_array, move_batches: List[List[Move]]) -> Tuple[
    List[np.ndarray],
    List[List[Tuple[Tuple[int, int], Tuple[int, int]]]],
    List[Dict[str, List[Tuple[int, int]]]],
]:
    """Replay *move_batches* against a scratch copy of *atom_array* through
    the real (collision/failure-aware) ``AtomArray.move_atoms`` — the
    simulation shared by every panel/frame renderer in this module.

    Returns ``(snapshots, movements_per_step, failures_per_step)``:
    ``snapshots[0]`` is the initial occupancy; ``snapshots[k]`` (``k>=1``)
    is the state after ``move_batches[k-1]``. ``movements_per_step`` /
    ``failures_per_step`` are 0-indexed per batch (length
    ``len(move_batches)``, one entry per transition — one shorter than
    ``snapshots``).
    """
    try:
        from atommovr.utils.AtomArray import AtomArray as _AA
    except Exception:
        _AA = None

    M = np.array(atom_array.matrix[:, :, 0], dtype=int)
    R, C = M.shape

    if _AA is not None:
        sim = _AA(atom_array.shape, n_species=atom_array.n_species)
        sim.matrix = atom_array.matrix.copy()
        sim.target = atom_array.target.copy()
        sim.error_model = atom_array.error_model
        sim.params = atom_array.params
    else:
        sim = None

    snapshots: List[np.ndarray] = [M.copy()]
    movements_per_step: List[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = []
    failures_per_step: List[Dict[str, List[Tuple[int, int]]]] = []

    for batch in move_batches:
        intended: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for mv in batch:
            if isinstance(mv, Move):
                intended.append(
                    (
                        (int(mv.from_row), int(mv.from_col)),
                        (int(mv.to_row), int(mv.to_col)),
                    )
                )

        prev_state = snapshots[-1].copy()
        failure_markers = _init_failure_markers()
        if sim is not None:
            (failed_moves, flags), _ = sim.move_atoms(batch)
            next_state = sim.matrix[:, :, 0].copy()
            flag_map = {int(idx): int(flags[i]) for i, idx in enumerate(failed_moves)}
            for mv_idx, mv in enumerate(batch):
                flag = flag_map.get(mv_idx, 0)
                movetype = getattr(mv, "movetype", None)
                if flag == 1:
                    failure_markers["pickup"].append(
                        (int(mv.from_row), int(mv.from_col))
                    )
                elif flag == 2:
                    failure_markers["putdown"].append(
                        (int(mv.from_row), int(mv.from_col))
                    )
                elif flag == 3:
                    failure_markers["noatom"].append(
                        (int(mv.from_row), int(mv.from_col))
                    )
                elif flag == 4:
                    failure_markers["crossed"].append(
                        (int(mv.from_row), int(mv.from_col))
                    )
                elif movetype == MoveType.ILLEGAL_MOVE:
                    failure_markers["collision"].append(
                        (int(mv.to_row), int(mv.to_col))
                    )
                elif movetype == MoveType.EJECT_MOVE:
                    failure_markers["eject"].append(
                        (int(mv.from_row), int(mv.from_col))
                    )
        else:
            # Fallback: naive update without collision handling
            next_state = prev_state.copy()
            for (fr, fc), (tr, tc) in intended:
                if 0 <= fr < R and 0 <= fc < C and next_state[fr, fc] == 1:
                    next_state[fr, fc] = 0
                    if 0 <= tr < R and 0 <= tc < C:
                        next_state[tr, tc] = 1

        # Determine realized moves using pre-batch legality to avoid drawing skip-overs
        realized: List[Tuple[Tuple[int, int], Tuple[int, int]]] = list(intended)

        snapshots.append(next_state)
        movements_per_step.append(realized)
        failures_per_step.append(failure_markers)

    return snapshots, movements_per_step, failures_per_step


def _draw_grid_panel(
    ax,
    state_mat: np.ndarray,
    R: int,
    C: int,
    *,
    title: str,
    moves: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
    failure_markers: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    fontsize: int = 8,
) -> None:
    """Draw one lattice-occupancy panel (atoms + move arrows + failure
    markers) into *ax* — the single-panel drawing logic shared by
    :func:`visualize_move_batches` (one panel per subplot in a grid figure)
    and :func:`render_move_batch_frames` (one panel per standalone GIF
    frame).
    """
    ax.set_xlim(-0.5, C - 0.5)
    ax.set_ylim(-0.5, R - 0.5)
    ax.set_xticks(range(C))
    ax.set_yticks(range(R))
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.grid(True)
    ax.tick_params(labelsize=6)

    rs, cs = np.where(state_mat == 1)
    ax.plot(list(cs), list(rs), "bo", markersize=6, zorder=2)
    ax.set_title(title, fontsize=fontsize)

    if moves:
        for (r0, c0), (r1, c1) in moves:
            # Clamp arrow head if destination is out-of-bounds (ejection)
            in_bounds = 0 <= r1 < R and 0 <= c1 < C
            rr = r1 if in_bounds else min(max(r1, 0), R - 1)
            cc = c1 if in_bounds else min(max(c1, 0), C - 1)
            ax.annotate(
                "",
                xy=(cc, rr),
                xytext=(c0, r0),
                arrowprops=dict(
                    arrowstyle="->",
                    color="r",
                    linewidth=1.2,
                    linestyle="dashed" if not in_bounds else "solid",
                ),
                zorder=10,
            )

    if failure_markers:
        for key, (color, size, lw, zorder, label) in _FAILURE_MARKER_STYLE.items():
            coords = failure_markers.get(key)
            if not coords:
                continue
            xs = [c for _, c in coords]
            ys = [r for r, _ in coords]
            ax.scatter(
                xs,
                ys,
                marker="x",
                color=color,
                s=size,
                linewidths=lw,
                zorder=zorder,
                label=label,
            )


def visualize_move_batches(
    atom_array,
    move_batches: List[List[Move]],
    save_path: Optional[str] = None,
    title_suffix: str = "",
    max_cols: int = 3,
):
    """
    Visualize batched moves (as returned by algorithms) over the lattice.

    - Subplot 0 shows the initial occupancy.
    - Subplot k>0 shows state after batch k with red arrows for that batch.

    Parameters
    - atom_array: `AtomArray` instance (used to get initial state and shape)
    - move_batches: list of lists of `Move`
    - save_path: optional output path. If None, saves to
      `figs/resorting/{title_suffix}.svg`.
    - title_suffix: appended to the filename/title for disambiguation.
    - max_cols: maximum subplot columns per figure.
    """
    snapshots, movements_per_step, failures_per_step = _simulate_move_batches(
        atom_array, move_batches
    )
    n_plots = len(snapshots)
    if n_plots == 0:
        return
    R, C = snapshots[0].shape

    nrows, ncols = _resolve_grid_dims(n_plots, max_cols)
    figsize = (3 * ncols, 3 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.array(axes, dtype=object).reshape(-1)

    for idx, ax in enumerate(axes):
        if idx >= n_plots:
            ax.axis("off")
            continue
        title = "Initial" if idx == 0 else f"Step {idx}"
        moves = movements_per_step[idx - 1] if idx > 0 else None
        failure_markers = failures_per_step[idx - 1] if idx > 0 else None
        _draw_grid_panel(
            ax,
            snapshots[idx],
            R,
            C,
            title=title,
            moves=moves,
            failure_markers=failure_markers,
        )

    for j in range(n_plots, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path is None:
        os.makedirs("figs/resorting", exist_ok=True)
        save_path = f"figs/resorting/{title_suffix}.svg"
    else:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_move_batch_frames(
    atom_array,
    move_batches: List[List[Move]],
    *,
    figsize: Tuple[float, float] = (4.5, 4.5),
    dpi: int = 100,
) -> List[np.ndarray]:
    """Render each round snapshot (initial state + one frame per move
    batch — same simulation and per-panel drawing as
    :func:`visualize_move_batches`) as an independent RGB ``uint8`` array,
    for animating into a GIF — e.g. via
    ``awg_controller.src.session_recorder.SessionRecorder.save_move_visualization``.

    Unlike :func:`visualize_move_batches` (one static multi-panel figure),
    this returns one array per snapshot at a fixed ``figsize``/``dpi`` (so
    every frame is the same pixel size, required for a valid GIF). Each
    frame uses a standalone ``Agg``-backed ``Figure``/``FigureCanvasAgg``
    (not ``matplotlib.pyplot``), so this never touches the process's
    interactive backend. Returns ``[]`` if *move_batches* is empty.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    snapshots, movements_per_step, failures_per_step = _simulate_move_batches(
        atom_array, move_batches
    )
    if not snapshots:
        return []
    R, C = snapshots[0].shape

    frames: List[np.ndarray] = []
    for idx, state_mat in enumerate(snapshots):
        fig = Figure(figsize=figsize, dpi=dpi, layout="constrained")
        canvas = FigureCanvasAgg(fig)
        ax = fig.subplots()
        title = "Initial" if idx == 0 else f"Step {idx}"
        moves = movements_per_step[idx - 1] if idx > 0 else None
        failure_markers = failures_per_step[idx - 1] if idx > 0 else None
        _draw_grid_panel(
            ax,
            state_mat,
            R,
            C,
            title=title,
            moves=moves,
            failure_markers=failure_markers,
            fontsize=11,
        )
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        frames.append(rgba[:, :, :3].copy())

    return frames


def visualize_batch_moves_on_image(
    atom_array,
    move_batches: List[List[Move]],
    save_path: str | None = None,
    title_suffix: str = "",
    max_cols: int = 3,
):
    """
    Visualize batched moves (as returned by algorithms) over the realistic atom image.

    - Subplot 0 shows the initial occupancy.
    - Subplot k>0 shows state after batch k with red arrows for that batch.

    Parameters
    - atom_array: `AtomArray` instance (used to get initial state and shape)
    - move_batches: list of lists of `Move`
    - save_path: optional output path. If None, saves to
      `figs/resorting/{title_suffix}_image.svg`.
    - title_suffix: appended to the filename/title for disambiguation.
    - max_cols: maximum subplot columns per figure.
    """
    try:
        from atommovr.utils.AtomArray import AtomArray as _AA
    except Exception:
        _AA = None

    base_state = np.array(atom_array.matrix[:, :, 0], dtype=int)
    R, C = base_state.shape

    if _AA is not None:
        sim = _AA(atom_array.shape, n_species=atom_array.n_species)
        sim.matrix = atom_array.matrix.copy()
        sim.target = atom_array.target.copy()
        sim.error_model = atom_array.error_model
        sim.params = atom_array.params
    else:
        sim = None

    snapshots: List[np.ndarray] = [base_state.copy()]
    realized_batches: List[List[tuple[tuple[int, int], tuple[int, int]]]] = []
    failures_per_step: List[dict[str, List[tuple[int, int]]]] = []

    def _init_failure_markers() -> dict[str, List[tuple[int, int]]]:
        return {
            "pickup": [],
            "putdown": [],
            "noatom": [],
            "crossed": [],
            "collision": [],
            "eject": [],
        }

    for batch in move_batches:
        intended: List[tuple[tuple[int, int], tuple[int, int]]] = []
        for mv in batch:
            if isinstance(mv, Move):
                intended.append(
                    (
                        (int(mv.from_row), int(mv.from_col)),
                        (int(mv.to_row), int(mv.to_col)),
                    )
                )

        prev_state = snapshots[-1].copy()
        failure_markers = _init_failure_markers()
        if sim is not None:
            (failed_moves, flags), _ = sim.move_atoms(batch)
            next_state = sim.matrix[:, :, 0].copy()
            flag_map = {int(idx): int(flags[i]) for i, idx in enumerate(failed_moves)}
            for mv_idx, mv in enumerate(batch):
                flag = flag_map.get(mv_idx, 0)
                movetype = getattr(mv, "movetype", None)
                if flag == 1:
                    failure_markers["pickup"].append(
                        (int(mv.from_row), int(mv.from_col))
                    )
                elif flag == 2:
                    failure_markers["putdown"].append(
                        (int(mv.from_row), int(mv.from_col))
                    )
                elif flag == 3:
                    failure_markers["noatom"].append(
                        (int(mv.from_row), int(mv.from_col))
                    )
                elif flag == 4:
                    failure_markers["crossed"].append(
                        (int(mv.from_row), int(mv.from_col))
                    )
                elif movetype == MoveType.ILLEGAL_MOVE:
                    failure_markers["collision"].append(
                        (int(mv.to_row), int(mv.to_col))
                    )
                elif movetype == MoveType.EJECT_MOVE:
                    failure_markers["eject"].append(
                        (int(mv.from_row), int(mv.from_col))
                    )
        else:
            next_state = prev_state.copy()
            for (fr, fc), (tr, tc) in intended:
                if 0 <= fr < R and 0 <= fc < C and next_state[fr, fc] == 1:
                    next_state[fr, fc] = 0
                    if 0 <= tr < R and 0 <= tc < C:
                        next_state[tr, tc] = 1

        realized: List[tuple[tuple[int, int], tuple[int, int]]] = []
        for (fr, fc), (tr, tc) in intended:
            realized.append(((fr, fc), (tr, tc)))

        snapshots.append(next_state)
        realized_batches.append(realized)
        failures_per_step.append(failure_markers)

    def _grid_pixel_centers(rows: int, cols: int, img_shape: Tuple[int, int]):
        h, w = img_shape
        if rows == 0 or cols == 0:
            return np.zeros((rows, cols, 2)), 0.0, 0.0
        start_row = h // max(rows, 1)
        start_col = w // max(cols, 1)
        row_step = max(1, h // (rows + 1))
        col_step = max(1, w // (cols + 1))
        centers = np.zeros((rows, cols, 2), dtype=float)
        for r in range(rows):
            for c in range(cols):
                centers[r, c, 0] = start_row + r * row_step  # + row_step // 10
                centers[r, c, 1] = start_col + c * col_step  # + col_step // 10
        return centers, float(row_step), float(col_step)

    image_shape = getattr(atom_array, "image_shape", None)
    if not (isinstance(image_shape, tuple) and len(image_shape) == 2):
        image_shape = (256, 256)

    centers, row_step_px, col_step_px = _grid_pixel_centers(R, C, image_shape)

    try:
        blob_detector = BlobDetection(shape=(R, C))
    except Exception:
        blob_detector = None

    rendered_images: List[np.ndarray] = []
    blob_centers: List[np.ndarray] = []
    for state in snapshots:
        img = generate_gaussian_image_from_binary_grid(
            state,
            sigma=1.2,
            brightness_factor=1.0,
            image_shape=image_shape,
            noise_level=0.02,
            stripe_intensity=0.003,
        )
        rendered_images.append(img)
        if blob_detector is not None:
            cents, _ = blob_detector.extract(img)
        else:
            cents = np.empty((0, 2), dtype=float)
        blob_centers.append(cents)

    n_plots = len(rendered_images)
    if n_plots == 0:
        return

    nrows, ncols = _resolve_grid_dims(n_plots, max_cols)
    figsize = (4 * ncols, 4 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.array(axes, dtype=object).reshape(-1)

    def _clamp_pixel(y: float, x: float) -> tuple[float, float]:
        return (
            float(np.clip(y, 0, image_shape[0] - 1)),
            float(np.clip(x, 0, image_shape[1] - 1)),
        )

    for idx in range(len(axes)):
        ax = axes[idx]
        if idx >= n_plots:
            ax.axis("off")
            continue

        img = rendered_images[idx]
        ax.imshow(img, cmap="Blues", origin="upper", interpolation="nearest")
        ax.axis("off")
        title = "Initial" if idx == 0 else f"Batch {idx}"
        ax.set_title(title)

        cents = blob_centers[idx]
        if cents.size:
            ax.scatter(
                cents[:, 1],
                cents[:, 0],
                s=40,
                facecolors="none",
                edgecolors="white",
                linewidths=1.0,
                alpha=0.9,
            )

        if idx == 0:
            continue

        moves = realized_batches[idx - 1]
        for (sr, sc), (tr, tc) in moves:
            if not (0 <= sr < R and 0 <= sc < C):
                continue
            y0, x0 = centers[sr, sc]
            in_bounds = 0 <= tr < R and 0 <= tc < C
            if in_bounds:
                y1, x1 = centers[tr, tc]
            else:
                delta_r = row_step_px if tr >= sr else -row_step_px
                delta_c = col_step_px if tc >= sc else -col_step_px
                if tr == sr:
                    delta_r = 0.0
                if tc == sc:
                    delta_c = 0.0
                y1, x1 = _clamp_pixel(y0 + delta_r, x0 + delta_c)

            ax.scatter(
                x0,
                y0,
                marker="x",
                color="tab:red",
                s=60,
                linewidths=2.0,
                zorder=15,
            )
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="->",
                    color="tab:red",
                    linewidth=1.5,
                    linestyle="solid" if in_bounds else "dashed",
                ),
                zorder=14,
            )
            if in_bounds:
                ax.scatter(
                    x1,
                    y1,
                    marker="o",
                    facecolors="none",
                    edgecolors="lime",
                    s=60,
                    linewidths=1.5,
                    zorder=16,
                )

        failure_markers = failures_per_step[idx - 1]

        def _plot_failures(coords: List[tuple[int, int]], color: str):
            if not coords:
                return
            ys = []
            xs = []
            for r, c in coords:
                if 0 <= r < R and 0 <= c < C:
                    y, x = centers[r, c]
                    ys.append(y)
                    xs.append(x)
            if xs:
                ax.scatter(
                    xs, ys, marker="x", color=color, s=80, linewidths=2.2, zorder=20
                )

        _plot_failures(failure_markers["pickup"], "gold")
        _plot_failures(failure_markers["putdown"], "magenta")
        _plot_failures(failure_markers["noatom"], "gray")
        _plot_failures(failure_markers["crossed"], "red")
        _plot_failures(failure_markers["collision"], "black")
        _plot_failures(failure_markers["eject"], "lime")

    plt.tight_layout()
    if save_path is None:
        os.makedirs("figs/resorting", exist_ok=True)
        save_path = f"figs/resorting/{title_suffix}_image.png"
    else:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
