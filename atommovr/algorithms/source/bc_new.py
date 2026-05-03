from __future__ import annotations
import copy
import numpy as np
import hashlib
from numpy.typing import NDArray
from typing import Tuple

from atommovr.utils.Move import Move
from atommovr.utils.AtomArray import AtomArray
from atommovr.utils.errormodels import ZeroNoise
from atommovr.utils.move_utils import move_atoms_noiseless, get_move_list_from_AOD_cmds
from atommovr.algorithms.source.bc_controller_helpers import (
    _int_sum,
    _as_2d_state,
    move_across_rows,
    get_all_moves_btwn_rows_cols,
    compress_move_rounds_conservative,
)
from atommovr.algorithms.source.ejection import ejection

BALANCE_BATCH_FRACTION: float = 0.8
PREBALANCE_BATCH_FRACTION: float = 0.8


def stable_digest(arr) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def bcv2(
    array: AtomArray,
    do_ejection: bool = False,
    batch_fractions: list[float] | None = None,
) -> tuple[NDArray[np.uint8], list[list[Move]], bool]:
    """
    Third iteration of the Balance and Compact algorithm
    (originally proposed for optical lattices).
    """
    # checks and quick returns
    if len(np.shape(array.matrix)) > 2 and np.shape(array.matrix)[2] == 2:
        raise ValueError(
            f"Atom array has shape {np.shape(array.matrix)}, which is not correct for single species. Did you meant to use a dual species algorithm?"
        )
    if batch_fractions is None:
        batch_fractions = [PREBALANCE_BATCH_FRACTION, BALANCE_BATCH_FRACTION]

    success_flag = False
    arr1 = copy.deepcopy(array)
    arr1.error_model = ZeroNoise()
    init_mat_copy = array.matrix.copy()
    start_row, start_col, end_row, end_col = get_target_locs(arr1)

    # 1. prebalance (making sure target rows/cols have enough atoms)
    master_move_list, success_flag = prebalance(
        arr1.matrix, arr1.target, batch_size_fraction=batch_fractions[0]
    )
    master_move_list = compress_move_rounds_conservative(
        init_mat_copy, master_move_list
    )

    _, _ = arr1.evaluate_moves(master_move_list)
    if success_flag:
        # 2. balance (distributing atoms between target rows according to needs)
        assignments = get_all_balance_assignments(start_row, end_row)
        for assignment in assignments:
            try:
                bal_moves = balance_rows(
                    arr1.matrix,
                    arr1.target,
                    assignment[0],
                    assignment[1],
                    balance_batch_fraction=batch_fractions[1],
                )
                if assignment[0] != assignment[1] and len(bal_moves) > 0:
                    _, _ = arr1.evaluate_moves(bal_moves)
                    master_move_list.extend(bal_moves)
            except ValueError:
                return arr1.matrix, master_move_list, False

        # 3. compact
        try:
            com_moves = compact(arr1)
        except ValueError:
            return arr1.matrix, master_move_list, False

        if len(com_moves) > 0:
            _, _ = arr1.evaluate_moves(com_moves)
            master_move_list.extend(com_moves)

    if do_ejection:
        eject_moves, final_config = ejection(
            arr1.matrix,
            arr1.target,
            [0, len(arr1.matrix) - 1, 0, len(arr1.matrix[0]) - 1],
        )
        _, _ = arr1.evaluate_moves(eject_moves)
        master_move_list.extend(eject_moves)
        # 3.1 Check if the configuration is the same as the target configuration
        if np.array_equal(arr1.matrix, arr1.target.reshape(np.shape(arr1.matrix))):
            success_flag = True
    else:
        # 3.2 Check if the configuration (inside range of target) the same as the target configuration
        effective_config = np.multiply(
            arr1.matrix, arr1.target.reshape(np.shape(arr1.matrix))
        )
        if np.array_equal(effective_config, arr1.target.reshape(np.shape(arr1.matrix))):
            success_flag = True

    return arr1.matrix, master_move_list, success_flag


def choose_best_atom_set_1d(
    init_config: np.ndarray,
    target_config: np.ndarray,
) -> np.ndarray:
    """
    Choose the contiguous subset of row atoms that should be assigned to the target.

    Why this exists
    ---------------
    The compact vote logic does not need the full 1D transport schedule for each
    row on every iteration. It only needs to know which current atoms belong to
    the best noncrossing assignment to the target support. This helper isolates
    that selection step without constructing AtomArray objects or simulating
    moves through the full move pipeline.

    The selected subset matches the current middle-fill policy:
    among contiguous subsets of the current row atoms with the required target
    cardinality, choose one minimizing the largest travel distance to the
    corresponding target columns, with the same central-window initialization
    used by the existing helper.

    Parameters
    ----------
    init_config
        Initial 1-row occupancy array with shape ``(1, n_cols, 1)`` or
        ``(1, n_cols)``.
    target_config
        Target 1-row occupancy array with shape ``(1, n_cols, 1)`` or
        ``(1, n_cols)``.

    Returns
    -------
    np.ndarray
        Sorted 1D integer array of chosen atom columns, with dtype ``np.intp``.

    Raises
    ------
    ValueError
        If the inputs do not describe one row, or if the row contains fewer
        atoms than target sites.
    """
    init_row: np.ndarray
    target_row: np.ndarray

    if init_config.ndim == 3:
        if init_config.shape[0] != 1 or init_config.shape[2] != 1:
            raise ValueError(
                "init_config must have shape (1, n_cols, 1) for 1D row helpers."
            )
        init_row = init_config[0, :, 0]
    elif init_config.ndim == 2:
        if init_config.shape[0] != 1:
            raise ValueError(
                "init_config must have shape (1, n_cols) for 1D row helpers."
            )
        init_row = init_config[0, :]
    else:
        raise ValueError(
            f"init_config must be 2D or 3D; got shape {init_config.shape}."
        )

    if target_config.ndim == 3:
        if target_config.shape[0] != 1 or target_config.shape[2] != 1:
            raise ValueError(
                "target_config must have shape (1, n_cols, 1) for 1D row helpers."
            )
        target_row = target_config[0, :, 0]
    elif target_config.ndim == 2:
        if target_config.shape[0] != 1:
            raise ValueError(
                "target_config must have shape (1, n_cols) for 1D row helpers."
            )
        target_row = target_config[0, :]
    else:
        raise ValueError(
            f"target_config must be 2D or 3D; got shape {target_config.shape}."
        )

    target_indices: np.ndarray = np.flatnonzero(target_row).astype(np.intp, copy=False)
    atom_indices: np.ndarray = np.flatnonzero(init_row).astype(np.intp, copy=False)

    n_targets: int = int(target_indices.size)
    n_atoms: int = int(atom_indices.size)
    n_cols: int = int(init_row.size)

    if n_targets == 0:
        return np.zeros(0, dtype=np.intp)
    if n_atoms < n_targets:
        raise ValueError(
            "choose_best_atom_set_1d requires at least as many row atoms as target sites."
        )
    if n_atoms == n_targets:
        return atom_indices.copy()

    avg_targ_pos: int = int(np.ceil(np.mean(target_indices)))
    radius: int = 0

    left_bound: int = 0
    right_bound: int = 0
    while True:
        left_bound = max(0, avg_targ_pos - radius)
        right_bound = min(n_cols, avg_targ_pos + radius + 1)
        center_region: np.ndarray = init_row[left_bound:right_bound]
        n_atoms_in_center_region: int = int(np.sum(center_region, dtype=np.int64))
        if n_atoms_in_center_region >= n_targets:
            break
        radius += 1

    first_local_idx: int = int(np.flatnonzero(center_region)[0])
    first_atom_loc: int = int(left_bound + first_local_idx)

    first_matches: np.ndarray = np.flatnonzero(atom_indices == first_atom_loc).astype(
        np.intp, copy=False
    )
    if first_matches.size == 0:
        raise RuntimeError(
            "choose_best_atom_set_1d failed to locate the first central atom."
        )
    first_list_ind: int = int(first_matches[0])

    current_r_atom_set: np.ndarray = atom_indices[
        first_list_ind : first_list_ind + n_targets
    ].copy()
    dist_r_current: int = find_largest_dist_to_move(target_indices, current_r_atom_set)
    right_count: int = 0

    while first_list_ind + right_count + n_targets < n_atoms:
        right_atom_set: np.ndarray = atom_indices[
            first_list_ind
            + right_count
            + 1 : first_list_ind
            + right_count
            + n_targets
            + 1
        ]
        dist_right: int = find_largest_dist_to_move(target_indices, right_atom_set)
        if dist_right > dist_r_current:
            break
        current_r_atom_set = right_atom_set.copy()
        dist_r_current = dist_right
        right_count += 1

    current_l_atom_set: np.ndarray = atom_indices[
        first_list_ind : first_list_ind + n_targets
    ].copy()
    dist_l_current: int = find_largest_dist_to_move(target_indices, current_l_atom_set)
    left_count: int = 0

    while first_list_ind - left_count - 1 >= 0:
        left_atom_set: np.ndarray = atom_indices[
            first_list_ind
            - left_count
            - 1 : first_list_ind
            - left_count
            + n_targets
            - 1
        ]
        dist_left: int = find_largest_dist_to_move(target_indices, left_atom_set)
        if dist_left > dist_l_current:
            break
        current_l_atom_set = left_atom_set.copy()
        dist_l_current = dist_left
        left_count += 1

    if dist_l_current < dist_r_current:
        return current_l_atom_set.astype(np.intp, copy=False)

    return current_r_atom_set.astype(np.intp, copy=False)


def first_round_edges_for_best_set(
    init_config: np.ndarray,
    target_config: np.ndarray,
    best_atom_set: np.ndarray,
) -> set[tuple[int, int]]:
    """
    Return the first-step horizontal edges for the chosen 1D atom subset.

    Why this exists
    ---------------
    The compact vote logic only needs to know, for each currently chosen row atom,
    whether its first legal unit step under the noncrossing target assignment is
    to the left, to the right, or to stay put. It does not need the full 1D move
    schedule. This helper constructs exactly that first-round edge set for the
    chosen contiguous subset.

    Parameters
    ----------
    init_config
        Initial 1-row occupancy array with shape ``(1, n_cols, 1)`` or
        ``(1, n_cols)``.
    target_config
        Target 1-row occupancy array with shape ``(1, n_cols, 1)`` or
        ``(1, n_cols)``.
    best_atom_set
        Sorted chosen atom columns, typically from ``choose_best_atom_set_1d(...)``.

    Returns
    -------
    set[tuple[int, int]]
        Set of first-round horizontal edges ``(from_col, to_col)`` for the chosen
        noncrossing assignment.

    Raises
    ------
    ValueError
        If the chosen atom count does not match the target-site count.
    """
    target_row: np.ndarray
    if target_config.ndim == 3:
        if target_config.shape[0] != 1 or target_config.shape[2] != 1:
            raise ValueError(
                "target_config must have shape (1, n_cols, 1) for 1D row helpers."
            )
        target_row = target_config[0, :, 0]
    elif target_config.ndim == 2:
        if target_config.shape[0] != 1:
            raise ValueError(
                "target_config must have shape (1, n_cols) for 1D row helpers."
            )
        target_row = target_config[0, :]
    else:
        raise ValueError(
            f"target_config must be 2D or 3D; got shape {target_config.shape}."
        )

    target_indices: np.ndarray = np.flatnonzero(target_row).astype(np.intp, copy=False)
    chosen_atoms: np.ndarray = np.asarray(best_atom_set, dtype=np.intp)

    if chosen_atoms.size != target_indices.size:
        raise ValueError(
            "best_atom_set size must match the number of target sites in the row."
        )

    edges: set[tuple[int, int]] = set()

    atom_col: int
    target_col: int
    for atom_col, target_col in zip(
        chosen_atoms.tolist(), target_indices.tolist(), strict=True
    ):
        if atom_col < target_col:
            edges.add((int(atom_col), int(atom_col + 1)))
        elif atom_col > target_col:
            edges.add((int(atom_col), int(atom_col - 1)))

    return edges


def special_case_algo_1d(
    init_config: np.ndarray, target_config: np.ndarray
) -> tuple[list, list]:
    arr_copy = AtomArray(np.shape(init_config)[:2])
    arr_copy.target = copy.deepcopy(target_config)
    arr_copy.matrix = copy.deepcopy(init_config)

    # first, find the column indices of the target sites
    # and those of the sites with atoms
    target_indices = np.where(arr_copy.target == 1)[1]
    atom_indices = np.where(arr_copy.matrix == 1)[1]

    if len(target_indices) != len(atom_indices):
        raise Exception(
            f"Number of atoms ({len(atom_indices)}) does not equal number of target sites ({len(target_indices)})."
        )

    # second, we can pair the atoms and make a list
    pairs = []
    for ind, target_index in enumerate(target_indices):
        atom_index = atom_indices[ind]
        pair = (target_index, atom_index)
        pairs.append(pair)
    # lastly, we can move atoms towards their target positions
    target_prepared = np.array_equal(arr_copy.target, arr_copy.matrix)
    move_set = []
    while not target_prepared:
        move_list = []
        for i, pair in enumerate(pairs):
            target_index, atom_index = pair
            if target_index != atom_index:
                new_atom_index = int(atom_index + np.sign(target_index - atom_index))
                move = Move(0, atom_index, 0, new_atom_index)
                move_list.append(move)
                pairs[i] = (target_index, new_atom_index)
        if move_list != []:
            _, _ = arr_copy.evaluate_moves([move_list])
            move_set.append(move_list)
        else:
            break
    return move_set, atom_indices.tolist()


# utility function that calculates the longest move distance between target sites and atom sites
def find_largest_dist_to_move(target_inds, atom_inds):
    if len(target_inds) > len(atom_inds):
        return np.inf
    max_dist = 0
    for ind, target_loc in enumerate(target_inds):
        atom_loc = atom_inds[ind]
        distance = np.abs(target_loc - atom_loc)
        if distance > max_dist:
            max_dist = distance
    return max_dist


def middle_fill_algo_1d(
    init_config: np.ndarray, target_config: np.ndarray
) -> Tuple[list, list]:
    """
    Choose a contiguous set of atoms for 1D row compaction and generate the
    corresponding move rounds.

    Why this exists
    ---------------
    In rows with more atoms than target sites, there are many possible subsets of
    atoms that could be used to realize the target support. This helper chooses a
    "central" contiguous subset and then refines that choice by comparing adjacent
    candidate subsets, favoring the one with the smaller worst-case travel
    distance.

    Parameters
    ----------
    init_config
        Initial 1-row occupancy configuration with shape ``(1, n_cols, 1)``.
    target_config
        Target 1-row occupancy configuration with shape ``(1, n_cols, 1)``.

    Returns
    -------
    tuple[list, list]
        ``(move_rounds, best_atom_set)`` where ``move_rounds`` is the sequence of
        horizontal move rounds and ``best_atom_set`` is the selected set of atom
        column indices.
    """
    arr_copy = AtomArray(np.shape(init_config)[:2])
    arr_copy.target = copy.deepcopy(target_config)
    arr_copy.matrix = copy.deepcopy(init_config)

    target_indices: np.ndarray = np.where(arr_copy.target == 1)[1]
    atom_indices: np.ndarray = np.where(arr_copy.matrix == 1)[1]
    n_targets: int = len(target_indices)
    n_atoms: int = len(atom_indices)
    n_cols: int = int(arr_copy.matrix.shape[1])

    if n_targets == n_atoms:
        return special_case_algo_1d(init_config, target_config)
    elif n_targets > n_atoms or n_targets == 0:
        return [], []

    # Find a central window that contains at least `n_targets` atoms.
    avg_targ_pos: int = int(np.ceil(np.mean(target_indices)))
    count: int = 0
    sufficient_atoms: bool = False
    left_bound: int = 0
    right_bound: int = 0

    while not sufficient_atoms:
        left_bound = max(0, avg_targ_pos - count)
        right_bound = min(n_cols, avg_targ_pos + count + 1)
        center_region: np.ndarray = arr_copy.matrix[0, left_bound:right_bound]

        n_atoms_in_center_region: int = _int_sum(center_region)
        sufficient_atoms = n_targets <= n_atoms_in_center_region
        if not sufficient_atoms:
            count += 1

    first_local_idx: int = int(np.where(center_region == 1)[0][0])
    first_atom_loc: int = left_bound + first_local_idx

    # Convert the first atom location to the corresponding index in atom_indices.
    first_matches: np.ndarray = np.where(atom_indices == first_atom_loc)[0]
    if first_matches.size == 0:
        raise RuntimeError(
            "middle_fill_algo_1d failed to locate the first central atom inside "
            "the sorted atom index list."
        )
    first_list_ind: int = int(first_matches[0])

    # Explore candidate contiguous atom sets to the right.
    current_r_atom_set: np.ndarray = atom_indices[
        first_list_ind : first_list_ind + n_targets
    ]
    dist_r_current: int = find_largest_dist_to_move(target_indices, current_r_atom_set)
    right_count: int = 0

    while first_list_ind + right_count + n_targets < n_atoms:
        right_atom_set: np.ndarray = atom_indices[
            first_list_ind
            + right_count
            + 1 : first_list_ind
            + right_count
            + n_targets
            + 1
        ]
        dist_right: int = find_largest_dist_to_move(target_indices, right_atom_set)
        if dist_right > dist_r_current:
            break
        current_r_atom_set = right_atom_set
        dist_r_current = dist_right
        right_count += 1

    # Explore candidate contiguous atom sets to the left.
    current_l_atom_set: np.ndarray = atom_indices[
        first_list_ind : first_list_ind + n_targets
    ]
    dist_l_current: int = find_largest_dist_to_move(target_indices, current_l_atom_set)
    left_count: int = 0

    while first_list_ind - left_count - 1 >= 0:
        left_atom_set: np.ndarray = atom_indices[
            first_list_ind
            - left_count
            - 1 : first_list_ind
            - left_count
            + n_targets
            - 1
        ]
        dist_left: int = find_largest_dist_to_move(target_indices, left_atom_set)
        if dist_left > dist_l_current:
            break
        current_l_atom_set = left_atom_set
        dist_l_current = dist_left
        left_count += 1

    if dist_l_current < dist_r_current:
        best_atom_set: np.ndarray = current_l_atom_set
    else:
        best_atom_set = current_r_atom_set

    pairs: list[tuple[int, int]] = []
    for ind, target_index in enumerate(target_indices):
        atom_index: int = int(best_atom_set[ind])
        pair: tuple[int, int] = (int(target_index), atom_index)
        pairs.append(pair)

    target_prepared: bool = np.array_equal(arr_copy.target, arr_copy.matrix)
    move_set: list[list[Move]] = []
    while not target_prepared:
        move_list: list[Move] = []
        for i, pair in enumerate(pairs):
            target_index, atom_index = pair
            if target_index != atom_index:
                new_atom_index: int = int(
                    atom_index + np.sign(target_index - atom_index)
                )
                move: Move = Move(0, atom_index, 0, new_atom_index)
                move_list.append(move)
                pairs[i] = (target_index, new_atom_index)

        if len(move_list) == 0:
            break

        arr_copy.move_atoms(move_list)
        move_set.append(move_list)
        target_prepared = np.array_equal(arr_copy.target, arr_copy.matrix)

    return move_set, best_atom_set.tolist()


# Balance and Compact


def _target_col_bounds_for_rows(
    target_config: np.ndarray,
    start_row: int,
    end_row: int,
) -> tuple[int, int]:
    """
    Return the inclusive target-column bounds relevant to a row interval.

    Why this exists
    ---------------
    The controller-level horizontal cut-capacity helper needs a preferred target
    column window. For row-balancing and prebalance, the natural window is the
    target support within the active row interval, not the full array width.

    Parameters
    ----------
    target_config
        Target occupancy array with shape ``(rows, cols, 1)`` or ``(rows, cols)``.
    start_row
        Inclusive first row of the interval.
    end_row
        Inclusive last row of the interval.

    Returns
    -------
    tuple[int, int]
        Inclusive ``(start_col, end_col)`` bounds. If the interval contains no
        target support, returns the full row width.
    """
    if target_config.ndim == 3:
        target_2d: np.ndarray = target_config[:, :, 0]
    else:
        target_2d = target_config

    interval_target: np.ndarray = target_2d[start_row : end_row + 1, :]
    rr: np.ndarray
    cc: np.ndarray
    rr, cc = np.where(interval_target == 1)

    n_cols: int = int(target_2d.shape[1])
    if cc.size == 0:
        return 0, n_cols - 1
    return int(cc.min()), int(cc.max())


def balance_rows(
    init_config: np.ndarray,
    target_config: np.ndarray,
    i: int,
    j: int,
    balance_batch_fraction: float = BALANCE_BATCH_FRACTION,
) -> list[list[Move]]:
    """
    Balance atom supply between the two child halves of ``[i, j]``.

    Why this exists
    ---------------
    Recursive BC balancing works by ensuring that each child half of an interval
    contains enough atoms to realize the target support inside that child. This
    function computes the exact net atom transfer required across the cut between
    the two halves and delegates the actual routing to ``move_across_rows(...)``.

    Contract
    --------
    - If the interval is already feasible on both halves, return no moves.
    - If total atom number inside ``[i, j]`` is insufficient, raise ``ValueError``.
    - Otherwise move exactly the required net number of atoms across the cut, or
      raise ``RuntimeError`` if the current helper/controller decomposition cannot
      realize that transfer.

    Parameters
    ----------
    init_config
        Current single-species occupancy array.
    target_config
        Target single-species occupancy array.
    i
        Inclusive start row of the active interval.
    j
        Inclusive end row of the active interval.

    Returns
    -------
    list[list[Move]]
        Parallel move rounds that realize the required balancing transfer.

    Raises
    ------
    ValueError
        If the total atom count in ``[i, j]`` is insufficient to satisfy the
        target support in that interval.
    RuntimeError
        If the exact required transfer cannot be completed by the current
        controller/helper decomposition.
    """
    if i == j:
        return []

    n_rows_involved: int = j - i + 1
    m: int = i + (n_rows_involved // 2)

    n_req_top: int = _int_sum(target_config[i:m, :])
    n_atoms_top: int = _int_sum(init_config[i:m, :])
    n_req_bot: int = _int_sum(target_config[m : j + 1, :])
    n_atoms_bot: int = _int_sum(init_config[m : j + 1, :])

    diff_top: int = n_atoms_top - n_req_top
    diff_bot: int = n_atoms_bot - n_req_bot

    if (diff_top + diff_bot) < 0:
        raise ValueError(
            f"Insufficient number of atoms: deficit in rows {i}-{m-1} is {diff_top} "
            f"and deficit in rows {m}-{j} is {diff_bot}."
        )

    # Both halves already feasible.
    if diff_top >= 0 and diff_bot >= 0:
        return []

    current_state: np.ndarray = init_config.copy()
    target_start_col: int
    target_end_col: int
    target_start_col, target_end_col = _target_col_bounds_for_rows(target_config, i, j)

    n_to_move: int
    transferred: int
    move_rounds: list[list[Move]]
    status: dict[str, int | float | str | bool | None]

    if diff_top < 0 and diff_bot > 0:
        # Move atoms bottom -> top across cut (m) -> (m-1).
        n_to_move = min(-diff_top, diff_bot)
        current_state, move_rounds, transferred, status = move_across_rows(
            state=current_state,
            boundary_src_row=m,
            boundary_dst_row=m - 1,
            source_search_limit_row=j,
            destination_search_limit_row=i,
            R=n_to_move,
            C=balance_batch_fraction,
            target_start_col=target_start_col,
            target_end_col=target_end_col,
        )
    elif diff_bot < 0 and diff_top > 0:
        # Move atoms top -> bottom across cut (m-1) -> (m).
        n_to_move = min(-diff_bot, diff_top)
        current_state, move_rounds, transferred, status = move_across_rows(
            state=current_state,
            boundary_src_row=m - 1,
            boundary_dst_row=m,
            source_search_limit_row=i,
            destination_search_limit_row=j,
            R=n_to_move,
            C=balance_batch_fraction,
            target_start_col=target_start_col,
            target_end_col=target_end_col,
        )
    else:
        # This branch should only happen when both halves are already feasible,
        # which we already returned above. Keep it explicit.
        return []

    if transferred != n_to_move:
        top_atoms_after: int = _int_sum(current_state[i:m, :])
        top_req_after: int = _int_sum(target_config[i:m, :])
        bot_atoms_after: int = _int_sum(current_state[m : j + 1, :])
        bot_req_after: int = _int_sum(target_config[m : j + 1, :])

        raise RuntimeError(
            "move_across_rows failed to complete transfer: "
            f"requested {n_to_move}, transferred {transferred}. "
            f"Top after = {top_atoms_after} (need {top_req_after}); "
            f"bottom after = {bot_atoms_after} (need {bot_req_after}). "
            f"Controller status: {status}."
        )

    # Final exact-feasibility check.
    top_atoms: int = _int_sum(current_state[i:m, :])
    top_req: int = _int_sum(target_config[i:m, :])
    bot_atoms: int = _int_sum(current_state[m : j + 1, :])
    bot_req: int = _int_sum(target_config[m : j + 1, :])

    if top_atoms < top_req or bot_atoms < bot_req:
        raise RuntimeError(
            "move_across_rows completed the nominal transfer count but did not leave "
            f"child halves feasible. Top: {top_atoms}/{top_req}, "
            f"bottom: {bot_atoms}/{bot_req}."
        )

    return move_rounds


def prebalance(
    init_config: np.ndarray,
    target_config: np.ndarray,
    batch_size_fraction: float = PREBALANCE_BATCH_FRACTION,
) -> tuple[list[list[Move]], bool]:
    """
    Ensure the target-row band contains enough atoms before recursive row balancing.

    Why this exists
    ---------------
    Recursive ``balance_rows(...)`` assumes the target-row band already contains at
    least as many atoms in total as the target support inside that band. This
    function fills that band by pulling atoms across its top and bottom boundaries
    using the same controller machinery as the balancing step.

    The micro-objective here is simpler than full recursive balancing:
    populate the target-row band with enough total atoms, then stop. The function
    should not spend extra rounds after the band first becomes sufficient, though
    opportunistic oversupply within a successful final round is allowed.

    Parameters
    ----------
    init_config
        Current single-species occupancy array.
    target_config
        Target single-species occupancy array.
    batch_size_fraction
        Minimum number of atoms to transfer in parallel (i.e. fraction of the row)

    Returns
    -------
    tuple[list[list[Move]], bool]
        ``(move_rounds, ok)`` where ``ok`` indicates whether the target-row band
        ended with sufficient total atom count.

    Notes
    -----
    - If the target is empty, returns ``([], False)``.
    - If the global atom count is insufficient, returns ``([], False)``.
    - If the target-row band is already sufficiently populated, returns
      ``([], True)``.
    - If the controller/helper decomposition makes no progress while the band
      remains deficient, raises ``RuntimeError``.
    - If the loop re-enters a previously seen full state before the band is
      sufficiently populated, raises ``RuntimeError``.
    """
    target_view: np.ndarray = (
        target_config[:, :, 0] if target_config.ndim == 3 else target_config
    )
    rr: np.ndarray
    cc: np.ndarray
    rr, cc = np.where(target_view == 1)

    if rr.size == 0:
        return [], False

    start_row: int = int(rr.min())
    end_row: int = int(rr.max())
    target_start_col: int = int(cc.min())
    target_end_col: int = int(cc.max())

    n_targets: int = _int_sum(target_config[start_row : end_row + 1, :])
    n_atoms_global: int = _int_sum(init_config)

    if n_atoms_global < n_targets:
        return [], False

    current_state: np.ndarray = init_config.copy()
    all_rounds: list[list[Move]] = []

    n_atoms_row_region: int = _int_sum(current_state[start_row : end_row + 1, :])
    if n_atoms_row_region >= n_targets:
        return [], True

    n_rows: int = int(current_state.shape[0])

    # Repeated-state guard for multi-step cycles.
    seen_signatures: set[bytes] = set()

    while _int_sum(current_state[start_row : end_row + 1, :]) < n_targets:
        signature: bytes = current_state.tobytes()
        if signature in seen_signatures:
            raise RuntimeError(
                "prebalance re-entered a previously seen state while the target-row "
                "band remains deficient. This indicates a controller cycle under "
                "the current helper decomposition."
            )
        seen_signatures.add(signature)

        state_before: np.ndarray = current_state.copy()
        band_before: int = _int_sum(current_state[start_row : end_row + 1, :])
        deficit: int = n_targets - band_before

        made_progress: bool = False

        # --------------------------------------------------------------
        # Try sourcing from above first.
        # --------------------------------------------------------------
        if start_row > 0:
            atoms_above: int = _int_sum(current_state[:start_row, :])
            if atoms_above > 0:
                request_above: int = min(deficit, atoms_above)

                next_state: np.ndarray
                move_rounds: list[list[Move]]
                transferred: int
                status: dict[str, int | float | str | bool | None]
                next_state, move_rounds, transferred, status = move_across_rows(
                    state=current_state,
                    boundary_src_row=start_row - 1,
                    boundary_dst_row=start_row,
                    source_search_limit_row=0,
                    destination_search_limit_row=end_row,
                    R=request_above,
                    C=batch_size_fraction,
                    target_start_col=target_start_col,
                    target_end_col=target_end_col,
                )

                if transferred > 0:
                    current_state = next_state
                    all_rounds.extend(move_rounds)
                    made_progress = True

        # --------------------------------------------------------------
        # If still deficient, try sourcing from below.
        # --------------------------------------------------------------
        if (
            _int_sum(current_state[start_row : end_row + 1, :]) < n_targets
            and end_row < n_rows - 1
        ):
            atoms_below: int = _int_sum(current_state[end_row + 1 :, :])
            if atoms_below > 0:
                band_now: int = _int_sum(current_state[start_row : end_row + 1, :])
                deficit_now: int = n_targets - band_now
                request_below: int = min(deficit_now, atoms_below)

                next_state, move_rounds, transferred, status = move_across_rows(
                    state=current_state,
                    boundary_src_row=end_row + 1,
                    boundary_dst_row=end_row,
                    source_search_limit_row=n_rows - 1,
                    destination_search_limit_row=start_row,
                    R=request_below,
                    C=batch_size_fraction,
                    target_start_col=target_start_col,
                    target_end_col=target_end_col,
                )

                if transferred > 0:
                    current_state = next_state
                    all_rounds.extend(move_rounds)
                    made_progress = True

        band_after: int = _int_sum(current_state[start_row : end_row + 1, :])
        state_changed: bool = not np.array_equal(current_state, state_before)

        if band_after >= n_targets:
            return all_rounds, True

        if not made_progress and not state_changed and band_after == band_before:
            raise RuntimeError(
                "prebalance made no progress while target-row band remains deficient. "
                "This indicates a jammed or unreachable routing configuration under "
                "the current controller/helper decomposition."
            )

    return all_rounds, True


def get_all_moves_btwn_rows_from_rows(
    from_row: np.ndarray,
    to_row: np.ndarray,
    from_row_ind: int,
    to_row_ind: int,
) -> tuple[list[Move], int]:
    """
    Build a greedy, parallelizable move set that transfers atoms between two rows.

    Why this exists
    ---------------
    BCv2 calls row-to-row transfer many times. The dominant cost at high call counts
    is *Python overhead* (slicing, repeated attribute lookups, repeated bounds checks),
    not the simple local matching itself.

    This helper isolates the core logic so callers that already have the row slices
    can avoid reslicing `init_config` and can reuse temporary arrays in future
    optimizations.

    Contract
    --------
    - `from_row` and `to_row` are 1D integer arrays with occupancy in {0,1}.
    - The greedy policy matches the existing behavior:
        For each atom at column c (processed in increasing c),
        choose destination in priority order: c-1, c, c+1, subject to destination vacancy.
    - Each destination column is used at most once.

    Parameters
    ----------
    from_row, to_row
        1D occupancy arrays for source and destination rows (values 0/1).
    from_row_ind, to_row_ind
        Absolute row indices used to construct Move objects.

    Returns
    -------
    moves, n_moves
        Move list and its length.
    """
    if from_row_ind < 0 or to_row_ind < 0:
        raise IndexError

    # Fast exits
    if from_row.size == 0:
        return [], 0

    # `flatnonzero(from_row)` is equivalent to indices where from_row != 0
    src_cols = np.flatnonzero(from_row)
    if src_cols.size == 0:
        return [], 0

    # Free destination slots as a boolean array we can mutate.
    free = to_row == 0
    if not free.any():
        return [], 0

    n_cols = int(free.size)
    moves: list[Move] = []
    append = moves.append  # localize for speed

    # Main greedy loop: minimal Python work per source.
    for c in src_cols:
        ci = int(c)

        left = ci - 1
        if left >= 0 and free[left]:
            append(Move(from_row_ind, ci, to_row_ind, left))
            free[left] = False
            continue

        if free[ci]:
            append(Move(from_row_ind, ci, to_row_ind, ci))
            free[ci] = False
            continue

        right = ci + 1
        if right < n_cols and free[right]:
            append(Move(from_row_ind, ci, to_row_ind, right))
            free[right] = False
            continue

    return moves, len(moves)


def get_all_moves_btwn_rows(
    init_config: np.ndarray,
    from_row_ind: int,
    to_row_ind: int,
) -> tuple[list[Move], int]:
    """
    Backwards-compatible wrapper returning `Move` objects.

    Why this exists
    ---------------
    The rest of BCv2 expects a `list[Move]`. Internally we compute the same matching
    using `get_all_moves_btwn_rows_cols` and only construct `Move` objects once we
    know we have at least one move.
    """
    from_cols, to_cols, n_moves = get_all_moves_btwn_rows_cols(
        init_config, from_row_ind, to_row_ind
    )
    if n_moves == 0:
        return [], 0

    moves = [
        Move(from_row_ind, int(fc), to_row_ind, int(tc))
        for fc, tc in zip(from_cols, to_cols, strict=True)
    ]
    return moves, n_moves


def get_all_moves_btwn_rows_faster(
    init_config: np.ndarray,
    from_row_ind: int,
    to_row_ind: int,
) -> tuple[list[Move], int]:
    """
    Wrapper around `get_all_moves_btwn_rows_from_rows` that slices rows from `init_config`.

    Why this exists
    ---------------
    Maintains the existing BCv2 API, but routes through the optimized core routine.

    Parameters
    ----------
    init_config
        2D occupancy array (rows, cols) with values in {0,1}.
    from_row_ind, to_row_ind
        Row indices.

    Returns
    -------
    moves, n_moves
        Move list and its length.
    """
    if init_config.ndim != 3:
        raise ValueError(
            f"init_config must be 3D (rows, cols); got ndim={init_config.ndim}."
        )

    from_row = init_config[from_row_ind, :, 0]
    to_row = init_config[to_row_ind, :, 0]
    return get_all_moves_btwn_rows_from_rows(from_row, to_row, from_row_ind, to_row_ind)


def get_all_moves_btwn_cols(init_config, from_col_ind, to_col_ind):
    from_col = init_config[:, from_col_ind]
    to_col = init_config[:, to_col_ind, :]

    available_source = np.flatnonzero(from_col == 1)

    free = (to_col[:, 0] == 0).copy()  # bool
    moves = []

    for atom_row in available_source:
        dest = None
        if atom_row - 1 >= 0 and free[atom_row - 1]:
            dest = atom_row - 1
        elif free[atom_row]:
            dest = atom_row
        elif atom_row + 1 < free.size and free[atom_row + 1]:
            dest = atom_row + 1

        if dest is not None:
            moves.append(Move(int(atom_row), from_col_ind, int(dest), to_col_ind))
            free[dest] = False

    return moves, len(moves)


def get_all_balance_assignments(start, end):
    assignments = []
    i = start
    j = end
    new_assignments = [(i, j)]
    n_a = len(new_assignments)
    while n_a > 0:
        assignment_list = []
        for assignment in new_assignments:
            i = assignment[0]
            j = assignment[1]
            next_layer = get_next_balance_assignment(i, j)
            assignment_list.extend(next_layer)
        assignments.extend(new_assignments)
        if len(assignment_list) > 0:
            new_assignments = assignment_list
        else:
            break
    return assignments


def get_next_balance_assignment(i, j):
    n_rows_involved = j - i + 1
    m = i + (n_rows_involved // 2)
    next_list = []
    if i != j and i < j:
        next_list.append((i, m - 1))
        next_list.append((m, j))
    return next_list


def get_target_locs(array):
    """
    Return the bounding box (start_row, start_col, end_row, end_col) of the target region.

    Notes
    -----
    Vectorized implementation to avoid O(H*W) Python loops.
    """
    targ = array.target
    if targ.ndim == 3:
        targ2 = targ[:, :, 0]
    else:
        targ2 = targ

    rr, cc = np.where(targ2 == 1)
    if rr.size == 0:
        # No target sites: treat as empty region
        return 0, 0, -1, -1

    return int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())


def _compact_state_signature(matrix: np.ndarray) -> bytes:
    """
    Return a hashable signature for compact-cycle detection.

    Parameters
    ----------
    matrix
        Current single-species occupancy state.

    Returns
    -------
    bytes
        Byte representation of the current state.
    """
    return matrix.tobytes()


def _target_overlap_count(matrix: np.ndarray, target: np.ndarray) -> int:
    """
    Return the number of currently filled target sites.

    Parameters
    ----------
    matrix
        Current occupancy matrix.
    target
        Target occupancy matrix.

    Returns
    -------
    int
        Number of occupied target sites.
    """
    return int(np.sum(matrix * target, dtype=np.int64))


def compact(array) -> list[list[Move]]:
    """
    Compact atoms horizontally into the target rectangle using legal shared AOD frames.

    Why this exists
    ---------------
    Under the updated collision model, the fully symmetric inward crunch is not a
    physically admissible shared AOD command pattern: near the condensation column,
    inward tones from both sides can create colliding tweezers / pile-ups.

    This implementation compares four admissible candidate templates each round:
    - left-center-deleted symmetric template
    - right-center-deleted symmetric template
    - pure right-moving template
    - pure left-moving template

    The chosen template must actually change the state. Among state-changing
    candidates, we prefer those that maximize target overlap increase, with vote
    sum used as a tie-breaker. A repeated-state guard is included to catch cycles
    loudly rather than silently looping forever.
    """
    arr1 = copy.deepcopy(array)

    start_row, start_col, end_row, end_col = get_target_locs(arr1)
    n_rows: int = len(arr1.target)
    n_cols: int = len(arr1.target[0])

    if end_row < start_row or end_col < start_col:
        return []

    target_2d: np.ndarray = _as_2d_state(arr1.target)
    matrix_2d: np.ndarray = _as_2d_state(arr1.matrix)

    required_per_row: np.ndarray = np.sum(
        target_2d[start_row : end_row + 1, :],
        axis=1,
        dtype=np.int64,
    )
    atoms_per_row: np.ndarray = np.sum(
        matrix_2d[start_row : end_row + 1, :],
        axis=1,
        dtype=np.int64,
    )

    insufficient_mask: np.ndarray = atoms_per_row < required_per_row
    if np.any(insufficient_mask):
        bad_rows: np.ndarray = np.where(insufficient_mask)[0] + start_row
        details: str = ", ".join(
            f"row {int(row)}: have {int(atoms_per_row[row - start_row])}, "
            f"need {int(required_per_row[row - start_row])}"
            for row in bad_rows
        )
        raise ValueError(
            "compact() requires each target row to already have enough atoms for "
            f"horizontal compaction, but found insufficient rows: {details}."
        )

    global_move_set: list[list[Move]] = []
    seen_signatures: set[bytes] = set()

    # Per-row cached metadata used by the vote logic.
    best_atom_cols_by_row: dict[int, np.ndarray] = {}
    best_atom_mask_by_row: dict[int, np.ndarray] = {}
    left_edge_mask_by_row: dict[int, np.ndarray] = {}
    right_edge_mask_by_row: dict[int, np.ndarray] = {}

    # Initially, every target row needs metadata.
    dirty_rows: set[int] = set(range(start_row, end_row + 1))

    def _refresh_row_metadata(row: int) -> None:
        """
        Recompute compact vote metadata for one row.

        Notes
        -----
        This is the hot-path replacement for the old middle_fill_algo_1d(...)
        call inside compact. It computes only the chosen atom subset and the
        first-step horizontal edges needed by the vote tally.
        """
        row_init: np.ndarray = arr1.matrix[row, :, :].reshape(1, n_cols, 1)
        row_target: np.ndarray = arr1.target[row, :, :].reshape(1, n_cols, 1)

        best_atom_cols: np.ndarray = choose_best_atom_set_1d(
            row_init,
            row_target,
        )
        first_edges: set[tuple[int, int]] = first_round_edges_for_best_set(
            row_init,
            row_target,
            best_atom_set=best_atom_cols,
        )

        best_atom_mask: np.ndarray = np.zeros(n_cols, dtype=np.bool_)
        if best_atom_cols.size > 0:
            best_atom_mask[best_atom_cols] = True

        left_edge_mask: np.ndarray = np.zeros(n_cols, dtype=np.bool_)
        right_edge_mask: np.ndarray = np.zeros(n_cols, dtype=np.bool_)
        src_col: int
        dst_col: int
        for src_col, dst_col in first_edges:
            if dst_col == src_col - 1:
                left_edge_mask[src_col] = True
            elif dst_col == src_col + 1:
                right_edge_mask[src_col] = True

        best_atom_cols_by_row[row] = best_atom_cols
        best_atom_mask_by_row[row] = best_atom_mask
        left_edge_mask_by_row[row] = left_edge_mask
        right_edge_mask_by_row[row] = right_edge_mask

    while True:
        signature: bytes = _compact_state_signature(arr1.matrix)
        if signature in seen_signatures:
            raise RuntimeError(
                "compact() re-entered a previously seen state."
                "This indicates a horizontal compaction cycle."
            )
        seen_signatures.add(signature)

        for row in sorted(dirty_rows):
            _refresh_row_metadata(row)
        dirty_rows.clear()

        col_counts: np.ndarray = np.sum(
            arr1.matrix[start_row : end_row + 1, start_col : end_col + 1, 0],
            axis=0,
            dtype=np.int64,
        )
        min_col_ind: int = int(start_col + int(np.argmin(col_counts)))

        n_target_rows: int = end_row - start_row + 1
        r_vote_tally: np.ndarray = np.zeros(n_target_rows, dtype=np.float64)
        l_vote_tally: np.ndarray = np.zeros(n_target_rows, dtype=np.float64)

        for i, row in enumerate(range(start_row, end_row + 1)):
            atom_in_row: int = int(arr1.matrix[row, min_col_ind, 0])
            if atom_in_row != 0:
                r_vote_tally[i] = -np.e
                l_vote_tally[i] = -np.e

        for col in range(n_cols):
            move_dir: int = int(np.sign(min_col_ind - col))

            if move_dir == -1:
                for i, row in enumerate(range(start_row, end_row + 1)):
                    if r_vote_tally[i] == -np.e:
                        continue
                    if int(arr1.matrix[row, col, 0]) != 1:
                        continue
                    if not bool(best_atom_mask_by_row[row][col]):
                        continue

                    vote: int = int(bool(left_edge_mask_by_row[row][col]))
                    r_vote_tally[i] += -1 + 2 * vote

            elif move_dir == 1:
                for i, row in enumerate(range(start_row, end_row + 1)):
                    if l_vote_tally[i] == -np.e:
                        continue
                    if int(arr1.matrix[row, col, 0]) != 1:
                        continue
                    if not bool(best_atom_mask_by_row[row][col]):
                        continue

                    vote = int(bool(right_edge_mask_by_row[row][col]))
                    l_vote_tally[i] += -1 + 2 * vote

        total_vote_tally: np.ndarray = r_vote_tally + l_vote_tally

        r_comh_AOD_cmds: np.ndarray = np.zeros(n_cols, dtype=np.uint8)
        l_comh_AOD_cmds: np.ndarray = np.zeros(n_cols, dtype=np.uint8)
        r_comv_AOD_cmds: np.ndarray = np.zeros(n_rows, dtype=np.uint8)
        l_comv_AOD_cmds: np.ndarray = np.zeros(n_rows, dtype=np.uint8)
        r_vote_sum: float = 0.0
        l_vote_sum: float = 0.0

        for row_ind in range(n_target_rows):
            n_r_votes: float = float(r_vote_tally[row_ind])
            n_l_votes: float = float(l_vote_tally[row_ind])
            abs_row: int = row_ind + start_row
            if n_r_votes > 0:
                r_comv_AOD_cmds[abs_row] = np.uint8(1)
                r_vote_sum += n_r_votes
            elif n_l_votes > 0:
                l_comv_AOD_cmds[abs_row] = np.uint8(1)
                l_vote_sum += n_l_votes

        if min_col_ind + 1 < n_cols:
            r_comh_AOD_cmds[min_col_ind + 1 :] = np.uint8(3)
        if min_col_ind > 0:
            l_comh_AOD_cmds[:min_col_ind] = np.uint8(2)

        left_del_comh_AOD_cmds: np.ndarray = np.zeros(n_cols, dtype=np.uint8)
        right_del_comh_AOD_cmds: np.ndarray = np.zeros(n_cols, dtype=np.uint8)
        left_del_comv_AOD_cmds: np.ndarray = np.zeros(n_rows, dtype=np.uint8)
        right_del_comv_AOD_cmds: np.ndarray = np.zeros(n_rows, dtype=np.uint8)

        if min_col_ind > 0:
            left_del_comh_AOD_cmds[:min_col_ind] = np.uint8(2)
            right_del_comh_AOD_cmds[:min_col_ind] = np.uint8(2)
        if min_col_ind + 1 < n_cols:
            left_del_comh_AOD_cmds[min_col_ind + 1 :] = np.uint8(3)
            right_del_comh_AOD_cmds[min_col_ind + 1 :] = np.uint8(3)

        if min_col_ind - 1 >= 0:
            left_del_comh_AOD_cmds[min_col_ind - 1] = np.uint8(0)
        if min_col_ind + 1 < n_cols:
            right_del_comh_AOD_cmds[min_col_ind + 1] = np.uint8(0)

        left_deleted_collision_mask: np.ndarray = np.zeros(n_rows, dtype=np.bool_)
        right_deleted_collision_mask: np.ndarray = np.zeros(n_rows, dtype=np.bool_)

        if min_col_ind - 2 >= 0:
            left_deleted_collision_mask |= arr1.matrix[:, min_col_ind - 1, 0].astype(
                bool
            ) & arr1.matrix[:, min_col_ind - 2, 0].astype(bool)
        if min_col_ind + 1 < n_cols:
            left_deleted_collision_mask |= arr1.matrix[:, min_col_ind + 1, 0].astype(
                bool
            ) & arr1.matrix[:, min_col_ind, 0].astype(bool)

        if min_col_ind + 2 < n_cols:
            right_deleted_collision_mask |= arr1.matrix[:, min_col_ind + 1, 0].astype(
                bool
            ) & arr1.matrix[:, min_col_ind + 2, 0].astype(bool)
        if min_col_ind - 1 >= 0:
            right_deleted_collision_mask |= arr1.matrix[:, min_col_ind - 1, 0].astype(
                bool
            ) & arr1.matrix[:, min_col_ind, 0].astype(bool)

        left_deleted_vote_sum: float = 0.0
        right_deleted_vote_sum: float = 0.0

        for row_ind in range(n_target_rows):
            abs_row: int = row_ind + start_row
            row_vote: float = float(total_vote_tally[row_ind])
            if row_vote <= 0:
                continue

            if not bool(left_deleted_collision_mask[abs_row]):
                left_del_comv_AOD_cmds[abs_row] = np.uint8(1)
                left_deleted_vote_sum += row_vote

            if not bool(right_deleted_collision_mask[abs_row]):
                right_del_comv_AOD_cmds[abs_row] = np.uint8(1)
                right_deleted_vote_sum += row_vote

        left_deleted_moves: list[Move] = get_move_list_from_AOD_cmds(
            left_del_comh_AOD_cmds,
            left_del_comv_AOD_cmds,
        )
        right_deleted_moves: list[Move] = get_move_list_from_AOD_cmds(
            right_del_comh_AOD_cmds,
            right_del_comv_AOD_cmds,
        )
        r_moves: list[Move] = get_move_list_from_AOD_cmds(
            r_comh_AOD_cmds,
            r_comv_AOD_cmds,
        )
        l_moves: list[Move] = get_move_list_from_AOD_cmds(
            l_comh_AOD_cmds,
            l_comv_AOD_cmds,
        )

        moves_options: list[list[Move]] = [
            left_deleted_moves,
            right_deleted_moves,
            r_moves,
            l_moves,
        ]
        vote_sums: list[float] = [
            left_deleted_vote_sum,
            right_deleted_vote_sum,
            r_vote_sum,
            l_vote_sum,
        ]

        before_matrix: np.ndarray = arr1.matrix.copy()
        before_overlap: int = _target_overlap_count(before_matrix, arr1.target)

        best_move_list: list[Move] = []
        best_score: tuple[int, float] | None = None

        for move_list, vote_sum in zip(moves_options, vote_sums, strict=True):
            if len(move_list) == 0:
                continue

            after_state: np.ndarray = move_atoms_noiseless(
                before_matrix.copy(), move_list
            )
            state_changed: bool = not np.array_equal(before_matrix, after_state)
            if not state_changed:
                continue

            after_overlap: int = _target_overlap_count(after_state, arr1.target)
            overlap_gain: int = after_overlap - before_overlap
            score: tuple[int, float] = (int(overlap_gain), float(vote_sum))

            if best_score is None or score > best_score:
                best_score = score
                best_move_list = move_list

        if len(best_move_list) == 0:
            break

        touched_rows: set[int] = {int(move.from_row) for move in best_move_list}
        arr1.move_atoms(best_move_list)
        global_move_set.append(best_move_list)

        dirty_rows |= touched_rows

    return global_move_set
