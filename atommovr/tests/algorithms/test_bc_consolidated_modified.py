from __future__ import annotations

import copy
import time
from collections.abc import Callable

import numpy as np
import pytest
import hashlib

import atommovr.utils as movr
from atommovr.algorithms.source import bc_new
from atommovr.tests.support.helpers import (
    _n_atoms,
    _replay_and_check_noiseless_conservation,
)
from atommovr.utils.AtomArray import AtomArray

def _make_single_row_case(
    atom_cols: list[int],
    target_cols: list[int],
    n_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a single-row occupancy/target pair."""
    init_config: np.ndarray = np.zeros((1, n_cols, 1), dtype=np.uint8)
    target_config: np.ndarray = np.zeros((1, n_cols, 1), dtype=np.uint8)
    init_config[0, atom_cols, 0] = np.uint8(1)
    target_config[0, target_cols, 0] = np.uint8(1)
    return init_config, target_config

def _make_atom_array(
    matrix_2d: np.ndarray,
    target_2d: np.ndarray,
) -> AtomArray:
    """Build a single-species AtomArray for BC tests."""
    n_rows: int
    n_cols: int
    n_rows, n_cols = matrix_2d.shape
    arr: AtomArray = AtomArray((n_rows, n_cols))
    arr.matrix = matrix_2d.astype(np.uint8, copy=False).reshape(n_rows, n_cols, 1)
    arr.target = target_2d.astype(np.uint8, copy=False).reshape(n_rows, n_cols, 1)
    return arr



def _serialize_move_rounds(
    move_rounds: list[list[movr.Move]],
) -> list[list[tuple[int, int, int, int]]]:
    """Convert move rounds into plain tuples for exact comparison."""
    return [
        [(m.from_row, m.from_col, m.to_row, m.to_col) for m in round_moves]
        for round_moves in move_rounds
    ]


def _make_random_prebalance_case(
    rng: np.random.Generator,
    n_rows: int,
    n_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a random globally feasible single-species prebalance case."""
    target: np.ndarray = np.zeros((n_rows, n_cols, 1), dtype=np.uint8)
    init: np.ndarray = np.zeros((n_rows, n_cols, 1), dtype=np.uint8)

    band_height: int = int(rng.integers(1, n_rows + 1))
    start_row: int = int(rng.integers(0, n_rows - band_height + 1))
    end_row: int = start_row + band_height - 1

    n_band_sites: int = band_height * n_cols
    n_targets: int = int(rng.integers(1, n_band_sites + 1))

    target_flat: np.ndarray = target[start_row : end_row + 1, :, 0].reshape(-1)
    target_inds: np.ndarray = rng.choice(n_band_sites, size=n_targets, replace=False)
    target_flat[target_inds] = np.uint8(1)

    n_total_sites: int = n_rows * n_cols
    n_atoms: int = int(rng.integers(n_targets, n_total_sites + 1))

    init_flat: np.ndarray = init[:, :, 0].reshape(-1)
    init_inds: np.ndarray = rng.choice(n_total_sites, size=n_atoms, replace=False)
    init_flat[init_inds] = np.uint8(1)

    return init, target

def _half_counts(
    state: np.ndarray,
    target: np.ndarray,
    i: int,
    j: int,
) -> tuple[int, int, int, int, int]:
    """Return top/bottom atom and requirement counts for interval ``[i, j]``."""
    m: int = i + ((j - i + 1) // 2)
    top_atoms: int = bc_new._int_sum(state[i:m, :, :])
    bot_atoms: int = bc_new._int_sum(state[m : j + 1, :, :])
    top_req: int = bc_new._int_sum(target[i:m, :, :])
    bot_req: int = bc_new._int_sum(target[m : j + 1, :, :])
    return m, top_atoms, bot_atoms, top_req, bot_req



def _make_random_balance_case(
    rng: np.random.Generator,
    n_rows: int,
    n_cols: int,
    i: int,
    j: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a random globally feasible ``balance_rows`` case."""
    target: np.ndarray = np.zeros((n_rows, n_cols, 1), dtype=np.uint8)
    init: np.ndarray = np.zeros((n_rows, n_cols, 1), dtype=np.uint8)

    max_sites: int = (j - i + 1) * n_cols
    n_targets: int = int(rng.integers(0, max_sites + 1))
    target_choices: np.ndarray = rng.choice(max_sites, size=n_targets, replace=False)
    target_flat: np.ndarray = target[i : j + 1, :, 0].reshape(-1)
    target_flat[target_choices] = np.uint8(1)

    extra_atoms: int = int(rng.integers(0, max(2, max_sites - n_targets + 1)))
    n_atoms: int = min(max_sites, n_targets + extra_atoms)
    init_choices: np.ndarray = rng.choice(max_sites, size=n_atoms, replace=False)
    init_flat: np.ndarray = init[i : j + 1, :, 0].reshape(-1)
    init_flat[init_choices] = np.uint8(1)

    return init, target



def _make_row_sufficient_case(
    rng: np.random.Generator,
    n_rows: int,
    n_cols: int,
    target_width: int,
    extra_atoms_max: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a random case satisfying compact's row-wise sufficiency precondition."""
    target_2d: np.ndarray = np.zeros((n_rows, n_cols), dtype=np.uint8)
    target_start: int = int(rng.integers(0, n_cols - target_width + 1))
    target_2d[:, target_start : target_start + target_width] = np.uint8(1)

    matrix_2d: np.ndarray = np.zeros((n_rows, n_cols), dtype=np.uint8)
    for row in range(n_rows):
        row_target_count: int = int(np.sum(target_2d[row], dtype=np.int64))
        row_atom_count: int = min(
            n_cols,
            row_target_count + int(rng.integers(0, extra_atoms_max + 1)),
        )
        cols: np.ndarray = rng.choice(n_cols, size=row_atom_count, replace=False)
        matrix_2d[row, cols] = np.uint8(1)

    return matrix_2d, target_2d

def _sample_center_biased_size(
    rng: np.random.Generator,
    max_size: int,
    center: int,
    decay: float = 0.6,
) -> int:
    """
    Sample an integer size in [1, max_size] with exponential bias around `center`.

    Why this exists
    ---------------
    For BCv2 randomized tests, fully uniform rectangle sizes overweight very small
    and very large rectangles relative to the centered "middle fill" regimes that
    the algorithm is most naturally designed for. This helper biases the sampled
    size toward a chosen center with exponentially decaying probability away from it.

    Parameters
    ----------
    rng
        Random number generator.
    max_size
        Maximum allowed sampled size.
    center
        Preferred size to bias around.
    decay
        Exponential decay factor in (0, 1). Smaller values give a sharper peak.

    Returns
    -------
    int
        Sampled size in [1, max_size].
    """
    support: np.ndarray = np.arange(1, max_size + 1, dtype=np.int64)
    weights: np.ndarray = decay ** np.abs(support - int(center))
    probs: np.ndarray = weights / np.sum(weights, dtype=np.float64)
    return int(rng.choice(support, p=probs))

def _make_random_bcv2_feasible_case(
    rng: np.random.Generator,
    n_rows: int,
    n_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a random globally feasible BCv2 case with a fully filled rectangular target.

    The rectangle size is sampled with exponential bias around half-filling in each
    dimension, which better matches the intended BCv2 operating regime than a fully
    uniform rectangle-size distribution.
    """
    target_2d: np.ndarray = np.zeros((n_rows, n_cols), dtype=np.uint8)

    roi_height: int = _sample_center_biased_size(
        rng=rng,
        max_size=n_rows,
        center=max(1, n_rows // 2),
        decay=0.7,
    )
    roi_width: int = _sample_center_biased_size(
        rng=rng,
        max_size=n_cols,
        center=max(1, n_cols // 2),
        decay=0.7,
    )

    start_row: int = int(rng.integers(0, n_rows - roi_height + 1))
    start_col: int = int(rng.integers(0, n_cols - roi_width + 1))
    end_row: int = start_row + roi_height
    end_col: int = start_col + roi_width

    # Fully filled rectangular target region.
    target_2d[start_row:end_row, start_col:end_col] = np.uint8(1)

    n_targets: int = int(roi_height * roi_width)

    matrix_2d: np.ndarray = np.zeros((n_rows, n_cols), dtype=np.uint8)
    n_total_sites: int = n_rows * n_cols
    n_atoms: int = int(rng.integers(n_targets, n_total_sites + 1))

    init_flat: np.ndarray = matrix_2d.reshape(-1)
    atom_inds: np.ndarray = rng.choice(n_total_sites, size=n_atoms, replace=False)
    init_flat[atom_inds] = np.uint8(1)

    return matrix_2d, target_2d

def _time_function(
    func: Callable,
    *args: object,
    repeat: int = 5,
) -> float:
    """Return the best wall-clock runtime over several repeats."""
    best: float = np.inf
    for _ in range(repeat):
        t0: float = time.perf_counter()
        func(*args)
        dt: float = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best



def _legacy_special_case_algo_1d(
    init_config: np.ndarray,
    target_config: np.ndarray,
) -> tuple[list[list[movr.Move]], np.ndarray]:
    """Reference copy of the pre-refactor equal-count 1D helper."""
    arr_copy: AtomArray = AtomArray(np.shape(init_config)[:2])
    arr_copy.target = copy.deepcopy(target_config)
    arr_copy.matrix = copy.deepcopy(init_config)

    target_indices: np.ndarray = np.where(arr_copy.target == 1)[1]
    atom_indices: np.ndarray = np.where(arr_copy.matrix == 1)[1]

    if len(target_indices) != len(atom_indices):
        raise Exception(
            f"Number of atoms ({len(atom_indices)}) does not equal number of target sites ({len(target_indices)})."
        )

    pairs: list[tuple[int, int]] = []
    for ind, target_index in enumerate(target_indices):
        atom_index: int = int(atom_indices[ind])
        pairs.append((int(target_index), atom_index))

    target_prepared: bool = bool(np.array_equal(arr_copy.target, arr_copy.matrix))
    move_set: list[list[movr.Move]] = []
    while not target_prepared:
        move_list: list[movr.Move] = []
        for k, pair in enumerate(pairs):
            target_index: int
            atom_index: int
            target_index, atom_index = pair
            if target_index != atom_index:
                new_atom_index: int = int(atom_index + np.sign(target_index - atom_index))
                move_list.append(movr.Move(0, atom_index, 0, new_atom_index))
                pairs[k] = (target_index, new_atom_index)
        if move_list != []:
            _, _ = arr_copy.evaluate_moves([move_list])
            move_set.append(move_list)
        else:
            break
    return move_set, atom_indices



def _legacy_middle_fill_algo_1d_clamped(
    init_config: np.ndarray,
    target_config: np.ndarray,
) -> tuple[list[list[movr.Move]], list[int]]:
    """Reference-style middle-fill helper with clamped edge handling."""
    arr_copy: AtomArray = AtomArray(list(np.shape(init_config)[:2]))
    arr_copy.target = copy.deepcopy(target_config)
    arr_copy.matrix = copy.deepcopy(init_config)

    target_indices: np.ndarray = np.where(arr_copy.target == 1)[1]
    atom_indices: np.ndarray = np.where(arr_copy.matrix == 1)[1]
    n_targets: int = int(len(target_indices))
    n_atoms: int = int(len(atom_indices))

    if n_targets == 0 or n_atoms == 0 or n_targets > n_atoms:
        return [], []
    if n_targets == n_atoms:
        return _legacy_special_case_algo_1d(init_config, target_config)

    avg_targ_pos: int = int(np.ceil(np.mean(target_indices)))
    count: int = 0
    sufficient_atoms: bool = False
    n_cols: int = int(arr_copy.matrix.shape[1])

    while not sufficient_atoms:
        left: int = max(0, avg_targ_pos - count)
        right: int = min(n_cols, avg_targ_pos + count + 1)
        center_region: np.ndarray = arr_copy.matrix[0, left:right]
        n_atoms_in_center_region: int = int(np.sum(center_region, dtype=np.int64))
        sufficient_atoms = bool(n_targets <= n_atoms_in_center_region)
        if not sufficient_atoms:
            count += 1

    first_atom_loc: int = int(np.where(center_region == 1)[0][0] + left)
    anchor_idx: int = int(np.where(atom_indices == first_atom_loc)[0][0])

    best_atom_set: np.ndarray | None = None
    best_cost: int | None = None

    start_min: int = max(0, anchor_idx - n_targets + 1)
    start_max: int = min(anchor_idx, n_atoms - n_targets)

    for start in range(start_min, start_max + 1):
        candidate: np.ndarray = atom_indices[start : start + n_targets]
        cost: int = int(np.max(np.abs(candidate - target_indices), initial=0))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_atom_set = candidate.copy()

    assert best_atom_set is not None

    pairs: list[tuple[int, int]] = []
    for ind, target_index in enumerate(target_indices):
        pairs.append((int(target_index), int(best_atom_set[ind])))

    arr_copy = AtomArray(list(np.shape(init_config)[:2]))
    arr_copy.target = copy.deepcopy(target_config)
    arr_copy.matrix = copy.deepcopy(init_config)

    move_set: list[list[movr.Move]] = []
    target_prepared: bool = bool(np.array_equal(arr_copy.target, arr_copy.matrix))

    while not target_prepared:
        move_list: list[movr.Move] = []
        for idx, pair in enumerate(pairs):
            target_index: int
            atom_index: int
            target_index, atom_index = pair
            if target_index != atom_index:
                new_atom_index: int = int(atom_index + np.sign(target_index - atom_index))
                move: movr.Move = movr.Move(0, atom_index, 0, new_atom_index)
                move_list.append(move)
                pairs[idx] = (target_index, new_atom_index)

        if move_list != []:
            _, _ = arr_copy.evaluate_moves([move_list])
            move_set.append(move_list)
            target_prepared = bool(np.array_equal(arr_copy.target, arr_copy.matrix))
        else:
            break

    return move_set, best_atom_set.tolist()


class TestGET_TARGET_LOCS:
    def test_returns_bounding_box(self) -> None:
        aa: AtomArray = AtomArray(shape=[4, 5], n_species=1)
        aa.target[:, :, 0] = np.uint8(0)
        aa.target[1, 2, 0] = np.uint8(1)
        aa.target[3, 4, 0] = np.uint8(1)

        sr: int
        sc: int
        er: int
        ec: int
        sr, sc, er, ec = bc_new.get_target_locs(aa)
        assert (sr, sc, er, ec) == (1, 2, 3, 4)

    def test_returns_empty_box_for_empty_target(self) -> None:
        aa: AtomArray = AtomArray(shape=[4, 5], n_species=1)
        aa.target[:, :, 0] = np.uint8(0)

        assert bc_new.get_target_locs(aa) == (0, 0, -1, -1)


class TestGET_ALL_BALANCE_ASSIGNMENTS:
    def test_returns_expected_recursive_intervals_for_0_to_3(self) -> None:
        out: list[tuple[int, int]] = bc_new.get_all_balance_assignments(0, 3)
        expected: set[tuple[int, int]] = {
            (0, 3),
            (0, 1),
            (2, 3),
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
        }
        assert set(out) == expected

    def test_singleton_interval_returns_singleton_only(self) -> None:
        assert bc_new.get_all_balance_assignments(2, 2) == [(2, 2)]

    def test_two_row_interval_returns_root_and_children(self) -> None:
        out: list[tuple[int, int]] = bc_new.get_all_balance_assignments(4, 5)
        assert out == [(4, 5), (4, 4), (5, 5)]


class TestPREBALANCE:
    def test_returns_false_when_global_insufficient(self) -> None:
        init: np.ndarray = np.zeros((3, 3, 1), dtype=np.uint8)
        targ: np.ndarray = np.zeros((3, 3, 1), dtype=np.uint8)
        targ[1, 1, 0] = np.uint8(1)
        targ[1, 2, 0] = np.uint8(1)
        init[0, 0, 0] = np.uint8(1)

        moves, ok = bc_new.prebalance(init, targ)
        assert moves == []
        assert ok is False

    def test_empty_target_returns_no_moves_and_false(self) -> None:
        init: np.ndarray = np.zeros((4, 4, 1), dtype=np.uint8)
        init[0, 0, 0] = np.uint8(1)
        targ: np.ndarray = np.zeros((4, 4, 1), dtype=np.uint8)

        moves, ok = bc_new.prebalance(init, targ)
        assert moves == []
        assert ok is False

    def test_produces_moves_that_increase_atoms_in_target_rows(self) -> None:
        init: np.ndarray = np.zeros((5, 5, 1), dtype=np.uint8)
        targ: np.ndarray = np.zeros((5, 5, 1), dtype=np.uint8)

        targ[2, 1, 0] = np.uint8(1)
        targ[2, 2, 0] = np.uint8(1)
        targ[3, 2, 0] = np.uint8(1)

        init[0, 1, 0] = np.uint8(1)
        init[0, 2, 0] = np.uint8(1)
        init[4, 2, 0] = np.uint8(1)

        moves, ok = bc_new.prebalance(init, targ)
        assert ok is True

        aa: AtomArray = AtomArray(shape=[5, 5], n_species=1)
        aa.matrix = init.copy()
        aa.target = targ.copy()

        before: int = bc_new._int_sum(aa.matrix[2:4, :, :])
        n_targets: int = bc_new._int_sum(targ[2:4, :, :])
        _replay_and_check_noiseless_conservation(aa, moves)
        after: int = bc_new._int_sum(aa.matrix[2:4, :, :])

        assert after >= before
        assert after >= n_targets
    

    def test_REGRESSION_prebalance_passes_blocked_case(self) -> None:
        init = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 1, 1],
                [0, 0, 1, 0, 0, 0, 1],
            ],
            dtype=np.uint8,
        )[:, :, None]

        targ = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )[:, :, None]

        moves, ok = bc_new.prebalance(init, targ)
        

        aa: AtomArray = AtomArray(shape=[7, 6], n_species=1)
        aa.matrix = init.copy()
        aa.target = targ.copy()

        before: int = bc_new._int_sum(aa.matrix[0:3, :, :])
        n_targets: int = bc_new._int_sum(targ[0:3, :, :])
        _replay_and_check_noiseless_conservation(aa, moves)
        after: int = bc_new._int_sum(aa.matrix[0:3, :, :])

        assert ok
        assert after >= before
        assert after >= n_targets
        assert np.array_equal(np.multiply(aa.matrix, aa.target), aa.target)

class TestBALANCE_ROWS:
    def test_raises_on_insufficient_atoms(self) -> None:
        init: np.ndarray = np.zeros((4, 4, 1), dtype=np.uint8)
        targ: np.ndarray = np.zeros((4, 4, 1), dtype=np.uint8)

        targ[0, 0, 0] = np.uint8(1)
        targ[0, 1, 0] = np.uint8(1)
        targ[2, 0, 0] = np.uint8(1)
        targ[2, 1, 0] = np.uint8(1)

        init[0, 0, 0] = np.uint8(1)
        init[2, 0, 0] = np.uint8(1)

        with pytest.raises(ValueError, match="Insufficient number of atoms"):
            bc_new.balance_rows(init, targ, 0, 3)

    def test_returns_empty_when_already_balanced(self) -> None:
        init: np.ndarray = np.zeros((4, 4, 1), dtype=np.uint8)
        targ: np.ndarray = np.zeros((4, 4, 1), dtype=np.uint8)

        targ[0, 0, 0] = np.uint8(1)
        targ[2, 0, 0] = np.uint8(1)
        init[0, 0, 0] = np.uint8(1)
        init[2, 0, 0] = np.uint8(1)

        assert bc_new.balance_rows(init, targ, 0, 3) == []

    def test_returns_empty_when_both_halves_already_feasible(self) -> None:
        init: np.ndarray = np.zeros((4, 4, 1), dtype=np.uint8)
        targ: np.ndarray = np.zeros((4, 4, 1), dtype=np.uint8)

        targ[0, 0, 0] = np.uint8(1)
        init[0, 0, 0] = np.uint8(1)
        init[1, 1, 0] = np.uint8(1)

        targ[2, 0, 0] = np.uint8(1)
        init[2, 0, 0] = np.uint8(1)
        init[2, 1, 0] = np.uint8(1)
        init[3, 2, 0] = np.uint8(1)

        assert bc_new.balance_rows(init, targ, 0, 3) == []

    def test_single_row_interval_returns_empty(self) -> None:
        init: np.ndarray = np.zeros((3, 4, 1), dtype=np.uint8)
        targ: np.ndarray = np.zeros((3, 4, 1), dtype=np.uint8)
        init[1, 1, 0] = np.uint8(1)
        targ[1, 2, 0] = np.uint8(1)

        assert bc_new.balance_rows(init, targ, 1, 1) == []

    def test_two_row_interval_moves_atom_downward_exactly(self) -> None:
        init: np.ndarray = np.zeros((2, 4, 1), dtype=np.uint8)
        targ: np.ndarray = np.zeros((2, 4, 1), dtype=np.uint8)

        targ[0, 0, 0] = np.uint8(1)
        targ[1, 1, 0] = np.uint8(1)

        init[0, 1, 0] = np.uint8(1)
        init[0, 3, 0] = np.uint8(1)

        move_rounds: list[list[movr.Move]] = bc_new.balance_rows(init, targ, 0, 1)

        assert _serialize_move_rounds(move_rounds) == [[(0, 1, 1, 1)]]

        aa: AtomArray = AtomArray(shape=[2, 4], n_species=1)
        aa.matrix = init.copy()
        aa.target = targ.copy()
        _replay_and_check_noiseless_conservation(aa, move_rounds)

        _, top_atoms, bot_atoms, top_req, bot_req = _half_counts(aa.matrix, targ, 0, 1)
        assert top_atoms >= top_req
        assert bot_atoms >= bot_req

    def test_two_row_interval_moves_atom_upward_exactly(self) -> None:
        init: np.ndarray = np.zeros((2, 4, 1), dtype=np.uint8)
        targ: np.ndarray = np.zeros((2, 4, 1), dtype=np.uint8)

        targ[0, 0, 0] = np.uint8(1)
        targ[1, 1, 0] = np.uint8(1)

        init[1, 1, 0] = np.uint8(1)
        init[1, 3, 0] = np.uint8(1)

        move_rounds: list[list[movr.Move]] = bc_new.balance_rows(init, targ, 0, 1)

        assert _serialize_move_rounds(move_rounds) == [[(1, 1, 0, 1)]]

        aa: AtomArray = AtomArray(shape=[2, 4], n_species=1)
        aa.matrix = init.copy()
        aa.target = targ.copy()
        _replay_and_check_noiseless_conservation(aa, move_rounds)

        _, top_atoms, bot_atoms, top_req, bot_req = _half_counts(aa.matrix, targ, 0, 1)
        assert top_atoms >= top_req
        assert bot_atoms >= bot_req

    def test_moves_atoms_from_surplus_half_to_deficit_half(self) -> None:
        init: np.ndarray = np.zeros((4, 4, 1), dtype=np.uint8)
        targ: np.ndarray = np.zeros((4, 4, 1), dtype=np.uint8)

        targ[0, 0, 0] = np.uint8(1)
        targ[1, 1, 0] = np.uint8(1)
        targ[2, 0, 0] = np.uint8(1)

        init[2, 0, 0] = np.uint8(1)
        init[2, 1, 0] = np.uint8(1)
        init[3, 2, 0] = np.uint8(1)

        move_rounds: list[list[movr.Move]] = bc_new.balance_rows(init, targ, 0, 3)

        aa: AtomArray = AtomArray(shape=[4, 4], n_species=1)
        aa.matrix = init.copy()
        aa.target = targ.copy()
        _replay_and_check_noiseless_conservation(aa, move_rounds)

        _, top_atoms, bot_atoms, top_req, bot_req = _half_counts(aa.matrix, targ, 0, 3)
        assert top_atoms >= top_req
        assert bot_atoms >= bot_req

    def test_replay_preserves_child_half_feasibility(self) -> None:
        init: np.ndarray = np.zeros((8, 5, 1), dtype=np.uint8)
        targ: np.ndarray = np.zeros((8, 5, 1), dtype=np.uint8)

        targ[0, 0, 0] = np.uint8(1)
        targ[0, 1, 0] = np.uint8(1)
        targ[1, 0, 0] = np.uint8(1)
        targ[1, 1, 0] = np.uint8(1)
        targ[2, 0, 0] = np.uint8(1)
        targ[3, 0, 0] = np.uint8(1)
        targ[4, 0, 0] = np.uint8(1)
        targ[5, 0, 0] = np.uint8(1)

        init[4, 0, 0] = np.uint8(1)
        init[4, 1, 0] = np.uint8(1)
        init[5, 0, 0] = np.uint8(1)
        init[5, 1, 0] = np.uint8(1)
        init[6, 0, 0] = np.uint8(1)
        init[6, 1, 0] = np.uint8(1)
        init[7, 0, 0] = np.uint8(1)
        init[7, 1, 0] = np.uint8(1)

        move_rounds: list[list[movr.Move]] = bc_new.balance_rows(init, targ, 0, 7)

        aa: AtomArray = AtomArray(shape=[8, 5], n_species=1)
        aa.matrix = init.copy()
        aa.target = targ.copy()
        _replay_and_check_noiseless_conservation(aa, move_rounds)

        _, top_atoms, bot_atoms, top_req, bot_req = _half_counts(aa.matrix, targ, 0, 7)
        assert top_atoms >= top_req
        assert bot_atoms >= bot_req

    def test_random_postcondition(self) -> None:
        """Random feasible cases should preserve atom number and child feasibility."""
        rng: np.random.Generator = np.random.default_rng(0)

        for n_rows, n_cols in [(2, 4), (4, 5), (6, 7), (8, 8)]:
            for _ in range(200):
                init: np.ndarray
                targ: np.ndarray
                init, targ = _make_random_balance_case(
                    rng=rng,
                    n_rows=n_rows,
                    n_cols=n_cols,
                    i=0,
                    j=n_rows - 1,
                )

                if bc_new._int_sum(init) < bc_new._int_sum(targ):
                    continue

                move_rounds: list[list[movr.Move]] = bc_new.balance_rows(
                    init,
                    targ,
                    0,
                    n_rows - 1,
                )

                aa: AtomArray = AtomArray(shape=[n_rows, n_cols], n_species=1)
                aa.matrix = init.copy()
                aa.target = targ.copy()
                n_before: int = _n_atoms(aa.matrix)
                _replay_and_check_noiseless_conservation(aa, move_rounds)
                n_after: int = _n_atoms(aa.matrix)

                _, top_atoms, bot_atoms, top_req, bot_req = _half_counts(
                    aa.matrix,
                    targ,
                    0,
                    n_rows - 1,
                )
                assert n_after == n_before
                assert top_atoms >= top_req
                assert bot_atoms >= bot_req

class TestBALANCE_PHASE_INTERFACE:
    def test_full_recursive_balance_leaves_every_target_row_feasible(self) -> None:
        rng: np.random.Generator = np.random.default_rng(12)

        for n_rows, n_cols in [(4, 5), (6, 7), (8, 8)]:
            for _ in range(60):
                init: np.ndarray
                targ: np.ndarray
                init, targ = _make_random_prebalance_case(rng, n_rows, n_cols)

                pre_moves: list[list[movr.Move]]
                _: object
                ok: bool
                pre_moves, ok = bc_new.prebalance(init, targ)
                assert ok is True

                aa: AtomArray = AtomArray(shape=[n_rows, n_cols], n_species=1)
                aa.matrix = init.copy()
                aa.target = targ.copy()
                _replay_and_check_noiseless_conservation(aa, pre_moves)

                start_row: int
                start_col: int
                end_row: int
                end_col: int
                start_row, start_col, end_row, end_col = bc_new.get_target_locs(aa)

                assignments: list[tuple[int, int]] = bc_new.get_all_balance_assignments(
                    start_row,
                    end_row,
                )

                for i, j in assignments:
                    move_rounds: list[list[movr.Move]] = bc_new.balance_rows(
                        aa.matrix,
                        aa.target,
                        i,
                        j,
                    )
                    if len(move_rounds) > 0:
                        _replay_and_check_noiseless_conservation(aa, move_rounds)

                for row in range(start_row, end_row + 1):
                    row_atoms: int = int(np.sum(aa.matrix[row, :, :], dtype=np.int64))
                    row_req: int = int(np.sum(targ[row, :, :], dtype=np.int64))
                    assert row_atoms >= row_req, (
                        f"Row {row} ended balance with {row_atoms} atoms but "
                        f"requires {row_req}."
                    )

class TestSPECIAL_CASE_ALGO_1D_EXPECTED_OUTPUT:
    def test_returns_empty_when_no_atoms_and_no_targets(self) -> None:
        init_config, target_config = _make_single_row_case([], [], 6)

        move_rounds, atom_set = bc_new.special_case_algo_1d(init_config, target_config)

        assert move_rounds == []
        assert atom_set == []

    def test_returns_empty_move_set_when_already_matched(self) -> None:
        init_config, target_config = _make_single_row_case([1, 3, 5], [1, 3, 5], 7)

        move_rounds, atom_set = bc_new.special_case_algo_1d(init_config, target_config)

        assert atom_set == [1, 3, 5]
        assert move_rounds == []

    def test_moves_atoms_in_parallel_toward_targets(self) -> None:
        init_config, target_config = _make_single_row_case([0, 4], [1, 3], 5)

        move_rounds, atom_set = bc_new.special_case_algo_1d(init_config, target_config)

        assert atom_set == [0, 4]
        assert _serialize_move_rounds(move_rounds) == [
            [(0, 0, 0, 1), (0, 4, 0, 3)],
        ]

    def test_generates_multiple_rounds_when_distance_exceeds_one(self) -> None:
        init_config, target_config = _make_single_row_case([0, 5], [2, 3], 6)

        move_rounds, atom_set = bc_new.special_case_algo_1d(init_config, target_config)

        assert atom_set == [0, 5]
        assert _serialize_move_rounds(move_rounds) == [
            [(0, 0, 0, 1), (0, 5, 0, 4)],
            [(0, 1, 0, 2), (0, 4, 0, 3)],
        ]


class TestMIDDLE_FILL_ALGO_1D_EXPECTED_OUTPUT:
    def test_returns_empty_if_no_targets(self) -> None:
        init_config, target_config = _make_single_row_case([1, 4], [], 6)

        move_rounds, atom_set = bc_new.middle_fill_algo_1d(init_config, target_config)

        assert move_rounds == []
        assert atom_set == []

    def test_returns_empty_if_insufficient_atoms(self) -> None:
        init_config, target_config = _make_single_row_case([2], [1, 2, 3], 6)

        move_rounds, atom_set = bc_new.middle_fill_algo_1d(init_config, target_config)

        assert move_rounds == []
        assert atom_set == []

    def test_reduces_to_special_case_when_counts_match(self) -> None:
        init_config, target_config = _make_single_row_case([0, 3, 5], [1, 2, 4], 6)

        move_rounds_mid, atom_set_mid = bc_new.middle_fill_algo_1d(init_config, target_config)
        move_rounds_sp, atom_set_sp = bc_new.special_case_algo_1d(init_config, target_config)

        assert atom_set_mid == atom_set_sp
        assert _serialize_move_rounds(move_rounds_mid) == _serialize_move_rounds(move_rounds_sp)

    def test_selects_obvious_centered_block(self) -> None:
        init_config: np.ndarray
        target_config: np.ndarray
        init_config, target_config = _make_single_row_case(
            [0, 2, 4, 6, 8],
            [3, 4, 5],
            9,
        )

        move_rounds: list[list[movr.Move]]
        atom_set: list[int]
        move_rounds, atom_set = bc_new.middle_fill_algo_1d(init_config, target_config)

        assert atom_set == [2, 4, 6]
        assert _serialize_move_rounds(move_rounds) == [
            [(0, 2, 0, 3), (0, 6, 0, 5)],
        ]

    def test_selects_leftmost_best_window_when_unique(self) -> None:
        init_config, target_config = _make_single_row_case(
            [0, 1, 4, 7, 8],
            [0, 1, 2],
            9,
        )

        move_rounds, atom_set = bc_new.middle_fill_algo_1d(init_config, target_config)

        assert atom_set == [0, 1, 4]
        assert _serialize_move_rounds(move_rounds) == [
            [(0, 4, 0, 3)],
            [(0, 3, 0, 2)],
        ]

    def test_selects_right_block_near_right_edge(self) -> None:
        init_config, target_config = _make_single_row_case(
            [1, 3, 5, 7, 8],
            [6, 7, 8],
            9,
        )

        move_rounds, atom_set = bc_new.middle_fill_algo_1d(init_config, target_config)

        assert atom_set == [5, 7, 8]
        assert _serialize_move_rounds(move_rounds) == [
            [(0, 5, 0, 6)],
        ]


class TestMIDDLE_FILL_ALGO_1D_RANDOM_VALIDITY:
    def test_random_outputs_are_structurally_valid(self) -> None:
        rng = np.random.default_rng(0)

        for n_cols in [5, 8, 12, 24]:
            for _ in range(300):
                n_atoms = int(rng.integers(0, n_cols + 1))
                n_targets = int(rng.integers(0, n_cols + 1))

                atom_cols = sorted(rng.choice(n_cols, size=n_atoms, replace=False).tolist())
                target_cols = sorted(rng.choice(n_cols, size=n_targets, replace=False).tolist())

                init_config, target_config = _make_single_row_case(atom_cols, target_cols, n_cols)
                move_rounds, atom_set = bc_new.middle_fill_algo_1d(init_config, target_config)

                if n_targets == 0 or n_atoms < n_targets:
                    assert move_rounds == []
                    assert atom_set == []
                    continue

                assert len(atom_set) == n_targets
                assert all(col in atom_cols for col in atom_set)
                assert list(atom_set) == sorted(atom_set)

                serialized = _serialize_move_rounds(move_rounds)
                for round_moves in serialized:
                    src_cols = [m[1] for m in round_moves]
                    dst_cols = [m[3] for m in round_moves]
                    assert len(src_cols) == len(set(src_cols))
                    assert len(dst_cols) == len(set(dst_cols))
                    assert all(abs(dst - src) == 1 for _, src, _, dst in round_moves)
    
    def test_middle_fill_algo_1d_does_not_crash_when_center_window_expands_past_left_edge(self) -> None:
        init_config: np.ndarray = np.zeros((1, 12, 1), dtype=np.uint8)
        target_config: np.ndarray = np.zeros((1, 12, 1), dtype=np.uint8)

        target_config[0, 5:9, 0] = np.uint8(1)
        init_config[0, [0, 1, 2, 3, 9, 11], 0] = np.uint8(1)

        move_rounds, best_atom_set = bc_new.middle_fill_algo_1d(init_config, target_config)

        assert isinstance(move_rounds, list)
        assert len(best_atom_set) == int(np.sum(target_config, dtype=np.int64))

class TestCHOOSE_BEST_ATOM_SET_1D:
    def test_returns_empty_if_no_targets(self) -> None:
        init_config, target_config = _make_single_row_case([1, 4], [], 6)

        best_atom_set: np.ndarray = bc_new.choose_best_atom_set_1d(
            init_config,
            target_config,
        )

        assert isinstance(best_atom_set, np.ndarray)
        assert best_atom_set.dtype == np.intp
        assert best_atom_set.tolist() == []

    def test_raises_if_insufficient_atoms(self) -> None:
        init_config, target_config = _make_single_row_case([2], [1, 2, 3], 6)

        with pytest.raises(ValueError, match="at least as many row atoms as target sites"):
            bc_new.choose_best_atom_set_1d(init_config, target_config)

    def test_reduces_to_special_case_when_counts_match(self) -> None:
        init_config, target_config = _make_single_row_case([0, 3, 5], [1, 2, 4], 6)

        best_atom_set: np.ndarray = bc_new.choose_best_atom_set_1d(
            init_config,
            target_config,
        )
        _, atom_set_sp = bc_new.special_case_algo_1d(init_config, target_config)

        assert best_atom_set.tolist() == atom_set_sp

    def test_selects_obvious_centered_block(self) -> None:
        init_config, target_config = _make_single_row_case(
            [0, 2, 4, 6, 8],
            [3, 4, 5],
            9,
        )

        best_atom_set: np.ndarray = bc_new.choose_best_atom_set_1d(
            init_config,
            target_config,
        )

        assert best_atom_set.tolist() == [2, 4, 6]

    def test_selects_leftmost_best_window_when_unique(self) -> None:
        init_config, target_config = _make_single_row_case(
            [0, 1, 4, 7, 8],
            [0, 1, 2],
            9,
        )

        best_atom_set: np.ndarray = bc_new.choose_best_atom_set_1d(
            init_config,
            target_config,
        )

        assert best_atom_set.tolist() == [0, 1, 4]

    def test_selects_right_block_near_right_edge(self) -> None:
        init_config, target_config = _make_single_row_case(
            [1, 3, 5, 7, 8],
            [6, 7, 8],
            9,
        )

        best_atom_set: np.ndarray = bc_new.choose_best_atom_set_1d(
            init_config,
            target_config,
        )

        assert best_atom_set.tolist() == [5, 7, 8]

    def test_does_not_crash_when_center_window_expands_past_left_edge(self) -> None:
        init_config: np.ndarray = np.zeros((1, 12, 1), dtype=np.uint8)
        target_config: np.ndarray = np.zeros((1, 12, 1), dtype=np.uint8)

        target_config[0, 5:9, 0] = np.uint8(1)
        init_config[0, [0, 1, 2, 3, 9, 11], 0] = np.uint8(1)

        best_atom_set: np.ndarray = bc_new.choose_best_atom_set_1d(
            init_config,
            target_config,
        )

        assert isinstance(best_atom_set, np.ndarray)
        assert best_atom_set.dtype == np.intp
        assert len(best_atom_set) == int(np.sum(target_config, dtype=np.int64))

    def test_random_outputs_are_structurally_valid(self) -> None:
        rng: np.random.Generator = np.random.default_rng(0)

        for n_cols in [5, 8, 12, 24]:
            for _ in range(300):
                n_atoms: int = int(rng.integers(0, n_cols + 1))
                n_targets: int = int(rng.integers(0, n_cols + 1))

                atom_cols: list[int] = sorted(
                    rng.choice(n_cols, size=n_atoms, replace=False).tolist()
                )
                target_cols: list[int] = sorted(
                    rng.choice(n_cols, size=n_targets, replace=False).tolist()
                )

                init_config: np.ndarray
                target_config: np.ndarray
                init_config, target_config = _make_single_row_case(
                    atom_cols,
                    target_cols,
                    n_cols,
                )

                if n_targets == 0:
                    best_atom_set = bc_new.choose_best_atom_set_1d(
                        init_config,
                        target_config,
                    )
                    assert best_atom_set.tolist() == []
                    continue

                if n_atoms < n_targets:
                    with pytest.raises(ValueError):
                        bc_new.choose_best_atom_set_1d(init_config, target_config)
                    continue

                best_atom_set = bc_new.choose_best_atom_set_1d(
                    init_config,
                    target_config,
                )

                assert len(best_atom_set) == n_targets
                assert best_atom_set.dtype == np.intp
                assert list(best_atom_set) == sorted(best_atom_set.tolist())
                assert all(int(col) in atom_cols for col in best_atom_set.tolist())

                atom_indices: list[int] = sorted(atom_cols)
                chosen_positions: list[int] = [
                    atom_indices.index(int(col)) for col in best_atom_set.tolist()
                ]
                expected_positions: list[int] = list(
                    range(chosen_positions[0], chosen_positions[0] + n_targets)
                )
                assert chosen_positions == expected_positions


class TestFIRST_ROUND_EDGES_FOR_BEST_SET:
    def test_returns_empty_when_no_targets(self) -> None:
        init_config, target_config = _make_single_row_case([], [], 6)
        best_atom_set: np.ndarray = np.asarray([], dtype=np.intp)

        edges: set[tuple[int, int]] = bc_new.first_round_edges_for_best_set(
            init_config,
            target_config,
            best_atom_set=best_atom_set,
        )

        assert edges == set()

    def test_returns_empty_when_already_matched(self) -> None:
        init_config, target_config = _make_single_row_case([1, 3, 5], [1, 3, 5], 7)
        best_atom_set: np.ndarray = np.asarray([1, 3, 5], dtype=np.intp)

        edges: set[tuple[int, int]] = bc_new.first_round_edges_for_best_set(
            init_config,
            target_config,
            best_atom_set=best_atom_set,
        )

        assert edges == set()

    def test_returns_parallel_first_step_edges_toward_targets(self) -> None:
        init_config, target_config = _make_single_row_case([0, 4], [1, 3], 5)
        best_atom_set: np.ndarray = np.asarray([0, 4], dtype=np.intp)

        edges: set[tuple[int, int]] = bc_new.first_round_edges_for_best_set(
            init_config,
            target_config,
            best_atom_set=best_atom_set,
        )

        assert edges == {(0, 1), (4, 3)}

    def test_returns_first_step_edges_when_distance_exceeds_one(self) -> None:
        init_config, target_config = _make_single_row_case([0, 5], [2, 3], 6)
        best_atom_set: np.ndarray = np.asarray([0, 5], dtype=np.intp)

        edges: set[tuple[int, int]] = bc_new.first_round_edges_for_best_set(
            init_config,
            target_config,
            best_atom_set=best_atom_set,
        )

        assert edges == {(0, 1), (5, 4)}

    def test_matches_old_middle_fill_first_round_on_centered_case(self) -> None:
        init_config, target_config = _make_single_row_case(
            [0, 2, 4, 6, 8],
            [3, 4, 5],
            9,
        )

        best_atom_set: np.ndarray = bc_new.choose_best_atom_set_1d(
            init_config,
            target_config,
        )
        edges: set[tuple[int, int]] = bc_new.first_round_edges_for_best_set(
            init_config,
            target_config,
            best_atom_set=best_atom_set,
        )

        move_rounds_old, atom_set_old = bc_new.middle_fill_algo_1d(init_config, target_config)

        assert best_atom_set.tolist() == atom_set_old
        assert edges == {(2, 3), (6, 5)}
        assert edges == {
            (int(move.from_col), int(move.to_col))
            for move in move_rounds_old[0]
        }

    def test_matches_old_middle_fill_first_round_on_left_case(self) -> None:
        init_config, target_config = _make_single_row_case(
            [0, 1, 4, 7, 8],
            [0, 1, 2],
            9,
        )

        best_atom_set: np.ndarray = bc_new.choose_best_atom_set_1d(
            init_config,
            target_config,
        )
        edges: set[tuple[int, int]] = bc_new.first_round_edges_for_best_set(
            init_config,
            target_config,
            best_atom_set=best_atom_set,
        )

        move_rounds_old, atom_set_old = bc_new.middle_fill_algo_1d(init_config, target_config)

        assert best_atom_set.tolist() == atom_set_old
        assert edges == {(4, 3)}
        assert edges == {
            (int(move.from_col), int(move.to_col))
            for move in move_rounds_old[0]
        }

    def test_matches_old_middle_fill_first_round_on_right_case(self) -> None:
        init_config, target_config = _make_single_row_case(
            [1, 3, 5, 7, 8],
            [6, 7, 8],
            9,
        )

        best_atom_set: np.ndarray = bc_new.choose_best_atom_set_1d(
            init_config,
            target_config,
        )
        edges: set[tuple[int, int]] = bc_new.first_round_edges_for_best_set(
            init_config,
            target_config,
            best_atom_set=best_atom_set,
        )

        move_rounds_old, atom_set_old = bc_new.middle_fill_algo_1d(init_config, target_config)

        assert best_atom_set.tolist() == atom_set_old
        assert edges == {(5, 6)}
        assert edges == {
            (int(move.from_col), int(move.to_col))
            for move in move_rounds_old[0]
        }

    def test_matches_old_special_case_first_round_when_counts_match(self) -> None:
        init_config, target_config = _make_single_row_case([0, 3, 5], [1, 2, 4], 6)

        best_atom_set: np.ndarray = bc_new.choose_best_atom_set_1d(
            init_config,
            target_config,
        )
        edges: set[tuple[int, int]] = bc_new.first_round_edges_for_best_set(
            init_config,
            target_config,
            best_atom_set=best_atom_set,
        )

        move_rounds_old, atom_set_old = bc_new.special_case_algo_1d(init_config, target_config)

        assert best_atom_set.tolist() == atom_set_old
        assert edges == {
            (int(move.from_col), int(move.to_col))
            for move in move_rounds_old[0]
        }

    def test_raises_if_best_atom_count_does_not_match_target_count(self) -> None:
        init_config, target_config = _make_single_row_case([0, 2, 4], [1, 3], 5)
        bad_best_atom_set: np.ndarray = np.asarray([0], dtype=np.intp)

        with pytest.raises(ValueError, match="size must match"):
            bc_new.first_round_edges_for_best_set(
                init_config,
                target_config,
                best_atom_set=bad_best_atom_set,
            )

    def test_random_outputs_are_structurally_valid(self) -> None:
        rng: np.random.Generator = np.random.default_rng(1)

        for n_cols in [5, 8, 12, 24]:
            for _ in range(300):
                n_atoms: int = int(rng.integers(0, n_cols + 1))
                n_targets: int = int(rng.integers(0, n_cols + 1))

                atom_cols: list[int] = sorted(
                    rng.choice(n_cols, size=n_atoms, replace=False).tolist()
                )
                target_cols: list[int] = sorted(
                    rng.choice(n_cols, size=n_targets, replace=False).tolist()
                )

                init_config: np.ndarray
                target_config: np.ndarray
                init_config, target_config = _make_single_row_case(
                    atom_cols,
                    target_cols,
                    n_cols,
                )

                if n_targets == 0:
                    best_atom_set: np.ndarray = bc_new.choose_best_atom_set_1d(
                        init_config,
                        target_config,
                    )
                    edges: set[tuple[int, int]] = bc_new.first_round_edges_for_best_set(
                        init_config,
                        target_config,
                        best_atom_set=best_atom_set,
                    )
                    assert edges == set()
                    continue

                if n_atoms < n_targets:
                    continue

                best_atom_set = bc_new.choose_best_atom_set_1d(
                    init_config,
                    target_config,
                )
                edges = bc_new.first_round_edges_for_best_set(
                    init_config,
                    target_config,
                    best_atom_set=best_atom_set,
                )

                src_cols: list[int] = [src for src, _ in edges]
                dst_cols: list[int] = [dst for _, dst in edges]

                assert len(src_cols) == len(set(src_cols))
                assert len(dst_cols) == len(set(dst_cols))
                assert all(abs(int(dst) - int(src)) == 1 for src, dst in edges)
                assert all(int(src) in best_atom_set.tolist() for src in src_cols)

                target_indices: np.ndarray = np.flatnonzero(target_config[0, :, 0]).astype(
                    np.intp,
                    copy=False,
                )
                for atom_col, target_col in zip(
                    best_atom_set.tolist(),
                    target_indices.tolist(),
                    strict=True,
                ):
                    if atom_col < target_col:
                        assert (int(atom_col), int(atom_col + 1)) in edges
                    elif atom_col > target_col:
                        assert (int(atom_col), int(atom_col - 1)) in edges
                    else:
                        assert all(int(src) != int(atom_col) for src in src_cols)


class TestCHOOSE_BEST_ATOM_SET_1D_COMPATIBILITY:
    @pytest.mark.parametrize("n_cols", [5, 8, 12, 24])
    def test_matches_old_middle_fill_atom_set_on_small_random_cases(self, n_cols: int) -> None:
        rng: np.random.Generator = np.random.default_rng(20260416 + n_cols)

        for _ in range(400):
            n_atoms: int = int(rng.integers(0, n_cols + 1))
            n_targets: int = int(rng.integers(0, n_cols + 1))

            atom_cols: list[int] = sorted(
                rng.choice(n_cols, size=n_atoms, replace=False).tolist()
            )
            target_cols: list[int] = sorted(
                rng.choice(n_cols, size=n_targets, replace=False).tolist()
            )

            init_config: np.ndarray
            target_config: np.ndarray
            init_config, target_config = _make_single_row_case(
                atom_cols,
                target_cols,
                n_cols,
            )

            if n_targets == 0:
                best_atom_set: np.ndarray = bc_new.choose_best_atom_set_1d(
                    init_config,
                    target_config,
                )
                move_rounds_old, atom_set_old = bc_new.middle_fill_algo_1d(
                    init_config,
                    target_config,
                )
                assert move_rounds_old == []
                assert best_atom_set.tolist() == atom_set_old
                continue

            if n_atoms < n_targets:
                with pytest.raises(ValueError):
                    bc_new.choose_best_atom_set_1d(init_config, target_config)
                move_rounds_old, atom_set_old = bc_new.middle_fill_algo_1d(
                    init_config,
                    target_config,
                )
                assert move_rounds_old == []
                assert atom_set_old == []
                continue

            best_atom_set = bc_new.choose_best_atom_set_1d(
                init_config,
                target_config,
            )
            _, atom_set_old = bc_new.middle_fill_algo_1d(
                init_config,
                target_config,
            )

            assert best_atom_set.tolist() == atom_set_old


class TestFIRST_ROUND_EDGES_FOR_BEST_SET_COMPATIBILITY:
    @pytest.mark.parametrize("n_cols", [5, 8, 12, 24])
    def test_matches_old_middle_fill_first_round_on_small_random_cases(self, n_cols: int) -> None:
        rng: np.random.Generator = np.random.default_rng(20260429 + n_cols)

        for _ in range(400):
            n_atoms: int = int(rng.integers(0, n_cols + 1))
            n_targets: int = int(rng.integers(0, n_cols + 1))

            atom_cols: list[int] = sorted(
                rng.choice(n_cols, size=n_atoms, replace=False).tolist()
            )
            target_cols: list[int] = sorted(
                rng.choice(n_cols, size=n_targets, replace=False).tolist()
            )

            init_config: np.ndarray
            target_config: np.ndarray
            init_config, target_config = _make_single_row_case(
                atom_cols,
                target_cols,
                n_cols,
            )

            if n_targets == 0 or n_atoms < n_targets:
                continue

            best_atom_set: np.ndarray = bc_new.choose_best_atom_set_1d(
                init_config,
                target_config,
            )
            edges_new: set[tuple[int, int]] = bc_new.first_round_edges_for_best_set(
                init_config,
                target_config,
                best_atom_set=best_atom_set,
            )

            move_rounds_old, atom_set_old = bc_new.middle_fill_algo_1d(
                init_config,
                target_config,
            )

            assert best_atom_set.tolist() == atom_set_old

            if len(move_rounds_old) == 0:
                assert edges_new == set()
            else:
                edges_old: set[tuple[int, int]] = {
                    (int(move.from_col), int(move.to_col))
                    for move in move_rounds_old[0]
                }
                assert edges_new == edges_old

class TestCOMPACT:
    def test_returns_empty_for_empty_target(self) -> None:
        aa: AtomArray = AtomArray(shape=[4, 5], n_species=1)
        aa.matrix[:, :, 0] = np.uint8(0)
        aa.matrix[0, 0, 0] = np.uint8(1)
        aa.matrix[3, 4, 0] = np.uint8(1)
        aa.target[:, :, 0] = np.uint8(0)

        assert bc_new.compact(aa) == []

    def test_returns_empty_when_target_already_filled_exactly(self) -> None:
        aa: AtomArray = AtomArray(shape=[3, 5], n_species=1)
        aa.matrix[:, :, 0] = np.uint8(0)
        aa.target[:, :, 0] = np.uint8(0)
        aa.matrix[1, 1, 0] = np.uint8(1)
        aa.matrix[1, 2, 0] = np.uint8(1)
        aa.target[1, 1, 0] = np.uint8(1)
        aa.target[1, 2, 0] = np.uint8(1)

        assert bc_new.compact(aa) == []

    def test_returns_empty_when_target_region_is_prepared_but_surplus_exists_elsewhere(self) -> None:
        aa: AtomArray = AtomArray(shape=[3, 7], n_species=1)
        aa.matrix[:, :, 0] = np.uint8(0)
        aa.target[:, :, 0] = np.uint8(0)

        aa.target[1, 2, 0] = np.uint8(1)
        aa.target[1, 3, 0] = np.uint8(1)
        aa.target[1, 4, 0] = np.uint8(1)

        aa.matrix[1, 2, 0] = np.uint8(1)
        aa.matrix[1, 3, 0] = np.uint8(1)
        aa.matrix[1, 4, 0] = np.uint8(1)
        aa.matrix[0, 0, 0] = np.uint8(1)
        aa.matrix[2, 6, 0] = np.uint8(1)

        assert bc_new.compact(aa) == []

    def test_raises_when_any_target_row_is_rowwise_insufficient(self) -> None:
        aa: AtomArray = AtomArray(shape=[2, 5], n_species=1)
        aa.matrix[:, :, 0] = np.uint8(0)
        aa.target[:, :, 0] = np.uint8(0)

        aa.target[0, 1, 0] = np.uint8(1)
        aa.target[0, 2, 0] = np.uint8(1)
        aa.target[1, 1, 0] = np.uint8(1)
        aa.target[1, 2, 0] = np.uint8(1)

        aa.matrix[0, 0, 0] = np.uint8(1)
        aa.matrix[0, 4, 0] = np.uint8(1)
        aa.matrix[1, 0, 0] = np.uint8(1)

        with pytest.raises(ValueError, match='compact\(\) requires each target row'):
            bc_new.compact(aa)

    def test_compact_terminates_and_conserves_atoms(self) -> None:
        aa: AtomArray = AtomArray(shape=[6, 6], n_species=1)
        aa.target[:, :, 0] = np.uint8(0)
        aa.target[2, 2, 0] = np.uint8(1)
        aa.target[2, 3, 0] = np.uint8(1)
        aa.target[3, 2, 0] = np.uint8(1)
        aa.target[3, 3, 0] = np.uint8(1)

        aa.matrix[:, :, 0] = np.uint8(0)
        aa.matrix[2, 0, 0] = np.uint8(1)
        aa.matrix[2, 5, 0] = np.uint8(1)
        aa.matrix[3, 0, 0] = np.uint8(1)
        aa.matrix[3, 5, 0] = np.uint8(1)

        before_atoms: int = int(np.sum(aa.matrix, dtype=np.int64))
        move_rounds: list[list[movr.Move]] = bc_new.compact(aa)

        _replay_and_check_noiseless_conservation(aa, move_rounds)

        after_atoms: int = int(np.sum(aa.matrix, dtype=np.int64))
        assert after_atoms == before_atoms
        assert np.array_equal(aa.matrix * aa.target, aa.target)

    def test_compact_emits_no_duplicate_destinations_per_round(self) -> None:
        aa: AtomArray = AtomArray(shape=[6, 6], n_species=1)
        aa.target[:, :, 0] = np.uint8(0)
        aa.target[2, 2, 0] = np.uint8(1)
        aa.target[2, 3, 0] = np.uint8(1)
        aa.target[3, 2, 0] = np.uint8(1)
        aa.target[3, 3, 0] = np.uint8(1)

        aa.matrix[:, :, 0] = np.uint8(0)
        aa.matrix[2, 0, 0] = np.uint8(1)
        aa.matrix[2, 1, 0] = np.uint8(1)
        aa.matrix[3, 2, 0] = np.uint8(1)
        aa.matrix[3, 3, 0] = np.uint8(1)

        move_rounds: list[list[movr.Move]] = bc_new.compact(aa)

        for k, round_moves in enumerate(move_rounds):
            srcs: list[tuple[int, int]] = [(m.from_row, m.from_col) for m in round_moves]
            dests: list[tuple[int, int]] = [(m.to_row, m.to_col) for m in round_moves]
            assert len(srcs) == len(set(srcs)), f"Round {k} has duplicate sources: {srcs}"
            assert len(dests) == len(set(dests)), f"Round {k} has duplicate destinations: {dests}"

    def test_compact_prepares_target_when_row_counts_are_already_sufficient(self) -> None:
        aa: AtomArray = AtomArray(shape=[4, 7], n_species=1)
        aa.matrix[:, :, 0] = np.uint8(0)
        aa.target[:, :, 0] = np.uint8(0)

        aa.target[1, 2, 0] = np.uint8(1)
        aa.target[1, 3, 0] = np.uint8(1)
        aa.target[2, 2, 0] = np.uint8(1)
        aa.target[2, 3, 0] = np.uint8(1)

        aa.matrix[1, 0, 0] = np.uint8(1)
        aa.matrix[1, 6, 0] = np.uint8(1)
        aa.matrix[2, 0, 0] = np.uint8(1)
        aa.matrix[2, 6, 0] = np.uint8(1)

        move_rounds: list[list[movr.Move]] = bc_new.compact(aa)

        aa2: AtomArray = AtomArray(shape=[4, 7], n_species=1)
        aa2.matrix = aa.matrix.copy()
        aa2.target = aa.target.copy()
        _replay_and_check_noiseless_conservation(aa2, move_rounds)

        assert np.array_equal(aa2.matrix * aa2.target, aa2.target)

    def test_compact_moves_row_even_if_pivot_column_is_already_occupied(self) -> None:
        aa: AtomArray = AtomArray(shape=[3, 7], n_species=1)
        aa.matrix[:, :, 0] = np.uint8(0)
        aa.target[:, :, 0] = np.uint8(0)

        aa.target[1, 2, 0] = np.uint8(1)
        aa.target[1, 3, 0] = np.uint8(1)
        aa.target[1, 4, 0] = np.uint8(1)

        aa.matrix[1, 0, 0] = np.uint8(1)
        aa.matrix[1, 3, 0] = np.uint8(1)
        aa.matrix[1, 6, 0] = np.uint8(1)

        move_rounds: list[list[movr.Move]] = bc_new.compact(aa)

        aa2: AtomArray = AtomArray(shape=[3, 7], n_species=1)
        aa2.matrix = aa.matrix.copy()
        aa2.target = aa.target.copy()
        _replay_and_check_noiseless_conservation(aa2, move_rounds)

        assert np.array_equal(aa2.matrix * aa2.target, aa2.target)

    def test_compact_single_row_array_prepares_target(self) -> None:
        init_config: np.ndarray
        target_config: np.ndarray
        init_config, target_config = _make_single_row_case([0, 2, 6], [2, 3, 4], 7)

        aa: AtomArray = _make_atom_array(init_config[:, :, 0], target_config[:, :, 0])
        move_rounds: list[list[movr.Move]] = bc_new.compact(aa)

        aa2: AtomArray = _make_atom_array(init_config[:, :, 0], target_config[:, :, 0])
        _replay_and_check_noiseless_conservation(aa2, move_rounds)

        assert np.array_equal(aa2.matrix * aa2.target, aa2.target)
    
    def test_REGRESSION_compact_doesnt_go_into_infinite_loop(self) -> None:
        mat = np.array([[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0],
                        [1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [1, 0, 0, 0, 0]], dtype = np.uint8)
    
        target_2d = np.array([[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1]], dtype=np.uint8)
        
        aa: AtomArray = _make_atom_array(mat, target_2d)
        move_rounds: list[list[movr.Move]] = bc_new.compact(aa)

        aa2: AtomArray = _make_atom_array(mat, target_2d)
        _replay_and_check_noiseless_conservation(aa2, move_rounds)

        assert np.array_equal(aa2.matrix * aa2.target, aa2.target)
    
    def test_REGRESSION_compact_does_not_revisit_states(self) -> None:
        tar = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype = np.uint8)
        
        mat = np.array([[0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
                        [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
                        [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
                        [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
                        [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                        [0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
                        [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
                        [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1],
                        [0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0]], dtype = np.uint8)
        
        aa: AtomArray = _make_atom_array(mat, tar)
        print(bc_new.__file__)
        print(bc_new.compact.__code__.co_firstlineno)
        print("matrix shape/dtype:", aa.matrix.shape, aa.matrix.dtype)
        print("target shape/dtype:", aa.target.shape, aa.target.dtype)
        def stable_digest(arr) -> str:
            return hashlib.sha256(arr.tobytes()).hexdigest()
        print("matrix bytes hash:", hash(aa.matrix.tobytes()))
        print("target bytes hash:", hash(aa.target.tobytes()))
        move_rounds: list[list[movr.Move]] = bc_new.compact(aa)
        

        aa2: AtomArray = _make_atom_array(mat, tar)
        _replay_and_check_noiseless_conservation(aa2, move_rounds)

        assert np.array_equal(aa2.matrix * aa2.target, aa2.target)
    
    def test_REGRESSION_compact_does_not_revisit_states_case2(self) -> None:
        tar = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype = np.uint8)
        
        mat = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                        [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
                        [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
                        [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1],
                        [1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
                        [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                        [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1],
                        [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
                        [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
                        [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0]], dtype = np.uint8)
        
        aa: AtomArray = _make_atom_array(mat, tar)

        def stable_digest(arr) -> str:
            return hashlib.sha256(arr.tobytes()).hexdigest()

        print("=== BCv2 compact debug ===")
        print("compact file:",bc_new. __file__)
        print("first line no:",bc_new.compact.__code__.co_firstlineno)
        print("matrix shape/dtype:", aa.matrix.shape, aa.matrix.dtype)
        print("target shape/dtype:", aa.target.shape, aa.target.dtype)
        print("matrix stable digest:", stable_digest(aa.matrix))
        print("target stable digest:", stable_digest(aa.target))
        print("matrix:", aa.matrix[:, :, 0].astype(int))
        print("target:", aa.target[:, :, 0].astype(int))
        move_rounds: list[list[movr.Move]] = bc_new.compact(aa)
        

        aa2: AtomArray = _make_atom_array(mat, tar)
        _replay_and_check_noiseless_conservation(aa2, move_rounds)

        assert np.array_equal(aa2.matrix * aa2.target, aa2.target)
    
    @pytest.mark.parametrize('stuck_state', [
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
         [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1],
         [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0],
         [0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1]]])
    def test_REGRESSION_compact_succeeds_on_repeated_config(self, stuck_state) -> None:
        mat = np.array(stuck_state, dtype = np.uint8)

        tar = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype = np.uint8)
        
        aa: AtomArray = _make_atom_array(mat, tar)

        def stable_digest(arr) -> str:
            return hashlib.sha256(arr.tobytes()).hexdigest()

        print("=== BCv2 compact debug ===")
        print("compact file:",bc_new. __file__)
        print("first line no:",bc_new.compact.__code__.co_firstlineno)
        print("matrix shape/dtype:", aa.matrix.shape, aa.matrix.dtype)
        print("target shape/dtype:", aa.target.shape, aa.target.dtype)
        print("matrix stable digest:", stable_digest(aa.matrix))
        print("target stable digest:", stable_digest(aa.target))
        print("matrix:", aa.matrix[:, :, 0].astype(int))
        print("target:", aa.target[:, :, 0].astype(int))
        move_rounds: list[list[movr.Move]] = bc_new.compact(aa)
        

        aa2: AtomArray = _make_atom_array(mat, tar)
        _replay_and_check_noiseless_conservation(aa2, move_rounds)

        assert np.array_equal(aa2.matrix * aa2.target, aa2.target)

class TestBCV2:
    def test_returns_failure_when_compact_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        aa: AtomArray = AtomArray(shape=[2, 5], n_species=1)
        aa.matrix[:, :, 0] = np.uint8(0)
        aa.target[:, :, 0] = np.uint8(0)

        aa.target[0, 1, 0] = np.uint8(1)
        aa.target[0, 2, 0] = np.uint8(1)
        aa.target[1, 1, 0] = np.uint8(1)
        aa.target[1, 2, 0] = np.uint8(1)

        aa.matrix[0, 0, 0] = np.uint8(1)
        aa.matrix[0, 4, 0] = np.uint8(1)
        aa.matrix[1, 0, 0] = np.uint8(1)
        aa.matrix[1, 4, 0] = np.uint8(1)

        def _boom(_array: AtomArray) -> list[list[movr.Move]]:
            raise ValueError("forced compact failure")

        monkeypatch.setattr(bc_new, "compact", _boom)

        final_matrix: np.ndarray
        master_moves: list[list[movr.Move]]
        success_flag: bool
        final_matrix, master_moves, success_flag = bc_new.bcv2(aa, do_ejection=False)

        assert success_flag is False
        assert isinstance(master_moves, list)
        assert isinstance(final_matrix, np.ndarray)
    
    def test_bcv2_replay_matches_returned_final_matrix_and_fills_target_mask(self) -> None:
        matrix_2d: np.ndarray = np.zeros((6, 6), dtype=np.uint8)
        target_2d: np.ndarray = np.zeros((6, 6), dtype=np.uint8)

        target_2d[2, 2] = np.uint8(1)
        target_2d[2, 3] = np.uint8(1)
        target_2d[3, 2] = np.uint8(1)
        target_2d[3, 3] = np.uint8(1)

        matrix_2d[0, 0] = np.uint8(1)
        matrix_2d[0, 5] = np.uint8(1)
        matrix_2d[5, 0] = np.uint8(1)
        matrix_2d[5, 5] = np.uint8(1)

        arr: AtomArray = _make_atom_array(matrix_2d, target_2d)
        n0: int = _n_atoms(arr.matrix)

        final_matrix: np.ndarray
        master_moves: list[list[movr.Move]]
        success_flag: bool
        final_matrix, master_moves, success_flag = bc_new.bcv2(arr, do_ejection=False)

        replay_arr: AtomArray = _make_atom_array(matrix_2d, target_2d)
        _replay_and_check_noiseless_conservation(replay_arr, master_moves)

        assert success_flag is True
        assert np.array_equal(replay_arr.matrix, final_matrix)
        assert _n_atoms(final_matrix) == n0
        assert np.array_equal(final_matrix * replay_arr.target, replay_arr.target)

    def test_bcv2_reports_success_and_returns_no_moves_when_already_prepared(self) -> None:
        matrix_2d: np.ndarray = np.array(
            [
                [0, 1, 1, 0],
                [0, 1, 1, 0],
            ],
            dtype=np.uint8,
        )
        target_2d: np.ndarray = matrix_2d.copy()

        arr: AtomArray = _make_atom_array(matrix_2d, target_2d)

        final_matrix: np.ndarray
        master_moves: list[list[movr.Move]]
        success_flag: bool
        final_matrix, master_moves, success_flag = bc_new.bcv2(arr, do_ejection=False)

        assert success_flag is True
        assert master_moves == []
        assert np.array_equal(final_matrix[:, :, 0], target_2d)

    
    def test_bcv2_REGRESSION_20x4_array_fails_on_compact(self):
        matrix_2d = np.array([[0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1],
                              [1, 1, 1, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 1],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [1, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0]], dtype=np.uint8)

        target_2d = np.array([[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1]], dtype=np.uint8)
        
        arr: AtomArray = _make_atom_array(matrix_2d, target_2d)
        n_before: int = _n_atoms(arr.matrix)
        n_rows = 20
        n_cols = 5

        final_matrix: np.ndarray
        master_moves: list[list[movr.Move]]
        success_flag: bool
        final_matrix, master_moves, success_flag = bc_new.bcv2(arr, do_ejection=False)

        replay_arr: AtomArray = _make_atom_array(matrix_2d, target_2d)
        _replay_and_check_noiseless_conservation(replay_arr, master_moves)

        assert np.array_equal(
            replay_arr.matrix,
            final_matrix,
        ), (
            "Returned final_matrix does not match noiseless replay. "
            f"shape=({n_rows}, {n_cols})"
        )

        assert _n_atoms(final_matrix) == n_before, (
            "BCv2 changed the total atom number on a noiseless single-species case. "
            f"shape=({n_rows}, {n_cols})"
        )

        effective_config: np.ndarray = np.multiply(
            final_matrix,
            replay_arr.target.reshape(final_matrix.shape),
        )
        assert np.array_equal(
            effective_config,
            replay_arr.target.reshape(final_matrix.shape),
        ), (
            "BCv2 failed to fill the target mask on a globally feasible case. "
            f"shape=({n_rows}, {n_cols})"
        )

        assert success_flag is True, (
            "BCv2 returned success_flag=False on a globally feasible case whose "
            "noiseless replay fills the target mask. "
            f"shape=({n_rows}, {n_cols})"
        )


class TestBCV2_RANDOM_END_TO_END:
    """
    Randomized end-to-end feasibility/regression tests for single-species BCv2.

    Written contract
    ----------------
    For globally feasible single-species cases (enough atoms to realize the target
    support), BCv2 should either:
    - succeed and produce a final matrix that fills the target mask after noiseless
      replay of the returned move rounds, or
    - expose a concrete failing seed/configuration that can be promoted to a named
      regression test.

    This test is a falsification search, not a proof of universal correctness.
    """
    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("n_rows", "n_cols", "n_trials"),
        [
            (20, 5, 200),
            (30, 6, 200),
            (5, 20, 200),
            (6, 30, 200),
            (40, 8, 200),
            (8, 40, 200),
            (60, 5, 200),
            # (93, 89, 50),
        ],
    )
    def test_random_globally_feasible_cases_fill_target_mask(
        self,
        n_rows: int,
        n_cols: int,
        n_trials: int,
    ) -> None:
        """
        For random globally feasible cases, BCv2 should replay to the returned final
        matrix, conserve atom number, and fill the target support mask.
        """
        rng: np.random.Generator = np.random.default_rng(20260401 + 17 * n_rows + n_cols)

        for trial_index in range(n_trials):
            matrix_2d: np.ndarray
            target_2d: np.ndarray
            matrix_2d, target_2d = _make_random_bcv2_feasible_case(
                rng=rng,
                n_rows=n_rows,
                n_cols=n_cols,
            )

            arr: AtomArray = _make_atom_array(matrix_2d, target_2d)
            n_before: int = _n_atoms(arr.matrix)

            final_matrix: np.ndarray
            master_moves: list[list[movr.Move]]
            success_flag: bool
            try:
                final_matrix, master_moves, success_flag = bc_new.bcv2(arr, do_ejection=False)
            except UnboundLocalError:
                print(f'index is {trial_index}')
                raise UnboundLocalError

            replay_arr: AtomArray = _make_atom_array(matrix_2d, target_2d)
            try:
                _replay_and_check_noiseless_conservation(replay_arr, master_moves)
            except AssertionError:
                print(f'index is {trial_index}')

            assert np.array_equal(
                replay_arr.matrix,
                final_matrix,
            ), (
                "Returned final_matrix does not match noiseless replay. "
                f"shape=({n_rows}, {n_cols}), trial={trial_index}"
            )

            assert _n_atoms(final_matrix) == n_before, (
                "BCv2 changed the total atom number on a noiseless single-species case. "
                f"shape=({n_rows}, {n_cols}), trial={trial_index}"
            )

            effective_config: np.ndarray = np.multiply(
                final_matrix,
                replay_arr.target.reshape(final_matrix.shape),
            )
            assert np.array_equal(
                effective_config,
                replay_arr.target.reshape(final_matrix.shape),
            ), (
                "BCv2 failed to fill the target mask on a globally feasible case. "
                f"shape=({n_rows}, {n_cols}), trial={trial_index}"
            )

            assert success_flag is True, (
                "BCv2 returned success_flag=False on a globally feasible case whose "
                "noiseless replay fills the target mask. "
                f"shape=({n_rows}, {n_cols}), trial={trial_index}"
            )

    def test_failure_seed_template_for_regression_promotion(self) -> None:
        """
        Template reminder for promoting discovered random failures into named regressions.

        Notes
        -----
        If the randomized test above ever finds a failing configuration, copy the
        printed/inspected matrix_2d and target_2d into a dedicated named regression
        test instead of relying on the random seed alone.
        """
        assert True


@pytest.mark.performance
class TestSPECIAL_CASE_ALGO_1D_PERFORMANCE:
    def test_special_case_algo_runtime_smoke(self) -> None:
        rng: np.random.Generator = np.random.default_rng(0)
        n_cols: int = 256
        n_atoms: int = 96

        atom_cols: list[int] = sorted(rng.choice(n_cols, size=n_atoms, replace=False).tolist())
        target_cols: list[int] = sorted(rng.choice(n_cols, size=n_atoms, replace=False).tolist())
        init_config: np.ndarray
        target_config: np.ndarray
        init_config, target_config = _make_single_row_case(atom_cols, target_cols, n_cols)

        t_new: float = _time_function(
            bc_new.special_case_algo_1d,
            init_config,
            target_config,
            repeat=7,
        )

        assert t_new > 0.0


@pytest.mark.performance
class TestMIDDLE_FILL_ALGO_1D_PERFORMANCE:
    def test_middle_fill_algo_runtime_smoke(self) -> None:
        rng: np.random.Generator = np.random.default_rng(1)
        n_cols: int = 256
        n_atoms: int = 128
        n_targets: int = 80

        atom_cols: list[int] = sorted(rng.choice(n_cols, size=n_atoms, replace=False).tolist())
        target_cols: list[int] = sorted(rng.choice(n_cols, size=n_targets, replace=False).tolist())
        init_config: np.ndarray
        target_config: np.ndarray
        init_config, target_config = _make_single_row_case(atom_cols, target_cols, n_cols)

        t_new: float = _time_function(
            bc_new.middle_fill_algo_1d,
            init_config,
            target_config,
            repeat=7,
        )

        assert t_new > 0.0


@pytest.mark.performance
class TestCOMPACT_PERFORMANCE:
    @staticmethod
    def _make_case(
        rng: np.random.Generator,
        n_rows: int,
        n_cols: int,
        target_width: int,
        extra_atoms: int,
    ) -> AtomArray:
        """
        Build a row-wise feasible compact case.

        Why this exists
        ---------------
        `compact()` is horizontal-only, so every target row must already contain at
        least as many atoms as the number of target sites in that row. This helper
        enforces that precondition while still placing atoms away from the target
        columns so compact has real work to do.
        """
        target_2d: np.ndarray = np.zeros((n_rows, n_cols), dtype=np.uint8)
        target_start: int = int(rng.integers(0, n_cols - target_width + 1))
        target_cols: np.ndarray = np.arange(target_start, target_start + target_width)
        target_2d[:, target_start : target_start + target_width] = np.uint8(1)

        matrix_2d: np.ndarray = np.zeros((n_rows, n_cols), dtype=np.uint8)

        # Distribute the extra atoms across rows, but always guarantee that each row
        # has at least `target_width` atoms so compact's row-wise precondition holds.
        extra_per_row: np.ndarray = np.zeros(n_rows, dtype=np.int64)
        for _ in range(extra_atoms):
            row_idx: int = int(rng.integers(0, n_rows))
            if target_width + int(extra_per_row[row_idx]) < n_cols:
                extra_per_row[row_idx] += 1

        for row in range(n_rows):
            n_row_atoms: int = target_width + int(extra_per_row[row])

            # Prefer non-target columns first so the row is feasible but not already prepared.
            non_target_cols: np.ndarray = np.setdiff1d(
                np.arange(n_cols, dtype=np.int64),
                target_cols,
                assume_unique=True,
            )

            chosen: np.ndarray
            if n_row_atoms <= non_target_cols.size:
                chosen = rng.choice(non_target_cols, size=n_row_atoms, replace=False)
            else:
                chosen_non_target: np.ndarray = non_target_cols.copy()
                remaining: int = n_row_atoms - chosen_non_target.size
                chosen_target: np.ndarray = rng.choice(
                    target_cols,
                    size=remaining,
                    replace=False,
                )
                chosen = np.concatenate([chosen_non_target, chosen_target])

            matrix_2d[row, chosen] = np.uint8(1)

        return _make_atom_array(matrix_2d, target_2d)

    def test_compact_runtime_smoke(self) -> None:
        rng: np.random.Generator = np.random.default_rng(2)
        cases: list[AtomArray] = [
            self._make_case(rng, 8, 32, 12, 10),
            self._make_case(rng, 10, 40, 15, 12),
            self._make_case(rng, 12, 48, 18, 16),
        ]

        def _run_new(case: AtomArray) -> list[list[movr.Move]]:
            arr: AtomArray = _make_atom_array(case.matrix[:, :, 0], case.target[:, :, 0])
            return bc_new.compact(arr)

        total_new: float = 0.0
        for case in cases:
            total_new += _time_function(_run_new, case, repeat=3)

        assert total_new > 0.0


@pytest.mark.performance
class TestBCV2_PERFORMANCE:
    def test_bcv2_scales_reasonably_on_medium_instances(self) -> None:
        rng: np.random.Generator = np.random.default_rng(3)

        def _make_bcv2_case(n_rows: int, n_cols: int, target_width: int) -> AtomArray:
            target_2d: np.ndarray = np.zeros((n_rows, n_cols), dtype=np.uint8)
            target_start: int = (n_cols - target_width) // 2
            target_2d[:, target_start : target_start + target_width] = np.uint8(1)

            n_targets: int = int(np.sum(target_2d, dtype=np.int64))
            n_atoms: int = min(n_rows * n_cols, n_targets + n_rows)

            occ: np.ndarray = np.zeros(n_rows * n_cols, dtype=np.uint8)
            occ[rng.choice(occ.size, size=n_atoms, replace=False)] = np.uint8(1)
            matrix_2d: np.ndarray = occ.reshape(n_rows, n_cols)

            return _make_atom_array(matrix_2d, target_2d)

        case_small: AtomArray = _make_bcv2_case(8, 24, 10)
        case_large: AtomArray = _make_bcv2_case(12, 36, 14)

        t_small: float = _time_function(bc_new.bcv2, case_small, False, repeat=3)
        t_large: float = _time_function(bc_new.bcv2, case_large, False, repeat=3)

        assert t_small > 0.0
        assert t_large > 0.0
        assert t_large >= t_small


class TestPREBALANCE_ADDITIONAL_CONTRACTS:
    def test_returns_success_with_no_moves_when_target_band_already_sufficient(self) -> None:
        """A sufficiently populated target-row band should short-circuit immediately."""
        init: np.ndarray = np.zeros((5, 6, 1), dtype=np.uint8)
        targ: np.ndarray = np.zeros((5, 6, 1), dtype=np.uint8)

        targ[1, 2, 0] = np.uint8(1)
        targ[2, 1, 0] = np.uint8(1)
        targ[2, 3, 0] = np.uint8(1)
        targ[3, 2, 0] = np.uint8(1)

        init[1, 0, 0] = np.uint8(1)
        init[2, 0, 0] = np.uint8(1)
        init[2, 5, 0] = np.uint8(1)
        init[3, 5, 0] = np.uint8(1)
        init[0, 1, 0] = np.uint8(1)
        init[4, 4, 0] = np.uint8(1)

        moves, ok = bc_new.prebalance(init, targ)
        assert moves == []
        assert ok is True

    def test_only_above_feasible_case_succeeds(self) -> None:
        """Prebalance should be able to fill the band using atoms sourced only from above."""
        init: np.ndarray = np.zeros((6, 6, 1), dtype=np.uint8)
        targ: np.ndarray = np.zeros((6, 6, 1), dtype=np.uint8)

        targ[2, 1, 0] = np.uint8(1)
        targ[2, 2, 0] = np.uint8(1)
        targ[3, 2, 0] = np.uint8(1)
        targ[3, 3, 0] = np.uint8(1)

        init[0, 0, 0] = np.uint8(1)
        init[0, 2, 0] = np.uint8(1)
        init[1, 1, 0] = np.uint8(1)
        init[1, 4, 0] = np.uint8(1)

        moves, ok = bc_new.prebalance(init, targ)
        assert ok is True

        aa: AtomArray = AtomArray(shape=[6, 6], n_species=1)
        aa.matrix = init.copy()
        aa.target = targ.copy()
        _replay_and_check_noiseless_conservation(aa, moves)

        assert bc_new._int_sum(aa.matrix[2:4, :, :]) >= bc_new._int_sum(targ[2:4, :, :])

    def test_only_below_feasible_case_succeeds(self) -> None:
        """Prebalance should be able to fill the band using atoms sourced only from below."""
        init: np.ndarray = np.zeros((6, 6, 1), dtype=np.uint8)
        targ: np.ndarray = np.zeros((6, 6, 1), dtype=np.uint8)

        targ[2, 1, 0] = np.uint8(1)
        targ[2, 2, 0] = np.uint8(1)
        targ[3, 2, 0] = np.uint8(1)
        targ[3, 3, 0] = np.uint8(1)

        init[4, 0, 0] = np.uint8(1)
        init[4, 2, 0] = np.uint8(1)
        init[5, 1, 0] = np.uint8(1)
        init[5, 4, 0] = np.uint8(1)

        moves, ok = bc_new.prebalance(init, targ)
        assert ok is True

        aa: AtomArray = AtomArray(shape=[6, 6], n_species=1)
        aa.matrix = init.copy()
        aa.target = targ.copy()
        _replay_and_check_noiseless_conservation(aa, moves)

        assert bc_new._int_sum(aa.matrix[2:4, :, :]) >= bc_new._int_sum(targ[2:4, :, :])

    def test_stops_once_band_sufficiency_is_reached(self) -> None:
        """
        Prebalance should not spend extra rounds after the target-row band first becomes sufficient.

        Opportunistic oversupply within the final successful round is allowed.
        """
        init: np.ndarray = np.zeros((5, 5, 1), dtype=np.uint8)
        targ: np.ndarray = np.zeros((5, 5, 1), dtype=np.uint8)

        targ[2, 1, 0] = np.uint8(1)
        targ[2, 2, 0] = np.uint8(1)
        targ[2, 3, 0] = np.uint8(1)

        init[1, 0, 0] = np.uint8(1)
        init[1, 2, 0] = np.uint8(1)
        init[3, 4, 0] = np.uint8(1)

        moves, ok = bc_new.prebalance(init, targ)
        assert ok is True

        aa: AtomArray = AtomArray(shape=[5, 5], n_species=1)
        aa.matrix = init.copy()
        aa.target = targ.copy()

        band_counts: list[int] = [bc_new._int_sum(aa.matrix[2:3, :, :])]
        target_count: int = bc_new._int_sum(targ[2:3, :, :])

        for round_moves in moves:
            _replay_and_check_noiseless_conservation(aa, [round_moves])
            band_counts.append(bc_new._int_sum(aa.matrix[2:3, :, :]))

        hit_index: int | None = None
        for idx, count in enumerate(band_counts):
            if count >= target_count:
                hit_index = idx
                break

        assert hit_index is not None
        assert len(moves) == hit_index


class TestBALANCE_ROWS_ADDITIONAL_CONTRACTS:
    def test_moves_exactly_requested_net_transfer_on_small_case(self) -> None:
        """
        balance_rows should remain an exact balancing primitive.

        When one half is deficient by exactly one atom and the other has exactly one
        movable surplus, the routine should realize that exact net transfer.
        """
        init: np.ndarray = np.zeros((4, 5, 1), dtype=np.uint8)
        targ: np.ndarray = np.zeros((4, 5, 1), dtype=np.uint8)

        targ[0, 0, 0] = np.uint8(1)
        targ[1, 1, 0] = np.uint8(1)
        targ[1, 3, 0] = np.uint8(1)
        targ[2, 2, 0] = np.uint8(1)

        init[0, 0, 0] = np.uint8(1)
        init[1, 4, 0] = np.uint8(1)
        init[2, 2, 0] = np.uint8(1)
        init[3, 1, 0] = np.uint8(1)

        before_top: int = bc_new._int_sum(init[0:2, :, :])
        before_bot: int = bc_new._int_sum(init[2:4, :, :])

        move_rounds: list[list[movr.Move]] = bc_new.balance_rows(init, targ, 0, 3)

        aa: AtomArray = AtomArray(shape=[4, 5], n_species=1)
        aa.matrix = init.copy()
        aa.target = targ.copy()
        _replay_and_check_noiseless_conservation(aa, move_rounds)

        after_top: int = bc_new._int_sum(aa.matrix[0:2, :, :])
        after_bot: int = bc_new._int_sum(aa.matrix[2:4, :, :])

        assert after_top - before_top == 1
        assert after_bot - before_bot == -1

    def test_moves_exactly_requested_net_transfer_on_mirrored_small_case(self) -> None:
        """Mirror-direction version of the exact-transfer contract."""
        init: np.ndarray = np.zeros((4, 5, 1), dtype=np.uint8)
        targ: np.ndarray = np.zeros((4, 5, 1), dtype=np.uint8)

        targ[0, 0, 0] = np.uint8(1)
        targ[2, 1, 0] = np.uint8(1)
        targ[2, 3, 0] = np.uint8(1)
        targ[3, 2, 0] = np.uint8(1)

        init[0, 0, 0] = np.uint8(1)
        init[1, 4, 0] = np.uint8(1)
        init[2, 2, 0] = np.uint8(1)
        init[3, 1, 0] = np.uint8(1)

        before_top: int = bc_new._int_sum(init[0:2, :, :])
        before_bot: int = bc_new._int_sum(init[2:4, :, :])

        move_rounds: list[list[movr.Move]] = bc_new.balance_rows(init, targ, 0, 3)

        aa: AtomArray = AtomArray(shape=[4, 5], n_species=1)
        aa.matrix = init.copy()
        aa.target = targ.copy()
        _replay_and_check_noiseless_conservation(aa, move_rounds)

        after_top: int = bc_new._int_sum(aa.matrix[0:2, :, :])
        after_bot: int = bc_new._int_sum(aa.matrix[2:4, :, :])

        assert after_top - before_top == -1
        assert after_bot - before_bot == 1

    def test_moves_stay_within_active_interval(self) -> None:
        """All balance_rows moves should remain inside the active row interval [i, j]."""
        init: np.ndarray = np.zeros((6, 5, 1), dtype=np.uint8)
        targ: np.ndarray = np.zeros((6, 5, 1), dtype=np.uint8)

        targ[1, 0, 0] = np.uint8(1)
        targ[2, 1, 0] = np.uint8(1)
        targ[3, 2, 0] = np.uint8(1)
        targ[4, 3, 0] = np.uint8(1)

        init[1, 4, 0] = np.uint8(1)
        init[3, 0, 0] = np.uint8(1)
        init[4, 1, 0] = np.uint8(1)
        init[4, 4, 0] = np.uint8(1)

        move_rounds: list[list[movr.Move]] = bc_new.balance_rows(init, targ, 1, 4)

        for round_moves in move_rounds:
            for move in round_moves:
                assert 1 <= int(move.from_row) <= 4
                assert 1 <= int(move.to_row) <= 4

    def test_raises_runtimeerror_when_transfer_decomposition_stalls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Globally sufficient atom count but decomposition-limited routing should raise RuntimeError.
        """
        init: np.ndarray = np.zeros((4, 5, 1), dtype=np.uint8)
        targ: np.ndarray = np.zeros((4, 5, 1), dtype=np.uint8)

        targ[0, 0, 0] = np.uint8(1)
        targ[1, 1, 0] = np.uint8(1)
        targ[1, 3, 0] = np.uint8(1)
        targ[2, 2, 0] = np.uint8(1)

        init[0, 0, 0] = np.uint8(1)
        init[1, 4, 0] = np.uint8(1)
        init[2, 2, 0] = np.uint8(1)
        init[3, 1, 0] = np.uint8(1)

        def _stall(*args, **kwargs):
            state_in: np.ndarray = args[0] if args else kwargs["state"]
            return (
                state_in.copy(),
                [],
                0,
                {
                    "kind": "cannot_solve_within_constraints",
                    "transferred": 0,
                    "remaining_R": 1,
                    "n_rounds": 0,
                    "last_bottleneck": "cut_capacity",
                    "source_status_kind": None,
                    "cut_status_kind": "cannot_solve_within_constraints",
                    "C": 0.3,
                },
            )

        monkeypatch.setattr(bc_new, "move_across_rows", _stall)

        with pytest.raises(RuntimeError, match="move_across_rows failed to complete transfer"):
            bc_new.balance_rows(init, targ, 0, 3)


class TestBALANCE_PHASE_INTERFACE_ADDITIONAL_CONTRACTS:
    def test_recursive_balance_handles_small_exact_transfer_case(self) -> None:
        """
        Integration seam: prebalance + recursive balance should still handle a small
        exact transfer case where the balancing primitive must not under-transfer.
        """
        init: np.ndarray = np.zeros((4, 5, 1), dtype=np.uint8)
        targ: np.ndarray = np.zeros((4, 5, 1), dtype=np.uint8)

        targ[0, 0, 0] = np.uint8(1)
        targ[1, 1, 0] = np.uint8(1)
        targ[1, 3, 0] = np.uint8(1)
        targ[2, 2, 0] = np.uint8(1)

        init[0, 0, 0] = np.uint8(1)
        init[1, 4, 0] = np.uint8(1)
        init[2, 2, 0] = np.uint8(1)
        init[3, 1, 0] = np.uint8(1)

        pre_moves, ok = bc_new.prebalance(init, targ)
        assert ok is True

        aa: AtomArray = AtomArray(shape=[4, 5], n_species=1)
        aa.matrix = init.copy()
        aa.target = targ.copy()
        _replay_and_check_noiseless_conservation(aa, pre_moves)

        assignments: list[tuple[int, int]] = bc_new.get_all_balance_assignments(0, 3)
        for i, j in assignments:
            move_rounds: list[list[movr.Move]] = bc_new.balance_rows(
                aa.matrix,
                aa.target,
                i,
                j,
            )
            if move_rounds:
                _replay_and_check_noiseless_conservation(aa, move_rounds)

        for row in range(0, 4):
            row_atoms: int = int(np.sum(aa.matrix[row, :, :], dtype=np.int64))
            row_req: int = int(np.sum(targ[row, :, :], dtype=np.int64))
            assert row_atoms >= row_req
