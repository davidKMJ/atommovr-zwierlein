import itertools

import numpy as np
import pytest

from atommovr.utils.Move import Move
from atommovr.utils.move_utils import move_atoms_noiseless
from atommovr.algorithms.source.bc_controller_helpers import (
    direct_cut_capacity,
    ensure_source_supply,
    compress_move_rounds_conservative,
    get_all_moves_btwn_rows_cols,
    perform_transfer,
    source_supply_at_boundary,
    ensure_boundary_row_cut_capacity_horizontal,
    ensure_boundary_row_cut_capacity_vertical,
    ensure_cut_capacity,
    move_across_rows,
    _plan_horizontal_rounds_to_target,
)

import atommovr.algorithms.source.bc_controller_helpers as helpers

"""
Behavioral-contract tests for BCv2 boundary-transfer controller helpers.

These tests lock the intended semantics of the first controller primitives before
higher-level integration work begins.

Contract summary
----------------
1. `source_supply_at_boundary(state, boundary_src_row)` returns the exact number
   of atoms currently present on the boundary source row. This is the immediate
   source-side metric `S`; it does not look deeper into the source region and
   does not encode transfer accessibility.

2. `direct_cut_capacity(state, boundary_src_row, boundary_dst_row)` returns the
   exact immediate geometric transfer capacity `T` across the cut under the
   local rule `|src_col - dst_col| <= 1`. This is a true maximum-cardinality
   matching quantity, not a reachable-vacancy count and not a planner-specific
   approximation.

3. `get_all_moves_btwn_rows_cols(...)` is a planning helper, not the capacity
   metric itself.
   - If `n_transfer_needed == 0`, it returns the full exact direct-transfer
     plan.
   - If `n_transfer_needed > 0`, it returns a bounded direct-transfer plan of
     size `min(n_transfer_needed, T)`.
   - If some pure mode (same-column only, all-left, or all-right) already
     contains enough legal transfers to satisfy the bounded request, the planner
     must use that pure-mode path rather than paying for the exact mixed-mode
     matcher.
   - When truncation is required, the returned subset is chosen
     deterministically by destination-centrality priority, avoiding the old
     implicit left-prefix bias.

4. `perform_transfer(...)` executes only the currently available direct
   cross-cut transfers. It does not source atoms from deeper rows, clear
   destination space, or iterate internally. Its execution count is therefore

       n_moved = min(remaining, T)

   using the currently available direct plan.

5. `ensure_source_supply(...)` is a vertical source-side relay routine.
   - It may touch only rows in the inclusive interval between
     `boundary_src_row` and `search_limit_row`.
   - It must not move atoms across the cut.
   - In `fill_mode="exact"`, it should not intentionally overfill the boundary
     row beyond the requested increase `delta_S`.
   - In `fill_mode="opportunistic"`, it may realize a larger boundary-supply
     increase than `delta_S`, but must use the same number of internal support
     rounds as the exact-mode relay plan for that request.
   - It returns structured status information (`success`, `partial_blocked`,
     `insufficient_atoms`) rather than silently degrading its objective.
"""


def _make_state(rows: list[list[int]]) -> np.ndarray:
    """Construct a single-species BCv2 state with shape ``(rows, cols, 1)``."""
    arr_2d: np.ndarray = np.asarray(rows, dtype=np.uint8)
    return arr_2d[:, :, np.newaxis]


def _apply_planned_row_transfer(
    state: np.ndarray,
    boundary_src_row: int,
    boundary_dst_row: int,
    from_cols: np.ndarray,
    to_cols: np.ndarray,
) -> np.ndarray:
    """Apply a row-transfer plan directly to a copied state for test verification."""
    new_state: np.ndarray = state.copy()
    src_col: int
    dst_col: int
    for src_col, dst_col in zip(from_cols.tolist(), to_cols.tolist(), strict=True):
        new_state[boundary_src_row, src_col, 0] = np.uint8(0)
        new_state[boundary_dst_row, dst_col, 0] = np.uint8(1)
    return new_state


def _assert_plan_is_legal(
    state: np.ndarray,
    boundary_src_row: int,
    boundary_dst_row: int,
    from_cols: np.ndarray,
    to_cols: np.ndarray,
) -> None:
    """Assert that a returned row-local transfer plan is physically admissible."""
    assert len(from_cols) == len(to_cols)
    assert len(set(from_cols.tolist())) == len(from_cols)
    assert len(set(to_cols.tolist())) == len(to_cols)

    src_col: int
    dst_col: int
    for src_col, dst_col in zip(from_cols.tolist(), to_cols.tolist(), strict=True):
        assert int(state[boundary_src_row, src_col, 0]) == 1
        assert int(state[boundary_dst_row, dst_col, 0]) == 0
        assert abs(src_col - dst_col) <= 1


def _bruteforce_direct_cut_capacity(
    source_row: np.ndarray,
    destination_row: np.ndarray,
) -> int:
    """Compute exact direct-cut capacity by brute force on a single row pair."""
    src_cols: list[int] = np.flatnonzero(source_row).astype(int).tolist()
    dst_vac_cols: list[int] = np.flatnonzero(destination_row == 0).astype(int).tolist()

    edges: list[tuple[int, int]] = []
    src_col: int
    dst_col: int
    for src_col in src_cols:
        for dst_col in dst_vac_cols:
            if abs(src_col - dst_col) <= 1:
                edges.append((src_col, dst_col))

    best: int = 0
    n_edges: int = len(edges)
    subset_size: int
    for subset_size in range(n_edges + 1):
        subset: tuple[tuple[int, int], ...]
        for subset in itertools.combinations(edges, subset_size):
            used_src: set[int] = set()
            used_dst: set[int] = set()
            valid: bool = True
            edge_src: int
            edge_dst: int
            for edge_src, edge_dst in subset:
                if edge_src in used_src or edge_dst in used_dst:
                    valid = False
                    break
                used_src.add(edge_src)
                used_dst.add(edge_dst)
            if valid:
                best = max(best, len(subset))

    return best


def _test_pure_mode_matches(
    source_row: np.ndarray,
    destination_row: np.ndarray,
    shift: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return the pure-mode row-local transfer plan for test-side oracle checks."""
    free_mask: np.ndarray = destination_row == 0
    n_cols: int = int(source_row.size)

    if shift == 0:
        match_mask: np.ndarray = source_row.astype(bool) & free_mask
        from_cols: np.ndarray = np.flatnonzero(match_mask).astype(np.intp, copy=False)
        to_cols: np.ndarray = from_cols.copy()
        return from_cols, to_cols, int(from_cols.size)

    if shift == -1:
        if n_cols <= 1:
            empty: np.ndarray = np.zeros(0, dtype=np.intp)
            return empty, empty, 0
        match_mask = source_row[1:].astype(bool) & free_mask[:-1]
        src_local: np.ndarray = np.flatnonzero(match_mask).astype(np.intp, copy=False)
        from_cols = src_local + 1
        to_cols = src_local
        return from_cols, to_cols, int(from_cols.size)

    if shift == 1:
        if n_cols <= 1:
            empty = np.zeros(0, dtype=np.intp)
            return empty, empty, 0
        match_mask = source_row[:-1].astype(bool) & free_mask[1:]
        src_local = np.flatnonzero(match_mask).astype(np.intp, copy=False)
        from_cols = src_local
        to_cols = src_local + 1
        return from_cols, to_cols, int(from_cols.size)

    raise ValueError(f"shift must be -1, 0, or 1; got {shift}")


def _test_vacancy_center_twice(destination_row: np.ndarray) -> int:
    """Return twice the arithmetic mean of destination vacancy columns."""
    vacancy_cols: np.ndarray = np.flatnonzero(destination_row == 0).astype(np.intp, copy=False)
    if vacancy_cols.size == 0:
        raise ValueError("destination_row must contain at least one vacancy.")
    return int(2 * np.sum(vacancy_cols, dtype=np.int64) // int(vacancy_cols.size))


def _test_truncate_by_destination_centrality(
    from_cols: np.ndarray,
    to_cols: np.ndarray,
    destination_row: np.ndarray,
    n_keep: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply the deterministic centrality truncation rule on the test side."""
    if n_keep <= 0 or from_cols.size == 0:
        empty: np.ndarray = np.zeros(0, dtype=np.intp)
        return empty, empty, 0
    if n_keep >= int(from_cols.size):
        return from_cols, to_cols, int(from_cols.size)

    center_twice: int = _test_vacancy_center_twice(destination_row)
    order: list[int] = list(range(int(from_cols.size)))
    order.sort(
        key=lambda idx: (
            abs(2 * int(to_cols[idx]) - center_twice),
            abs(int(to_cols[idx]) - int(from_cols[idx])),
            int(to_cols[idx]),
            int(from_cols[idx]),
        )
    )
    keep_idx: np.ndarray = np.asarray(order[:n_keep], dtype=np.intp)
    return from_cols[keep_idx], to_cols[keep_idx], int(keep_idx.size)


def _test_best_pure_bounded_plan(
    source_row: np.ndarray,
    destination_row: np.ndarray,
    n_needed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return the expected bounded pure-mode plan under the controller contract."""
    center_twice: int = _test_vacancy_center_twice(destination_row)
    candidates: list[tuple[tuple[int, int, int], np.ndarray, np.ndarray]] = []

    mode_rank: int
    shift: int
    for mode_rank, shift in enumerate((0, -1, 1)):
        pure_from: np.ndarray
        pure_to: np.ndarray
        pure_n: int
        pure_from, pure_to, pure_n = _test_pure_mode_matches(
            source_row=source_row,
            destination_row=destination_row,
            shift=shift,
        )
        if pure_n < n_needed:
            continue

        chosen_from: np.ndarray
        chosen_to: np.ndarray
        n_chosen: int
        chosen_from, chosen_to, n_chosen = _test_truncate_by_destination_centrality(
            from_cols=pure_from,
            to_cols=pure_to,
            destination_row=destination_row,
            n_keep=n_needed,
        )
        assert n_chosen == n_needed

        score: tuple[int, int, int] = (
            int(np.sum(np.abs(2 * chosen_to.astype(np.int64) - center_twice), dtype=np.int64)),
            int(np.sum(np.abs(chosen_to.astype(np.int64) - chosen_from.astype(np.int64)), dtype=np.int64)),
            mode_rank,
        )
        candidates.append((score, chosen_from, chosen_to))

    if len(candidates) == 0:
        empty: np.ndarray = np.zeros(0, dtype=np.intp)
        return empty, empty, 0

    candidates.sort(key=lambda item: item[0])
    best_from: np.ndarray = candidates[0][1]
    best_to: np.ndarray = candidates[0][2]
    return best_from, best_to, int(best_from.size)

class TestCOMPRESS_MOVE_ROUNDS_CONSERVATIVE:
    def test_returns_empty_for_empty_schedule(self) -> None:
        """An empty schedule should remain empty."""
        state: np.ndarray = _make_state_from_rows(
            [0, 1, 0],
            [1, 0, 1],
        )

        compressed: list[list[Move]] = compress_move_rounds_conservative(
            state=state,
            move_rounds=[],
        )

        assert compressed == []

    def test_parallelizes_vertical_hole_propagation(self) -> None:
        """
        Consecutive vacancy-propagation moves along one column should merge when
        simultaneous replay gives the same state as the original sequential
        schedule.
        """
        state: np.ndarray = _make_state_from_rows(
            [1],
            [1],
            [1],
            [0],
        )
        move_rounds: list[list[Move]] = [
            [Move(2, 0, 3, 0)],
            [Move(1, 0, 2, 0)],
            [Move(0, 0, 1, 0)],
        ]

        compressed: list[list[Move]] = compress_move_rounds_conservative(
            state=state,
            move_rounds=move_rounds,
        )

        assert len(compressed) == 1
        assert len(compressed[0]) == 3

        original_final: np.ndarray = _replay_rounds(state=state, move_rounds=move_rounds)
        compressed_final: np.ndarray = _replay_rounds(
            state=state,
            move_rounds=compressed,
        )
        np.testing.assert_array_equal(compressed_final, original_final)

    def test_does_not_merge_atom_transport_across_empty_rows(self) -> None:
        """
        A single atom moving through empty rows should remain sequential because
        simultaneous replay is not equivalent to repeated one-step transport.
        """
        state: np.ndarray = _make_state_from_rows(
            [1],
            [0],
            [0],
            [0],
        )
        move_rounds: list[list[Move]] = [
            [Move(0, 0, 1, 0)],
            [Move(1, 0, 2, 0)],
            [Move(2, 0, 3, 0)],
        ]

        compressed: list[list[Move]] = compress_move_rounds_conservative(
            state=state,
            move_rounds=move_rounds,
        )

        assert compressed == move_rounds

        original_final: np.ndarray = _replay_rounds(state=state, move_rounds=move_rounds)
        compressed_final: np.ndarray = _replay_rounds(
            state=state,
            move_rounds=compressed,
        )
        np.testing.assert_array_equal(compressed_final, original_final)

    def test_merges_only_safe_prefix_of_schedule(self) -> None:
        """
        The compressor should merge an initial replay-equivalent vacancy wave
        without forcing later unrelated rounds into the same batch.
        """
        state: np.ndarray = _make_state_from_rows(
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
        )
        move_rounds: list[list[Move]] = [
            [Move(2, 0, 3, 0)],
            [Move(1, 0, 2, 0)],
            [Move(0, 0, 1, 0)],
            [Move(3, 1, 2, 1)],
        ]

        compressed: list[list[Move]] = compress_move_rounds_conservative(
            state=state,
            move_rounds=move_rounds,
        )

        assert len(compressed) == 2
        assert len(compressed[0]) == 3
        assert compressed[1] == move_rounds[3]

        original_final: np.ndarray = _replay_rounds(state=state, move_rounds=move_rounds)
        compressed_final: np.ndarray = _replay_rounds(
            state=state,
            move_rounds=compressed,
        )
        np.testing.assert_array_equal(compressed_final, original_final)

    def test_does_not_merge_unparallelizable_moves(self) -> None:
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 0],
        )

        move_rounds: list[list[Move]] = [
            [Move(1, 1, 0, 1)],
            [Move(2, 1, 1, 1)],
            [Move(3, 1, 2, 1)],
            [Move(3, 2, 3, 1), Move(3, 3, 3, 2)],
        ]

        compressed: list[list[Move]] = compress_move_rounds_conservative(
            state=state,
            move_rounds=move_rounds,
        )

        assert len(compressed) == 2
        assert len(compressed[0]) == 3
        assert len(compressed[1]) == 2
        assert compressed[1] == move_rounds[3]

        original_final: np.ndarray = _replay_rounds(state=state, move_rounds=move_rounds)
        compressed_final: np.ndarray = _replay_rounds(
            state=state,
            move_rounds=compressed,
        )
        np.testing.assert_array_equal(compressed_final, original_final)


class TestSOURCE_SUPPLY_AT_BOUNDARY:
    def test_counts_atoms_on_boundary_row(self) -> None:
        state: np.ndarray = _make_state(
            [
                [0, 1, 0, 1],
                [1, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        )

        assert source_supply_at_boundary(state, 0) == 2
        assert source_supply_at_boundary(state, 1) == 2
        assert source_supply_at_boundary(state, 2) == 1

    def test_raises_for_row_out_of_bounds(self) -> None:
        state: np.ndarray = _make_state([[1, 0], [0, 1]])

        with pytest.raises(IndexError):
            source_supply_at_boundary(state, -1)

        with pytest.raises(IndexError):
            source_supply_at_boundary(state, 3)

    def test_raises_for_wrong_shape(self) -> None:
        state_2d: np.ndarray = np.asarray([[1, 0], [0, 1]], dtype=np.uint8)

        with pytest.raises(ValueError):
            source_supply_at_boundary(state_2d, 0)

    @pytest.mark.parametrize("moves", [
        [[Move(2, 2, 2, 1)], [Move(2, 3, 2, 2)]],
        [[Move(2, 2, 2, 1)], [Move(0, 2, 0, 1)]],
        [[Move(2, 2, 2, 1)], [Move(0, 2, 0, 1)],[Move(2, 3, 2, 2)]],
        [[Move(2, 2, 2, 1)], [Move(0, 2, 0, 1), Move(2, 3, 2, 2)]],
        [[Move(2, 2, 2, 1), Move(2, 3, 2, 2)], [Move(0, 2, 0, 1)]],
        [[Move(1, 1, 0, 1)], [Move(2, 4, 1, 4)]],
        [[Move(1, 1, 0, 1)], [Move(1, 3, 0, 3)], [Move(2, 3, 1, 3)]],
        [[Move(1, 1, 0, 1)], [Move(2, 3, 1, 3), Move(1, 3, 0, 3)]],
        [[Move(1, 1, 0, 1)], [Move(1, 3, 0, 3), Move(2, 3, 1, 3)]],
        [[Move(2, 3, 1, 3), Move(1, 3, 0, 3)], [Move(1, 1, 0, 1)]],
                                       ])
    def test_mergeable_moves_are_merged_into_single_round(self, moves):
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
        )
        com_moves = compress_move_rounds_conservative(state, moves)
        assert len(com_moves) == 1
    
    @pytest.mark.parametrize("moves", [
        [[Move(1, 1, 0, 1)], [Move(2, 3, 1, 3)]],
        [[Move(0, 2, 0, 1)], [Move(2, 4, 1, 4)]],
        [[Move(1, 1, 0, 1)], [Move(2, 3, 1, 3)], [Move(1, 3, 0, 3)]],
        [[Move(2, 2, 2, 1)], [Move(1, 3, 1, 4)]],
                                       ])
    def test_nonmergeable_moves_are_not_merged(self, moves):
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
        )
        com_moves = compress_move_rounds_conservative(state, moves)
        assert len(com_moves) == len(moves)


class TestDIRECT_CUT_CAPACITY:
    def test_returns_zero_when_no_source_atoms(self) -> None:
        state: np.ndarray = _make_state(
            [
                [0, 0, 0],
                [1, 0, 1],
            ]
        )

        assert direct_cut_capacity(state, 0, 1) == 0

    def test_returns_zero_when_no_destination_vacancies(self) -> None:
        state: np.ndarray = _make_state(
            [
                [1, 0, 1],
                [1, 1, 1],
            ]
        )

        assert direct_cut_capacity(state, 0, 1) == 0

    @pytest.mark.parametrize(
        ("source_row", "destination_row", "expected"),
        [
            ([1, 0], [1, 0], 1),
            ([0, 1], [0, 1], 1),
            ([1, 1, 0], [1, 0, 1], 1),
            ([1, 0, 1], [0, 0, 0], 2),
            ([1, 0, 0, 1], [1, 0, 0, 1], 2),
            ([1, 0, 1, 1], [0, 0, 1, 0], 3),
            ([0, 0, 0, 1, 1], [0, 0, 1, 1, 1], 0),
            ([1, 0, 1, 1], [0, 1, 1, 0], 2),
            ([1, 1, 1], [1, 1, 1], 0),
            ([0, 0, 0], [0, 0, 0], 0),
        ],
    )
    def test_handbuilt_edge_cases(
        self,
        source_row: list[int],
        destination_row: list[int],
        expected: int,
    ) -> None:
        state: np.ndarray = _make_state([source_row, destination_row])
        assert direct_cut_capacity(state, 0, 1) == expected

    def test_matches_bruteforce_oracle_on_small_random_rows(self) -> None:
        rng: np.random.Generator = np.random.default_rng(0)

        n_cols: int
        for n_cols in range(1, 7):
            for _ in range(200):
                source_row: np.ndarray = rng.integers(0, 2, size=n_cols, dtype=np.uint8)
                destination_row: np.ndarray = rng.integers(0, 2, size=n_cols, dtype=np.uint8)
                state: np.ndarray = _make_state([source_row.tolist(), destination_row.tolist()])

                observed: int = direct_cut_capacity(state, 0, 1)
                expected: int = _bruteforce_direct_cut_capacity(
                    source_row=source_row,
                    destination_row=destination_row,
                )
                assert observed == expected


class TestGET_ALL_MOVES_BTWN_ROWS_COLS:
    def test_get_all_moves_btwn_rows_cols_raises_for_negative_source_row(self) -> None:
        state: np.ndarray = _make_state([[1, 0], [0, 0]])
        with pytest.raises(IndexError):
            get_all_moves_btwn_rows_cols(state, -1, 1, n_transfer_needed=0)

    def test_get_all_moves_btwn_rows_cols_raises_for_negative_destination_row(self) -> None:
        state: np.ndarray = _make_state([[1, 0], [0, 0]])
        with pytest.raises(IndexError):
            get_all_moves_btwn_rows_cols(state, 0, -1, n_transfer_needed=0)

    def test_get_all_moves_btwn_rows_cols_raises_for_source_row_out_of_bounds(self) -> None:
        state: np.ndarray = _make_state([[1, 0], [0, 0]])
        with pytest.raises(IndexError):
            get_all_moves_btwn_rows_cols(state, 2, 1, n_transfer_needed=0)

    def test_get_all_moves_btwn_rows_cols_raises_for_destination_row_out_of_bounds(self) -> None:
        state: np.ndarray = _make_state([[1, 0], [0, 0]])
        with pytest.raises(IndexError):
            get_all_moves_btwn_rows_cols(state, 0, 2, n_transfer_needed=0)

    def test_get_all_moves_btwn_rows_cols_raises_for_wrong_ndim(self) -> None:
        state_2d: np.ndarray = np.asarray([[1, 0], [0, 0]], dtype=np.uint8)
        with pytest.raises(ValueError):
            get_all_moves_btwn_rows_cols(state_2d, 0, 1, n_transfer_needed=0)

    def test_get_all_moves_btwn_rows_cols_raises_for_wrong_trailing_dimension(self) -> None:
        state_bad: np.ndarray = np.zeros((2, 3, 2), dtype=np.uint8)
        with pytest.raises(ValueError):
            get_all_moves_btwn_rows_cols(state_bad, 0, 1, n_transfer_needed=0)

    def test_get_all_moves_btwn_rows_cols_raises_for_negative_n_transfer_needed(self) -> None:
        state: np.ndarray = _make_state([[1, 0], [0, 0]])
        with pytest.raises(ValueError):
            get_all_moves_btwn_rows_cols(state, 0, 1, n_transfer_needed=-1)

    def test_planner_outputs_are_self_consistent(self) -> None:
        state: np.ndarray = _make_state([[1, 0, 1, 1], [0, 1, 0, 0]])
        from_cols, to_cols, n_planned = get_all_moves_btwn_rows_cols(
            state,
            0,
            1,
            n_transfer_needed=2,
        )
        assert len(from_cols) == n_planned
        assert len(to_cols) == n_planned

    @pytest.mark.parametrize("n_needed", [1, 2, 3])
    def test_bounded_planner_returns_requested_count_on_mixed_case(
        self,
        n_needed: int,
    ) -> None:
        state: np.ndarray = _make_state([[1, 0, 1, 1], [0, 0, 1, 0]])
        from_cols, to_cols, n_planned = get_all_moves_btwn_rows_cols(
            state,
            0,
            1,
            n_transfer_needed=n_needed,
        )
        assert n_planned == n_needed
        assert len(from_cols) == n_needed
        assert len(to_cols) == n_needed
        _assert_plan_is_legal(state, 0, 1, from_cols, to_cols)

    def test_unbounded_planner_returns_full_plan_on_mixed_case(self) -> None:
        state: np.ndarray = _make_state([[1, 0, 1, 1], [0, 0, 1, 0]])
        from_cols, to_cols, n_planned = get_all_moves_btwn_rows_cols(
            state,
            0,
            1,
            n_transfer_needed=0,
        )
        assert n_planned == 3
        assert len(from_cols) == 3
        assert len(to_cols) == 3
        _assert_plan_is_legal(state, 0, 1, from_cols, to_cols)

    def test_unbounded_planner_matches_exact_capacity_metric(self) -> None:
        rng: np.random.Generator = np.random.default_rng(1)
        n_cols: int
        for n_cols in range(1, 7):
            for _ in range(200):
                source_row: np.ndarray = rng.integers(0, 2, size=n_cols, dtype=np.uint8)
                destination_row: np.ndarray = rng.integers(0, 2, size=n_cols, dtype=np.uint8)
                state: np.ndarray = _make_state([source_row.tolist(), destination_row.tolist()])

                exact_capacity: int = direct_cut_capacity(state, 0, 1)
                from_cols, to_cols, n_planned = get_all_moves_btwn_rows_cols(
                    state,
                    0,
                    1,
                    n_transfer_needed=0,
                )
                assert n_planned == exact_capacity
                assert len(from_cols) == exact_capacity
                assert len(to_cols) == exact_capacity
                _assert_plan_is_legal(state, 0, 1, from_cols, to_cols)

    def test_bounded_planner_uses_best_pure_mode_when_pure_is_sufficient(self) -> None:
        rng: np.random.Generator = np.random.default_rng(2)
        n_cols: int
        for n_cols in range(1, 7):
            for _ in range(150):
                source_row: np.ndarray = rng.integers(0, 2, size=n_cols, dtype=np.uint8)
                destination_row: np.ndarray = rng.integers(0, 2, size=n_cols, dtype=np.uint8)
                state: np.ndarray = _make_state([source_row.tolist(), destination_row.tolist()])

                full_capacity: int = direct_cut_capacity(state, 0, 1)
                n_needed: int
                for n_needed in range(1, full_capacity + 1):
                    expected_from, expected_to, expected_n = _test_best_pure_bounded_plan(
                        source_row=source_row,
                        destination_row=destination_row,
                        n_needed=n_needed,
                    )
                    if expected_n == 0:
                        continue

                    observed_from, observed_to, observed_n = get_all_moves_btwn_rows_cols(
                        state,
                        0,
                        1,
                        n_transfer_needed=n_needed,
                    )
                    assert observed_n == expected_n
                    assert np.array_equal(observed_from, expected_from)
                    assert np.array_equal(observed_to, expected_to)
                    _assert_plan_is_legal(state, 0, 1, observed_from, observed_to)

    def test_bounded_planner_prefers_central_destinations_within_pure_mode(self) -> None:
        state: np.ndarray = _make_state([[1, 0, 1, 1, 0, 1], [0, 1, 0, 0, 1, 0]])
        from_cols, to_cols, n_planned = get_all_moves_btwn_rows_cols(
            state,
            0,
            1,
            n_transfer_needed=2,
        )
        assert n_planned == 2
        assert set(to_cols.tolist()) == {2, 3}
        _assert_plan_is_legal(state, 0, 1, from_cols, to_cols)

    def test_bounded_planner_truncates_exact_fallback_by_centrality(self) -> None:
        state: np.ndarray = _make_state([[1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 1, 0]])
        full_from, full_to, full_n = get_all_moves_btwn_rows_cols(state, 0, 1, n_transfer_needed=0)
        assert full_n == 3

        bounded_from, bounded_to, bounded_n = get_all_moves_btwn_rows_cols(
            state,
            0,
            1,
            n_transfer_needed=2,
        )

        expected_from, expected_to, expected_n = _test_truncate_by_destination_centrality(
            from_cols=full_from,
            to_cols=full_to,
            destination_row=state[1, :, 0],
            n_keep=2,
        )
        assert bounded_n == 2
        assert expected_n == 2
        assert np.array_equal(bounded_from, expected_from)
        assert np.array_equal(bounded_to, expected_to)
        _assert_plan_is_legal(state, 0, 1, bounded_from, bounded_to)

    def test_bounded_planner_truncates_unbounded_exact_plan_when_pure_is_insufficient(self) -> None:
        rng: np.random.Generator = np.random.default_rng(4)
        n_cols: int
        for n_cols in range(1, 7):
            for _ in range(250):
                source_row: np.ndarray = rng.integers(0, 2, size=n_cols, dtype=np.uint8)
                destination_row: np.ndarray = rng.integers(0, 2, size=n_cols, dtype=np.uint8)
                state: np.ndarray = _make_state([source_row.tolist(), destination_row.tolist()])

                capacity: int = direct_cut_capacity(state, 0, 1)
                if capacity <= 1:
                    continue

                best_pure_count: int = max(
                    _test_pure_mode_matches(source_row, destination_row, shift=0)[2],
                    _test_pure_mode_matches(source_row, destination_row, shift=-1)[2],
                    _test_pure_mode_matches(source_row, destination_row, shift=1)[2],
                )

                n_needed: int
                for n_needed in range(1, capacity):
                    if best_pure_count >= n_needed:
                        continue

                    full_from, full_to, full_n = get_all_moves_btwn_rows_cols(
                        state,
                        0,
                        1,
                        n_transfer_needed=0,
                    )
                    expected_from, expected_to, expected_n = _test_truncate_by_destination_centrality(
                        from_cols=full_from,
                        to_cols=full_to,
                        destination_row=destination_row,
                        n_keep=n_needed,
                    )
                    observed_from, observed_to, observed_n = get_all_moves_btwn_rows_cols(
                        state,
                        0,
                        1,
                        n_transfer_needed=n_needed,
                    )
                    assert full_n == capacity
                    assert observed_n == expected_n
                    assert np.array_equal(observed_from, expected_from)
                    assert np.array_equal(observed_to, expected_to)
                    _assert_plan_is_legal(state, 0, 1, observed_from, observed_to)

    def test_bounded_fallback_output_is_subset_of_unbounded_exact_plan(self) -> None:
        rng: np.random.Generator = np.random.default_rng(5)
        n_cols: int
        for n_cols in range(1, 7):
            for _ in range(250):
                source_row: np.ndarray = rng.integers(0, 2, size=n_cols, dtype=np.uint8)
                destination_row: np.ndarray = rng.integers(0, 2, size=n_cols, dtype=np.uint8)
                state: np.ndarray = _make_state([source_row.tolist(), destination_row.tolist()])

                capacity: int = direct_cut_capacity(state, 0, 1)
                if capacity <= 1:
                    continue

                best_pure_count: int = max(
                    _test_pure_mode_matches(source_row, destination_row, shift=0)[2],
                    _test_pure_mode_matches(source_row, destination_row, shift=-1)[2],
                    _test_pure_mode_matches(source_row, destination_row, shift=1)[2],
                )

                n_needed: int
                for n_needed in range(1, capacity):
                    if best_pure_count >= n_needed:
                        continue

                    full_from, full_to, full_n = get_all_moves_btwn_rows_cols(
                        state,
                        0,
                        1,
                        n_transfer_needed=0,
                    )
                    bounded_from, bounded_to, bounded_n = get_all_moves_btwn_rows_cols(
                        state,
                        0,
                        1,
                        n_transfer_needed=n_needed,
                    )
                    full_edges: set[tuple[int, int]] = set(
                        zip(full_from.tolist(), full_to.tolist(), strict=True)
                    )
                    bounded_edges: set[tuple[int, int]] = set(
                        zip(bounded_from.tolist(), bounded_to.tolist(), strict=True)
                    )
                    assert full_n == capacity
                    assert bounded_n == n_needed
                    assert bounded_edges.issubset(full_edges)
                    _assert_plan_is_legal(state, 0, 1, bounded_from, bounded_to)


class TestPERFORM_TRANSFER:
    def test_returns_no_moves_when_remaining_is_zero(self) -> None:
        state: np.ndarray = _make_state([[1, 0, 1], [0, 0, 0]])
        new_state, moves, n_moved = perform_transfer(state, 0, 0, 1)
        assert n_moved == 0
        assert moves == []
        assert np.array_equal(new_state, state)

    def test_moves_full_capacity_when_remaining_exceeds_capacity(self) -> None:
        state: np.ndarray = _make_state([[1, 0, 1], [0, 0, 0]])
        new_state, moves, n_moved = perform_transfer(state, 10, 0, 1)
        assert n_moved == 2
        assert len(moves) == 2
        assert source_supply_at_boundary(new_state, 0) == 0
        assert int(np.sum(new_state[1, :, 0], dtype=np.int64)) == 2

    def test_preserves_total_atom_number(self) -> None:
        state: np.ndarray = _make_state([[1, 1, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]])
        total_before: int = int(np.sum(state, dtype=np.int64))
        new_state, moves, n_moved = perform_transfer(state, 2, 0, 1)
        total_after: int = int(np.sum(new_state, dtype=np.int64))
        assert len(moves) == n_moved
        assert total_after == total_before

    def test_raises_for_negative_remaining(self) -> None:
        state: np.ndarray = _make_state([[1, 0], [0, 0]])
        with pytest.raises(ValueError):
            perform_transfer(state, -1, 0, 1)

    def test_raises_for_wrong_shape(self) -> None:
        state_2d: np.ndarray = np.asarray([[1, 0], [0, 1]], dtype=np.uint8)
        with pytest.raises(ValueError):
            perform_transfer(state_2d, 1, 0, 1)

    def test_moves_exactly_requested_count_when_remaining_is_below_capacity(self) -> None:
        state: np.ndarray = _make_state([[1, 0, 1, 1], [0, 1, 0, 0]])
        new_state, moves, n_moved = perform_transfer(state, 2, 0, 1)
        assert n_moved == 2
        assert len(moves) == 2
        assert source_supply_at_boundary(new_state, 0) == 1
        assert int(np.sum(new_state[1, :, 0], dtype=np.int64)) == 3
        assert int(np.sum(new_state, dtype=np.int64)) == int(np.sum(state, dtype=np.int64))

    def test_matches_exact_capacity_when_remaining_is_large(self) -> None:
        rng: np.random.Generator = np.random.default_rng(3)
        n_cols: int
        for n_cols in range(1, 7):
            for _ in range(150):
                source_row: np.ndarray = rng.integers(0, 2, size=n_cols, dtype=np.uint8)
                destination_row: np.ndarray = rng.integers(0, 2, size=n_cols, dtype=np.uint8)
                state: np.ndarray = _make_state([source_row.tolist(), destination_row.tolist()])
                expected_capacity: int = direct_cut_capacity(state, 0, 1)
                total_before: int = int(np.sum(state, dtype=np.int64))
                new_state, moves, n_moved = perform_transfer(
                    state,
                    remaining=n_cols + 5,
                    boundary_src_row=0,
                    boundary_dst_row=1,
                )
                assert n_moved == expected_capacity
                assert len(moves) == expected_capacity
                assert int(np.sum(new_state, dtype=np.int64)) == total_before

    def test_exact_state_matches_direct_application_of_returned_plan(self) -> None:
        state: np.ndarray = _make_state([[1, 0, 1, 1], [0, 0, 1, 0]])
        new_state, moves, n_moved = perform_transfer(state, 2, 0, 1)
        from_cols: np.ndarray = np.asarray([int(move.from_col) for move in moves], dtype=np.intp)
        to_cols: np.ndarray = np.asarray([int(move.to_col) for move in moves], dtype=np.intp)
        expected_state: np.ndarray = _apply_planned_row_transfer(
            state=state,
            boundary_src_row=0,
            boundary_dst_row=1,
            from_cols=from_cols,
            to_cols=to_cols,
        )
        assert n_moved == 2
        assert np.array_equal(new_state, expected_state)

    @pytest.mark.parametrize(
        "state_input",
        [
            [[0, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [1, 1, 1]],
        ],
    )
    def test_returns_no_moves_when_cut_capacity_is_zero(self, state_input: list[list[int]]) -> None:
        state: np.ndarray = _make_state(state_input)
        new_state, moves, n_moved = perform_transfer(state, 2, 0, 1)
        assert n_moved == 0
        assert moves == []
        assert np.array_equal(new_state, state)

class Test_PLAN_HORIZONTAL_ROUNDS_TO_TARGET:
    def test_returns_empty_when_row_already_matches_target(self) -> None:
        """
        The planner should emit no rounds when the current and target rows already
        agree, so helper/controller layers do not see fake work.
        """
        current_row: np.ndarray = np.array(
            [0, 1, 0, 1, 1, 0, 0],
            dtype=np.uint8,
        )
        target_row: np.ndarray = current_row.copy()

        move_rounds: list[list[Move]] = _plan_horizontal_rounds_to_target(
            current_row=current_row,
            target_row=target_row,
            boundary_dst_row=3,
        )

        assert move_rounds == []

    def test_parallelizes_mixed_left_and_right_motion_without_conflict(self) -> None:
        """
        The planner should batch simultaneously legal left- and right-moving unit
        steps in the same round when they do not conflict.
        """
        current_row: np.ndarray = np.array(
            [0, 1, 1, 1, 0, 1, 1, 0],
            dtype=np.uint8,
        )
        target_row: np.ndarray = np.array(
            [1, 1, 0, 0, 1, 1, 0, 1],
            dtype=np.uint8,
        )

        move_rounds: list[list[Move]] = _plan_horizontal_rounds_to_target(
            current_row=current_row,
            target_row=target_row,
            boundary_dst_row=2,
        )

        assert len(move_rounds) == 1

        round_moves: list[Move] = move_rounds[0]
        assert len(round_moves) == 4

        observed_edges: set[tuple[int, int]] = {
            (int(move.from_col), int(move.to_col)) for move in round_moves
        }
        expected_edges: set[tuple[int, int]] = {(1, 0), (2, 1), (3, 4), (6, 7)}
        assert observed_edges == expected_edges

class TestENSURE_SOURCE_SUPPLY:
    def test_raises_for_wrong_shape(self) -> None:
        state_2d: np.ndarray = np.asarray([[1, 0], [0, 1]], dtype=np.uint8)
        with pytest.raises(ValueError):
            ensure_source_supply(
                state=state_2d,
                boundary_src_row=1,
                search_limit_row=0,
                delta_S=1,
            )

    def test_raises_for_negative_delta_S(self) -> None:
        state: np.ndarray = _make_state([[0, 0], [1, 0]])
        with pytest.raises(ValueError):
            ensure_source_supply(
                state=state,
                boundary_src_row=1,
                search_limit_row=0,
                delta_S=-1,
            )

    def test_raises_for_invalid_fill_mode(self) -> None:
        state: np.ndarray = _make_state([[0, 0], [1, 0]])
        with pytest.raises(ValueError):
            ensure_source_supply(
                state=state,
                boundary_src_row=1,
                search_limit_row=0,
                delta_S=1,
                fill_mode="bad_mode",  # type: ignore[arg-type]
            )

    def test_raises_for_boundary_row_out_of_bounds(self) -> None:
        state: np.ndarray = _make_state([[0, 0], [1, 0]])
        with pytest.raises(IndexError):
            ensure_source_supply(
                state=state,
                boundary_src_row=2,
                search_limit_row=0,
                delta_S=1,
            )

    def test_raises_for_search_limit_row_out_of_bounds(self) -> None:
        state: np.ndarray = _make_state([[0, 0], [1, 0]])
        with pytest.raises(IndexError):
            ensure_source_supply(
                state=state,
                boundary_src_row=1,
                search_limit_row=-1,
                delta_S=1,
            )

    def test_raises_when_search_limit_row_equals_boundary_src_row(self) -> None:
        state: np.ndarray = _make_state([[0, 0], [1, 0], [0, 1]])
        with pytest.raises(ValueError):
            ensure_source_supply(
                state=state,
                boundary_src_row=1,
                search_limit_row=1,
                delta_S=1,
            )

    def test_returns_success_immediately_when_delta_S_is_zero(self) -> None:
        state: np.ndarray = _make_state([[0, 1, 0], [1, 0, 0]])
        new_state, move_rounds, achieved_delta_S, status = ensure_source_supply(
            state=state,
            boundary_src_row=1,
            search_limit_row=0,
            delta_S=0,
            fill_mode="exact",
        )
        assert np.array_equal(new_state, state)
        assert move_rounds == []
        assert achieved_delta_S == 0
        assert status["kind"] == "success"
        assert status["unmet_delta_S"] == 0

    def test_exact_mode_reaches_exact_requested_boundary_increase_when_possible(self) -> None:
        state: np.ndarray = _make_state([[1, 0, 1], [0, 0, 0]])
        before_supply: int = source_supply_at_boundary(state, 1)
        new_state, move_rounds, achieved_delta_S, status = ensure_source_supply(
            state=state,
            boundary_src_row=1,
            search_limit_row=0,
            delta_S=1,
            fill_mode="exact",
        )
        after_supply: int = source_supply_at_boundary(new_state, 1)
        assert achieved_delta_S == 1
        assert after_supply - before_supply == 1
        assert status["kind"] == "success"
        assert status["unmet_delta_S"] == 0
        assert len(move_rounds) > 0

    def test_opportunistic_mode_allows_overfill_within_same_round_budget(self) -> None:
        state: np.ndarray = _make_state([[1, 0, 1], [0, 0, 0]])
        before_supply: int = source_supply_at_boundary(state, 1)
        new_state, move_rounds, achieved_delta_S, status = ensure_source_supply(
            state=state,
            boundary_src_row=1,
            search_limit_row=0,
            delta_S=1,
            fill_mode="opportunistic",
        )
        after_supply: int = source_supply_at_boundary(new_state, 1)
        assert achieved_delta_S >= 1
        assert after_supply - before_supply == achieved_delta_S
        assert status["kind"] == "success"
        assert status["unmet_delta_S"] == 0

    def test_exact_vs_opportunistic_randomized_comparison(self) -> None:
        rng: np.random.Generator = np.random.default_rng(17)
        n_cols: int
        for n_cols in range(2, 7):
            for _ in range(200):
                source_row: np.ndarray = rng.integers(0, 2, size=n_cols, dtype=np.uint8)
                boundary_row: np.ndarray = rng.integers(0, 2, size=n_cols, dtype=np.uint8)
                state: np.ndarray = _make_state([source_row.tolist(), boundary_row.tolist()])
                exact_state, exact_rounds, exact_delta, exact_status = ensure_source_supply(
                    state=state,
                    boundary_src_row=1,
                    search_limit_row=0,
                    delta_S=1,
                    fill_mode="exact",
                )
                opp_state, opp_rounds, opp_delta, opp_status = ensure_source_supply(
                    state=state,
                    boundary_src_row=1,
                    search_limit_row=0,
                    delta_S=1,
                    fill_mode="opportunistic",
                )
                assert opp_delta >= exact_delta
                assert len(opp_rounds) == len(exact_rounds)
                assert int(np.sum(exact_state, dtype=np.int64)) == int(np.sum(state, dtype=np.int64))
                assert int(np.sum(opp_state, dtype=np.int64)) == int(np.sum(state, dtype=np.int64))

    def test_returns_insufficient_atoms_when_source_interval_cannot_supply_request(self) -> None:
        state: np.ndarray = _make_state([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
        new_state, move_rounds, achieved_delta_S, status = ensure_source_supply(
            state=state,
            boundary_src_row=2,
            search_limit_row=1,
            delta_S=2,
            fill_mode="exact",
        )
        assert achieved_delta_S == 0
        assert status["kind"] == "insufficient_atoms"
        assert status["unmet_delta_S"] == 2
        assert np.array_equal(new_state, state)
        assert move_rounds == []

    def test_partial_blocked_commits_progress_when_some_supply_reaches_boundary(self) -> None:
        state: np.ndarray = _make_state(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 1, 0, 0],
            ]
        )
        before_supply: int = source_supply_at_boundary(state, 2)
        new_state, move_rounds, achieved_delta_S, status = ensure_source_supply(
            state=state,
            boundary_src_row=2,
            search_limit_row=0,
            delta_S=2,
            fill_mode="exact",
        )
        after_supply: int = source_supply_at_boundary(new_state, 2)
        assert status["kind"] == "partial_blocked"
        assert achieved_delta_S > 0
        assert after_supply - before_supply == achieved_delta_S
        assert int(status["unmet_delta_S"]) > 0
        assert isinstance(status["blocking_row"], int)
        assert len(move_rounds) > 0

    def test_partial_blocked_with_zero_progress_is_allowed_when_immediately_blocked(self) -> None:
        state: np.ndarray = _make_state(
            [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ]
        )
        new_state, move_rounds, achieved_delta_S, status = ensure_source_supply(
            state=state,
            boundary_src_row=2,
            search_limit_row=0,
            delta_S=2,
            fill_mode="exact",
        )
        if status["kind"] == "partial_blocked":
            assert achieved_delta_S == 0
            assert isinstance(status["blocking_row"], int)
            assert int(status["unmet_delta_S"]) > 0
            assert np.array_equal(new_state, state)
            assert move_rounds == []

    def test_multi_layer_relay_success(self) -> None:
        state: np.ndarray = _make_state(
            [
                [1, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        before_supply: int = source_supply_at_boundary(state, 2)
        new_state, move_rounds, achieved_delta_S, status = ensure_source_supply(
            state=state,
            boundary_src_row=2,
            search_limit_row=0,
            delta_S=1,
            fill_mode="exact",
        )
        after_supply: int = source_supply_at_boundary(new_state, 2)
        assert status["kind"] == "success"
        assert achieved_delta_S == 1
        assert after_supply - before_supply == 1
        assert len(move_rounds) >= 2

    def test_never_touches_rows_outside_allowed_interval_when_search_limit_above_boundary(self) -> None:
        state: np.ndarray = _make_state(
            [
                [1, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 1],
            ]
        )
        protected_top: np.ndarray = state[0, :, :].copy()
        protected_bottom: np.ndarray = state[3, :, :].copy()
        new_state, move_rounds, achieved_delta_S, status = ensure_source_supply(
            state=state,
            boundary_src_row=2,
            search_limit_row=1,
            delta_S=1,
            fill_mode="exact",
        )
        assert np.array_equal(new_state[0, :, :], protected_top)
        assert np.array_equal(new_state[3, :, :], protected_bottom)

    def test_never_touches_rows_outside_allowed_interval_when_search_limit_below_boundary(self) -> None:
        state: np.ndarray = _make_state(
            [
                [1, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 1],
            ]
        )
        protected_top: np.ndarray = state[0, :, :].copy()
        protected_bottom: np.ndarray = state[3, :, :].copy()
        new_state, move_rounds, achieved_delta_S, status = ensure_source_supply(
            state=state,
            boundary_src_row=1,
            search_limit_row=2,
            delta_S=1,
            fill_mode="exact",
        )
        assert np.array_equal(new_state[0, :, :], protected_top)
        assert np.array_equal(new_state[3, :, :], protected_bottom)

    def test_total_atom_number_is_conserved(self) -> None:
        state: np.ndarray = _make_state([[1, 0, 1], [0, 1, 0], [0, 0, 0]])
        total_before: int = int(np.sum(state, dtype=np.int64))
        new_state, move_rounds, achieved_delta_S, status = ensure_source_supply(
            state=state,
            boundary_src_row=2,
            search_limit_row=0,
            delta_S=2,
            fill_mode="opportunistic",
        )
        total_after: int = int(np.sum(new_state, dtype=np.int64))
        assert total_after == total_before

    def test_achieved_delta_matches_actual_boundary_supply_change(self) -> None:
        rng: np.random.Generator = np.random.default_rng(23)
        n_cols: int
        for n_cols in range(2, 7):
            for _ in range(150):
                state: np.ndarray = _make_state(
                    [
                        rng.integers(0, 2, size=n_cols, dtype=np.uint8).tolist(),
                        rng.integers(0, 2, size=n_cols, dtype=np.uint8).tolist(),
                        rng.integers(0, 2, size=n_cols, dtype=np.uint8).tolist(),
                    ]
                )
                before_supply: int = source_supply_at_boundary(state, 2)
                new_state, move_rounds, achieved_delta_S, status = ensure_source_supply(
                    state=state,
                    boundary_src_row=2,
                    search_limit_row=0,
                    delta_S=1,
                    fill_mode="opportunistic",
                )
                after_supply: int = source_supply_at_boundary(new_state, 2)
                assert achieved_delta_S == after_supply - before_supply

    def test_randomized_moves_stay_within_interval_and_move_toward_boundary(self) -> None:
        rng: np.random.Generator = np.random.default_rng(31)
        n_cols: int
        for n_cols in range(2, 7):
            for _ in range(150):
                state: np.ndarray = _make_state(
                    [
                        rng.integers(0, 2, size=n_cols, dtype=np.uint8).tolist(),
                        rng.integers(0, 2, size=n_cols, dtype=np.uint8).tolist(),
                        rng.integers(0, 2, size=n_cols, dtype=np.uint8).tolist(),
                        rng.integers(0, 2, size=n_cols, dtype=np.uint8).tolist(),
                    ]
                )
                boundary_src_row: int = 2
                search_limit_row: int = 0
                new_state, move_rounds, achieved_delta_S, status = ensure_source_supply(
                    state=state,
                    boundary_src_row=boundary_src_row,
                    search_limit_row=search_limit_row,
                    delta_S=1,
                    fill_mode="exact",
                )
                low_row: int = min(boundary_src_row, search_limit_row)
                high_row: int = max(boundary_src_row, search_limit_row)
                for move_round in move_rounds:
                    for move in move_round:
                        assert low_row <= int(move.from_row) <= high_row
                        assert low_row <= int(move.to_row) <= high_row
                        assert abs(int(move.to_row) - int(move.from_row)) == 1
                        assert abs(boundary_src_row - int(move.to_row)) < abs(boundary_src_row - int(move.from_row))


def _make_state_from_rows(*rows: list[int]) -> np.ndarray:
    """Build a BCv2-style single-species state array from 2D row data."""
    arr_2d: np.ndarray = np.asarray(rows, dtype=np.uint8)
    if arr_2d.ndim != 2:
        raise ValueError("rows must define a 2D occupancy grid.")
    return arr_2d[:, :, None]


def _row_count(state: np.ndarray, row: int) -> int:
    """Return the atom count on one row."""
    return int(np.sum(state[row, :, 0], dtype=np.int64))


def _total_count(state: np.ndarray) -> int:
    """Return the total atom count in the full state."""
    return int(np.sum(state[:, :, 0], dtype=np.int64))


def _horizontal_ceiling(
    state: np.ndarray,
    boundary_src_row: int,
    boundary_dst_row: int,
) -> int:
    """Return the eventual horizontal-only cut-capacity ceiling ``min(S, V_dst)``."""
    src_supply: int = _row_count(state, boundary_src_row)
    dst_vacancies: int = int(np.sum(state[boundary_dst_row, :, 0] == 0, dtype=np.int64))
    return min(src_supply, dst_vacancies)


def _replay_horizontal_rounds(
    state: np.ndarray,
    move_rounds: list[list[Move]],
    boundary_dst_row: int,
) -> np.ndarray:
    """Replay returned move rounds while checking horizontal-round legality."""
    work_state: np.ndarray = state.copy()

    round_index: int
    round_moves: list[Move]
    for round_index, round_moves in enumerate(move_rounds):
        from_cols_seen: set[int] = set()
        to_cols_seen: set[int] = set()

        move: Move
        for move in round_moves:
            assert int(move.from_row) == boundary_dst_row, (
                f"Round {round_index} contains move from unexpected row: {move}."
            )
            assert int(move.to_row) == boundary_dst_row, (
                f"Round {round_index} contains move to unexpected row: {move}."
            )
            assert abs(int(move.to_col) - int(move.from_col)) == 1, (
                f"Round {round_index} contains non-unit horizontal move: {move}."
            )

            from_col: int = int(move.from_col)
            to_col: int = int(move.to_col)

            assert from_col not in from_cols_seen, (
                f"Round {round_index} reuses source column {from_col}."
            )
            assert to_col not in to_cols_seen, (
                f"Round {round_index} reuses destination column {to_col}."
            )

            from_cols_seen.add(from_col)
            to_cols_seen.add(to_col)

            assert int(work_state[boundary_dst_row, from_col, 0]) == 1, (
                f"Round {round_index} tries to move from an empty site at "
                f"({boundary_dst_row}, {from_col})."
            )

        total_before: int = int(np.sum(work_state[:, :, 0], dtype=np.int64))
        row_before: int = int(np.sum(work_state[boundary_dst_row, :, 0], dtype=np.int64))

        next_state: np.ndarray = move_atoms_noiseless(work_state.copy(), round_moves)

        total_after: int = int(np.sum(next_state[:, :, 0], dtype=np.int64))
        row_after: int = int(np.sum(next_state[boundary_dst_row, :, 0], dtype=np.int64))

        assert total_after == total_before, (
            f"Round {round_index} lost or created atoms under noiseless realization."
        )
        assert row_after == row_before, (
            f"Round {round_index} changed destination-row atom count under "
            f"noiseless realization."
        )

        work_state = next_state

    return work_state


def _capacity_after_each_round(
    state: np.ndarray,
    move_rounds: list[list[Move]],
    boundary_src_row: int,
    boundary_dst_row: int,
) -> list[int]:
    """Return exact direct cut capacity after each replayed round, including the initial value."""
    capacities: list[int] = [direct_cut_capacity(state, boundary_src_row, boundary_dst_row)]
    work_state: np.ndarray = state.copy()

    round_moves: list[Move]
    for round_moves in move_rounds:
        work_state = move_atoms_noiseless(work_state.copy(), round_moves)
        capacities.append(direct_cut_capacity(work_state, boundary_src_row, boundary_dst_row))

    return capacities


def _assert_only_destination_boundary_row_touched(
    before: np.ndarray,
    after: np.ndarray,
    boundary_dst_row: int,
) -> None:
    """Assert that only the destination boundary row changed."""
    changed_mask: np.ndarray = before[:, :, 0] != after[:, :, 0]
    changed_rows: np.ndarray = np.flatnonzero(np.any(changed_mask, axis=1))

    if changed_rows.size == 0:
        return

    assert changed_rows.tolist() == [boundary_dst_row], (
        f"Helper changed rows {changed_rows.tolist()}, but only destination boundary "
        f"row {boundary_dst_row} may be touched."
    )


def _assert_strict_horizontal_helper_invariants(
    before: np.ndarray,
    after: np.ndarray,
    move_rounds: list[list[Move]],
    boundary_src_row: int,
    boundary_dst_row: int,
    achieved_delta_t: int,
    status: dict[str, int | bool | str],
) -> None:
    """Assert the core helper invariants that should hold in every non-raising path."""
    replayed_state: np.ndarray = _replay_horizontal_rounds(
        state=before,
        move_rounds=move_rounds,
        boundary_dst_row=boundary_dst_row,
    )
    assert np.array_equal(replayed_state, after)

    _assert_only_destination_boundary_row_touched(
        before=before,
        after=after,
        boundary_dst_row=boundary_dst_row,
    )

    assert _row_count(after, boundary_dst_row) == _row_count(before, boundary_dst_row)
    assert _total_count(after) == _total_count(before)
    assert np.array_equal(after[boundary_src_row, :, 0], before[boundary_src_row, :, 0])

    t_before: int = direct_cut_capacity(before, boundary_src_row, boundary_dst_row)
    t_after: int = direct_cut_capacity(after, boundary_src_row, boundary_dst_row)

    assert achieved_delta_t == t_after - t_before
    assert int(status["T_before"]) == t_before
    assert int(status["T_after"]) == t_after
    assert int(status["achieved_delta_T"]) == achieved_delta_t
    assert int(status["n_rounds"]) == len(move_rounds)

    if str(status["kind"]) == "success":
        assert achieved_delta_t >= int(status["requested_delta_T"])


class TestENSURE_BOUNDARY_ROW_CUT_CAPACITY_HORIZONTAL:
    """
    Contract tests for horizontal boundary-row cut-capacity realignment.

    Written contract
    ----------------
    ``ensure_boundary_row_cut_capacity_horizontal(...)`` is a destination-side
    helper that acts only on the destination boundary row, using horizontal
    parallel single-site non-colliding moves, to increase exact direct cut
    capacity ``T`` by at least ``delta_T`` while preserving that row's atom
    count.

    The helper:
    - may touch only ``boundary_dst_row``,
    - may use only horizontal unit moves within that row,
    - must preserve both global atom count and destination-row atom count,
    - is exact about achieved ``delta_T`` via recomputed
      ``direct_cut_capacity(...)``,
    - should reject impossible requests up front when
      ``T_before + delta_T > min(S, V_dst)``,
    - should not spend extra rounds solely to continue improving ``T`` after
      first reaching the requested gain,
    - should prefer staying within the target window when an equally effective
      within-target solution exists,
    - and, on tiny obvious cases, should prefer the least disruptive successful
      realignment when the intended action is operationally unambiguous.
    """

    def test_returns_success_and_leaves_state_unchanged_when_delta_t_is_zero(self) -> None:
        """A zero request should succeed without executing any moves."""
        state: np.ndarray = _make_state_from_rows(
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
        )

        t_before: int = direct_cut_capacity(state, 0, 1)

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                delta_T=0,
                target_start_col=1,
                target_end_col=5,
            )
        )

        assert np.array_equal(new_state, state)
        assert move_rounds == []
        assert achieved_delta_t == 0

        assert status["kind"] == "success"
        assert status["T_before"] == t_before
        assert status["T_after"] == t_before
        assert status["requested_delta_T"] == 0
        assert status["achieved_delta_T"] == 0
        assert status["used_outside_target_cols"] is False
        assert status["n_rounds"] == 0

    def test_raises_on_invalid_arguments(self) -> None:
        """Invalid shape, indices, and delta should fail loudly."""
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1],
            [0, 1, 0],
        )

        with pytest.raises(ValueError):
            ensure_boundary_row_cut_capacity_horizontal(
                state=state[:, :, 0],
                boundary_src_row=0,
                boundary_dst_row=1,
                delta_T=1,
                target_start_col=0,
                target_end_col=2,
            )

        with pytest.raises(ValueError):
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                delta_T=-1,
                target_start_col=0,
                target_end_col=2,
            )

        with pytest.raises(IndexError):
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=-1,
                boundary_dst_row=1,
                delta_T=1,
                target_start_col=0,
                target_end_col=2,
            )

        with pytest.raises(IndexError):
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=2,
                delta_T=1,
                target_start_col=0,
                target_end_col=2,
            )

        with pytest.raises(ValueError):
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                delta_T=1,
                target_start_col=2,
                target_end_col=1,
            )

    def test_returns_infeasible_request_when_requested_gain_exceeds_horizontal_only_ceiling(
        self,
    ) -> None:
        """The helper should reject impossible horizontal-only gain requests up front."""
        state: np.ndarray = _make_state_from_rows(
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
        )
        t_before: int = direct_cut_capacity(state, 0, 1)

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                delta_T=3,
                target_start_col=1,
                target_end_col=5,
            )
        )

        assert np.array_equal(new_state, state)
        assert move_rounds == []
        assert achieved_delta_t == 0

        assert status["kind"] == "infeasible_request"
        assert status["T_before"] == t_before
        assert status["T_after"] == t_before
        assert status["requested_delta_T"] == 3
        assert status["achieved_delta_T"] == 0
        assert status["used_outside_target_cols"] is False
        assert status["n_rounds"] == 0

    def test_source_row_empty_makes_any_positive_gain_infeasible(self) -> None:
        """If ``S = 0``, horizontal rearrangement cannot improve cut capacity."""
        state: np.ndarray = _make_state_from_rows(
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1],
        )

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                delta_T=1,
                target_start_col=0,
                target_end_col=4,
            )
        )

        assert np.array_equal(new_state, state)
        assert move_rounds == []
        assert achieved_delta_t == 0
        assert status["kind"] == "infeasible_request"

    def test_destination_row_all_full_makes_any_positive_gain_infeasible(self) -> None:
        """If ``V_dst = 0``, horizontal rearrangement cannot improve cut capacity."""
        state: np.ndarray = _make_state_from_rows(
            [1, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
        )

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                delta_T=1,
                target_start_col=0,
                target_end_col=4,
            )
        )

        assert np.array_equal(new_state, state)
        assert move_rounds == []
        assert achieved_delta_t == 0
        assert status["kind"] == "infeasible_request"

    def test_already_horizontally_saturated_makes_any_positive_gain_infeasible(self) -> None:
        """If ``T`` already equals ``min(S, V_dst)``, any positive ``delta_T`` is infeasible."""
        state: np.ndarray = _make_state_from_rows(
            [0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1],
        )
        t_before: int = direct_cut_capacity(state, 0, 1)
        ceiling: int = _horizontal_ceiling(state, 0, 1)

        assert t_before == ceiling

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                delta_T=1,
                target_start_col=0,
                target_end_col=4,
            )
        )

        assert np.array_equal(new_state, state)
        assert move_rounds == []
        assert achieved_delta_t == 0
        assert status["kind"] == "infeasible_request"

    def test_deeper_rows_are_irrelevant_to_horizontal_helper_behavior(self) -> None:
        """Rows deeper in the ROI should not affect the helper's touch set or bookkeeping."""
        state: np.ndarray = _make_state_from_rows(
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
        )

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                delta_T=1,
                target_start_col=1,
                target_end_col=5,
            )
        )

        _assert_strict_horizontal_helper_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )
        assert np.array_equal(new_state[2:, :, 0], state[2:, :, 0])

    def test_edge_column_behavior_is_handled_correctly(self) -> None:
        """Moves involving edge columns should stay legal and replay correctly."""
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 1],
        )

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                delta_T=1,
                target_start_col=0,
                target_end_col=5,
            )
        )

        _assert_strict_horizontal_helper_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

    def test_success_increases_capacity_by_at_least_requested_delta_and_preserves_row_atom_count(
        self,
    ) -> None:
        """Successful horizontal realignment should raise ``T`` while preserving row count."""
        state: np.ndarray = _make_state_from_rows(
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1, 1, 1],
        )

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                delta_T=1,
                target_start_col=1,
                target_end_col=5,
            )
        )

        _assert_strict_horizontal_helper_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

    def test_parallelizes_mixed_direction_chain_at_leaf_level(self) -> None:
        """
        The leaf horizontal helper should realize obvious same-row parallelism
        directly, without relying on the coordinator.
        """
        state: np.ndarray = _make_state_from_rows(
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        )

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                delta_T=3,
                target_start_col=1,
                target_end_col=12,
            )
        )

        assert status["kind"] == "success"
        assert len(move_rounds) == 2
        assert achieved_delta_t == 3

        _assert_strict_horizontal_helper_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

    def test_prefers_staying_within_target_window_when_an_equal_effectiveness_solution_exists(
        self,
    ) -> None:
        """The helper should avoid outside-target motion when it is not needed."""
        state: np.ndarray = _make_state_from_rows(
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
        )

        target_start_col: int = 1
        target_end_col: int = 5

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                delta_T=1,
                target_start_col=target_start_col,
                target_end_col=target_end_col,
            )
        )

        assert status["kind"] == "success"
        assert achieved_delta_t >= 1
        assert status["used_outside_target_cols"] is False

        round_moves: list[Move]
        for round_moves in move_rounds:
            move: Move
            for move in round_moves:
                assert target_start_col <= int(move.from_col) <= target_end_col
                assert target_start_col <= int(move.to_col) <= target_end_col

        _assert_strict_horizontal_helper_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

    def test_reports_when_outside_target_motion_was_used(self) -> None:
        """Outside-target motion should be surfaced in metadata when it is required."""
        state: np.ndarray = _make_state_from_rows(
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
        )

        target_start_col: int = 1
        target_end_col: int = 4

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                delta_T=1,
                target_start_col=target_start_col,
                target_end_col=target_end_col,
            )
        )

        assert status["kind"] == "success"
        assert achieved_delta_t >= 1
        assert status["used_outside_target_cols"] is True

        touched_outside: bool = False
        round_moves: list[Move]
        for round_moves in move_rounds:
            move: Move
            for move in round_moves:
                from_col: int = int(move.from_col)
                to_col: int = int(move.to_col)
                if (
                    from_col < target_start_col
                    or from_col > target_end_col
                    or to_col < target_start_col
                    or to_col > target_end_col
                ):
                    touched_outside = True

        assert touched_outside is True

        _assert_strict_horizontal_helper_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

    def test_canonical_tiny_case_prefers_single_obvious_one_move_solution(self) -> None:
        """
        On a tiny obvious case, lock the least disruptive one-move behavior.

        Here the intended action is operationally unambiguous by eye: shift the
        rightmost destination-row atom one step right, stay within the target
        window, and achieve the requested gain in one round with one move.
        """
        state: np.ndarray = _make_state_from_rows(
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
        )

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                delta_T=1,
                target_start_col=1,
                target_end_col=5,
            )
        )

        assert status["kind"] == "success"
        assert len(move_rounds) == 1
        assert len(move_rounds[0]) == 1

        move: Move = move_rounds[0][0]
        assert (int(move.from_row), int(move.from_col), int(move.to_row), int(move.to_col)) == (
            1,
            4,
            1,
            5,
        )
        assert status["used_outside_target_cols"] is False
        assert achieved_delta_t >= 1

        _assert_strict_horizontal_helper_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

    @pytest.mark.parametrize(("delta_t",), [(1,), (2,)])
    def test_patterned_harder_case_succeeds_for_delta_t_one_and_two(self, delta_t: int) -> None:
        """A nontrivial vacancy pattern should support moderate requested gains."""
        src_row: list[int] = [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1]
        dst_row: list[int] = [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0]
        state: np.ndarray = _make_state_from_rows(src_row, dst_row)

        t_before: int = direct_cut_capacity(state, 0, 1)
        assert t_before == 2

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                delta_T=delta_t,
                target_start_col=2,
                target_end_col=11,
            )
        )

        assert status["kind"] == "success"
        _assert_strict_horizontal_helper_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

    @pytest.mark.parametrize(("delta_t",), [(3,), (4,)])
    def test_patterned_harder_case_returns_infeasible_request_when_gain_exceeds_ceiling(
        self,
        delta_t: int,
    ) -> None:
        """If requested gain exceeds ``min(S, V_dst) - T_before``, fail immediately."""
        src_row: list[int] = [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1]
        dst_row: list[int] = [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0]
        state: np.ndarray = _make_state_from_rows(src_row, dst_row)

        t_before: int = direct_cut_capacity(state, 0, 1)
        assert t_before == 2

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                delta_T=delta_t,
                target_start_col=2,
                target_end_col=11,
            )
        )

        assert np.array_equal(new_state, state)
        assert move_rounds == []
        assert achieved_delta_t == 0
        assert status["kind"] == "infeasible_request"

    @pytest.mark.parametrize(
        ("src_row", "dst_row", "delta_t"),
        [
            ([0, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0], 1),
            ([1, 1, 0, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0], 1),
            ([0, 1, 0, 1, 0, 1, 0], [1, 1, 1, 0, 1, 0, 0], 1),
            (
                [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
                2,
            ),
        ],
    )
    def test_success_cases_preserve_source_row_and_recompute_capacity_consistently(
        self,
        src_row: list[int],
        dst_row: list[int],
        delta_t: int,
    ) -> None:
        """Returned bookkeeping should agree with exact recomputation of ``T``."""
        extra_row: list[int] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1][: len(src_row)]
        state: np.ndarray = _make_state_from_rows(src_row, dst_row, extra_row)

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                delta_T=delta_t,
                target_start_col=0,
                target_end_col=len(src_row) - 1,
            )
        )

        _assert_strict_horizontal_helper_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
    def test_randomized_feasible_requests_preserve_invariants(self, seed: int) -> None:
        """Random feasible requests should satisfy the full invariant set."""
        rng: np.random.Generator = np.random.default_rng(seed)
        n_cols: int = 12

        for _ in range(25):
            src_row: np.ndarray = rng.integers(0, 2, size=n_cols, dtype=np.uint8)
            dst_row: np.ndarray = rng.integers(0, 2, size=n_cols, dtype=np.uint8)
            extra_row: np.ndarray = rng.integers(0, 2, size=n_cols, dtype=np.uint8)

            state: np.ndarray = _make_state_from_rows(
                src_row.tolist(),
                dst_row.tolist(),
                extra_row.tolist(),
            )

            t_before: int = direct_cut_capacity(state, 0, 1)
            ceiling: int = _horizontal_ceiling(state, 0, 1)
            max_gain: int = ceiling - t_before

            if max_gain < 0:
                raise RuntimeError(
                    "Internal test construction bug: horizontal ceiling cannot be "
                    "smaller than current cut capacity."
                )

            delta_t: int = int(rng.integers(0, max_gain + 1))

            new_state, move_rounds, achieved_delta_t, status = (
                ensure_boundary_row_cut_capacity_horizontal(
                    state=state,
                    boundary_src_row=0,
                    boundary_dst_row=1,
                    delta_T=delta_t,
                    target_start_col=2,
                    target_end_col=n_cols - 3,
                )
            )

            _assert_strict_horizontal_helper_invariants(
                before=state,
                after=new_state,
                move_rounds=move_rounds,
                boundary_src_row=0,
                boundary_dst_row=1,
                achieved_delta_t=achieved_delta_t,
                status=status,
            )

    def test_avoids_gratuitous_post_success_rounds_for_canonical_one_round_case(self) -> None:
        """Once the requested gain is first achieved, the helper should stop the episode."""
        state: np.ndarray = _make_state_from_rows(
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
        )
        delta_t: int = 1

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                delta_T=delta_t,
                target_start_col=1,
                target_end_col=5,
            )
        )

        assert status["kind"] == "success"
        capacities: list[int] = _capacity_after_each_round(
            state=state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
        )

        t_before: int = capacities[0]
        first_hit_index: int | None = None
        for idx, t_val in enumerate(capacities[1:], start=1):
            if t_val - t_before >= delta_t:
                first_hit_index = idx
                break

        assert first_hit_index is not None
        assert len(move_rounds) == first_hit_index == 1

        _assert_strict_horizontal_helper_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

    def test_avoids_gratuitous_post_success_rounds_for_patterned_one_round_delta_two_case(self) -> None:
        """
        On the patterned case, ``delta_T = 2`` should be achieved within one round.

        We intentionally do not over-specify the exact move subset here, because we
        have not yet declared a global priority between staying within the target
        window and minimizing moved atoms when both are not simultaneously optimal.
        """
        state: np.ndarray = _make_state_from_rows(
            [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
        )
        delta_t: int = 2

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                delta_T=delta_t,
                target_start_col=2,
                target_end_col=11,
            )
        )

        assert status["kind"] == "success"
        capacities: list[int] = _capacity_after_each_round(
            state=state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
        )

        t_before: int = capacities[0]
        first_hit_index: int | None = None
        for idx, t_val in enumerate(capacities[1:], start=1):
            if t_val - t_before >= delta_t:
                first_hit_index = idx
                break

        assert first_hit_index is not None
        assert len(move_rounds) == first_hit_index == 1

        _assert_strict_horizontal_helper_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

    def test_non_success_feasible_branch_must_use_cannot_solve_within_constraints_status(
        self,
    ) -> None:
        """
        If a feasible request is not completed, the helper must use the agreed status.

        This test does not force a particular geometry to trigger the branch. It
        instead locks the status semantics if the implementation ever reports a
        non-success despite the loose horizontal ceiling not ruling the request out.
        """
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0, 1, 0],
        )

        t_before: int = direct_cut_capacity(state, 0, 1)
        ceiling: int = _horizontal_ceiling(state, 0, 1)
        delta_t: int = max(0, ceiling - t_before)

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_horizontal(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                delta_T=delta_t,
                target_start_col=1,
                target_end_col=5,
            )
        )

        _assert_strict_horizontal_helper_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

        if status["kind"] != "success":
            assert status["kind"] == "cannot_solve_within_constraints"

def _make_state_from_rows(*rows: list[int]) -> np.ndarray:
    """Build a BCv2-style single-species state array from 2D row data."""
    arr_2d: np.ndarray = np.asarray(rows, dtype=np.uint8)
    if arr_2d.ndim != 2:
        raise ValueError("rows must define a 2D occupancy grid.")
    return arr_2d[:, :, None]


def _row_count(state: np.ndarray, row: int) -> int:
    """Return the atom count on one row."""
    return int(np.sum(state[row, :, 0], dtype=np.int64))


def _total_count(state: np.ndarray) -> int:
    """Return the total atom count in the full state."""
    return int(np.sum(state[:, :, 0], dtype=np.int64))


def _interval_bounds(
    boundary_dst_row: int,
    search_limit_row: int,
) -> tuple[int, int]:
    """Return inclusive row bounds for the allowed destination interval."""
    return min(boundary_dst_row, search_limit_row), max(boundary_dst_row, search_limit_row)


def _region_atom_count(
    state: np.ndarray,
    boundary_dst_row: int,
    search_limit_row: int,
) -> int:
    """Return the total atom count in the allowed destination interval."""
    low_row, high_row = _interval_bounds(boundary_dst_row, search_limit_row)
    return int(np.sum(state[low_row : high_row + 1, :, 0], dtype=np.int64))


def _destination_vacancy_upper_bound(
    state: np.ndarray,
    boundary_src_row: int,
    boundary_dst_row: int,
    search_limit_row: int,
) -> int:
    """
    Return the loose destination-side upper bound on additional cut capacity.

    Notes
    -----
    This is the agreed coarse precheck:
    total vacancies in the allowed destination interval minus current transfer capacity.
    """
    low_row, high_row = _interval_bounds(boundary_dst_row, search_limit_row)
    n_vacancies_region: int = int(
        np.sum(state[low_row : high_row + 1, :, 0] == 0, dtype=np.int64)
    )
    t_before: int = direct_cut_capacity(state, boundary_src_row, boundary_dst_row)
    return n_vacancies_region - t_before


def _replay_vertical_rounds(
    state: np.ndarray,
    move_rounds: list[list[Move]],
    boundary_src_row: int,
    boundary_dst_row: int,
    search_limit_row: int,
) -> np.ndarray:
    """
    Replay returned move rounds while checking vertical-helper legality.

    Notes
    -----
    Daisy-chain realizations are allowed so long as the joint noiseless
    application preserves atom count and all moves stay within the allowed
    destination interval.
    """
    work_state: np.ndarray = state.copy()
    low_row, high_row = _interval_bounds(boundary_dst_row, search_limit_row)

    round_index: int
    round_moves: list[Move]
    for round_index, round_moves in enumerate(move_rounds):
        sources_seen: set[tuple[int, int]] = set()
        dests_seen: set[tuple[int, int]] = set()

        move: Move
        for move in round_moves:
            from_row: int = int(move.from_row)
            to_row: int = int(move.to_row)
            from_col: int = int(move.from_col)
            to_col: int = int(move.to_col)

            assert low_row <= from_row <= high_row, (
                f"Round {round_index} contains move from row outside allowed "
                f"destination interval: {move}."
            )
            assert low_row <= to_row <= high_row, (
                f"Round {round_index} contains move to row outside allowed "
                f"destination interval: {move}."
            )
            assert abs(to_row - from_row) <= 1, (
                f"Round {round_index} contains non-adjacent-row move: {move}."
            )
            assert abs(to_col - from_col) <= 1, (
                f"Round {round_index} contains move violating local row-to-row "
                f"geometry |src_col - dst_col| <= 1: {move}."
            )

            src_site: tuple[int, int] = (from_row, from_col)
            dst_site: tuple[int, int] = (to_row, to_col)

            assert src_site not in sources_seen, (
                f"Round {round_index} reuses source site {src_site}."
            )
            assert dst_site not in dests_seen, (
                f"Round {round_index} reuses destination site {dst_site}."
            )

            sources_seen.add(src_site)
            dests_seen.add(dst_site)

            assert int(work_state[from_row, from_col, 0]) == 1, (
                f"Round {round_index} tries to move from an empty site at "
                f"{src_site}."
            )

        total_before: int = _total_count(work_state)
        region_before: int = _region_atom_count(
            work_state,
            boundary_dst_row=boundary_dst_row,
            search_limit_row=search_limit_row,
        )

        next_state: np.ndarray = move_atoms_noiseless(work_state.copy(), round_moves)

        total_after: int = _total_count(next_state)
        region_after: int = _region_atom_count(
            next_state,
            boundary_dst_row=boundary_dst_row,
            search_limit_row=search_limit_row,
        )

        assert total_after == total_before, (
            f"Round {round_index} lost or created atoms under noiseless realization."
        )
        assert region_after == region_before, (
            f"Round {round_index} changed destination-interval atom count under "
            f"noiseless realization."
        )

        work_state = next_state

    return work_state


def _assert_only_destination_interval_touched(
    before: np.ndarray,
    after: np.ndarray,
    boundary_dst_row: int,
    search_limit_row: int,
) -> None:
    """Assert that only the allowed destination interval changed."""
    low_row, high_row = _interval_bounds(boundary_dst_row, search_limit_row)
    changed_mask: np.ndarray = before[:, :, 0] != after[:, :, 0]
    changed_rows: np.ndarray = np.flatnonzero(np.any(changed_mask, axis=1))

    if changed_rows.size == 0:
        return

    actual_rows: list[int] = changed_rows.tolist()
    for row in actual_rows:
        assert low_row <= row <= high_row, (
            f"Helper changed row {row}, but only rows in the inclusive interval "
            f"[{low_row}, {high_row}] may be touched."
        )


def _assert_rows_outside_interval_unchanged(
    before: np.ndarray,
    after: np.ndarray,
    boundary_dst_row: int,
    search_limit_row: int,
) -> None:
    """Assert that rows outside the destination interval are exactly unchanged."""
    low_row, high_row = _interval_bounds(boundary_dst_row, search_limit_row)
    if low_row > 0:
        assert np.array_equal(after[:low_row, :, 0], before[:low_row, :, 0])
    if high_row + 1 < before.shape[0]:
        assert np.array_equal(after[high_row + 1 :, :, 0], before[high_row + 1 :, :, 0])


def _assert_no_move_crosses_into_source_row(
    move_rounds: list[list[Move]],
    boundary_src_row: int,
) -> None:
    """Assert that no returned move targets the source row."""
    round_moves: list[Move]
    for round_moves in move_rounds:
        move: Move
        for move in round_moves:
            assert int(move.to_row) != boundary_src_row, (
                f"Returned move illegally targets the source row: {move}."
            )


def _assert_strict_vertical_helper_invariants(
    before: np.ndarray,
    after: np.ndarray,
    move_rounds: list[list[Move]],
    boundary_src_row: int,
    boundary_dst_row: int,
    search_limit_row: int,
    achieved_delta_t: int,
    status: dict[str, int | str | None],
) -> None:
    """Assert the core helper invariants that should hold in every non-raising path."""
    replayed_state: np.ndarray = _replay_vertical_rounds(
        state=before,
        move_rounds=move_rounds,
        boundary_src_row=boundary_src_row,
        boundary_dst_row=boundary_dst_row,
        search_limit_row=search_limit_row,
    )
    assert np.array_equal(replayed_state, after)

    _assert_only_destination_interval_touched(
        before=before,
        after=after,
        boundary_dst_row=boundary_dst_row,
        search_limit_row=search_limit_row,
    )
    _assert_rows_outside_interval_unchanged(
        before=before,
        after=after,
        boundary_dst_row=boundary_dst_row,
        search_limit_row=search_limit_row,
    )
    _assert_no_move_crosses_into_source_row(
        move_rounds=move_rounds,
        boundary_src_row=boundary_src_row,
    )

    assert _total_count(after) == _total_count(before)
    assert _region_atom_count(after, boundary_dst_row, search_limit_row) == _region_atom_count(
        before, boundary_dst_row, search_limit_row
    )
    assert np.array_equal(after[boundary_src_row, :, 0], before[boundary_src_row, :, 0])

    t_before: int = direct_cut_capacity(before, boundary_src_row, boundary_dst_row)
    t_after: int = direct_cut_capacity(after, boundary_src_row, boundary_dst_row)

    assert achieved_delta_t == t_after - t_before
    assert int(status["T_before"]) == t_before
    assert int(status["T_after"]) == t_after
    assert int(status["achieved_delta_T"]) == achieved_delta_t
    assert int(status["n_rounds"]) == len(move_rounds)

    if str(status["kind"]) == "success":
        assert achieved_delta_t >= int(status["requested_delta_T"])


class TestENSURE_BOUNDARY_ROW_CUT_CAPACITY_VERTICAL:
    """
    Contract tests for vertical destination-side cut-capacity clearing.

    Written contract
    ----------------
    ``ensure_boundary_row_cut_capacity_vertical(...)`` is a destination-side
    helper that acts only within the inclusive interval between
    ``boundary_dst_row`` and ``search_limit_row``. It uses destination-side
    adjacent-row transfers with the same local row-to-row geometry as
    ``perform_transfer(...)`` (i.e. ``|src_col - dst_col| <= 1``) to increase
    exact direct cut capacity ``T`` by at least ``delta_T``.

    The helper:
    - may touch only rows in the allowed destination interval,
    - may not touch the source row or cross the cut,
    - must preserve both global atom count and the atom count inside the allowed
      destination interval,
    - is exact about achieved ``delta_T`` via recomputed
      ``direct_cut_capacity(...)``,
    - should reject impossible requests up front when the coarse destination-side
      upper bound ``(n_vacancies_in_interval - T_before)`` rules them out,
    - should not spend extra destination-side clearing rounds solely to continue
      improving ``T`` after first reaching the requested gain,
    - and should return one of:
        ``"success"``, ``"infeasible_request"``,
        ``"cannot_solve_within_constraints"``.
    """

    def test_returns_success_and_leaves_state_unchanged_when_delta_t_is_zero(self) -> None:
        """A zero request should succeed without executing any moves."""
        state: np.ndarray = _make_state_from_rows(
            [1, 1, 0, 1, 0],
            [1, 1, 1, 0, 1],
            [0, 0, 1, 0, 0],
        )

        t_before: int = direct_cut_capacity(state, 0, 1)

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_vertical(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                search_limit_row=2,
                delta_T=0,
            )
        )

        assert np.array_equal(new_state, state)
        assert move_rounds == []
        assert achieved_delta_t == 0

        assert status["kind"] == "success"
        assert status["T_before"] == t_before
        assert status["T_after"] == t_before
        assert status["requested_delta_T"] == 0
        assert status["achieved_delta_T"] == 0
        assert status["n_rounds"] == 0

    def test_raises_on_invalid_arguments(self) -> None:
        """Invalid shape, indices, and delta should fail loudly."""
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        )

        with pytest.raises(ValueError):
            ensure_boundary_row_cut_capacity_vertical(
                state=state[:, :, 0],
                boundary_src_row=0,
                boundary_dst_row=1,
                search_limit_row=2,
                delta_T=1,
            )

        with pytest.raises(ValueError):
            ensure_boundary_row_cut_capacity_vertical(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                search_limit_row=2,
                delta_T=-1,
            )

        with pytest.raises(IndexError):
            ensure_boundary_row_cut_capacity_vertical(
                state=state,
                boundary_src_row=-1,
                boundary_dst_row=1,
                search_limit_row=2,
                delta_T=1,
            )

        with pytest.raises(IndexError):
            ensure_boundary_row_cut_capacity_vertical(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=3,
                search_limit_row=2,
                delta_T=1,
            )

        with pytest.raises(IndexError):
            ensure_boundary_row_cut_capacity_vertical(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                search_limit_row=3,
                delta_T=1,
            )

        with pytest.raises(ValueError):
            ensure_boundary_row_cut_capacity_vertical(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                search_limit_row=1,
                delta_T=1,
            )

    def test_returns_infeasible_request_when_requested_gain_exceeds_destination_side_upper_bound(
        self,
    ) -> None:
        """The helper should reject impossible destination-side gain requests up front."""
        state: np.ndarray = _make_state_from_rows(
            [1, 1, 0, 1, 0],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1],
        )

        upper_bound: int = _destination_vacancy_upper_bound(state, 0, 1, 2)
        assert upper_bound <= 0

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_vertical(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                search_limit_row=2,
                delta_T=1,
            )
        )

        assert np.array_equal(new_state, state)
        assert move_rounds == []
        assert achieved_delta_t == 0
        assert status["kind"] == "infeasible_request"

    def test_positive_delta_is_treated_literally_even_if_t_before_is_already_large(self) -> None:
        """
        A positive request still means "gain more T", not "already sufficient".

        This locks the helper/controller boundary: the helper should not inject a
        controller-like notion of sufficiency.
        """
        state: np.ndarray = _make_state_from_rows(
            [0, 1, 0, 1, 0, 1],
            [0, 0, 1, 1, 1, 0],
            [1, 1, 0, 0, 0, 1],
        )

        t_before: int = direct_cut_capacity(state, 0, 1)
        assert t_before > 0

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_vertical(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                search_limit_row=2,
                delta_T=1,
            )
        )

        _assert_strict_vertical_helper_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            search_limit_row=2,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

    def test_source_row_is_never_touched(self) -> None:
        """The source row must remain unchanged."""
        state: np.ndarray = _make_state_from_rows(
            [0, 1, 0, 1, 0, 1],
            [1, 1, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0],
        )
        src_before: np.ndarray = state[0, :, 0].copy()

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_vertical(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                search_limit_row=2,
                delta_T=1,
            )
        )

        _assert_strict_vertical_helper_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            search_limit_row=2,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )
        assert np.array_equal(new_state[0, :, 0], src_before)

    def test_preserves_destination_interval_atom_count(self) -> None:
        """Destination-side clearing should preserve atom count in the allowed interval."""
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1, 0, 1, 0],
            [1, 1, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 1],
        )

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_vertical(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                search_limit_row=3,
                delta_T=1,
            )
        )

        _assert_strict_vertical_helper_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            search_limit_row=3,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

    def test_canonical_one_round_vertical_success(self) -> None:
        """
        A simple one-round destination-side clearing move should raise T.

        The boundary destination row has an atom that can be pushed one row deeper
        to create a new usable vacancy at the cut boundary.
        """
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1],
        )

        t_before: int = direct_cut_capacity(state, 0, 1)

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_vertical(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                search_limit_row=2,
                delta_T=1,
            )
        )

        assert status["kind"] == "success"
        assert len(move_rounds) == 1
        assert len(move_rounds[0]) > 0

        t_after: int = direct_cut_capacity(new_state, 0, 1)
        assert t_after - t_before >= 1

        _assert_strict_vertical_helper_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            search_limit_row=2,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

    def test_canonical_two_round_vertical_success(self) -> None:
        """
        A deeper jam should require exactly two rounds of destination-side clearing.
        """
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1],
        )

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_vertical(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                search_limit_row=3,
                delta_T=1,
            )
        )

        assert status["kind"] == "success"
        assert len(move_rounds) == 2

        _assert_strict_vertical_helper_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            search_limit_row=3,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

    def test_two_row_interval_edge_case(self) -> None:
        """
        When only one deeper row is available, the helper must solve or fail within that
        two-row destination interval without touching anything else.
        """
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1],
        )

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_vertical(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                search_limit_row=2,
                delta_T=1,
            )
        )

        _assert_strict_vertical_helper_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            search_limit_row=2,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )
        assert np.array_equal(new_state[3:, :, 0], state[3:, :, 0])

    def test_avoids_gratuitous_post_success_rounds(self) -> None:
        """Once the requested gain is first achieved, the helper should stop the episode."""
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1],
        )
        delta_t: int = 1

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_vertical(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                search_limit_row=2,
                delta_T=delta_t,
            )
        )

        assert status["kind"] == "success"

        capacities: list[int] = [direct_cut_capacity(state, 0, 1)]
        work_state: np.ndarray = state.copy()
        round_moves: list[Move]
        for round_moves in move_rounds:
            work_state = move_atoms_noiseless(work_state.copy(), round_moves)
            capacities.append(direct_cut_capacity(work_state, 0, 1))

        t_before: int = capacities[0]
        first_hit_index: int | None = None
        for idx, t_val in enumerate(capacities[1:], start=1):
            if t_val - t_before >= delta_t:
                first_hit_index = idx
                break

        assert first_hit_index is not None
        assert len(move_rounds) == first_hit_index

        _assert_strict_vertical_helper_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            search_limit_row=2,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

    def test_reverse_direction_success_case_has_positive_gain(self) -> None:
        """The helper should also work when the allowed interval extends upward in index."""
        state: np.ndarray = _make_state_from_rows(
            [0, 0, 1, 1, 0, 1],
            [0, 1, 1, 1, 1, 0],
            [1, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0],
        )

        t_before: int = direct_cut_capacity(state, 3, 2)

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_vertical(
                state=state,
                boundary_src_row=3,
                boundary_dst_row=2,
                search_limit_row=0,
                delta_T=1,
            )
        )

        assert status["kind"] == "success"
        t_after: int = direct_cut_capacity(new_state, 3, 2)
        assert t_after - t_before >= 1

        _assert_strict_vertical_helper_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=3,
            boundary_dst_row=2,
            search_limit_row=0,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
    def test_randomized_requests_not_ruled_out_by_coarse_precheck_preserve_invariants(
        self,
        seed: int,
    ) -> None:
        """
        Random requests not ruled out by the coarse destination-side precheck should
        satisfy the full invariant set.
        """
        rng: np.random.Generator = np.random.default_rng(seed)
        n_rows: int = 5
        n_cols: int = 10

        for _ in range(24):
            rows: list[np.ndarray] = [
                rng.integers(0, 2, size=n_cols, dtype=np.uint8) for _ in range(n_rows)
            ]
            state: np.ndarray = _make_state_from_rows(*[row.tolist() for row in rows])

            if bool(rng.integers(0, 2)):
                boundary_src_row = 0
                boundary_dst_row = 1
                search_limit_row = int(rng.integers(boundary_dst_row + 1, n_rows))
            else:
                boundary_src_row = n_rows - 1
                boundary_dst_row = n_rows - 2
                search_limit_row = int(rng.integers(0, boundary_dst_row))

            upper_bound: int = _destination_vacancy_upper_bound(
                state,
                boundary_src_row=boundary_src_row,
                boundary_dst_row=boundary_dst_row,
                search_limit_row=search_limit_row,
            )
            if upper_bound < 0:
                raise RuntimeError(
                    "Internal test construction bug: destination-side vacancy upper "
                    "bound cannot be negative."
                )

            delta_t: int = int(rng.integers(0, upper_bound + 1))

            new_state, move_rounds, achieved_delta_t, status = (
                ensure_boundary_row_cut_capacity_vertical(
                    state=state,
                    boundary_src_row=boundary_src_row,
                    boundary_dst_row=boundary_dst_row,
                    search_limit_row=search_limit_row,
                    delta_T=delta_t,
                )
            )

            _assert_strict_vertical_helper_invariants(
                before=state,
                after=new_state,
                move_rounds=move_rounds,
                boundary_src_row=boundary_src_row,
                boundary_dst_row=boundary_dst_row,
                search_limit_row=search_limit_row,
                achieved_delta_t=achieved_delta_t,
                status=status,
            )

    def test_returns_cannot_solve_within_constraints_when_coarse_bound_is_positive_but_vertical_path_jams(
        self,
    ) -> None:
        """
        Coarse vacancy availability is not enough: the helper can still be blocked by
        row-to-row geometry and should then report the constraint-limited status.

        This configuration is designed so that there are vacancies deeper in the
        destination interval, so the coarse bound is positive, but the boundary and
        next destination rows are geometrically jammed in a way that should surface
        a blocked destination row for the controller to hand off to the horizontal
        unblocker.
        """
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1],
        )

        upper_bound: int = _destination_vacancy_upper_bound(state, 0, 1, 3)
        assert upper_bound > 0

        new_state, move_rounds, achieved_delta_t, status = (
            ensure_boundary_row_cut_capacity_vertical(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                search_limit_row=3,
                delta_T=1,
            )
        )

        _assert_strict_vertical_helper_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            search_limit_row=3,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

        assert status["kind"] in {"success", "cannot_solve_within_constraints"}
        if status["kind"] != "success":
            assert status["kind"] == "cannot_solve_within_constraints"
            assert status["blocking_row"] is not None

def _replay_rounds(
    state: np.ndarray,
    move_rounds: list[list[Move]],
) -> np.ndarray:
    """Replay all returned rounds using move_atoms_noiseless."""
    work_state: np.ndarray = state.copy()

    round_index: int
    round_moves: list[Move]
    for round_index, round_moves in enumerate(move_rounds):
        sources_seen: set[tuple[int, int]] = set()
        dests_seen: set[tuple[int, int]] = set()

        move: Move
        for move in round_moves:
            src_site: tuple[int, int] = (int(move.from_row), int(move.from_col))
            dst_site: tuple[int, int] = (int(move.to_row), int(move.to_col))

            assert src_site not in sources_seen, (
                f"Round {round_index} reuses source site {src_site}."
            )
            assert dst_site not in dests_seen, (
                f"Round {round_index} reuses destination site {dst_site}."
            )
            sources_seen.add(src_site)
            dests_seen.add(dst_site)

            assert int(work_state[int(move.from_row), int(move.from_col), 0]) == 1, (
                f"Round {round_index} tries to move from an empty site {src_site}."
            )

        total_before: int = _total_count(work_state)
        next_state: np.ndarray = move_atoms_noiseless(work_state.copy(), round_moves)
        total_after: int = _total_count(next_state)

        assert total_after == total_before, (
            f"Round {round_index} lost or created atoms under noiseless realization."
        )
        work_state = next_state

    return work_state


def _assert_no_move_crosses_cut(
    move_rounds: list[list[Move]],
    boundary_src_row: int,
    boundary_dst_row: int,
) -> None:
    """Assert that the coordinator does not perform direct cross-cut transfer."""
    round_moves: list[Move]
    for round_moves in move_rounds:
        move: Move
        for move in round_moves:
            from_row: int = int(move.from_row)
            to_row: int = int(move.to_row)
            assert not (
                from_row == boundary_src_row and to_row == boundary_dst_row
            ), f"Coordinator illegally crossed the cut with move {move}."
            assert not (
                from_row == boundary_dst_row and to_row == boundary_src_row
            ), f"Coordinator illegally crossed the cut with move {move}."


def _assert_source_side_unchanged(
    before: np.ndarray,
    after: np.ndarray,
    boundary_src_row: int,
    boundary_dst_row: int,
) -> None:
    """Assert that all rows on the source side are unchanged."""
    if boundary_src_row < boundary_dst_row:
        if boundary_src_row >= 0:
            assert np.array_equal(
                after[: boundary_src_row + 1, :, 0],
                before[: boundary_src_row + 1, :, 0],
            )
    else:
        assert np.array_equal(
            after[boundary_src_row:, :, 0],
            before[boundary_src_row:, :, 0],
        )


def _assert_cut_capacity_coordinator_invariants(
    before: np.ndarray,
    after: np.ndarray,
    move_rounds: list[list[Move]],
    boundary_src_row: int,
    boundary_dst_row: int,
    achieved_delta_t: int,
    status: dict[str, int | str | bool | None],
) -> None:
    """Assert invariants that should hold for all non-raising coordinator returns."""
    replayed_state: np.ndarray = _replay_rounds(before, move_rounds)
    assert np.array_equal(replayed_state, after)

    assert _total_count(after) == _total_count(before)
    assert np.array_equal(after[boundary_src_row, :, 0], before[boundary_src_row, :, 0])
    _assert_source_side_unchanged(before, after, boundary_src_row, boundary_dst_row)
    _assert_no_move_crosses_cut(move_rounds, boundary_src_row, boundary_dst_row)

    t_before: int = direct_cut_capacity(before, boundary_src_row, boundary_dst_row)
    t_after: int = direct_cut_capacity(after, boundary_src_row, boundary_dst_row)

    assert achieved_delta_t == t_after - t_before
    assert int(status["T_before"]) == t_before
    assert int(status["T_after"]) == t_after
    assert int(status["achieved_delta_T"]) == achieved_delta_t
    assert int(status["n_rounds"]) == len(move_rounds)

    if str(status["kind"]) == "success":
        requested_delta_t: int = int(status["requested_delta_T"])
        assert achieved_delta_t >= requested_delta_t
        if requested_delta_t == 0:
            assert status["chosen_mode"] == "none"
        else:
            assert status["chosen_mode"] in {"horizontal", "vertical", "vertical_then_horizontal", "horizontal_then_vertical"}
    else:
        assert achieved_delta_t < int(status["requested_delta_T"])
        assert status["chosen_mode"] in {"horizontal", "vertical", "none", "vertical_then_horizontal", "horizontal_then_vertical"}
        if status["horizontal_status_kind"] is not None:
            assert status["horizontal_status_kind"] != "success"
        if status["vertical_status_kind"] is not None:
            assert status["vertical_status_kind"] != "success"

    if status["kind"] == "success":
        assert status["blocking_row"] is None
    elif status["vertical_status_kind"] == "infeasible_request":
        assert status["blocking_row"] is None


def _horizontal_coarse_cap(
    state: np.ndarray,
    boundary_src_row: int,
    boundary_dst_row: int,
) -> int:
    """Loose horizontal-only additional cut-capacity ceiling."""
    src_supply: int = int(np.sum(state[boundary_src_row, :, 0], dtype=np.int64))
    dst_vacancies: int = int(np.sum(state[boundary_dst_row, :, 0] == 0, dtype=np.int64))
    t_before: int = direct_cut_capacity(state, boundary_src_row, boundary_dst_row)
    return max(0, min(src_supply, dst_vacancies) - t_before)


def _vertical_coarse_cap(
    state: np.ndarray,
    boundary_src_row: int,
    boundary_dst_row: int,
    destination_search_limit_row: int,
) -> int:
    """Loose destination-side vertical-only additional cut-capacity ceiling."""
    low_row: int = min(boundary_dst_row, destination_search_limit_row)
    high_row: int = max(boundary_dst_row, destination_search_limit_row)
    n_vacancies_region: int = int(
        np.sum(state[low_row : high_row + 1, :, 0] == 0, dtype=np.int64)
    )
    t_before: int = direct_cut_capacity(state, boundary_src_row, boundary_dst_row)
    return max(0, n_vacancies_region - t_before)


class TestENSURE_CUT_CAPACITY:
    """
    Contract tests for the cut-capacity coordinator.

    Written contract
    ----------------
    ``ensure_cut_capacity(...)`` is a coordinator helper for the transfer-limited
    micro-objective. It does not source atoms from the source side and does not
    perform direct transfer across the cut. Instead, it tries to increase direct
    cut capacity ``T`` by at least ``delta_T`` by coordinating two lower-level
    helpers:

    1. boundary-row horizontal cut-capacity increase,
    2. destination-side vertical cut-capacity increase.

    First-version fallback policy
    -----------------------------
    - Try the horizontal helper first.
    - If horizontal succeeds, return immediately.
    - Otherwise try the vertical helper.
    - If vertical succeeds, return immediately.
    - If both fail, return structured non-success with metadata from both attempts.

    The coordinator:
    - is exact about achieved ``delta_T`` via recomputed
      ``direct_cut_capacity(...)``,
    - should not perform gratuitous extra work after the first successful helper,
    - should propagate vertical ``blocking_row`` metadata when vertical fails,
    - and should return one of:
        ``"success"``, ``"infeasible_request"``,
        ``"cannot_solve_within_constraints"``.
    """

    def test_returns_success_and_leaves_state_unchanged_when_delta_t_is_zero(self) -> None:
        """A zero request should succeed without executing any moves."""
        state: np.ndarray = _make_state_from_rows(
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
        )
        t_before: int = direct_cut_capacity(state, 0, 1)

        new_state, move_rounds, achieved_delta_t, status = ensure_cut_capacity(
            state=state,
            boundary_src_row=0,
            boundary_dst_row=1,
            destination_search_limit_row=2,
            delta_T=0,
            target_start_col=1,
            target_end_col=5,
        )

        assert np.array_equal(new_state, state)
        assert move_rounds == []
        assert achieved_delta_t == 0
        assert status["kind"] == "success"
        assert status["chosen_mode"] == "none"
        assert status["T_before"] == t_before
        assert status["T_after"] == t_before
        assert status["requested_delta_T"] == 0
        assert status["achieved_delta_T"] == 0
        assert status["n_rounds"] == 0

    def test_raises_on_invalid_arguments(self) -> None:
        """Invalid shape, indices, interval, and delta should fail loudly."""
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        )

        with pytest.raises(ValueError):
            ensure_cut_capacity(
                state=state[:, :, 0],
                boundary_src_row=0,
                boundary_dst_row=1,
                destination_search_limit_row=2,
                delta_T=1,
                target_start_col=0,
                target_end_col=2,
            )

        with pytest.raises(ValueError):
            ensure_cut_capacity(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                destination_search_limit_row=2,
                delta_T=-1,
                target_start_col=0,
                target_end_col=2,
            )

        with pytest.raises(IndexError):
            ensure_cut_capacity(
                state=state,
                boundary_src_row=-1,
                boundary_dst_row=1,
                destination_search_limit_row=2,
                delta_T=1,
                target_start_col=0,
                target_end_col=2,
            )

        with pytest.raises(IndexError):
            ensure_cut_capacity(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=3,
                destination_search_limit_row=2,
                delta_T=1,
                target_start_col=0,
                target_end_col=2,
            )

        with pytest.raises(IndexError):
            ensure_cut_capacity(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                destination_search_limit_row=3,
                delta_T=1,
                target_start_col=0,
                target_end_col=2,
            )

        with pytest.raises(ValueError):
            ensure_cut_capacity(
                state=state,
                boundary_src_row=0,
                boundary_dst_row=1,
                destination_search_limit_row=2,
                delta_T=1,
                target_start_col=2,
                target_end_col=1,
            )

    def test_current_fallback_policy_prefers_horizontal_when_both_are_plausible(self) -> None:
        """One explicit test locks the current fallback order for later comparison."""
        state: np.ndarray = _make_state_from_rows(
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
        )

        new_state, move_rounds, achieved_delta_t, status = ensure_cut_capacity(
            state=state,
            boundary_src_row=0,
            boundary_dst_row=1,
            destination_search_limit_row=2,
            delta_T=1,
            target_start_col=1,
            target_end_col=5,
        )

        assert status["kind"] == "success"
        assert status["chosen_mode"] == "horizontal"
        assert status["horizontal_status_kind"] == "success"
        assert status["vertical_status_kind"] is None

        _assert_cut_capacity_coordinator_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

    def test_horizontal_helper_parallelizes_chain_move(self) -> None:
        """A horizontally solvable case should parallelize moves when possible."""
        state: np.ndarray = _make_state_from_rows(
            [0, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 1],
        )

        new_state, move_rounds, achieved_delta_t, status = ensure_cut_capacity(
            state=state,
            boundary_src_row=0,
            boundary_dst_row=1,
            destination_search_limit_row=1,
            delta_T=1,
            target_start_col=1,
            target_end_col=5,
        )

        assert status["kind"] == "success"
        assert status["chosen_mode"] =="horizontal"
        assert len(move_rounds) == 1
        assert achieved_delta_t == 1
    
    
    def test_horizontal_helper_parallelizes_chain_move_in_two_rounds(self) -> None:
        """A horizontally solvable case should parallelize moves when possible."""
        state: np.ndarray = _make_state_from_rows(
            [0, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 1],
        )

        new_state, move_rounds, achieved_delta_t, status = ensure_cut_capacity(
            state=state,
            boundary_src_row=0,
            boundary_dst_row=1,
            destination_search_limit_row=1,
            delta_T=2,
            target_start_col=1,
            target_end_col=5,
        )

        assert status["kind"] == "success"
        assert status["chosen_mode"] =="horizontal"
        assert len(move_rounds) == 2
        assert achieved_delta_t == 2
    
    def test_horizontal_helper_parallelizes_chain_move_in_two_rounds_left_and_right_moving(self) -> None:
        """A horizontally solvable case should parallelize moves when possible, 
           even if atoms are moving in different directions."""
        state: np.ndarray = _make_state_from_rows(
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        )

        new_state, move_rounds, achieved_delta_t, status = ensure_cut_capacity(
            state=state,
            boundary_src_row=0,
            boundary_dst_row=1,
            destination_search_limit_row=1,
            delta_T=3,
            target_start_col=1,
            target_end_col=12,
        )

        assert status["kind"] == "success"
        assert status["chosen_mode"] =="horizontal"
        assert len(move_rounds) == 2
        assert achieved_delta_t == 3
    
    def test_horizontal_helper_parallelizes_chain_move_in_min_rounds_hard(self) -> None:
        """A horizontally solvable case should parallelize moves when possible, 
           even if atoms are moving in different directions."""
        state: np.ndarray = _make_state_from_rows(
            [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        )

        new_state, move_rounds, achieved_delta_t, status = ensure_cut_capacity(
            state=state,
            boundary_src_row=0,
            boundary_dst_row=1,
            destination_search_limit_row=1,
            delta_T=3,
            target_start_col=1,
            target_end_col=12,
        )

        assert status["kind"] == "success"
        assert status["chosen_mode"] =="horizontal"
        assert len(move_rounds) == 1
        assert achieved_delta_t == 3

    def test_succeeds_on_case_solved_by_horizontal_helper(self) -> None:
        """A horizontally solvable case should succeed and preserve coordinator invariants."""
        state: np.ndarray = _make_state_from_rows(
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
        )

        new_state, move_rounds, achieved_delta_t, status = ensure_cut_capacity(
            state=state,
            boundary_src_row=0,
            boundary_dst_row=1,
            destination_search_limit_row=2,
            delta_T=1,
            target_start_col=1,
            target_end_col=5,
        )

        assert status["kind"] == "success"
        assert status["chosen_mode"] in {"horizontal", "vertical"}
        assert len(move_rounds) == 1

        _assert_cut_capacity_coordinator_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

    def test_REGRESSION_succeeds_when_both_horiz_and_vert_clearing_needed(self) -> None:
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 0],
        )

        new_state, move_rounds, achieved_delta_t, status = ensure_cut_capacity(
            state=state,
            boundary_src_row=2,
            boundary_dst_row=1,
            destination_search_limit_row=0,
            delta_T=1,
            target_start_col=0,
            target_end_col=6,
        )

        assert status["kind"] == "success"
        assert status["chosen_mode"] in {"vertical_then_horizontal"}
        assert len(move_rounds) == 2 # something like [[Move(1,1,0,1)], [Move(1,2,1,1), Move(1,3,1,2)]]

        _assert_cut_capacity_coordinator_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=2,
            boundary_dst_row=1,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )
    
    def test_REGRESSION_succeeds_when_both_horiz_and_vert_clearing_needed_in_multiple_rounds(self) -> None:
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 0],
        )

        new_state, move_rounds, achieved_delta_t, status = ensure_cut_capacity(
            state=state,
            boundary_src_row=4,
            boundary_dst_row=3,
            destination_search_limit_row=0,
            delta_T=1,
            target_start_col=0,
            target_end_col=6,
        )

        assert status["kind"] == "success"
        assert status["chosen_mode"] in {"vertical_then_horizontal"}
        assert len(move_rounds) == 4

        _assert_cut_capacity_coordinator_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=4,
            boundary_dst_row=3,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )
    

    def test_succeeds_on_case_solved_by_vertical_helper_after_at_most_one_failed_attempt(self) -> None:
        """
        A vertically solvable case should succeed even if the coordinator first tries
        another mode that cannot complete the request.
        """
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1],
        )

        new_state, move_rounds, achieved_delta_t, status = ensure_cut_capacity(
            state=state,
            boundary_src_row=0,
            boundary_dst_row=1,
            destination_search_limit_row=1,
            delta_T=1,
            target_start_col=1,
            target_end_col=3,
        )

        assert status["kind"] == "success"
        assert status["chosen_mode"] in {"horizontal", "vertical"}

        _assert_cut_capacity_coordinator_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

    def test_when_both_modes_could_succeed_coordinator_commits_to_only_one(self) -> None:
        """The coordinator should not apply both helpers in a single episode."""
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1, 0, 1, 0],
            [1, 1, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
        )

        new_state, move_rounds, achieved_delta_t, status = ensure_cut_capacity(
            state=state,
            boundary_src_row=0,
            boundary_dst_row=1,
            destination_search_limit_row=3,
            delta_T=1,
            target_start_col=1,
            target_end_col=4,
        )

        assert status["kind"] == "success"
        assert status["chosen_mode"] in {"horizontal", "vertical"}
        if status["chosen_mode"] == "horizontal":
            assert status["horizontal_status_kind"] == "success"
            assert status["vertical_status_kind"] is None
        else:
            assert status["vertical_status_kind"] == "success"
            assert status["horizontal_status_kind"] in {
                None,
                "infeasible_request",
                "cannot_solve_within_constraints",
            }

        _assert_cut_capacity_coordinator_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=0,
            boundary_dst_row=1,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )


    def test_propagates_vertical_blocking_row_when_vertical_fails(self) -> None:
        """
        When vertical discovers a deeper relay bottleneck, the coordinator should
        propagate the blocking row so a later controller layer can call a horizontal
        unblocker there.
        """
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1],
        )

        _, _, _, status = ensure_cut_capacity(
            state=state,
            boundary_src_row=0,
            boundary_dst_row=1,
            destination_search_limit_row=3,
            delta_T=1,
            target_start_col=1,
            target_end_col=5,
        )

        if status["vertical_status_kind"] == "cannot_solve_within_constraints":
            assert status["blocking_row"] is not None

    def test_metadata_propagates_horizontal_flags_when_horizontal_is_used(self) -> None:
        """Horizontal-specific metadata should be preserved on horizontal success."""
        state: np.ndarray = _make_state_from_rows(
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
        )

        _, _, _, status = ensure_cut_capacity(
            state=state,
            boundary_src_row=0,
            boundary_dst_row=1,
            destination_search_limit_row=2,
            delta_T=1,
            target_start_col=1,
            target_end_col=4,
        )

        if status["chosen_mode"] == "horizontal":
            assert "used_outside_target_cols" in status

    def test_reverse_direction_vertical_fallback_case(self) -> None:
        """Explicit reverse-direction fallback/invariant case for readability."""
        state: np.ndarray = _make_state_from_rows(
            [0, 0, 0, 1, 1],
            [0, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
        )

        new_state, move_rounds, achieved_delta_t, status = ensure_cut_capacity(
            state=state,
            boundary_src_row=3,
            boundary_dst_row=2,
            destination_search_limit_row=0,
            delta_T=1,
            target_start_col=1,
            target_end_col=3,
        )

        _assert_cut_capacity_coordinator_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            boundary_src_row=3,
            boundary_dst_row=2,
            achieved_delta_t=achieved_delta_t,
            status=status,
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
    def test_randomized_requests_around_coarse_cap_preserve_invariants(self, seed: int) -> None:
        """
        Random requests sampled inside and slightly above a coarse combined cap
        should satisfy the full invariant set.
        """
        rng: np.random.Generator = np.random.default_rng(seed)
        n_rows: int = 5
        n_cols: int = 10

        for _ in range(28):
            rows: list[np.ndarray] = [
                rng.integers(0, 2, size=n_cols, dtype=np.uint8) for _ in range(n_rows)
            ]
            state: np.ndarray = _make_state_from_rows(*[row.tolist() for row in rows])

            if bool(rng.integers(0, 2)):
                boundary_src_row = 0
                boundary_dst_row = 1
                destination_search_limit_row = int(rng.integers(boundary_dst_row + 1, n_rows))
            else:
                boundary_src_row = n_rows - 1
                boundary_dst_row = n_rows - 2
                destination_search_limit_row = int(rng.integers(0, boundary_dst_row))

            h_cap: int = _horizontal_coarse_cap(
                state=state,
                boundary_src_row=boundary_src_row,
                boundary_dst_row=boundary_dst_row,
            )
            v_cap: int = _vertical_coarse_cap(
                state=state,
                boundary_src_row=boundary_src_row,
                boundary_dst_row=boundary_dst_row,
                destination_search_limit_row=destination_search_limit_row,
            )
            coarse_cap: int = max(h_cap, v_cap)

            delta_t: int = int(rng.integers(0, coarse_cap + 3))
            target_start_col: int = 1
            target_end_col: int = n_cols - 2

            new_state, move_rounds, achieved_delta_t, status = ensure_cut_capacity(
                state=state,
                boundary_src_row=boundary_src_row,
                boundary_dst_row=boundary_dst_row,
                destination_search_limit_row=destination_search_limit_row,
                delta_T=delta_t,
                target_start_col=target_start_col,
                target_end_col=target_end_col,
            )

            _assert_cut_capacity_coordinator_invariants(
                before=state,
                after=new_state,
                move_rounds=move_rounds,
                boundary_src_row=boundary_src_row,
                boundary_dst_row=boundary_dst_row,
                achieved_delta_t=achieved_delta_t,
                status=status,
            )


class TestENSURE_CUT_CAPACITY_WITH_STUBS:
    """Coordinator unit-style tests with stubbed helper results."""

    def test_stubbed_horizontal_success_short_circuits_vertical(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If horizontal succeeds, vertical should not be called."""
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
        )
        calls: dict[str, int] = {"horizontal": 0, "vertical": 0}
        horiz_state: np.ndarray = state.copy()
        horiz_state[1, 2, 0] = np.uint8(0)
        horiz_state[1, 1, 0] = np.uint8(1)

        def fake_horizontal(**kwargs):
            calls["horizontal"] += 1
            return (
                horiz_state,
                [[Move(1, 2, 1, 1)]],
                1,
                {
                    "kind": "success",
                    "T_before": 0,
                    "T_after": 1,
                    "requested_delta_T": 1,
                    "achieved_delta_T": 1,
                    "n_rounds": 1,
                    "used_outside_target_cols": False,
                },
            )

        def fake_vertical(**kwargs):
            calls["vertical"] += 1
            raise AssertionError("vertical should not be called after horizontal success")

        monkeypatch.setattr(
            helpers,
            "ensure_boundary_row_cut_capacity_horizontal",
            fake_horizontal,
        )
        monkeypatch.setattr(
            helpers,
            "ensure_boundary_row_cut_capacity_vertical",
            fake_vertical,
        )

        _, move_rounds, achieved_delta_t, status = ensure_cut_capacity(
            state=state,
            boundary_src_row=0,
            boundary_dst_row=1,
            destination_search_limit_row=2,
            delta_T=1,
            target_start_col=0,
            target_end_col=2,
        )

        assert calls["horizontal"] == 1
        assert calls["vertical"] == 0
        assert achieved_delta_t == 1
        assert status["kind"] == "success"
        assert status["chosen_mode"] == "horizontal"
        assert move_rounds == [[Move(1, 2, 1, 1)]]

    def test_stubbed_vertical_success_after_horizontal_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If horizontal fails and vertical succeeds, coordinator should return vertical success."""
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
        )
        calls: dict[str, int] = {"horizontal": 0, "vertical": 0}
        vert_state: np.ndarray = state.copy()
        vert_state[1, 0, 0] = np.uint8(0)
        vert_state[2, 0, 0] = np.uint8(1)

        def fake_horizontal(**kwargs):
            calls["horizontal"] += 1
            return (
                state.copy(),
                [],
                0,
                {
                    "kind": "cannot_solve_within_constraints",
                    "T_before": 0,
                    "T_after": 0,
                    "requested_delta_T": 1,
                    "achieved_delta_T": 0,
                    "n_rounds": 0,
                    "used_outside_target_cols": False,
                },
            )

        def fake_vertical(**kwargs):
            calls["vertical"] += 1
            return (
                vert_state,
                [[Move(1, 0, 2, 0)]],
                1,
                {
                    "kind": "success",
                    "T_before": 0,
                    "T_after": 1,
                    "requested_delta_T": 1,
                    "achieved_delta_T": 1,
                    "n_rounds": 1,
                    "blocking_row": None,
                },
            )

        monkeypatch.setattr(
            helpers,
            "ensure_boundary_row_cut_capacity_horizontal",
            fake_horizontal,
        )
        monkeypatch.setattr(
            helpers,
            "ensure_boundary_row_cut_capacity_vertical",
            fake_vertical,
        )

        _, _, achieved_delta_t, status = ensure_cut_capacity(
            state=state,
            boundary_src_row=0,
            boundary_dst_row=1,
            destination_search_limit_row=2,
            delta_T=1,
            target_start_col=0,
            target_end_col=2,
        )

        assert calls["horizontal"] == 1
        assert calls["vertical"] == 1
        assert achieved_delta_t == 1
        assert status["kind"] == "success"
        assert status["chosen_mode"] == "vertical"
        assert status["horizontal_status_kind"] == "cannot_solve_within_constraints"
        assert status["vertical_status_kind"] == "success"

    def test_stubbed_combined_failure_preserves_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If both subhelpers fail, coordinator should preserve both statuses and blocking_row."""
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
        )

        def fake_horizontal(**kwargs):
            return (
                state.copy(),
                [],
                0,
                {
                    "kind": "infeasible_request",
                    "T_before": 0,
                    "T_after": 0,
                    "requested_delta_T": 2,
                    "achieved_delta_T": 0,
                    "n_rounds": 0,
                    "used_outside_target_cols": False,
                },
            )

        def fake_vertical(**kwargs):
            return (
                state.copy(),
                [],
                0,
                {
                    "kind": "cannot_solve_within_constraints",
                    "T_before": 0,
                    "T_after": 0,
                    "requested_delta_T": 2,
                    "achieved_delta_T": 0,
                    "n_rounds": 0,
                    "blocking_row": 2,
                },
            )

        monkeypatch.setattr(
            helpers,
            "ensure_boundary_row_cut_capacity_horizontal",
            fake_horizontal,
        )
        monkeypatch.setattr(
            helpers,
            "ensure_boundary_row_cut_capacity_vertical",
            fake_vertical,
        )

        _, move_rounds, achieved_delta_t, status = ensure_cut_capacity(
            state=state,
            boundary_src_row=0,
            boundary_dst_row=1,
            destination_search_limit_row=2,
            delta_T=2,
            target_start_col=0,
            target_end_col=2,
        )

        assert move_rounds == []
        assert achieved_delta_t == 0
        assert status["kind"] == "cannot_solve_within_constraints"
        assert status["horizontal_status_kind"] == "infeasible_request"
        assert status["vertical_status_kind"] == "cannot_solve_within_constraints"
        assert status["blocking_row"] == 2

def _assert_controller_invariants(
    before: np.ndarray,
    after: np.ndarray,
    move_rounds: list[list[Move]],
    transferred: int,
    requested_R: int,
    status: dict[str, int | float | str | bool | None],
) -> None:
    replayed_state: np.ndarray = _replay_rounds(before, move_rounds)
    assert np.array_equal(replayed_state, after)
    assert _total_count(after) == _total_count(before)
    assert int(status["transferred"]) == transferred
    assert int(status["n_rounds"]) == len(move_rounds)
    assert 0 <= transferred <= requested_R
    assert int(status["remaining_R"]) == requested_R - transferred
    if str(status["kind"]) == "success":
        assert transferred > 0 or requested_R == 0
    else:
        assert str(status["last_bottleneck"]) in {"source", "cut_capacity", "none"}


class TestMOVE_ACROSS_ROWS:
    def test_zero_R_returns_success_without_moves(self) -> None:
        state: np.ndarray = _make_state_from_rows(
            [1, 1, 0, 1],
            [0, 0, 1, 0],
        )
        new_state, move_rounds, transferred, status = move_across_rows(
            state=state,
            boundary_src_row=0,
            boundary_dst_row=1,
            source_search_limit_row=0,
            destination_search_limit_row=1,
            R=0,
            C=0.3,
            target_start_col=0,
            target_end_col=3,
        )
        assert np.array_equal(new_state, state)
        assert move_rounds == []
        assert transferred == 0
        assert status["kind"] == "success"
        assert status["remaining_R"] == 0

    @pytest.mark.parametrize(
        ("C", "expected_batch_goal"),
        [
            (0.20, 2),
            (0.21, 3),
            (0.40, 4),
            (0.41, 5),
        ],
    )
    def test_batch_goal_rounding_edge_cases(self, C: float, expected_batch_goal: int) -> None:
        n_cols: int = 10
        computed: int = min(7, int(np.ceil(C * n_cols)))
        assert computed == expected_batch_goal

    def test_transfers_exactly_min_R_T_when_T_meets_batch_goal(self) -> None:
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 1, 1],
        )
        new_state, move_rounds, transferred, status = move_across_rows(
            state=state,
            boundary_src_row=0,
            boundary_dst_row=1,
            source_search_limit_row=0,
            destination_search_limit_row=1,
            R=2,
            C=0.3,
            target_start_col=0,
            target_end_col=5,
        )
        assert transferred == 2
        assert status["kind"] == "success"
        _assert_controller_invariants(
            before=state,
            after=new_state,
            move_rounds=move_rounds,
            transferred=transferred,
            requested_R=2,
            status=status,
        )

    def test_randomized_controller_runs_preserve_invariants(self) -> None:
        rng: np.random.Generator = np.random.default_rng(0)
        n_rows: int = 5
        n_cols: int = 10
        for _ in range(12):
            rows: list[np.ndarray] = [
                rng.integers(0, 2, size=n_cols, dtype=np.uint8) for _ in range(n_rows)
            ]
            state: np.ndarray = _make_state_from_rows(*[row.tolist() for row in rows])
            boundary_src_row: int = 0
            boundary_dst_row: int = 1
            R: int = int(rng.integers(0, 5))
            C: float = float(rng.choice(np.array([0.2, 0.3, 0.4], dtype=float)))
            new_state, move_rounds, transferred, status = move_across_rows(
                state=state,
                boundary_src_row=boundary_src_row,
                boundary_dst_row=boundary_dst_row,
                source_search_limit_row=0,
                destination_search_limit_row=4,
                R=R,
                C=C,
                target_start_col=1,
                target_end_col=n_cols - 2,
            )
            _assert_controller_invariants(
                before=state,
                after=new_state,
                move_rounds=move_rounds,
                transferred=transferred,
                requested_R=R,
                status=status,
            )

    @pytest.mark.parametrize('C', [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_REGRESSION_move_across_rows_recurses_for_source_shortage(self, C) -> None:
        state: np.ndarray = _make_state_from_rows(
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 0],
            [1, 1, 0, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        )

        new_state, move_rounds, transferred, status = move_across_rows(
            state=state,
            boundary_src_row=2,
            boundary_dst_row=3,
            source_search_limit_row=0,
            destination_search_limit_row=6,
            R=6,
            C=C,
            target_start_col=0,
            target_end_col=7,
        )

        assert transferred == 6

class TestMOVE_ACROSS_ROWS_WITH_STUBS:
    def test_cut_stall_short_circuits_without_calling_source_helper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state: np.ndarray = _make_state_from_rows(
            [1, 0, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
        )
        calls: dict[str, int] = {"source": 0, "cut": 0}

        def fake_source(**kwargs):
            calls["source"] += 1
            raise AssertionError("source helper should not be called in this stalled cut episode.")

        def fake_cut(**kwargs):
            calls["cut"] += 1
            return (
                state.copy(),
                [],
                0,
                {"kind": "cannot_solve_within_constraints", "achieved_delta_T": 0, "n_rounds": 0},
            )

        monkeypatch.setattr(helpers, "ensure_source_supply", fake_source)
        monkeypatch.setattr(helpers, "ensure_cut_capacity", fake_cut)

        _, _, transferred, status = move_across_rows(
            state=state,
            boundary_src_row=0,
            boundary_dst_row=1,
            source_search_limit_row=2,
            destination_search_limit_row=2,
            R=1,
            C=0.5,
            target_start_col=0,
            target_end_col=3,
        )
        assert calls["source"] == 0
        assert calls["cut"] == 1
        assert transferred == 0
        assert status["last_bottleneck"] == "cut_capacity"
