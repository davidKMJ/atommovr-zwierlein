from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Final, Any, Literal

from atommovr.utils.Move import Move
from atommovr.utils.move_utils import move_atoms_noiseless, get_AOD_cmds_from_move_list, get_move_list_from_AOD_cmds
from atommovr.utils.core import _int_sum

## Low-level helpers

def _as_2d_state(state: np.ndarray) -> np.ndarray:
    """
    Normalize BCv2 internal occupancy representation to a 2D (rows, cols) view.

    Why this exists
    ---------------
    BCv2 is logically single-species and most internal helpers reason about a
    2D occupancy grid. In the wider package, single-species matrices are often
    stored as (rows, cols, 1). This helper keeps the BCv2 internals robust to
    that representation without forcing the rest of the algorithm to care.
    """
    arr = np.asarray(state)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]
    raise ValueError(
        f"BCv2 expected 2D or (rows, cols, 1) single-species state; got shape {arr.shape}."
    )

## Helpers for compressing move lists


def _flatten_move_rounds(move_rounds: list[list[Move]]) -> list[Move]:
    """Return the concatenated move list for a list of rounds."""
    flat_moves: list[Move] = []
    single_round_moves: list[Move]
    for single_round_moves in move_rounds:
        flat_moves.extend(single_round_moves)
    return flat_moves


def _replay_rounds(
    state: np.ndarray,
    move_rounds: list[list[Move]],
) -> np.ndarray:
    """Replay a move schedule exactly as written."""
    work_state: np.ndarray = state.copy()
    round_moves: list[Move]
    for round_moves in move_rounds:
        work_state = move_atoms_noiseless(work_state.copy(), round_moves)
    return work_state


def _try_apply_round(
    state: np.ndarray,
    round_moves: list[Move],
) -> np.ndarray | None:
    """Attempt to apply one simultaneous round."""
    try:
        return move_atoms_noiseless(state.copy(), round_moves)
    except Exception:
        return None
    
def _has_unique_move_endpoints(round_moves: list[Move]) -> bool:
    """
    Return whether a simultaneous round uses unique sources and destinations.

    Why this exists
    ---------------
    The decoded AOD round is the physical artifact emitted by the compressor.
    Before accepting it, we want a cheap structural sanity check that rejects
    obvious simultaneous-round pathologies such as duplicate destinations or
    duplicate sources.

    Parameters
    ----------
    round_moves
        Candidate simultaneous move list.

    Returns
    -------
    bool
        ``True`` when every source site and every destination site appears at
        most once in the round, else ``False``.
    """
    seen_sources: set[tuple[int, int]] = set()
    seen_destinations: set[tuple[int, int]] = set()

    move: Move
    for move in round_moves:
        src: tuple[int, int] = (int(move.from_row), int(move.from_col))
        dst: tuple[int, int] = (int(move.to_row), int(move.to_col))

        if src in seen_sources or dst in seen_destinations:
            return False

        seen_sources.add(src)
        seen_destinations.add(dst)

    return True


def _try_decode_parallel_block(
    state: np.ndarray,
    original_round_block: list[list[Move]],
) -> list[Move] | None:
    """
    Decode one candidate original-round block into a single AOD-realized round.

    Why this exists
    ---------------
    The compressor should keep the original round structure for the reference
    replay, but the AOD encoder needs the union of all moves in the candidate
    block. This helper performs that encode/decode step and returns the decoded
    simultaneous round if the block is representable.

    Parameters
    ----------
    state
        Occupancy state at the start of the candidate block, with shape
        ``(rows, cols, 1)``.
    original_round_block
        Consecutive original rounds being proposed for merger.

    Returns
    -------
    list[Move] | None
        The decoded simultaneous round implied by one admissible AOD frame, or
        ``None`` if the candidate block is not safely AOD-parallelizable.
    """
    flat_moves: list[Move] = _flatten_move_rounds(original_round_block)
    matrix_2d: np.ndarray = state[:, :, 0]

    try:
        horiz_cmds: np.ndarray
        vert_cmds: np.ndarray
        parallel_success_flag: bool
        horiz_cmds, vert_cmds, parallel_success_flag = get_AOD_cmds_from_move_list(
            matrix=matrix_2d,
            move_seq=flat_moves,
            verify=False, #True
        )
    except Exception:
        return None

    if not parallel_success_flag:
        return None

    try:
        decoded_round: list[Move] = get_move_list_from_AOD_cmds(
            horiz_AOD_cmds=horiz_cmds,
            vert_AOD_cmds=vert_cmds,
        )
    except Exception:
        return None

    return decoded_round


def compress_move_rounds_conservative(
    state: np.ndarray,
    move_rounds: list[list[Move]],
) -> list[list[Move]]:
    """
    Greedily merge consecutive original rounds when one decoded AOD round is
    legal and replay-equivalent.

    Why this exists
    ---------------
    Some BCv2 controller paths emit serial schedules even when several adjacent
    rounds can be realized by one shared AOD frame. The physically relevant
    merged artifact is not the naive concatenation of the original moves, but
    the move list decoded from that shared AOD frame. At the same time, the
    *reference semantics* must preserve the original round structure, because
    some original rounds can contain legal same-round chains whose correctness
    would be lost by replaying a flattened concatenation sequentially.

    This helper therefore uses two representations of each candidate merge:

    1. a block of original rounds, kept with full round structure for the
       sequential reference replay, and
    2. a single decoded round obtained from the AOD encoder/decoder, which is
       emitted only if it is legal and yields the same end state.

    Parameters
    ----------
    state
        Initial single-species occupancy array with shape ``(rows, cols, 1)``.
    move_rounds
        Existing sequential move rounds.

    Returns
    -------
    list[list[Move]]
        Compressed decoded move rounds.

    Raises
    ------
    ValueError
        If ``state`` is not a single-species 3D occupancy array.
    RuntimeError
        If an original input round cannot be replayed from the current working
        state. That indicates the compressor was called on an invalid schedule.
    """
    if state.ndim != 3 or state.shape[2] != 1:
        raise ValueError(
            "state must have shape (rows, cols, 1) for BCv2 single-species helpers."
        )
    move_rounds = [list(r) for r in move_rounds if len(r) > 0]
    if len(move_rounds) == 0:
        return []

    work_state: np.ndarray = state.copy()
    compressed_rounds: list[list[Move]] = []

    candidate_block: list[list[Move]] = [list(move_rounds[0])]
    candidate_start_state: np.ndarray = work_state.copy()

    candidate_decoded_round: list[Move] = list(move_rounds[0])
    candidate_end_state: np.ndarray | None = _try_apply_round(
        state=candidate_start_state.copy(),
        round_moves=candidate_decoded_round,
    )
    if candidate_end_state is None:
        raise RuntimeError(
            "compress_move_rounds_conservative received an input round that cannot "
            "be replayed from the current state."
        )

    incoming_round: list[Move]
    for incoming_round in move_rounds[1:]:
        trial_block: list[list[Move]] = candidate_block + [list(incoming_round)]

        decoded_trial_round: list[Move] | None = _try_decode_parallel_block(
            state=candidate_start_state.copy(),
            original_round_block=trial_block,
        )

        merge_ok: bool = False
        decoded_trial_state: np.ndarray | None = None

        if decoded_trial_round is not None and _has_unique_move_endpoints(decoded_trial_round):
            sequential_trial_state: np.ndarray = _replay_rounds(
                state=candidate_start_state.copy(),
                move_rounds=trial_block,
            )
            decoded_trial_state = _try_apply_round(
                state=candidate_start_state.copy(),
                round_moves=decoded_trial_round,
            )
            merge_ok = (
                decoded_trial_state is not None
                and np.array_equal(decoded_trial_state, sequential_trial_state)
            )

        if merge_ok:
            candidate_block = trial_block
            candidate_decoded_round = decoded_trial_round  # type: ignore[assignment]
            candidate_end_state = decoded_trial_state
            continue

        compressed_rounds.append(candidate_decoded_round)
        work_state = candidate_end_state

        candidate_block = [list(incoming_round)]
        candidate_start_state = work_state.copy()
        candidate_decoded_round = list(incoming_round)
        candidate_end_state = _try_apply_round(
            state=candidate_start_state.copy(),
            round_moves=candidate_decoded_round,
        )
        if candidate_end_state is None:
            raise RuntimeError(
                "compress_move_rounds_conservative received an input round that "
                "cannot be replayed from the current state after committing the "
                "previous block."
            )

    compressed_rounds.append(candidate_decoded_round)
    return compressed_rounds

## Helpers for calculating S, R, and T

def source_supply_at_boundary(
    state: np.ndarray,
    boundary_src_row: int,
) -> int:
    """
    Return the number of atoms currently present on the boundary source row.

    Why this exists
    ---------------
    The transfer controller must distinguish a sourcing bottleneck from a
    blocking bottleneck. This helper defines the immediate source-side metric

        S = number of atoms on the boundary source row

    and intentionally does nothing more. It does not look deeper into the
    source region, and it does not ask whether those atoms are currently usable
    for a direct transfer.

    Parameters
    ----------
    state : np.ndarray
        Single-species occupancy array with shape ``(rows, cols, 1)``.
    boundary_src_row : int
        Row index of the source row adjacent to the transfer cut.

    Returns
    -------
    int
        Number of atoms on ``boundary_src_row``.

    Raises
    ------
    IndexError
        If ``boundary_src_row`` is out of bounds.
    ValueError
        If ``state`` is not a 3D single-species occupancy array.
    """
    if state.ndim != 3 or state.shape[2] != 1:
        raise ValueError(
            "state must have shape (rows, cols, 1) for BCv2 single-species helpers."
        )

    n_rows: int = int(state.shape[0])
    if boundary_src_row < 0 or boundary_src_row >= n_rows:
        raise IndexError(
            f"boundary_src_row={boundary_src_row} is out of bounds for {n_rows} rows."
        )

    return _int_sum(state[boundary_src_row, :, 0])


def direct_cut_capacity(
    state: np.ndarray,
    boundary_src_row: int,
    boundary_dst_row: int,
) -> int:
    """
    Return the exact immediate transfer capacity across the cut.

    Why this exists
    ---------------
    The controller needs the metric

        T = maximum number of atoms that can cross the cut in one direct round
            right now

    using only the current boundary source row and boundary destination row.
    This helper computes that quantity exactly under the local transfer rule
    ``|src_col - dst_col| <= 1``.

    A simple reachable-vacancy mask is not sufficient because direct transfer is
    a one-to-one matching problem: each source atom and each destination vacancy
    can be used at most once. The local structure makes the exact problem cheap:
    after processing column ``c``, the only unresolved objects that can still
    matter at column ``c + 1`` are the source atom at ``c`` and the destination
    vacancy at ``c``. That yields a four-state dynamic program.

    Parameters
    ----------
    state : np.ndarray
        Single-species occupancy array with shape ``(rows, cols, 1)``.
    boundary_src_row : int
        Source row adjacent to the cut.
    boundary_dst_row : int
        Destination row adjacent to the cut.

    Returns
    -------
    int
        Exact number of direct transfers currently possible across the cut.

    Raises
    ------
    IndexError
        If either boundary row index is out of bounds.
    ValueError
        If ``state`` is not a 3D single-species occupancy array.
    """
    if state.ndim != 3 or state.shape[2] != 1:
        raise ValueError(
            "state must have shape (rows, cols, 1) for BCv2 single-species helpers."
        )

    n_rows: int = int(state.shape[0])
    if boundary_src_row < 0 or boundary_src_row >= n_rows:
        raise IndexError(
            f"boundary_src_row={boundary_src_row} is out of bounds for {n_rows} rows."
        )
    if boundary_dst_row < 0 or boundary_dst_row >= n_rows:
        raise IndexError(
            f"boundary_dst_row={boundary_dst_row} is out of bounds for {n_rows} rows."
        )

    src_row: np.ndarray = state[boundary_src_row, :, 0].astype(np.bool_, copy=False)
    dst_vac: np.ndarray = state[boundary_dst_row, :, 0] == 0

    if not np.any(src_row) or not np.any(dst_vac):
        return 0

    n_cols: int = int(src_row.size)
    neg_inf: int = -10**9

    # Bit 0: unresolved source at column c remains available to c+1.
    # Bit 1: unresolved vacancy at column c remains available to c+1.
    scores: np.ndarray = np.full(4, neg_inf, dtype=np.int64)
    scores[0] = 0

    for c in range(n_cols):
        src_here: bool = bool(src_row[c])
        vac_here: bool = bool(dst_vac[c])

        next_scores: np.ndarray = np.full(4, neg_inf, dtype=np.int64)

        for state_bits in range(4):
            base_score: int = int(scores[state_bits])
            if base_score == neg_inf:
                continue

            prev_src_open: bool = bool(state_bits & 0b01)
            prev_vac_open: bool = bool(state_bits & 0b10)

            # Choice 0: no local match at this frontier.
            next_src_open: bool = src_here
            next_vac_open: bool = vac_here
            next_state_bits: int = int(next_src_open) | (int(next_vac_open) << 1)
            if base_score > int(next_scores[next_state_bits]):
                next_scores[next_state_bits] = base_score

            # Choice A: (c - 1) -> c
            if prev_src_open and vac_here:
                next_src_open = src_here
                next_vac_open = False
                next_state_bits = int(next_src_open) | (int(next_vac_open) << 1)
                if base_score + 1 > int(next_scores[next_state_bits]):
                    next_scores[next_state_bits] = base_score + 1

            # Choice B: c -> (c - 1)
            if src_here and prev_vac_open:
                next_src_open = False
                next_vac_open = vac_here
                next_state_bits = int(next_src_open) | (int(next_vac_open) << 1)
                if base_score + 1 > int(next_scores[next_state_bits]):
                    next_scores[next_state_bits] = base_score + 1

            # Choice C: c -> c
            if src_here and vac_here:
                next_src_open = False
                next_vac_open = False
                next_state_bits = int(next_src_open) | (int(next_vac_open) << 1)
                if base_score + 1 > int(next_scores[next_state_bits]):
                    next_scores[next_state_bits] = base_score + 1

            # Choice A + B:
            if prev_src_open and vac_here and src_here and prev_vac_open:
                next_src_open = False
                next_vac_open = False
                next_state_bits = int(next_src_open) | (int(next_vac_open) << 1)
                if base_score + 2 > int(next_scores[next_state_bits]):
                    next_scores[next_state_bits] = base_score + 2

        scores = next_scores

    return int(np.max(scores))


## Helpers for realizing a transfer
def perform_transfer(
    state: np.ndarray,
    remaining: int,
    boundary_src_row: int,
    boundary_dst_row: int,
) -> tuple[np.ndarray, list[Move], int]:
    """
    Execute the maximal direct transfer currently possible across the cut.

    Why this exists
    ---------------
    The BCv2 transfer controller separates transfer from support work. This
    helper performs only the direct cross-cut transfer that is available right
    now. It does not source atoms, clear space deeper in the destination
    region, or retry alternative routing plans.

    If the controller has decided that transfer is worthwhile, this helper must
    execute exactly

        min(remaining, T)

    direct transfers, where ``T`` is the current direct cut capacity.

    Parameters
    ----------
    state : np.ndarray
        Single-species occupancy array with shape ``(rows, cols, 1)``.
    remaining : int
        Number of atoms still needing to cross the cut.
    boundary_src_row : int
        Source row adjacent to the cut.
    boundary_dst_row : int
        Destination row adjacent to the cut.

    Returns
    -------
    tuple[np.ndarray, list[Move], int]
        ``(new_state, moves_run, n_moved)``.

    Raises
    ------
    ValueError
        If ``remaining`` is negative or ``state`` has the wrong shape.
    IndexError
        If either boundary row index is out of bounds.
    RuntimeError
        If the helper fails to realize exactly the direct transfer count that it
        claims to execute.
    """
    if state.ndim != 3 or state.shape[2] != 1:
        raise ValueError(
            "state must have shape (rows, cols, 1) for BCv2 single-species helpers."
        )
    if remaining < 0:
        raise ValueError(f"remaining must be nonnegative; got {remaining}.")

    n_rows: int = int(state.shape[0])
    if boundary_src_row < 0 or boundary_src_row >= n_rows:
        raise IndexError(
            f"boundary_src_row={boundary_src_row} is out of bounds for {n_rows} rows."
        )
    if boundary_dst_row < 0 or boundary_dst_row >= n_rows:
        raise IndexError(
            f"boundary_dst_row={boundary_dst_row} is out of bounds for {n_rows} rows."
        )

    if remaining == 0:
        return state.copy(), [], 0

    from_cols: np.ndarray
    to_cols: np.ndarray
    n_movable: int
    from_cols, to_cols, n_movable = get_all_moves_btwn_rows_cols(
        state,
        boundary_src_row,
        boundary_dst_row,
        n_transfer_needed=remaining,
    )

    if n_movable == 0:
        return state.copy(), [], 0

    n_run: int = min(int(remaining), int(n_movable))
    if n_run <= 0:
        return state.copy(), [], 0

    if len(from_cols) < n_run or len(to_cols) < n_run:
        raise RuntimeError(
            "get_all_moves_btwn_rows_cols returned an inconsistent transfer plan: "
            f"n_movable={n_movable}, len(from_cols)={len(from_cols)}, "
            f"len(to_cols)={len(to_cols)}, n_run={n_run}."
        )

    moves_run: list[Move] = [
        Move(
        boundary_src_row,
        int(src_col),
        boundary_dst_row,
        int(dst_col),
        )
        for src_col, dst_col in zip(from_cols[:n_run], to_cols[:n_run], strict=True)
    ]

    if len(moves_run) == 0:
        return state.copy(), [], 0

    new_state: np.ndarray = move_atoms_noiseless(state.copy(), moves_run)
    n_moved: int = len(moves_run)

    if n_moved != n_run:
        raise RuntimeError(
            "perform_transfer constructed the wrong number of direct moves: "
            f"expected {n_run}, built {n_moved}."
        )

    src_before: int = _int_sum(state[boundary_src_row, :, 0])
    src_after: int = _int_sum(new_state[boundary_src_row, :, 0])
    dst_before: int = _int_sum(state[boundary_dst_row, :, 0])
    dst_after: int = _int_sum(new_state[boundary_dst_row, :, 0])

    if (src_before - src_after) != n_moved:
        raise RuntimeError(
            "perform_transfer did not reduce source boundary occupancy by the "
            f"executed transfer count. Expected delta {n_moved}, got "
            f"{src_before - src_after}."
        )
    if (dst_after - dst_before) != n_moved:
        raise RuntimeError(
            "perform_transfer did not increase destination boundary occupancy by "
            f"the executed transfer count. Expected delta {n_moved}, got "
            f"{dst_after - dst_before}."
        )

    return new_state, moves_run, n_moved


def _score_edge(
    src_col: int,
    dst_col: int,
    vacancy_center_twice: int,
) -> tuple[int, int, int]:
    """
    Return the lexicographic score contribution of a single transfer edge.

    Why this exists
    ---------------
    The local row-to-row transfer problem has a clear primary objective:
    maximize the number of transferred atoms. Among equally large transfer sets,
    we prefer exact same-column transfers because they disturb the configuration
    least. As a final weak tie-break, we prefer filling vacancies closer to the
    center of the current vacancy distribution to reduce one-sided pile-up.

    Parameters
    ----------
    src_col
        Source column index.
    dst_col
        Destination column index.
    vacancy_center_twice
        Twice the arithmetic mean of the vacancy columns. Using the doubled value
        keeps the tie-break integer-valued.

    Returns
    -------
    tuple of int
        Lexicographic score contribution
        ``(n_transfers, n_same_column, centrality_score)``.
    """
    n_transfers: int = 1
    n_same_column: int = int(src_col == dst_col)
    centrality_score: int = -abs(2 * dst_col - vacancy_center_twice)
    return n_transfers, n_same_column, centrality_score


def _pure_mode_matches(
    from_row: np.ndarray,
    free_mask: np.ndarray,
    shift: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Compute a pure-mode transfer plan for a single adjacent-row shift pattern.

    Why this exists
    ---------------
    Many easy cases are solved by one simple transfer mode alone:
    exact same-column fill, pure left-shift fill, or pure right-shift fill.
    These are extremely cheap to evaluate with boolean logic and can be used as
    a fast path before falling back to the exact local matcher.

    Parameters
    ----------
    from_row
        Boolean/int occupancy mask of the source row with shape ``(n_cols,)``.
    free_mask
        Boolean vacancy mask of the destination row with shape ``(n_cols,)``.
    shift
        Destination shift relative to the source column.
        Allowed values are ``-1, 0, +1``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        ``(from_cols, to_cols, n_moves)`` for the pure mode.
    """
    if shift not in (-1, 0, 1):
        raise ValueError(f"shift must be -1, 0, or 1; got {shift}")

    n_cols: int = int(from_row.size)

    if shift == 0:
        match_mask: np.ndarray = from_row.astype(bool) & free_mask
        from_cols: np.ndarray = np.flatnonzero(match_mask).astype(np.intp, copy=False)
        to_cols: np.ndarray = from_cols.copy()
        return from_cols, to_cols, int(from_cols.size)

    if shift == -1:
        if n_cols <= 1:
            empty: np.ndarray = np.zeros(0, dtype=np.intp)
            return empty, empty, 0
        match_mask = from_row[1:].astype(bool) & free_mask[:-1]
        src_local: np.ndarray = np.flatnonzero(match_mask).astype(np.intp, copy=False)
        from_cols = src_local + 1
        to_cols = src_local
        return from_cols, to_cols, int(from_cols.size)

    # shift == +1
    if n_cols <= 1:
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty, 0
    match_mask = from_row[:-1].astype(bool) & free_mask[1:]
    src_local = np.flatnonzero(match_mask).astype(np.intp, copy=False)
    from_cols = src_local
    to_cols = src_local + 1
    return from_cols, to_cols, int(from_cols.size)

def _optimal_local_matches(
    from_row: np.ndarray,
    free_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Solve the exact local maximum-cardinality matching problem for ``|src - dst| <= 1``.

    Why this exists
    ---------------
    The cheap pure-mode scans catch easy cases, but mixed plans can be better:
    some same-column transfers, some left shifts, and some right shifts in the
    same round. Because each source column can connect only to destination
    vacancies at ``c - 1``, ``c``, or ``c + 1``, the exact matching problem has
    very small local structure and can be solved by a frontier dynamic program.

    This implementation optimizes only the primary objective:
    maximize the number of direct transfers. That is the quantity needed by the
    controller and by `perform_transfer(...)`. If we later want a secondary
    tie-break (for example preferring same-column transfers), that can be added
    after the cardinality path is stable and tested.

    Parameters
    ----------
    from_row
        Boolean/int source-row occupancy mask with shape ``(n_cols,)``.
    free_mask
        Boolean destination-row vacancy mask with shape ``(n_cols,)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        ``(from_cols, to_cols, n_moves)`` for an exact maximum-cardinality local
        transfer plan.

    Raises
    ------
    RuntimeError
        If internal DP bookkeeping becomes inconsistent.
    """
    n_cols: int = int(from_row.size)

    if n_cols != int(free_mask.size):
        raise RuntimeError(
            "from_row and free_mask must have the same length for local matching."
        )

    if n_cols == 0:
        empty: np.ndarray = np.zeros(0, dtype=np.intp)
        return empty, empty, 0

    # Frontier state bits after processing column c:
    #   bit 0 -> unresolved source at column c remains available for c + 1
    #   bit 1 -> unresolved vacancy at column c remains available for c + 1
    n_states: Final[int] = 4
    neg_inf: int = -10**9

    scores: np.ndarray = np.full(n_states, neg_inf, dtype=np.int64)
    scores[0] = 0

    prev_state_table: np.ndarray = np.full((n_cols, n_states), -1, dtype=np.int64)
    choice_table: list[list[tuple[tuple[int, int], ...] | None]] = [
        [None] * n_states for _ in range(n_cols)
    ]

    for c in range(n_cols):
        src_here: bool = bool(from_row[c])
        vac_here: bool = bool(free_mask[c])

        next_scores: np.ndarray = np.full(n_states, neg_inf, dtype=np.int64)

        for state_bits in range(n_states):
            base_score: int = int(scores[state_bits])
            if base_score == neg_inf:
                continue

            prev_src_open: bool = bool(state_bits & 0b01)
            prev_vac_open: bool = bool(state_bits & 0b10)

            # Valid local edge subsets at this frontier:
            #
            # A: (c - 1) -> c
            # B: c -> (c - 1)
            # C: c -> c
            #
            # The only valid two-edge subset is A + B.
            local_choices: tuple[tuple[tuple[int, int], ...], ...] = (
                (),
                (((c - 1), c),) if prev_src_open and vac_here else (),
                ((c, (c - 1)),) if src_here and prev_vac_open else (),
                ((c, c),) if src_here and vac_here else (),
                (((c - 1), c), (c, (c - 1)))
                if prev_src_open and vac_here and src_here and prev_vac_open
                else (),
            )

            for edges in local_choices:
                used_prev_src: bool = False
                used_src_here: bool = False
                used_prev_vac: bool = False
                used_vac_here: bool = False
                valid: bool = True

                for src_col, dst_col in edges:
                    if src_col == c - 1:
                        if not prev_src_open or used_prev_src:
                            valid = False
                            break
                        used_prev_src = True
                    elif src_col == c:
                        if not src_here or used_src_here:
                            valid = False
                            break
                        used_src_here = True
                    else:
                        valid = False
                        break

                    if dst_col == c - 1:
                        if not prev_vac_open or used_prev_vac:
                            valid = False
                            break
                        used_prev_vac = True
                    elif dst_col == c:
                        if not vac_here or used_vac_here:
                            valid = False
                            break
                        used_vac_here = True
                    else:
                        valid = False
                        break

                if not valid:
                    continue

                next_src_open: bool = src_here and (not used_src_here)
                next_vac_open: bool = vac_here and (not used_vac_here)
                next_state_bits: int = int(next_src_open) | (int(next_vac_open) << 1)

                cand_score: int = base_score + len(edges)
                if cand_score > int(next_scores[next_state_bits]):
                    next_scores[next_state_bits] = cand_score
                    prev_state_table[c, next_state_bits] = state_bits
                    choice_table[c][next_state_bits] = edges

        scores = next_scores

    end_state: int = int(np.argmax(scores))
    best_score: int = int(scores[end_state])

    if best_score == neg_inf:
        raise RuntimeError(
            "_optimal_local_matches found no valid terminal DP state. "
            "This indicates a frontier-transition bug."
        )

    if best_score == 0:
        empty: np.ndarray = np.zeros(0, dtype=np.intp)
        return empty, empty, 0

    edges_rev: list[tuple[int, int]] = []
    state_bits: int = end_state

    for c in range(n_cols - 1, -1, -1):
        edges = choice_table[c][state_bits]
        if edges is not None and len(edges) > 0:
            edges_rev.extend(edges)

        prev_state: int = int(prev_state_table[c, state_bits])
        if c > 0 and prev_state < 0:
            raise RuntimeError(
                "_optimal_local_matches backtracking failed due to missing predecessor state."
            )
        if prev_state >= 0:
            state_bits = prev_state

    edges_rev.reverse()

    from_cols: np.ndarray = np.asarray([src for src, _ in edges_rev], dtype=np.intp)
    to_cols: np.ndarray = np.asarray([dst for _, dst in edges_rev], dtype=np.intp)
    n_moves: int = int(from_cols.size)

    if to_cols.size != from_cols.size or n_moves != best_score:
        raise RuntimeError(
            "_optimal_local_matches returned inconsistent output sizes: "
            f"len(from_cols)={from_cols.size}, len(to_cols)={to_cols.size}, "
            f"best_score={best_score}."
        )

    return from_cols, to_cols, n_moves


def _vacancy_center_twice(free_mask: np.ndarray) -> int:
    """
    Return twice the arithmetic mean of destination vacancy columns.

    Why this exists
    ---------------
    When truncating a legal transfer plan, we want a deterministic policy that
    avoids the previous implicit left-prefix bias. We therefore prioritize moves
    whose destination columns are closest to the center of the current vacancy
    pattern.

    Using twice the arithmetic mean keeps all scoring integer-valued.
    """
    vacancy_cols: np.ndarray = np.flatnonzero(free_mask).astype(np.intp, copy=False)
    if vacancy_cols.size == 0:
        raise ValueError("free_mask must contain at least one vacancy.")
    return int(
        2 * np.sum(vacancy_cols, dtype=np.int64) // int(vacancy_cols.size)
    )


def _plan_truncation_order(
    from_cols: np.ndarray,
    to_cols: np.ndarray,
    vacancy_center_twice: int,
) -> list[int]:
    """
    Return deterministic truncation order indices for a legal transfer plan.

    Why this exists
    ---------------
    A bounded planner should not inherit an arbitrary prefix-order bias from the
    underlying exact solver. We instead choose a deterministic subset ordering
    that prefers:

    1. destination columns closest to the center of the vacancy pattern,
    2. smaller move distance,
    3. then a stable deterministic column order.

    Notes
    -----
    This removes the strong left-prefix bias from truncation. Perfectly
    symmetric ties still require some deterministic tie-break, so a tiny amount
    of directional asymmetry can remain in exact-tie cases.
    """
    indices: list[int] = list(range(int(from_cols.size)))
    indices.sort(
        key=lambda idx: (
            abs(2 * int(to_cols[idx]) - vacancy_center_twice),
            abs(int(to_cols[idx]) - int(from_cols[idx])),
            int(to_cols[idx]),
            int(from_cols[idx]),
        )
    )
    return indices


def _truncate_plan_by_destination_centrality(
    from_cols: np.ndarray,
    to_cols: np.ndarray,
    n_keep: int,
    free_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Truncate a legal transfer plan using deterministic destination-centrality priority.

    Parameters
    ----------
    from_cols
        Source columns of a legal one-to-one transfer plan.
    to_cols
        Destination columns of a legal one-to-one transfer plan.
    n_keep
        Number of moves to keep.
    free_mask
        Destination vacancy mask used to define centrality.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        ``(from_cols_kept, to_cols_kept, n_kept)``.
    """
    if n_keep < 0:
        raise ValueError(f"n_keep must be nonnegative; got {n_keep}.")
    if from_cols.size != to_cols.size:
        raise ValueError("from_cols and to_cols must have the same length.")

    n_available: int = int(from_cols.size)
    if n_keep == 0 or n_available == 0:
        empty: np.ndarray = np.zeros(0, dtype=np.intp)
        return empty, empty, 0
    if n_keep >= n_available:
        return from_cols, to_cols, n_available

    vacancy_center_twice: int = _vacancy_center_twice(free_mask)
    order: list[int] = _plan_truncation_order(
        from_cols=from_cols,
        to_cols=to_cols,
        vacancy_center_twice=vacancy_center_twice,
    )
    keep_idx: np.ndarray = np.asarray(order[:n_keep], dtype=np.intp)
    return from_cols[keep_idx], to_cols[keep_idx], int(keep_idx.size)


def _subset_score(
    from_cols: np.ndarray,
    to_cols: np.ndarray,
    vacancy_center_twice: int,
    mode_rank: int,
) -> tuple[int, int, int]:
    """
    Score a bounded transfer subset for deterministic pure-mode selection.

    Why this exists
    ---------------
    When several pure modes can satisfy a bounded request, we need a
    deterministic way to choose between them. We prioritize subsets whose
    destination columns are more central, then smaller move distance, then a
    fixed mode rank.
    """
    centrality_penalty: int = int(
        np.sum(np.abs(2 * to_cols.astype(np.int64) - vacancy_center_twice), dtype=np.int64)
    )
    move_distance_penalty: int = int(
        np.sum(np.abs(to_cols.astype(np.int64) - from_cols.astype(np.int64)), dtype=np.int64)
    )
    return centrality_penalty, move_distance_penalty, mode_rank


def get_all_moves_btwn_rows_cols(
    init_config: np.ndarray,
    from_row_ind: int,
    to_row_ind: int,
    n_transfer_needed: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Return a direct row-to-row transfer plan as integer column arrays.

    Why this exists
    ---------------
    This helper supports two distinct controller use-cases:

    1. ``n_transfer_needed == 0``:
       return the full exact direct-transfer plan.
    2. ``n_transfer_needed > 0``:
       return only enough direct transfers to satisfy that request, if possible.

    The bounded path first tries cheap pure modes (same-column only, all-left,
    all-right). If any pure mode already contains enough legal transfers, we do
    not compute the exact mixed-mode matcher; instead, we deterministically
    truncate the best pure candidate by destination centrality. We only pay for
    the exact local matcher when the best pure mode is insufficient.

    Parameters
    ----------
    init_config
        Single-species 3D occupancy array with shape ``(rows, cols, 1)``.
    from_row_ind
        Source row index.
    to_row_ind
        Destination row index.
    n_transfer_needed
        Requested number of direct transfers to plan. A value of ``0`` means
        "return the full exact plan."

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        ``(from_cols, to_cols, n_planned)``.

    Raises
    ------
    IndexError
        If either row index is negative.
    ValueError
        If ``init_config`` is not a 3D single-species occupancy array or if
        ``n_transfer_needed`` is negative.
    """
    if from_row_ind < 0 or to_row_ind < 0:
        raise IndexError

    if init_config.ndim != 3:
        raise ValueError(f"init_config must be 3D; got ndim={init_config.ndim}")
    if init_config.shape[2] != 1:
        raise ValueError(
            f"init_config must have shape (rows, cols, 1); got shape {init_config.shape}."
        )
    if n_transfer_needed < 0:
        raise ValueError(
            f"n_transfer_needed must be nonnegative; got {n_transfer_needed}."
        )

    from_row: np.ndarray = init_config[from_row_ind, :, 0]
    to_row: np.ndarray = init_config[to_row_ind, :, 0]

    src_count: int = int(np.sum(from_row, dtype=np.int64))
    if src_count == 0:
        empty: np.ndarray = np.zeros(0, dtype=np.intp)
        return empty, empty, 0

    free_mask: np.ndarray = to_row == 0
    vac_count: int = int(np.sum(free_mask, dtype=np.int64))
    if vac_count == 0:
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty, 0

    max_possible: int = min(src_count, vac_count)
    n_target: int = max_possible if n_transfer_needed == 0 else min(max_possible, n_transfer_needed)
    vacancy_center_twice: int = _vacancy_center_twice(free_mask)

    pure_modes: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    mode_rank: int
    shift: int
    for mode_rank, shift in enumerate((0, -1, 1)):
        pure_from: np.ndarray
        pure_to: np.ndarray
        pure_n: int
        pure_from, pure_to, pure_n = _pure_mode_matches(from_row, free_mask, shift=shift)
        pure_modes.append((pure_from, pure_to, pure_n, mode_rank))

    # Fast path:
    # - bounded request: if any pure mode already contains enough moves, use the
    #   best pure candidate and truncate it centrally.
    # - unbounded request: if any pure mode saturates max_possible, return the
    #   best full pure candidate.
    viable_pure_candidates: list[
        tuple[tuple[int, int, int], np.ndarray, np.ndarray]
    ] = []
    pure_from: np.ndarray
    pure_to: np.ndarray
    pure_n: int
    mode_rank = 0
    for pure_from, pure_to, pure_n, mode_rank in pure_modes:
        if pure_n < n_target:
            continue
        chosen_from: np.ndarray
        chosen_to: np.ndarray
        n_chosen: int
        chosen_from, chosen_to, n_chosen = _truncate_plan_by_destination_centrality(
            from_cols=pure_from,
            to_cols=pure_to,
            n_keep=n_target,
            free_mask=free_mask,
        )
        score: tuple[int, int, int] = _subset_score(
            from_cols=chosen_from,
            to_cols=chosen_to,
            vacancy_center_twice=vacancy_center_twice,
            mode_rank=mode_rank,
        )
        viable_pure_candidates.append((score, chosen_from, chosen_to))

    if len(viable_pure_candidates) > 0:
        viable_pure_candidates.sort(key=lambda item: item[0])
        best_from: np.ndarray = viable_pure_candidates[0][1]
        best_to: np.ndarray = viable_pure_candidates[0][2]
        return best_from, best_to, int(best_from.size)

    exact_from: np.ndarray
    exact_to: np.ndarray
    exact_n: int
    exact_from, exact_to, exact_n = _optimal_local_matches(from_row, free_mask)

    if n_transfer_needed == 0:
        return exact_from, exact_to, exact_n

    chosen_from, chosen_to, n_chosen = _truncate_plan_by_destination_centrality(
        from_cols=exact_from,
        to_cols=exact_to,
        n_keep=n_target,
        free_mask=free_mask,
    )
    return chosen_from, chosen_to, n_chosen

## 

def _status_dict(
    kind: str,
    blocking_row: int | None,
    unmet_delta_S: int,
) -> dict[str, int | str | None]:
    """Build the structured status payload used by ensure_source_supply."""
    return {
        "kind": kind,
        "blocking_row": blocking_row,
        "unmet_delta_S": int(unmet_delta_S),
    }


def _count_row_atoms(state: np.ndarray, row: int) -> int:
    """Return the occupancy count on one row."""
    return _int_sum(state[row, :, 0])


def _interval_bounds(
    boundary_src_row: int,
    search_limit_row: int,
) -> tuple[int, int, int]:
    """
    Return ``(low_row, high_row, step_toward_boundary)`` for the allowed interval.

    Notes
    -----
    ``step_toward_boundary`` is ``+1`` when rows move upward in index toward the
    boundary and ``-1`` when they move downward in index toward the boundary.
    """
    low_row: int = min(boundary_src_row, search_limit_row)
    high_row: int = max(boundary_src_row, search_limit_row)
    step_toward_boundary: int = 1 if boundary_src_row > search_limit_row else -1
    return low_row, high_row, step_toward_boundary


def _vacancy_center_twice_from_row(target_row: np.ndarray) -> int:
    """Return twice the arithmetic mean of vacancy columns on a target row."""
    vacancy_cols: np.ndarray = np.flatnonzero(target_row == 0).astype(np.intp, copy=False)
    if vacancy_cols.size == 0:
        raise ValueError("target_row must contain at least one vacancy.")
    return int(2 * np.sum(vacancy_cols, dtype=np.int64) // int(vacancy_cols.size))


def _select_same_column_transfer_cols(
    state: np.ndarray,
    from_row: int,
    to_row: int,
    cap: int | None,
) -> np.ndarray:
    """
    Return same-column transfer columns, optionally truncated by destination centrality.

    Notes
    -----
    Only columns with an atom on ``from_row`` and a vacancy on ``to_row`` are
    eligible. If ``cap`` is provided, the returned subset is truncated
    deterministically by proximity to the center of the destination vacancy
    pattern, then by column index.
    """
    source_mask: np.ndarray = state[from_row, :, 0].astype(np.bool_, copy=False)
    target_vac_mask: np.ndarray = state[to_row, :, 0] == 0
    movable_cols: np.ndarray = np.flatnonzero(source_mask & target_vac_mask).astype(
        np.intp, copy=False
    )

    if cap is None or movable_cols.size <= int(cap):
        return movable_cols

    center_twice: int = _vacancy_center_twice_from_row(state[to_row, :, 0])
    ordered: list[int] = movable_cols.tolist()
    ordered.sort(
        key=lambda col: (
            abs(2 * int(col) - center_twice),
            int(col),
        )
    )
    return np.asarray(ordered[: int(cap)], dtype=np.intp)


def _apply_vertical_round(
    state: np.ndarray,
    from_row: int,
    to_row: int,
    cap: int | None,
) -> tuple[np.ndarray, list[Move]]:
    """
    Apply one same-column vertical relay round from ``from_row`` to ``to_row``.

    Notes
    -----
    This helper is intentionally vertical-only: every returned move satisfies
    ``move.to_col == move.from_col``.
    """
    cols: np.ndarray = _select_same_column_transfer_cols(
        state=state,
        from_row=from_row,
        to_row=to_row,
        cap=cap,
    )
    if cols.size == 0:
        return state.copy(), []

    moves: list[Move] = [
        Move(int(from_row), int(col), int(to_row), int(col))
        for col in cols.tolist()
    ]
    new_state: np.ndarray = move_atoms_noiseless(state.copy(), moves)
    return new_state, moves


def _relay_exact_to_target(
    state: np.ndarray,
    target_row: int,
    search_limit_row: int,
    delta_needed: int,
    step_toward_boundary: int,
) -> tuple[np.ndarray, list[list[Move]], int, dict[str, int | str | None]]:
    """
    Recursively source atoms into ``target_row`` using exact-mode semantics.

    Why this exists
    ---------------
    ``ensure_source_supply(..., fill_mode="exact")`` needs a narrow relay routine
    that only moves atoms one row closer to the boundary at a time, never
    overfills the requested increase intentionally, and returns structured
    blocked status when it cannot complete the request after the top-level
    global insufficiency precheck has passed.
    """
    if delta_needed < 0:
        raise ValueError(f"delta_needed must be nonnegative; got {delta_needed}.")
    if delta_needed == 0:
        return (
            state.copy(),
            [],
            0,
            _status_dict("success", None, 0),
        )

    low_row: int
    high_row: int
    low_row, high_row, _ = _interval_bounds(target_row, search_limit_row)
    upstream_row: int = target_row - step_toward_boundary

    if upstream_row < low_row or upstream_row > high_row:
        return (
            state.copy(),
            [],
            0,
            _status_dict("partial_blocked", target_row, delta_needed),
        )

    work_state: np.ndarray = state.copy()
    move_rounds: list[list[Move]] = []
    achieved_delta: int = 0

    # First, take any immediately available same-column transfers into the target.
    work_state, direct_moves = _apply_vertical_round(
        state=work_state,
        from_row=upstream_row,
        to_row=target_row,
        cap=delta_needed,
    )
    if len(direct_moves) > 0:
        move_rounds.append(direct_moves)
        achieved_delta += len(direct_moves)

    remaining_need: int = int(delta_needed - achieved_delta)
    if remaining_need == 0:
        return (
            work_state,
            move_rounds,
            achieved_delta,
            _status_dict("success", None, 0),
        )

    upstream_supply: int = _count_row_atoms(work_state, upstream_row)
    if upstream_supply >= remaining_need:
        # Enough atoms are sitting one row away, but they cannot enter the target.
        return (
            work_state,
            move_rounds,
            achieved_delta,
            _status_dict("partial_blocked", target_row, remaining_need),
        )

    next_upstream_row: int = upstream_row - step_toward_boundary
    if next_upstream_row < low_row or next_upstream_row > high_row:
        return (
            work_state,
            move_rounds,
            achieved_delta,
            _status_dict("partial_blocked", upstream_row, remaining_need),
        )

    deficit_into_upstream: int = int(remaining_need - upstream_supply)

    # Recursively source enough atoms into the upstream row.
    work_state, deeper_rounds, _, deeper_status = _relay_exact_to_target(
        state=work_state,
        target_row=upstream_row,
        search_limit_row=search_limit_row,
        delta_needed=deficit_into_upstream,
        step_toward_boundary=step_toward_boundary,
    )
    move_rounds.extend(deeper_rounds)

    # After deeper sourcing, try again to feed the original target.
    work_state, second_direct_moves = _apply_vertical_round(
        state=work_state,
        from_row=upstream_row,
        to_row=target_row,
        cap=remaining_need,
    )
    if len(second_direct_moves) > 0:
        move_rounds.append(second_direct_moves)
        achieved_delta += len(second_direct_moves)

    remaining_need = int(delta_needed - achieved_delta)
    if remaining_need == 0:
        return (
            work_state,
            move_rounds,
            achieved_delta,
            _status_dict("success", None, 0),
        )

    upstream_supply_now: int = _count_row_atoms(work_state, upstream_row)
    if upstream_supply_now >= remaining_need:
        return (
            work_state,
            move_rounds,
            achieved_delta,
            _status_dict("partial_blocked", target_row, remaining_need),
        )

    blocking_row_raw: int | str | None = deeper_status["blocking_row"]
    blocking_row: int | None = None if blocking_row_raw is None else int(blocking_row_raw)
    if blocking_row is None:
        blocking_row = upstream_row

    return (
        work_state,
        move_rounds,
        achieved_delta,
        _status_dict("partial_blocked", blocking_row, remaining_need),
    )

def relay_exact(
    work_state: np.ndarray,
    target_row: int,
    delta_needed: int,
    step_toward_boundary: int,
    low_row: int,
    high_row: int
    ) -> tuple[np.ndarray, list[list[Move]], int, dict[str, int | str | None]]:
    if delta_needed == 0:
        return work_state.copy(), [], 0, _status_dict("success", None, 0)

    upstream_row: int = target_row - step_toward_boundary
    if upstream_row < low_row or upstream_row > high_row:
        return work_state.copy(), [], 0, _status_dict("partial_blocked", target_row, delta_needed)

    moved_state, moved_now_round, moved_now = perform_transfer(
        work_state,
        remaining=delta_needed,
        boundary_src_row=upstream_row,
        boundary_dst_row=target_row,
    )
    rounds: list[list[Move]] = []
    achieved: int = 0
    if moved_now > 0:
        rounds.append(moved_now_round)
        achieved += moved_now

    remaining_need: int = int(delta_needed - achieved)
    if remaining_need == 0:
        return moved_state, rounds, achieved, _status_dict("success", None, 0)

    upstream_supply: int = _count_row_atoms(moved_state, upstream_row)
    if upstream_supply >= remaining_need:
        return moved_state, rounds, achieved, _status_dict("partial_blocked", target_row, remaining_need)

    next_upstream_row: int = upstream_row - step_toward_boundary
    if next_upstream_row < low_row or next_upstream_row > high_row:
        return moved_state, rounds, achieved, _status_dict("partial_blocked", upstream_row, remaining_need)

    deficit_into_upstream: int = int(remaining_need - upstream_supply)
    moved_state, deeper_rounds, _, deeper_status = relay_exact(
        moved_state,
        upstream_row,
        deficit_into_upstream,
        step_toward_boundary=step_toward_boundary,
        low_row=low_row,
        high_row=high_row,
    )
    rounds.extend(deeper_rounds)

    moved_state, second_round, second_moved = perform_transfer(
        moved_state,
        remaining=remaining_need,
        boundary_src_row=upstream_row,
        boundary_dst_row=target_row,
    )
    if second_moved > 0:
        rounds.append(second_round)
        achieved += second_moved

    remaining_need = int(delta_needed - achieved)
    if remaining_need == 0:
        return moved_state, rounds, achieved, _status_dict("success", None, 0)

    upstream_supply_now: int = _count_row_atoms(moved_state, upstream_row)
    if upstream_supply_now >= remaining_need:
        return moved_state, rounds, achieved, _status_dict("partial_blocked", target_row, remaining_need)

    blocking_row_raw: int | str | None = deeper_status["blocking_row"]
    blocking_row: int | None = None if blocking_row_raw is None else int(blocking_row_raw)
    if blocking_row is None:
        blocking_row = upstream_row
    return moved_state, rounds, achieved, _status_dict("partial_blocked", blocking_row, remaining_need)


def ensure_source_supply(
    state: np.ndarray,
    boundary_src_row: int,
    search_limit_row: int,
    delta_S: int,
    fill_mode: Literal["exact", "opportunistic"] = "opportunistic",
) -> tuple[np.ndarray, list[list[Move]], int, dict[str, int | str | None]]:
    """
    Increase boundary-row source supply using only source-side adjacent-row relay moves.

    Notes
    -----
    This helper uses ``perform_transfer(...)`` as its adjacent-row relay
    primitive. Therefore source-side support moves are allowed to use the same
    local row-to-row geometry as direct transfer planning, including diagonal
    column shifts with ``|to_col - from_col| <= 1``. The helper may touch only
    rows in the inclusive interval between ``boundary_src_row`` and
    ``search_limit_row`` and never crosses the cut.

    ``"insufficient_atoms"`` is determined only by the global upstream atom
    count in the allowed interval. After that precheck passes, any failure to
    complete is reported as ``"partial_blocked"``.
    """
    if state.ndim != 3 or state.shape[2] != 1:
        raise ValueError(
            "state must have shape (rows, cols, 1) for BCv2 single-species helpers."
        )
    if delta_S < 0:
        raise ValueError(f"delta_S must be nonnegative; got {delta_S}.")
    if fill_mode not in {"exact", "opportunistic"}:
        raise ValueError(
            f"fill_mode must be 'exact' or 'opportunistic'; got {fill_mode!r}."
        )

    n_rows: int = int(state.shape[0])
    if boundary_src_row < 0 or boundary_src_row >= n_rows:
        raise IndexError(
            f"boundary_src_row={boundary_src_row} is out of bounds for {n_rows} rows."
        )
    if search_limit_row < 0 or search_limit_row >= n_rows:
        raise IndexError(
            f"search_limit_row={search_limit_row} is out of bounds for {n_rows} rows."
        )
    if boundary_src_row == search_limit_row:
        raise ValueError(
            "ensure_source_supply requires a nontrivial source interval: "
            "search_limit_row must differ from boundary_src_row."
        )

    initial_state: np.ndarray = state.copy()
    initial_boundary_supply: int = source_supply_at_boundary(initial_state, boundary_src_row)

    if delta_S == 0:
        return initial_state, [], 0, _status_dict("success", None, 0)

    low_row: int = min(boundary_src_row, search_limit_row)
    high_row: int = max(boundary_src_row, search_limit_row)
    step_toward_boundary: int = 1 if boundary_src_row > search_limit_row else -1

    global_upstream_atoms: int = 0
    for row in range(low_row, high_row + 1):
        if row != boundary_src_row:
            global_upstream_atoms += _count_row_atoms(initial_state, row)
    if global_upstream_atoms < delta_S:
        return initial_state, [], 0, _status_dict("insufficient_atoms", None, delta_S)

    
    if fill_mode == "exact":
        new_state, move_rounds, achieved_delta_S, status = relay_exact(
            initial_state,
            boundary_src_row,
            delta_S,
            step_toward_boundary=step_toward_boundary,
            low_row=low_row,
            high_row=high_row,
        )
        actual_delta: int = source_supply_at_boundary(new_state, boundary_src_row) - initial_boundary_supply
        if actual_delta != achieved_delta_S:
            raise RuntimeError(
                "ensure_source_supply exact-mode bookkeeping mismatch: "
                f"reported achieved_delta_S={achieved_delta_S}, actual boundary delta={actual_delta}."
            )
        status["unmet_delta_S"] = max(0, int(delta_S - achieved_delta_S))
        if achieved_delta_S >= delta_S:
            status["kind"] = "success"
            status["blocking_row"] = None
        return new_state, move_rounds, achieved_delta_S, status

    exact_state, exact_rounds, exact_achieved, exact_status = relay_exact(
        initial_state,
        boundary_src_row,
        delta_S,
        step_toward_boundary=step_toward_boundary,
        low_row=low_row,
        high_row=high_row,
    )

    opportunistic_state: np.ndarray = initial_state.copy()
    opportunistic_rounds: list[list[Move]] = []

    planned_interfaces: list[tuple[int, int]] = []
    for round_moves in exact_rounds:
        if len(round_moves) == 0:
            raise RuntimeError("Exact relay plan contains an empty move round.")
        planned_interfaces.append((int(round_moves[0].from_row), int(round_moves[0].to_row)))

    n_cols: int = int(state.shape[1])
    for from_row_i, to_row_i in planned_interfaces:
        opportunistic_state, opp_round, _ = perform_transfer(
            opportunistic_state,
            remaining=n_cols,
            boundary_src_row=from_row_i,
            boundary_dst_row=to_row_i,
        )
        opportunistic_rounds.append(opp_round)

    achieved_delta_S = source_supply_at_boundary(opportunistic_state, boundary_src_row) - initial_boundary_supply
    if achieved_delta_S >= delta_S:
        status = _status_dict("success", None, 0)
    else:
        unmet_delta_S: int = max(0, int(delta_S - achieved_delta_S))
        if exact_status["kind"] == "partial_blocked":
            blocking_row_raw: int | str | None = exact_status["blocking_row"]
            blocking_row: int | None = None if blocking_row_raw is None else int(blocking_row_raw)
            status = _status_dict("partial_blocked", blocking_row, unmet_delta_S)
        else:
            status = _status_dict(
                str(exact_status["kind"]),
                None if exact_status["blocking_row"] is None else int(exact_status["blocking_row"]),
                unmet_delta_S,
            )
    return opportunistic_state, opportunistic_rounds, achieved_delta_S, status


def _iter_local_matching_choices(
    prev_src_open: bool,
    prev_vac_open: bool,
    src_here: bool,
    vac_here: bool,
    c: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """
    Return all valid local edge subsets at one DP frontier column.

    Notes
    -----
    This mirrors the frontier structure used in ``direct_cut_capacity(...)`` and
    related exact local match planners. It is split out here so the horizontal
    boundary-row helper can jointly optimize over candidate vacancy patterns and
    the resulting direct cut capacity.
    """
    return (
        (),
        (((c - 1), c),) if prev_src_open and vac_here else (),
        ((c, (c - 1)),) if src_here and prev_vac_open else (),
        ((c, c),) if src_here and vac_here else (),
        (((c - 1), c), (c, (c - 1)))
        if prev_src_open and vac_here and src_here and prev_vac_open
        else (),
    )

def _choose_target_vacancy_mask_for_horizontal_capacity(
    src_row: np.ndarray,
    dst_row: np.ndarray,
    target_capacity: int,
    target_start_col: int,
    target_end_col: int,
    preserve_outside_target: bool,
) -> np.ndarray | None:
    """
    Choose a destination-row vacancy mask that reaches a requested cut capacity.

    Why this exists
    ---------------
    The horizontal boundary-row helper should reason directly about the quantity
    it is trying to improve: the exact direct cut capacity between a fixed source
    row and the destination boundary row. Since any vacancy pattern with the same
    total vacancy count is eventually reachable by horizontal rearrangement, the
    core planning question is to choose a target vacancy pattern that:

    1. has the correct total vacancy count,
    2. achieves at least the requested direct cut capacity,
    3. changes the current row as little as possible, and
    4. optionally leaves sites outside the target window untouched.

    The dynamic program below optimizes over vacancy patterns column-by-column.
    It embeds the same four-state local matching frontier used by the exact
    cut-capacity helper, while also tracking how many vacancies have been placed.
    Among feasible patterns, it prefers:

    1. more overlap with the current vacancy pattern,
    2. smaller total vacancy displacement,
    3. a deterministic lexicographic reconstruction.
    """
    n_cols: int = int(src_row.size)
    cur_vac: np.ndarray = dst_row == 0
    current_vac_cols: np.ndarray = np.flatnonzero(cur_vac).astype(np.intp, copy=False)
    n_vac: int = int(current_vac_cols.size)

    if target_capacity < 0:
        raise ValueError(f"target_capacity must be nonnegative; got {target_capacity}.")
    if target_capacity == 0:
        return cur_vac.copy()
    if n_vac == 0:
        return None

    src_count: int = int(np.sum(src_row, dtype=np.int64))
    max_possible_capacity: int = min(src_count, n_vac)
    if target_capacity > max_possible_capacity:
        return None

    def _capacity_from_src_and_vacancy_mask(
        src_mask: np.ndarray,
        vac_mask: np.ndarray,
    ) -> int:
        """Return exact local cut capacity for a source row and vacancy mask."""
        neg_inf_local: int = -10**9
        scores_local: np.ndarray = np.full(4, neg_inf_local, dtype=np.int64)
        scores_local[0] = 0

        for col in range(n_cols):
            src_here_local: bool = bool(src_mask[col])
            vac_here_local: bool = bool(vac_mask[col])
            next_scores_local: np.ndarray = np.full(4, neg_inf_local, dtype=np.int64)

            for state_bits_local in range(4):
                base_score_local: int = int(scores_local[state_bits_local])
                if base_score_local == neg_inf_local:
                    continue

                prev_src_open_local: bool = bool(state_bits_local & 0b01)
                prev_vac_open_local: bool = bool(state_bits_local & 0b10)

                next_src_open_local: bool = src_here_local
                next_vac_open_local: bool = vac_here_local
                next_state_bits_local: int = (
                    int(next_src_open_local) | (int(next_vac_open_local) << 1)
                )
                if base_score_local > int(next_scores_local[next_state_bits_local]):
                    next_scores_local[next_state_bits_local] = base_score_local

                if prev_src_open_local and vac_here_local:
                    next_src_open_local = src_here_local
                    next_vac_open_local = False
                    next_state_bits_local = (
                        int(next_src_open_local) | (int(next_vac_open_local) << 1)
                    )
                    if base_score_local + 1 > int(next_scores_local[next_state_bits_local]):
                        next_scores_local[next_state_bits_local] = base_score_local + 1

                if src_here_local and prev_vac_open_local:
                    next_src_open_local = False
                    next_vac_open_local = vac_here_local
                    next_state_bits_local = (
                        int(next_src_open_local) | (int(next_vac_open_local) << 1)
                    )
                    if base_score_local + 1 > int(next_scores_local[next_state_bits_local]):
                        next_scores_local[next_state_bits_local] = base_score_local + 1

                if src_here_local and vac_here_local:
                    next_src_open_local = False
                    next_vac_open_local = False
                    next_state_bits_local = (
                        int(next_src_open_local) | (int(next_vac_open_local) << 1)
                    )
                    if base_score_local + 1 > int(next_scores_local[next_state_bits_local]):
                        next_scores_local[next_state_bits_local] = base_score_local + 1

                if (
                    prev_src_open_local
                    and vac_here_local
                    and src_here_local
                    and prev_vac_open_local
                ):
                    next_src_open_local = False
                    next_vac_open_local = False
                    next_state_bits_local = (
                        int(next_src_open_local) | (int(next_vac_open_local) << 1)
                    )
                    if base_score_local + 2 > int(next_scores_local[next_state_bits_local]):
                        next_scores_local[next_state_bits_local] = base_score_local + 2

            scores_local = next_scores_local

        return int(np.max(scores_local))

    current_capacity: int = _capacity_from_src_and_vacancy_mask(src_row, cur_vac)
    if current_capacity >= target_capacity:
        return cur_vac.copy()

    neg_inf: int = -10**9
    large_disp: int = 10**9
    n_states: int = 4

    overlap_scores: np.ndarray = np.full(
        (n_vac + 1, target_capacity + 1, n_states),
        neg_inf,
        dtype=np.int64,
    )
    disp_costs: np.ndarray = np.full(
        (n_vac + 1, target_capacity + 1, n_states),
        large_disp,
        dtype=np.int64,
    )
    overlap_scores[0, 0, 0] = 0
    disp_costs[0, 0, 0] = 0

    next_overlap: np.ndarray = np.empty_like(overlap_scores)
    next_disp: np.ndarray = np.empty_like(disp_costs)

    prev_v_used_table: np.ndarray = np.full(
        (n_cols, n_vac + 1, target_capacity + 1, n_states),
        -1,
        dtype=np.int64,
    )
    prev_match_table: np.ndarray = np.full(
        (n_cols, n_vac + 1, target_capacity + 1, n_states),
        -1,
        dtype=np.int64,
    )
    prev_state_table: np.ndarray = np.full(
        (n_cols, n_vac + 1, target_capacity + 1, n_states),
        -1,
        dtype=np.int64,
    )
    vac_choice_table: np.ndarray = np.full(
        (n_cols, n_vac + 1, target_capacity + 1, n_states),
        -1,
        dtype=np.int8,
    )

    # Precomputed local transition templates.
    # Each tuple is:
    # (used_prev_src, used_src_here, used_prev_vac, used_vac_here, match_gain)
    _LOCAL_TRANSITIONS: dict[tuple[bool, bool, bool, bool], tuple[tuple[int, int, int, int, int], ...]] = {
        (False, False, False, False): ((0, 0, 0, 0, 0),),
        (False, False, False, True): (
            (0, 0, 0, 0, 0),  # no edge
        ),
        (False, False, True, False): (
            (0, 0, 0, 0, 0),  # no edge
        ),
        (False, False, True, True): (
            (0, 0, 0, 0, 0),  # no edge
        ),
        (False, True, False, False): (
            (0, 0, 0, 0, 0),  # no edge
        ),
        (False, True, False, True): (
            (0, 0, 0, 0, 0),  # no edge
            (0, 1, 0, 1, 1),  # C: c -> c
        ),
        (False, True, True, False): (
            (0, 0, 0, 0, 0),  # no edge
            (0, 1, 1, 0, 1),  # B: c -> c-1
        ),
        (False, True, True, True): (
            (0, 0, 0, 0, 0),  # no edge
            (0, 1, 1, 0, 1),  # B
            (0, 1, 0, 1, 1),  # C
        ),
        (True, False, False, False): (
            (0, 0, 0, 0, 0),  # no edge
        ),
        (True, False, False, True): (
            (0, 0, 0, 0, 0),  # no edge
            (1, 0, 0, 1, 1),  # A: c-1 -> c
        ),
        (True, False, True, False): (
            (0, 0, 0, 0, 0),  # no edge
        ),
        (True, False, True, True): (
            (0, 0, 0, 0, 0),  # no edge
            (1, 0, 0, 1, 1),  # A
        ),
        (True, True, False, False): (
            (0, 0, 0, 0, 0),  # no edge
        ),
        (True, True, False, True): (
            (0, 0, 0, 0, 0),  # no edge
            (1, 0, 0, 1, 1),  # A
            (0, 1, 0, 1, 1),  # C
        ),
        (True, True, True, False): (
            (0, 0, 0, 0, 0),  # no edge
            (0, 1, 1, 0, 1),  # B
        ),
        (True, True, True, True): (
            (0, 0, 0, 0, 0),  # no edge
            (1, 0, 0, 1, 1),  # A
            (0, 1, 1, 0, 1),  # B
            (0, 1, 0, 1, 1),  # C
            (1, 1, 1, 1, 2),  # A + B
        ),
    }

    for c in range(n_cols):
        src_here: bool = bool(src_row[c])
        cur_vac_here: bool = bool(cur_vac[c])

        next_overlap.fill(neg_inf)
        next_disp.fill(large_disp)

        if preserve_outside_target and not (target_start_col <= c <= target_end_col):
            vac_options: tuple[bool, ...] = (cur_vac_here,)
        else:
            vac_options = (False, True)

        v_used_min: int = max(0, n_vac - (n_cols - (c + 1)))
        v_used_max: int = min(n_vac, c + 1)

        for v_used in range(v_used_min, v_used_max + 1):
            match_upper: int = min(target_capacity, v_used)

            for match_count in range(match_upper + 1):
                for state_bits in range(n_states):
                    base_overlap: int = int(overlap_scores[v_used, match_count, state_bits])
                    if base_overlap == neg_inf:
                        continue
                    base_disp: int = int(disp_costs[v_used, match_count, state_bits])

                    prev_src_open: bool = bool(state_bits & 0b01)
                    prev_vac_open: bool = bool(state_bits & 0b10)

                    for vac_here in vac_options:
                        new_v_used: int = v_used + int(vac_here)
                        if new_v_used > n_vac:
                            continue

                        overlap_gain: int = int(vac_here == cur_vac_here)
                        disp_gain: int = 0
                        if vac_here:
                            vac_index: int = new_v_used - 1
                            disp_gain = abs(c - int(current_vac_cols[vac_index]))

                        transitions = _LOCAL_TRANSITIONS[
                            (prev_src_open, src_here, prev_vac_open, vac_here)
                        ]

                        for (
                            _used_prev_src,
                            used_src_here,
                            _used_prev_vac,
                            used_vac_here,
                            match_gain,
                        ) in transitions:
                            next_src_open: bool = src_here and (not bool(used_src_here))
                            next_vac_open: bool = vac_here and (not bool(used_vac_here))
                            next_state_bits: int = (
                                int(next_src_open) | (int(next_vac_open) << 1)
                            )

                            new_match_count_raw: int = match_count + match_gain
                            new_match_count: int = min(target_capacity, new_match_count_raw)
                            cand_overlap: int = base_overlap + overlap_gain
                            cand_disp: int = base_disp + disp_gain

                            old_overlap: int = int(
                                next_overlap[new_v_used, new_match_count, next_state_bits]
                            )
                            old_disp: int = int(
                                next_disp[new_v_used, new_match_count, next_state_bits]
                            )
                            cand_better: bool = (
                                cand_overlap > old_overlap
                                or (cand_overlap == old_overlap and cand_disp < old_disp)
                            )
                            if not cand_better:
                                continue

                            next_overlap[new_v_used, new_match_count, next_state_bits] = cand_overlap
                            next_disp[new_v_used, new_match_count, next_state_bits] = cand_disp
                            prev_v_used_table[c, new_v_used, new_match_count, next_state_bits] = v_used
                            prev_match_table[c, new_v_used, new_match_count, next_state_bits] = match_count
                            prev_state_table[c, new_v_used, new_match_count, next_state_bits] = state_bits
                            vac_choice_table[c, new_v_used, new_match_count, next_state_bits] = int(vac_here)

        overlap_scores, next_overlap = next_overlap, overlap_scores
        disp_costs, next_disp = next_disp, disp_costs

    best_state: tuple[int, int, int] | None = None
    best_score: tuple[int, int] | None = None

    for state_bits in range(n_states):
        overlap_val: int = int(overlap_scores[n_vac, target_capacity, state_bits])
        if overlap_val == neg_inf:
            continue
        disp_val: int = int(disp_costs[n_vac, target_capacity, state_bits])
        cand_score: tuple[int, int] = (overlap_val, -disp_val)
        if best_score is None or cand_score > best_score:
            best_score = cand_score
            best_state = (n_vac, target_capacity, state_bits)

    if best_state is None:
        return None

    target_vac_rev: list[bool] = []
    v_used: int
    match_count: int
    state_bits: int
    v_used, match_count, state_bits = best_state

    for c in range(n_cols - 1, -1, -1):
        vac_choice_raw: int = int(vac_choice_table[c, v_used, match_count, state_bits])
        if vac_choice_raw < 0:
            raise RuntimeError(
                "Horizontal vacancy-pattern DP backtracking failed due to missing choice state."
            )
        target_vac_rev.append(bool(vac_choice_raw))

        prev_v_used: int = int(prev_v_used_table[c, v_used, match_count, state_bits])
        prev_match: int = int(prev_match_table[c, v_used, match_count, state_bits])
        prev_state: int = int(prev_state_table[c, v_used, match_count, state_bits])

        if c > 0 and (prev_v_used < 0 or prev_match < 0 or prev_state < 0):
            raise RuntimeError(
                "Horizontal vacancy-pattern DP backtracking failed due to missing predecessor state."
            )
        if prev_v_used >= 0:
            v_used, match_count, state_bits = prev_v_used, prev_match, prev_state

    target_vac_rev.reverse()
    return np.asarray(target_vac_rev, dtype=np.bool_)

def _choose_target_vacancy_mask_for_horizontal_capacity_slow(
    src_row: np.ndarray,
    dst_row: np.ndarray,
    target_capacity: int,
    target_start_col: int,
    target_end_col: int,
    preserve_outside_target: bool,
) -> np.ndarray | None:
    """
    Choose a destination-row vacancy mask that reaches a requested cut capacity.

    Why this exists
    ---------------
    The horizontal boundary-row helper should reason directly about the quantity
    it is trying to improve: the exact direct cut capacity between a fixed source
    row and the destination boundary row. Since any vacancy pattern with the same
    total vacancy count is eventually reachable by horizontal rearrangement, the
    core planning question is to choose a target vacancy pattern that:

    1. has the correct total vacancy count,
    2. achieves at least the requested direct cut capacity,
    3. changes the current row as little as possible, and
    4. optionally leaves sites outside the target window untouched.

    The dynamic program below optimizes over vacancy patterns column-by-column.
    It embeds the same four-state local matching frontier used by the exact
    cut-capacity helper, while also tracking how many vacancies have been placed.
    Among feasible patterns, it prefers:

    1. more overlap with the current vacancy pattern,
    2. smaller total vacancy displacement,
    3. a deterministic lexicographic reconstruction.
    """
    n_cols: int = int(src_row.size)
    cur_vac: np.ndarray = dst_row == 0
    current_vac_cols: np.ndarray = np.flatnonzero(cur_vac).astype(np.intp, copy=False)
    n_vac: int = int(current_vac_cols.size)

    if target_capacity < 0:
        raise ValueError(f"target_capacity must be nonnegative; got {target_capacity}.")
    if target_capacity == 0:
        return cur_vac.copy()
    if n_vac == 0:
        return None

    neg_inf: int = -10**9
    n_states: int = 4
    # max_matches: int = min(int(np.sum(src_row, dtype=np.int64)), n_vac)

    overlap_scores: np.ndarray = np.full(
        (n_vac + 1, target_capacity + 1, n_states),
        neg_inf,
        dtype=np.int64,
    )
    disp_costs: np.ndarray = np.full(
        (n_vac + 1, target_capacity + 1, n_states),
        10**9,
        dtype=np.int64,
    )
    overlap_scores[0, 0, 0] = 0
    disp_costs[0, 0, 0] = 0

    prev_v_used_table: np.ndarray = np.full(
        (n_cols, n_vac + 1, target_capacity + 1, n_states),
        -1,
        dtype=np.int64,
    )
    prev_match_table: np.ndarray = np.full(
        (n_cols, n_vac + 1, target_capacity + 1, n_states),
        -1,
        dtype=np.int64,
    )
    prev_state_table: np.ndarray = np.full(
        (n_cols, n_vac + 1, target_capacity + 1, n_states),
        -1,
        dtype=np.int64,
    )
    vac_choice_table: np.ndarray = np.full(
        (n_cols, n_vac + 1, target_capacity + 1, n_states),
        -1,
        dtype=np.int8,
    )

    for c in range(n_cols):
        src_here: bool = bool(src_row[c])
        next_overlap: np.ndarray = np.full_like(overlap_scores, neg_inf)
        next_disp: np.ndarray = np.full_like(disp_costs, 10**9)

        if preserve_outside_target and not (target_start_col <= c <= target_end_col):
            vac_options: tuple[bool, ...] = (bool(cur_vac[c]),)
        else:
            vac_options = (False, True)

        for v_used in range(n_vac + 1):
            for match_count in range(target_capacity + 1):
                for state_bits in range(n_states):
                    base_overlap: int = int(overlap_scores[v_used, match_count, state_bits])
                    if base_overlap == neg_inf:
                        continue
                    base_disp: int = int(disp_costs[v_used, match_count, state_bits])

                    prev_src_open: bool = bool(state_bits & 0b01)
                    prev_vac_open: bool = bool(state_bits & 0b10)

                    for vac_here in vac_options:
                        new_v_used: int = v_used + int(vac_here)
                        if new_v_used > n_vac:
                            continue

                        overlap_gain: int = int(vac_here == bool(cur_vac[c]))
                        disp_gain: int = 0
                        if vac_here:
                            vac_index: int = new_v_used - 1
                            disp_gain = abs(c - int(current_vac_cols[vac_index]))

                        local_choices = _iter_local_matching_choices(
                            prev_src_open=prev_src_open,
                            prev_vac_open=prev_vac_open,
                            src_here=src_here,
                            vac_here=vac_here,
                            c=c,
                        )

                        for edges in local_choices:
                            used_prev_src: bool = False
                            used_src_here: bool = False
                            used_prev_vac: bool = False
                            used_vac_here: bool = False
                            valid: bool = True

                            for src_col, dst_col in edges:
                                if src_col == c - 1:
                                    if not prev_src_open or used_prev_src:
                                        valid = False
                                        break
                                    used_prev_src = True
                                elif src_col == c:
                                    if not src_here or used_src_here:
                                        valid = False
                                        break
                                    used_src_here = True
                                else:
                                    valid = False
                                    break

                                if dst_col == c - 1:
                                    if not prev_vac_open or used_prev_vac:
                                        valid = False
                                        break
                                    used_prev_vac = True
                                elif dst_col == c:
                                    if not vac_here or used_vac_here:
                                        valid = False
                                        break
                                    used_vac_here = True
                                else:
                                    valid = False
                                    break

                            if not valid:
                                continue

                            next_src_open: bool = src_here and (not used_src_here)
                            next_vac_open: bool = vac_here and (not used_vac_here)
                            next_state_bits: int = int(next_src_open) | (int(next_vac_open) << 1)

                            new_match_count_raw: int = match_count + len(edges)
                            new_match_count: int = min(target_capacity, new_match_count_raw)
                            cand_overlap: int = base_overlap + overlap_gain
                            cand_disp: int = base_disp + disp_gain

                            old_overlap: int = int(next_overlap[new_v_used, new_match_count, next_state_bits])
                            old_disp: int = int(next_disp[new_v_used, new_match_count, next_state_bits])
                            cand_better: bool = (
                                cand_overlap > old_overlap
                                or (cand_overlap == old_overlap and cand_disp < old_disp)
                            )
                            if not cand_better:
                                continue

                            next_overlap[new_v_used, new_match_count, next_state_bits] = cand_overlap
                            next_disp[new_v_used, new_match_count, next_state_bits] = cand_disp
                            prev_v_used_table[c, new_v_used, new_match_count, next_state_bits] = v_used
                            prev_match_table[c, new_v_used, new_match_count, next_state_bits] = match_count
                            prev_state_table[c, new_v_used, new_match_count, next_state_bits] = state_bits
                            vac_choice_table[c, new_v_used, new_match_count, next_state_bits] = int(vac_here)

        overlap_scores = next_overlap
        disp_costs = next_disp

    best_state: tuple[int, int, int] | None = None
    best_score: tuple[int, int] | None = None

    for state_bits in range(n_states):
        overlap_val: int = int(overlap_scores[n_vac, target_capacity, state_bits])
        if overlap_val == neg_inf:
            continue
        disp_val: int = int(disp_costs[n_vac, target_capacity, state_bits])
        cand_score: tuple[int, int] = (overlap_val, -disp_val)
        if best_score is None or cand_score > best_score:
            best_score = cand_score
            best_state = (n_vac, target_capacity, state_bits)

    if best_state is None:
        return None

    target_vac_rev: list[bool] = []
    v_used, match_count, state_bits = best_state
    for c in range(n_cols - 1, -1, -1):
        vac_choice_raw: int = int(vac_choice_table[c, v_used, match_count, state_bits])
        if vac_choice_raw < 0:
            raise RuntimeError(
                "Horizontal vacancy-pattern DP backtracking failed due to missing choice state."
            )
        target_vac_rev.append(bool(vac_choice_raw))

        prev_v_used: int = int(prev_v_used_table[c, v_used, match_count, state_bits])
        prev_match: int = int(prev_match_table[c, v_used, match_count, state_bits])
        prev_state: int = int(prev_state_table[c, v_used, match_count, state_bits])
        if c > 0 and (prev_v_used < 0 or prev_match < 0 or prev_state < 0):
            raise RuntimeError(
                "Horizontal vacancy-pattern DP backtracking failed due to missing predecessor state."
            )
        if prev_v_used >= 0:
            v_used, match_count, state_bits = prev_v_used, prev_match, prev_state

    target_vac_rev.reverse()
    return np.asarray(target_vac_rev, dtype=np.bool_)



def _target_row_from_vacancy_mask(vacancy_mask: np.ndarray) -> np.ndarray:
    """Return the occupancy row corresponding to a boolean vacancy mask."""
    return (~vacancy_mask).astype(np.uint8, copy=False)



def _plan_horizontal_rounds_to_target(
    current_row: np.ndarray,
    target_row: np.ndarray,
    boundary_dst_row: int,
) -> list[list[Move]]:
    """
    Plan horizontal move rounds that transform one row occupancy pattern into another.

    Why this exists
    ---------------
    The horizontal cut-capacity helper needs a same-row transport planner that
    respects one-dimensional exclusion while still exploiting obvious parallelism.
    With indistinguishable atoms and left-to-right pairing, the transport is the
    natural non-crossing matching on a row.

    The key scheduling detail is directional:
    - left-moving chains should be scheduled from left to right,
    - right-moving chains should be scheduled from right to left.

    That ordering allows a whole chain to advance in one round when each atom
    moves into a site vacated by the next atom in the same direction.
    """
    current_atom_cols: list[int] = (
        np.flatnonzero(current_row).astype(np.intp, copy=False).tolist()
    )
    target_atom_cols: list[int] = (
        np.flatnonzero(target_row).astype(np.intp, copy=False).tolist()
    )

    if len(current_atom_cols) != len(target_atom_cols):
        raise RuntimeError(
            "Horizontal row planner received rows with different atom counts."
        )
    if current_atom_cols == target_atom_cols:
        return []

    work_cols: list[int] = [int(col) for col in current_atom_cols]
    target_cols: list[int] = [int(col) for col in target_atom_cols]
    n_cols: int = int(current_row.size)
    move_rounds: list[list[Move]] = []
    max_rounds: int = max(1, n_cols * n_cols)

    for _ in range(max_rounds):
        if work_cols == target_cols:
            break

        occupancy: np.ndarray = np.zeros(n_cols, dtype=np.bool_)
        for col in work_cols:
            occupancy[col] = True

        round_moves: list[Move] = []
        new_cols: list[int] = work_cols.copy()

        left_indices: list[int] = [
            idx
            for idx, (cur_col, tgt_col) in enumerate(zip(work_cols, target_cols, strict=True))
            if cur_col > tgt_col
        ]
        right_indices: list[int] = [
            idx
            for idx, (cur_col, tgt_col) in enumerate(zip(work_cols, target_cols, strict=True))
            if cur_col < tgt_col
        ]

        for idx in left_indices:
            cur_col: int = int(work_cols[idx])
            next_col: int = cur_col - 1
            if next_col < 0:
                raise RuntimeError(
                    "Horizontal row planner attempted to move outside row bounds."
                )
            if occupancy[next_col]:
                continue
            round_moves.append(
                Move(boundary_dst_row, cur_col, boundary_dst_row, next_col)
            )
            occupancy[cur_col] = False
            occupancy[next_col] = True
            new_cols[idx] = next_col

        for idx in reversed(right_indices):
            cur_col = int(work_cols[idx])
            next_col = cur_col + 1
            if next_col >= n_cols:
                raise RuntimeError(
                    "Horizontal row planner attempted to move outside row bounds."
                )
            if occupancy[next_col]:
                continue
            round_moves.append(
                Move(boundary_dst_row, cur_col, boundary_dst_row, next_col)
            )
            occupancy[cur_col] = False
            occupancy[next_col] = True
            new_cols[idx] = next_col

        if len(round_moves) == 0:
            raise RuntimeError(
                "Horizontal row planner stalled before reaching target occupancy pattern."
            )

        move_rounds.append(round_moves)
        work_cols = new_cols

    if work_cols != target_cols:
        raise RuntimeError(
            "Horizontal row planner exceeded its round budget before reaching the target pattern."
        )
    return move_rounds



def _moves_touch_outside_target(
    move_rounds: list[list[Move]],
    target_start_col: int,
    target_end_col: int,
) -> bool:
    """Return whether any move source or destination lies outside the target window."""
    for round_moves in move_rounds:
        for move in round_moves:
            if (
                int(move.from_col) < target_start_col
                or int(move.from_col) > target_end_col
                or int(move.to_col) < target_start_col
                or int(move.to_col) > target_end_col
            ):
                return True
    return False


def ensure_boundary_row_cut_capacity_horizontal(
    state: np.ndarray,
    boundary_src_row: int,
    boundary_dst_row: int,
    delta_T: int,
    target_start_col: int,
    target_end_col: int,
) -> tuple[np.ndarray, list[list[Move]], int, dict[str, int | bool | str]]:
    """
    Increase direct cut capacity by horizontally realigning the destination boundary row.

    Why this exists
    ---------------
    When transfer across a cut is limited by poor alignment between source-row
    atoms and destination-row vacancies, the controller needs a helper that
    changes *only* the destination boundary row. This helper performs that
    narrow micro-objective:

    - it uses only horizontal, unit-step, same-row moves,
    - it preserves the destination-row atom count,
    - it never touches the source row or deeper destination rows,
    - and it is exact about the achieved improvement in direct cut capacity.

    The helper first performs a cheap horizontal-only feasibility precheck using
    the eventual ceiling ``min(S, V_dst)``. It then searches for a target vacancy
    pattern that reaches the requested cut capacity while changing the current row
    as little as possible. If a solution exists that keeps sites outside the
    target window unchanged, that class is preferred. Otherwise, the helper may
    spill outside the target window and reports that in its metadata.
    """
    if state.ndim != 3 or state.shape[2] != 1:
        raise ValueError(
            "state must have shape (rows, cols, 1) for BCv2 single-species helpers."
        )
    if delta_T < 0:
        raise ValueError(f"delta_T must be nonnegative; got {delta_T}.")

    n_rows: int = int(state.shape[0])
    n_cols: int = int(state.shape[1])
    if boundary_src_row < 0 or boundary_src_row >= n_rows:
        raise IndexError(
            f"boundary_src_row={boundary_src_row} is out of bounds for {n_rows} rows."
        )
    if boundary_dst_row < 0 or boundary_dst_row >= n_rows:
        raise IndexError(
            f"boundary_dst_row={boundary_dst_row} is out of bounds for {n_rows} rows."
        )
    if target_start_col < 0 or target_start_col >= n_cols:
        raise IndexError(
            f"target_start_col={target_start_col} is out of bounds for {n_cols} columns."
        )
    if target_end_col < 0 or target_end_col >= n_cols:
        raise IndexError(
            f"target_end_col={target_end_col} is out of bounds for {n_cols} columns."
        )
    if target_start_col > target_end_col:
        raise ValueError(
            "target_start_col must be <= target_end_col for the horizontal boundary helper."
        )

    initial_state: np.ndarray = state.copy()
    dst_count_before: int = _count_row_atoms(initial_state, boundary_dst_row)
    t_before: int = direct_cut_capacity(initial_state, boundary_src_row, boundary_dst_row)

    if delta_T == 0:
        return (
            initial_state,
            [],
            0,
            {
                "kind": "success",
                "T_before": int(t_before),
                "T_after": int(t_before),
                "requested_delta_T": 0,
                "achieved_delta_T": 0,
                "used_outside_target_cols": False,
                "n_rounds": 0,
            },
        )

    src_supply: int = _count_row_atoms(initial_state, boundary_src_row)
    dst_vacancies: int = int(np.sum(initial_state[boundary_dst_row, :, 0] == 0, dtype=np.int64))
    horizontal_ceiling: int = min(src_supply, dst_vacancies)
    target_capacity: int = int(t_before + delta_T)

    if target_capacity > horizontal_ceiling:
        return (
            initial_state,
            [],
            0,
            {
                "kind": "infeasible_request",
                "T_before": int(t_before),
                "T_after": int(t_before),
                "requested_delta_T": int(delta_T),
                "achieved_delta_T": 0,
                "used_outside_target_cols": False,
                "n_rounds": 0,
            },
        )

    src_row: np.ndarray = initial_state[boundary_src_row, :, 0].astype(np.bool_, copy=False)
    dst_row: np.ndarray = initial_state[boundary_dst_row, :, 0].astype(np.uint8, copy=True)

    chosen_vac_mask: np.ndarray | None = _choose_target_vacancy_mask_for_horizontal_capacity(
        src_row=src_row,
        dst_row=dst_row,
        target_capacity=target_capacity,
        target_start_col=target_start_col,
        target_end_col=target_end_col,
        preserve_outside_target=True,
    )
    preserve_outside_target: bool = True
    if chosen_vac_mask is None:
        chosen_vac_mask = _choose_target_vacancy_mask_for_horizontal_capacity(
            src_row=src_row,
            dst_row=dst_row,
            target_capacity=target_capacity,
            target_start_col=target_start_col,
            target_end_col=target_end_col,
            preserve_outside_target=False,
        )
        preserve_outside_target = False

    if chosen_vac_mask is None:
        return (
            initial_state,
            [],
            0,
            {
                "kind": "cannot_solve_within_constraints",
                "T_before": int(t_before),
                "T_after": int(t_before),
                "requested_delta_T": int(delta_T),
                "achieved_delta_T": 0,
                "used_outside_target_cols": False,
                "n_rounds": 0,
            },
        )

    target_row: np.ndarray = _target_row_from_vacancy_mask(chosen_vac_mask)
    move_rounds: list[list[Move]] = _plan_horizontal_rounds_to_target(
        current_row=dst_row,
        target_row=target_row,
        boundary_dst_row=boundary_dst_row,
    )

    work_state: np.ndarray = initial_state.copy()
    executed_rounds: list[list[Move]] = []
    t_after: int = t_before
    for round_moves in move_rounds:
        work_state = move_atoms_noiseless(work_state.copy(), round_moves)
        executed_rounds.append(round_moves)
        t_after = direct_cut_capacity(work_state, boundary_src_row, boundary_dst_row)
        if t_after - t_before >= delta_T:
            break

    achieved_delta_T: int = int(t_after - t_before)
    used_outside_target_cols: bool = _moves_touch_outside_target(
        executed_rounds,
        target_start_col=target_start_col,
        target_end_col=target_end_col,
    )
    if preserve_outside_target and used_outside_target_cols:
        raise RuntimeError(
            "Horizontal boundary helper claimed to preserve sites outside the target window, "
            "but executed moves touched outside-target columns."
        )

    if _count_row_atoms(work_state, boundary_dst_row) != dst_count_before:
        raise RuntimeError(
            "Horizontal boundary helper changed the destination boundary-row atom count."
        )

    status_kind: str = "success" if achieved_delta_T >= delta_T else "cannot_solve_within_constraints"
    status: dict[str, int | bool | str] = {
        "kind": status_kind,
        "T_before": int(t_before),
        "T_after": int(t_after),
        "requested_delta_T": int(delta_T),
        "achieved_delta_T": int(achieved_delta_T),
        "used_outside_target_cols": bool(used_outside_target_cols),
        "n_rounds": int(len(executed_rounds)),
    }
    return work_state, executed_rounds, achieved_delta_T, status


def ensure_boundary_row_cut_capacity_vertical(
    state: np.ndarray,
    boundary_src_row: int,
    boundary_dst_row: int,
    search_limit_row: int,
    delta_T: int,
) -> tuple[np.ndarray, list[list[Move]], int, dict[str, int | str | None]]:
    """
    Increase direct cut capacity by vertically clearing atoms deeper into the destination region.

    Why this exists
    ---------------
    When the source boundary row already has enough supply but direct transfer
    across the cut is still bottlenecked, one possible cause is that the
    destination boundary row is too crowded. This helper addresses that specific
    micro-objective by moving atoms from the destination boundary row farther
    into the destination-side interval, using only adjacent-row destination-side
    transfers. It does not touch the source side, does not cross the cut, and
    does not perform horizontal unblocking.

    Contract
    --------
    - Only rows in the inclusive interval between ``boundary_dst_row`` and
      ``search_limit_row`` may be touched.
    - Returned progress is measured only by the exact change in
      ``direct_cut_capacity(state, boundary_src_row, boundary_dst_row)``.
    - The helper stops as soon as the requested gain is reached or exceeded. It
      does not intentionally spend extra destination-side clearing rounds solely
      to push capacity beyond ``delta_T``.
    - The coarse infeasibility precheck is destination-side only:
      ``n_vacancies_in_interval - T_before``.

    Parameters
    ----------
    state
        Single-species occupancy array with shape ``(rows, cols, 1)``.
    boundary_src_row
        Source row adjacent to the cut.
    boundary_dst_row
        Destination row adjacent to the cut.
    search_limit_row
        Furthest destination-side row the helper is allowed to touch.
    delta_T
        Requested increase in direct cut capacity.

    Returns
    -------
    tuple[np.ndarray, list[list[Move]], int, dict[str, int | str | None]]
        ``(new_state, move_rounds, achieved_delta_T, status)``.

    Raises
    ------
    ValueError
        If ``state`` has the wrong shape, ``delta_T`` is negative, or the
        destination interval is trivial.
    IndexError
        If any row index is out of bounds.
    RuntimeError
        If internal bookkeeping becomes inconsistent.
    """
    if state.ndim != 3 or state.shape[2] != 1:
        raise ValueError(
            "state must have shape (rows, cols, 1) for BCv2 single-species helpers."
        )
    if delta_T < 0:
        raise ValueError(f"delta_T must be nonnegative; got {delta_T}.")

    n_rows: int = int(state.shape[0])
    if boundary_src_row < 0 or boundary_src_row >= n_rows:
        raise IndexError(
            f"boundary_src_row={boundary_src_row} is out of bounds for {n_rows} rows."
        )
    if boundary_dst_row < 0 or boundary_dst_row >= n_rows:
        raise IndexError(
            f"boundary_dst_row={boundary_dst_row} is out of bounds for {n_rows} rows."
        )
    if search_limit_row < 0 or search_limit_row >= n_rows:
        raise IndexError(
            f"search_limit_row={search_limit_row} is out of bounds for {n_rows} rows."
        )
    if boundary_dst_row == search_limit_row:
        raise ValueError(
            "ensure_boundary_row_cut_capacity_vertical requires a nontrivial "
            "destination interval: search_limit_row must differ from boundary_dst_row."
        )

    def _status(
        kind: str,
        t_before: int,
        t_after: int,
        requested_delta_t: int,
        achieved_delta_t: int,
        n_rounds: int,
        blocking_row: int | None,
    ) -> dict[str, int | str | None]:
        """Build the structured status payload."""
        return {
            "kind": kind,
            "T_before": int(t_before),
            "T_after": int(t_after),
            "requested_delta_T": int(requested_delta_t),
            "achieved_delta_T": int(achieved_delta_t),
            "n_rounds": int(n_rounds),
            "blocking_row": blocking_row,
        }

    initial_state: np.ndarray = state.copy()
    t_before: int = direct_cut_capacity(
        initial_state,
        boundary_src_row=boundary_src_row,
        boundary_dst_row=boundary_dst_row,
    )

    if delta_T == 0:
        return (
            initial_state,
            [],
            0,
            _status(
                kind="success",
                t_before=t_before,
                t_after=t_before,
                requested_delta_t=0,
                achieved_delta_t=0,
                n_rounds=0,
                blocking_row=None,
            ),
        )

    low_row: int = min(boundary_dst_row, search_limit_row)
    high_row: int = max(boundary_dst_row, search_limit_row)
    step_away: int = 1 if search_limit_row > boundary_dst_row else -1

    n_vacancies_region: int = int(
        np.sum(initial_state[low_row : high_row + 1, :, 0] == 0, dtype=np.int64)
    )
    coarse_upper_bound: int = n_vacancies_region - t_before
    if delta_T > coarse_upper_bound:
        return (
            initial_state,
            [],
            0,
            _status(
                kind="infeasible_request",
                t_before=t_before,
                t_after=t_before,
                requested_delta_t=delta_T,
                achieved_delta_t=0,
                n_rounds=0,
                blocking_row=None,
            ),
        )

    work_state: np.ndarray = initial_state.copy()
    move_rounds: list[list[Move]] = []
    n_cols: int = int(state.shape[1])

    def clear_one_step_from_row(
        curr_state: np.ndarray,
        row_to_clear: int,
    ) -> tuple[np.ndarray, list[list[Move]], bool, int | None]:
        """
        Clear at least one atom from ``row_to_clear`` one step farther from the cut.
        """
        downstream_row: int = row_to_clear + step_away
        if downstream_row < low_row or downstream_row > high_row:
            return curr_state, [], False, row_to_clear

        next_state: np.ndarray
        direct_moves: list[Move]
        n_moved: int
        next_state, direct_moves, n_moved = perform_transfer(
            curr_state,
            remaining=n_cols,
            boundary_src_row=row_to_clear,
            boundary_dst_row=downstream_row,
        )
        if n_moved > 0:
            return next_state, [direct_moves], True, None

        n_atoms_here: int = int(np.sum(curr_state[row_to_clear, :, 0], dtype=np.int64))
        if n_atoms_here == 0:
            return curr_state, [], False, None

        # NEW:
        # The immediate cut (row_to_clear -> downstream_row) may be blocked by poor
        # alignment even though downstream vacancies exist. Before recursing deeper,
        # try to increase cut capacity on this exact cut.
        cut_state: np.ndarray
        cut_rounds: list[list[Move]]
        cut_achieved: int
        cut_status: dict[str, int | str | bool | None]
        (
            cut_state,
            cut_rounds,
            cut_achieved,
            cut_status,
        ) = ensure_cut_capacity(
            state=curr_state,
            boundary_src_row=row_to_clear,
            boundary_dst_row=downstream_row,
            destination_search_limit_row=search_limit_row,
            delta_T=1,
            target_start_col=0,
            target_end_col=n_cols - 1,
        )

        if len(cut_rounds) > 0:
            retry_state: np.ndarray
            retry_moves: list[Move]
            retry_n_moved: int
            retry_state, retry_moves, retry_n_moved = perform_transfer(
                cut_state,
                remaining=n_cols,
                boundary_src_row=row_to_clear,
                boundary_dst_row=downstream_row,
            )
            if retry_n_moved > 0:
                return retry_state, cut_rounds + [retry_moves], True, None

            # Even if the retry still cannot move immediately, the cut-repair rounds
            # changed state and should count as structural progress.
            return cut_state, cut_rounds, True, None

        # If the immediate cut could not be repaired locally, then recurse deeper.
        deeper_state: np.ndarray
        deeper_rounds: list[list[Move]]
        deeper_progress: bool
        deeper_blocking_row: int | None
        deeper_state, deeper_rounds, deeper_progress, deeper_blocking_row = (
            clear_one_step_from_row(curr_state, downstream_row)
        )

        if deeper_progress:
            retry_state: np.ndarray
            retry_moves: list[Move]
            retry_n_moved: int
            retry_state, retry_moves, retry_n_moved = perform_transfer(
                deeper_state,
                remaining=n_cols,
                boundary_src_row=row_to_clear,
                boundary_dst_row=downstream_row,
            )
            if retry_n_moved > 0:
                return retry_state, deeper_rounds + [retry_moves], True, None

            return deeper_state, deeper_rounds, True, None

        return deeper_state, deeper_rounds, False, row_to_clear

    current_t: int = t_before
    while current_t - t_before < delta_T:
        state_before_episode: np.ndarray = work_state.copy()
        t_before_episode: int = current_t

        work_state, new_rounds, made_progress, blocking_row = clear_one_step_from_row(
            work_state,
            boundary_dst_row,
        )

        if len(new_rounds) > 0:
            move_rounds.extend(new_rounds)

        current_t = direct_cut_capacity(
            work_state,
            boundary_src_row=boundary_src_row,
            boundary_dst_row=boundary_dst_row,
        )
        achieved_delta_t: int = current_t - t_before

        if achieved_delta_t >= delta_T:
            status: dict[str, int | str | None] = _status(
                kind="success",
                t_before=t_before,
                t_after=current_t,
                requested_delta_t=delta_T,
                achieved_delta_t=achieved_delta_t,
                n_rounds=len(move_rounds),
                blocking_row=None,
            )
            if achieved_delta_t != int(status["achieved_delta_T"]):
                raise RuntimeError(
                    "ensure_boundary_row_cut_capacity_vertical success bookkeeping "
                    "mismatch in achieved_delta_T."
                )
            return work_state, move_rounds, achieved_delta_t, status

        no_state_change: bool = np.array_equal(work_state, state_before_episode)
        no_metric_change: bool = current_t == t_before_episode
        if (not made_progress) and no_state_change and no_metric_change:
            status = _status(
                kind="cannot_solve_within_constraints",
                t_before=t_before,
                t_after=current_t,
                requested_delta_t=delta_T,
                achieved_delta_t=achieved_delta_t,
                n_rounds=len(move_rounds),
                blocking_row=blocking_row,
            )
            return work_state, move_rounds, achieved_delta_t, status

    achieved_delta_t = current_t - t_before
    status = _status(
        kind="success",
        t_before=t_before,
        t_after=current_t,
        requested_delta_t=delta_T,
        achieved_delta_t=achieved_delta_t,
        n_rounds=len(move_rounds),
        blocking_row=None,
    )
    return work_state, move_rounds, achieved_delta_t, status

def _make_status(
        kind: str,
        t_before: int,
        t_after: int,
        requested_delta_t: int,
        achieved_delta_t: int,
        n_rounds: int,
        chosen_mode: str,
        horizontal_status_kind: str | None,
        vertical_status_kind: str | None,
        blocking_row: int | None,
        used_outside_target_cols: bool | None,
    ) -> dict[str, int | str | bool | None]:
        """Build the structured coordinator status payload."""
        return {
            "kind": kind,
            "T_before": int(t_before),
            "T_after": int(t_after),
            "requested_delta_T": int(requested_delta_t),
            "achieved_delta_T": int(achieved_delta_t),
            "n_rounds": int(n_rounds),
            "chosen_mode": chosen_mode,
            "horizontal_status_kind": horizontal_status_kind,
            "vertical_status_kind": vertical_status_kind,
            "blocking_row": blocking_row,
            "used_outside_target_cols": used_outside_target_cols,
        }
def ensure_cut_capacity(
    state: np.ndarray,
    boundary_src_row: int,
    boundary_dst_row: int,
    destination_search_limit_row: int,
    delta_T: int,
    target_start_col: int,
    target_end_col: int,
) -> tuple[np.ndarray, list[list[Move]], int, dict[str, int | str | bool | None]]:
    """
    Coordinate destination-side cut-capacity improvement.

    Why this exists
    ---------------
    Row-to-row transfer across a cut can be bottlenecked by poor destination-side
    geometry. The controller currently has two local mechanisms for increasing
    direct cut capacity:

    1. horizontal realignment within the destination boundary row, and
    2. vertical clearing deeper into the destination interval.

    The policy implemented here is intentionally simple:
    - take horizontal immediately only when it can satisfy the full request in a
      single round,
    - otherwise prefer vertical when a nontrivial destination interval exists,
    - and compose horizontal then vertical or vertical then horizontal when that
      is the smallest way to make useful progress.

    Important subtlety
    ------------------
    Horizontal or vertical repair may be useful even when it does not immediately
    increase direct cut capacity. The coordinator therefore treats any non-empty
    move list as structural progress worth returning to the parent controller.
    """
    if state.ndim != 3 or state.shape[2] != 1:
        raise ValueError(
            "state must have shape (rows, cols, 1) for BCv2 single-species helpers."
        )
    if delta_T < 0:
        raise ValueError(f"delta_T must be nonnegative; got {delta_T}.")
    if target_start_col > target_end_col:
        raise ValueError(
            "target_start_col must be <= target_end_col; "
            f"got {target_start_col} > {target_end_col}."
        )

    n_rows: int = int(state.shape[0])
    if boundary_src_row < 0 or boundary_src_row >= n_rows:
        raise IndexError(
            f"boundary_src_row={boundary_src_row} is out of bounds for {n_rows} rows."
        )
    if boundary_dst_row < 0 or boundary_dst_row >= n_rows:
        raise IndexError(
            f"boundary_dst_row={boundary_dst_row} is out of bounds for {n_rows} rows."
        )
    if destination_search_limit_row < 0 or destination_search_limit_row >= n_rows:
        raise IndexError(
            "destination_search_limit_row="
            f"{destination_search_limit_row} is out of bounds for {n_rows} rows."
        )

    initial_state: np.ndarray = state.copy()
    t_before: int = direct_cut_capacity(
        initial_state,
        boundary_src_row=boundary_src_row,
        boundary_dst_row=boundary_dst_row,
    )

    if delta_T == 0:
        return (
            initial_state,
            [],
            0,
            _make_status(
                kind="success",
                t_before=t_before,
                t_after=t_before,
                requested_delta_t=0,
                achieved_delta_t=0,
                n_rounds=0,
                chosen_mode="none",
                horizontal_status_kind=None,
                vertical_status_kind=None,
                blocking_row=None,
                used_outside_target_cols=False,
            ),
        )

    horiz_state: np.ndarray
    horiz_rounds: list[list[Move]]
    horiz_achieved: int
    horiz_status: dict[str, int | str | bool | None]
    (
        horiz_state,
        horiz_rounds,
        horiz_achieved,
        horiz_status,
    ) = ensure_boundary_row_cut_capacity_horizontal(
        state=initial_state,
        boundary_src_row=boundary_src_row,
        boundary_dst_row=boundary_dst_row,
        delta_T=delta_T,
        target_start_col=target_start_col,
        target_end_col=target_end_col,
    )

    horiz_kind: str = str(horiz_status["kind"])
    horiz_used_outside: bool = bool(
        horiz_status.get("used_outside_target_cols", False)
    )

    horizontal_finishes_in_one_round: bool = (
        horiz_kind == "success"
        and int(horiz_achieved) >= int(delta_T)
        and len(horiz_rounds) == 1
    )
    if horizontal_finishes_in_one_round:
        return (
            horiz_state,
            horiz_rounds,
            int(horiz_achieved),
            _make_status(
                kind="success",
                t_before=t_before,
                t_after=int(horiz_status["T_after"]),
                requested_delta_t=delta_T,
                achieved_delta_t=int(horiz_achieved),
                n_rounds=len(horiz_rounds),
                chosen_mode="horizontal",
                horizontal_status_kind=horiz_kind,
                vertical_status_kind=None,
                blocking_row=None,
                used_outside_target_cols=horiz_used_outside,
            ),
        )

    vertical_available: bool = destination_search_limit_row != boundary_dst_row

    horizontal_made_structural_progress: bool = len(horiz_rounds) > 0
    if horizontal_made_structural_progress and vertical_available:
        horiz_vert_state: np.ndarray
        horiz_vert_rounds: list[list[Move]]
        horiz_vert_achieved: int
        horiz_vert_status: dict[str, int | str | None]
        (
            horiz_vert_state,
            horiz_vert_rounds,
            horiz_vert_achieved,
            horiz_vert_status,
        ) = ensure_boundary_row_cut_capacity_vertical(
            state=horiz_state,
            boundary_src_row=boundary_src_row,
            boundary_dst_row=boundary_dst_row,
            search_limit_row=destination_search_limit_row,
            delta_T=max(0, int(delta_T) - int(horiz_achieved)),
        )

        horiz_vert_kind: str = str(horiz_vert_status["kind"])
        total_rounds_hv: list[list[Move]] = list(horiz_rounds) + list(horiz_vert_rounds)
        total_achieved_hv: int = int(horiz_achieved) + int(horiz_vert_achieved)
        t_after_hv: int = direct_cut_capacity(
            horiz_vert_state,
            boundary_src_row=boundary_src_row,
            boundary_dst_row=boundary_dst_row,
        )

        if total_achieved_hv >= int(delta_T):
            return (
                horiz_vert_state,
                total_rounds_hv,
                total_achieved_hv,
                _make_status(
                    kind="success",
                    t_before=t_before,
                    t_after=t_after_hv,
                    requested_delta_t=delta_T,
                    achieved_delta_t=total_achieved_hv,
                    n_rounds=len(total_rounds_hv),
                    chosen_mode="horizontal_then_vertical",
                    horizontal_status_kind=horiz_kind,
                    vertical_status_kind=horiz_vert_kind,
                    blocking_row=None,
                    used_outside_target_cols=horiz_used_outside,
                ),
            )

        if len(total_rounds_hv) > 0:
            blocking_row_hv: int | None = (
                None
                if horiz_vert_status.get("blocking_row") is None
                else int(horiz_vert_status["blocking_row"])
            )
            return (
                horiz_vert_state,
                total_rounds_hv,
                total_achieved_hv,
                _make_status(
                    kind="cannot_solve_within_constraints",
                    t_before=t_before,
                    t_after=t_after_hv,
                    requested_delta_t=delta_T,
                    achieved_delta_t=total_achieved_hv,
                    n_rounds=len(total_rounds_hv),
                    chosen_mode="horizontal_then_vertical",
                    horizontal_status_kind=horiz_kind,
                    vertical_status_kind=horiz_vert_kind,
                    blocking_row=blocking_row_hv,
                    used_outside_target_cols=horiz_used_outside,
                ),
            )

    if not vertical_available:
        final_kind: str
        if horiz_kind == "cannot_solve_within_constraints":
            final_kind = "cannot_solve_within_constraints"
        elif horiz_kind == "success" and int(horiz_achieved) >= int(delta_T):
            final_kind = "success"
        else:
            final_kind = "infeasible_request"

        return (
            horiz_state,
            horiz_rounds,
            int(horiz_achieved),
            _make_status(
                kind=final_kind,
                t_before=t_before,
                t_after=int(horiz_status["T_after"]),
                requested_delta_t=delta_T,
                achieved_delta_t=int(horiz_achieved),
                n_rounds=len(horiz_rounds),
                chosen_mode="horizontal" if len(horiz_rounds) > 0 else "none",
                horizontal_status_kind=horiz_kind,
                vertical_status_kind=None,
                blocking_row=None,
                used_outside_target_cols=horiz_used_outside,
            ),
        )

    vert_state: np.ndarray
    vert_rounds: list[list[Move]]
    vert_achieved: int
    vert_status: dict[str, int | str | None]
    (
        vert_state,
        vert_rounds,
        vert_achieved,
        vert_status,
    ) = ensure_boundary_row_cut_capacity_vertical(
        state=initial_state,
        boundary_src_row=boundary_src_row,
        boundary_dst_row=boundary_dst_row,
        search_limit_row=destination_search_limit_row,
        delta_T=delta_T,
    )

    vert_kind: str = str(vert_status["kind"])

    if vert_kind == "success" and int(vert_achieved) >= int(delta_T):
        return (
            vert_state,
            vert_rounds,
            int(vert_achieved),
            _make_status(
                kind="success",
                t_before=t_before,
                t_after=int(vert_status["T_after"]),
                requested_delta_t=delta_T,
                achieved_delta_t=int(vert_achieved),
                n_rounds=len(vert_rounds),
                chosen_mode="vertical",
                horizontal_status_kind=horiz_kind,
                vertical_status_kind=vert_kind,
                blocking_row=None,
                used_outside_target_cols=horiz_used_outside,
            ),
        )

    vertical_made_structural_progress: bool = len(vert_rounds) > 0
    if vertical_made_structural_progress:
        remaining_delta_t: int = max(0, int(delta_T) - int(vert_achieved))

        horiz2_state: np.ndarray
        horiz2_rounds: list[list[Move]]
        horiz2_achieved: int
        horiz2_status: dict[str, int | str | bool | None]
        (
            horiz2_state,
            horiz2_rounds,
            horiz2_achieved,
            horiz2_status,
        ) = ensure_boundary_row_cut_capacity_horizontal(
            state=vert_state,
            boundary_src_row=boundary_src_row,
            boundary_dst_row=boundary_dst_row,
            delta_T=remaining_delta_t,
            target_start_col=target_start_col,
            target_end_col=target_end_col,
        )

        horiz2_kind: str = str(horiz2_status["kind"])
        horiz2_used_outside: bool = bool(
            horiz2_status.get("used_outside_target_cols", False)
        )
        total_rounds_vh: list[list[Move]] = list(vert_rounds) + list(horiz2_rounds)
        total_achieved_vh: int = int(vert_achieved) + int(horiz2_achieved)
        t_after_vh: int = direct_cut_capacity(
            horiz2_state,
            boundary_src_row=boundary_src_row,
            boundary_dst_row=boundary_dst_row,
        )

        if total_achieved_vh >= int(delta_T):
            return (
                horiz2_state,
                total_rounds_vh,
                total_achieved_vh,
                _make_status(
                    kind="success",
                    t_before=t_before,
                    t_after=t_after_vh,
                    requested_delta_t=delta_T,
                    achieved_delta_t=total_achieved_vh,
                    n_rounds=len(total_rounds_vh),
                    chosen_mode="vertical_then_horizontal",
                    horizontal_status_kind=horiz2_kind,
                    vertical_status_kind=vert_kind,
                    blocking_row=None,
                    used_outside_target_cols=horiz2_used_outside,
                ),
            )

        if len(total_rounds_vh) > 0:
            blocking_row_vh: int | None = (
                None
                if vert_status.get("blocking_row") is None
                else int(vert_status["blocking_row"])
            )
            return (
                horiz2_state,
                total_rounds_vh,
                total_achieved_vh,
                _make_status(
                    kind="cannot_solve_within_constraints",
                    t_before=t_before,
                    t_after=t_after_vh,
                    requested_delta_t=delta_T,
                    achieved_delta_t=total_achieved_vh,
                    n_rounds=len(total_rounds_vh),
                    chosen_mode="vertical_then_horizontal",
                    horizontal_status_kind=horiz2_kind,
                    vertical_status_kind=vert_kind,
                    blocking_row=blocking_row_vh,
                    used_outside_target_cols=horiz2_used_outside,
                ),
            )

    if len(horiz_rounds) > 0:
        return (
            horiz_state,
            horiz_rounds,
            int(horiz_achieved),
            _make_status(
                kind="cannot_solve_within_constraints",
                t_before=t_before,
                t_after=int(horiz_status["T_after"]),
                requested_delta_t=delta_T,
                achieved_delta_t=int(horiz_achieved),
                n_rounds=len(horiz_rounds),
                chosen_mode="horizontal",
                horizontal_status_kind=horiz_kind,
                vertical_status_kind=vert_kind,
                blocking_row=None,
                used_outside_target_cols=horiz_used_outside,
            ),
        )

    final_kind: str
    if (
        horiz_kind == "cannot_solve_within_constraints"
        or vert_kind == "cannot_solve_within_constraints"
    ):
        final_kind = "cannot_solve_within_constraints"
    else:
        final_kind = "infeasible_request"

    blocking_row: int | None = (
        None
        if vert_status.get("blocking_row") is None
        else int(vert_status["blocking_row"])
    )

    return (
        initial_state,
        [],
        0,
        _make_status(
            kind=final_kind,
            t_before=t_before,
            t_after=t_before,
            requested_delta_t=delta_T,
            achieved_delta_t=0,
            n_rounds=0,
            chosen_mode="none",
            horizontal_status_kind=horiz_kind,
            vertical_status_kind=vert_kind,
            blocking_row=blocking_row,
            used_outside_target_cols=horiz_used_outside,
        ),
    )
# -----------------------------------------------------------------------------
# Small controller helpers moved to module scope
# -----------------------------------------------------------------------------


def _move_across_rows_batch_goal(
    remaining_R: int,
    C: float,
    n_cols: int,
) -> int:
    """
    Return the minimum worthwhile batch size for the current remaining demand.

    Parameters
    ----------
    remaining_R
        Remaining number of atoms that must cross the cut.
    C
        Batch-size fraction in ``[0, 1)``.
    n_cols
        Number of columns in the array.

    Returns
    -------
    int
        ``min(remaining_R, ceil(C * n_cols))``.
    """
    return min(int(remaining_R), int(np.ceil(C * n_cols)))


def _make_move_across_rows_status(
    kind: str,
    transferred: int,
    remaining_R: int,
    n_rounds: int,
    last_bottleneck: str,
    source_status_kind: str | None,
    cut_status_kind: str | None,
    C: float,
) -> dict[str, int | float | str | bool | None]:
    """
    Build the structured controller status payload.

    Parameters
    ----------
    kind
        Overall controller outcome.
    transferred
        Number of atoms transferred across the cut.
    remaining_R
        Remaining transfer demand.
    n_rounds
        Total number of executed move rounds.
    last_bottleneck
        Last active bottleneck category.
    source_status_kind
        Status propagated from source-side remediation, if any.
    cut_status_kind
        Status propagated from cut-capacity remediation, if any.
    C
        Batch-size fraction used by the controller.

    Returns
    -------
    dict[str, int | float | str | bool | None]
        Structured status dictionary.
    """
    return {
        "kind": kind,
        "transferred": int(transferred),
        "remaining_R": int(remaining_R),
        "n_rounds": int(n_rounds),
        "last_bottleneck": last_bottleneck,
        "source_status_kind": source_status_kind,
        "cut_status_kind": cut_status_kind,
        "C": float(C),
    }


def _move_across_rows_state_signature(
    state: np.ndarray,
    remaining_R: int,
) -> tuple[int, bytes]:
    """
    Build a cycle-detection signature for the outer controller loop.

    Why this exists
    ---------------
    The controller can make locally valid helper progress that does not produce
    monotone global progress toward completing the requested transfer. If the
    system returns to a previously seen ``(state, remaining_R)`` pair, the outer
    loop has entered a cycle and should fail loudly rather than spin forever.

    Parameters
    ----------
    state
        Current occupancy state.
    remaining_R
        Remaining number of atoms to transfer.

    Returns
    -------
    tuple[int, bytes]
        Hashable signature for cycle detection.
    """
    return int(remaining_R), state.tobytes()

def move_across_rows_old(
    state: np.ndarray,
    boundary_src_row: int,
    boundary_dst_row: int,
    source_search_limit_row: int,
    destination_search_limit_row: int,
    R: int,
    C: float,
    target_start_col: int,
    target_end_col: int,
) -> tuple[np.ndarray, list[list[Move]], int, dict[str, int | float | str | bool | None]]:
    """
    Transfer atoms across a single source/destination row cut using helper-guided control.

    Why this exists
    ---------------
    This function is the controller shell for one row-to-row transfer problem in
    BCv2. Its job is not to invent new local move logic, but to orchestrate the
    existing helper micro-objectives in a transparent order:

    1. If the current cut already supports a worthwhile transfer batch, transfer.
    2. Otherwise, if the source boundary row does not have enough supply to justify
       such a batch, repair source supply.
    3. Otherwise, repair cut capacity.
    4. Repeat until the requested transfer count is completed or the current helper
       set cannot make progress.

    The controller uses a minimum worthwhile batch size
    ``batch_goal = min(R_remaining, ceil(C * n_cols))`` so it does not spend time
    executing transfers that are too small to justify the overhead of the episode.

    Parameters
    ----------
    state
        Single-species occupancy array with shape ``(rows, cols, 1)``.
    boundary_src_row
        Source row adjacent to the cut.
    boundary_dst_row
        Destination row adjacent to the cut.
    source_search_limit_row
        Furthest source-side row that source-supply remediation may touch.
    destination_search_limit_row
        Furthest destination-side row that cut-capacity remediation may touch.
    R
        Requested number of atoms to transfer across the cut.
    C
        Batch-size fraction in ``[0, 1)`` used to define the worthwhile transfer
        threshold ``ceil(C * n_cols)``.
    target_start_col
        Inclusive left boundary of the target column window used by horizontal
        cut-capacity remediation.
    target_end_col
        Inclusive right boundary of the target column window used by horizontal
        cut-capacity remediation.

    Returns
    -------
    tuple[np.ndarray, list[list[Move]], int, dict[str, int | float | str | bool | None]]
        ``(new_state, move_rounds, transferred, status)``.

    Raises
    ------
    ValueError
        If the state shape is invalid, ``R`` is negative, ``C`` is outside
        ``[0, 1)``, or the target column interval is invalid.
    IndexError
        If any row index is out of bounds.
    """
    if state.ndim != 3 or state.shape[2] != 1:
        raise ValueError(
            "state must have shape (rows, cols, 1) for BCv2 single-species control."
        )
    if R < 0:
        raise ValueError(f"R must be nonnegative; got {R}.")
    if not (0.0 <= C < 1.0):
        raise ValueError(f"C must satisfy 0 <= C < 1; got {C}.")
    if target_start_col > target_end_col:
        raise ValueError(
            "target_start_col must be <= target_end_col; "
            f"got {target_start_col} > {target_end_col}."
        )

    n_rows: int = int(state.shape[0])
    n_cols: int = int(state.shape[1])

    for name, row in (
        ("boundary_src_row", boundary_src_row),
        ("boundary_dst_row", boundary_dst_row),
        ("source_search_limit_row", source_search_limit_row),
        ("destination_search_limit_row", destination_search_limit_row),
    ):
        if row < 0 or row >= n_rows:
            raise IndexError(f"{name}={row} is out of bounds for {n_rows} rows.")

    def _source_supply(curr_state: np.ndarray) -> int:
        """
        Return source-boundary supply.

        Notes
        -----
        This intentionally uses the boundary source row only. The controller's
        source-side helper is responsible for increasing this quantity when it is
        the active bottleneck.
        """
        return int(np.sum(curr_state[boundary_src_row, :, 0], dtype=np.int64))

    def _batch_goal(remaining_R: int) -> int:
        """
        Return the minimum worthwhile batch size for the current remaining demand.
        """
        return min(remaining_R, int(np.ceil(C * n_cols)))

    def _make_status(
        kind: str,
        transferred: int,
        remaining_R: int,
        n_rounds: int,
        last_bottleneck: str,
        source_status_kind: str | None,
        cut_status_kind: str | None,
    ) -> dict[str, int | float | str | bool | None]:
        """
        Build the structured controller status payload.
        """
        return {
            "kind": kind,
            "transferred": int(transferred),
            "remaining_R": int(remaining_R),
            "n_rounds": int(n_rounds),
            "last_bottleneck": last_bottleneck,
            "source_status_kind": source_status_kind,
            "cut_status_kind": cut_status_kind,
            "C": float(C),
        }

    initial_state: np.ndarray = state.copy()
    if R == 0:
        return (
            initial_state,
            [],
            0,
            _make_status(
                kind="success",
                transferred=0,
                remaining_R=0,
                n_rounds=0,
                last_bottleneck="none",
                source_status_kind=None,
                cut_status_kind=None,
            ),
        )

    work_state: np.ndarray = initial_state.copy()
    all_rounds: list[list[Move]] = []
    transferred_total: int = 0
    remaining_R: int = int(R)

    while remaining_R > 0:
        batch_goal: int = _batch_goal(remaining_R)
        S: int = _source_supply(work_state)
        T: int = direct_cut_capacity(
            state=work_state,
            boundary_src_row=boundary_src_row,
            boundary_dst_row=boundary_dst_row,
        )

        # ------------------------------------------------------------------
        # Branch 1: immediate transfer when the cut already supports a
        # worthwhile batch.
        # ------------------------------------------------------------------
        if T >= batch_goal:
            transfer_count: int = min(remaining_R, T)

            next_state: np.ndarray
            transfer_moves: list[Move]
            n_moved: int
            next_state, transfer_moves, n_moved = perform_transfer(
                work_state,
                remaining=transfer_count,
                boundary_src_row=boundary_src_row,
                boundary_dst_row=boundary_dst_row,
            )

            if n_moved <= 0:
                return (
                    work_state,
                    all_rounds,
                    transferred_total,
                    _make_status(
                        kind="cannot_solve_within_constraints",
                        transferred=transferred_total,
                        remaining_R=remaining_R,
                        n_rounds=len(all_rounds),
                        last_bottleneck="cut_capacity",
                        source_status_kind=None,
                        cut_status_kind="cannot_solve_within_constraints",
                    ),
                )

            all_rounds.append(transfer_moves)
            work_state = next_state
            transferred_total += int(n_moved)
            remaining_R -= int(n_moved)
            continue

        # ------------------------------------------------------------------
        # Branch 2: source-limited, so repair source supply first.
        # ------------------------------------------------------------------
        if S < batch_goal:
            source_state: np.ndarray
            source_rounds: list[list[Move]]
            source_achieved: int
            source_status: dict[str, Any]
            (
                source_state,
                source_rounds,
                source_achieved,
                source_status,
            ) = ensure_source_supply(
                state=work_state,
                boundary_src_row=boundary_src_row,
                search_limit_row=source_search_limit_row,
                delta_S=max(0, batch_goal - S),
            )

            source_kind: str = str(source_status["kind"])
            if source_rounds:
                all_rounds.extend(source_rounds)
                work_state = source_state

            if source_achieved > 0:
                continue

            return (
                work_state,
                all_rounds,
                transferred_total,
                _make_status(
                    kind=(
                        "cannot_solve_within_constraints"
                        if source_kind == "cannot_solve_within_constraints"
                        else "infeasible_request"
                    ),
                    transferred=transferred_total,
                    remaining_R=remaining_R,
                    n_rounds=len(all_rounds),
                    last_bottleneck="source",
                    source_status_kind=source_kind,
                    cut_status_kind=None,
                ),
            )

        # ------------------------------------------------------------------
        # Branch 3: source supply is good enough, so repair cut capacity.
        # ------------------------------------------------------------------
        cut_state: np.ndarray
        cut_rounds: list[list[Move]]
        cut_achieved: int
        cut_status: dict[str, Any]
        (
            cut_state,
            cut_rounds,
            cut_achieved,
            cut_status,
        ) = ensure_cut_capacity(
            state=work_state,
            boundary_src_row=boundary_src_row,
            boundary_dst_row=boundary_dst_row,
            destination_search_limit_row=destination_search_limit_row,
            delta_T=max(0, batch_goal - T),
            target_start_col=target_start_col,
            target_end_col=target_end_col,
        )

        cut_kind: str = str(cut_status["kind"])
        if cut_rounds:
            all_rounds.extend(cut_rounds)
        if cut_achieved > 0:
            work_state = cut_state
            continue

        return (
            work_state,
            all_rounds,
            transferred_total,
            _make_status(
                kind=(
                    "cannot_solve_within_constraints"
                    if cut_kind == "cannot_solve_within_constraints"
                    else "infeasible_request"
                ),
                transferred=transferred_total,
                remaining_R=remaining_R,
                n_rounds=len(all_rounds),
                last_bottleneck="cut_capacity",
                source_status_kind=None,
                cut_status_kind=cut_kind,
            ),
        )

    return (
        work_state,
        all_rounds,
        transferred_total,
        _make_status(
            kind="success",
            transferred=transferred_total,
            remaining_R=remaining_R,
            n_rounds=len(all_rounds),
            last_bottleneck="none",
            source_status_kind=None,
            cut_status_kind=None,
        ),
    )


def move_across_rows(
    state: np.ndarray,
    boundary_src_row: int,
    boundary_dst_row: int,
    source_search_limit_row: int,
    destination_search_limit_row: int,
    R: int,
    C: float,
    target_start_col: int,
    target_end_col: int,
) -> tuple[np.ndarray, list[list[Move]], int, dict[str, int | float | str | bool | None]]:
    """
    Transfer atoms across a single source/destination row cut using helper-guided control.

    Why this exists
    ---------------
    This function is the controller shell for one row-to-row transfer problem in
    BCv2. Its job is not to invent new local move logic, but to orchestrate the
    existing helper micro-objectives in a transparent order:

    1. If the current cut already supports a worthwhile transfer batch, transfer.
    2. Otherwise, if the source boundary row does not have enough supply to justify
       such a batch, repair source supply.
    3. Otherwise, repair cut capacity.
    4. Repeat until the requested transfer count is completed or the current helper
       set cannot make progress.

    The controller uses a minimum worthwhile batch size
    ``batch_goal = min(R_remaining, ceil(C * n_cols))`` so it does not spend time
    executing transfers that are too small to justify the overhead of the episode.

    Parameters
    ----------
    state
        Single-species occupancy array with shape ``(rows, cols, 1)``.
    boundary_src_row
        Source row adjacent to the cut.
    boundary_dst_row
        Destination row adjacent to the cut.
    source_search_limit_row
        Furthest source-side row that source-supply remediation may touch.
    destination_search_limit_row
        Furthest destination-side row that cut-capacity remediation may touch.
    R
        Requested number of atoms to transfer across the cut.
    C
        Batch-size fraction in ``[0, 1)`` used to define the worthwhile transfer
        threshold ``ceil(C * n_cols)``.
    target_start_col
        Inclusive left boundary of the target column window used by horizontal
        cut-capacity remediation.
    target_end_col
        Inclusive right boundary of the target column window used by horizontal
        cut-capacity remediation.

    Returns
    -------
    tuple[np.ndarray, list[list[Move]], int, dict[str, int | float | str | bool | None]]
        ``(new_state, move_rounds, transferred, status)``.

    Raises
    ------
    ValueError
        If the state shape is invalid, ``R`` is negative, ``C`` is outside
        ``[0, 1)``, or the target column interval is invalid.
    IndexError
        If any row index is out of bounds.
    RuntimeError
        If the outer controller loop exceeds its iteration cap or revisits a
        previously seen controller state.
    """
    if state.ndim != 3 or state.shape[2] != 1:
        raise ValueError(
            "state must have shape (rows, cols, 1) for BCv2 single-species control."
        )
    if R < 0:
        raise ValueError(f"R must be nonnegative; got {R}.")
    if not (0.0 <= C < 1.0):
        raise ValueError(f"C must satisfy 0 <= C < 1; got {C}.")
    if target_start_col > target_end_col:
        raise ValueError(
            "target_start_col must be <= target_end_col; "
            f"got {target_start_col} > {target_end_col}."
        )

    n_rows: int = int(state.shape[0])
    n_cols: int = int(state.shape[1])

    for name, row in (
        ("boundary_src_row", boundary_src_row),
        ("boundary_dst_row", boundary_dst_row),
        ("source_search_limit_row", source_search_limit_row),
        ("destination_search_limit_row", destination_search_limit_row),
    ):
        if row < 0 or row >= n_rows:
            raise IndexError(f"{name}={row} is out of bounds for {n_rows} rows.")

    initial_state: np.ndarray = state.copy()
    if R == 0:
        return (
            initial_state,
            [],
            0,
            _make_move_across_rows_status(
                kind="success",
                transferred=0,
                remaining_R=0,
                n_rounds=0,
                last_bottleneck="none",
                source_status_kind=None,
                cut_status_kind=None,
                C=C,
            ),
        )

    work_state: np.ndarray = initial_state.copy()
    all_rounds: list[list[Move]] = []
    transferred_total: int = 0
    remaining_R: int = int(R)

    # Outer-loop safety guards.
    seen_signatures: set[tuple[int, bytes]] = set()
    max_outer_iters: int = max(100, 8 * n_rows * n_cols + 8 * int(R))

    for _ in range(max_outer_iters):
        if remaining_R <= 0:
            return (
                work_state,
                all_rounds,
                transferred_total,
                _make_move_across_rows_status(
                    kind="success",
                    transferred=transferred_total,
                    remaining_R=remaining_R,
                    n_rounds=len(all_rounds),
                    last_bottleneck="none",
                    source_status_kind=None,
                    cut_status_kind=None,
                    C=C,
                ),
            )

        signature: tuple[int, bytes] = _move_across_rows_state_signature(
            work_state,
            remaining_R,
        )
        if signature in seen_signatures:
            raise RuntimeError(
                "move_across_rows entered a repeated controller state without "
                "completing the requested transfer."
            )
        seen_signatures.add(signature)

        batch_goal: int = _move_across_rows_batch_goal(
            remaining_R=remaining_R,
            C=C,
            n_cols=n_cols,
        )
        S: int = source_supply_at_boundary(
            state=work_state,
            boundary_src_row=boundary_src_row,
        )
        T: int = direct_cut_capacity(
            state=work_state,
            boundary_src_row=boundary_src_row,
            boundary_dst_row=boundary_dst_row,
        )

        # ------------------------------------------------------------------
        # Branch 1: immediate transfer when the cut already supports a
        # worthwhile batch.
        # ------------------------------------------------------------------
        if T >= batch_goal:
            transfer_count: int = min(remaining_R, T)

            next_state: np.ndarray
            transfer_moves: list[Move]
            n_moved: int
            next_state, transfer_moves, n_moved = perform_transfer(
                work_state,
                remaining=transfer_count,
                boundary_src_row=boundary_src_row,
                boundary_dst_row=boundary_dst_row,
            )

            if n_moved <= 0:
                return (
                    work_state,
                    all_rounds,
                    transferred_total,
                    _make_move_across_rows_status(
                        kind="cannot_solve_within_constraints",
                        transferred=transferred_total,
                        remaining_R=remaining_R,
                        n_rounds=len(all_rounds),
                        last_bottleneck="cut_capacity",
                        source_status_kind=None,
                        cut_status_kind="cannot_solve_within_constraints",
                        C=C,
                    ),
                )

            all_rounds.append(transfer_moves)
            work_state = next_state
            transferred_total += int(n_moved)
            remaining_R -= int(n_moved)
            continue

        # ------------------------------------------------------------------
        # Branch 2: source-limited, so repair source supply first.
        # ------------------------------------------------------------------
        if S < batch_goal:
            source_state: np.ndarray
            source_rounds: list[list[Move]]
            source_achieved: int
            source_status: dict[str, Any]
            source_deficit: int = max(0, batch_goal - S)
            (
                source_state,
                source_rounds,
                source_achieved,
                source_status,
            ) = ensure_source_supply(
                state=work_state,
                boundary_src_row=boundary_src_row,
                search_limit_row=source_search_limit_row,
                delta_S=source_deficit,
            )

            source_kind: str = str(source_status["kind"])
            if source_rounds:
                all_rounds.extend(source_rounds)
                work_state = source_state

            if source_achieved > 0:
                continue

            # Canonical recursive source-child:
            # if local source-supply repair made no progress, try to import the
            # missing atoms across the adjacent outward cut.
            step_away_from_target: int = (
                -1 if source_search_limit_row < boundary_src_row else 1
            )
            child_boundary_dst_row: int = boundary_src_row
            child_boundary_src_row: int = boundary_src_row + step_away_from_target

            child_is_in_bounds: bool = 0 <= child_boundary_src_row < n_rows
            child_has_smaller_span: bool = (
                abs(source_search_limit_row - child_boundary_src_row)
                < abs(source_search_limit_row - boundary_src_row)
            )

            if (
                child_is_in_bounds
                and child_has_smaller_span
                and source_kind != "insufficient_atoms"
                and source_deficit > 0
            ):
                child_state: np.ndarray
                child_rounds: list[list[Move]]
                child_transferred: int
                child_status: dict[str, Any]
                (
                    child_state,
                    child_rounds,
                    child_transferred,
                    child_status,
                ) = move_across_rows(
                    state=work_state,
                    boundary_src_row=child_boundary_src_row,
                    boundary_dst_row=child_boundary_dst_row,
                    source_search_limit_row=source_search_limit_row,
                    destination_search_limit_row=boundary_src_row,
                    R=source_deficit,
                    C=C,
                    target_start_col=target_start_col,
                    target_end_col=target_end_col,
                )

                if child_rounds:
                    all_rounds.extend(child_rounds)

                if child_transferred > 0:
                    work_state = child_state
                    continue

            return (
                work_state,
                all_rounds,
                transferred_total,
                _make_move_across_rows_status(
                    kind=(
                        "cannot_solve_within_constraints"
                        if source_kind == "cannot_solve_within_constraints"
                        else "infeasible_request"
                    ),
                    transferred=transferred_total,
                    remaining_R=remaining_R,
                    n_rounds=len(all_rounds),
                    last_bottleneck="source",
                    source_status_kind=source_kind,
                    cut_status_kind=None,
                    C=C,
                ),
            )

        # ------------------------------------------------------------------
        # Branch 3: source supply is good enough, so repair cut capacity.
        # ------------------------------------------------------------------
        cut_state: np.ndarray
        cut_rounds: list[list[Move]]
        cut_achieved: int
        cut_status: dict[str, Any]
        (
            cut_state,
            cut_rounds,
            cut_achieved,
            cut_status,
        ) = ensure_cut_capacity(
            state=work_state,
            boundary_src_row=boundary_src_row,
            boundary_dst_row=boundary_dst_row,
            destination_search_limit_row=destination_search_limit_row,
            delta_T=max(0, batch_goal - T),
            target_start_col=target_start_col,
            target_end_col=target_end_col,
        )

        cut_kind: str = str(cut_status["kind"])
        if cut_rounds:
            all_rounds.extend(cut_rounds)
            work_state = cut_state
            continue

        # if cut_achieved > 0:
        #     work_state = cut_state
        #     continue

        return (
            work_state,
            all_rounds,
            transferred_total,
            _make_move_across_rows_status(
                kind=(
                    "cannot_solve_within_constraints"
                    if cut_kind == "cannot_solve_within_constraints"
                    else "infeasible_request"
                ),
                transferred=transferred_total,
                remaining_R=remaining_R,
                n_rounds=len(all_rounds),
                last_bottleneck="cut_capacity",
                source_status_kind=None,
                cut_status_kind=cut_kind,
                C=C,
            ),
        )

    raise RuntimeError(
        "move_across_rows exceeded its outer iteration cap without completing the "
        "requested transfer."
    )


## debuggers

@dataclass
class MoveAcrossRowsTraceStep:
    iteration: int
    remaining_R_before: int
    batch_goal: int
    source_supply: int
    cut_capacity: int
    branch: str
    requested_delta: int
    helper_rounds: int
    transferred_this_step: int
    remaining_R_after: int
    source_status_kind: str | None
    cut_status_kind: str | None
    note: str


def _state_signature(state: np.ndarray, remaining_R: int) -> tuple[int, bytes]:
    """Return a cycle-detection signature for the traced controller state."""
    return int(remaining_R), state.tobytes()


def trace_move_across_rows(
    state: np.ndarray,
    boundary_src_row: int,
    boundary_dst_row: int,
    source_search_limit_row: int,
    destination_search_limit_row: int,
    R: int,
    C: float,
    target_start_col: int,
    target_end_col: int,
    max_iters: int = 500,
):
    """Run a traced copy of move_across_rows and record one row per outer iteration."""
    if state.ndim != 3 or state.shape[2] != 1:
        raise ValueError("state must have shape (rows, cols, 1).")

    n_rows: int = int(state.shape[0])
    n_cols: int = int(state.shape[1])
    for name, row in (
        ("boundary_src_row", boundary_src_row),
        ("boundary_dst_row", boundary_dst_row),
        ("source_search_limit_row", source_search_limit_row),
        ("destination_search_limit_row", destination_search_limit_row),
    ):
        if row < 0 or row >= n_rows:
            raise IndexError(f"{name}={row} out of bounds for {n_rows} rows.")

    work_state: np.ndarray = state.copy()
    all_rounds: list[list[Move]] = []
    transferred_total: int = 0
    remaining_R: int = int(R)
    trace: list[MoveAcrossRowsTraceStep] = []
    seen: set[tuple[int, bytes]] = set()

    if remaining_R == 0:
        status = {
            "kind": "success",
            "transferred": 0,
            "remaining_R": 0,
            "n_rounds": 0,
            "last_bottleneck": "none",
            "source_status_kind": None,
            "cut_status_kind": None,
            "C": float(C),
        }
        return work_state, all_rounds, transferred_total, status, trace

    for it in range(max_iters):
        sig = _state_signature(work_state, remaining_R)
        if sig in seen:
            status = {
                "kind": "cycle_detected",
                "transferred": transferred_total,
                "remaining_R": remaining_R,
                "n_rounds": len(all_rounds),
                "last_bottleneck": "none",
                "source_status_kind": None,
                "cut_status_kind": None,
                "C": float(C),
            }
            trace.append(
                MoveAcrossRowsTraceStep(
                    iteration=it,
                    remaining_R_before=remaining_R,
                    batch_goal=-1,
                    source_supply=-1,
                    cut_capacity=-1,
                    branch="cycle",
                    requested_delta=0,
                    helper_rounds=0,
                    transferred_this_step=0,
                    remaining_R_after=remaining_R,
                    source_status_kind=None,
                    cut_status_kind=None,
                    note="repeated outer controller state",
                )
            )
            return work_state, all_rounds, transferred_total, status, trace
        seen.add(sig)

        batch_goal: int = _move_across_rows_batch_goal(
            remaining_R=remaining_R,
            C=C,
            n_cols=n_cols,
        )
        S: int = source_supply_at_boundary(
            state=work_state,
            boundary_src_row=boundary_src_row,
        )
        T: int = direct_cut_capacity(
            state=work_state,
            boundary_src_row=boundary_src_row,
            boundary_dst_row=boundary_dst_row,
        )

        if T >= batch_goal:
            transfer_count: int = min(remaining_R, T)
            next_state, transfer_moves, n_moved = perform_transfer(
                work_state,
                remaining=transfer_count,
                boundary_src_row=boundary_src_row,
                boundary_dst_row=boundary_dst_row,
            )
            if n_moved <= 0:
                status = {
                    "kind": "cannot_solve_within_constraints",
                    "transferred": transferred_total,
                    "remaining_R": remaining_R,
                    "n_rounds": len(all_rounds),
                    "last_bottleneck": "cut_capacity",
                    "source_status_kind": None,
                    "cut_status_kind": "cannot_solve_within_constraints",
                    "C": float(C),
                }
                trace.append(
                    MoveAcrossRowsTraceStep(
                        iteration=it,
                        remaining_R_before=remaining_R,
                        batch_goal=batch_goal,
                        source_supply=S,
                        cut_capacity=T,
                        branch="transfer",
                        requested_delta=transfer_count,
                        helper_rounds=0,
                        transferred_this_step=0,
                        remaining_R_after=remaining_R,
                        source_status_kind=None,
                        cut_status_kind="cannot_solve_within_constraints",
                        note="perform_transfer returned zero moved",
                    )
                )
                return work_state, all_rounds, transferred_total, status, trace

            all_rounds.append(transfer_moves)
            work_state = next_state
            transferred_total += int(n_moved)
            before_R = remaining_R
            remaining_R -= int(n_moved)

            trace.append(
                MoveAcrossRowsTraceStep(
                    iteration=it,
                    remaining_R_before=before_R,
                    batch_goal=batch_goal,
                    source_supply=S,
                    cut_capacity=T,
                    branch="transfer",
                    requested_delta=transfer_count,
                    helper_rounds=1,
                    transferred_this_step=int(n_moved),
                    remaining_R_after=remaining_R,
                    source_status_kind=None,
                    cut_status_kind=None,
                    note="direct transfer",
                )
            )

            if remaining_R <= 0:
                status = {
                    "kind": "success",
                    "transferred": transferred_total,
                    "remaining_R": remaining_R,
                    "n_rounds": len(all_rounds),
                    "last_bottleneck": "none",
                    "source_status_kind": None,
                    "cut_status_kind": None,
                    "C": float(C),
                }
                return work_state, all_rounds, transferred_total, status, trace
            continue

        if S < batch_goal:
            req: int = max(0, batch_goal - S)
            source_state, source_rounds, source_achieved, source_status = ensure_source_supply(
                state=work_state,
                boundary_src_row=boundary_src_row,
                search_limit_row=source_search_limit_row,
                delta_S=req,
            )
            source_kind: str = str(source_status["kind"])

            trace.append(
                MoveAcrossRowsTraceStep(
                    iteration=it,
                    remaining_R_before=remaining_R,
                    batch_goal=batch_goal,
                    source_supply=S,
                    cut_capacity=T,
                    branch="source",
                    requested_delta=req,
                    helper_rounds=len(source_rounds),
                    transferred_this_step=0,
                    remaining_R_after=remaining_R,
                    source_status_kind=source_kind,
                    cut_status_kind=None,
                    note=f"achieved_delta_S={int(source_achieved)}",
                )
            )

            if source_rounds:
                all_rounds.extend(source_rounds)
                work_state = source_state

            if source_achieved > 0:
                continue

            status = {
                "kind": (
                    "cannot_solve_within_constraints"
                    if source_kind == "cannot_solve_within_constraints"
                    else "infeasible_request"
                ),
                "transferred": transferred_total,
                "remaining_R": remaining_R,
                "n_rounds": len(all_rounds),
                "last_bottleneck": "source",
                "source_status_kind": source_kind,
                "cut_status_kind": None,
                "C": float(C),
            }
            return work_state, all_rounds, transferred_total, status, trace

        req = max(0, batch_goal - T)
        cut_state, cut_rounds, cut_achieved, cut_status = ensure_cut_capacity(
            state=work_state,
            boundary_src_row=boundary_src_row,
            boundary_dst_row=boundary_dst_row,
            destination_search_limit_row=destination_search_limit_row,
            delta_T=req,
            target_start_col=target_start_col,
            target_end_col=target_end_col,
        )
        if cut_rounds:
            all_rounds.extend(cut_rounds)
        cut_kind: str = str(cut_status["kind"])

        trace.append(
            MoveAcrossRowsTraceStep(
                iteration=it,
                remaining_R_before=remaining_R,
                batch_goal=batch_goal,
                source_supply=S,
                cut_capacity=T,
                branch="cut",
                requested_delta=req,
                helper_rounds=len(cut_rounds),
                transferred_this_step=0,
                remaining_R_after=remaining_R,
                source_status_kind=None,
                cut_status_kind=cut_kind,
                note=f"achieved_delta_T={int(cut_achieved)}",
            )
        )

        if cut_achieved > 0:
            work_state = cut_state
            continue

        status = {
            "kind": (
                "cannot_solve_within_constraints"
                if cut_kind == "cannot_solve_within_constraints"
                else "infeasible_request"
            ),
            "transferred": transferred_total,
            "remaining_R": remaining_R,
            "n_rounds": len(all_rounds),
            "last_bottleneck": "cut_capacity",
            "source_status_kind": None,
            "cut_status_kind": cut_kind,
            "C": float(C),
        }
        return work_state, all_rounds, transferred_total, status, trace

    status = {
        "kind": "iteration_cap_hit",
        "transferred": transferred_total,
        "remaining_R": remaining_R,
        "n_rounds": len(all_rounds),
        "last_bottleneck": "none",
        "source_status_kind": None,
        "cut_status_kind": None,
        "C": float(C),
    }
    return work_state, all_rounds, transferred_total, status, trace


def trace_to_dataframe(trace: list[MoveAcrossRowsTraceStep]) -> pd.DataFrame:
    """Convert a move_across_rows trace to a DataFrame for easier inspection."""
    return pd.DataFrame([step.__dict__ for step in trace])
