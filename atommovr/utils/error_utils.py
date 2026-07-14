"""
Vectorized utilities for applying error events to `Move` lists.

This module implements fast bitmask updates for error processes without
overwriting previous events.

Workflow
--------
1) Allocate event_mask = zeros(N, uint64)
2) For each error process:
     - build `eligible` boolean mask (shape (N,))
     - call apply_bernoulli_event_inplace(event_mask, eligible, p, bit, rng)
3) Call suppress_inplace(event_mask) from failure_policy
4) Resolve primary events via resolve_primary_events(event_mask)
5) Write primary FailureEvent back into Move objects
"""

from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np
from numpy.typing import NDArray

from atommovr.utils.Move import Move
from atommovr.utils.failure_policy import (
    FailureBit,
    bit_value,
    suppress_inplace,
    resolve_primary_events,
)


def set_event_bit_inplace(
    event_mask: NDArray[np.uint64],
    eligible: NDArray[np.bool_],
    bit: FailureBit,
) -> None:
    """
    Deterministically set a FailureBit for all eligible moves.

    Parameters
    ----------
    event_mask
        Array of shape (N,) with dtype uint64 encoding per-move event bitmasks.
        Updated in-place.
    eligible
        Boolean mask of shape (N,) selecting which moves are affected.
    bit
        FailureBit to set (bit position, not 1<<bit).
    """
    event_size = event_mask.size
    elig_size = eligible.size
    if event_size != elig_size:
        raise ValueError(
            f"Size mismatch between 'event_mask' ({event_size}) and 'eligible' ({elig_size})."
        )
    if event_size == 0:
        return
    if elig_size == 0:
        return
    if event_mask.ndim != 1 or eligible.ndim != 1:
        raise ValueError(
            f"Parameters 'event_mask' and 'eligible' must be 1D arrays, not {event_mask.ndim}D and {eligible.ndim}D."
        )
    event_mask[eligible] |= bit_value(bit)


def apply_bernoulli_event_inplace(
    event_mask: np.ndarray,
    eligible: np.ndarray,
    p_fail: float,
    bit: FailureBit,
    rng: np.random.Generator,
) -> None:
    """
    Vectorized Bernoulli failure application.

    Parameters
    ----------
    event_mask : numpy.ndarray
        uint64 array of shape (N,) mutated in-place; ORs in `bit` for failures.
    eligible : numpy.ndarray
        bool array of shape (N,) selecting moves to consider.
    p_fail : float
        Failure probability in [0, 1].
    bit : FailureBit
        Which failure bit to set for failed eligible moves.
    rng : numpy.random.Generator
        RNG used for sampling (enables deterministic tests).

    Returns
    -------
    None
    """
    # sanity checks
    if event_mask.ndim != 1 or eligible.ndim != 1:
        raise ValueError(
            f"Parameters 'event_mask' and 'eligible' must be 1D arrays, not {event_mask.ndim}D and {eligible.ndim}D."
        )
    if event_mask.size != eligible.size:
        raise ValueError(
            f"Size mismatch between 'event_mask' ({event_mask.size}) and 'eligible' ({eligible.size})."
        )
    if event_mask.size == 0:
        return
    if p_fail == 0.0:
        return
    elif p_fail == 1.0:
        fail = eligible
    elif p_fail < 0 or p_fail > 1:
        raise ValueError(f"Parameter 'p_fail' must be in [0,1], not {p_fail}.")
    else:
        r = rng.random(event_mask.shape[0])
        fail = eligible & (r < p_fail)

    if np.any(fail):
        event_mask[fail] |= bit_value(bit)


def eligible_from_indices(n: int, indices: Sequence[int]) -> np.ndarray:
    """Build a boolean eligibility mask of length n from a list of indices."""
    m = np.zeros(n, dtype=bool)
    if len(indices) > 0:
        m[np.asarray(indices, dtype=np.intp)] = True
    return m


def eligible_from_moves(
    all_moves: Sequence[Move], subset_moves: Iterable[Move]
) -> np.ndarray:
    """
    Build eligibility mask from Move object identity (same object instance).

    Notes
    -----
    This uses object identity, not Move.__eq__. That’s usually what you want
    when subsets are created by filtering the original list.
    """
    n = len(all_moves)
    eligible = np.zeros(n, dtype=bool)
    id_to_idx = {id(m): i for i, m in enumerate(all_moves)}
    for m in subset_moves:
        idx = id_to_idx.get(id(m))
        if idx is not None:
            eligible[idx] = True
    return eligible


def write_primary_events_to_moves(
    moves: Sequence[Move], primary_events: np.ndarray
) -> None:
    """
    Write FailureEvent codes back into Move objects (updates fail_flag too).

    Parameters
    ----------
    moves : Sequence[Move]
        List of Move objects.
    primary_events : numpy.ndarray
        Integer array (N,) of FailureEvent codes.

    Returns
    -------
    None
    """
    for mv, ev in zip(moves, primary_events, strict=True):
        mv.set_failure_event(int(ev))


def finalize_events_to_moves(
    moves: Sequence[Move],
    event_mask: np.ndarray,
    store_mask_on_move: bool = False,
) -> np.ndarray:
    """
    Suppress irrelevant events, resolve primary event, and write to Move objects.

    Parameters
    ----------
    moves : Sequence[Move]
        Move objects to update.
    event_mask : numpy.ndarray
        uint64 event bitmask array.
    store_mask_on_move : bool
        If True, stores the raw bitmask on each Move as `move.fail_mask`.

    Returns
    -------
    primary : numpy.ndarray
        (N,) array of primary FailureEvent codes.
    """
    suppress_inplace(event_mask)
    primary = resolve_primary_events(event_mask)
    write_primary_events_to_moves(moves, primary)

    if store_mask_on_move:
        for mv, mask in zip(moves, event_mask, strict=True):
            mv.fail_mask = int(mask)

    return primary
