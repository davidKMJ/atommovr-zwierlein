"""Shared move-timing utilities for simulation and hardware control.

* **Travel** — Chebyshev site distance × ``spacing`` / ``AOD_speed``.
  Shared by AWG TIMER / slopes / host transport waits and as the travel
  component of sim evolution time.  Floor: :data:`MIN_MOVE_DURATION_S`.

* **Phases** — ErrorModel pickup / accel / decel / putdown (seconds).
  Used only in **sim** ``evolution_time`` / returned ``move_time``.
  Hardware control uses travel alone.

``Move.distance`` remains Euclidean (geometry / failure penalties).
``get_move_distance`` in ``core`` remains Manhattan and is **not** the
transport clock.
"""

from __future__ import annotations

from typing import Any, Sequence

MIN_MOVE_DURATION_S: float = 1e-6


def chebyshev_sites(
    from_row: int,
    from_col: int,
    to_row: int,
    to_col: int,
) -> int:
    """Chebyshev site count (parallel V/H AOD travel)."""
    return max(abs(to_row - from_row), abs(to_col - from_col))


def travel_duration_s(
    moves: Sequence[Any],
    spacing: float,
    AOD_speed: float,
) -> float:
    """Travel window (s) for a parallel move batch.

    Uses the longest Chebyshev distance in *moves*.  Empty → ``0.0``.
    Non-empty results are floored at :data:`MIN_MOVE_DURATION_S`.
    """
    if not moves:
        return 0.0
    max_cheb = max(
        chebyshev_sites(m.from_row, m.from_col, m.to_row, m.to_col) for m in moves
    )
    dist_m = max_cheb * float(spacing)
    duration_s = dist_m / max(float(AOD_speed), 1e-15)
    return max(duration_s, MIN_MOVE_DURATION_S)


def phase_duration_s(
    error_model: Any,
    *,
    pickup: bool = False,
    accel: bool = False,
    decel: bool = False,
    putdown: bool = False,
) -> float:
    """Sum selected ErrorModel phase times (seconds).  Sim evolution only."""
    total = 0.0
    if pickup:
        total += float(getattr(error_model, "pickup_time", 0.0) or 0.0)
    if accel:
        total += float(getattr(error_model, "accel_time", 0.0) or 0.0)
    if decel:
        total += float(getattr(error_model, "decel_time", 0.0) or 0.0)
    if putdown:
        total += float(getattr(error_model, "putdown_time", 0.0) or 0.0)
    return total


def all_phase_duration_s(error_model: Any) -> float:
    """Sum of all four ErrorModel phase times (sim batch overhead)."""
    return phase_duration_s(
        error_model, pickup=True, accel=True, decel=True, putdown=True
    )


def batch_evolution_time_s(
    moves: Sequence[Any],
    spacing: float,
    AOD_speed: float,
    phase_time_s: float = 0.0,
) -> float:
    """Sim evolution / benchmark time: travel + optional phases."""
    return travel_duration_s(moves, spacing, AOD_speed) + max(float(phase_time_s), 0.0)
