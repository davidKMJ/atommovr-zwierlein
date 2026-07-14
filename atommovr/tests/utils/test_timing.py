"""Tests for shared travel / phase timing (atommovr.utils.timing)."""

import pytest

from atommovr.utils.ErrorModel import ErrorModel
from atommovr.utils.Move import Move
from atommovr.utils.timing import (
    MIN_MOVE_DURATION_S,
    all_phase_duration_s,
    batch_evolution_time_s,
    chebyshev_sites,
    phase_duration_s,
    travel_duration_s,
)


class TestChebyshevSites:
    def test_diagonal(self):
        assert chebyshev_sites(0, 0, 2, 3) == 3

    def test_axis_aligned(self):
        assert chebyshev_sites(0, 0, 0, 5) == 5
        assert chebyshev_sites(1, 2, 4, 2) == 3

    def test_zero(self):
        assert chebyshev_sites(1, 1, 1, 1) == 0


class TestTravelDuration:
    def test_empty_zero(self):
        assert travel_duration_s([], spacing=5e-6, AOD_speed=0.1) == 0.0

    def test_one_site(self):
        moves = [Move(0, 0, 1, 0)]
        assert travel_duration_s(moves, 5e-6, 0.1) == pytest.approx(5e-6 / 0.1)

    def test_diagonal_uses_chebyshev_not_euclidean(self):
        moves = [Move(0, 0, 2, 3)]
        # Chebyshev = 3, not sqrt(13)
        assert travel_duration_s(moves, 5e-6, 0.1) == pytest.approx(3 * 5e-6 / 0.1)

    def test_batch_takes_longest(self):
        short = Move(0, 0, 1, 0)
        long_ = Move(0, 0, 5, 5)
        assert travel_duration_s([short, long_], 5e-6, 0.1) == pytest.approx(
            travel_duration_s([long_], 5e-6, 0.1)
        )

    def test_floor(self):
        # Tiny spacing / huge speed would underflow without floor
        moves = [Move(0, 0, 1, 0)]
        dur = travel_duration_s(moves, spacing=1e-15, AOD_speed=1e9)
        assert dur == pytest.approx(MIN_MOVE_DURATION_S)

    def test_aod_speed_inverse(self):
        moves = [Move(0, 0, 1, 0)]
        slow = travel_duration_s(moves, 5e-6, 0.1)
        fast = travel_duration_s(moves, 5e-6, 0.2)
        assert slow == pytest.approx(2 * fast)


class TestPhasesAndEvolution:
    def test_phase_duration_selective(self):
        em = ErrorModel(
            pickup_time=1e-4,
            accel_time=2e-4,
            decel_time=3e-4,
            putdown_time=4e-4,
        )
        assert phase_duration_s(em, pickup=True, putdown=True) == pytest.approx(5e-4)
        assert all_phase_duration_s(em) == pytest.approx(1e-3)

    def test_evolution_is_travel_plus_phases(self):
        moves = [Move(0, 0, 1, 0)]
        travel = travel_duration_s(moves, 5e-6, 0.1)
        evo = batch_evolution_time_s(moves, 5e-6, 0.1, phase_time_s=1e-4)
        assert evo == pytest.approx(travel + 1e-4)
