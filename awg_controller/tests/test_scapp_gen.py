"""
Tests for the SCAPP GPU-generation backend (``awg_controller.src.scapp_gen``).

Pure phase/amplitude math is exercised with plain numpy (no cupy/hardware
required); ``ScappFeeder``'s schedule-transition logic is exercised via a
"bare" instance constructed with ``__new__`` (bypassing ``__init__``, which
requires real hardware) so the exact production transition code is under
test, not a reimplementation of it.
"""

import math
import os
import sys
import threading

import numpy as np
import pytest
from scipy.integrate import quad

# Ensure repo root is on sys.path (mirrors test_controller_pipeline.py)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")

from awg_controller.src.awg_control import AODSettings, AWGBatch, RFRamp
from awg_controller.src.scapp_gen import (
    TWO_PI,
    ScappFeeder,
    ScappFeederConfig,
    ToneSegment,
    segment_instantaneous_phase,
    segment_total_phase,
)


def _seg(
    shape,
    f_start,
    f_end,
    duration_s,
    start_sample=0,
    phase_offset_rad=0.0,
    static_phase_rad=0.0,
    amplitude_pct=40.0,
):
    return ToneSegment(
        channel=0,
        tone_index=0,
        shape=shape,
        f_start=f_start,
        f_end=f_end,
        duration_s=duration_s,
        amplitude_pct=amplitude_pct,
        phase_offset_rad=phase_offset_rad,
        static_phase_rad=static_phase_rad,
        start_sample=start_sample,
    )


def _angle_close(a: float, b: float, tol: float = 1e-6) -> bool:
    diff = (a - b + math.pi) % TWO_PI - math.pi
    return abs(diff) < tol


# =====================================================================
# 1. Pure phase-math tests (hold / linear / scurve)
# =====================================================================


class TestScappSegmentMath:
    def test_hold_phase_matches_constant_frequency(self):
        seg = _seg("hold", f_start=70e6, f_end=70e6, duration_s=0.0)
        t = np.array([0.0, 1e-7, 5e-7, 1e-6])
        phase = segment_instantaneous_phase(seg, t)
        expected = TWO_PI * 70e6 * t
        np.testing.assert_allclose(phase, expected)

    def test_linear_phase_matches_numeric_integration(self):
        f_start, f_end, duration = 70e6, 90e6, 2e-6
        seg = _seg("linear", f_start, f_end, duration)

        def f(tau):
            if tau > duration:
                return f_end
            return f_start + (f_end - f_start) / duration * tau

        for t_val in [
            0.0,
            5e-7,
            1e-6,
            2e-6,
            3e-6,
        ]:  # last two probe the past-duration tail
            expected, _ = quad(f, 0.0, t_val, limit=200)
            expected_phase = TWO_PI * expected
            actual = float(segment_instantaneous_phase(seg, np.array([t_val]))[0])
            assert actual == pytest.approx(expected_phase, rel=1e-6, abs=1e-3)

    def test_scurve_phase_matches_numeric_integration(self):
        f_start, f_end, duration = 70e6, 90e6, 2e-6
        seg = _seg("scurve", f_start, f_end, duration)

        def f(tau):
            if tau > duration:
                return f_end
            return (
                f_start
                + (f_end - f_start) * (1 - math.cos(math.pi * tau / duration)) / 2
            )

        for t_val in [0.0, 5e-7, 1e-6, 2e-6, 3e-6]:
            expected, _ = quad(f, 0.0, t_val, limit=200)
            expected_phase = TWO_PI * expected
            actual = float(segment_instantaneous_phase(seg, np.array([t_val]))[0])
            assert actual == pytest.approx(expected_phase, rel=1e-6, abs=1e-3)

    def test_scurve_matches_linear_when_endpoints_equal(self):
        """Zero-motion segments (f_start == f_end) reduce to a constant
        frequency term regardless of shape (Δf == 0 collapses both ramp
        formulas to the same thing)."""
        seg_lin = _seg("linear", 75e6, 75e6, 1e-6)
        seg_s = _seg("scurve", 75e6, 75e6, 1e-6)
        t = np.array([0.0, 3e-7, 1e-6, 2e-6])
        np.testing.assert_allclose(
            segment_instantaneous_phase(seg_lin, t),
            segment_instantaneous_phase(seg_s, t),
        )

    def test_unknown_shape_raises(self):
        seg = _seg("bogus", 70e6, 70e6, 0.0)
        with pytest.raises(ValueError):
            segment_instantaneous_phase(seg, np.array([0.0]))


# =====================================================================
# 2. ScappFeeder schedule-transition tests (no hardware required)
# =====================================================================


def _bare_feeder(
    sample_rate_hz: float = 1e9, ramp_shape: str = "linear"
) -> ScappFeeder:
    """A ScappFeeder with only the state needed by ``_seed_segments`` /
    ``_transition_segments`` populated — bypasses ``__init__`` (and thus
    the real hardware/GPU requirement) via ``__new__``.
    """
    feeder = ScappFeeder.__new__(ScappFeeder)
    feeder._state_lock = threading.Lock()
    feeder._sample_rate_hz = sample_rate_hz
    feeder._config = ScappFeederConfig(ramp_shape=ramp_shape)
    feeder._active_segments = {}
    feeder._next_fill_sample = 0
    feeder._feeder_exception = None
    return feeder


def _single_tone_batch(f_start, f_end, duration_s, amplitude_pct=40.0, phase_deg=0.0):
    ramp = RFRamp(
        channel=0,
        core=0,
        f_start=f_start,
        f_end=f_end,
        amplitude_pct=amplitude_pct,
        phase_deg=phase_deg,
        tone_index=0,
    )
    return AWGBatch(ramps=[ramp], total_duration_s=duration_s)


class TestScappFeederPhaseContinuity:
    """Critical regression test: phase must stay continuous (mod 2*pi) at
    every schedule transition, chained across hold -> linear -> scurve ->
    hold, using the exact production ``_transition_segments`` mechanism.
    """

    def _assert_continuous_transition(self, feeder, batch):
        transition_sample = feeder._next_fill_sample
        old_seg = feeder._active_segments[(0, 0)]
        feeder._transition_segments(batch)
        new_seg = feeder._active_segments[(0, 0)]

        t_old = (transition_sample - old_seg.start_sample) / feeder._sample_rate_hz
        phase_end_old = float(segment_total_phase(old_seg, np.array([t_old]))[0])
        phase_start_new = float(segment_total_phase(new_seg, np.array([0.0]))[0])
        assert _angle_close(
            phase_end_old, phase_start_new
        ), f"phase discontinuity at transition: {phase_end_old} vs {phase_start_new}"
        return new_seg

    def test_transition_chain_hold_linear_scurve_hold(self):
        feeder = _bare_feeder(sample_rate_hz=1e9, ramp_shape="linear")
        feeder._seed_segments(_single_tone_batch(70e6, 70e6, 0.0))

        # hold -> linear, after some hold samples have already streamed
        feeder._next_fill_sample = 1000
        self._assert_continuous_transition(feeder, _single_tone_batch(70e6, 90e6, 1e-6))

        # linear -> scurve, well past the linear ramp's own duration
        feeder._config.ramp_shape = "scurve"
        feeder._next_fill_sample += 2_000_000
        self._assert_continuous_transition(feeder, _single_tone_batch(90e6, 60e6, 3e-6))

        # scurve -> hold
        feeder._next_fill_sample += 5_000_000
        self._assert_continuous_transition(feeder, _single_tone_batch(60e6, 60e6, 0.0))

    def test_transition_with_nonzero_static_phase_held_constant(self):
        """RFConverter always emits phase_deg=0 for every ramp it produces,
        so the only static-phase scenario that occurs in practice is a
        constant per-tone offset across a transition — verify that case
        stays continuous. (A *changing* static phase between batches is a
        deliberate discontinuity, not a bug — not exercised here.)
        """
        feeder = _bare_feeder(sample_rate_hz=1e9, ramp_shape="linear")
        ramp = RFRamp(
            channel=0,
            core=0,
            f_start=70e6,
            f_end=70e6,
            amplitude_pct=40.0,
            phase_deg=45.0,
            tone_index=0,
        )
        feeder._seed_segments(AWGBatch(ramps=[ramp], total_duration_s=0.0))
        feeder._next_fill_sample = 12345
        self._assert_continuous_transition(
            feeder,
            AWGBatch(
                ramps=[
                    RFRamp(
                        channel=0,
                        core=0,
                        f_start=70e6,
                        f_end=95e6,
                        amplitude_pct=40.0,
                        phase_deg=45.0,
                        tone_index=0,
                    )
                ],
                total_duration_s=5e-7,
            ),
        )

    def test_transition_lands_exactly_on_fill_boundary(self):
        feeder = _bare_feeder()
        feeder._seed_segments(_single_tone_batch(70e6, 70e6, 0.0))
        feeder._next_fill_sample = 524288
        feeder._transition_segments(_single_tone_batch(70e6, 80e6, 1e-6))
        assert feeder._active_segments[(0, 0)].start_sample == 524288

    def test_holding_batch_always_uses_hold_shape(self):
        """A zero-duration batch always becomes a 'hold' segment even when
        the configured ramp_shape is 'scurve' — shape is derived from
        batch.total_duration_s, not blindly from config."""
        feeder = _bare_feeder(ramp_shape="scurve")
        feeder._seed_segments(_single_tone_batch(70e6, 70e6, 0.0))
        feeder._transition_segments(_single_tone_batch(70e6, 70e6, 0.0))
        assert feeder._active_segments[(0, 0)].shape == "hold"


# =====================================================================
# 3. Config defaults
# =====================================================================


class TestScappFeederConfig:
    def test_defaults(self):
        cfg = ScappFeederConfig()
        assert cfg.notify_samples == 512 * 1024
        assert cfg.dma_buffer_samples == 32 * 1024 * 1024
        assert cfg.fill_start_threshold_promille == 800
        assert cfg.sample_rate_hz is None
        assert cfg.ramp_shape == "linear"
        assert 0.0 < cfg.fill_time_warn_fraction <= 1.0
        assert cfg.join_timeout_s > 0.0


# =====================================================================
# 4. Simulation-mode / no-hardware guarantees
# =====================================================================


class TestScappFeederSimulationGuard:
    def test_module_imports_without_hardware_or_gpu(self):
        import awg_controller.src.scapp_gen as sg

        assert sg.ToneSegment is not None
        assert sg.ScappFeeder is not None
        assert sg.segment_instantaneous_phase is not None

    def test_controller_default_backend_runs_in_sim_mode(self):
        sys.path.insert(0, _SCRIPTS_DIR)
        from atommovr_controller import (
            atommovrController,
            HardwareConfig,
            SoftwareConfig,
        )

        sw = SoftwareConfig(
            aod_settings=AODSettings(grid_rows=4, grid_cols=3),
            max_rounds=1,
        )
        hw = HardwareConfig()
        ctrl = atommovrController(sw, hw)
        try:
            assert ctrl.backend == "scapp"
            assert ctrl._card is None
            assert ctrl._feeder is None

            holding = ctrl.rf_converter.holding_config()
            ctrl._output_batch(holding)  # logs + sleeps 0s, must not raise
            ctrl._send_holding()
        finally:
            ctrl.shutdown()
