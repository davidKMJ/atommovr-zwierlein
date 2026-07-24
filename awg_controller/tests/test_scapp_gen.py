"""
Tests for the SCAPP GPU-generation backend (``awg_controller.src.scapp``).

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
from awg_controller.src.scapp import (
    TWO_PI,
    ScappFeeder,
    ScappFeederConfig,
    ToneSegment,
    segment_instantaneous_frequency,
    segment_instantaneous_phase,
    segment_total_phase,
    synthesize_round_frequency_trajectory,
    synthesize_round_waveform,
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


class TestSegmentInstantaneousFrequency:
    """segment_instantaneous_frequency is the analytical f(t) counterpart to
    segment_instantaneous_phase — used for the "pure" (FFT-free) frequency
    trajectory panel in SessionRecorder.save_spectrogram.
    """

    def test_hold_is_constant_f_end_regardless_of_t(self):
        seg = _seg("hold", f_start=70e6, f_end=90e6, duration_s=0.0)
        t = np.array([0.0, 1e-6, 5e-6, 1e3])
        np.testing.assert_allclose(segment_instantaneous_frequency(seg, t), 90e6)

    def test_linear_matches_closed_form(self):
        f_start, f_end, duration = 70e6, 90e6, 2e-6
        seg = _seg("linear", f_start, f_end, duration)
        t = np.array([0.0, 5e-7, 1e-6, 2e-6, 3e-6])  # last one past duration
        expected = np.where(
            t <= duration, f_start + (f_end - f_start) * (t / duration), f_end
        )
        np.testing.assert_allclose(
            segment_instantaneous_frequency(seg, t), expected, rtol=1e-9
        )

    def test_scurve_matches_closed_form(self):
        f_start, f_end, duration = 70e6, 90e6, 2e-6
        seg = _seg("scurve", f_start, f_end, duration)
        t = np.array([0.0, 5e-7, 1e-6, 2e-6, 3e-6])
        t_c = np.minimum(t, duration)
        expected = f_start + 0.5 * (f_end - f_start) * (
            1.0 - np.cos(math.pi * t_c / duration)
        )
        np.testing.assert_allclose(
            segment_instantaneous_frequency(seg, t), expected, rtol=1e-9
        )

    @pytest.mark.parametrize("shape", ["linear", "scurve"])
    def test_matches_derivative_of_phase(self, shape):
        """f(t) must be the derivative of segment_instantaneous_phase's
        integral — cross-checked numerically away from the t=0 edge and the
        t=duration kink, where one-sided finite differences are inherently
        imprecise regardless of formula correctness.
        """
        seg = _seg(shape, f_start=70e6, f_end=90e6, duration_s=2e-6)
        t = np.linspace(0.0, 3e-6, 3000)
        dt = t[1] - t[0]
        f_analytic = segment_instantaneous_frequency(seg, t)
        f_numeric = np.gradient(segment_instantaneous_phase(seg, t), dt) / TWO_PI

        kink_idx = int(round(seg.duration_s / dt))
        interior = np.ones(t.shape, dtype=bool)
        interior[:5] = False
        interior[-5:] = False
        interior[max(kink_idx - 5, 0) : kink_idx + 5] = False
        assert np.max(np.abs(f_analytic - f_numeric)[interior]) < 50.0  # Hz

    def test_scurve_matches_linear_when_endpoints_equal(self):
        seg_lin = _seg("linear", 75e6, 75e6, 1e-6)
        seg_s = _seg("scurve", 75e6, 75e6, 1e-6)
        t = np.array([0.0, 3e-7, 1e-6, 2e-6])
        np.testing.assert_allclose(
            segment_instantaneous_frequency(seg_lin, t),
            segment_instantaneous_frequency(seg_s, t),
        )
        np.testing.assert_allclose(segment_instantaneous_frequency(seg_lin, t), 75e6)

    def test_unknown_shape_raises(self):
        seg = _seg("bogus", 70e6, 70e6, 0.0)
        with pytest.raises(ValueError):
            segment_instantaneous_frequency(seg, np.array([0.0]))


class TestSynthesizeRoundFrequencyTrajectory:
    """synthesize_round_frequency_trajectory: the analytical, FFT-free
    per-tone f(t) used for the "pure" frequency trajectory spectrogram panel.
    """

    def test_single_moving_batch(self):
        ramp = RFRamp(
            channel=0,
            core=0,
            f_start=70e6,
            f_end=90e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        batches = [AWGBatch(ramps=[ramp], total_duration_s=2e-6)]
        traj = synthesize_round_frequency_trajectory(
            batches, ramp_shape="linear", points_per_batch=5
        )
        assert set(traj) == {(0, 0)}
        times, freqs = traj[(0, 0)]
        assert times[0] == pytest.approx(0.0)
        assert times[-1] == pytest.approx(2e-6)
        assert freqs[0] == pytest.approx(70e6)
        assert freqs[-1] == pytest.approx(90e6)
        assert np.all(np.diff(freqs) >= 0)  # monotonically increasing

    def test_non_moving_ramp_is_flat(self):
        ramp = RFRamp(
            channel=0,
            core=0,
            f_start=80e6,
            f_end=80e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        batches = [AWGBatch(ramps=[ramp], total_duration_s=2e-6)]
        _times, freqs = synthesize_round_frequency_trajectory(batches)[(0, 0)]
        np.testing.assert_allclose(freqs, 80e6)

    def test_cumulative_time_across_batches(self):
        """A hold batch (duration=0) contributes a single point and doesn't
        advance the time offset; subsequent batches' times stack on top of
        prior batches' actual durations.
        """
        ramp1 = RFRamp(
            channel=0,
            core=0,
            f_start=70e6,
            f_end=90e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        ramp2 = RFRamp(
            channel=0,
            core=0,
            f_start=90e6,
            f_end=90e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        ramp3 = RFRamp(
            channel=0,
            core=0,
            f_start=90e6,
            f_end=70e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        batches = [
            AWGBatch(ramps=[ramp1], total_duration_s=2e-6),
            AWGBatch(ramps=[ramp2], total_duration_s=0.0),
            AWGBatch(ramps=[ramp3], total_duration_s=3e-6),
        ]
        times, freqs = synthesize_round_frequency_trajectory(
            batches, points_per_batch=4
        )[(0, 0)]
        assert times[-1] == pytest.approx(5e-6)  # 2e-6 + 0 + 3e-6
        assert freqs.max() == pytest.approx(90e6)
        assert freqs.min() == pytest.approx(70e6)
        assert freqs[-1] == pytest.approx(70e6)  # ends back at 70 MHz

    def test_empty_batches_returns_empty(self):
        assert synthesize_round_frequency_trajectory([]) == {}

    def test_regression_break_inserted_at_genuine_discontinuity(self):
        """RFConverter can rebuild a non-targeted tone's ramp from its
        nominal resting frequency rather than where an earlier batch in the
        same round actually left it — a real commanded frequency jump, not
        a continuous ramp. The trajectory must insert a NaN there so a line
        plot doesn't draw a misleading connecting "drop", instead of
        silently connecting f_end of one batch to a mismatched f_start of
        the next.
        """
        moved_away = RFRamp(
            channel=0,
            core=0,
            f_start=70e6,
            f_end=90e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        # Next batch doesn't target tone_index=0: rebuilt as a "hold" at its
        # *nominal* 70 MHz, not the 90 MHz it actually reached above.
        reset_to_nominal = RFRamp(
            channel=0,
            core=0,
            f_start=70e6,
            f_end=70e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        batches = [
            AWGBatch(ramps=[moved_away], total_duration_s=2e-6),
            AWGBatch(ramps=[reset_to_nominal], total_duration_s=2e-6),
        ]
        times, freqs = synthesize_round_frequency_trajectory(batches)[(0, 0)]
        nan_idx = np.where(np.isnan(freqs))[0]
        assert nan_idx.size == 1
        # The break sits exactly at the batch boundary (t=2e-6), separating
        # the first batch's ramp-up from the second's mismatched restart.
        assert times[nan_idx[0]] == pytest.approx(2e-6)
        assert freqs[nan_idx[0] - 1] == pytest.approx(90e6)  # end of batch 1
        assert freqs[nan_idx[0] + 1] == pytest.approx(70e6)  # start of batch 2

    def test_no_break_when_batches_are_frequency_continuous(self):
        """The normal, well-behaved case (f_start matches the previous
        batch's f_end) must not be broken -- only genuine mismatches are.
        """
        ramp1 = RFRamp(
            channel=0,
            core=0,
            f_start=70e6,
            f_end=90e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        ramp2 = RFRamp(
            channel=0,
            core=0,
            f_start=90e6,
            f_end=70e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        batches = [
            AWGBatch(ramps=[ramp1], total_duration_s=2e-6),
            AWGBatch(ramps=[ramp2], total_duration_s=2e-6),
        ]
        _times, freqs = synthesize_round_frequency_trajectory(batches)[(0, 0)]
        assert not np.any(np.isnan(freqs))

    def test_break_tol_hz_controls_mismatch_threshold(self):
        """A tiny f_start/f_end mismatch within break_tol_hz must not be
        treated as a real discontinuity."""
        ramp1 = RFRamp(
            channel=0,
            core=0,
            f_start=70e6,
            f_end=90e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        ramp2 = RFRamp(
            channel=0,
            core=0,
            f_start=90e6 + 0.1,  # 0.1 Hz off from the true prior f_end
            f_end=70e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        batches = [
            AWGBatch(ramps=[ramp1], total_duration_s=2e-6),
            AWGBatch(ramps=[ramp2], total_duration_s=2e-6),
        ]
        _times, freqs = synthesize_round_frequency_trajectory(
            batches, break_tol_hz=1.0
        )[(0, 0)]
        assert not np.any(np.isnan(freqs))


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
    feeder._last_transition_sample = None
    feeder._dropped_transition_count = 0
    feeder._last_dropped_warn_s = 0.0
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

    def test_regression_dropped_transition_counted_when_submits_outrun_fill_loop(self):
        """Two submit_batch-equivalent calls landing before _next_fill_sample
        advances means the fill loop never rendered the first one's segments
        — it never reached the DAC. This is the exact mechanism by which a
        burst of moves shorter than one notify_samples window can silently
        vanish; _transition_segments must detect and count it.
        """
        feeder = _bare_feeder()
        feeder._seed_segments(_single_tone_batch(70e6, 70e6, 0.0))
        assert feeder.dropped_transition_count == 0

        # _next_fill_sample untouched between these two calls, mimicking the
        # fill loop not having advanced past the target boundary yet.
        feeder._transition_segments(_single_tone_batch(70e6, 80e6, 3e-6))
        feeder._transition_segments(_single_tone_batch(80e6, 90e6, 3e-6))
        assert feeder.dropped_transition_count == 1

        # A third call in the same still-unadvanced window drops again.
        feeder._transition_segments(_single_tone_batch(90e6, 100e6, 3e-6))
        assert feeder.dropped_transition_count == 2

    def test_no_drop_counted_when_fill_loop_advances_between_submits(self):
        """The normal, non-dropping case: the fill loop advances
        _next_fill_sample (as it does once per rendered chunk in
        _fill_loop) before the next submit_batch call arrives.
        """
        feeder = _bare_feeder()
        feeder._seed_segments(_single_tone_batch(70e6, 70e6, 0.0))

        feeder._transition_segments(_single_tone_batch(70e6, 80e6, 3e-6))
        feeder._next_fill_sample += feeder._config.notify_samples
        feeder._transition_segments(_single_tone_batch(80e6, 90e6, 3e-6))

        assert feeder.dropped_transition_count == 0

    def test_seed_segments_resets_drop_tracking(self):
        """A fresh _seed_segments (e.g. feeder restart) must not treat the
        first post-seed _transition_segments call as a drop just because it
        lands at the same sample (0) a pre-restart call already used.
        """
        feeder = _bare_feeder()
        feeder._seed_segments(_single_tone_batch(70e6, 70e6, 0.0))
        feeder._transition_segments(_single_tone_batch(70e6, 80e6, 3e-6))
        assert feeder.dropped_transition_count == 0  # single call, nothing to drop yet

        feeder._seed_segments(_single_tone_batch(80e6, 80e6, 0.0))
        assert feeder._last_transition_sample is None
        feeder._transition_segments(_single_tone_batch(80e6, 90e6, 3e-6))
        assert (
            feeder.dropped_transition_count == 0
        )  # reset by _seed_segments, not a drop

    def test_regression_submit_batch_sleeps_extra_notify_period(self, monkeypatch):
        """submit_batch's sleep must cover the move's own duration *plus*
        one full notify_samples window — a cheap safeguard so the *next*
        submission is very likely to land after the fill loop has already
        advanced past this one's ticket (see
        _maybe_warn_dropped_transition), instead of racing it.
        """
        import awg_controller.src.scapp as scapp_module

        feeder = _bare_feeder(sample_rate_hz=1e9)
        feeder._config = ScappFeederConfig(notify_samples=1024, sample_rate_hz=1e9)
        feeder._seed_segments(_single_tone_batch(70e6, 70e6, 0.0))

        sleeps = []
        monkeypatch.setattr(scapp_module.time, "sleep", lambda s: sleeps.append(s))

        feeder.submit_batch(_single_tone_batch(70e6, 80e6, 3e-6))

        notify_period_s = feeder._config.notify_samples / feeder._sample_rate_hz
        assert len(sleeps) == 1
        assert sleeps[0] == pytest.approx(3e-6 + notify_period_s)

    def test_regression_submit_batch_sleeps_at_least_one_notify_period_for_zero_duration(
        self, monkeypatch
    ):
        """Even a nominally zero-duration batch sent via submit_batch (not
        the usual submit_holding path) must still wait out one notify
        period — otherwise its own segments could be overwritten before
        the fill loop ever renders them, same failure mode as a real move.
        """
        import awg_controller.src.scapp as scapp_module

        feeder = _bare_feeder(sample_rate_hz=1e9)
        feeder._config = ScappFeederConfig(notify_samples=1024, sample_rate_hz=1e9)
        feeder._seed_segments(_single_tone_batch(70e6, 70e6, 0.0))

        sleeps = []
        monkeypatch.setattr(scapp_module.time, "sleep", lambda s: sleeps.append(s))

        feeder.submit_batch(_single_tone_batch(70e6, 70e6, 0.0))

        notify_period_s = feeder._config.notify_samples / feeder._sample_rate_hz
        assert len(sleeps) == 1
        assert sleeps[0] == pytest.approx(notify_period_s)


# =====================================================================
# 2b. Offline waveform synthesis (for spectrogram visualization)
# =====================================================================


class TestSynthesizeRoundWaveform:
    def test_empty_batches_returns_empty_dict(self):
        assert synthesize_round_waveform([], sample_rate_hz=1e6) == {}

    def test_sample_counts_match_durations(self):
        batches = [
            _single_tone_batch(70e6, 70e6, 2e-6),
            _single_tone_batch(70e6, 90e6, 3e-6),
        ]
        waveforms = synthesize_round_waveform(batches, sample_rate_hz=10e6)
        assert set(waveforms) == {0}
        assert waveforms[0].shape == (50,)  # (2+3)us * 10MHz

    def test_hold_tone_is_a_pure_sinusoid_at_f_end(self):
        f = 70e6
        rate = 1e9
        batches = [_single_tone_batch(f, f, 2e-6)]
        waveforms = synthesize_round_waveform(batches, sample_rate_hz=rate)
        samples = waveforms[0]
        t = np.arange(samples.size) / rate
        expected = 0.40 * np.sin(TWO_PI * f * t)
        np.testing.assert_allclose(samples, expected, atol=1e-9)

    def test_multi_channel_batches_produce_one_waveform_per_channel(self):
        ramp0 = RFRamp(
            channel=0,
            core=0,
            f_start=70e6,
            f_end=80e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        ramp1 = RFRamp(
            channel=1,
            core=0,
            f_start=60e6,
            f_end=65e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        batches = [AWGBatch(ramps=[ramp0, ramp1], total_duration_s=1e-6)]
        waveforms = synthesize_round_waveform(batches, sample_rate_hz=10e6)
        assert set(waveforms) == {0, 1}
        assert waveforms[0].shape == waveforms[1].shape == (10,)
        assert not np.allclose(waveforms[0], waveforms[1])

    def test_phase_continuous_across_batch_boundary(self):
        """The waveform's own instantaneous phase (not just the underlying
        ToneSegment bookkeeping already covered by
        TestScappFeederPhaseContinuity) must not jump at a batch boundary."""
        rate = 1e9
        batches = [
            _single_tone_batch(70e6, 70e6, 1e-6),
            _single_tone_batch(70e6, 90e6, 1e-6),
        ]
        samples = synthesize_round_waveform(batches, sample_rate_hz=rate)[0]
        boundary = int(round(1e-6 * rate))
        # Consecutive-sample deltas should be small everywhere, including
        # right at the boundary — a phase jump would show up as an outlier.
        deltas = np.abs(np.diff(samples))
        assert deltas[boundary - 1] < 5 * np.median(deltas) + 1e-3


# =====================================================================
# 3. Config defaults
# =====================================================================


class TestScappFeederConfig:
    def test_defaults(self):
        cfg = ScappFeederConfig()
        assert cfg.notify_samples == 1024
        assert cfg.dma_buffer_samples == 32 * 1024 * 1024
        assert cfg.fill_start_threshold_promille == 800
        assert cfg.sample_rate_hz is None
        assert cfg.ramp_shape == "linear"
        assert 0.0 < cfg.fill_time_warn_fraction <= 1.0
        assert cfg.join_timeout_s > 0.0

    def test_regression_notify_samples_matched_to_m4i_6631x8_and_move_timescale(self):
        """notify_samples sets the granularity at which a submitted move can
        actually take effect on the DAC (see ScappFeederConfig docstring).
        The default must be small enough, at the M4i.6631-x8's real 1.25 GS/s
        max rate, to resolve even MIN_MOVE_DURATION_S (the shortest a move
        batch can ever be, atommovr.utils.timing) — the original 512 KiS
        default gave a ~419 µs update period, over 400x longer than that.
        """
        from atommovr.utils.timing import MIN_MOVE_DURATION_S
        from awg_controller.src.awg_control import M4I_6631_X8_MAX_SAMPLE_RATE_HZ

        cfg = ScappFeederConfig()
        notify_period_s = cfg.notify_samples / M4I_6631_X8_MAX_SAMPLE_RATE_HZ
        assert notify_period_s < MIN_MOVE_DURATION_S

    def test_dma_buffer_must_be_multiple_of_notify_samples(self):
        """The vendored spcm package's automatic buffer handling only works
        correctly when dma_buffer_samples is an exact multiple of
        notify_samples (spcm/classes_data_transfer.py:
        `_pre_buffer_transfer`'s `exception_num_samples` check) — enforced
        here at construction time instead of failing deep inside the driver.
        """
        cfg = ScappFeederConfig()  # defaults must satisfy this themselves
        assert cfg.dma_buffer_samples % cfg.notify_samples == 0

        with pytest.raises(ValueError):
            ScappFeederConfig(notify_samples=1000, dma_buffer_samples=32_000_001)

        # A valid, non-default combination must not raise.
        ScappFeederConfig(notify_samples=1024, dma_buffer_samples=1024 * 100)

    def test_notify_samples_must_be_positive(self):
        with pytest.raises(ValueError):
            ScappFeederConfig(notify_samples=0)


# =====================================================================
# 4. Simulation-mode / no-hardware guarantees
# =====================================================================


class TestScappFeederSimulationGuard:
    def test_module_imports_without_hardware_or_gpu(self):
        import awg_controller.src.scapp as sg

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
