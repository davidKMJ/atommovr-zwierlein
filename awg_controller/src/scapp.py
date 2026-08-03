"""
SCAPP (CUDA GPU-direct RDMA) generation backend for the Spectrum Instrumentation
AWG card.
===========================================================================

Default AWG backend (see ``dds_strategies.py`` for the legacy DDS backend,
kept as an opt-in alternative). Unlike DDS — where writing one frequency
command lets the FPGA hold that tone indefinitely with no further CPU
involvement — SCAPP requires the host to continuously supply samples to the
card's DMA FIFO in real time. There is no fixed "core count": each AOD
output channel is a software-synthesized sum of sine tones, computed on the
GPU (CuPy) and streamed to the card via RDMA (``spcm.SCAPPTransfer``).

Architecture
------------
A dedicated background thread (`ScappFeeder`) owns the continuous
``for card_buffer in scapp_transfer`` loop and keeps streaming the current
waveform (holding or ramping). The main control-loop thread only calls
``submit_batch``/``submit_holding`` to update what's played next; both stay
synchronous (block for ``batch.total_duration_s``) to match the pacing
contract the legacy `_output_batch`/`_send_holding` methods already have.

Since ``RFConverter.holding_config()``/``convert_moves()`` always emit a
full-grid batch (every tone, every call — the amplitude-safety invariant),
every ``submit_*`` call replaces *every* tone's trajectory simultaneously:
no per-tone bookkeeping divergence, no partial-update races. Phase
continuity across a transition is guaranteed by evaluating the outgoing
segment's closed-form phase at the exact sample the incoming segment takes
over, and carrying that as the incoming segment's phase offset.

Safety
------
* **Maximum output voltage MUST stay below 2.0 V** in all scripts.
* Per-channel tone amplitudes are normalised so their sum never exceeds
  ``MAX_AMPLITUDE_PCT_PER_CHANNEL`` (40 %), bounding the digital sum to
  well within full scale regardless of tone count or phase alignment.
* Always verify amplifier output with an oscilloscope before connecting
  to the AOD. Excessive voltage will damage the AOD driver.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from awg_controller.scripts.atommover_controller import HardwareConfig

from awg_controller.src.awg_control import AODSettings, AWGBatch

# Optional hardware driver (same broadened guard as dds_strategies.py: an
# installed-but-driverless `spcm` package raises a bare Exception from
# spcm_core/pyspcm.py when the vendor driver .so isn't found, not ImportError).
try:
    import spcm
    from spcm import SpcmException

    _HW_AVAILABLE = True
except Exception:
    spcm = None  # type: ignore[assignment]
    SpcmException = Exception  # type: ignore[assignment,misc]
    _HW_AVAILABLE = False

try:
    import cupy as cp

    _GPU_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore[assignment]
    _GPU_AVAILABLE = False

log = logging.getLogger(__name__)

TWO_PI: float = 2.0 * math.pi

#: Hard safety ceiling, asserted at setup time (not per-sample).
MAX_SAFE_OUTPUT_V: float = 2.0

#: Minimum interval between repeated throughput warnings, to avoid log spam
#: when the GPU genuinely can't keep up with real time. Also gates the GPU
#: stream sync needed to measure elapsed fill-loop time accurately (see
#: ScappFeeder._maybe_warn_throughput) -- syncing every iteration would
#: itself cost more than the entire real-time budget at notify_samples=16384.
_THROUGHPUT_WARN_INTERVAL_S: float = 5.0

#: Minimum interval between amplitude-clipping integrity checks (see
#: ScappFeeder._maybe_check_clipping). Reading back whether anything
#: exceeded the amplitude budget requires a host/GPU sync, so -- like the
#: throughput check -- it's throttled rather than paid every iteration. The
#: clip itself stays unconditional and is unaffected by this throttling.
_CLIP_CHECK_INTERVAL_S: float = 5.0

#: Minimum interval between repeated dropped-transition warnings (see
#: ScappFeeder._maybe_warn_dropped_transition), to avoid log spam when a
#: whole burst of moves lands inside one notify_samples window.
_DROPPED_TRANSITION_WARN_INTERVAL_S: float = 5.0


# ---------------------------------------------------------------------------
# Per-tone trajectory + phase math (pure, hardware-free — unit-testable with
# plain numpy, and used unmodified by the real fill loop with cupy).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToneSegment:
    """Closed-form frequency/phase trajectory for one tone on one channel.

    ``shape`` selects the frequency profile over ``[0, duration_s]``:

    * ``"hold"`` — constant at ``f_end`` (``duration_s`` is informational,
      not used in the phase formula).
    * ``"linear"`` — constant-slope sweep from ``f_start`` to ``f_end``.
    * ``"scurve"`` — raised-cosine (smooth-acceleration) sweep from
      ``f_start`` to ``f_end``.

    Both ramp shapes hold at ``f_end`` for ``t > duration_s`` (the segment
    stays valid indefinitely until the next transition replaces it).
    """

    channel: int
    tone_index: int
    shape: str  # "hold" | "linear" | "scurve"
    f_start: float
    f_end: float
    duration_s: float
    amplitude_pct: float
    phase_offset_rad: float
    static_phase_rad: float
    start_sample: int


def segment_instantaneous_phase(seg: ToneSegment, t_local, xp=np):
    """Vectorized ``2*pi * integral(f(tau), tau=0..t)`` term (rad).

    Does NOT include ``phase_offset_rad``/``static_phase_rad`` — callers add
    those separately. ``t_local`` is seconds since ``seg.start_sample``,
    as an ``xp`` array (``xp=numpy`` in tests, ``xp=cupy`` in the real fill
    loop — the formulas are identical either way).
    """
    if seg.shape == "hold":
        return TWO_PI * seg.f_end * t_local

    duration = seg.duration_s
    t_c = xp.minimum(t_local, duration)
    tail = TWO_PI * seg.f_end * xp.maximum(t_local - duration, 0.0)

    if seg.shape == "linear":
        slope = (seg.f_end - seg.f_start) / duration
        ramp = TWO_PI * (seg.f_start * t_c + 0.5 * slope * t_c * t_c)
        return ramp + tail

    if seg.shape == "scurve":
        delta_f = seg.f_end - seg.f_start
        ramp = TWO_PI * (
            seg.f_start * t_c
            + 0.5
            * delta_f
            * (t_c - (duration / math.pi) * xp.sin(math.pi * t_c / duration))
        )
        return ramp + tail

    raise ValueError(
        f"Unknown ToneSegment.shape {seg.shape!r}; expected 'hold'/'linear'/'scurve'."
    )


def segment_total_phase(seg: ToneSegment, t_local, xp=np):
    """Full instantaneous phase (rad), including offsets. Convenience wrapper
    used by tests and the fill loop for the final ``sin(phase)`` evaluation.
    """
    return (
        seg.phase_offset_rad
        + segment_instantaneous_phase(seg, t_local, xp=xp)
        + seg.static_phase_rad
    )


def segment_instantaneous_frequency(seg: ToneSegment, t_local, xp=np):
    """Vectorized instantaneous frequency ``f(t)`` (Hz) for *seg* — the
    analytical ramp shape a tone is commanded to follow, independent of
    phase/FFT. Used for the "pure" frequency-vs-time trajectory plot
    (:func:`synthesize_round_frequency_trajectory`), not the real-time fill
    loop (which only needs phase, via :func:`segment_total_phase`).

    Unlike phase, frequency is **not cumulative** — it depends only on where
    ``t_local`` falls within *this* segment's own ``duration_s``, so (unlike
    :func:`segment_instantaneous_phase`) there's no separate "tail" term:
    clamping ``t_local`` at ``duration_s`` already lands exactly on
    ``f_end`` for both ``"linear"`` and ``"scurve"``, and holds there.
    """
    if seg.shape == "hold":
        return t_local * 0.0 + seg.f_end

    duration = seg.duration_s
    t_c = xp.minimum(t_local, duration)

    if seg.shape == "linear":
        slope = (seg.f_end - seg.f_start) / duration
        return seg.f_start + slope * t_c

    if seg.shape == "scurve":
        delta_f = seg.f_end - seg.f_start
        return seg.f_start + 0.5 * delta_f * (1.0 - xp.cos(math.pi * t_c / duration))

    raise ValueError(
        f"Unknown ToneSegment.shape {seg.shape!r}; expected 'hold'/'linear'/'scurve'."
    )


def _transition_tone_segments(
    segments: Dict[Tuple[int, int], ToneSegment],
    batch: AWGBatch,
    transition_sample: int,
    sample_rate_hz: float,
    ramp_shape: str,
) -> Dict[Tuple[int, int], ToneSegment]:
    """Pure (hardware/GPU-free) core of :meth:`ScappFeeder._transition_segments`.

    Computes the next ``{(channel, tone_index): ToneSegment}`` state at
    ``transition_sample``, carrying each tone's phase continuously across the
    boundary (see :meth:`ScappFeeder._transition_segments` for why). Factored
    out so offline waveform synthesis (:func:`synthesize_round_waveform`) can
    reuse the exact same math without touching feeder state/locks.
    """
    shape = "hold" if batch.total_duration_s <= 0 else ramp_shape
    new_segments: Dict[Tuple[int, int], ToneSegment] = {}
    for ramp in batch.ramps:
        key = (ramp.channel, ramp.tone_index)
        old = segments[key]
        t_old = (transition_sample - old.start_sample) / sample_rate_hz
        phase_at_transition = (
            old.phase_offset_rad
            + float(segment_instantaneous_phase(old, np.array([t_old]), xp=np)[0])
        ) % TWO_PI
        new_segments[key] = ToneSegment(
            channel=ramp.channel,
            tone_index=ramp.tone_index,
            shape=shape,
            f_start=ramp.f_start,
            f_end=ramp.f_end,
            duration_s=batch.total_duration_s,
            amplitude_pct=ramp.amplitude_pct,
            phase_offset_rad=phase_at_transition,
            static_phase_rad=math.radians(ramp.phase_deg),
            start_sample=transition_sample,
        )
    return new_segments


def synthesize_round_waveform(
    batches: Sequence[AWGBatch],
    sample_rate_hz: float,
    *,
    ramp_shape: str = "linear",
) -> Dict[int, np.ndarray]:
    """Offline (CPU/numpy) synthesis of the per-channel waveform SCAPP would
    stream to the card for *batches* played back-to-back.

    Reuses the exact phase-continuous segment math the real-time GPU fill
    loop uses (:func:`_transition_tone_segments`, :func:`segment_total_phase`)
    so the result matches what actually reaches the AOD under the scapp
    backend — this is for offline visualization/analysis (e.g. spectrograms
    via :meth:`awg_controller.src.session_recorder.SessionRecorder.save_spectrogram`),
    not part of the real-time fill path itself.

    The first batch's ``f_start`` per tone is treated as the pre-existing
    hold frequency (mirrors how :meth:`ScappFeeder.start` seeds from the
    holding batch). Zero-duration batches contribute no samples but still
    update the carried phase/frequency state.

    Returns ``{channel: samples}``, float64 in roughly ``[-1, 1]`` (sum of
    unit-amplitude tones scaled by ``amplitude_pct``), one entry per channel
    seen across *batches*. Empty dict if *batches* is empty.
    """
    if not batches:
        return {}

    segments: Dict[Tuple[int, int], ToneSegment] = {
        (ramp.channel, ramp.tone_index): ToneSegment(
            channel=ramp.channel,
            tone_index=ramp.tone_index,
            shape="hold",
            f_start=ramp.f_start,
            f_end=ramp.f_start,
            duration_s=0.0,
            amplitude_pct=ramp.amplitude_pct,
            phase_offset_rad=0.0,
            static_phase_rad=math.radians(ramp.phase_deg),
            start_sample=0,
        )
        for ramp in batches[0].ramps
    }

    channels = sorted({ramp.channel for batch in batches for ramp in batch.ramps})
    chunks: Dict[int, list] = {ch: [] for ch in channels}
    next_sample = 0

    for batch in batches:
        n_samples = int(round(batch.total_duration_s * sample_rate_hz))
        segments = _transition_tone_segments(
            segments, batch, next_sample, sample_rate_hz, ramp_shape
        )
        if n_samples > 0:
            abs_sample = next_sample + np.arange(n_samples, dtype=np.int64)
            for ch in channels:
                total = np.zeros(n_samples, dtype=np.float64)
                for (seg_ch, _tone_idx), seg in segments.items():
                    if seg_ch != ch:
                        continue
                    t_local = (abs_sample - seg.start_sample).astype(
                        np.float64
                    ) / sample_rate_hz
                    phase = segment_total_phase(seg, t_local, xp=np)
                    total = total + np.sin(phase) * (seg.amplitude_pct / 100.0)
                chunks[ch].append(total)
            next_sample += n_samples

    return {
        ch: (np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64))
        for ch, parts in chunks.items()
    }


def synthesize_round_frequency_trajectory(
    batches: Sequence[AWGBatch],
    *,
    ramp_shape: str = "linear",
    points_per_batch: int = 64,
    break_tol_hz: float = 1.0,
) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    """Analytical (phase/FFT-free) per-tone frequency trajectory for
    *batches* played back-to-back — the exact ``f(t)`` ramp shape
    (:func:`segment_instantaneous_frequency`) each tone is commanded to
    follow, i.e. the *design intent* rather than the synthesized waveform.

    Distinct from :func:`synthesize_round_waveform`, which sums sine tones
    for FFT/spectrogram analysis: instantaneous frequency isn't cumulative
    across batches (unlike phase), so this needs no ``ToneSegment``
    phase-continuity bookkeeping — each batch's ramp is fully described by
    its own ``(f_start, f_end, duration_s, shape)``.

    A tone's ``f_start`` in one batch doesn't always match its ``f_end`` in
    the previous batch it appeared in — e.g. ``RFConverter.convert_moves``
    rebuilds every non-targeted tone's ramp from its *nominal* resting
    frequency each batch, regardless of where an earlier batch in the same
    round actually left it, so a tone that moved once and isn't re-targeted
    gets a real, physically-commanded frequency jump on the next batch
    boundary. Drawing a straight connecting line across that jump would
    misrepresent it as a continuous ramp, so a ``NaN`` separator is inserted
    into that tone's trajectory instead — matplotlib breaks the line there
    rather than drawing it.

    Returns ``{(channel, tone_index): (times_s, freqs_hz)}`` spanning the
    full round (cumulative time across *batches*), one entry per tone seen
    in *batches* — including non-moving tones, which naturally evaluate flat
    since ``f_start == f_end`` collapses both ramp shapes to a constant (see
    :func:`segment_instantaneous_frequency`); callers wanting only moving
    tones should filter the returned keys themselves (e.g. by checking
    ``f_start != f_end`` on the corresponding ramps in *batches*).
    ``points_per_batch`` sets the plotting resolution per batch (a
    zero-duration/holding batch contributes a single point). ``break_tol_hz``
    is the mismatch threshold for inserting a break.
    """
    trajectories: Dict[Tuple[int, int], Tuple[list, list]] = {}
    last_f_end: Dict[Tuple[int, int], float] = {}
    t_offset = 0.0
    for batch in batches:
        duration = max(float(batch.total_duration_s), 0.0)
        shape = "hold" if duration <= 0 else ramp_shape
        t_local = (
            np.linspace(0.0, duration, max(int(points_per_batch), 2))
            if duration > 0
            else np.zeros(1)
        )
        for ramp in batch.ramps:
            key = (ramp.channel, ramp.tone_index)
            times_list, freqs_list = trajectories.setdefault(key, ([], []))
            prev_end = last_f_end.get(key)
            if prev_end is not None and abs(ramp.f_start - prev_end) > break_tol_hz:
                times_list.append(np.array([t_offset]))
                freqs_list.append(np.array([np.nan]))

            seg = ToneSegment(
                channel=ramp.channel,
                tone_index=ramp.tone_index,
                shape=shape,
                f_start=ramp.f_start,
                f_end=ramp.f_end,
                duration_s=duration,
                amplitude_pct=ramp.amplitude_pct,
                phase_offset_rad=0.0,
                static_phase_rad=0.0,
                start_sample=0,
            )
            freqs = segment_instantaneous_frequency(seg, t_local, xp=np)
            times_list.append(t_offset + t_local)
            freqs_list.append(freqs)
            last_f_end[key] = float(ramp.f_end)
        t_offset += duration

    return {
        key: (np.concatenate(ts), np.concatenate(fs))
        for key, (ts, fs) in trajectories.items()
    }


# ---------------------------------------------------------------------------
# Feeder configuration + thread
# ---------------------------------------------------------------------------


@dataclass
class ScappFeederConfig:
    """Tuning knobs for :class:`ScappFeeder`.

    ``notify_samples`` / ``dma_buffer_samples`` sizing
    ----------------------------------------------------
    ``notify_samples`` sets the *only* granularity at which the streamed
    waveform can change trajectory: :meth:`ScappFeeder._transition_segments`
    always stamps a new ``submit_batch``/``submit_holding`` at the current
    ``_next_fill_sample``, which only advances once per completed
    ``notify_samples``-sized chunk in :meth:`ScappFeeder._fill_loop`. If two
    ``submit_batch`` calls land within the same chunk period, the second
    silently overwrites the first's segments before the fill loop ever
    renders them — the earlier move's chirp never reaches the DAC.

    Move-batch durations here are set by ``atommovr.utils.timing`` (Chebyshev
    distance × ``PhysicalParams.spacing`` / ``AOD_speed``, floored at
    ``MIN_MOVE_DURATION_S`` — 50 µs; raised from an earlier 1 µs value, see
    that constant's docstring). Below ``MIN_MOVE_DURATION_S``, distance/speed
    differences between moves stop mattering: every move whose raw travel
    time is smaller floors to exactly 50 µs, so ``notify_samples`` only ever
    needs to resolve *that* floor, not any faster real move. The reference
    SCAPP example (``spcm-examples/10_cuda_scapp/5_scapp_gen_fifo_sine.py``)
    used ``notify_samples = 512 * 1024`` for an always-on 2-tone demo with no
    latency requirement — at the M4i.6631-x8's 1.25 GS/s max rate
    (``awg_controller.src.awg_control.M4I_6631_X8_MAX_SAMPLE_RATE_HZ``) that's
    a ~419 µs chunk period, ~8x longer than the 50 µs move floor.

    Empirically (see ``dropped_transition_count`` /
    ``ScappFeeder._maybe_warn_dropped_transition``, and the offline
    replay-simulation this default was tuned against), a notify_samples value
    picked with too little margin below the real move cadence starts
    dropping submissions once the control loop's ``time.sleep()`` pacing is
    precise enough to approach that cadence (on a real OS the *actual*
    scheduling granularity of that sleep — not the nominal move duration —
    is what determines whether this bites; it's easy for that to be
    optimistic on a well-tuned/real-time system).

    The default below (``16384``, i.e. 16 KiS) drives the notify period to
    **~13.1 µs at the M4i.6631-x8's 1.25 GS/s max rate** — about 3.8x margin
    under the 50 µs ``MIN_MOVE_DURATION_S`` floor, and ~16x more real-time
    budget per fill-loop iteration than the previous 1024-sample
    (~0.82 µs) default, which left far less headroom than the fixed
    Python/CuPy dispatch overhead of even one fill iteration needs (see
    :class:`_ChannelToneArrays`/:meth:`ScappFeeder._sum_channel` for the
    per-iteration cost this budget has to cover). It stays a multiple of
    1024 samples, so it doesn't fight the vendored ``spcm`` package's
    4096-byte buffer-alignment quantum (1024 samples × 4 bytes/sample-pair
    = 4096 B).

    This is still smaller than anything demonstrated in
    ``spcm-examples/10_cuda_scapp`` (that folder's smallest continuous-FIFO
    example is 64 KiS; ours is 4x smaller) — going this small trades away
    some of the throughput headroom those examples were chosen to preserve.
    Each fill-loop iteration has fixed per-call overhead (Python/CuPy
    dispatch, lock acquire, iterating every active tone segment) that
    doesn't shrink with ``notify_samples``, so more iterations per second
    could still make that fixed overhead dominate the real-time budget and
    cause actual GPU/DMA underruns — a different, and likely worse, failure
    mode than the dropped-transition problem this is meant to fix (a real
    hardware fault, not just a silently-skipped move). **This value is
    unverified — it must be validated on real hardware**: watch
    :meth:`ScappFeeder._maybe_warn_throughput` for real-time-budget warnings
    and check ``dropped_transition_count`` after a run; raise
    ``notify_samples`` (in steps, e.g. 32 KiS/64 KiS) if either fires,
    verifying with an oscilloscope, until both stay clean — and if
    ``MIN_MOVE_DURATION_S`` is ever lowered again, ``notify_samples`` must
    come down with it.

    ``dma_buffer_samples`` must be an **exact multiple** of ``notify_samples``
    (enforced in ``__post_init__``) — the vendored ``spcm`` package's
    automatic buffer handling only works correctly under that constraint
    (see ``spcm/classes_data_transfer.py``: ``_pre_buffer_transfer``'s
    ``exception_num_samples`` check).
    """

    #: GPU buffer fill block size (samples). See sizing discussion above —
    #: tuned against MIN_MOVE_DURATION_S (50 µs), still smaller than any
    #: precedented spcm-examples/10_cuda_scapp value, and needs on-hardware
    #: throughput validation.
    notify_samples: int = 16384
    #: Total RDMA-pinned DMA buffer size (samples); must be an exact
    #: multiple of ``notify_samples``. ~26.8 ms of underrun cushion at the
    #: M4i.6631-x8's 1.25 GS/s max rate.
    dma_buffer_samples: int = 32 * 1024 * 1024
    #: card.start() fires once the on-board buffer fill level crosses this
    #: (per-mille, 0-1000), matching the reference example's warm-up gate.
    fill_start_threshold_promille: int = 800
    #: None -> use the card's maximum sample rate (M4i.6631-x8: 1.25 GS/s,
    #: see ``awg_controller.src.awg_control.M4I_6631_X8_MAX_SAMPLE_RATE_HZ``).
    sample_rate_hz: Optional[float] = None
    #: Default frequency-ramp shape for non-static moves ("linear"|"scurve").
    ramp_shape: str = "linear"
    #: Soft GPU-throughput warning threshold, as a fraction of the real-time
    #: budget (notify_samples / sample_rate_hz) a single fill iteration may
    #: consume before a warning is logged. Not a hard limit.
    fill_time_warn_fraction: float = 0.5
    #: Timeout (s) for feeder-thread startup / shutdown joins.
    join_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if self.notify_samples <= 0:
            raise ValueError("notify_samples must be positive.")
        if self.dma_buffer_samples % self.notify_samples != 0:
            raise ValueError(
                f"dma_buffer_samples ({self.dma_buffer_samples}) must be an "
                f"exact multiple of notify_samples ({self.notify_samples}) — "
                "spcm's automatic buffer handling requires this."
            )


@dataclass
class _ChannelToneArrays:
    """GPU-resident, precomputed trajectory arrays for every tone active on
    one channel — rebuilt via :func:`_build_channel_arrays` once per
    submitted batch (see :meth:`ScappFeeder._seed_segments` /
    :meth:`ScappFeeder._transition_segments`), *not* once per fill-loop
    iteration.

    This is what lets :meth:`ScappFeeder._sum_channel` evaluate every active
    tone's phase in a handful of vectorized GPU ops regardless of tone
    count, instead of a Python ``for`` loop issuing one GPU kernel launch
    per tone per iteration. At ``notify_samples=16384`` the real-time budget
    per fill iteration is ~13.1 microseconds at the card's max sample rate
    — still comparable to the fixed dispatch latency of a handful of kernel
    launches — so a per-tone loop's overhead scales directly with tone count,
    and was the dominant way a live grid (dozens of active tones per round)
    blew that budget and caused a DMA underrun.

    Every field here that only depends on the *segment* (not on the sample
    index being rendered) is folded into a single constant once per
    submitted batch, instead of being recomputed from ``f_start``/``slope``/
    etc. on every fill-loop iteration. ``two_pi_f_start``/``two_pi_f_end``/
    ``pi_slope`` (linear/hold) and ``pi_delta_f``/``delta_f_safe_dur``/
    ``pi_over_safe_dur`` (scurve) are the algebraically-expanded phase
    formulas from :func:`segment_instantaneous_phase` — see
    :meth:`ScappFeeder._sum_channel` for the expansion. ``amplitude_frac``
    likewise already has the card's ``max_sample_value`` baked in (see
    :func:`_build_channel_arrays`), so the fill loop doesn't need a separate
    whole-buffer multiply per channel per iteration.

    ``is_scurve`` distinguishes only "scurve" from "everything else":
    "hold" (``duration=0``) and "linear" tones share one formula, because
    clamping ``t_c = min(t_local, duration)`` makes the linear ramp term
    evaluate to exactly zero when ``duration=0`` regardless of slope,
    collapsing it to the same ``tail`` term the standalone "hold" branch in
    :func:`segment_instantaneous_phase` uses — a closed-form identity, not
    an approximation (see :func:`_build_channel_arrays`). ``has_scurve`` is a
    plain Python ``bool`` (not a GPU array), computed once on the host, so
    :meth:`ScappFeeder._sum_channel` can skip the scurve terms (and the
    ``cp.where`` select) entirely in Python at zero GPU cost whenever no
    active tone uses that shape — since every ``submit_batch``/
    ``submit_holding`` call re-stamps *every* tone with the same shape (see
    module docstring), this is almost always true.

    All arrays are shape ``(n_tones, 1)`` (including ``start_sample``) so
    they broadcast against a ``(1, n_samples)`` sample-index row in
    :meth:`ScappFeeder._sum_channel` with no per-call reshaping.
    """

    start_sample: "cp.ndarray"
    duration: "cp.ndarray"
    two_pi_f_start: "cp.ndarray"
    two_pi_f_end: "cp.ndarray"
    pi_slope: "cp.ndarray"
    pi_delta_f: "cp.ndarray"
    delta_f_safe_dur: "cp.ndarray"
    pi_over_safe_dur: "cp.ndarray"
    is_scurve: "cp.ndarray"
    has_scurve: bool
    amplitude_frac: "cp.ndarray"
    offset_plus_static: "cp.ndarray"
    n_tones: int


def _build_channel_arrays(
    segments: Dict[Tuple[int, int], ToneSegment], channel: int, max_value: float = 1.0
) -> "_ChannelToneArrays":
    """Pack every :class:`ToneSegment` active on *channel* into the batched,
    GPU-resident form :class:`_ChannelToneArrays` needs.

    Runs on plain Python/numpy on the way in — segment dicts are small
    (tens to ~hundreds of tones) and this only runs once per submitted
    batch, not once per fill iteration — with a single host->device upload
    per field. All the algebra that doesn't depend on the sample index
    (``TWO_PI * f_start``, the slope/scurve constants, ``amplitude_frac *
    max_value``, ``phase_offset + static_phase``) is done here, once, so
    :meth:`ScappFeeder._sum_channel` never repeats it on the real-time path
    (see :class:`_ChannelToneArrays` docstring for the expanded formulas).

    ``max_value`` (the card's ``max_sample_value``) is folded into
    ``amplitude_frac`` here instead of being applied as a separate
    whole-buffer multiply in :meth:`ScappFeeder._fill_loop` every iteration.
    """
    rows = [seg for (ch, _tone_idx), seg in segments.items() if ch == channel]
    n = len(rows)
    f_start = np.empty(n, dtype=np.float64)
    f_end = np.empty(n, dtype=np.float64)
    duration = np.empty(n, dtype=np.float64)
    amplitude_frac = np.empty(n, dtype=np.float64)
    phase_offset = np.empty(n, dtype=np.float64)
    static_phase = np.empty(n, dtype=np.float64)
    start_sample = np.empty(n, dtype=np.int64)
    is_scurve = np.empty(n, dtype=bool)
    for i, seg in enumerate(rows):
        f_start[i] = seg.f_start
        f_end[i] = seg.f_end
        duration[i] = seg.duration_s
        amplitude_frac[i] = seg.amplitude_pct / 100.0
        phase_offset[i] = seg.phase_offset_rad
        static_phase[i] = seg.static_phase_rad
        start_sample[i] = seg.start_sample
        is_scurve[i] = seg.shape == "scurve"

    # np.where avoids a literal 0/0 in the slope division for "hold" tones
    # (duration=0); the result is discarded anyway since t_c=min(t_local, 0)
    # zeroes out everything it multiplies (see class docstring).
    safe_duration = np.where(duration > 0, duration, 1.0)
    delta_f = f_end - f_start
    slope = delta_f / safe_duration

    # Algebraically-expanded phase constants (see _sum_channel for the
    # derivation) -- computed once per batch instead of once per iteration.
    two_pi_f_start = TWO_PI * f_start
    two_pi_f_end = TWO_PI * f_end
    pi_slope = math.pi * slope
    pi_delta_f = math.pi * delta_f
    delta_f_safe_dur = delta_f * safe_duration
    pi_over_safe_dur = math.pi / safe_duration

    def col(a: np.ndarray) -> "cp.ndarray":
        return cp.asarray(a)[:, None]

    return _ChannelToneArrays(
        start_sample=col(start_sample),
        duration=col(duration),
        two_pi_f_start=col(two_pi_f_start),
        two_pi_f_end=col(two_pi_f_end),
        pi_slope=col(pi_slope),
        pi_delta_f=col(pi_delta_f),
        delta_f_safe_dur=col(delta_f_safe_dur),
        pi_over_safe_dur=col(pi_over_safe_dur),
        is_scurve=col(is_scurve),
        has_scurve=bool(is_scurve.any()),
        amplitude_frac=col(amplitude_frac * max_value),
        offset_plus_static=col(phase_offset + static_phase),
        n_tones=n,
    )


# ---------------------------------------------------------------------------
# Fusable per-sample phase/weight math for _sum_channel's real-time hot path.
# ---------------------------------------------------------------------------
#
# Every line in these two functions is a plain elementwise expression (no
# reductions, no Python branching on array contents), which is exactly what
# ``cp.fuse`` compiles into a *single* GPU kernel instead of one kernel
# launch per operator -- collapsing this ~9-op chain from ~9 launches to 1
# (see :meth:`ScappFeeder._sum_channel`, which still does the final
# ``cp.sum`` reduction separately, since fused reductions are less portable
# across cupy versions than a plain elementwise fuse). On real hardware,
# fixed per-launch dispatch overhead dominates at the sample counts used
# here, so cutting launch count directly cuts wall-clock time -- unlike
# reducing FLOPs, which barely matters at this scale.
#
# Kept as plain module-level functions (not methods) so ``cp.fuse()`` -- a
# decorator, not something with special support for bound methods -- can
# wrap them directly. Conditionally wrapped below only when a real cupy is
# imported: numpy has no ``fuse``, so the unfused fallback runs as-is under
# the test suite's numpy-simulated ``cp`` (identical math either way --
# fusion changes execution strategy, not results).


def _linear_phase_weight(
    abs_sample,
    start_sample,
    duration,
    two_pi_f_start,
    two_pi_f_end,
    pi_slope,
    offset_plus_static,
    amplitude_frac,
    sample_rate_hz,
):
    """Per-tone-per-sample ``sin(phase) * amplitude`` for "hold"/"linear"
    tones (see :meth:`ScappFeeder._sum_channel` for the formula derivation).
    """
    t_local = (abs_sample - start_sample) / sample_rate_hz
    t_c = cp.minimum(t_local, duration)
    tail = two_pi_f_end * cp.maximum(t_local - duration, 0.0)
    ramp = two_pi_f_start * t_c + pi_slope * (t_c * t_c)
    phase = ramp + tail + offset_plus_static
    return cp.sin(phase) * amplitude_frac


def _full_phase_weight(
    abs_sample,
    start_sample,
    duration,
    two_pi_f_start,
    two_pi_f_end,
    pi_slope,
    pi_delta_f,
    delta_f_safe_dur,
    pi_over_safe_dur,
    is_scurve,
    offset_plus_static,
    amplitude_frac,
    sample_rate_hz,
):
    """Per-tone-per-sample ``sin(phase) * amplitude``, general form
    covering a mix of "hold"/"linear" and "scurve" tones on one channel.
    """
    t_local = (abs_sample - start_sample) / sample_rate_hz
    t_c = cp.minimum(t_local, duration)
    tail = two_pi_f_end * cp.maximum(t_local - duration, 0.0)
    shared = two_pi_f_start * t_c
    linear_extra = pi_slope * (t_c * t_c)
    scurve_extra = pi_delta_f * t_c - delta_f_safe_dur * cp.sin(pi_over_safe_dur * t_c)
    ramp = shared + cp.where(is_scurve, scurve_extra, linear_extra)
    phase = ramp + tail + offset_plus_static
    return cp.sin(phase) * amplitude_frac


if _GPU_AVAILABLE and hasattr(cp, "fuse"):
    _linear_phase_weight = cp.fuse()(_linear_phase_weight)
    _full_phase_weight = cp.fuse()(_full_phase_weight)


class ScappFeeder:
    """Owns the continuous SCAPP GPU buffer-fill loop for one card.

    Life-cycle::

        feeder = ScappFeeder(card, hw_config, aod_settings, config)
        feeder.start(initial_holding_batch)   # blocks until card.start() fires
        feeder.submit_batch(batch)            # blocks for batch.total_duration_s
        feeder.submit_holding(holding_batch)  # returns immediately
        feeder.stop()

    Requires ``spcm`` (real driver) and ``cupy``; construction is cheap and
    hardware-free, but :meth:`start` requires both to be available.
    """

    def __init__(
        self,
        card: "spcm.Card",
        hw_config: "HardwareConfig",
        aod_settings: AODSettings,
        feeder_config: Optional[ScappFeederConfig] = None,
    ) -> None:
        self._card = card
        self._hw = hw_config
        self._aod = aod_settings
        self._config = feeder_config or ScappFeederConfig(
            ramp_shape=aod_settings.ramp_shape
        )

        self._state_lock = threading.Lock()
        self._active_segments: Dict[Tuple[int, int], ToneSegment] = {}
        # GPU-resident vectorized form of _active_segments, rebuilt whenever
        # it changes (_seed_segments / _transition_segments) -- see
        # _ChannelToneArrays / _sum_channel for why the fill loop indexes
        # into this instead of iterating _active_segments per tone.
        self._channel_arrays: Dict[int, "_ChannelToneArrays"] = {}
        self._next_fill_sample: int = 0
        self._feeder_exception: Optional[BaseException] = None

        self._stop_event = threading.Event()
        self._start_event = threading.Event()
        self._started = False
        self._thread: Optional[threading.Thread] = None

        self._scapp_transfer = None
        self._sample_rate_hz: float = 0.0
        self._max_value: int = 1
        self._sample_dtype = None
        self._local_sample_idx = None
        self._last_throughput_warn_s: float = 0.0

        # Dropped-transition detection (see _maybe_warn_dropped_transition):
        # tracks whether the fill loop has advanced past a submitted
        # transition's target sample before the *next* submit_batch/
        # submit_holding call replaces it.
        self._last_transition_sample: Optional[int] = None
        self._dropped_transition_count: int = 0
        self._last_dropped_warn_s: float = 0.0

        # Throttle state for the fill loop's diagnostic checks -- both
        # gate a GPU stream sync, so they're paid only periodically rather
        # than every iteration (see _maybe_warn_throughput /
        # _maybe_check_clipping).
        self._last_clip_check_s: float = 0.0

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def start(self, initial_holding: AWGBatch) -> None:
        """One-time hardware setup + spawn the background fill thread.

        Blocks until the DMA buffer has pre-filled past
        ``fill_start_threshold_promille`` and ``card.start()`` has been
        issued — by the time this returns, samples are already flowing to
        the DAC (matching the reference example's warm-up behavior).
        """
        if not _HW_AVAILABLE:
            raise RuntimeError("spcm is not available; cannot start ScappFeeder.")
        if not _GPU_AVAILABLE:
            raise RuntimeError("cupy is not available; cannot start ScappFeeder.")

        if self._hw.max_amplitude_v > MAX_SAFE_OUTPUT_V:
            raise ValueError(
                f"max_amplitude_v={self._hw.max_amplitude_v} V exceeds "
                f"{MAX_SAFE_OUTPUT_V} V hard safety ceiling."
            )

        self._card.card_mode(spcm.SPC_REP_FIFO_SINGLE)
        self._card.timeout(5 * spcm.units.s)

        trigger = spcm.Trigger(self._card)
        trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)

        channels = spcm.Channels(self._card, card_enable=spcm.CHANNEL0 | spcm.CHANNEL1)
        channels.enable(True)
        channels.output_load(self._hw.output_load_ohms * spcm.units.ohm)
        channels.amp(self._hw.max_amplitude_v * spcm.units.V)

        clock = spcm.Clock(self._card)
        clock.mode(spcm.SPC_CM_INTPLL)
        if self._config.sample_rate_hz is not None:
            rate = clock.sample_rate(
                self._config.sample_rate_hz * spcm.units.Hz, return_unit=spcm.units.Hz
            )
        else:
            rate = clock.sample_rate(max=True, return_unit=spcm.units.Hz)
        self._sample_rate_hz = _magnitude(rate)

        max_tone_freq = max(self._aod.f_max_v, self._aod.f_max_h)
        if max_tone_freq >= 0.5 * self._sample_rate_hz:
            raise ValueError(
                f"Tone frequency {max_tone_freq / 1e6:.1f} MHz exceeds Nyquist "
                f"({self._sample_rate_hz / 2e6:.1f} MHz) at sample_rate="
                f"{self._sample_rate_hz / 1e6:.1f} MHz."
            )

        self._max_value = self._card.max_sample_value()

        self._scapp_transfer = spcm.SCAPPTransfer(
            self._card, direction=spcm.Direction.Generation
        )
        self._scapp_transfer.notify_samples(self._config.notify_samples)
        self._scapp_transfer.allocate_buffer(self._config.dma_buffer_samples)
        self._scapp_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)
        self._sample_dtype = self._scapp_transfer.numpy_type()

        # Constant per chunk (only chunk_start shifts each iteration) --
        # precomputed once instead of relaunching cp.arange every fill
        # iteration in the real-time loop.
        self._local_sample_idx = cp.arange(self._config.notify_samples, dtype=cp.int64)
        # A channel with zero active tones (e.g. an unpopulated axis) always
        # renders silence -- cache that instead of launching a fresh
        # cp.zeros every iteration (see _sum_channel).
        self._zero_chunk = cp.zeros(self._config.notify_samples, dtype=cp.float64)
        # Counts iterations since the last _maybe_warn_throughput sync, so
        # its log line can report an amortized per-iteration compute cost
        # alongside the raw (possibly backlog-inflated) figure -- see that
        # method's docstring.
        self._iters_since_throughput_check = 0

        self._seed_segments(initial_holding)
        self._warmup_kernels()

        self._stop_event = threading.Event()
        self._start_event = threading.Event()
        self._started = False
        self._thread = threading.Thread(
            target=self._fill_loop, name="ScappFeeder", daemon=True
        )
        self._thread.start()

        if not self._start_event.wait(timeout=self._config.join_timeout_s):
            self.stop()
            raise RuntimeError("SCAPP feeder did not reach the fill threshold in time.")
        self._raise_if_failed()

        log.info(
            f"SCAPP feeder started (sample_rate={self._sample_rate_hz / 1e6:.1f} MHz, "
            f"notify_samples={self._config.notify_samples}, ramp_shape={self._config.ramp_shape})."
        )

    def stop(self, timeout_s: Optional[float] = None) -> None:
        """Signal shutdown, abort the DMA transfer, and join the fill thread."""
        self._stop_event.set()
        if self._card is not None:
            try:
                self._card.stop(spcm.M2CMD_DATA_STOPDMA | spcm.M2CMD_CARD_STOP)
            except Exception:
                log.warning(
                    "Error stopping SCAPP DMA transfer during shutdown", exc_info=True
                )
        if self._thread is not None:
            self._thread.join(timeout_s or self._config.join_timeout_s)
            if self._thread.is_alive():
                log.error("SCAPP feeder thread did not stop within timeout.")
            self._thread = None

    # ------------------------------------------------------------------
    # Public API used by the controller
    # ------------------------------------------------------------------

    def submit_batch(self, batch: AWGBatch) -> None:
        """Replace the active schedule with *batch*, then block for its
        travel duration plus one ``notify_samples`` window — mirrors the
        legacy `_output_batch` pacing contract, with the extra margin
        guarding against the *next* submit_batch/submit_holding call
        landing on the same ``_next_fill_sample`` "ticket" this one used,
        before the fill loop has ever rendered it (see
        :meth:`_transition_segments` / :meth:`_maybe_warn_dropped_transition`
        for the failure mode this avoids).

        Not an absolute guarantee: the fill loop's real cadence can run
        slightly behind ``notify_samples / sample_rate_hz`` under load —
        watch :meth:`_maybe_warn_throughput`. But since a chunk boundary
        only needs to have been *reached* once (not this batch's segments
        specifically rendered) for the drop to be avoided, waiting a full
        extra notify period comfortably covers the common case; check
        :attr:`dropped_transition_count` after a real run to confirm.
        """
        self._raise_if_failed()
        self._transition_segments(batch)
        self._raise_if_failed()
        notify_period_s = self._config.notify_samples / self._sample_rate_hz
        time.sleep(batch.total_duration_s + notify_period_s)

    def submit_holding(self, batch: AWGBatch) -> None:
        """Replace the active schedule with the static holding *batch*.
        Does not block (mirrors legacy `_send_holding`).
        """
        self._raise_if_failed()
        self._transition_segments(batch)
        self._raise_if_failed()

    @property
    def sample_rate_hz(self) -> float:
        return self._sample_rate_hz

    @property
    def last_error(self) -> Optional[BaseException]:
        with self._state_lock:
            return self._feeder_exception

    @property
    def dropped_transition_count(self) -> int:
        """Cumulative count of submitted transitions overwritten before the
        fill loop ever rendered them — i.e. moves that never physically
        reached the DAC because they were submitted faster than one
        ``notify_samples`` window. See :meth:`_maybe_warn_dropped_transition`.
        """
        with self._state_lock:
            return self._dropped_transition_count

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _raise_if_failed(self) -> None:
        with self._state_lock:
            exc = self._feeder_exception
        if exc is not None:
            raise RuntimeError("SCAPP feeder thread failed") from exc

    def _seed_segments(self, batch: AWGBatch) -> None:
        segments: Dict[Tuple[int, int], ToneSegment] = {}
        for ramp in batch.ramps:
            segments[(ramp.channel, ramp.tone_index)] = ToneSegment(
                channel=ramp.channel,
                tone_index=ramp.tone_index,
                shape="hold",
                f_start=ramp.f_start,
                f_end=ramp.f_end,
                duration_s=0.0,
                amplitude_pct=ramp.amplitude_pct,
                phase_offset_rad=0.0,
                static_phase_rad=math.radians(ramp.phase_deg),
                start_sample=0,
            )
        with self._state_lock:
            self._active_segments = segments
            self._channel_arrays = {
                0: _build_channel_arrays(segments, 0, self._max_value),
                1: _build_channel_arrays(segments, 1, self._max_value),
            }
            self._next_fill_sample = 0
            self._last_transition_sample = None

    def _warmup_kernels(self) -> None:
        """Force every CUDA kernel the fill loop will need -- including the
        ``cp.fuse``-compiled ones in :func:`_linear_phase_weight` /
        :func:`_full_phase_weight`, which JIT-compile lazily on first call
        -- to compile here, synchronously, before :meth:`_fill_loop` ever
        runs, instead of paying that cost on the loop's first live
        iteration where it eats straight into the DMA buffer's underrun
        cushion for no reason.

        Observed on real hardware: a single first-call warning showing
        ``compute=0.61 ms over 1 iteration(s)`` -- ruling out backlog
        (see :meth:`_maybe_warn_throughput`) -- immediately followed by a
        DMA underrun. That one slow iteration landed exactly because it
        was simultaneously paying for (a) ``cp.fuse``'s first-call JIT
        compile of a kernel signature never exercised before (a
        general-purpose warm-up of the *unfused* ``cp.sin``/``cp.minimum``/
        etc. primitives, run from the caller before ``start()``, does not
        pre-compile this — ``cp.fuse`` traces and compiles the *decorated
        function* itself, a distinct kernel) and (b) :meth:`_maybe_check_clipping`
        and :meth:`_maybe_warn_throughput` both firing on the same
        iteration, since both throttle states start unset. This runs the
        exact same call sequence once, using the real just-seeded
        ``_channel_arrays`` (so the compiled kernel signatures match
        real playback exactly), then marks both throttles as just-checked
        so the fill loop's actual first iteration doesn't redundantly
        re-pay the (now-cheap, but nonzero) sync cost either.
        """
        abs_sample = self._local_sample_idx  # chunk_start=0 for this dry run
        ch0 = self._sum_channel(self._channel_arrays[0], abs_sample)
        ch1 = self._sum_channel(self._channel_arrays[1], abs_sample)
        int(cp.count_nonzero(cp.abs(ch0) > self._max_value))
        int(cp.count_nonzero(cp.abs(ch1) > self._max_value))
        ch0 = cp.clip(ch0, -self._max_value, self._max_value)
        ch1 = cp.clip(ch1, -self._max_value, self._max_value)
        ch0.astype(self._sample_dtype)
        ch1.astype(self._sample_dtype)
        cp.cuda.get_current_stream().synchronize()

        now = time.monotonic()
        self._last_clip_check_s = now
        self._last_throughput_warn_s = now

    def _transition_segments(self, batch: AWGBatch) -> None:
        """Atomically replace every tone's trajectory at the exact start of
        the next unfilled chunk, carrying phase continuously across the
        boundary (see module docstring for why this is race-free).

        The carried ``phase_offset_rad`` deliberately excludes
        ``static_phase_rad``: each segment re-adds its own ``static_phase_rad``
        at evaluation time (see ``segment_total_phase``), so folding the
        outgoing segment's static phase into the carry would double-count it
        whenever the incoming segment's static phase is the same value (the
        only case that occurs in practice — ``RFConverter`` always emits
        ``phase_deg=0``).

        Each chunk the fill loop renders uses a single ``_active_segments``
        snapshot for its entire ``notify_samples`` span (see ``_fill_loop``)
        — so if this call's ``transition_sample`` is the same one the
        *previous* call used, the fill loop cannot have advanced past it in
        between, meaning the previous call's segments were replaced here
        before ever being rendered into a chunk: that move never reached the
        DAC. See :meth:`_maybe_warn_dropped_transition`.
        """
        with self._state_lock:
            transition_sample = self._next_fill_sample
            self._maybe_warn_dropped_transition(transition_sample)
            self._last_transition_sample = transition_sample
            self._active_segments = _transition_tone_segments(
                self._active_segments,
                batch,
                transition_sample,
                self._sample_rate_hz,
                self._config.ramp_shape,
            )
            self._channel_arrays = {
                0: _build_channel_arrays(self._active_segments, 0, self._max_value),
                1: _build_channel_arrays(self._active_segments, 1, self._max_value),
            }

    def _maybe_warn_dropped_transition(self, transition_sample: int) -> None:
        """Must be called with ``self._state_lock`` already held (from
        :meth:`_transition_segments`) — not reentrant-safe otherwise.

        Detects the case described in :meth:`_transition_segments`'s
        docstring: this call's ``transition_sample`` matching the previous
        call's means the fill loop hasn't advanced past that boundary since,
        so the previous call's segments are about to be discarded without
        ever having been rendered. Always counts it; logging itself is
        throttled (a fast burst of moves can trip this on every single
        call) — poll :attr:`dropped_transition_count` for the exact total.
        """
        if (
            self._last_transition_sample is None
            or transition_sample != self._last_transition_sample
        ):
            return
        self._dropped_transition_count += 1
        now = time.monotonic()
        if now - self._last_dropped_warn_s < _DROPPED_TRANSITION_WARN_INTERVAL_S:
            return
        self._last_dropped_warn_s = now
        notify_period_us = (
            self._config.notify_samples / self._sample_rate_hz * 1e6
            if self._sample_rate_hz
            else float("nan")
        )
        log.warning(
            "SCAPP feeder: a submitted move's segments were overwritten "
            "before the fill loop ever rendered them — that move never "
            f"reached the DAC (dropped_transition_count="
            f"{self._dropped_transition_count}). Submissions are arriving "
            f"faster than one notify_samples window ({self._config.notify_samples} "
            f"samples ≈ {notify_period_us:.1f} µs at the current sample "
            "rate); consider batching moves or shrinking notify_samples."
        )

    def _sum_channel(self, arrays: "_ChannelToneArrays", abs_sample):
        """Vectorized total waveform for one channel: every active tone's
        phase evaluated in one broadcasted expression instead of a per-tone
        Python loop issuing a GPU kernel launch per tone (see
        :class:`_ChannelToneArrays`), with that whole expression compiled
        into a single fused GPU kernel by :func:`_linear_phase_weight` /
        :func:`_full_phase_weight` instead of one launch per operator — see
        those functions' module-level comment for why fixed per-launch
        dispatch overhead (not FLOPs) is what dominates wall-clock time
        here, and why fusing it away matters.

        Every term depends on ``abs_sample`` (the one thing that actually
        changes between fill-loop iterations); everything that doesn't was
        already folded into ``arrays.*`` once per submitted batch by
        :func:`_build_channel_arrays` — e.g. ``linear_ramp =
        TWO_PI*(f_start*t_c + 0.5*slope*t_c**2)`` expands to
        ``two_pi_f_start*t_c + pi_slope*t_c**2`` (``TWO_PI*0.5 == pi``), and
        the scurve term similarly to ``two_pi_f_start*t_c + pi_delta_f*t_c -
        delta_f_safe_dur*sin(pi_over_safe_dur*t_c)``. ``has_scurve`` (a host
        ``bool``, not a device array) picks the cheaper fused function
        entirely in Python, at zero GPU cost, whenever every active tone
        shares the "hold"/"linear" formula (the common case — see
        :class:`_ChannelToneArrays`).
        """
        if arrays.n_tones == 0:
            # Cached in start() -- no GPU allocation/kernel launch for a
            # channel that's always silent.
            return self._zero_chunk

        abs_sample_row = abs_sample[None, :]
        if arrays.has_scurve:
            weighted = _full_phase_weight(
                abs_sample_row,
                arrays.start_sample,
                arrays.duration,
                arrays.two_pi_f_start,
                arrays.two_pi_f_end,
                arrays.pi_slope,
                arrays.pi_delta_f,
                arrays.delta_f_safe_dur,
                arrays.pi_over_safe_dur,
                arrays.is_scurve,
                arrays.offset_plus_static,
                arrays.amplitude_frac,
                self._sample_rate_hz,
            )
        else:
            weighted = _linear_phase_weight(
                abs_sample_row,
                arrays.start_sample,
                arrays.duration,
                arrays.two_pi_f_start,
                arrays.two_pi_f_end,
                arrays.pi_slope,
                arrays.offset_plus_static,
                arrays.amplitude_frac,
                self._sample_rate_hz,
            )
        return cp.sum(weighted, axis=0)

    def _maybe_start_trigger(self) -> None:
        if self._started:
            return
        fill = self._scapp_transfer.fill_size_promille()
        if fill > self._config.fill_start_threshold_promille:
            self._card.start(spcm.M2CMD_CARD_ENABLETRIGGER)
            self._started = True
            self._start_event.set()

    def _maybe_warn_throughput(
        self, t0: float, wait_s: float, n_active_tones: int
    ) -> None:
        """Throttled by ``_THROUGHPUT_WARN_INTERVAL_S`` -- and that throttle
        gates more than the log call here: getting an accurate ``compute_s``
        requires syncing the GPU stream (all the fill-loop's cupy ops are
        launched async), and a sync every iteration would itself cost more
        than the entire real-time budget at ``notify_samples=16384``. So
        outside the periodic check, this call is a cheap no-op and the loop
        stays fully async.

        ``wait_s`` (time blocked in ``for card_buffer in self._scapp_transfer``
        -- i.e. inside the driver's ``wait_dma()`` -- since the *previous*
        iteration finished) is passed in already measured and needs no sync:
        it's a plain wall-clock delta around a blocking call, not GPU work.
        Reported separately from ``compute_s`` (this iteration's GPU work,
        synced here) because they have entirely different possible causes --
        a large ``wait_s`` points at the driver/DMA side (unrelated to
        anything in this file), while a large ``compute_s`` points at the
        fill loop's own GPU work. Note ``compute_s`` -- unlike ``wait_s`` --
        can be inflated by more than just this one iteration: since this
        throttled sync is the *only* place the stream is synced, if the loop
        has been issuing async work faster than the GPU drains it for a
        while, this call also waits out whatever backlog piled up since the
        last check, not just this iteration's own ops. ``self.
        _iters_since_throughput_check`` (incremented every iteration by
        :meth:`_fill_loop`, reset here) makes that visible: dividing
        ``compute_s`` by it gives an amortized per-iteration figure --
        if that's small while raw ``compute_s`` is large, the raw number
        is backlog, not a real per-iteration cost.
        """
        now = time.monotonic()
        if now - self._last_throughput_warn_s < _THROUGHPUT_WARN_INTERVAL_S:
            return
        self._last_throughput_warn_s = now
        n_iters = self._iters_since_throughput_check
        self._iters_since_throughput_check = 0
        cp.cuda.get_current_stream().synchronize()
        compute_s = time.perf_counter() - t0
        budget_s = self._config.notify_samples / self._sample_rate_hz
        if wait_s + compute_s <= self._config.fill_time_warn_fraction * budget_s:
            return
        amortized_compute_s = compute_s / max(n_iters, 1)
        log.warning(
            f"SCAPP fill loop running close to real-time budget: "
            f"wait={wait_s * 1e3:.2f} ms, compute={compute_s * 1e3:.2f} ms "
            f"over {n_iters} iteration(s) since the last check "
            f"(amortized {amortized_compute_s * 1e3:.4f} ms/iteration) "
            f"vs {budget_s * 1e3:.2f} ms budget ({n_active_tones} active tones)."
        )

    def _maybe_check_clipping(self, ch0_raw, ch1_raw) -> None:
        """Throttled the same way as :meth:`_maybe_warn_throughput`, and for
        the same reason: reading back whether anything exceeded the
        amplitude budget needs a host/GPU sync (``int()`` on a cupy
        reduction), so it's checked periodically instead of every
        iteration. This only gates the "should never happen" diagnostic
        log -- the clip applied in :meth:`_fill_loop` is unconditional and
        unaffected by this throttling, so the amplitude-safety invariant
        still holds every single iteration regardless of the check cadence.
        """
        now = time.monotonic()
        if now - self._last_clip_check_s < _CLIP_CHECK_INTERVAL_S:
            return
        self._last_clip_check_s = now
        n_clipped = int(cp.count_nonzero(cp.abs(ch0_raw) > self._max_value)) + int(
            cp.count_nonzero(cp.abs(ch1_raw) > self._max_value)
        )
        if n_clipped:
            log.error(
                f"SCAPP fill loop clipped {n_clipped} samples this chunk — "
                "amplitude-budget invariant violated upstream."
            )

    def _fill_loop(self) -> None:
        try:
            # Marks when the previous iteration's body finished, i.e. right
            # before this thread blocks inside `for card_buffer in
            # self._scapp_transfer` (the driver's wait_dma()) waiting for
            # the next chunk. The gap between that and t0 below is real
            # wall-clock time spent blocked in the driver, not in any of
            # this file's GPU work -- see _maybe_warn_throughput.
            t_prev_iter_end = time.perf_counter()
            for card_buffer in self._scapp_transfer:
                if self._stop_event.is_set():
                    break

                t0 = time.perf_counter()
                wait_s = t0 - t_prev_iter_end
                self._iters_since_throughput_check += 1
                with self._state_lock:
                    n_active_tones = len(self._active_segments)
                    channel_arrays = self._channel_arrays
                    chunk_start = self._next_fill_sample
                    self._next_fill_sample += self._config.notify_samples

                abs_sample = chunk_start + self._local_sample_idx

                # max_value is already baked into arrays.amplitude_frac (see
                # _build_channel_arrays), so no separate whole-buffer scale
                # multiply is needed here per channel per iteration.
                ch0 = self._sum_channel(channel_arrays[0], abs_sample)
                ch1 = self._sum_channel(channel_arrays[1], abs_sample)

                # Throttled diagnostic read-back (see _maybe_check_clipping);
                # the clip itself is unconditional so the amplitude-safety
                # invariant holds every iteration regardless.
                self._maybe_check_clipping(ch0, ch1)
                ch0 = cp.clip(ch0, -self._max_value, self._max_value)
                ch1 = cp.clip(ch1, -self._max_value, self._max_value)

                card_buffer[0, :] = ch0.astype(self._sample_dtype)
                card_buffer[1, :] = ch1.astype(self._sample_dtype)

                self._maybe_start_trigger()
                self._maybe_warn_throughput(t0, wait_s, n_active_tones)
                t_prev_iter_end = time.perf_counter()
        except (
            BaseException
        ) as exc:  # noqa: BLE001 - must capture and hand off, not swallow
            log.exception("SCAPP feeder thread failed")
            with self._state_lock:
                self._feeder_exception = exc
            self._stop_event.set()
            self._start_event.set()  # unblock start() waiters so they see the failure


def _magnitude(value) -> float:
    """Normalise a possibly-pint-quantity spcm return value to a plain float
    in base (SI) units.
    """
    if hasattr(value, "to_base_units"):
        return float(value.to_base_units().magnitude)
    if hasattr(value, "magnitude"):
        return float(value.magnitude)
    return float(value)
