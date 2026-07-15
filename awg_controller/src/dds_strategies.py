"""
DDS execution strategies for the Spectrum Instrumentation AWG card.
===================================================================

Four interchangeable paradigms for driving AOD frequency changes during
atom rearrangement.  All implement the :class:`DDSStrategy` interface and
can be swapped transparently in :class:`atommovrController`.

Strategies
----------
1. **DDSStreamingStrategy** - Current production approach:
   ``DDSCommandQueue`` + ``TIMER`` trigger + FIFO pre-fill.
2. **DDSRampStrategy** - FPGA-level frequency ramps via
   ``frequency_slope()`` (spcm examples 03, 04, 12).
3. **DDSPatternStrategy** - Pre-loaded patterns with ``CARD`` trigger
   synchronisation (spcm example 15).
4. **DDSCameraTriggeredStrategy** - External camera TTL replaces
   ``trigger.force()`` for fully hardware-synchronised feedback
   (spcm examples 09 + 15).

Timing
------
Move pacing is governed by **travel** —
`AWGBatch.total_duration_s` from `atommovr.utils.timing.travel_duration_s`
(Chebyshev × spacing / `AOD_speed`).

`HardwareConfig.trigger_timer_s` is the idle / holding TIMER only.

Safety
------
* **Maximum output voltage MUST stay below 2.0 V** in all scripts.
* Always verify amplifier output with an oscilloscope before connecting
  to the AOD.  Excessive voltage will damage the AOD driver.
"""

from __future__ import annotations

import abc
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from awg_controller.src.awg_control import AWGBatch, RFRamp

# Optional hardware driver (same guard as atommovr_controller.py)
try:
    import spcm

    _HW_AVAILABLE = True
except ImportError:
    spcm = None  # type: ignore[assignment]
    _HW_AVAILABLE = False

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transport timing (single authority: AWGBatch.total_duration_s / AOD_speed)
# ---------------------------------------------------------------------------

#: Idle-buffer target time used when sizing FIFO prefill writes.
PREFILL_BUFFER_S: float = 10.0

#: Cap on prefill writes so µs-scale idle timers cannot explode the queue.
PREFILL_MAX_COUNT: int = 64

#: Floor applied only to the prefill-count divisor (not to move duration).
PREFILL_MIN_TIMER_S: float = 1e-3


def transport_duration_s(batch: Any) -> float:
    """Return the **travel** window for *batch* (seconds).

    This is ``AWGBatch.total_duration_s`` from
    :func:`atommovr.utils.timing.travel_duration_s` (Chebyshev × spacing /
    ``AOD_speed``).  Holding batches return ``0.0``.
    """
    return max(float(getattr(batch, "total_duration_s", 0.0) or 0.0), 0.0)


def wait_transport(duration_s: float, already_waited_s: float = 0.0) -> None:
    """Sleep the remaining transport window after hardware work.

    Parameters
    ----------
    duration_s :
        Authoritative physics duration from :func:`transport_duration_s`.
    already_waited_s :
        Wall time already spent (e.g. pattern poll).  Only the remainder
        is slept so abrupt strategies never under-wait relative to
        ``AOD_speed``.
    """
    remaining = float(duration_s) - max(float(already_waited_s), 0.0)
    if remaining > 0.0:
        time.sleep(remaining)


def prefill_count_for_timer(timer_s: float) -> int:
    """Number of idle holding writes used to pre-fill the DDS FIFO.

    Uses a fixed ~``PREFILL_BUFFER_S`` seconds of *idle* ticks, clamped so
    short timers cannot request hundreds of thousands of writes.  Move
    pacing itself is **not** derived from this value.
    """
    t = max(float(timer_s), PREFILL_MIN_TIMER_S)
    return max(1, min(PREFILL_MAX_COUNT, math.floor(PREFILL_BUFFER_S / t)))


def program_timer(dds: Any, interval_s: float) -> None:
    """Program the card TIMER period used for the next trigger events."""
    dds.trg_timer(float(interval_s))


# ---------------------------------------------------------------------------
# Configuration data-classes
# ---------------------------------------------------------------------------


@dataclass
class RampConfig:
    """Tunables for :class:`DDSRampStrategy`.

    Attributes
    ----------
    ramp_stepsize : int
        Passed to ``dds.freq_ramp_stepsize()``.  Controls how many clock
        cycles elapse between FPGA frequency-register updates.
        Lower → finer ramp resolution.  1000 is the spcm example default.
    use_scurve : bool
        When *True*, the linear ramp is replaced by a piecewise-linear
        cosine S-curve (minimum-jerk profile, spcm example 12).
    scurve_segments : int
        Number of linear segments that approximate the S-curve.
        More segments → smoother profile, but more DDS commands.
        Segment duration is ``transport_duration_s / scurve_segments``.
    """

    ramp_stepsize: int = 1000
    use_scurve: bool = False
    scurve_segments: int = 16


@dataclass
class PatternConfig:
    """Tunables for :class:`DDSPatternStrategy`.

    Attributes
    ----------
    poll_interval_s : float
        Sleep between ``queue_cmd_count()`` polls (seconds).
    poll_timeout_s : float
        Maximum wait before declaring a pattern execution timeout.
        Safety ceiling only — transport pacing uses ``AOD_speed``.
    """

    poll_interval_s: float = 0.001
    poll_timeout_s: float = 10.0


@dataclass
class CameraTriggerConfig:
    """Tunables for :class:`DDSCameraTriggeredStrategy`.

    .. warning::

       ``trigger_level_v`` **MUST** be below 2.0 V.  The constructor
       raises ``ValueError`` if this limit is violated.

    Attributes
    ----------
    trigger_level_v : float
        External trigger threshold voltage.  Default 1.5 V.
    trigger_coupling : str
        ``"DC"`` or ``"AC"``.
    trigger_edge : str
        ``"rising"`` or ``"falling"``.
    trigger_termination_ohms : float
        Input termination impedance.
    poll_interval_s : float
        Sleep between ``queue_cmd_count()`` polls.
    poll_timeout_s : float
        Maximum wait for camera trigger + pattern completion.
        Camera frame rate is independent of ``AOD_speed``; this is only
        a safety timeout.  After the pattern completes, any remaining
        transport window is still slept.
    """

    trigger_level_v: float = 1.5
    trigger_coupling: str = "DC"
    trigger_edge: str = "rising"
    trigger_termination_ohms: float = 50.0
    poll_interval_s: float = 0.001
    poll_timeout_s: float = 30.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

#: Maximum safe trigger level.  Hard-coded to prevent accidental damage.
MAX_SAFE_TRIGGER_LEVEL_V: float = 2.0


def _core_const(idx: int):
    """Map an integer core index to the corresponding ``spcm.SPCM_DDS_COREn`` constant."""
    if spcm is None:
        return idx
    attr = f"SPCM_DDS_CORE{idx}"
    return getattr(spcm, attr, idx)


def _group_ramps_by_channel(batch: Any) -> Dict[int, list]:
    """Return ``{channel: [ramp, …]}`` sorted by channel number."""
    by_channel: Dict[int, list] = {}
    for ramp in batch.ramps:
        by_channel.setdefault(ramp.channel, []).append(ramp)
    return dict(sorted(by_channel.items()))


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class DDSStrategy(abc.ABC):
    """Interface that every DDS execution strategy must implement.

    The :class:`atommovrController` delegates all hardware-specific DDS
    logic to a ``DDSStrategy`` instance, making the paradigm swappable
    without touching the control-loop code.

    **Timing contract**: move pacing is governed by
    ``AWGBatch.total_duration_s`` (from ``AOD_speed``).
    ``HardwareConfig.trigger_timer_s`` is the *idle / holding* TIMER
    only, used at configure/prefill and restored after each move batch.

    Life-cycle (called by the controller)::

        dds = strategy.create_dds(card, channels)
        strategy.configure(dds, card, hw_config, core_map)
        strategy.prefill(dds, holding_batch, hw_config)
        strategy.start(stack)
        # … main loop …
        strategy.execute_batch(dds, batch)
        strategy.send_holding(dds, holding_batch)
        # … end of round …
        strategy.finalize_sequence(dds)
        strategy.shutdown(dds, card)
    """

    #: Idle / holding TIMER period (s).  Set in :meth:`configure`.
    _idle_timer_s: float = 0.2

    # -- identity ----------------------------------------------------------

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short, unique, human-readable name (e.g. ``"ramp"``)."""

    # -- hardware setup ----------------------------------------------------

    @abc.abstractmethod
    def create_dds(self, card: Any, channels: Any) -> Any:
        """Create the DDS driver object (``DDSCommandQueue`` or ``DDS``)."""

    @abc.abstractmethod
    def configure(
        self,
        dds: Any,
        card: Any,
        hw_config: Any,
        core_map: Dict[int, List[int]],
    ) -> None:
        """One-time configuration: trigger mode, core assignments, DMA."""

    @abc.abstractmethod
    def prefill(self, dds: Any, holding_batch: Any, hw_config: Any) -> None:
        """Pre-fill the command buffer (no-op for pattern strategies)."""

    @abc.abstractmethod
    def start(self, stack: Any) -> None:
        """Start the card stack for streaming / pattern execution."""

    # -- batch execution ---------------------------------------------------

    @abc.abstractmethod
    def execute_batch(self, dds: Any, batch: Any) -> None:
        """Execute one move batch on the DDS hardware."""

    @abc.abstractmethod
    def send_holding(self, dds: Any, batch: Any) -> None:
        """Send a static holding configuration (no motion)."""

    # -- optional hooks ----------------------------------------------------

    def finalize_sequence(self, dds: Any) -> None:
        """Called after all batches in a rearrangement round (optional)."""

    def shutdown(self, dds: Any, card: Any = None) -> None:
        """Strategy-specific cleanup (optional)."""

    # -- utilities ---------------------------------------------------------

    def _store_idle_timer(self, hw_config: Any) -> float:
        """Cache and return the idle TIMER from *hw_config*."""
        self._idle_timer_s = float(hw_config.trigger_timer_s)
        return self._idle_timer_s

    def _restore_idle_timer(self, dds: Any) -> None:
        """Re-program the card TIMER to the idle / holding period."""
        program_timer(dds, self._idle_timer_s)

    @staticmethod
    def compute_slope(ramp: Any) -> float:
        """Compute the linear frequency slope (Hz / s) for *ramp*.

        Uses ``ramp.duration_s`` (physics / ``AOD_speed``).  Returns 0.0
        for static ramps (no motion or zero duration).
        """
        if ramp.duration_s <= 0 or ramp.f_start == ramp.f_end:
            return 0.0
        return (ramp.f_end - ramp.f_start) / ramp.duration_s


# =========================================================================
# Strategy 1 - Streaming (current production approach)
# =========================================================================


class DDSStreamingStrategy(DDSStrategy):
    """FIFO streaming with ``DDSCommandQueue`` and ``TIMER`` trigger.

    This is the current production approach, extracted from
    ``atommovr_controller.py`` and ``cli.py``.

    **Paradigm**: ``DDSCommandQueue`` + ``SPCM_DDS_TRG_SRC_TIMER``
    + FIFO pre-fill + continuous ``write_to_card()`` calls.

    **Timing**: for each move batch the card TIMER is set to
    ``batch.total_duration_s`` (from ``AOD_speed``), the hop is queued,
    and the host waits that same duration.  Idle/holding uses
    ``HardwareConfig.trigger_timer_s``.

    Advantages
        * Simple command model (set freq → exec_at_trg → write).
        * Battle-tested in cli.py production code.

    Limitations
        * Abrupt frequency hops (instantaneous at trigger event).
        * Continuous FIFO feeding required (underrun risk).
        * Buffer pre-fill adds startup latency.
    """

    @property
    def name(self) -> str:
        return "streaming"

    # -- setup -------------------------------------------------------------

    def create_dds(self, card: Any, channels: Any) -> Any:
        dds = spcm.DDSCommandQueue(card, channels=channels)
        dds.reset()
        dds.data_transfer_mode(spcm.SPCM_DDS_DTM_DMA)
        return dds

    def configure(
        self,
        dds: Any,
        card: Any,
        hw_config: Any,
        core_map: Dict[int, List[int]],
    ) -> None:
        ch1_cores = core_map[1]
        if len(ch1_cores) <= 1:
            dds.cores_on_channel(1, _core_const(20))
        else:
            dds.cores_on_channel(1, *[_core_const(c) for c in ch1_cores])

        idle_s = self._store_idle_timer(hw_config)
        dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)
        program_timer(dds, idle_s)

    def prefill(self, dds: Any, holding_batch: Any, hw_config: Any) -> None:
        count = prefill_count_for_timer(hw_config.trigger_timer_s)
        for _ in range(count):
            self._write_batch(dds, holding_batch)
            dds.exec_at_trg()
            dds.write_to_card()
            dds.mode = dds.WRITE_MODE.WAIT_IF_FULL

    def start(self, stack: Any) -> None:
        stack.start(
            spcm.M2CMD_CARD_ENABLETRIGGER,
            spcm.M2CMD_CARD_FORCETRIGGER,
        )

    # -- execution ---------------------------------------------------------

    def execute_batch(self, dds: Any, batch: Any) -> None:
        duration_s = transport_duration_s(batch)
        if duration_s > 0:
            program_timer(dds, duration_s)

        self._write_batch(dds, batch)
        dds.exec_at_trg()
        dds.write_to_card()

        if dds.status() & spcm.SPCM_DDS_STAT_QUEUE_UNDERRUN:
            log.warning("DDS queue underrun detected (streaming strategy)!")

        wait_transport(duration_s)
        if duration_s > 0:
            self._restore_idle_timer(dds)

    def send_holding(self, dds: Any, batch: Any) -> None:
        self._restore_idle_timer(dds)
        self.execute_batch(dds, batch)

    # -- internal ----------------------------------------------------------

    @staticmethod
    def _write_batch(dds: Any, batch: Any) -> None:
        """Write freq / phase / amp for every core, grouped by channel."""
        by_channel = _group_ramps_by_channel(batch)
        for _ch, ramps in by_channel.items():
            for ramp in ramps:
                dds[ramp.core].freq(float(ramp.f_end))
                dds[ramp.core].phase(ramp.phase_deg * spcm.units.deg)
                dds[ramp.core].amp(ramp.amplitude_pct * spcm.units.percent)
            dds.exec_at_trg()


# =========================================================================
# Strategy 2 - Hardware Frequency Ramps
# =========================================================================


class DDSRampStrategy(DDSStrategy):
    """FPGA-level frequency ramps via ``frequency_slope()``.

    Based on spcm DDS examples 03, 04, and 12.

    Instead of abrupt frequency hops, the FPGA autonomously increments
    each core's frequency register at a computed slope (Hz / s).  This
    produces smooth transport trajectories, reducing atom loss.

    **Paradigm**: ``DDSCommandQueue`` + ``SPCM_DDS_TRG_SRC_TIMER``
    + ``frequency_slope()`` for smooth sweeps.

    Batch execution (3 trigger events per move)::

        Trigger 1 → set initial frequencies + amplitudes
        Trigger 2 → activate frequency_slope on each core  (ramp starts)
        Trigger 3 → frequency_slope(0) + set exact final freq  (ramp stops)

    The ramp runs between triggers 2 and 3 for exactly
    ``batch.total_duration_s`` (from ``AOD_speed``).  The slope is::

        slope = (f_end - f_start) / total_duration_s

    **S-curve support** (example 12):  When ``RampConfig.use_scurve`` is
    *True*, the linear ramp is replaced with a piecewise-linear cosine
    profile.  Each segment lasts ``total_duration_s / n_segments``; slope
    math uses the same ``T = total_duration_s``.

    Advantages
        * Smooth frequency transitions (no abrupt hops).
        * FPGA handles interpolation — zero software timing jitter.
        * Optional S-curve for jerk-free transport.

    Limitations
        * 3 trigger events per batch (vs. 1 for streaming); host wait is
          ``3 × duration`` (linear) or ``(n_seg+2) × segment_dt`` (S-curve).
        * More DDS commands (especially with S-curve segments).

    Parameters
    ----------
    config : RampConfig, optional
    """

    def __init__(self, config: Optional[RampConfig] = None) -> None:
        self.config = config or RampConfig()

    @property
    def name(self) -> str:
        return "ramp"

    # -- setup -------------------------------------------------------------

    def create_dds(self, card: Any, channels: Any) -> Any:
        dds = spcm.DDSCommandQueue(card, channels=channels)
        dds.reset()
        dds.data_transfer_mode(spcm.SPCM_DDS_DTM_DMA)
        return dds

    def configure(
        self,
        dds: Any,
        card: Any,
        hw_config: Any,
        core_map: Dict[int, List[int]],
    ) -> None:
        ch1_cores = core_map[1]
        if len(ch1_cores) <= 1:
            dds.cores_on_channel(1, _core_const(20))
        else:
            dds.cores_on_channel(1, *[_core_const(c) for c in ch1_cores])

        idle_s = self._store_idle_timer(hw_config)
        dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)
        program_timer(dds, idle_s)

        # Ramp step-size: how many clock cycles between FPGA freq updates
        dds.freq_ramp_stepsize(self.config.ramp_stepsize)

    def prefill(self, dds: Any, holding_batch: Any, hw_config: Any) -> None:
        count = prefill_count_for_timer(hw_config.trigger_timer_s)
        for _ in range(count):
            for ramp in holding_batch.ramps:
                dds[ramp.core].freq(float(ramp.f_end))
                dds[ramp.core].phase(ramp.phase_deg * spcm.units.deg)
                dds[ramp.core].amp(ramp.amplitude_pct * spcm.units.percent)
                dds[ramp.core].frequency_slope(0.0)
            dds.exec_at_trg()
            dds.write_to_card()
            dds.mode = dds.WRITE_MODE.WAIT_IF_FULL

    def start(self, stack: Any) -> None:
        stack.start(
            spcm.M2CMD_CARD_ENABLETRIGGER,
            spcm.M2CMD_CARD_FORCETRIGGER,
        )

    # -- execution ---------------------------------------------------------

    def execute_batch(self, dds: Any, batch: Any) -> None:
        if not batch.ramps:
            return

        if transport_duration_s(batch) <= 0:
            self.send_holding(dds, batch)
            return

        if self.config.use_scurve:
            self._execute_scurve(dds, batch)
        else:
            self._execute_linear(dds, batch)

    def send_holding(self, dds: Any, batch: Any) -> None:
        """Set static frequencies with zero slopes."""
        self._restore_idle_timer(dds)
        for ramp in batch.ramps:
            dds[ramp.core].freq(float(ramp.f_end))
            dds[ramp.core].phase(ramp.phase_deg * spcm.units.deg)
            dds[ramp.core].amp(ramp.amplitude_pct * spcm.units.percent)
            dds[ramp.core].frequency_slope(0.0)
        dds.exec_at_trg()
        dds.write_to_card()

    # -- linear ramp -------------------------------------------------------

    def _execute_linear(self, dds: Any, batch: Any) -> None:
        """Three-trigger linear ramp (spcm examples 03 / 04)."""
        duration_s = transport_duration_s(batch)
        program_timer(dds, duration_s)

        # Trigger 1 — initial frequencies + amplitudes
        for ramp in batch.ramps:
            dds[ramp.core].freq(float(ramp.f_start))
            dds[ramp.core].phase(ramp.phase_deg * spcm.units.deg)
            dds[ramp.core].amp(ramp.amplitude_pct * spcm.units.percent)
        dds.exec_at_trg()

        # Trigger 2 — start ramps (slope spans one transport window)
        for ramp in batch.ramps:
            slope = self._slope_for_duration(ramp, duration_s)
            dds[ramp.core].frequency_slope(slope)
        dds.exec_at_trg()

        # Trigger 3 — stop ramps + exact final frequencies
        for ramp in batch.ramps:
            dds[ramp.core].frequency_slope(0.0)
            dds[ramp.core].freq(float(ramp.f_end))
        dds.exec_at_trg()

        dds.write_to_card()

        # Wait for all 3 trigger intervals (middle one is the transport)
        time.sleep(duration_s * 3)
        self._restore_idle_timer(dds)

    # -- S-curve ramp (example 12) -----------------------------------------

    def _execute_scurve(self, dds: Any, batch: Any) -> None:
        """Piecewise-linear cosine S-curve (spcm example 12)."""
        n_seg = self.config.scurve_segments
        duration_s = transport_duration_s(batch)
        segment_dt = duration_s / n_seg
        program_timer(dds, segment_dt)

        # Pre-compute per-core segment slopes over the same T = duration_s
        all_slopes: List[List[float]] = [
            self._compute_scurve_slopes(ramp, n_seg, duration_s=duration_s)
            for ramp in batch.ramps
        ]

        # Step 1 — initial frequencies
        for ramp in batch.ramps:
            dds[ramp.core].freq(float(ramp.f_start))
            dds[ramp.core].phase(ramp.phase_deg * spcm.units.deg)
            dds[ramp.core].amp(ramp.amplitude_pct * spcm.units.percent)
        dds.exec_at_trg()

        # Steps 2 … N+1 — piecewise slopes (N × segment_dt = duration_s)
        for seg_idx in range(n_seg):
            for ramp_idx, ramp in enumerate(batch.ramps):
                dds[ramp.core].frequency_slope(all_slopes[ramp_idx][seg_idx])
            dds.exec_at_trg()

        # Final step — stop ramps + exact end frequencies
        for ramp in batch.ramps:
            dds[ramp.core].frequency_slope(0.0)
            dds[ramp.core].freq(float(ramp.f_end))
        dds.exec_at_trg()

        dds.write_to_card()

        # Total trigger events: 1 (init) + n_seg (slopes) + 1 (final)
        total_triggers = 1 + n_seg + 1
        time.sleep(segment_dt * total_triggers)
        self._restore_idle_timer(dds)

    # -- slope helpers -----------------------------------------------------

    @staticmethod
    def _slope_for_duration(ramp: Any, duration_s: float) -> float:
        """Compute slope so the ramp covers ``delta_f`` in *duration_s*."""
        delta_f = ramp.f_end - ramp.f_start
        if abs(delta_f) < 1e-3 or duration_s <= 0:
            return 0.0
        return delta_f / duration_s

    # Back-compat alias used by older tests / callers
    _slope_for_timer = _slope_for_duration

    @staticmethod
    def _compute_scurve_slopes(
        ramp: Any,
        n_segments: int,
        duration_s: Optional[float] = None,
    ) -> List[float]:
        """Piecewise-linear slopes for a cosine S-curve profile.

        The raised-cosine position profile is::

            f(t) = f_start + Δf · (1 − cos(π · t / T)) / 2

        Its derivative gives the instantaneous slope at each segment
        midpoint.  ``T`` and each segment's wall-clock length must agree:
        ``T = duration_s``, ``dt = T / n_segments``.

        Based on spcm example 12.
        """
        delta_f = ramp.f_end - ramp.f_start
        T = float(duration_s) if duration_s is not None else float(ramp.duration_s)
        if abs(delta_f) < 1e-3 or T <= 0:
            return [0.0] * n_segments

        slopes: List[float] = []
        for i in range(n_segments):
            t_mid = (i + 0.5) * T / n_segments
            slope = delta_f * math.pi / (2 * T) * math.sin(math.pi * t_mid / T)
            slopes.append(slope)
        return slopes


# =========================================================================
# Strategy 3 - Pattern-Based
# =========================================================================


class DDSPatternStrategy(DDSStrategy):
    """Pre-loaded patterns with ``CARD`` trigger synchronisation.

    Based on spcm DDS example 15 (repeated patterns).

    Instead of continuous FIFO streaming, a complete move pattern is
    pre-loaded to the card.  ``TIMER`` paces the inter-step transitions;
    a final ``CARD`` trigger acts as a pause / sync point.
    ``trigger.force()`` starts each pattern, and ``queue_cmd_count()``
    is polled for completion.

    **Timing**: inter-step TIMER is set to ``batch.total_duration_s``.
    After the pattern queue drains, any remaining transport window is
    slept so the host never under-waits relative to ``AOD_speed``.

    **Paradigm**: ``spcm.DDS`` + ``TRG_SRC_TIMER`` during steps
    + ``TRG_SRC_CARD`` at end + ``trigger.force()`` + polling.

    Pattern execution sequence::

        1.  trg_src(TIMER)
        2.  Set initial frequencies → exec_at_trg()
        3.  Set final   frequencies → exec_at_trg()
        4.  trg_src(CARD) → exec_at_trg()          ← pause point
        5.  write_to_card()
        6.  trigger.force()                         ← start pattern
        7.  poll queue_cmd_count() until 0           ← wait
        8.  sleep remaining transport window

    Advantages
        * No FIFO underrun risk (pattern is pre-loaded).
        * Deterministic execution timing.
        * Natural sync point between patterns.

    Limitations
        * Abrupt frequency hops (like streaming).
        * Polling overhead between patterns.
        * Pattern-load latency per batch.

    Parameters
    ----------
    config : PatternConfig, optional
    """

    def __init__(self, config: Optional[PatternConfig] = None) -> None:
        self.config = config or PatternConfig()
        self._trigger: Any = None

    @property
    def name(self) -> str:
        return "pattern"

    # -- setup -------------------------------------------------------------

    def create_dds(self, card: Any, channels: Any) -> Any:
        # spcm.DDS (not DDSCommandQueue) for pattern approach
        dds = spcm.DDS(card, channels=channels)
        dds.reset()
        dds.data_transfer_mode(spcm.SPCM_DDS_DTM_DMA)
        return dds

    def configure(
        self,
        dds: Any,
        card: Any,
        hw_config: Any,
        core_map: Dict[int, List[int]],
    ) -> None:
        ch1_cores = core_map[1]
        if len(ch1_cores) <= 1:
            dds.cores_on_channel(1, _core_const(20))
        else:
            dds.cores_on_channel(1, *[_core_const(c) for c in ch1_cores])

        idle_s = self._store_idle_timer(hw_config)
        dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)
        program_timer(dds, idle_s)

        # Disable all external trigger sources — use force() only
        self._trigger = spcm.Trigger(card)
        self._trigger.or_mask(spcm.SPC_TM_NONE)

    def prefill(self, dds: Any, holding_batch: Any, hw_config: Any) -> None:
        # No FIFO pre-fill needed; just set the initial idle state
        for ramp in holding_batch.ramps:
            dds[ramp.core].amp(ramp.amplitude_pct * spcm.units.percent)
            dds[ramp.core].freq(float(ramp.f_end))
            dds[ramp.core].phase(ramp.phase_deg * spcm.units.deg)
        dds.exec_at_trg()
        dds.write_to_card()

    def start(self, stack: Any) -> None:
        stack.start(
            spcm.M2CMD_CARD_ENABLETRIGGER,
            spcm.M2CMD_CARD_FORCETRIGGER,
        )

    # -- execution ---------------------------------------------------------

    def execute_batch(self, dds: Any, batch: Any) -> None:
        if not batch.ramps:
            return

        duration_s = transport_duration_s(batch)
        if duration_s > 0:
            program_timer(dds, duration_s)

        # TIMER for inter-step pacing
        dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)

        t0 = time.monotonic()

        # Load initial frequencies
        for ramp in batch.ramps:
            dds[ramp.core].freq(float(ramp.f_start))
            dds[ramp.core].phase(ramp.phase_deg * spcm.units.deg)
            dds[ramp.core].amp(ramp.amplitude_pct * spcm.units.percent)
        dds.exec_at_trg()

        # Load final frequencies (the actual move)
        for ramp in batch.ramps:
            dds[ramp.core].freq(float(ramp.f_end))
        dds.exec_at_trg()

        # Pause point: CARD trigger at end
        dds.trg_src(spcm.SPCM_DDS_TRG_SRC_CARD)
        dds.exec_at_trg()

        # Flush and start
        dds.write_to_card()
        if self._trigger is not None:
            self._trigger.force()

        # Wait for completion, then any remaining AOD_speed window
        self._poll_completion(dds)
        wait_transport(duration_s, already_waited_s=time.monotonic() - t0)
        if duration_s > 0:
            self._restore_idle_timer(dds)

    def send_holding(self, dds: Any, batch: Any) -> None:
        self._restore_idle_timer(dds)
        dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)

        for ramp in batch.ramps:
            dds[ramp.core].freq(float(ramp.f_end))
            dds[ramp.core].phase(ramp.phase_deg * spcm.units.deg)
            dds[ramp.core].amp(ramp.amplitude_pct * spcm.units.percent)
        dds.exec_at_trg()

        # Pause after holding is set
        dds.trg_src(spcm.SPCM_DDS_TRG_SRC_CARD)
        dds.exec_at_trg()

        dds.write_to_card()
        if self._trigger is not None:
            self._trigger.force()
        self._poll_completion(dds)

    def shutdown(self, dds: Any, card: Any = None) -> None:
        self._trigger = None

    # -- internal ----------------------------------------------------------

    def _poll_completion(self, dds: Any) -> None:
        deadline = time.monotonic() + self.config.poll_timeout_s
        while dds.queue_cmd_count() > 0:
            if time.monotonic() > deadline:
                log.error(
                    "Pattern execution timed out after "
                    f"{self.config.poll_timeout_s} s"
                )
                break
            time.sleep(self.config.poll_interval_s)


# =========================================================================
# Strategy 4 - Camera-Triggered Pattern Execution
# =========================================================================


class DDSCameraTriggeredStrategy(DDSStrategy):
    """External camera TTL triggers pattern execution.

    Combines spcm examples 09 (external trigger) and 15 (patterns).

    Identical to :class:`DDSPatternStrategy` except that the pattern
    start signal comes from a hardware TTL edge on the card's ``ext0``
    input (e.g. a camera *frame-ready* pulse) instead of a software
    ``trigger.force()``.  This creates a fully hardware-synchronised
    feedback loop::

        camera exposure → TTL pulse → AWG pattern start → AOD → atoms

    **Timing**: RF inter-step TIMER follows ``batch.total_duration_s``.
    The camera TTL period is *not* derived from ``AOD_speed``.  After the
    pattern queue drains, any remaining transport window is slept.

    .. danger::

       **``trigger_level_v`` MUST be below 2.0 V.**  The constructor
       raises ``ValueError`` immediately if this limit is exceeded.
       Exceeding 2.0 V risks permanent damage to the AOD amplifier.

    Batch execution sequence::

        1.  trg_src(TIMER)
        2.  Load frequency steps → exec_at_trg()
        3.  trg_src(CARD) at end → exec_at_trg()   ← wait for camera
        4.  write_to_card()
        5.  (hardware waits for ext0 TTL edge)
        6.  poll queue_cmd_count() until 0
        7.  sleep remaining transport window

    Advantages
        * Fully hardware-synchronised (zero software jitter).
        * Deterministic camera → transport timing.
        * No FIFO underrun risk.

    Limitations
        * Requires physical TTL wiring (camera → ext0).
        * Loop rate limited by camera frame rate.
        * More complex experimental setup.
        * ``trigger_level_v`` calibration critical.

    Parameters
    ----------
    config : CameraTriggerConfig, optional
    """

    def __init__(self, config: Optional[CameraTriggerConfig] = None) -> None:
        self.config = config or CameraTriggerConfig()
        self._trigger: Any = None

        # SAFETY: hard fail if trigger level is unsafe
        if self.config.trigger_level_v >= MAX_SAFE_TRIGGER_LEVEL_V:
            raise ValueError(
                f"Trigger level {self.config.trigger_level_v} V "
                f">= {MAX_SAFE_TRIGGER_LEVEL_V} V safety limit!  "
                f"Risk of AOD amplifier damage."
            )

    @property
    def name(self) -> str:
        return "camera_triggered"

    # -- setup -------------------------------------------------------------

    def create_dds(self, card: Any, channels: Any) -> Any:
        dds = spcm.DDS(card, channels=channels)
        dds.reset()
        dds.data_transfer_mode(spcm.SPCM_DDS_DTM_DMA)
        return dds

    def configure(
        self,
        dds: Any,
        card: Any,
        hw_config: Any,
        core_map: Dict[int, List[int]],
    ) -> None:
        ch1_cores = core_map[1]
        if len(ch1_cores) <= 1:
            dds.cores_on_channel(1, _core_const(20))
        else:
            dds.cores_on_channel(1, *[_core_const(c) for c in ch1_cores])

        idle_s = self._store_idle_timer(hw_config)
        dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)
        program_timer(dds, idle_s)

        # --- external trigger from camera (spcm example 09) ---
        self._trigger = spcm.Trigger(card)
        self._trigger.or_mask(spcm.SPC_TMASK_EXT0)

        if self.config.trigger_edge == "rising":
            self._trigger.ext0_mode(spcm.SPC_TM_POS)
        else:
            self._trigger.ext0_mode(spcm.SPC_TM_NEG)

        # SAFETY: level must be < 2.0 V (enforced in __init__)
        self._trigger.ext0_level0(self.config.trigger_level_v * spcm.units.V)

        if self.config.trigger_coupling == "DC":
            self._trigger.ext0_coupling(spcm.COUPLING_DC)
        else:
            self._trigger.ext0_coupling(spcm.COUPLING_AC)

        # Input termination (50 Ω)
        if hasattr(spcm, "SPCM_50OHM_ACTIVE"):
            self._trigger.ext0_termination(spcm.SPCM_50OHM_ACTIVE)

        log.info(
            "Camera trigger configured: ext0, "
            f"level={self.config.trigger_level_v} V, "
            f"edge={self.config.trigger_edge}, "
            f"coupling={self.config.trigger_coupling}"
        )

    def prefill(self, dds: Any, holding_batch: Any, hw_config: Any) -> None:
        for ramp in holding_batch.ramps:
            dds[ramp.core].amp(ramp.amplitude_pct * spcm.units.percent)
            dds[ramp.core].freq(float(ramp.f_end))
            dds[ramp.core].phase(ramp.phase_deg * spcm.units.deg)
        dds.exec_at_trg()
        dds.write_to_card()

    def start(self, stack: Any) -> None:
        # Enable trigger but do NOT force — wait for camera TTL
        stack.start(spcm.M2CMD_CARD_ENABLETRIGGER)

    # -- execution ---------------------------------------------------------

    def execute_batch(self, dds: Any, batch: Any) -> None:
        if not batch.ramps:
            return

        duration_s = transport_duration_s(batch)
        if duration_s > 0:
            program_timer(dds, duration_s)

        # TIMER for inter-step pacing
        dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)

        t0 = time.monotonic()

        # Load initial frequencies
        for ramp in batch.ramps:
            dds[ramp.core].freq(float(ramp.f_start))
            dds[ramp.core].phase(ramp.phase_deg * spcm.units.deg)
            dds[ramp.core].amp(ramp.amplitude_pct * spcm.units.percent)
        dds.exec_at_trg()

        # Load final frequencies
        for ramp in batch.ramps:
            dds[ramp.core].freq(float(ramp.f_end))
        dds.exec_at_trg()

        # Wait for camera TTL at end
        dds.trg_src(spcm.SPCM_DDS_TRG_SRC_CARD)
        dds.exec_at_trg()

        dds.write_to_card()

        # No force trigger — ext0 TTL will start execution
        self._poll_completion(dds)
        wait_transport(duration_s, already_waited_s=time.monotonic() - t0)
        if duration_s > 0:
            self._restore_idle_timer(dds)

    def send_holding(self, dds: Any, batch: Any) -> None:
        """Send holding config.  Uses force-trigger (no camera event expected)."""
        self._restore_idle_timer(dds)
        dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)

        for ramp in batch.ramps:
            dds[ramp.core].freq(float(ramp.f_end))
            dds[ramp.core].phase(ramp.phase_deg * spcm.units.deg)
            dds[ramp.core].amp(ramp.amplitude_pct * spcm.units.percent)
        dds.exec_at_trg()

        dds.trg_src(spcm.SPCM_DDS_TRG_SRC_CARD)
        dds.exec_at_trg()
        dds.write_to_card()

        # For holding, force-trigger since camera may not fire
        if self._trigger is not None:
            self._trigger.force()
        self._poll_completion(dds)

    def shutdown(self, dds: Any, card: Any = None) -> None:
        self._trigger = None

    # -- internal ----------------------------------------------------------

    def _poll_completion(self, dds: Any) -> None:
        deadline = time.monotonic() + self.config.poll_timeout_s
        while dds.queue_cmd_count() > 0:
            if time.monotonic() > deadline:
                log.error(
                    "Camera-triggered pattern timed out after "
                    f"{self.config.poll_timeout_s} s.  "
                    "Check camera TTL connection to ext0."
                )
                break
            time.sleep(self.config.poll_interval_s)


# =========================================================================
# Strategy registry
# =========================================================================

STRATEGY_REGISTRY: Dict[str, type] = {
    "streaming": DDSStreamingStrategy,
    "ramp": DDSRampStrategy,
    "pattern": DDSPatternStrategy,
    "camera_triggered": DDSCameraTriggeredStrategy,
}


def get_strategy(name: str, **kwargs: Any) -> DDSStrategy:
    """Create a strategy instance by name.

    Parameters
    ----------
    name : str
        One of ``"streaming"``, ``"ramp"``, ``"pattern"``,
        ``"camera_triggered"``.
    **kwargs
        Forwarded to the strategy's configuration dataclass.

    Returns
    -------
    DDSStrategy

    Raises
    ------
    ValueError
        Unknown strategy name.
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown DDS strategy '{name}'.  "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )

    _CONFIG_MAP = {
        "ramp": RampConfig,
        "pattern": PatternConfig,
        "camera_triggered": CameraTriggerConfig,
    }

    cls = STRATEGY_REGISTRY[name]
    if name in _CONFIG_MAP and kwargs:
        config = _CONFIG_MAP[name](**kwargs)
        return cls(config=config)
    return cls()
