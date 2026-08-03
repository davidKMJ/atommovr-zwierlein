#!/usr/bin/env python3
"""Standalone streaming + recording monitor for a Siglent SVA1015X.

Not a package module -- run it directly, in parallel with the AWG/ramp
controller, as its own process:

    python awg_controller/scripts/sva1015x_monitor.py \\
        --resource TCPIP0::192.168.1.50::INSTR --threshold-dbm -20

Connects over VISA (LAN or USB, via ``pyvisa``/``pyvisa-py`` -- see the
``hardware`` extra in ``pyproject.toml``), polls the active trace, and
writes everything under a timestamped ``runs/sva1015x_<stamp>/`` folder:

* ``events.jsonl``     -- append-only ledger of every logged event (see below)
* ``summary.csv``      -- one row per sweep: timestamp, peak power/freq, whether
  a capture burst is currently active
* ``latest_trace.npz`` -- the most recent full sweep (``freqs_hz``, ``trace_dbm``,
  ``timestamp``), overwritten every poll regardless of recording state -- written
  atomically (temp file + rename) so a concurrent reader never sees a partial
  write. This is what a live "plot the analyzer's screen" viewer should poll,
  since it doesn't require a second VISA connection to the instrument.
* ``capture_NNNN/`` -- full-resolution trace snapshots (``.npz``), one folder
  per triggered recording burst, plus a ``meta.json`` with start/stop reason

Event system
------------
Two custom ``logging`` levels sit between the standard ones:

* ``MARKER``  (25, between INFO and WARNING) -- annotate the timeline only,
  no side effect. Use ``log.marker(...)`` for things like "monitor started",
  "N sweeps recorded", "instrument reconnected".
* ``TRIGGER`` (35, between WARNING and ERROR) -- actually starts/stops a
  recording burst. Use ``log.trigger(msg, extra={"fields": {"action": "start"}})``
  (or ``"stop"``) -- any other ``log.trigger(...)`` call is just a ledger
  entry with no side effect, same as a marker.

Every record, at any level, lands in ``events.jsonl`` via
``JsonlEventHandler`` -- that's the durable timeline. ``TriggerRecordingHandler``
additionally watches for ``TRIGGER``-level records carrying a start/stop
``action`` and flips a shared ``RecordingSession`` on/off accordingly. The
built-in threshold check (``--threshold-dbm``/``--hysteresis-db``) is one
example trigger source; add more by calling ``log.trigger(...)`` with your
own condition.

SCPI note
---------
``InstrumentConfig``'s query/command strings match Siglent's SSA3000X/
SVA1000X-series programming guide as of this writing. Verify against your
unit's actual manual/firmware and adjust the dataclass fields if your
syntax differs -- nothing else in this file needs to change.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pyvisa
except ImportError:
    pyvisa = None

# ---------------------------------------------------------------------------
# Custom logging levels
# ---------------------------------------------------------------------------

MARKER = 25
TRIGGER = 35
logging.addLevelName(MARKER, "MARKER")
logging.addLevelName(TRIGGER, "TRIGGER")


def _marker(self: logging.Logger, message: str, *args, **kwargs) -> None:
    if self.isEnabledFor(MARKER):
        self._log(MARKER, message, args, **kwargs)


def _trigger(self: logging.Logger, message: str, *args, **kwargs) -> None:
    if self.isEnabledFor(TRIGGER):
        self._log(TRIGGER, message, args, **kwargs)


logging.Logger.marker = _marker
logging.Logger.trigger = _trigger


class JsonlEventHandler(logging.Handler):
    """Append every log record as one JSON line -- the durable event ledger.

    Markers, triggers, and ordinary INFO/WARNING/ERROR records all land
    here, so the whole run has one append-only timeline that can be
    replayed or cross-referenced against the recorded traces later.
    """

    def __init__(self, path: Path):
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        payload = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload["fields"] = fields
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")


def _write_latest_trace(
    run_dir: Path, freqs_hz: np.ndarray, trace_dbm: np.ndarray, timestamp: str
) -> None:
    """Atomically overwrite ``run_dir/latest_trace.npz`` with the newest sweep.

    Writes to a temp file and ``os.replace``s it into place so a concurrent
    reader (e.g. a notebook polling this file to plot the live trace) never
    observes a partially-written file -- ``os.replace`` is atomic on both
    POSIX and Windows.
    """
    tmp_path = run_dir / ".latest_trace.npz.tmp"
    final_path = run_dir / "latest_trace.npz"
    with tmp_path.open("wb") as fh:
        np.savez(fh, freqs_hz=freqs_hz, trace_dbm=trace_dbm, timestamp=timestamp)
    os.replace(tmp_path, final_path)


# ---------------------------------------------------------------------------
# Recording session (owns the on-disk capture directory for one burst)
# ---------------------------------------------------------------------------


class RecordingSession:
    """Full-resolution trace capture, active only between a start/stop trigger."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.active = False
        self.capture_dir: Optional[Path] = None
        self._index = 0
        self._capture_count = 0

    def start(self, reason: str) -> None:
        self._capture_count += 1
        self.capture_dir = self.run_dir / f"capture_{self._capture_count:04d}"
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self._index = 0
        self.active = True
        (self.capture_dir / "meta.json").write_text(
            json.dumps(
                {
                    "reason": reason,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
        )

    def stop(self, reason: str) -> None:
        if not self.active:
            return
        self.active = False
        if self.capture_dir is not None:
            meta_path = self.capture_dir / "meta.json"
            meta = json.loads(meta_path.read_text())
            meta["stopped_at"] = datetime.now(timezone.utc).isoformat()
            meta["stop_reason"] = reason
            meta["n_traces"] = self._index
            meta_path.write_text(json.dumps(meta, indent=2))
        self.capture_dir = None

    def save_trace(
        self, freqs_hz: np.ndarray, trace_dbm: np.ndarray, timestamp: str
    ) -> None:
        if not self.active or self.capture_dir is None:
            return
        np.savez(
            self.capture_dir / f"trace_{self._index:05d}.npz",
            freqs_hz=freqs_hz,
            trace_dbm=trace_dbm,
            timestamp=timestamp,
        )
        self._index += 1


class TriggerRecordingHandler(logging.Handler):
    """React to TRIGGER-level records: start/stop the shared ``RecordingSession``.

    Only records logged via ``log.trigger(..., extra={"fields": {"action":
    "start"|"stop"}})`` change recording state -- a plain ``log.trigger(...)``
    with no ``action`` field (or any other level) is a no-op here, so it's
    safe to also route ordinary ERROR/WARNING records through the same
    logger without accidentally toggling a capture.
    """

    def __init__(self, session: RecordingSession):
        super().__init__(level=TRIGGER)
        self.session = session

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno != TRIGGER:
            return
        fields = getattr(record, "fields", None) or {}
        action = fields.get("action")
        if action == "start":
            self.session.start(reason=record.getMessage())
        elif action == "stop":
            self.session.stop(reason=record.getMessage())


# ---------------------------------------------------------------------------
# Instrument + monitor configuration
# ---------------------------------------------------------------------------


@dataclass
class InstrumentConfig:
    """VISA resource + SCPI strings for the SVA1015X. See module docstring's
    "SCPI note" -- adjust the query/command strings here if your firmware's
    dialect differs; nothing else in this file needs to change."""

    resource: str = "TCPIP0::192.168.1.50::INSTR"
    visa_library: str = "@py"  # pure-Python pyvisa-py backend; "" for NI-VISA
    timeout_ms: int = 5000
    continuous_sweep_cmd: str = ":INIT:CONT ON"
    trace_query: str = ":TRAC:DATA? TRACE1"
    start_freq_query: str = ":FREQ:STAR?"
    stop_freq_query: str = ":FREQ:STOP?"


@dataclass
class MonitorConfig:
    poll_interval_s: float = 1.0
    threshold_dbm: float = -20.0
    hysteresis_db: float = 3.0
    run_root: Path = Path("runs")
    console_level: int = logging.INFO


class SVA1015XMonitor:
    """Connects to the SVA1015X, streams trace data, and drives the event log."""

    def __init__(self, instrument: InstrumentConfig, monitor: MonitorConfig):
        self.instrument_cfg = instrument
        self.monitor_cfg = monitor
        self.inst = None
        self._above_threshold = False

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(monitor.run_root) / f"sva1015x_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.log = logging.getLogger("sva1015x")
        self.log.setLevel(logging.DEBUG)
        self.log.propagate = False
        self.log.handlers.clear()

        console = logging.StreamHandler()
        console.setLevel(monitor.console_level)
        console.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
        )
        self.log.addHandler(console)

        self.log.addHandler(JsonlEventHandler(self.run_dir / "events.jsonl"))

        self.session = RecordingSession(self.run_dir)
        self.log.addHandler(TriggerRecordingHandler(self.session))

    def connect(self) -> None:
        if pyvisa is None:
            raise RuntimeError(
                "pyvisa is required -- `pip install -e .[hardware]` or "
                "`pip install pyvisa pyvisa-py`."
            )
        rm = pyvisa.ResourceManager(self.instrument_cfg.visa_library or "")
        self.inst = rm.open_resource(self.instrument_cfg.resource)
        self.inst.timeout = self.instrument_cfg.timeout_ms
        idn = self.inst.query("*IDN?").strip()
        self.log.info(
            f"connected to {idn}",
            extra={"fields": {"resource": self.instrument_cfg.resource}},
        )
        self.inst.write(self.instrument_cfg.continuous_sweep_cmd)

    def _fetch_trace(self) -> tuple[np.ndarray, np.ndarray]:
        raw = self.inst.query(self.instrument_cfg.trace_query)
        trace_dbm = np.array(
            [float(v) for v in raw.strip().split(",") if v], dtype=np.float64
        )
        f_start = float(self.inst.query(self.instrument_cfg.start_freq_query))
        f_stop = float(self.inst.query(self.instrument_cfg.stop_freq_query))
        freqs_hz = np.linspace(f_start, f_stop, trace_dbm.size)
        return freqs_hz, trace_dbm

    def _check_threshold(
        self, freqs_hz: np.ndarray, trace_dbm: np.ndarray
    ) -> tuple[float, float]:
        """Peak-power threshold crossing -- the built-in example trigger
        source. Add more triggers by calling ``self.log.trigger(...)``
        elsewhere with your own condition."""
        idx = int(np.argmax(trace_dbm))
        peak_dbm = float(trace_dbm[idx])
        peak_hz = float(freqs_hz[idx])
        cfg = self.monitor_cfg

        if not self._above_threshold and peak_dbm >= cfg.threshold_dbm:
            self._above_threshold = True
            self.log.trigger(
                f"peak {peak_dbm:.1f} dBm @ {peak_hz / 1e6:.3f} MHz crossed threshold "
                f"{cfg.threshold_dbm:.1f} dBm -- starting capture",
                extra={
                    "fields": {
                        "action": "start",
                        "peak_dbm": peak_dbm,
                        "peak_hz": peak_hz,
                    }
                },
            )
        elif self._above_threshold and peak_dbm < cfg.threshold_dbm - cfg.hysteresis_db:
            self._above_threshold = False
            self.log.trigger(
                f"peak {peak_dbm:.1f} dBm dropped below threshold -- stopping capture",
                extra={
                    "fields": {
                        "action": "stop",
                        "peak_dbm": peak_dbm,
                        "peak_hz": peak_hz,
                    }
                },
            )

        return peak_dbm, peak_hz

    def run(self, duration_s: Optional[float] = None) -> None:
        self.connect()
        self.log.marker(
            "monitor_start", extra={"fields": {"run_dir": str(self.run_dir)}}
        )

        stop = False

        def _handle_sigint(signum, frame):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGINT, _handle_sigint)

        n = 0
        summary_path = self.run_dir / "summary.csv"
        try:
            with summary_path.open("w", encoding="utf-8") as summary:
                summary.write("timestamp,peak_dbm,peak_hz,recording_active\n")
                t0 = time.monotonic()
                while not stop:
                    if duration_s is not None and (time.monotonic() - t0) >= duration_s:
                        break

                    try:
                        freqs_hz, trace_dbm = self._fetch_trace()
                    except Exception:
                        self.log.exception("trace fetch failed")
                        time.sleep(self.monitor_cfg.poll_interval_s)
                        continue

                    peak_dbm, peak_hz = self._check_threshold(freqs_hz, trace_dbm)
                    now = datetime.now(timezone.utc)
                    _write_latest_trace(
                        self.run_dir, freqs_hz, trace_dbm, now.isoformat()
                    )
                    self.session.save_trace(freqs_hz, trace_dbm, now.isoformat())
                    summary.write(
                        f"{now.isoformat()},{peak_dbm:.3f},{peak_hz:.1f},{int(self.session.active)}\n"
                    )
                    summary.flush()

                    n += 1
                    if n % 50 == 0:
                        self.log.marker(
                            f"{n} sweeps recorded", extra={"fields": {"sweep_count": n}}
                        )

                    time.sleep(self.monitor_cfg.poll_interval_s)
        finally:
            if self.session.active:
                self.log.trigger(
                    "monitor stopping -- closing open capture",
                    extra={"fields": {"action": "stop"}},
                )
            self.log.marker("monitor_stop", extra={"fields": {"sweep_count": n}})
            self.close()

    def close(self) -> None:
        if self.inst is not None:
            try:
                self.inst.close()
            except Exception:
                pass
            self.inst = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--resource",
        default="TCPIP0::192.168.1.50::INSTR",
        help="VISA resource string for the SVA1015X (LAN, e.g. TCPIP0::<ip>::INSTR, or USB)",
    )
    p.add_argument(
        "--interval", type=float, default=1.0, help="seconds between trace fetches"
    )
    p.add_argument(
        "--threshold-dbm",
        type=float,
        default=-20.0,
        help="peak power that triggers a capture burst",
    )
    p.add_argument(
        "--hysteresis-db",
        type=float,
        default=3.0,
        help="peak must drop below (threshold - hysteresis) to stop the capture",
    )
    p.add_argument(
        "--run-root",
        default="runs",
        help="parent directory for the timestamped run folder",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="stop after this many seconds (default: run until Ctrl+C)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="console shows MARKER/TRIGGER events and above only",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    instrument_cfg = InstrumentConfig(resource=args.resource)
    monitor_cfg = MonitorConfig(
        poll_interval_s=args.interval,
        threshold_dbm=args.threshold_dbm,
        hysteresis_db=args.hysteresis_db,
        run_root=Path(args.run_root),
        console_level=MARKER if args.quiet else logging.INFO,
    )
    monitor = SVA1015XMonitor(instrument_cfg, monitor_cfg)
    try:
        monitor.run(duration_s=args.duration)
    except KeyboardInterrupt:
        monitor.close()


if __name__ == "__main__":
    main()
