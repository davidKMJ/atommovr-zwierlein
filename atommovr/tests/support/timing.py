from __future__ import annotations

import cProfile
import functools
import io
import os
import pstats
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar, cast

import numpy as np


P = ParamSpec("P")
R = TypeVar("R")


def _default_enabled() -> bool:
    """Return whether timing decorators should emit output by default."""
    raw: str = os.getenv("ATOMMOVR_TIMING", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _format_seconds(seconds: float) -> str:
    """Format a duration in seconds using readable engineering-style units."""
    if seconds >= 1.0:
        return f"{seconds:.6f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3f} ms"
    if seconds >= 1e-6:
        return f"{seconds * 1e6:.3f} us"
    return f"{seconds * 1e9:.3f} ns"


@dataclass
class TimingRecord:
    """
    Hold timing summary information for a single decorated function call.

    Parameters
    ----------
    name
        Function name used in reporting.
    samples
        Raw timing samples in seconds.
    metadata
        Optional user metadata.
    """

    name: str
    samples: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """
        Return the number of timing samples.

        Returns
        -------
        int
            Number of collected samples.
        """
        return len(self.samples)

    @property
    def mean(self) -> float:
        """
        Return the arithmetic mean runtime.

        Returns
        -------
        float
            Mean runtime in seconds.
        """
        return statistics.mean(self.samples)

    @property
    def median(self) -> float:
        """
        Return the median runtime.

        Returns
        -------
        float
            Median runtime in seconds.
        """
        return statistics.median(self.samples)

    @property
    def minimum(self) -> float:
        """
        Return the minimum runtime.

        Returns
        -------
        float
            Minimum runtime in seconds.
        """
        return min(self.samples)

    @property
    def maximum(self) -> float:
        """
        Return the maximum runtime.

        Returns
        -------
        float
            Maximum runtime in seconds.
        """
        return max(self.samples)

    @property
    def stdev(self) -> float:
        """
        Return the sample standard deviation when defined.

        Returns
        -------
        float
            Standard deviation in seconds. Returns 0.0 for one sample.
        """
        if self.count < 2:
            return 0.0
        return statistics.stdev(self.samples)

    def as_dict(self) -> dict[str, float | int]:
        """
        Convert the timing summary to a plain dictionary.

        Returns
        -------
        dict[str, float | int]
            Summary statistics.
        """
        return {
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "min": self.minimum,
            "max": self.maximum,
            "stdev": self.stdev,
        }


def timed(
    *,
    label: str | None = None,
    enabled: bool | None = None,
    logger: Callable[[str], None] = print,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorate a function with cheap wall-clock timing.

    Why this exists
    ---------------
    This is the lowest-friction timing tool for day-to-day development. Use it
    when you want to know roughly how long a single call took, without the
    measurement overhead of a profiler.

    Parameters
    ----------
    label
        Optional display label. Defaults to the function's qualified name.
    enabled
        Whether the decorator should emit timing output. If omitted, the
        ``ATOMMOVR_TIMING`` environment variable is used.
    logger
        Callable used to emit the formatted timing message.

    Returns
    -------
    Callable
        Decorated function that behaves identically but prints elapsed wall time.

    Notes
    -----
    This decorator stores the most recent runtime on
    ``wrapper._last_timing_record`` for later inspection.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        timing_enabled: bool = _default_enabled() if enabled is None else enabled
        timing_label: str = func.__qualname__ if label is None else label

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not timing_enabled:
                return func(*args, **kwargs)

            start: float = time.perf_counter()
            result: R = func(*args, **kwargs)
            elapsed: float = time.perf_counter() - start

            record: TimingRecord = TimingRecord(
                name=timing_label,
                samples=[elapsed],
            )
            setattr(wrapper, "_last_timing_record", record)
            logger(f"[timed] {timing_label}: {_format_seconds(elapsed)}")
            return result

        return cast(Callable[P, R], wrapper)

    return decorator


def benchmarked(
    *,
    repeats: int = 10,
    warmups: int = 2,
    label: str | None = None,
    enabled: bool | None = None,
    logger: Callable[[str], None] = print,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorate a function with repeated wall-clock benchmarking.

    Why this exists
    ---------------
    Single timings can be noisy, especially for small or medium-sized functions.
    This decorator runs warmups, then multiple measured executions, and reports
    stable summary statistics.

    Parameters
    ----------
    repeats
        Number of measured executions.
    warmups
        Number of warmup executions before measurement.
    label
        Optional display label.
    enabled
        Whether the decorator should emit timing output.
    logger
        Callable used to emit the formatted timing message.

    Returns
    -------
    Callable
        Decorated function. The wrapped function is still called only once for
        its return value; the benchmarking runs are additional executions.

    Notes
    -----
    Use this only for pure or effectively pure functions, or for functions where
    repeated execution on the same inputs is meaningful. Because this decorator
    executes the function multiple times, it is inappropriate for functions with
    important side effects unless you control those side effects externally.
    """

    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if warmups < 0:
        raise ValueError("warmups must be nonnegative.")

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        timing_enabled: bool = _default_enabled() if enabled is None else enabled
        timing_label: str = func.__qualname__ if label is None else label

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not timing_enabled:
                return func(*args, **kwargs)

            result: R = func(*args, **kwargs)

            for _ in range(warmups):
                func(*args, **kwargs)

            samples: list[float] = []
            for _ in range(repeats):
                start: float = time.perf_counter()
                func(*args, **kwargs)
                samples.append(time.perf_counter() - start)

            record: TimingRecord = TimingRecord(name=timing_label, samples=samples)
            setattr(wrapper, "_last_timing_record", record)

            logger(
                "[benchmarked] "
                f"{timing_label}: mean={_format_seconds(record.mean)}, "
                f"median={_format_seconds(record.median)}, "
                f"min={_format_seconds(record.minimum)}, "
                f"stdev={_format_seconds(record.stdev)}, "
                f"n={record.count}"
            )
            return result

        return cast(Callable[P, R], wrapper)

    return decorator


def profiled(
    *,
    sort_by: str = "cumtime",
    lines: int = 30,
    dump_path: str | Path | None = None,
    enabled: bool | None = None,
    logger: Callable[[str], None] = print,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorate a function with `cProfile`.

    Why this exists
    ---------------
    Use this on high-level algorithm entry points when you want to know which
    *functions* dominate runtime. It is not the right tool for subtle line-level
    comparisons, but it is excellent for finding the next big hotspot.

    Parameters
    ----------
    sort_by
        Sort key passed to ``pstats.Stats.sort_stats``.
    lines
        Number of lines of profile output to print.
    dump_path
        Optional path to dump the raw profiler stats.
    enabled
        Whether profiling output should be generated.
    logger
        Callable used to emit the formatted profile report.

    Returns
    -------
    Callable
        Decorated function that runs under `cProfile` and returns the original
        function result.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        profile_enabled: bool = _default_enabled() if enabled is None else enabled

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not profile_enabled:
                return func(*args, **kwargs)

            profiler: cProfile.Profile = cProfile.Profile()
            profiler.enable()
            result: R = func(*args, **kwargs)
            profiler.disable()

            if dump_path is not None:
                profiler.dump_stats(str(dump_path))

            buffer: io.StringIO = io.StringIO()
            stats: pstats.Stats = pstats.Stats(profiler, stream=buffer).sort_stats(sort_by)
            stats.print_stats(lines)
            setattr(wrapper, "_last_cprofile_text", buffer.getvalue())

            logger(
                f"[profiled] {func.__qualname__} "
                f"(sorted by {sort_by}, top {lines} lines)\n{buffer.getvalue()}"
            )
            return result

        return cast(Callable[P, R], wrapper)

    return decorator


def line_profiled(
    *,
    enabled: bool | None = None,
    logger: Callable[[str], None] = print,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorate a function with `line_profiler` when available.

    Why this exists
    ---------------
    This is the tool to use once you already know the hot function and want to
    understand which lines inside it are expensive. It requires the optional
    ``line_profiler`` dependency.

    Parameters
    ----------
    enabled
        Whether line-level profiling should run.
    logger
        Callable used to emit the formatted line-profile report.

    Returns
    -------
    Callable
        Decorated function. If `line_profiler` is not installed, the wrapper
        raises an informative ImportError when used with profiling enabled.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        profile_enabled: bool = _default_enabled() if enabled is None else enabled

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not profile_enabled:
                return func(*args, **kwargs)

            try:
                from line_profiler import LineProfiler
            except ImportError as exc:
                raise ImportError(
                    "line_profiled requires the optional 'line_profiler' package. "
                    "Install it with `python -m pip install line_profiler`."
                ) from exc

            profiler: Any = LineProfiler()
            profiler.add_function(func)
            result: R = profiler.runcall(func, *args, **kwargs)

            buffer: io.StringIO = io.StringIO()
            profiler.print_stats(stream=buffer)
            setattr(wrapper, "_last_line_profile_text", buffer.getvalue())

            logger(f"[line_profiled] {func.__qualname__}\n{buffer.getvalue()}")
            return result

        return cast(Callable[P, R], wrapper)

    return decorator


def estimate_runtime(
    func: Callable[P, R],
    *args: P.args,
    repeats: int = 10,
    warmups: int = 2,
    **kwargs: P.kwargs,
) -> TimingRecord:
    """
    Estimate function runtime without decorating the function.

    Parameters
    ----------
    func
        Function to benchmark.
    *args
        Positional arguments forwarded to the function.
    repeats
        Number of measured executions.
    warmups
        Number of warmup executions.
    **kwargs
        Keyword arguments forwarded to the function.

    Returns
    -------
    TimingRecord
        Timing summary record.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if warmups < 0:
        raise ValueError("warmups must be nonnegative.")

    for _ in range(warmups):
        func(*args, **kwargs)

    samples: list[float] = []
    for _ in range(repeats):
        start: float = time.perf_counter()
        func(*args, **kwargs)
        samples.append(time.perf_counter() - start)

    return TimingRecord(name=func.__qualname__, samples=samples)


def estimate_scaling(
    func_from_size: Callable[[int], Callable[[], Any]],
    sizes: list[int],
    *,
    repeats: int = 5,
    warmups: int = 1,
) -> dict[str, np.ndarray]:
    """
    Estimate empirical runtime scaling with problem size.

    Why this exists
    ---------------
    This helper is for questions like "does this look linear, quadratic, or
    worse?" You provide a builder that maps a system size to a zero-argument
    callable, so setup cost stays outside the timed region.

    Parameters
    ----------
    func_from_size
        Callable that accepts an integer size and returns a zero-argument
        function to benchmark at that size.
    sizes
        Problem sizes to measure.
    repeats
        Number of measured executions per size.
    warmups
        Number of warmup executions per size.

    Returns
    -------
    dict[str, np.ndarray]
        Arrays containing sizes, means, medians, minima, and standard
        deviations.

    Notes
    -----
    The returned data are suitable for plotting or for rough log-log slope
    estimation.
    """
    size_arr: list[int] = []
    mean_arr: list[float] = []
    median_arr: list[float] = []
    min_arr: list[float] = []
    stdev_arr: list[float] = []

    for size in sizes:
        timed_callable: Callable[[], Any] = func_from_size(size)
        record: TimingRecord = estimate_runtime(
            timed_callable,
            repeats=repeats,
            warmups=warmups,
        )
        size_arr.append(size)
        mean_arr.append(record.mean)
        median_arr.append(record.median)
        min_arr.append(record.minimum)
        stdev_arr.append(record.stdev)

    return {
        "sizes": np.asarray(size_arr, dtype=np.int64),
        "mean": np.asarray(mean_arr, dtype=np.float64),
        "median": np.asarray(median_arr, dtype=np.float64),
        "min": np.asarray(min_arr, dtype=np.float64),
        "stdev": np.asarray(stdev_arr, dtype=np.float64),
    }


def estimate_loglog_slope(
    sizes: np.ndarray,
    runtimes: np.ndarray,
) -> float:
    """
    Estimate the empirical scaling exponent from log-log data.

    Parameters
    ----------
    sizes
        Positive problem sizes.
    runtimes
        Positive measured runtimes corresponding to `sizes`.

    Returns
    -------
    float
        Least-squares slope of ``log(runtime)`` versus ``log(size)``.
    """
    if sizes.ndim != 1 or runtimes.ndim != 1:
        raise ValueError("sizes and runtimes must both be 1D.")
    if sizes.shape[0] != runtimes.shape[0]:
        raise ValueError("sizes and runtimes must have the same length.")
    if np.any(sizes <= 0) or np.any(runtimes <= 0):
        raise ValueError("sizes and runtimes must be strictly positive.")

    coeffs: np.ndarray = np.polyfit(np.log(sizes), np.log(runtimes), deg=1)
    return float(coeffs[0])


def compare_implementations(
    implementations: dict[str, Callable[[], Any]],
    *,
    repeats: int = 10,
    warmups: int = 2,
) -> dict[str, TimingRecord]:
    """
    Compare multiple zero-argument implementations on the same workload.

    Parameters
    ----------
    implementations
        Mapping from implementation name to zero-argument callable.
    repeats
        Number of measured executions per implementation.
    warmups
        Number of warmup executions per implementation.

    Returns
    -------
    dict[str, TimingRecord]
        Timing summaries keyed by implementation name.
    """
    results: dict[str, TimingRecord] = {}
    for name, impl in implementations.items():
        record: TimingRecord = estimate_runtime(
            impl,
            repeats=repeats,
            warmups=warmups,
        )
        record.name = name
        results[name] = record
    return results