"""Wall-clock timing, hardware capture and implementation-independent counters."""

from __future__ import annotations

import os
import platform
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator


class FitTimeout(BaseException):
    """Raised when a model exceeds its per-fold time budget.

    Deliberately a :class:`BaseException`, not an :class:`Exception`. The alarm
    fires wherever the model happens to be, which is usually deep inside a
    third-party library, and several of them swallow broad exceptions: sklearn's
    ``_fit_and_score`` catches ``Exception``, so a normal exception is absorbed
    into a "N fits failed" warning and the search simply carries on -- past the
    point where the itimer has already fired, leaving the rest of the fit
    genuinely unbounded. Inheriting from ``BaseException`` puts it in the same
    class as ``KeyboardInterrupt`` and lets it escape those handlers.

    Because of that, ``except Exception`` will not catch it: callers must name
    it explicitly.
    """


@contextmanager
def time_limit(seconds: float | None) -> Iterator[None]:
    """Abort the enclosed block once *seconds* have elapsed.

    Without this an unbounded baseline stalls the entire run: RIPPER and BRL are
    pure-Python learners whose cost grows steeply with row count, and a single
    large dataset can hold up every remaining one. Timing out a fold costs one
    row of results; not timing out costs the run.

    Uses ``SIGALRM``, so it is a no-op off the main thread or on platforms
    without it, rather than silently pretending to enforce a limit. The signal
    only interrupts at the interpreter's next check, so a long call inside a C
    extension is cut short when it returns, not mid-instruction.
    """
    unavailable = (
        not seconds
        or seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    )
    if unavailable:
        yield
        return

    def _raise(signum: int, frame: Any) -> None:
        raise FitTimeout(f"exceeded the {seconds:g}s fit budget")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


@dataclass
class Counters:
    """Implementation-independent work counters.

    These are deliberately hardware-agnostic so that search strategies remain
    comparable across machines and across languages.
    """

    trees_fitted: int = 0
    rules_generated: int = 0
    rules_after_filter: int = 0
    nodes_expanded: int = 0
    candidate_sets_evaluated: int = 0
    metric_evaluations: int = 0

    def merge(self, other: "Counters") -> "Counters":
        return Counters(
            trees_fitted=self.trees_fitted + other.trees_fitted,
            rules_generated=self.rules_generated + other.rules_generated,
            rules_after_filter=self.rules_after_filter + other.rules_after_filter,
            nodes_expanded=self.nodes_expanded + other.nodes_expanded,
            candidate_sets_evaluated=self.candidate_sets_evaluated
            + other.candidate_sets_evaluated,
            metric_evaluations=self.metric_evaluations + other.metric_evaluations,
        )

    def as_dict(self, prefix: str = "") -> dict[str, int]:
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


@dataclass
class Stopwatch:
    seconds: float = 0.0


@contextmanager
def timed() -> Iterator[Stopwatch]:
    watch = Stopwatch()
    start = time.perf_counter()
    try:
        yield watch
    finally:
        watch.seconds = time.perf_counter() - start


def _total_ram_gb() -> float | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        n_pages = os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None
    return round(page_size * n_pages / 1024**3, 2)


def hardware_info() -> dict[str, Any]:
    """Capture the hardware/OS facts a paper must disclose alongside timings."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "logical_cpus": os.cpu_count(),
        "total_ram_gb": _total_ram_gb(),
    }


@dataclass
class TimingRecord:
    stage: str
    seconds: float
    counters: Counters = field(default_factory=Counters)

    def as_row(self) -> dict[str, Any]:
        return {"stage": self.stage, "seconds": self.seconds, **self.counters.as_dict()}
