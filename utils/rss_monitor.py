"""Peak RSS (Resident Set Size) monitor for cost-comparison instrumentation.

Used by `scripts/run_paper_sweep.py` and `scripts/run_dpl_convergence.py`
to record per-slot peak memory footprint into HDF5 so block-DDA_Py and
block-VIEM.jl can be compared on equal cost axes (CLAUDE.md §5, §6).

Implementation: a daemon thread polls `/proc/self/status:VmRSS` at a
configurable interval (default 0.2 s) and tracks the maximum. The main
thread calls `reset()` just before the measured region and `peak_bytes`
afterwards. RSS is read from /proc to avoid any extra dependency on
psutil.
"""
from __future__ import annotations

import threading
import time


def _read_rss_kb() -> int:
    """Return the current VmRSS of this process in KiB (Linux only).

    Returns 0 if /proc/self/status is not readable.
    """
    try:
        with open("/proc/self/status") as fp:
            for line in fp:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


class RSSMonitor:
    """Background-thread peak-RSS sampler.

    Typical use:
        mon = RSSMonitor(interval=0.2).start()
        ...
        mon.reset()                       # start of measured region
        # ... do work ...
        peak = mon.peak_bytes             # end of measured region
        ...
        mon.stop()                        # at program exit
    """

    def __init__(self, interval: float = 0.2):
        self.interval = float(interval)
        self._peak_kb = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _poll(self) -> None:
        while not self._stop.is_set():
            rss = _read_rss_kb()
            with self._lock:
                if rss > self._peak_kb:
                    self._peak_kb = rss
            self._stop.wait(self.interval)

    def start(self) -> "RSSMonitor":
        """Start the background sampler. Returns self for chaining."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll, name="RSSMonitor", daemon=True)
        self._thread.start()
        # Seed with an immediate sample so peak_bytes is never 0 after reset.
        with self._lock:
            self._peak_kb = _read_rss_kb()
        return self

    def reset(self) -> None:
        """Reset the peak to the current RSS. Call at the start of each slot."""
        rss = _read_rss_kb()
        with self._lock:
            self._peak_kb = rss

    @property
    def peak_bytes(self) -> int:
        """Peak VmRSS observed since the last reset(), in bytes."""
        with self._lock:
            return int(self._peak_kb) * 1024

    @property
    def peak_gb(self) -> float:
        return self.peak_bytes / 1024 ** 3

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
