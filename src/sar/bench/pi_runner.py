"""Pi 4 benchmark harness: per-stage timing, tail latency, and system state (R7).

The measurement rules here are not stylistic. Each one exists because the
alternative produces a number that looks fine and is not comparable:

- **Per-stage timing.** NMS alone is routinely 20-40 percent of end-to-end at
  low resolution. Timing only the forward pass and calling it "inference" makes
  the result incomparable with every other paper.
- **Median, p95, p99 -- never the mean alone.** A mean of 90 ms with a p99 of
  400 ms is not a real-time system. Latency distributions are right-skewed, so
  the mean flatters and the tail is what breaks a deadline.
- **Warm-up discarded.** The first frames pay cold page cache, allocator growth
  and a clock ramp. Including them makes the benchmark unreproducible on a warm
  machine.
- **Throughput from the median, not the mean.** Same reason.

This module holds the pure, testable core: statistics, per-stage accounting and
the `vcgencmd` parsers. Device interaction lives in the runner script so the
maths can be tested on any machine, before the Pi is reachable.
"""

from __future__ import annotations

import re

# Full pipeline, in execution order. Reporting a subset is the failure mode
# described above, so the tuple is asserted in tests rather than assumed.
STAGES: tuple[str, ...] = (
    "capture",
    "decode",
    "letterbox",
    "forward",
    "nms",
    "postprocess",
)

DEFAULT_WARMUP = 100

_TEMP_RE = re.compile(r"temp=([0-9.]+)")
_CLOCK_RE = re.compile(r"frequency\(\d+\)=(\d+)")
_VOLTS_RE = re.compile(r"volt=([0-9.]+)")


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        raise ValueError("no samples")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = p * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def latency_summary(samples_ms: list[float], warmup: int = 0) -> dict:
    """Tail-aware latency summary in milliseconds.

    `warmup` frames are dropped from the FRONT before any statistic is computed.
    """
    if warmup:
        if warmup >= len(samples_ms):
            raise ValueError(
                f"warmup={warmup} discards all {len(samples_ms)} samples; "
                "collect more frames or lower the warm-up"
            )
        samples_ms = samples_ms[warmup:]
    if not samples_ms:
        raise ValueError("no samples to summarise")

    s = sorted(samples_ms)
    median = _percentile(s, 0.50)
    return {
        "n": len(s),
        "median_ms": round(median, 4),
        "p95_ms": round(_percentile(s, 0.95), 4),
        "p99_ms": round(_percentile(s, 0.99), 4),
        "min_ms": round(s[0], 4),
        "max_ms": round(s[-1], 4),
        # Deliberately derived from the median. A mean-derived FPS overstates
        # sustained throughput whenever the tail is heavy, which it always is.
        "fps_median": round(1000.0 / median, 4) if median > 0 else 0.0,
        # Reported for completeness only; never quote it as the headline.
        "mean_ms": round(sum(s) / len(s), 4),
    }


class StageTimer:
    """Accumulates per-stage durations for one benchmark run."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = {s: [] for s in STAGES}

    def record(self, stage: str, ms: float) -> None:
        if stage not in self._samples:
            raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
        self._samples[stage].append(ms)

    def samples(self, stage: str) -> list[float]:
        if stage not in self._samples:
            raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
        return self._samples[stage]

    def summary(self, warmup: int = DEFAULT_WARMUP) -> dict[str, dict]:
        """Per-stage latency summaries, skipping stages with no samples."""
        return {
            stage: latency_summary(vals, warmup=warmup)
            for stage, vals in self._samples.items()
            if vals
        }

    def stage_fractions(self, warmup: int = DEFAULT_WARMUP) -> dict[str, float]:
        """Share of end-to-end median time spent in each stage.

        This is the number that shows whether NMS is eating the budget, which is
        invisible if only total latency is reported.
        """
        med = {
            stage: latency_summary(vals, warmup=warmup)["median_ms"]
            for stage, vals in self._samples.items()
            if vals
        }
        total = sum(med.values())
        if total <= 0:
            return {k: 0.0 for k in med}
        return {k: v / total for k, v in med.items()}


def parse_measure_temp(output: str) -> float:
    """Degrees C from `vcgencmd measure_temp`."""
    m = _TEMP_RE.search(output or "")
    if not m:
        raise ValueError(f"no 'temp=' field in vcgencmd output {output!r}")
    return float(m.group(1))


def parse_measure_clock(output: str) -> int:
    """Hz from `vcgencmd measure_clock arm`."""
    m = _CLOCK_RE.search(output or "")
    if not m:
        raise ValueError(f"no 'frequency(N)=' field in vcgencmd output {output!r}")
    return int(m.group(1))


def parse_measure_volts(output: str) -> float:
    """Volts from `vcgencmd measure_volts core`."""
    m = _VOLTS_RE.search(output or "")
    if not m:
        raise ValueError(f"no 'volt=' field in vcgencmd output {output!r}")
    return float(m.group(1))


def thread_configs() -> tuple[int, ...]:
    """Thread counts to sweep.

    Four threads maximises isolated FPS and starves every co-tenant on a 4-core
    Pi. Both ends of that trade have to be measured, because isolated FPS is a
    marketing number and contended FPS is the operational one.
    """
    return (1, 2, 3, 4)
