"""Raspberry Pi system-state monitoring for the benchmark harness (R7).

This harness is the highest-integrity-risk component in the project: a subtly
wrong harness emits plausible numbers that are wrong, no reviewer catches it,
and the numbers ARE the paper. Hence the bit map is unit-tested rather than
trusted.

`vcgencmd get_throttled` returns a bit field. The four low bits are live state,
and bits 16-19 are the same four conditions latched since boot:

    bit  0 / 16   under-voltage detected / has occurred
    bit  1 / 17   ARM frequency capped   / has occurred
    bit  2 / 18   currently throttled    / has occurred
    bit  3 / 19   soft temperature limit active / has occurred

Why the latched bits matter as much as the live ones: a marginal 5 V supply or a
sagging BEC silently caps the clock. A benchmark that was throttled earlier in
the run is invalid even if `get_throttled` reads clean at the moment you look, so
every run must be checked before AND after and discarded if any latch is set.

Correction to the plan of record. `docs/plans/2026-08-03-sar-pi4-benchmark-plan.md`
R7 Step 1 asserts that 0x50005 sets soft_temp_limit_occurred. It does not:
0x50005 is bits 0, 2, 16 and 18, and soft-temp-limit-occurred is bit 19.
Implementing the plan's expectation would have mislabelled throttle events in
every reported run.
"""

from __future__ import annotations

import re

# bit position -> flag name. Tested for completeness and uniqueness.
THROTTLE_BITS: dict[int, str] = {
    0: "under_voltage_now",
    1: "arm_freq_capped_now",
    2: "throttled_now",
    3: "soft_temp_limit_now",
    16: "under_voltage_occurred",
    17: "arm_freq_capped_occurred",
    18: "throttled_occurred",
    19: "soft_temp_limit_occurred",
}

_OCCURRED_BITS = (16, 17, 18, 19)
# Capture whatever token follows the key, then validate it as hex separately.
# A stricter pattern would fail to match `throttled=garbage` at all and report
# "no throttled= field", which is misleading -- the field is present, its value
# is malformed, and those are different faults to diagnose on a live Pi.
_THROTTLED_RE = re.compile(r"throttled=(\S+)")


def parse_throttled(output: str) -> dict[str, bool]:
    """Decode `vcgencmd get_throttled` output into the eight documented flags.

    Raises rather than defaulting to zero on unparseable input. `vcgencmd` prints
    things like "VCHI initialization failed" when it cannot reach the firmware,
    and silently reading that as 0x0 would report a perfectly healthy Pi for a
    run that was never measured.

    Undocumented bits are ignored, not an error: firmware may add bits, and an
    unknown bit must not corrupt the eight flags that are documented.
    """
    text = (output or "").strip()
    m = _THROTTLED_RE.search(text)
    if not m:
        raise ValueError(
            f"no 'throttled=' field in vcgencmd output {text!r}; refusing to assume 0x0"
        )

    raw = m.group(1)
    try:
        value = int(raw, 16)
    except ValueError as exc:
        raise ValueError(f"throttled value {raw!r} is not hex") from exc

    return {name: bool(value & (1 << bit)) for bit, name in THROTTLE_BITS.items()}


def is_clean(flags: dict[str, bool]) -> bool:
    """True when no throttle condition is live or latched.

    The plan's gate: `get_throttled` must read 0x0 BEFORE a run starts. If it
    does not, fix the power supply -- every number taken under a capped clock is
    wrong.
    """
    return not any(flags.values())


def has_occurred_since_boot(flags: dict[str, bool]) -> bool:
    """True if any latched bit is set, ignoring live state.

    Use this to decide whether to DISCARD a completed run. A run can finish with
    clean live bits and still have been throttled partway through.
    """
    return any(flags[THROTTLE_BITS[b]] for b in _OCCURRED_BITS)


def describe(flags: dict[str, bool]) -> str:
    """Human-readable summary for logs and run manifests."""
    live = [n for n in THROTTLE_BITS.values() if n.endswith("_now") and flags[n]]
    latched = [n for n in THROTTLE_BITS.values() if n.endswith("_occurred") and flags[n]]
    if not live and not latched:
        return "clean (throttled=0x0)"
    parts = []
    if live:
        parts.append("live: " + ", ".join(live))
    if latched:
        parts.append("since boot: " + ", ".join(latched))
    return "; ".join(parts)
