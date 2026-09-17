import pytest

from src.sar.bench.sysmon import (
    THROTTLE_BITS,
    has_occurred_since_boot,
    is_clean,
    parse_throttled,
)


# --- bit decoding ---------------------------------------------------------

def test_parses_undervoltage_and_throttle_bits():
    # 0x50005 = 0b101_0000_0000_0000_0101 -> bits 0, 2, 16, 18.
    #
    # NOTE: the plan of record (docs/plans/2026-08-03-sar-pi4-benchmark-plan.md,
    # R7 Step 1) asserts soft_temp_limit_occurred is True for this value. That is
    # wrong: soft-temp-limit-occurred is bit 19, and bit 19 is NOT set in
    # 0x50005. Encoding the plan's expectation would have baked a wrong bit map
    # into the harness and mislabelled every throttle event in the paper.
    r = parse_throttled("throttled=0x50005")
    assert r["under_voltage_now"] is True
    assert r["arm_freq_capped_now"] is False
    assert r["throttled_now"] is True
    assert r["soft_temp_limit_now"] is False
    assert r["under_voltage_occurred"] is True
    assert r["arm_freq_capped_occurred"] is False
    assert r["throttled_occurred"] is True
    assert r["soft_temp_limit_occurred"] is False


def test_clean_state_is_all_false():
    assert not any(parse_throttled("throttled=0x0").values())


def test_soft_temp_limit_bits_decode_at_3_and_19():
    r = parse_throttled("throttled=0x80008")  # bits 3 and 19
    assert r["soft_temp_limit_now"] is True
    assert r["soft_temp_limit_occurred"] is True
    assert r["under_voltage_now"] is False


def test_arm_freq_capped_bits_decode_at_1_and_17():
    r = parse_throttled("throttled=0x20002")  # bits 1 and 17
    assert r["arm_freq_capped_now"] is True
    assert r["arm_freq_capped_occurred"] is True
    assert r["throttled_now"] is False


def test_all_eight_flags_set():
    r = parse_throttled("throttled=0xF000F")
    assert all(r.values())
    assert len(r) == 8


def test_every_documented_bit_has_exactly_one_flag():
    # Guards against a duplicated or missing entry in the bit map.
    assert len(THROTTLE_BITS) == 8
    assert len(set(THROTTLE_BITS.values())) == 8


# --- input handling -------------------------------------------------------

def test_accepts_surrounding_whitespace_and_newline():
    # subprocess output arrives with a trailing newline.
    assert parse_throttled("  throttled=0x50005\n")["under_voltage_now"] is True


def test_accepts_uppercase_hex():
    assert parse_throttled("throttled=0X50005")["throttled_now"] is True


def test_rejects_output_without_the_throttled_key():
    # vcgencmd prints "VCHI initialization failed" and similar on a bad call;
    # parsing that as 0 would silently report a healthy Pi.
    with pytest.raises(ValueError, match="throttled="):
        parse_throttled("VCHI initialization failed")


def test_rejects_empty_output():
    with pytest.raises(ValueError, match="throttled="):
        parse_throttled("")


def test_rejects_non_hex_value():
    with pytest.raises(ValueError, match="hex"):
        parse_throttled("throttled=notanumber")


def test_unknown_high_bits_do_not_crash_and_are_ignored():
    # Firmware may add bits later; an unknown bit must not corrupt the eight
    # documented flags.
    r = parse_throttled("throttled=0x100000")  # bit 20, undocumented
    assert not any(r.values())


# --- health helpers -------------------------------------------------------

def test_is_clean_only_when_every_flag_is_false():
    assert is_clean(parse_throttled("throttled=0x0")) is True
    assert is_clean(parse_throttled("throttled=0x1")) is False


def test_has_occurred_since_boot_ignores_the_live_bits():
    # A run that was throttled earlier is invalid even if it is fine right now.
    r = parse_throttled("throttled=0x10000")  # under-voltage occurred, nothing live
    assert has_occurred_since_boot(r) is True
    assert r["under_voltage_now"] is False


def test_has_occurred_since_boot_false_when_only_live_bits_set():
    r = parse_throttled("throttled=0x1")
    assert has_occurred_since_boot(r) is False
