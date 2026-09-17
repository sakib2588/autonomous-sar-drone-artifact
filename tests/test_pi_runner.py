import pytest

from src.sar.bench.pi_runner import (
    STAGES,
    StageTimer,
    latency_summary,
    parse_measure_clock,
    parse_measure_temp,
    parse_measure_volts,
    thread_configs,
)


# --- latency statistics ---------------------------------------------------

def test_summary_reports_median_p95_p99_not_mean_alone():
    # A mean of 90 ms with a p99 of 400 ms is not a real-time system. The plan
    # forbids reporting the mean as the headline number, so the summary must
    # carry the tail percentiles.
    s = latency_summary([10.0] * 99 + [1000.0])
    assert "median_ms" in s and "p95_ms" in s and "p99_ms" in s
    assert s["median_ms"] == pytest.approx(10.0)
    assert s["p99_ms"] > s["median_ms"]


def test_summary_percentiles_are_ordered():
    import random

    rng = random.Random(0)
    s = latency_summary([rng.expovariate(1 / 20) for _ in range(5000)])
    assert s["median_ms"] <= s["p95_ms"] <= s["p99_ms"] <= s["max_ms"]


def test_summary_reports_fps_from_the_median_not_the_mean():
    # Throughput quoted from a mean is inflated by the tail being symmetric,
    # which it never is for inference latency.
    s = latency_summary([20.0] * 50)
    assert s["fps_median"] == pytest.approx(50.0)


def test_summary_rejects_an_empty_sample():
    with pytest.raises(ValueError, match="no samples"):
        latency_summary([])


def test_warmup_frames_are_discarded():
    # First 100 frames are cold: page cache, allocator, and clock ramp. Including
    # them makes a benchmark unreproducible on a warm machine.
    slow_start = [500.0] * 100 + [10.0] * 100
    s = latency_summary(slow_start, warmup=100)
    assert s["median_ms"] == pytest.approx(10.0)
    assert s["n"] == 100


def test_warmup_larger_than_the_sample_raises():
    with pytest.raises(ValueError, match="warmup"):
        latency_summary([1.0] * 10, warmup=100)


# --- per-stage timing -----------------------------------------------------

def test_all_six_pipeline_stages_are_tracked():
    # Timing only the forward pass and calling it "inference" makes the number
    # incomparable to everyone else's. NMS alone is routinely 20-40 percent of
    # end-to-end at low resolution.
    assert STAGES == ("capture", "decode", "letterbox", "forward", "nms", "postprocess")


def test_stage_timer_accumulates_per_stage():
    t = StageTimer()
    t.record("forward", 10.0)
    t.record("forward", 20.0)
    t.record("nms", 5.0)
    assert t.samples("forward") == [10.0, 20.0]
    assert t.samples("nms") == [5.0]


def test_stage_timer_rejects_an_unknown_stage():
    # A typo would silently create a new bucket and vanish from the report.
    t = StageTimer()
    with pytest.raises(ValueError, match="unknown stage"):
        t.record("infrence", 1.0)


def test_stage_timer_summary_covers_every_recorded_stage():
    t = StageTimer()
    for s in STAGES:
        t.record(s, 5.0)
    out = t.summary(warmup=0)
    assert set(out) == set(STAGES)
    assert out["nms"]["median_ms"] == pytest.approx(5.0)


def test_stage_fraction_of_end_to_end_sums_to_one():
    t = StageTimer()
    t.record("forward", 60.0)
    t.record("nms", 40.0)
    frac = t.stage_fractions(warmup=0)
    assert frac["forward"] == pytest.approx(0.6)
    assert frac["nms"] == pytest.approx(0.4)
    assert sum(frac.values()) == pytest.approx(1.0)


# --- vcgencmd parsers -----------------------------------------------------

def test_parse_temp():
    assert parse_measure_temp("temp=47.2'C\n") == pytest.approx(47.2)


def test_parse_temp_rejects_garbage():
    with pytest.raises(ValueError, match="temp="):
        parse_measure_temp("VCHI initialization failed")


def test_parse_clock_returns_hz():
    assert parse_measure_clock("frequency(48)=1500000000\n") == 1_500_000_000


def test_parse_clock_rejects_garbage():
    with pytest.raises(ValueError, match="frequency"):
        parse_measure_clock("nope")


def test_parse_volts():
    assert parse_measure_volts("volt=0.8563V\n") == pytest.approx(0.8563)


def test_parse_volts_rejects_garbage():
    with pytest.raises(ValueError, match="volt="):
        parse_measure_volts("")


# --- sweep configuration --------------------------------------------------

def test_thread_configs_cover_one_to_four():
    # Four threads maximises isolated FPS and starves every co-tenant on a
    # 4-core Pi; both ends of that trade have to be measured.
    assert thread_configs() == (1, 2, 3, 4)
