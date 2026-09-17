import pytest

from src.sar.export.parity import compute_deltas


def _row(run, artifact_id, map50, map5095, ap_small):
    return {
        "run": run,
        "artifact_id": artifact_id,
        "mAP50": map50,
        "mAP50_95": map5095,
        "AP_small": ap_small,
        "size_bytes": 1000,
    }


def test_baseline_row_has_zero_delta():
    rows = [_row("r1", "pt_fp32", 0.70, 0.40, 0.20)]
    out = compute_deltas(rows)
    assert out[0]["delta_mAP50"] == 0.0
    assert out[0]["delta_mAP50_95"] == 0.0
    assert out[0]["delta_AP_small"] == 0.0


def test_delta_is_measured_against_the_same_runs_baseline():
    rows = [
        _row("r1", "pt_fp32", 0.70, 0.40, 0.20),
        _row("r2", "pt_fp32", 0.50, 0.30, 0.10),
        _row("r2", "tflite_int8", 0.45, 0.27, 0.05),
    ]
    out = {(r["run"], r["artifact_id"]): r for r in compute_deltas(rows)}
    # r2's int8 row must compare to r2's baseline (0.50), not r1's (0.70).
    assert out[("r2", "tflite_int8")]["delta_mAP50"] == pytest.approx(-0.05)


def test_delta_is_in_percentage_points_not_ratio():
    rows = [
        _row("r1", "pt_fp32", 0.70, 0.40, 0.20),
        _row("r1", "onnx_fp32", 0.68, 0.40, 0.20),
    ]
    out = {r["artifact_id"]: r for r in compute_deltas(rows)}
    assert out["onnx_fp32"]["delta_mAP50_pp"] == pytest.approx(-2.0)


def test_int8_drop_beyond_two_points_is_flagged_for_preprocessing_review():
    rows = [
        _row("r1", "pt_fp32", 0.70, 0.40, 0.20),
        _row("r1", "tflite_int8", 0.65, 0.36, 0.12),
    ]
    out = {r["artifact_id"]: r for r in compute_deltas(rows)}
    assert out["tflite_int8"]["review_preprocessing"] is True
    assert out["pt_fp32"]["review_preprocessing"] is False


def test_small_drop_is_not_flagged():
    rows = [
        _row("r1", "pt_fp32", 0.70, 0.40, 0.20),
        _row("r1", "tflite_int8", 0.695, 0.398, 0.199),
    ]
    out = {r["artifact_id"]: r for r in compute_deltas(rows)}
    assert out["tflite_int8"]["review_preprocessing"] is False


def test_a_gain_is_never_flagged_for_review():
    # Only losses matter here; an artifact scoring above its baseline is noise,
    # not a preprocessing bug, and must not raise a false alarm.
    rows = [
        _row("r1", "pt_fp32", 0.70, 0.40, 0.20),
        _row("r1", "onnx_fp32", 0.74, 0.43, 0.24),
    ]
    out = {r["artifact_id"]: r for r in compute_deltas(rows)}
    assert out["onnx_fp32"]["review_preprocessing"] is False


def test_missing_baseline_raises_rather_than_silently_reporting_zero():
    rows = [_row("r3", "onnx_fp32", 0.60, 0.35, 0.15)]
    with pytest.raises(ValueError, match="no pt_fp32 baseline"):
        compute_deltas(rows)
