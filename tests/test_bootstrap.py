import pytest

from src.sar.eval.bootstrap import (
    average_precision,
    match_detections,
    percentile_ci,
    recall_at_fp_budget,
)


# --- matching -------------------------------------------------------------

def test_one_detection_over_one_gt_is_a_true_positive():
    preds = [(0.9, (10, 10, 30, 30))]
    gts = [(10, 10, 30, 30)]
    flags, n_gt = match_detections(preds, gts, iou_thr=0.5)
    assert flags == [(0.9, True)]
    assert n_gt == 1


def test_detection_far_from_gt_is_a_false_positive():
    preds = [(0.9, (100, 100, 120, 120))]
    gts = [(10, 10, 30, 30)]
    flags, _ = match_detections(preds, gts, iou_thr=0.5)
    assert flags == [(0.9, False)]


def test_second_detection_on_the_same_gt_is_a_false_positive():
    # Duplicate detections must not both count; only the highest-confidence
    # match consumes the ground truth.
    preds = [(0.9, (10, 10, 30, 30)), (0.8, (11, 11, 31, 31))]
    gts = [(10, 10, 30, 30)]
    flags, _ = match_detections(preds, gts, iou_thr=0.5)
    assert flags[0] == (0.9, True)
    assert flags[1][1] is False


def test_matching_is_greedy_by_confidence_not_input_order():
    preds = [(0.4, (10, 10, 30, 30)), (0.95, (10, 10, 30, 30))]
    gts = [(10, 10, 30, 30)]
    flags, _ = match_detections(preds, gts, iou_thr=0.5)
    # Returned highest-confidence first, and that one owns the match.
    assert flags[0] == (0.95, True)
    assert flags[1][1] is False


def test_empty_frame_with_detections_yields_only_false_positives():
    preds = [(0.7, (5, 5, 15, 15)), (0.6, (50, 50, 60, 60))]
    flags, n_gt = match_detections(preds, [], iou_thr=0.5)
    assert n_gt == 0
    assert [f[1] for f in flags] == [False, False]


def test_frame_with_gt_and_no_detections_still_reports_its_gt():
    flags, n_gt = match_detections([], [(10, 10, 30, 30)], iou_thr=0.5)
    assert flags == []
    assert n_gt == 1


# --- average precision ----------------------------------------------------

def test_perfect_detector_scores_ap_one():
    flags = [(0.9, True), (0.8, True)]
    assert average_precision(flags, n_gt=2) == pytest.approx(1.0)


def test_detector_that_finds_nothing_scores_ap_zero():
    assert average_precision([], n_gt=5) == 0.0


def test_ap_is_zero_when_there_is_no_ground_truth():
    # No positives means recall is undefined; report 0 rather than dividing by
    # zero or silently returning 1.
    assert average_precision([(0.9, False)], n_gt=0) == 0.0


def test_half_the_ground_truth_found_cleanly_caps_ap_near_one_half():
    flags = [(0.9, True)]
    assert average_precision(flags, n_gt=2) == pytest.approx(0.5)


def test_false_positive_ranked_above_a_true_positive_lowers_ap():
    good = average_precision([(0.9, True), (0.8, False)], n_gt=1)
    bad = average_precision([(0.9, False), (0.8, True)], n_gt=1)
    assert bad < good


# --- FP budget ------------------------------------------------------------

def test_recall_at_fp_budget_picks_the_recall_maximising_threshold():
    # 2 GT, 2 frames. Budget 0.5 FP/frame = 1 FP allowed.
    flags = [(0.9, True), (0.7, False), (0.6, True), (0.5, False)]
    rec, thr, fpf = recall_at_fp_budget(flags, n_gt=2, n_frames=2, fp_per_frame=0.5)
    # Going down to 0.6 costs 1 FP and buys the second TP -> recall 1.0.
    assert rec == pytest.approx(1.0)
    assert thr == pytest.approx(0.6)
    assert fpf == pytest.approx(0.5)


def test_recall_at_fp_budget_respects_a_zero_budget():
    flags = [(0.9, True), (0.8, False), (0.7, True)]
    rec, _, fpf = recall_at_fp_budget(flags, n_gt=2, n_frames=2, fp_per_frame=0.0)
    assert rec == pytest.approx(0.5)
    assert fpf == 0.0


def test_recall_at_fp_budget_with_no_ground_truth_is_zero():
    rec, _, _ = recall_at_fp_budget([(0.9, False)], n_gt=0, n_frames=1, fp_per_frame=1.0)
    assert rec == 0.0


# --- CI -------------------------------------------------------------------

def test_percentile_ci_brackets_the_middle_of_the_distribution():
    lo, hi = percentile_ci(list(range(101)), alpha=0.05)
    assert lo == pytest.approx(2.5, abs=1)
    assert hi == pytest.approx(97.5, abs=1)


def test_percentile_ci_of_a_constant_is_that_constant():
    lo, hi = percentile_ci([7.0] * 50, alpha=0.05)
    assert lo == hi == 7.0


def test_percentile_ci_rejects_an_empty_sample():
    with pytest.raises(ValueError, match="empty"):
        percentile_ci([], alpha=0.05)


# --- histogram fast path --------------------------------------------------

def test_histogram_ap_matches_the_exact_ap_closely():
    # The bootstrap CI uses a binned confidence grid so 10,000 resamples are
    # vector adds instead of 10,000 sorts. It must agree with the exact AP.
    import random

    from src.sar.eval.bootstrap import ap_from_hist, sequence_histogram

    rng = random.Random(0)
    flags = [(rng.random(), rng.random() < 0.4) for _ in range(5000)]
    n_gt = 2500
    exact = average_precision(flags, n_gt)
    tp, fp = sequence_histogram(flags, n_bins=2000)
    approx = ap_from_hist(tp, fp, n_gt)
    assert approx == pytest.approx(exact, abs=0.005)


def test_histograms_are_additive_across_sequences():
    # Pooling two sequences must equal histogramming their concatenation --
    # this is the property that makes resampling a vector add.
    from src.sar.eval.bootstrap import sequence_histogram

    a = [(0.9, True), (0.5, False)]
    b = [(0.8, False), (0.4, True)]
    ta, fa = sequence_histogram(a, n_bins=100)
    tb, fb = sequence_histogram(b, n_bins=100)
    tc, fc = sequence_histogram(a + b, n_bins=100)
    assert list(ta + tb) == list(tc)
    assert list(fa + fb) == list(fc)


def test_histogram_ap_is_zero_without_ground_truth():
    from src.sar.eval.bootstrap import ap_from_hist, sequence_histogram

    tp, fp = sequence_histogram([(0.9, False)], n_bins=100)
    assert ap_from_hist(tp, fp, 0) == 0.0


# --- recall at budget, histogram path -------------------------------------
# Every script-level bootstrap loop in this repo goes through this function,
# and until now none of them was covered by a test.

def test_histogram_recall_matches_the_exact_recall_closely():
    import random

    from src.sar.eval.bootstrap import recall_at_fp_budget_hist, sequence_histogram

    rng = random.Random(1)
    flags = [(rng.random(), rng.random() < 0.35) for _ in range(4000)]
    n_gt, n_frames, budget = 1500, 500, 0.5
    exact, _, _ = recall_at_fp_budget(flags, n_gt, n_frames, budget)
    tp, fp = sequence_histogram(flags, n_bins=2000)
    approx = recall_at_fp_budget_hist(tp, fp, n_gt, n_frames, budget)
    assert approx == pytest.approx(exact, abs=0.005)


def test_histogram_recall_is_zero_without_ground_truth_or_frames():
    from src.sar.eval.bootstrap import recall_at_fp_budget_hist, sequence_histogram

    tp, fp = sequence_histogram([(0.9, True)], n_bins=100)
    assert recall_at_fp_budget_hist(tp, fp, 0, 10, 0.1) == 0.0
    assert recall_at_fp_budget_hist(tp, fp, 5, 0, 0.1) == 0.0


def test_histogram_recall_returns_zero_when_the_budget_buys_nothing():
    # The cheapest detection is already a false positive, so no threshold
    # satisfies a zero-FP budget and the honest answer is no recall at all.
    from src.sar.eval.bootstrap import recall_at_fp_budget_hist, sequence_histogram

    tp, fp = sequence_histogram([(0.9, False), (0.8, True)], n_bins=100)
    assert recall_at_fp_budget_hist(tp, fp, 1, 1, 0.0) == 0.0


def test_histogram_recall_rises_with_a_looser_budget():
    from src.sar.eval.bootstrap import recall_at_fp_budget_hist, sequence_histogram

    flags = [(0.9, False), (0.8, True), (0.7, False), (0.6, True)]
    tp, fp = sequence_histogram(flags, n_bins=100)
    tight = recall_at_fp_budget_hist(tp, fp, 2, 4, 0.25)
    loose = recall_at_fp_budget_hist(tp, fp, 2, 4, 1.0)
    assert loose >= tight


# --- the resampling unit --------------------------------------------------

def test_resampling_a_sequence_index_moves_whole_sequences():
    # The paper's intervals resample SEQUENCES, never tiles, because WiSARD
    # frames are consecutive video. This pins the property the script-level
    # loops rely on: indexing the per-sequence histogram array with a bootstrap
    # index must carry each drawn sequence's counts intact, so that drawing the
    # same sequence twice doubles its contribution.
    import numpy as np

    from src.sar.eval.bootstrap import sequence_histogram

    a = sequence_histogram([(0.9, True), (0.4, False)], n_bins=64)
    b = sequence_histogram([(0.7, False)], n_bins=64)
    tp_by_seq = np.stack([a[0], b[0]])
    fp_by_seq = np.stack([a[1], b[1]])

    drawn_twice = tp_by_seq[[0, 0]].sum(axis=0)
    assert list(drawn_twice) == list(tp_by_seq[0] * 2)

    both = tp_by_seq[[0, 1]].sum(axis=0)
    assert list(both) == list(tp_by_seq[0] + tp_by_seq[1])
    assert int(fp_by_seq[[1, 1]].sum()) == 2
