"""Sequence-level bootstrap for the cross-domain evaluation.

Why this module exists rather than a call to ultralytics' own metrics: the
resampling unit has to be the SEQUENCE, and mAP is not a mean over sequences,
so it cannot be produced by averaging per-sequence numbers.

WiSARD frames are consecutive video -- in the sample sequence, 251 of 264 frames
carry the same four people. Resampling tiles or boxes would treat near-duplicate
frames as independent observations and return a confidence interval several
times too narrow. The project protocol is bootstrap 10,000 resamples, percentile
2.5/97.5, resampling over images (here sequences), never over detections.

The implementation precomputes the expensive part once. Detection-to-ground-truth
matching is independent per tile, so each detection can be flagged TP or FP a
single time; a bootstrap iteration then only has to pool the flags of the
sampled sequences, sort by confidence, and sweep a precision-recall curve. That
turns 10,000 mAP computations into 10,000 cheap sweeps.
"""

from __future__ import annotations

Box = tuple[float, float, float, float]  # x0, y0, x1, y1
Pred = tuple[float, Box]  # confidence, box
Flag = tuple[float, bool]  # confidence, is_true_positive


def _iou(a: Box, b: Box) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def match_detections(preds: list[Pred], gts: list[Box], iou_thr: float = 0.5) -> tuple[list[Flag], int]:
    """Greedily match one tile's detections to its ground truth.

    Returns the detections flagged TP/FP, ordered by descending confidence, and
    the tile's ground-truth count. Each ground-truth box can be consumed only
    once, so a duplicate detection of the same object is a false positive --
    without that, a detector that fires twice on every person would look better
    than one that fires once.
    """
    ordered = sorted(preds, key=lambda p: -p[0])
    taken = [False] * len(gts)
    flags: list[Flag] = []
    for conf, box in ordered:
        best_i, best_iou = -1, iou_thr
        for i, g in enumerate(gts):
            if taken[i]:
                continue
            v = _iou(box, g)
            if v >= best_iou:
                best_i, best_iou = i, v
        if best_i >= 0:
            taken[best_i] = True
            flags.append((conf, True))
        else:
            flags.append((conf, False))
    return flags, len(gts)


def average_precision(flags: list[Flag], n_gt: int) -> float:
    """AP from pre-flagged detections, all-point interpolation.

    `n_gt` is the total ground-truth count over the same set of tiles the flags
    came from; it is what makes recall reachable, so it must be carried
    alongside the flags through every resample.
    """
    if n_gt <= 0 or not flags:
        return 0.0

    ordered = sorted(flags, key=lambda f: -f[0])
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    # All-point interpolation: accumulate precision * delta-recall, taking the
    # running max of precision to the right, which the reverse pass below
    # would need. Done forward with the standard monotone trick instead:
    points: list[tuple[float, float]] = []
    for _, is_tp in ordered:
        if is_tp:
            tp += 1
        else:
            fp += 1
        points.append((tp / n_gt, tp / (tp + fp)))

    # Make precision monotonically non-increasing from the right.
    best = 0.0
    for i in range(len(points) - 1, -1, -1):
        r, p = points[i]
        best = max(best, p)
        points[i] = (r, best)

    for r, p in points:
        if r > prev_recall:
            ap += (r - prev_recall) * p
            prev_recall = r
    return ap


def recall_at_fp_budget(
    flags: list[Flag], n_gt: int, n_frames: int, fp_per_frame: float
) -> tuple[float, float, float]:
    """Highest recall achievable without exceeding a false-positive budget.

    SAR is recall-dominant with a human reviewing every hit, so the operating
    point is NOT argmax-F1: it is the lowest confidence threshold whose
    false-positive rate still fits the operator's attention budget.

    Returns (recall, threshold, achieved_fp_per_frame).
    """
    if n_gt <= 0 or n_frames <= 0:
        return 0.0, 1.0, 0.0

    max_fp = fp_per_frame * n_frames
    ordered = sorted(flags, key=lambda f: -f[0])
    tp = fp = 0
    best = (0.0, 1.0, 0.0)
    for conf, is_tp in ordered:
        if is_tp:
            tp += 1
        else:
            fp += 1
        if fp > max_fp:
            break
        best = (tp / n_gt, conf, fp / n_frames)
    return best


DEFAULT_BINS = 2000


def sequence_histogram(flags: list[Flag], n_bins: int = DEFAULT_BINS):
    """Bin one sequence's detections into TP and FP counts per confidence bin.

    This is what makes 10,000 resamples tractable. The exact `average_precision`
    re-sorts every detection on every call; with millions of detections that is
    O(n log n) per resample and does not finish. Histograms are ADDITIVE, so a
    resample becomes a vector add over 38 arrays followed by one cumulative sum.

    Bin index rises with confidence, so a reverse cumulative sum walks the
    threshold from high confidence down, which is the order AP needs.
    """
    import numpy as np

    tp = np.zeros(n_bins, dtype=np.int64)
    fp = np.zeros(n_bins, dtype=np.int64)
    for conf, is_tp in flags:
        b = min(int(conf * n_bins), n_bins - 1)
        if is_tp:
            tp[b] += 1
        else:
            fp[b] += 1
    return tp, fp


def ap_from_hist(tp_hist, fp_hist, n_gt: int) -> float:
    """AP from binned counts, all-point interpolation.

    Within-bin ordering is lost, so this is an approximation to
    `average_precision`. At the default 2000 bins the agreement is within a few
    thousandths, which is far inside the width of any bootstrap interval. The
    reported POINT estimate still comes from the exact function; only the CI
    distribution uses this path.
    """
    import numpy as np

    if n_gt <= 0:
        return 0.0
    tp_c = np.cumsum(tp_hist[::-1])
    fp_c = np.cumsum(fp_hist[::-1])
    total = tp_c + fp_c
    keep = total > 0
    if not keep.any():
        return 0.0

    recall = tp_c[keep] / n_gt
    precision = tp_c[keep] / total[keep]

    # Monotone non-increasing precision from the right.
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    prev_r = 0.0
    ap = 0.0
    for r, p in zip(recall, precision):
        if r > prev_r:
            ap += (r - prev_r) * p
            prev_r = r
    return float(ap)


def recall_at_fp_budget_hist(tp_hist, fp_hist, n_gt: int, n_frames: int, fp_per_frame: float) -> float:
    """Recall at the FP budget, from binned counts. See `recall_at_fp_budget`."""
    import numpy as np

    if n_gt <= 0 or n_frames <= 0:
        return 0.0
    tp_c = np.cumsum(tp_hist[::-1])
    fp_c = np.cumsum(fp_hist[::-1])
    ok = fp_c <= fp_per_frame * n_frames
    if not ok.any():
        return 0.0
    return float(tp_c[ok].max() / n_gt)


def percentile_ci(samples: list[float], alpha: float = 0.05) -> tuple[float, float]:
    """Percentile confidence interval. No parametric assumption."""
    if not samples:
        raise ValueError("cannot take a percentile CI of an empty sample")
    s = sorted(samples)
    n = len(s)

    def q(p: float) -> float:
        if n == 1:
            return s[0]
        pos = p * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return s[lo] * (1 - frac) + s[hi] * frac

    return q(alpha / 2), q(1 - alpha / 2)
