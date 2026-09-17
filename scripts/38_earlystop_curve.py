"""Cross-domain accuracy against training epoch, from checkpoints already on disk.

THE QUESTION
------------
In-domain validation rises monotonically for the whole 120 epochs. Cross-domain
accuracy does not have to follow it, and a two-point check said it does not:
the epoch-50 checkpoint scores AP@0.5 0.3055 on WiSARD against 0.2570 for the
final model, a 19 percent relative LOSS from the training that improved the
in-domain number.

If that holds across the whole curve it means the source domain is being overfit,
early-stopping on it is free cross-domain accuracy, and every in-domain-only
report in this field is tuning toward a worse deployed model. That is worth more
than another ablation, and it costs no training at all -- `save_period=10` already
wrote the checkpoints.

WHAT IT DOES
------------
For each saved `epoch<N>.pt` (plus `last.pt` as the final epoch), stage it under a
throwaway run name so `29_dump_detections.py` picks it up unmodified, score the
full 37,058-tile WiSARD set, then compute AP@0.5 and the recall ceiling with the
same code the paper's numbers come from.

RESUMABLE. Rows already in the output CSV are skipped, so a crash costs one
checkpoint rather than the run. This matters: the machine died three times on
2026-08-25.

WHY STAGE COPIES INSTEAD OF EDITING 29
--------------------------------------
`29_dump_detections.py` hardcodes `checkpoints/<run>/weights/best.pt` and names its
cache `dets_<run>_sz<imgsz>.npz`. Copying each checkpoint to a distinct run name
gets a distinct cache for free and leaves the published caches untouched. Run
names stay flat -- a slash would end up inside the cache filename.

Nothing here deletes anything. The staged copies are left on disk (about 130 MB
per seed); remove them by hand once the curve is confirmed.

USAGE
  .venv/bin/python scripts/38_earlystop_curve.py --run y11n_tiled640_s42
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load_compare_helpers():
    """Reuse 37's AP/recall code so this curve and the control use one metric."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("cmp37", ROOT / "scripts" / "37_compare_negfrac.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def epochs_available(run: str) -> list[tuple[int, Path]]:
    """(epoch, checkpoint) pairs, with last.pt treated as the final epoch."""
    w = ROOT / "checkpoints" / run / "weights"
    out = [(int(p.stem.replace("epoch", "")), p) for p in w.glob("epoch*.pt")]
    last = w / "last.pt"
    if last.is_file():
        # results.csv is the authority on how many epochs actually completed
        rc = ROOT / "checkpoints" / run / "results.csv"
        n = len(rc.read_text().strip().splitlines()) - 1 if rc.is_file() else -1
        if n > 0 and all(e != n for e, _ in out):
            out.append((n, last))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training run whose checkpoints to sweep")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4, help="small by default: this may share the GPU")
    cli = ap.parse_args()

    cmp37 = _load_compare_helpers()
    out_csv = ROOT / "results" / f"earlystop_curve_{cli.run}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    done: set[int] = set()
    if out_csv.is_file():
        with open(out_csv) as f:
            done = {int(r["epoch"]) for r in csv.DictReader(f)}
        print(f"[resume] {len(done)} epochs already scored: {sorted(done)}")
    else:
        with open(out_csv, "w", newline="") as f:
            csv.writer(f).writerow(["run", "epoch", "imgsz", "ap50", "recall_ceiling", "n_dets", "n_gt"])

    todo = [(e, p) for e, p in epochs_available(cli.run) if e not in done]
    print(f"[plan] {len(todo)} checkpoints to score for {cli.run}: {[e for e, _ in todo]}")

    for epoch, ckpt in todo:
        stage = f"_es_{cli.run}_e{epoch:03d}"
        dst = ROOT / "checkpoints" / stage / "weights" / "best.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(ckpt, dst)          # copy, never move

        cache = ROOT / "results" / "cache" / f"dets_{stage}_sz{cli.imgsz}.npz"
        if not cache.is_file():
            print(f"[score] epoch {epoch} -> {stage}", flush=True)
            r = subprocess.run(
                [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "scripts" / "29_dump_detections.py"),
                 "--run", stage, "--imgsz", str(cli.imgsz), "--batch", str(cli.batch)],
                cwd=ROOT,
            )
            if r.returncode != 0 or not cache.is_file():
                print(f"[skip] epoch {epoch} failed (rc={r.returncode}); leaving it for the next pass")
                continue

        d = cmp37.load(stage, cli.imgsz)
        ap50 = cmp37.ap50(d)
        rec = cmp37.curve(d)[0]
        ceiling = float(rec[-1]) if len(rec) else 0.0
        with open(out_csv, "a", newline="") as f:
            csv.writer(f).writerow([cli.run, epoch, cli.imgsz, f"{ap50:.6f}",
                                    f"{ceiling:.6f}", len(d["conf"]), d["n_gt"]])
        print(f"  epoch {epoch:>3}  AP50 {ap50:.4f}  recall ceiling {ceiling:.4f}", flush=True)

    # Report the shape, because the shape is the finding.
    with open(out_csv) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["epoch"]))
    if rows:
        print(f"\ncross-domain AP@0.5 against training epoch, {cli.run}")
        for r in rows:
            print(f"  epoch {int(r['epoch']):>3}  AP50 {float(r['ap50']):.4f}  ceiling {float(r['recall_ceiling']):.4f}")
        best = max(rows, key=lambda r: float(r["ap50"]))
        final = rows[-1]
        drop = float(best["ap50"]) - float(final["ap50"])
        print(f"\n  peak   epoch {int(best['epoch'])} at AP50 {float(best['ap50']):.4f}")
        print(f"  final  epoch {int(final['epoch'])} at AP50 {float(final['ap50']):.4f}")
        if float(best["ap50"]) > 0:
            print(f"  training past the peak costs {drop:+.4f} AP50 "
                  f"({drop / float(best['ap50']) * 100:.1f} percent relative)")
        print("\n  A peak at the final epoch means there is no early-stopping effect and the")
        print("  two-point result was noise. Report that outcome as readily as the other one.")


if __name__ == "__main__":
    main()
