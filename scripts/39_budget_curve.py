"""How many labelled wilderness sequences do you actually need?

THE QUESTION
------------
The paper's Conclusion says "a few dozen labelled target-domain sequences buy more
than any inference-time adjustment." That is extrapolated from exactly ONE
fine-tuning run on 31 sequences, and it is the weakest sentence in the paper.

A rescue organisation deciding whether to fund a labelling effort needs the curve,
not the endpoint: does accuracy saturate at 4 sequences, or is it still climbing at
31? Those imply completely different budgets.

WHAT IT DOES
------------
Fine-tunes from the VisDrone-trained checkpoint on N target sequences for
N in {1, 2, 4, 8, 16, 31}, holding the validation and TEST splits fixed at the
sequence-disjoint sets `22_build_wisard_finetune_split.py` produced, then evaluates
every model on that same untouched test set.

Hyperparameters are copied exactly from the run that produced the paper's 6.1x
figure (`checkpoints/y11n_wisardft_s42/args.yaml`), so the N=31 point should
reproduce it. If it does not, something in this script is wrong -- treat N=31 as a
built-in control, not as a result.

WHICH SEQUENCES GET PICKED, AND WHY IT MATTERS
----------------------------------------------
Sequences are wildly uneven here: per-sequence person-box counts run 0 to 1,122 and
six sequences contain no person at all. Picking N sequences at random can hand N=4
either 2,000 boxes or none, which would make the curve a measurement of luck.

So the pick is seeded and stratified by box count: sequences are ordered by count,
then sampled across that order, so a small budget gets a representative spread
rather than the head or the tail. Repeating over several seeds is what separates
the curve from the draw -- run with --seeds 42,123,456.

RESUMABLE. A finished (N, seed) writes its row and is skipped on the next pass.

USAGE
  .venv/bin/python scripts/39_budget_curve.py --seeds 42
  .venv/bin/python scripts/39_budget_curve.py --seeds 42,123,456
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FT_CONFIG = ROOT / "configs" / "sar_wisard_finetune.yaml"
SOURCE_CKPT = ROOT / "checkpoints" / "y11n_tiled640_s42" / "weights" / "best.pt"


def sequence_of(tile_path: str) -> str:
    """Sequence key for a WiSARD tile, matching src/sar/data/sar_eval.py."""
    stem = Path(tile_path).stem
    base = stem.split("__")[0]
    return base.rpartition("_")[0] or base


def build_subsplit(n_seq: int, seed: int, ft_root: Path) -> tuple[Path, int, int]:
    """Write a train list holding n_seq sequences. Returns (path, n_tiles, n_boxes)."""
    train_txt = ft_root / "train.txt"
    by_seq: dict[str, list[str]] = defaultdict(list)
    for line in train_txt.read_text().split("\n"):
        if line.strip():
            by_seq[sequence_of(line)].append(line)

    # Box count per sequence, from the label file beside each tile.
    def boxes(paths: list[str]) -> int:
        total = 0
        for p in paths:
            lp = Path(p.replace("/images/", "/labels/")).with_suffix(".txt")
            if lp.is_file():
                total += sum(1 for ln in lp.read_text().splitlines() if ln.strip().startswith("0 "))
        return total

    counts = {s: boxes(v) for s, v in by_seq.items()}
    ordered = sorted(counts, key=lambda s: counts[s])          # ascending by box count
    if n_seq >= len(ordered):
        picked = ordered
    else:
        # even stride across the ordering, jittered by seed -> representative spread
        rng = random.Random(seed)
        step = len(ordered) / n_seq
        picked = []
        for i in range(n_seq):
            lo, hi = int(i * step), max(int(i * step), int((i + 1) * step) - 1)
            picked.append(ordered[rng.randint(lo, hi)])
        picked = list(dict.fromkeys(picked))
        # a collision from the jitter would silently shrink the budget
        pool = [s for s in ordered if s not in picked]
        while len(picked) < n_seq and pool:
            picked.append(pool.pop(rng.randrange(len(pool))))

    tiles = [p for s in picked for p in by_seq[s]]
    dest = ft_root / f"train_n{n_seq}_s{seed}.txt"
    dest.write_text("\n".join(tiles) + "\n")
    return dest, len(tiles), sum(counts[s] for s in picked)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="1,2,4,8,16,31")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=15,
                    help="the paper's run peaked at epoch 5 of 40; 15 is headroom, not a change of method")
    ap.add_argument("--batch", type=int, default=32)
    cli = ap.parse_args()

    base = yaml.safe_load(FT_CONFIG.read_text())
    ft_root = Path(base["path"])
    out_csv = ROOT / "results" / "budget_curve.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[int, int]] = set()
    if out_csv.is_file():
        with open(out_csv) as f:
            done = {(int(r["n_sequences"]), int(r["seed"])) for r in csv.DictReader(f)}
        print(f"[resume] {len(done)} points already measured")
    else:
        with open(out_csv, "w", newline="") as f:
            csv.writer(f).writerow(["n_sequences", "seed", "n_tiles", "n_boxes",
                                    "person_ap50", "person_recall", "epochs"])

    from ultralytics import YOLO

    for seed in [int(s) for s in cli.seeds.split(",")]:
        for n_seq in [int(b) for b in cli.budgets.split(",")]:
            if (n_seq, seed) in done:
                print(f"[skip] N={n_seq} seed={seed} already done")
                continue

            sub, n_tiles, n_boxes = build_subsplit(n_seq, seed, ft_root)
            cfg = dict(base)
            cfg["train"] = sub.name                     # val and test stay untouched
            cfg_path = ROOT / "configs" / f"_budget_n{n_seq}_s{seed}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

            name = f"y11n_ftbudget_n{n_seq}_s{seed}"
            print(f"\n[train] N={n_seq} seed={seed}: {n_tiles:,} tiles, {n_boxes:,} boxes", flush=True)

            model = YOLO(str(SOURCE_CKPT))
            model.train(
                data=str(cfg_path), epochs=cli.epochs, imgsz=640, batch=cli.batch,
                optimizer="SGD", lr0=1e-3, seed=seed, deterministic=True, cos_lr=True,
                patience=15, workers=2, warmup_epochs=1.0, pretrained=True,
                project="checkpoints", name=name, exist_ok=True, save_period=-1, verbose=False,
            )

            best = ROOT / "checkpoints" / name / "weights" / "best.pt"
            m = YOLO(str(best)).val(data=str(cfg_path), split="test", imgsz=640,
                                    batch=cli.batch, verbose=False)
            # class 0 is person in this three-class space
            ap50 = float(m.box.ap50[0]) if len(m.box.ap50) else float("nan")
            rec = float(m.box.r[0]) if len(m.box.r) else float("nan")

            with open(out_csv, "a", newline="") as f:
                csv.writer(f).writerow([n_seq, seed, n_tiles, n_boxes,
                                        f"{ap50:.6f}", f"{rec:.6f}", cli.epochs])
            print(f"  N={n_seq} seed={seed}: person AP50 {ap50:.4f}  recall {rec:.4f}", flush=True)

    with open(out_csv) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: (int(r["seed"]), int(r["n_sequences"])))
    print("\ntarget-domain data budget, person AP@0.5 on the fixed held-out test sequences")
    for r in rows:
        print(f"  N={int(r['n_sequences']):>2} seed={r['seed']}  "
              f"AP50 {float(r['person_ap50']):.4f}  recall {float(r['person_recall']):.4f}  "
              f"({int(r['n_boxes']):,} boxes)")
    print("\n  N=31 is the control: it should land near the paper's 0.4575. If it does not,")
    print("  distrust the whole curve and find the discrepancy before reporting anything.")


if __name__ == "__main__":
    main()
