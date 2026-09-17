#!/usr/bin/env python3
"""R5 step 3+4: test-split evaluation of the four sweep runs, plus the train manifest.

Runs `val` on the TEST split for every completed run and writes one tidy row per
(run, class) to results/ablation_test.csv, plus a run-level row with class="all".

Why read hyperparameters back out of checkpoints/<name>/args.yaml instead of
hardcoding them here: imgsz and data differ per run (416/640, tiled/full) and a
mismatch between train and val imgsz silently changes the numbers rather than
erroring. args.yaml is written by Ultralytics at train start, so it is the
authoritative record of what the checkpoint actually saw.

best.pt, not last.pt. last.pt is the final epoch; best.pt is the epoch that won
on val mAP50-95 and is what a deployment would ship. Both are recorded in the
manifest so the choice is auditable.

F1 is computed here as 2PR/(P+R); Ultralytics does not log it directly. It is
derived from the same P and R at the confidence that maximises F1 internally, so
it is consistent with the reported P/R pair but is NOT independent of them --
do not present it as a separate measurement.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "results"
RUNS = [
    "y11n_tiled640_s42",
    "y8n_tiled640_s42",
    "y11n_full640_s42",
    "y11n_tiled416_s42",
]
# Low, because the assigner-fallback OOM that killed run 2 was a host-RAM event
# and val on the 9,377-image test split is not throughput-critical.
WORKERS = 2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception as exc:  # noqa: BLE001 - manifest must still be written
        return f"unavailable: {exc}"


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def eval_run(name: str) -> tuple[list[dict], dict] | None:
    """Return (per-class rows, manifest entry) or None if the run has no best.pt."""
    from ultralytics import YOLO

    run_dir = CKPT / name
    best = run_dir / "weights" / "best.pt"
    args_path = run_dir / "args.yaml"
    if not best.exists() or not args_path.exists():
        print(f"[skip] {name}: missing best.pt or args.yaml", flush=True)
        return None

    args = yaml.safe_load(args_path.read_text())
    imgsz, data, batch = args["imgsz"], args["data"], args["batch"]
    print(f"[eval] {name} imgsz={imgsz} batch={batch} data={data}", flush=True)

    model = YOLO(str(best))
    m = model.val(
        data=str(ROOT / data),
        split="test",
        imgsz=imgsz,
        batch=batch,
        workers=WORKERS,
        project=str(RESULTS / "test_eval"),
        name=name,
        exist_ok=True,
        plots=False,
        verbose=True,
    )

    names = m.names  # {idx: class name}
    rows: list[dict] = []

    # Run-level. mean_results() -> (P, R, mAP50, mAP50-95)
    mp, mr, map50, map5095 = m.box.mean_results()
    rows.append(
        {
            "run": name,
            "split": "test",
            "class": "all",
            "imgsz": imgsz,
            "precision": round(mp, 5),
            "recall": round(mr, 5),
            "mAP50": round(map50, 5),
            "mAP50_95": round(map5095, 5),
            "f1": round(f1(mp, mr), 5),
        }
    )

    # Per-class. m.box.ap_class_index maps result rows back to class indices --
    # a class with zero test instances is absent from the results, so indexing
    # by position would silently attribute metrics to the wrong class.
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap5095 = m.box.class_result(i)
        rows.append(
            {
                "run": name,
                "split": "test",
                "class": names[int(c)],
                "imgsz": imgsz,
                "precision": round(p, 5),
                "recall": round(r, 5),
                "mAP50": round(ap50, 5),
                "mAP50_95": round(ap5095, 5),
                "f1": round(f1(p, r), 5),
            }
        )

    entry = {
        "run": name,
        "data_config": data,
        "imgsz": imgsz,
        "batch": batch,
        "epochs_requested": args.get("epochs"),
        "seed": args.get("seed"),
        "model_init": args.get("model"),
        "best_pt_sha256": sha256(best),
        "last_pt_sha256": sha256(run_dir / "weights" / "last.pt")
        if (run_dir / "weights" / "last.pt").exists()
        else None,
        "results_csv_rows": sum(1 for _ in (run_dir / "results.csv").open()) - 1
        if (run_dir / "results.csv").exists()
        else None,
    }
    return rows, entry


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    all_rows: list[dict] = []
    manifest_runs: list[dict] = []

    for name in RUNS:
        got = eval_run(name)
        if got is None:
            continue
        rows, entry = got
        all_rows.extend(rows)
        manifest_runs.append(entry)

    if not all_rows:
        print("no runs evaluated; refusing to write empty CSV", file=sys.stderr)
        return 1

    out_csv = RESULTS / "ablation_test.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"[write] {out_csv} ({len(all_rows)} rows)", flush=True)

    env_txt = RESULTS / "environment_train.txt"
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "environment_file": str(env_txt.relative_to(ROOT)) if env_txt.exists() else None,
        "environment_sha256": sha256(env_txt) if env_txt.exists() else None,
        "eval_split": "test",
        "weights_used": "best.pt",
        "runs": manifest_runs,
        "caveat": (
            "VisDrone test-dev is a harder person distribution than train/val "
            "(mean person side 14.68 px vs 20.88 px; 31.31% under 8 px vs 9.70%). "
            "This intra-VisDrone shift must be reported separately from the "
            "train-to-SAR domain gap, per notes/20260805-permanent-phase-status-and-blockers.md "
            "Section 7.2."
        ),
    }
    out_json = RESULTS / "train_manifest.json"
    out_json.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[write] {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
