"""R6 Steps 3 and 4: re-validate every exported artifact and record deltas.

The whole point of GATE-4 is that the exported FILE is measured, never the .pt.
Class-index reordering, letterbox and normalisation mismatch between Python
preprocessing and the exported graph, and INT8 collapse in the small-object
regime all appear here and nowhere else.

`review_preprocessing` flags any artifact losing more than 2 pp of mAP50 against
its own run's pt_fp32 baseline. That flag does NOT mean quantisation damage --
preprocessing mismatch is the more common cause and looks identical. Rule it out
before reporting an INT8 drop as a finding.

AP_small matters more than aggregate mAP for this paper: roughly 90 percent of
persons in this dataset are below 32x32 px, so aggregate mAP is dominated by
vehicles and would mask exactly the degradation that breaks the SAR claim.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"

BASELINE_ARTIFACT = "pt_fp32"
# Threshold from the plan of record: beyond this, diagnose preprocessing first.
PREPROCESSING_REVIEW_PP = 2.0


def compute_deltas(rows: list[dict]) -> list[dict]:
    """Attach deltas versus each run's own pt_fp32 baseline.

    Raises if a run has artifacts but no baseline: reporting a zero delta in that
    case would present an unvalidated artifact as matching the reference.
    """
    baselines = {r["run"]: r for r in rows if r["artifact_id"] == BASELINE_ARTIFACT}
    out: list[dict] = []
    for r in rows:
        base = baselines.get(r["run"])
        if base is None:
            raise ValueError(f"run {r['run']!r} has no pt_fp32 baseline to compare against")
        d50 = r["mAP50"] - base["mAP50"]
        d5095 = r["mAP50_95"] - base["mAP50_95"]
        dsmall = r["AP_small"] - base["AP_small"]
        out.append(
            {
                **r,
                "delta_mAP50": round(d50, 5),
                "delta_mAP50_95": round(d5095, 5),
                "delta_AP_small": round(dsmall, 5),
                "delta_mAP50_pp": round(d50 * 100, 3),
                "delta_mAP50_95_pp": round(d5095 * 100, 3),
                "delta_AP_small_pp": round(dsmall * 100, 3),
                "review_preprocessing": bool(d50 * 100 < -PREPROCESSING_REVIEW_PP),
            }
        )
    return out


def validate_artifact(path: Path, data_cfg: Path, imgsz: int, batch: int, device: str = "cpu") -> dict:
    """Run test-split val on one exported artifact and return its metrics.

    `device` defaults to cpu for exported artifacts. Ultralytics places input
    tensors on the GPU whenever CUDA is available, but the exported runtimes
    here are CPU-only: `.venv` carries plain `onnxruntime`, and NCNN and TFLite
    have no CUDA path at all. The mismatch surfaces as

        Error when binding input: There's no data transfer registered for
        copying tensors from Device:[DeviceType:1 ... VendorId:4318 ...]

    (VendorId 4318 = 0x10DE = NVIDIA). Forcing cpu also matches the deployment
    target -- every one of these artifacts exists to run on a Pi 4 CPU -- and
    accuracy is device-independent up to float ordering, so it does not weaken
    the parity comparison. Latency is not measured here; that is R7 on the Pi.

    pt_fp32 keeps the GPU so its rows reproduce results/ablation_test.csv
    exactly, which makes the two files cross-checkable.
    """
    from ultralytics import YOLO

    model = YOLO(str(path))
    m = model.val(
        data=str(data_cfg),
        split="test",
        imgsz=imgsz,
        batch=batch,
        workers=2,
        device=device,
        plots=False,
        verbose=False,
    )
    mp, mr, map50, map5095 = m.box.mean_results()
    # Ultralytics exposes size-stratified AP only through the COCO eval path;
    # fall back to the person-class AP, which is the size-limited class here and
    # is what the SAR claim actually rests on. The paper must label this column
    # as person AP, NOT as COCO AP-small.
    names = {int(k): v for k, v in m.names.items()} if isinstance(m.names, dict) else dict(enumerate(m.names))
    ap_small = float("nan")
    for i, c in enumerate(m.box.ap_class_index):
        if names[int(c)] == "person":
            ap_small = float(m.box.class_result(i)[3])
            break
    return {
        "precision": round(float(mp), 5),
        "recall": round(float(mr), 5),
        "mAP50": round(float(map50), 5),
        "mAP50_95": round(float(map5095), 5),
        "AP_small": round(ap_small, 5),
    }


def main() -> int:
    import argparse

    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--only-tflite", action="store_true", help="measure only tflite (run in .venv-tflite)")
    ap.add_argument("--skip-tflite", action="store_true", help="measure everything but tflite (run in .venv)")
    cli = ap.parse_args()

    # Measurement is split by interpreter because TFLite inference needs the
    # tensorflow stack that deliberately lives only in .venv-tflite, while the
    # pt_fp32 baseline needs the CUDA torch that lives only in .venv. Each leg
    # writes raw metrics; deltas are computed in the merge below once both
    # legs' raw files exist, since a leg on its own has no baseline to
    # compare against.
    raw_path = RESULTS / ("parity_raw_tflite.json" if cli.only_tflite else "parity_raw.json")

    manifests = [RESULTS / "export_manifest.json", RESULTS / "export_manifest_tflite.json"]
    artifacts: list[dict] = []
    for mf in manifests:
        if mf.exists():
            artifacts.extend(json.loads(mf.read_text())["artifacts"])

    # Every tflite variant runs in .venv-tflite, not just the one ultralytics
    # returns from an int8 export. Matching on the prefix keeps float32/float16/
    # full-integer rows in the same leg.
    is_tflite = lambda a: a["artifact_id"].startswith("tflite")  # noqa: E731
    if cli.only_tflite:
        artifacts = [a for a in artifacts if is_tflite(a)]
    elif cli.skip_tflite:
        artifacts = [a for a in artifacts if not is_tflite(a)]

    rows: list[dict] = []
    for a in artifacts:
        if a["status"] != "ok" or not a["path"]:
            print(f"[skip] {a['run']} {a['artifact_id']}: {a['status']}", flush=True)
            continue
        run_args = yaml.safe_load((ROOT / "checkpoints" / a["run"] / "args.yaml").read_text())
        data_cfg = ROOT / run_args["data"]
        # pt_fp32 on GPU so it reproduces ablation_test.csv exactly; every
        # exported artifact on CPU, where its runtime actually lives.
        device = "0" if a["artifact_id"] == BASELINE_ARTIFACT else "cpu"
        try:
            metrics = validate_artifact(
                ROOT / a["path"], data_cfg, a["imgsz"], run_args["batch"], device=device
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {a['run']} {a['artifact_id']}: {exc}", flush=True)
            continue
        rows.append(
            {
                "run": a["run"],
                "artifact_id": a["artifact_id"],
                "format": a["format"],
                "imgsz": a["imgsz"],
                "size_bytes": a["size_bytes"],
                **metrics,
            }
        )
        print(f"[ok] {a['run']} {a['artifact_id']} mAP50={metrics['mAP50']}", flush=True)

    if not rows:
        print("no artifacts validated; refusing to write an empty parity CSV")
        return 1

    raw_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print(f"[write] {raw_path} ({len(rows)} rows)", flush=True)

    # Merge every leg that has run. Deltas need the pt_fp32 baseline, which only
    # the non-tflite leg produces, so a tflite-only run before the main leg
    # simply records its raw numbers and defers the CSV.
    merged: list[dict] = []
    for f in (RESULTS / "parity_raw.json", RESULTS / "parity_raw_tflite.json"):
        if f.exists():
            merged.extend(json.loads(f.read_text())["rows"])

    if not any(r["artifact_id"] == BASELINE_ARTIFACT for r in merged):
        print(f"no {BASELINE_ARTIFACT} baseline yet; raw metrics saved, CSV deferred")
        return 0

    # Drop rows whose run has no baseline rather than aborting the whole merge:
    # a run that failed to export its .pt should not block the others.
    runs_with_baseline = {r["run"] for r in merged if r["artifact_id"] == BASELINE_ARTIFACT}
    dropped = [r for r in merged if r["run"] not in runs_with_baseline]
    for r in dropped:
        print(f"[skip] {r['run']} {r['artifact_id']}: run has no {BASELINE_ARTIFACT} baseline")
    merged = [r for r in merged if r["run"] in runs_with_baseline]

    out_rows = compute_deltas(merged)
    out_csv = RESULTS / "export_parity.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"[write] {out_csv} ({len(out_rows)} rows)", flush=True)

    flagged = [r for r in out_rows if r["review_preprocessing"]]
    if flagged:
        print(f"\n{len(flagged)} artifact(s) lost more than {PREPROCESSING_REVIEW_PP} pp mAP50:")
        for r in flagged:
            print(f"  {r['run']} {r['artifact_id']}: {r['delta_mAP50_pp']} pp")
        print("Diagnose preprocessing (letterbox pad colour, BGR/RGB order, scale,")
        print("class index order) BEFORE attributing this to quantisation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
