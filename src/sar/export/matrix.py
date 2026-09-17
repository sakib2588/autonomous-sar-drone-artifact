"""R6 Step 1 and Step 2: export every trained run to the deployment formats.

Artifact set, per the plan of record:

    pt_fp32      PyTorch FP32     reference, GPU only, never a paper accuracy row
    onnx_fp32    ONNX FP32        opset 12, simplified
    ncnn_fp32    NCNN FP32
    ncnn_fp16    NCNN FP16
    tflite_int8  TFLite INT8      calibrated on TRAIN images only

Ultralytics NCNN export produces FP32 or FP16 and never INT8. Real NCNN INT8
requires an offline ncnn2table plus ncnn2int8 pass, which is out of budget. Do
not describe any artifact here as "NCNN INT8".

Run the tflite leg with `.venv-tflite/bin/python`; every other leg runs in
`.venv`. The driver records which interpreter produced each artifact so the
manifest is reproducible.

Verified 2026-08-07 that split, fraction, int8, half, opset and simplify are all
in DEFAULT_CFG_DICT, so they survive the kwargs path rather than being silently
dropped -- without that, the calibration guard could not be enforced here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

from src.sar.export.calibration import (
    assert_calibration_split_is_train,
    calibration_fraction,
    calibration_image_count,
    calibration_memory_bytes,
)

ROOT = Path(__file__).resolve().parents[3]
CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "results"

RUNS = [
    "y11n_tiled640_s42",
    "y8n_tiled640_s42",
    "y11n_full640_s42",
    "y11n_tiled416_s42",
]

# (artifact id, ultralytics format, extra export kwargs, needs the tflite venv)
ARTIFACTS = [
    ("onnx_fp32", "onnx", {"opset": 12, "simplify": True}, False),
    ("ncnn_fp32", "ncnn", {}, False),
    ("ncnn_fp16", "ncnn", {"half": True}, False),
    ("tflite_int8", "tflite", {"int8": True}, True),
]

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}


def count_train_images(data_cfg: Path) -> int:
    cfg = yaml.safe_load(data_cfg.read_text())
    train_dir = Path(cfg["path"]) / cfg["train"]
    return sum(1 for p in train_dir.iterdir() if p.suffix.lower() in _IMG_SUFFIXES)


def export_one(run: str, artifact_id: str, fmt: str, extra: dict, imgsz: int, data_cfg: Path) -> dict:
    """Export a single artifact in-process. Returns a manifest entry."""
    from ultralytics import YOLO

    best = CKPT / run / "weights" / "best.pt"
    kwargs = dict(format=fmt, imgsz=imgsz, **extra)

    if extra.get("int8"):
        # The two guards. split must be train, and fraction must bound the
        # float32 concat that export_saved_model performs.
        n_train = count_train_images(data_cfg)
        fraction = calibration_fraction(n_train)
        assert_calibration_split_is_train("train")
        n_calib = calibration_image_count(n_train, fraction)
        mem_gb = calibration_memory_bytes(n_calib, imgsz) / 1024**3
        print(
            f"[calib] {run}: {n_calib} images from {n_train} train "
            f"(fraction={fraction:.6f}), concat ~{mem_gb:.2f} GiB",
            flush=True,
        )
        kwargs.update(data=str(data_cfg), split="train", fraction=fraction)

    model = YOLO(str(best))
    out = model.export(**kwargs)
    out_path = Path(out)

    # Ultralytics names NCNN output `<stem>_ncnn_model/` with no precision in
    # the name, so ncnn_fp32 and ncnn_fp16 land on the SAME path and the second
    # export silently overwrites the first -- leaving two manifest rows that
    # both point at the FP16 directory. Verified 2026-08-07 by smoke test.
    #
    # Move each artifact into a subdirectory named for its id rather than
    # renaming the file: AutoBackend infers the format from the trailing name
    # (`_ncnn_model`, `_saved_model`, `.onnx`), so renaming would break loading
    # in parity.py. The subdirectory keeps the basename Ultralytics expects.
    dest_dir = out_path.parent / artifact_id
    dest = dest_dir / out_path.name
    if out_path.parent.name != artifact_id:
        dest_dir.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
        out_path = Path(shutil.move(str(out_path), str(dest)))

    size = (
        sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file())
        if out_path.is_dir()
        else out_path.stat().st_size
    )
    return {
        "run": run,
        "artifact_id": artifact_id,
        "format": fmt,
        "path": str(out_path.relative_to(ROOT)) if out_path.is_relative_to(ROOT) else str(out_path),
        "imgsz": imgsz,
        "size_bytes": size,
        "interpreter": sys.executable,
        "export_kwargs": {k: v for k, v in kwargs.items() if k != "data"},
        "status": "ok",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-tflite", action="store_true", help="run only the tflite leg (for .venv-tflite)")
    ap.add_argument("--skip-tflite", action="store_true", help="skip the tflite leg (for .venv)")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    entries: list[dict] = []

    for run in RUNS:
        args_path = CKPT / run / "args.yaml"
        best = CKPT / run / "weights" / "best.pt"
        if not best.exists() or not args_path.exists():
            print(f"[skip] {run}: no best.pt", flush=True)
            continue
        run_args = yaml.safe_load(args_path.read_text())
        imgsz, data_cfg = run_args["imgsz"], ROOT / run_args["data"]

        # The reference row. Recorded, never used for a paper accuracy number.
        # Only the .venv leg writes it, so the two manifests do not duplicate it.
        if not args.only_tflite:
            entries.append(
                {
                    "run": run,
                    "artifact_id": "pt_fp32",
                    "format": "pt",
                    "path": str(best.relative_to(ROOT)),
                    "imgsz": imgsz,
                    "size_bytes": best.stat().st_size,
                    "interpreter": sys.executable,
                    "export_kwargs": {},
                    "status": "ok",
                }
            )

        for artifact_id, fmt, extra, needs_tflite in ARTIFACTS:
            if needs_tflite and args.skip_tflite:
                continue
            if not needs_tflite and args.only_tflite:
                continue
            try:
                entries.append(export_one(run, artifact_id, fmt, extra, imgsz, data_cfg))
                print(f"[ok] {run} {artifact_id}", flush=True)
            except Exception as exc:  # noqa: BLE001 - a failed format must not sink the matrix
                print(f"[FAIL] {run} {artifact_id}: {exc}", flush=True)
                entries.append(
                    {
                        "run": run,
                        "artifact_id": artifact_id,
                        "format": fmt,
                        "path": None,
                        "imgsz": imgsz,
                        "size_bytes": None,
                        "interpreter": sys.executable,
                        "export_kwargs": {},
                        "status": f"failed: {exc}",
                    }
                )

    out = RESULTS / ("export_manifest_tflite.json" if args.only_tflite else "export_manifest.json")
    # Overwrite rather than append: a re-run must replace its own leg's results,
    # not stack duplicate rows that would then inflate the GATE-4 coverage check.
    out.write_text(json.dumps({"artifacts": entries}, indent=2) + "\n")
    print(f"[write] {out} ({len(entries)} entries)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
