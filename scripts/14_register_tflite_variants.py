#!/usr/bin/env python3
"""Register every TFLite variant the export produced, not just the returned one.

Why this exists. `model.export(format="tflite", int8=True)` writes FIVE files
into `<stem>_saved_model/` and returns the path to `best_int8.tflite`. Taking
the return value at face value -- which `matrix.py` originally did -- records
one artifact and silently discards four.

That is not a bookkeeping detail. Measured on a Pi 4 (results/bench/):

    best_full_integer_quant.tflite   410.1 ms   true int8 I/O, FASTEST
    best_float32.tflite              495.2 ms
    best_float16.tflite              500.6 ms   no gain over float32 on A72
    best_int8.tflite                 761.2 ms   float32 I/O, 193/991 int8
                                                tensors, SLOWEST of the four

The returned `best_int8.tflite` is a hybrid: float I/O with partially quantised
weights, and on this core it is 54 percent slower than plain float32. So the
GATE-4 parity table was reporting accuracy for a file nobody would deploy, while
the latency table reported a different file. This script rebuilds the tflite
manifest from what is actually on disk so accuracy and latency describe the same
artifacts.

`best_integer_quant.tflite` is also produced; it is the per-tensor integer
variant with float I/O, kept for completeness so the matrix is exhaustive.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "results"

RUNS = [
    "y11n_tiled640_s42",
    "y8n_tiled640_s42",
    "y11n_full640_s42",
    "y11n_tiled416_s42",
]

# filename stem -> artifact id. Names chosen so the paper's table is unambiguous
# about which file each row describes.
VARIANTS = {
    "best_float32": "tflite_fp32",
    "best_float16": "tflite_fp16",
    "best_full_integer_quant": "tflite_int8_full",
    "best_integer_quant": "tflite_int8_pertensor",
    "best_int8": "tflite_int8_hybrid",
}


def main() -> int:
    entries: list[dict] = []
    for run in RUNS:
        args_path = CKPT / run / "args.yaml"
        if not args_path.exists():
            print(f"[skip] {run}: no args.yaml", flush=True)
            continue
        imgsz = yaml.safe_load(args_path.read_text())["imgsz"]
        weights = CKPT / run / "weights"

        for stem, artifact_id in VARIANTS.items():
            # The hybrid was moved into its own subdirectory by matrix.py's
            # collision fix; the rest sit directly in best_saved_model/.
            candidates = [
                weights / "best_saved_model" / f"{stem}.tflite",
                weights / "best_saved_model" / "tflite_int8" / f"{stem}.tflite",
            ]
            path = next((p for p in candidates if p.exists()), None)
            if path is None:
                print(f"[miss] {run} {artifact_id}: {stem}.tflite not found", flush=True)
                continue
            entries.append(
                {
                    "run": run,
                    "artifact_id": artifact_id,
                    "format": "tflite",
                    "path": str(path.relative_to(ROOT)),
                    "imgsz": imgsz,
                    "size_bytes": path.stat().st_size,
                    "interpreter": "(registered from disk)",
                    "export_kwargs": {"int8": True},
                    "status": "ok",
                }
            )
            print(f"[ok] {run} {artifact_id} <- {path.name} "
                  f"({path.stat().st_size / 1048576:.2f} MB)", flush=True)

    out = RESULTS / "export_manifest_tflite.json"
    out.write_text(json.dumps({"artifacts": entries}, indent=2) + "\n")
    print(f"\n[write] {out} ({len(entries)} entries)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
