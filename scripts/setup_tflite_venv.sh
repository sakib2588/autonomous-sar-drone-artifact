#!/usr/bin/env bash
# Builds .venv-tflite for the TFLite INT8 export leg ONLY.
#
# Why a second venv. `tensorflow<=2.19.0` pins numpy<2.2. The training venv
# (.venv) holds numpy 2.4.4 under torch 2.5.1+cu124 and ultralytics 8.3.253 --
# the exact install that produced all four sweep checkpoints. Installing
# TensorFlow into .venv would downgrade numpy underneath that stack, so a
# reproducibility claim about the checkpoints would no longer hold against the
# recorded results/environment_train.txt. Export is a one-way transformation,
# so it costs nothing to run that leg in its own interpreter.
#
# ---------------------------------------------------------------------------
# The dependency trap, resolved 2026-08-07 after four failed attempts.
#
# The naive install from the ultralytics check_requirements list does NOT work:
#
#   1. `onnx2tf` unpinned resolves to 2.6.8, which pins numpy==2.2.6 and
#      protobuf==7.35.1 EXACTLY. tensorflow 2.19.0 requires numpy<2.2.0 and
#      protobuf<6.0.0. Mutually exclusive. pip installs it anyway and only
#      warns -- leaving a venv where `import tensorflow` is broken but the
#      install "succeeded". Constrain onnx2tf to <2.0.
#
#   2. `protobuf>=5` (what ultralytics asks for) is too loose and lets 7.35.1
#      in through onnx2tf. It must be `protobuf>=5,<6`.
#
#   3. Some onnx2tf 1.x releases need `onnxsim==0.4.36`, which has no wheel and
#      builds from source via cmake. pip's build isolation does not see a cmake
#      installed inside the venv, so cmake must be on PATH for the install.
#      With the constraints above pip settles on onnx2tf 1.28.8, which takes
#      the prebuilt onnxsim wheel and needs no compiler -- but the PATH export
#      is kept so a future resolution that does need it still works.
#
# Verify with `pip check`, NOT with pip's exit code: pip exits 0 while printing
# "dependency conflicts" for exactly the case in point 1.
#
# NCNN is different: it installs into .venv with --no-deps so it cannot move any
# pin. It does need portalocker and tqdm at runtime, added the same way. tqdm
# was genuinely absent from .venv because ultralytics 8.3.253 ships its own
# TQDM implementation and never required it.
# ---------------------------------------------------------------------------
set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ"

echo "== installing ncnn into the training venv (no deps, no pin changes) =="
.venv/bin/pip install --no-deps ncnn portalocker tqdm

echo "== building .venv-tflite =="
python3 -m venv .venv-tflite
.venv-tflite/bin/pip install --upgrade pip

# Same ultralytics and torch VERSIONS as the training venv so the exporter code
# path is identical; CPU build because export is a graph transformation and
# needs no GPU. Saves ~2.5 GB and avoids a second CUDA toolchain.
# Ultralytics loads the CUDA-trained checkpoints via map_location, so the
# weights are unaffected.
.venv-tflite/bin/pip install \
  "torch==2.5.1" "torchvision==0.20.1" \
  --index-url https://download.pytorch.org/whl/cpu

# Stage 1: TensorFlow and ultralytics. Installed BEFORE the onnx chain so that
# a later resolver conflict is attributable rather than buried in one big graph.
.venv-tflite/bin/pip install \
  "tensorflow==2.19.0" \
  "tf_keras<=2.19.0" \
  "ultralytics==8.3.253"

# Stage 2: the onnx2tf chain, with the three constraints from the header.
# cmake on PATH covers the case where the resolver picks a source-built onnxsim.
.venv-tflite/bin/pip install cmake
PATH="$PROJ/.venv-tflite/bin:$PATH" .venv-tflite/bin/pip install \
  "onnx2tf>=1.26.3,<2.0" \
  "numpy>=1.26.0,<2.2.0" \
  "protobuf>=5,<6" \
  "tensorflow==2.19.0" \
  "sng4onnx>=1.0.1" \
  "onnx_graphsurgeon>=0.3.26" \
  "ai-edge-litert>=1.2.0" \
  "onnx>=1.12.0,<2.0.0" \
  "onnxslim>=0.1.71" \
  "onnxruntime" \
  --extra-index-url https://pypi.ngc.nvidia.com

echo "== verifying the training venv was NOT perturbed =="
.venv/bin/pip check
.venv/bin/python -c "
import numpy, torch, ultralytics, ncnn
assert numpy.__version__.startswith('2.4'), f'numpy moved to {numpy.__version__} -- training venv damaged'
print('ok  numpy', numpy.__version__, '| torch', torch.__version__, '| ultralytics', ultralytics.__version__)
print('ok  ncnn importable')
"

echo "== verifying the tflite venv =="
# pip check, not exit code -- pip exits 0 on the conflicts that matter here.
.venv-tflite/bin/pip check
.venv-tflite/bin/python -c "
import tensorflow as tf, numpy, onnx, onnx2tf, onnxsim, ultralytics, torch, google.protobuf
# Tuple compare, not string: '2.10' < '2.2' is True as strings and would let a
# tensorflow-breaking numpy through unnoticed.
nv = tuple(int(x) for x in numpy.__version__.split('.')[:2])
pv = tuple(int(x) for x in google.protobuf.__version__.split('.')[:2])
assert nv < (2, 2), f'numpy {numpy.__version__} breaks tensorflow 2.19 (needs <2.2)'
assert pv < (6, 0), f'protobuf {google.protobuf.__version__} breaks tensorflow 2.19 (needs <6)'
print('ok  tensorflow', tf.__version__, '| numpy', numpy.__version__, '| protobuf', google.protobuf.__version__)
print('ok  onnx2tf', onnx2tf.__version__, '| ultralytics', ultralytics.__version__, '| torch', torch.__version__)
"
echo "SETUP OK"
