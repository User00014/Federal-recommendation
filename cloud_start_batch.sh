#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ROUNDS="${ROUNDS:-1000}"
SEEDS="${SEEDS:-42}"
EXP_GROUPS="${EXP_GROUPS:-G0,G1,G2,G3,G4,G5,G6,G7,G8,G9,A1,A2,A3,A4L,A4H}"
BATCH_NAME="${BATCH_NAME:-batch_$(date +%Y%m%d-%H%M%S)}"
RUN_SMOKE="${RUN_SMOKE:-1}"
ALLOW_CPU="${ALLOW_CPU:-0}"

cd "$PROJECT_DIR"

echo "[STEP] project: $PROJECT_DIR"

echo "[STEP] create venv"
if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

if [ ! -f ".venv/bin/activate" ]; then
  echo "[ERR] .venv ???????????"
  rm -rf .venv
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate

echo "[STEP] pip install"
python -m pip install --upgrade pip setuptools wheel
if [ -f "requirements_cloud.txt" ]; then
  python -m pip install -r requirements_cloud.txt
elif [ -f "src/requirements" ]; then
  python -m pip install -r src/requirements
else
  echo "[ERR] requirements file not found"
  exit 1
fi

echo "[STEP] env check"
python - <<'PY'
import os
import sys
import torch

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print("gpu_0:", torch.cuda.get_device_name(0))

required = [
    "data/ratings.csv",
    "data/movies.csv",
    "data/tags.csv",
    "data/links.csv",
]
missing = [x for x in required if not os.path.exists(x)]
if missing:
    print("missing_data:", missing)
    sys.exit(3)
PY

GPU_OK=$(python - <<'PY'
import torch
print(1 if torch.cuda.is_available() else 0)
PY
)

if [ "$GPU_OK" != "1" ] && [ "$ALLOW_CPU" != "1" ]; then
  echo "[ERR] CUDA ?????????????CPU?????????"
  echo "[HINT] ????CPU????? ALLOW_CPU=1"
  exit 2
fi

if [ "$RUN_SMOKE" = "1" ]; then
  echo "[STEP] smoke test (1 round, G0)"
  python run_cloud_batch.py \
    --rounds 1 \
    --groups G0 \
    --seeds 42 \
    --batch-name "_smoke_$(date +%Y%m%d-%H%M%S)" \
    --users-per-round 2 \
    --local-epochs 1 \
    --disable-attack
fi

mkdir -p logs
OUT_FILE="logs/nohup_${BATCH_NAME}.out"
PID_FILE="logs/nohup_${BATCH_NAME}.pid"

CMD=(python run_cloud_batch.py --rounds "$ROUNDS" --groups "$EXP_GROUPS" --seeds "$SEEDS" --batch-name "$BATCH_NAME")

echo "[STEP] start batch"
echo "[CMD] ${CMD[*]}"
nohup "${CMD[@]}" > "$OUT_FILE" 2>&1 &
echo "$!" > "$PID_FILE"

echo "[OK] started"
echo "[INFO] PID: $(cat "$PID_FILE")"
echo "[INFO] OUT: $OUT_FILE"
echo "[INFO] BATCH_DIR: logs/$BATCH_NAME"
echo "[INFO] tail: tail -f $OUT_FILE"
