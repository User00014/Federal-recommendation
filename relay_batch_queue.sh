#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/Federated_Privacy_Project}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
CHECK_INTERVAL="${CHECK_INTERVAL:-45}"
QUEUE_TAG="${QUEUE_TAG:-relay_privacy_expand}"

cd "$PROJECT_DIR"

PY_BIN="${PY_BIN:-$PROJECT_DIR/.venv/bin/python}"
if [ ! -x "$PY_BIN" ]; then
  PY_BIN="python3"
fi

META_DIR="$PROJECT_DIR/logs/${QUEUE_TAG}_meta"
mkdir -p "$META_DIR"
DISPATCH_CSV="$META_DIR/dispatch.csv"
if [ ! -f "$DISPATCH_CSV" ]; then
  echo "queued_base,batch_name,groups,seeds,rounds,launched_at,pid" > "$DISPATCH_CSV"
fi

JOBS=(
  "full15_seed62|G0,G1,G2,G3,G4,G5,G6,G7,G8,G9,A1,A2,A3,A4L,A4H|62|1000"
  "privacy_scan_seed42|P1,P2,P3,P4,P5,P6|42|1000"
  "privacy_scan_seed52|P1,P2,P3,P4,P5,P6|52|1000"
)

running_count() {
  pgrep -af "python run_cloud_batch.py" | wc -l | tr -d ' '
}

unique_batch_name() {
  local base="$1"
  local ts
  local cand
  while true; do
    ts="$(date +%Y%m%d-%H%M%S)"
    cand="${base}_${ts}"
    if [ ! -e "$PROJECT_DIR/logs/$cand" ] && [ ! -e "$PROJECT_DIR/logs/nohup_${cand}.out" ]; then
      echo "$cand"
      return
    fi
    sleep 1
  done
}

echo "[QUEUE] start tag=$QUEUE_TAG max_parallel=$MAX_PARALLEL check_interval=${CHECK_INTERVAL}s python=$PY_BIN"

for spec in "${JOBS[@]}"; do
  IFS='|' read -r base groups seeds rounds <<< "$spec"

  while true; do
    rc="$(running_count)"
    if [ "$rc" -lt "$MAX_PARALLEL" ]; then
      break
    fi
    echo "[QUEUE] waiting slot... current_running=$rc target<$MAX_PARALLEL"
    sleep "$CHECK_INTERVAL"
  done

  batch_name="$(unique_batch_name "$base")"
  out_file="$PROJECT_DIR/logs/nohup_${batch_name}.out"

  nohup "$PY_BIN" run_cloud_batch.py \
    --rounds "$rounds" \
    --groups "$groups" \
    --seeds "$seeds" \
    --batch-name "$batch_name" > "$out_file" 2>&1 &
  pid="$!"

  echo "$base,$batch_name,$groups,$seeds,$rounds,$(date '+%F %T'),$pid" >> "$DISPATCH_CSV"
  echo "[QUEUE] launched base=$base batch=$batch_name pid=$pid"
  sleep 2
  tail -n 3 "$out_file" || true

done

echo "[QUEUE] all queued jobs dispatched."

echo "[QUEUE] monitor active run_cloud_batch until all done..."
while true; do
  rc="$(running_count)"
  echo "[QUEUE] running_count=$rc"
  if [ "$rc" -eq 0 ]; then
    echo "[QUEUE] all training jobs finished."
    break
  fi
  sleep 120
done
