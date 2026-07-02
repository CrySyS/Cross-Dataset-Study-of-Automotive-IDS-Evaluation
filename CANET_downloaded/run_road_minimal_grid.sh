#!/usr/bin/env bash
set -euo pipefail

# Minimal, non-intrusive CANet ROAD tuning runner:
# 1) Train each config for N epochs
# 2) Select best epoch by minimum validation loss from log
# 3) Run test with best epoch
# 4) Score with quantile sweep
# 5) Build a compact summary CSV

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
VENV_PY="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: Python not found at $VENV_PY"
  exit 1
fi

DATASET_DIR="${SCRIPT_DIR}/../data_raw/02_Road_cleaned/signal_extractions"
MODEL_DIR="${SCRIPT_DIR}/../../models/CANET_ROAD_V3"
RESULTS_DIR="${ROOT_DIR}/Results"
SELECTION_CONFIG="${SCRIPT_DIR}/road_signal_selection.json"
ID_CONFIG="${SCRIPT_DIR}/road_id_config_v3.json"
RUNS_DIR="${RUNS_DIR_OVERRIDE:-${RESULTS_DIR}/road_minimal_grid_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUNS_DIR"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-250}"
TRAIN_SLICES="${TRAIN_SLICES:-10}"
VALID_SLICES="${VALID_SLICES:-1}"
VALID_PROP="${VALID_PROP:-0.05}"
WINDOW_SIZE_BASE="${WINDOW_SIZE_BASE:-1}"

# Keep this intentionally small and practical.
CONFIGS=(
  "lr1e4_h5_w1 1e-4 5 1"
  "lr3e4_h5_w1 3e-4 5 1"
  "lr5e5_h5_w1 5e-5 5 1"
  "lr1e4_h8_w1 1e-4 8 1"
)

if [[ "${INCLUDE_WINDOW2:-0}" == "1" ]]; then
  CONFIGS+=("lr1e4_h5_w2 1e-4 5 2")
fi

QUANTILES=(0.95 0.99 0.995 0.999)

SUMMARY_CSV="${RUNS_DIR}/summary.csv"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  printf "run_name,experiment_id,best_epoch,best_val_loss,lr,hidden_size,window_size,quantile,avg_tpr,avg_tnr,avg_f1,macro_auc\n" > "$SUMMARY_CSV"
fi

if [[ -n "${CONFIG_FILTER:-}" ]]; then
  FILTERED_CONFIGS=()
  for cfg in "${CONFIGS[@]}"; do
    read -r RUN_NAME _ <<<"$cfg"
    if [[ ",${CONFIG_FILTER}," == *",${RUN_NAME},"* ]]; then
      FILTERED_CONFIGS+=("$cfg")
    fi
  done
  CONFIGS=("${FILTERED_CONFIGS[@]}")
fi

echo "Runs directory: $RUNS_DIR"

auto_select_best_epoch() {
  local log_file="$1"
  "$VENV_PY" - <<'PY' "$log_file"
import re, sys
from math import inf

log_path = sys.argv[1]
pat = re.compile(r"Epoch\s+(\d+)\s+validation loss sum:\s+([0-9eE+\-.]+)")
best_epoch = None
best_val = inf

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pat.search(line)
        if not m:
            continue
        e = int(m.group(1))
        v = float(m.group(2))
        if v < best_val:
            best_val = v
            best_epoch = e

if best_epoch is None:
    print("ERROR: no validation loss lines found", file=sys.stderr)
    sys.exit(2)

print(f"{best_epoch},{best_val:.12g}")
PY
}

extract_experiment_id() {
  local log_file="$1"
  local exp
  exp="$(grep -oE 'Saved weights: .*/RoadV3_[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}_epoch[0-9]+' "$log_file" | tail -1 | sed -E 's|.*RoadV3_([0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2})_epoch[0-9]+|\1|')"
  if [[ -z "$exp" ]]; then
    echo "ERROR: could not extract experiment_id from $log_file" >&2
    return 2
  fi
  echo "$exp"
}

append_metrics_to_summary() {
  local run_name="$1"
  local exp_id="$2"
  local best_epoch="$3"
  local best_val_loss="$4"
  local lr="$5"
  local hidden="$6"
  local window="$7"
  local q="$8"
  local metrics_json="$9"

  "$VENV_PY" - <<'PY' "$SUMMARY_CSV" "$run_name" "$exp_id" "$best_epoch" "$best_val_loss" "$lr" "$hidden" "$window" "$q" "$metrics_json"
import csv, json, math, statistics, sys

summary_csv, run_name, exp_id, best_epoch, best_val_loss, lr, hidden, window, q, metrics_json = sys.argv[1:]

with open(metrics_json, "r", encoding="utf-8") as f:
    j = json.load(f)

def safe_float(v):
    try:
        x = float(v)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None

# v3 writes averages under "averages_all"; keep backward compatibility as fallback.
avgs = j.get("averages_all") or j.get("averages", {})
avg_tpr = safe_float(avgs.get("avg_TPR")) or float("nan")
avg_tnr = safe_float(avgs.get("avg_TNR")) or float("nan")
avg_f1  = safe_float(avgs.get("avg_F1"))  or float("nan")
mac_auc = safe_float(avgs.get("avg_AUC")) or float("nan")

with open(summary_csv, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        run_name,
        exp_id,
        best_epoch,
        best_val_loss,
        lr,
        hidden,
        window,
        q,
        f"{avg_tpr:.6g}" if not math.isnan(avg_tpr) else "",
        f"{avg_tnr:.6g}" if not math.isnan(avg_tnr) else "",
        f"{avg_f1:.6g}" if not math.isnan(avg_f1) else "",
        f"{mac_auc:.6g}" if not math.isnan(mac_auc) else "",
    ])
PY
}

for cfg in "${CONFIGS[@]}"; do
  read -r RUN_NAME LR HIDDEN WINDOW <<<"$cfg"
  if [[ "$WINDOW" == "1" ]]; then
    WINDOW="$WINDOW_SIZE_BASE"
  fi

  RUN_DIR="${RUNS_DIR}/${RUN_NAME}"
  PRED_DIR="${RUN_DIR}/predictions"
  mkdir -p "$RUN_DIR"
  mkdir -p "$PRED_DIR"
  TRAIN_LOG="${RUN_DIR}/train.log"

  echo
  echo "=== TRAIN ${RUN_NAME} (lr=${LR}, hidden=${HIDDEN}, window=${WINDOW}) ==="
  "$VENV_PY" "${SCRIPT_DIR}/canet_road_v3_train.py" \
    --dataset-dir "$DATASET_DIR" \
    --model-dir "$MODEL_DIR" \
    --selection-config "$SELECTION_CONFIG" \
    --id-config "$ID_CONFIG" \
    --id-mps-source fixed \
    --epochs "$EPOCHS" \
    --learning-rate "$LR" \
    --hidden-size "$HIDDEN" \
    --window-size "$WINDOW" \
    --batch-size "$BATCH_SIZE" \
    --train-slices "$TRAIN_SLICES" \
    --valid-slices "$VALID_SLICES" \
    --valid-prop "$VALID_PROP" \
    2>&1 | tee "$TRAIN_LOG"

  EXP_ID="$(extract_experiment_id "$TRAIN_LOG")"
  BEST_INFO="$(auto_select_best_epoch "$TRAIN_LOG")"
  BEST_EPOCH="${BEST_INFO%%,*}"
  BEST_VAL_LOSS="${BEST_INFO##*,}"

  echo "Selected best epoch: ${BEST_EPOCH} (val_loss=${BEST_VAL_LOSS})"
  echo "Experiment ID: ${EXP_ID}"

  echo "=== TEST ${RUN_NAME} @ epoch ${BEST_EPOCH} ==="
  "$VENV_PY" "${SCRIPT_DIR}/canet_road_v3_test.py" \
    --dataset-dir "$DATASET_DIR" \
    --model-dir "$MODEL_DIR" \
    --results-dir "$PRED_DIR" \
    --experiment-id "$EXP_ID" \
    --epoch "$BEST_EPOCH" \
    --batch-size "$BATCH_SIZE" \
    --window-size "$WINDOW" \
    --valid-prop "$VALID_PROP" \
    2>&1 | tee "${RUN_DIR}/test.log"

  for q in "${QUANTILES[@]}"; do
    q_tag="q$(echo "$q" | tr -d '.')"
    OUT_CSV="${RUN_DIR}/metrics_${q_tag}.csv"
    OUT_JSON="${RUN_DIR}/metrics_${q_tag}_summary.json"

    echo "=== SCORE ${RUN_NAME} quantile=${q} ==="
    "$VENV_PY" "${SCRIPT_DIR}/canet_road_v3_score.py" \
      --results-dir "$PRED_DIR" \
      --experiment-id "$EXP_ID" \
      --quantile "$q" \
      --out-csv "$OUT_CSV" \
      --out-json "$OUT_JSON" \
      2>&1 | tee -a "${RUN_DIR}/score.log"

    append_metrics_to_summary "$RUN_NAME" "$EXP_ID" "$BEST_EPOCH" "$BEST_VAL_LOSS" "$LR" "$HIDDEN" "$WINDOW" "$q" "$OUT_JSON"
  done

done

echo
"$VENV_PY" - <<'PY' "$SUMMARY_CSV"
import pandas as pd, sys
p = sys.argv[1]
df = pd.read_csv(p)
print("Wrote:", p)
if not df.empty:
    print("Top by avg_tpr then macro_auc:")
    print(df.sort_values(["avg_tpr", "macro_auc"], ascending=False).head(10).to_string(index=False))
PY
