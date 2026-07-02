#!/usr/bin/env bash
set -euo pipefail

# Clean rerun for CANShield ROAD + CrySyS.
# - Archives old artifacts to avoid stale reuse.
# - Runs train/eval/visualization for each dataset.
# - Prints minimal progress to console.
# - Logs full command output to a timestamped file.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_PY=".venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PY}"

TS="$(date +%Y-%m-%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/canshield_clean_rerun_${TS}.log"

ARCHIVE_DIR="$ROOT_DIR/archive_clean_rerun_${TS}"
mkdir -p "$ARCHIVE_DIR"

DATASETS=(road crysys)

log() {
  local msg="$1"
  echo "[$(date +%H:%M:%S)] $msg" | tee -a "$LOG_FILE"
}

run_cmd() {
  local label="$1"
  shift
  log "START: $label"
  "$@" >>"$LOG_FILE" 2>&1
  log "DONE : $label"
}

move_if_exists() {
  local src="$1"
  local dst_dir="$2"
  local dst_name="$3"
  if [[ -e "$src" ]]; then
    mkdir -p "$dst_dir"
    local dst_path="$dst_dir/$dst_name"
    if [[ -e "$dst_path" ]]; then
      dst_path="${dst_path}_$(date +%s)"
    fi
    mv "$src" "$dst_path"
    log "ARCHIVE: moved $src -> $dst_path"
  fi
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  log "ERROR: Python executable not found or not executable: $PYTHON_BIN"
  exit 1
fi

cd "$ROOT_DIR"
log "Root directory: $ROOT_DIR"
log "Python: $PYTHON_BIN"
log "Log file: $LOG_FILE"
log "Archive directory: $ARCHIVE_DIR"

for ds in "${DATASETS[@]}"; do
  log "Preparing clean state for dataset: $ds"
  DST="$ARCHIVE_DIR/$ds"

  move_if_exists "$ROOT_DIR/artifacts/models/$ds" "$DST" "artifacts_models_${ds}"
  move_if_exists "$ROOT_DIR/artifacts/model_ckpts/$ds" "$DST" "artifacts_model_ckpts_${ds}"
  move_if_exists "$ROOT_DIR/artifacts/histories/$ds" "$DST" "artifacts_histories_${ds}"
  move_if_exists "$ROOT_DIR/artifacts/visualize/$ds" "$DST" "artifacts_visualize_${ds}"
  move_if_exists "$ROOT_DIR/artifacts/debug_inputs/$ds" "$DST" "artifacts_debug_inputs_${ds}"
  move_if_exists "$ROOT_DIR/artifacts/reconstruction_plots/$ds" "$DST" "artifacts_reconstruction_plots_${ds}"

  move_if_exists "$ROOT_DIR/plots/$ds" "$DST" "plots_${ds}"

  move_if_exists "$ROOT_DIR/data/thresholds/$ds" "$DST" "data_thresholds_${ds}"
  move_if_exists "$ROOT_DIR/data/label/$ds" "$DST" "data_label_${ds}"
  move_if_exists "$ROOT_DIR/data/results/$ds" "$DST" "data_results_${ds}"
  move_if_exists "$ROOT_DIR/data/prediction/${ds}_original" "$DST" "data_prediction_${ds}_original"
  move_if_exists "$ROOT_DIR/data/prediction/${ds}_lite" "$DST" "data_prediction_${ds}_lite"
done

for ds in "${DATASETS[@]}"; do
  run_cmd "train $ds" bash -lc "cd '$ROOT_DIR/src' && '$PYTHON_BIN' run_development_canshield.py --config-name '$ds'"
  run_cmd "evaluate $ds" bash -lc "cd '$ROOT_DIR/src' && '$PYTHON_BIN' run_evaluation_canshield.py --config-name '$ds'"
  run_cmd "visualize $ds" bash -lc "cd '$ROOT_DIR/src' && '$PYTHON_BIN' run_visualization_results.py --config-name '$ds'"
done

for ds in "${DATASETS[@]}"; do
  log "Output check for dataset: $ds"
  for p in \
    "$ROOT_DIR/artifacts/models/$ds" \
    "$ROOT_DIR/data/thresholds/$ds" \
    "$ROOT_DIR/data/prediction/${ds}_original" \
    "$ROOT_DIR/data/results/$ds" \
    "$ROOT_DIR/plots/$ds"; do
    if [[ -d "$p" ]]; then
      count="$(find "$p" -type f | wc -l | tr -d ' ')"
      log "OK   : $p (files=$count)"
    else
      log "MISS : $p"
    fi
  done
done

log "All done. Full logs: $LOG_FILE"
