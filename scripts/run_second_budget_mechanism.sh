#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="results/diagnostics_second_budget"
LOG_DIR="${OUT_ROOT}/logs"
RUN_INDEX="${OUT_ROOT}/run_index.tsv"
STATUS_FILE="${OUT_ROOT}/status.tsv"

mkdir -p "$LOG_DIR"

printf "setting_key\tlabel\tdataset\ttarget\tpercent\trun_dir\tdiag_dir\n" > "$RUN_INDEX"
printf "time\tsetting_key\tstage\tstatus\tdetail\n" > "$STATUS_FILE"

log_status() {
  local key="$1"
  local stage="$2"
  local status="$3"
  local detail="${4:-}"
  printf "%s\t%s\t%s\t%s\t%s\n" "$(date '+%F %T')" "$key" "$stage" "$status" "$detail" | tee -a "$STATUS_FILE"
}

latest_run_dir() {
  local dataset="$1"
  find "results/${dataset}/model56" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
    | sort -nr \
    | awk 'NR==1 {print $2}'
}

assert_three_runs() {
  local run_dir="$1"
  python - "$run_dir" <<'PY'
import json
import os
import sys

run_dir = sys.argv[1]
summary = os.path.join(run_dir, "summary_all_runs.json")
if not os.path.exists(summary):
    raise SystemExit(f"missing summary_all_runs.json in {run_dir}")
rows = json.load(open(summary))
if len(rows) != 3:
    raise SystemExit(f"expected 3 runs in {summary}, found {len(rows)}")
missing = [
    rid for rid in range(3)
    if not os.path.exists(os.path.join(run_dir, "checkpoints", f"best_model_run{rid}.pt"))
]
if missing:
    raise SystemExit(f"missing best checkpoints for runs {missing} in {run_dir}")
print(run_dir)
PY
}

run_training() {
  local key="$1"
  local label="$2"
  local dataset="$3"
  local target="$4"
  local percent="$5"
  local train_log="${LOG_DIR}/${key}_train.log"

  log_status "$key" "train" "start" "${label}"
  {
    echo "[Command]"
    echo "python main.py --model model56 --dataset ${dataset} --percent ${percent} --warmup_step 0.10 --d_model 180 --gcn_layers 10 --graph_n_head 6 --graph_readout att --lambda_mse 1.0 --lambda_rank 0.2 --lambda_consistency 0.0 --rounds 3 --device cuda --patience 3000 --predict_target ${target}"
    echo
    python main.py \
      --model model56 \
      --dataset "$dataset" \
      --percent "$percent" \
      --warmup_step 0.10 \
      --d_model 180 \
      --gcn_layers 10 \
      --graph_n_head 6 \
      --graph_readout att \
      --lambda_mse 1.0 \
      --lambda_rank 0.2 \
      --lambda_consistency 0.0 \
      --rounds 3 \
      --device cuda \
      --patience 3000 \
      --predict_target "$target"
  } > "$train_log" 2>&1

  local run_dir
  run_dir="$(latest_run_dir "$dataset")"
  assert_three_runs "$run_dir" >> "$train_log"
  log_status "$key" "train" "done" "$run_dir"

  local diag_dir="${OUT_ROOT}/${key}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$key" "$label" "$dataset" "$target" "$percent" "$run_dir" "$diag_dir" >> "$RUN_INDEX"
}

run_diagnostics_for_index() {
  tail -n +2 "$RUN_INDEX" | while IFS=$'\t' read -r key label dataset target percent run_dir diag_dir; do
    local diag_log="${LOG_DIR}/${key}_diagnostics.log"
    log_status "$key" "diagnostics" "start" "$diag_dir"
    python analysis/directional_diagnostics.py \
      --run_dir "$run_dir" \
      --runs 0 1 2 \
      --split test \
      --max_samples 0 \
      --device cuda \
      --out_dir "$diag_dir" > "$diag_log" 2>&1
    log_status "$key" "diagnostics" "done" "$diag_dir"
  done
}

run_training "NB101_Acc_172" "NB101 Acc @172" "nasbench101" "accuracy" "172"
run_training "NB101_Lat_172" "NB101 Lat @172" "nasbench101" "latency" "172"
run_training "NB201_Acc_469" "NB201 Acc @469" "nasbench201" "accuracy" "469"
run_training "NB201_Lat_469" "NB201 Lat @469" "nasbench201" "latency" "469"

run_diagnostics_for_index

log_status "ALL" "aggregate" "start" "${OUT_ROOT}/SECOND_BUDGET_MECHANISM_ANALYSIS.md"
python analysis/aggregate_mechanism_diagnostics.py \
  --root "$OUT_ROOT" \
  --title "Second-Budget DTA-GT Mechanism Diagnostics Handoff" \
  --report-name "SECOND_BUDGET_MECHANISM_ANALYSIS.md" \
  > "${LOG_DIR}/aggregate.log" 2>&1
log_status "ALL" "aggregate" "done" "${OUT_ROOT}/SECOND_BUDGET_MECHANISM_ANALYSIS.md"

echo "Second-budget mechanism diagnostics completed."
echo "Report: ${OUT_ROOT}/SECOND_BUDGET_MECHANISM_ANALYSIS.md"
