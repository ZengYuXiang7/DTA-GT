#!/usr/bin/env bash

set -e
cd "$(dirname "$0")"

# python preprocessing/gen_json_201.py
# ./generate_datasets.sh
# rm -rf ./data/nasbench201/rounds*/

PERCENTS="156 469 781 1563"
ROUNDS=3
warmup_step=0.10

# ============================================================
# accuracy 目标：main.py 会使用 val_acc_avg，并按 val tau 最大化保存 best
# ============================================================
for percent in $PERCENTS; do
  python main.py \
      --model          model56 \
      --dataset        nasbench201 \
      --percent        "$percent" \
      --warmup_step    "$warmup_step" \
      --graph_readout  att \
      --lambda_rank    0.2 \
      --rounds         "$ROUNDS" \
      --device         cuda \
      --patience       3000 \
      --predict_target accuracy
done
