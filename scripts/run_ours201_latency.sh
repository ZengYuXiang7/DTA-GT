#!/usr/bin/env bash

set -e
cd "$(dirname "$0")"

# ./generate_datasets.sh
# rm -rf ./data/nasbench201/rounds*/

PERCENTS="156 469 781 1563"
ROUNDS=3
warmup_step=0.10

# ============================================================
# 阶段1：结构探索 gcn_layers=10, rank=0.8 → 只跑 att
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
      --predict_target latency
done
