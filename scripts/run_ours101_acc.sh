#!/usr/bin/env bash

set -e
cd "$(dirname "$0")"

# 如需重新生成 101 accuracy 数据，先执行：
# ./generate_datasets.sh
# rm -rf ./data/nasbench101/rounds*/

PERCENTS="100 172 424 4236"
# PERCENTS="4236"
ROUNDS=3
warmup_step=0.10

for percent in $PERCENTS; do
  python main.py \
      --model          model56 \
      --dataset        nasbench101 \
      --percent        "$percent" \
      --warmup_step    "$warmup_step" \
      --d_model        180 \
      --gcn_layers     10 \
      --graph_n_head   6 \
      --graph_readout  att \
      --lambda_mse     1.0 \
      --lambda_rank    0.2 \
      --lambda_consistency 0.0 \
      --rounds         "$ROUNDS" \
      --device         cuda \
      --patience       3000 \
      --predict_target accuracy
done
