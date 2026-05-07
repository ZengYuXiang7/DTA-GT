#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

FORCE_JSON="${FORCE_JSON:-0}"
VALIDATE_DATASET="${VALIDATE_DATASET:-1}"
VALIDATION_RUNID="${VALIDATION_RUNID:-99999}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[Missing] $path"
    exit 1
  fi
}

maybe_generate_json() {
  local dataset="$1"
  local output="$2"
  shift 2

  if [[ "$FORCE_JSON" == "1" || ! -f "$output" ]]; then
    echo "[JSON] generating ${dataset}: $output"
    "$@"
  else
    echo "[JSON] exists, skip ${dataset}: $output"
  fi
}

echo "========== Raw File Check =========="
require_file "data/nasbench101/nasbench_full.tfrecord"
require_file "data/nasbench101/nasbench101_latency.csv"
require_file "data/nasbench201/NAS-Bench-201-v1_1-096897.pth"

echo
echo "========== JSON Generation =========="
maybe_generate_json \
  "nasbench101" \
  "data/nasbench101/nasbench101.json" \
  python preprocessing/gen_json_101.py

maybe_generate_json \
  "nasbench201" \
  "data/nasbench201/nasbench201.json" \
  python preprocessing/gen_json_201.py

echo
echo "========== Split-Field Generation =========="
python generate_data.py --dataset all

echo
echo "========== Clear DataLoader Cache =========="
rm -rf ./data/nasbench101/rounds*/
rm -rf ./data/nasbench201/rounds*/

if [[ "$VALIDATE_DATASET" == "1" ]]; then
  echo
  echo "========== Dataset Usability Check =========="
  python - <<PY
import logging
import os

import torch

from datasets.nasbench.dataset import NasbenchDataset

embed_type = "onehot_op"
runid = ${VALIDATION_RUNID@Q}

required = {
    "index",
    "adj",
    "ops",
    "validation_accuracy",
    "test_accuracy",
    "code",
    "code_rel_pos",
    "code_depth",
    "op_depth",
    "reachability",
    "in_degree",
    "out_degree",
    "dir_pe_rw",
    "dir_pe_ml",
    "dir_pe_ml2",
    "latency",
}

logger = logging.getLogger("dataset_check")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

for dataset in ("nasbench101", "nasbench201"):
    base = f"data/{dataset}/all_{dataset}.pt"
    meta_path = base[:-3] + ".meta.pt"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)

    meta = torch.load(meta_path, weights_only=False)
    fields = set(meta["fields"])
    missing = sorted(required - fields)
    if missing:
        raise RuntimeError(f"{dataset}: missing split fields: {missing}")

    norm = meta.get("norm_params", {})
    if "latency" not in norm:
        raise RuntimeError(f"{dataset}: missing latency norm_params")

    for field in sorted(required):
        field_path = base[:-3] + f".{field}.pt"
        if not os.path.exists(field_path):
            raise FileNotFoundError(field_path)

    train = NasbenchDataset(
        logger,
        dataset,
        "train",
        base,
        percent=10,
        embed_type=embed_type,
        runid=runid,
    )
    val = NasbenchDataset(
        logger,
        dataset,
        "val",
        base,
        percent=10,
        embed_type=embed_type,
        runid=runid,
    )

    train_sample = train[0]
    val_sample = val[0]
    for part, sample in (("train", train_sample), ("val", val_sample)):
        needed = {
            "ops",
            "code_adj",
            "op_depth",
            "reachability",
            "in_degree",
            "out_degree",
            "dir_pe_ml",
            "dir_pe_ml2",
            "latency",
        }
        missing_sample = sorted(needed - set(sample))
        if missing_sample:
            raise RuntimeError(f"{dataset} {part}: missing sample keys: {missing_sample}")

    print(
        f"[OK] {dataset}: length={meta['length']} "
        f"train={len(train)} val={len(val)} fields={len(fields)}"
    )

print("[OK] dataset check passed")
PY
fi

echo
echo "========== Done =========="
echo "Generated split-field datasets for nasbench101 and nasbench201."
echo "Use FORCE_JSON=1 to regenerate JSON, or VALIDATE_DATASET=0 to skip dataset checks."
