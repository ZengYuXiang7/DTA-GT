"""Generate NASBench-101 JSON used by generate_data.py.

This is the only top-level NASBench-101 preprocessing entry. It reads the
raw TFRecord, joins the measured latency CSV by architecture hash, validates
that every hash matches, and writes data/nasbench101/nasbench101.json.
"""

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

from nasbench.api import ModelSpec, NASBench
from tqdm import tqdm


DEFAULT_TFRECORD = Path("data/nasbench101/nasbench_full.tfrecord")
DEFAULT_LATENCY_CSV = Path("data/nasbench101/nasbench101_latency.csv")
DEFAULT_OUTPUT = Path("data/nasbench101/nasbench101.json")
INPUT = "input"
OUTPUT = "output"
CONV1X1 = "conv1x1-bn-relu"
CONV3X3 = "conv3x3-bn-relu"
MAXPOOL3X3 = "maxpool3x3"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord", type=Path, default=DEFAULT_TFRECORD)
    parser.add_argument("--latency-csv", type=Path, default=DEFAULT_LATENCY_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epoch", type=int, default=108)
    parser.add_argument(
        "--check-fields",
        action="store_true",
        help="print available NASBench-101 source fields and exit",
    )
    parser.add_argument("--check-limit", type=int, default=1000)
    return parser.parse_args()


def load_latency_map(path: Path):
    latency_by_hash = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"hash", "latency_mean_ms"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        for row in reader:
            module_hash = row["hash"]
            if module_hash in latency_by_hash:
                raise ValueError(f"duplicate hash in {path}: {module_hash}")
            latency_by_hash[module_hash] = float(row["latency_mean_ms"])
    return latency_by_hash


def transform_operations_category(ops):
    transform_dict = {
        INPUT: 0,
        CONV1X1: 1,
        CONV3X3: 2,
        MAXPOOL3X3: 3,
        OUTPUT: 4,
    }
    return [transform_dict[k] for k in ops]


def check_source_fields(nasbench: NASBench, limit: int):
    fixed_fields = set()
    computed_fields = set()
    query_fields = set()

    hashes = list(nasbench.hash_iterator())
    for unique_hash in hashes[:limit]:
        fixed, computed = nasbench.get_metrics_from_hash(unique_hash)
        fixed_fields.update(fixed.keys())
        for repeats in computed.values():
            for repeat_data in repeats:
                computed_fields.update(repeat_data.keys())

    for unique_hash in hashes[: min(limit, 100)]:
        fixed, _ = nasbench.get_metrics_from_hash(unique_hash)
        model_spec = ModelSpec(fixed["module_adjacency"], fixed["module_operations"])
        query_fields.update(nasbench.query(model_spec, epochs=108).keys())

    print("fixed_metrics fields:", sorted(fixed_fields))
    print("computed_metrics fields:", sorted(computed_fields))
    print("query fields:", sorted(query_fields))


def build_item(nasbench: NASBench, unique_hash: str, epoch: int, latency_by_hash: dict):
    fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
    repeats = computed_metrics[epoch]
    if len(repeats) != 3:
        raise ValueError(f"expected 3 repeats at epoch {epoch}, got {len(repeats)}")

    val_acc_avg = sum(x["final_validation_accuracy"] for x in repeats) / len(repeats)
    test_acc_avg = sum(x["final_test_accuracy"] for x in repeats) / len(repeats)
    training_time_avg = sum(x["final_training_time"] for x in repeats) / len(repeats)

    return {
        "hash": unique_hash,
        "test_accuracy": test_acc_avg,
        "validation_accuracy": val_acc_avg,
        "module_adjacency": fixed_metrics["module_adjacency"].tolist(),
        "module_operations": transform_operations_category(
            fixed_metrics["module_operations"]
        ),
        "parameters": fixed_metrics["trainable_parameters"],
        "training_time": training_time_avg,
        "latency": latency_by_hash[unique_hash],
    }


def gen_json_file(tfrecord: Path, latency_csv: Path, output: Path, epoch: int):
    nasbench = NASBench(str(tfrecord))
    latency_by_hash = load_latency_map(latency_csv)

    hashes = list(nasbench.hash_iterator())
    nas_hashes = set(hashes)
    latency_hashes = set(latency_by_hash)
    missing = nas_hashes - latency_hashes
    extra = latency_hashes - nas_hashes
    if missing or extra:
        raise ValueError(
            "latency csv hash mismatch: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    data_dict = OrderedDict()
    for i, unique_hash in enumerate(tqdm(hashes, desc="nasbench101 json")):
        data_dict[str(i)] = build_item(nasbench, unique_hash, epoch, latency_by_hash)

    with output.open("w") as f:
        json.dump(data_dict, f)

    print(f"wrote {output} ({len(data_dict)} samples)")


def main():
    args = parse_args()
    if args.check_fields:
        nasbench = NASBench(str(args.tfrecord))
        check_source_fields(nasbench, args.check_limit)
        return
    gen_json_file(args.tfrecord, args.latency_csv, args.output, args.epoch)


if __name__ == "__main__":
    main()
