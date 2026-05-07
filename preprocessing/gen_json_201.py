"""Generate NASBench-201 JSON used by generate_data.py."""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from nas_201_api import NASBench201API as API
from tqdm import tqdm


DEFAULT_API_PATH = Path("data/nasbench201/NAS-Bench-201-v1_1-096897.pth")
DEFAULT_OUTPUT = Path("data/nasbench201/nasbench201.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-path", type=Path, default=DEFAULT_API_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", type=str, default="cifar10_valid_converged")
    return parser.parse_args()


def train_and_eval(api, arch_index, nepoch=None, dataname=None, use_converged_lr=True):
    if dataname == "cifar10":
        raise ValueError("Do not allow cifar10 dataset")

    if use_converged_lr and dataname == "cifar10-valid":
        if nepoch is not None:
            raise ValueError("use_converged_lr=True requires nepoch=None")
        info = api.get_more_info(arch_index, dataname, None, "200", True)
        valid_acc = info["valid-accuracy"]
        time_cost = info["train-all-time"] + info["valid-per-time"]
        valid_acc_avg = api.get_more_info(
            arch_index, "cifar10-valid", None, "200", False
        )["valid-accuracy"]
        test_acc = api.get_more_info(arch_index, "cifar10", None, "200", True)[
            "test-accuracy"
        ]
        test_acc_avg = api.get_more_info(
            arch_index, "cifar10", None, "200", False
        )["test-accuracy"]
        return valid_acc, valid_acc_avg, time_cost, test_acc, test_acc_avg

    raise ValueError("Only cifar10_valid_converged is implemented")


def info2mat(api, arch_index: int, dataset: str):
    if dataset != "cifar10_valid_converged":
        raise ValueError("Only cifar10_valid_converged is implemented")

    arch_str = api.meta_archs[arch_index]
    ops = {
        "input": 0,
        "nor_conv_1x1": 1,
        "nor_conv_3x3": 2,
        "avg_pool_3x3": 3,
        "skip_connect": 4,
        "none": 5,
        "output": 6,
    }
    adj_mat = np.array(
        [
            [0, 1, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    nodes = ["input"]
    steps = arch_str.split("+")
    steps_coding = ["0", "0", "1", "0", "1", "2"]
    cont = 0
    for step in steps:
        for node in step.strip("|").split("|"):
            op_name, idx = node.split("~")
            if idx != steps_coding[cont]:
                raise ValueError(f"unexpected NASBench-201 arch format: {arch_str}")
            cont += 1
            nodes.append(op_name)
    nodes.append("output")

    ops_idx = [ops[k] for k in nodes]
    valid_acc, val_acc_avg, time_cost, test_acc, test_acc_avg = train_and_eval(
        api,
        arch_index,
        nepoch=None,
        dataname="cifar10-valid",
        use_converged_lr=True,
    )
    return {
        "test_accuracy": test_acc,
        "test_accuracy_avg": test_acc_avg,
        "validation_accuracy": valid_acc,
        "validation_accuracy_avg": val_acc_avg,
        "module_adjacency": adj_mat.tolist(),
        "module_operations": ops_idx,
        "training_time": time_cost,
        "latency": api.get_latency(arch_index, "cifar10-valid"),
    }


def gen_json_file(api_path: Path, output: Path, dataset: str):
    api = API(str(api_path))
    data_dict = OrderedDict()
    for index in tqdm(range(len(api)), desc="nasbench201 json"):
        data_dict[str(index)] = info2mat(api, index, dataset)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(data_dict, f)
    print(f"wrote {output} ({len(data_dict)} samples)")


def main():
    args = parse_args()
    gen_json_file(args.api_path, args.output, args.dataset)


if __name__ == "__main__":
    main()
