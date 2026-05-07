import pickle

import torch
from torch.utils.data import Dataset

import random
import time
import os

from tqdm import tqdm


def _to_tensor(x, dtype):
    if torch.is_tensor(x):
        return x.clone().detach().to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)


def _get_split_meta_path(base_path):
    stem = base_path[:-3] if base_path.endswith(".pt") else base_path
    return f"{stem}.meta.pt"


def _get_split_field_path(base_path, field):
    stem = base_path[:-3] if base_path.endswith(".pt") else base_path
    return f"{stem}.{field}.pt"


def _get_reachability(data):
    if "reachability" in data:
        return _to_tensor(data["reachability"], torch.int64)
    if "distance" in data:
        return (_to_tensor(data["distance"], torch.int64) > 0).to(torch.int64)
    raise KeyError("sample requires 'reachability' or legacy 'distance'")


class NasbenchDataset(Dataset):
    _split_meta_cache = {}
    _split_columns_cache = {}
    _legacy_data_cache = {}

    def __init__(
        self,
        logger,
        dataset,
        part,
        data_path,
        percent=0,
        lbd_consistency=0,
        embed_type="",
        sample_method="random",
        n_clusters=5,
        runid=0,
    ):
        self.part = part
        self.dataset = dataset
        self.consistency = lbd_consistency
        self.data_path = data_path
        self.percent = percent
        self.sample_method = sample_method
        self.n_clusters = n_clusters
        self.cache_dir = (
            f"./data/{dataset}/rounds{runid}"  # Cache directory
        )
        self.embed_type = embed_type

        t0 = time.time()
        logger.info(f"Building dataset {self.part} from .pth file")
        self.data = self._load()
        t = time.time() - t0
        if self.consistency > 0:
            total_samples = sum(len(data) for data in self.data)
            logger.info(
                f"Finish Loading dataset {self.part}; Number of sample groups: {len(self.data)}; "
                f"Total samples: {total_samples}; Time: {t:.2f}s"
            )
        else:
            logger.info(
                f"Finish Loading dataset {self.part}; Number of samples: {len(self.data)}; Time: {t:.2f}s"
            )

    def _load(self):
        # Check if cached data exists
        cluster_suffix = f"_k{self.n_clusters}" if self.sample_method == "balanced_cluster" else ""
        embed_suffix = "" if self.embed_type in ("", "onehot_op") else f"_{self.embed_type}"
        cache_file = os.path.join(
            self.cache_dir,
            f"{self.part}_{self.percent}_{self.sample_method}{cluster_suffix}{embed_suffix}_cached_data.pth",
        )

        if os.path.exists(cache_file):
            # If cache file exists, load it
            print(f"Loading cached data from {cache_file}")
            cached_data = torch.load(cache_file, weights_only=False)
            if self._cache_is_compatible(cached_data):
                return cached_data
            print(f"Ignoring stale cached data from {cache_file}")

        # If cache file does not exist, process data
        data_file = self.data_path
        loaded_data = []
        split_meta_path = _get_split_meta_path(data_file)

        if os.path.exists(split_meta_path):
            if (
                data_file in self._split_meta_cache
                and data_file in self._split_columns_cache
            ):
                split_meta = self._split_meta_cache[data_file]
                columns = self._split_columns_cache[data_file]
                print(f"Using in-memory split cache for {data_file}")
            else:
                split_meta = torch.load(split_meta_path, weights_only=False)
                fields = split_meta["fields"]
                columns = {}
                for field in tqdm(
                    fields,
                    desc=f"{self.dataset} load fields",
                    unit="field",
                ):
                    field_path = _get_split_field_path(data_file, field)
                    field_data = torch.load(field_path, weights_only=False)
                    columns[field] = field_data
                self._split_meta_cache[data_file] = split_meta
                self._split_columns_cache[data_file] = columns

            fields = split_meta["fields"]
            total_num = int(split_meta["length"])
            data_num = (
                int(self.percent) if self.percent > 1 else int(total_num * self.percent)
            )
            data_num = min(max(data_num, 0), total_num)

            # 使用全局随机种子生成排列，保存/恢复状态确保train和val一致
            all_keys = list(range(total_num))
            state = random.getstate()
            random.shuffle(all_keys)
            random.setstate(state)

            if self.part == "train":
                filtered_keys = [k for k in all_keys if len(columns["ops"][k]) >= 6]
                print(f"[random过滤] 总样本: {total_num}, 节点数>=6: {len(filtered_keys)}, "
                        f"过滤掉: {total_num - len(filtered_keys)}")
                keys = sorted(filtered_keys[:data_num])
            elif self.part == "val":
                val_end = min(total_num, data_num + 1024)
                keys = sorted(all_keys[data_num:val_end])
            elif self.part == "test":
                keys = list(range(total_num))
            else:
                keys = []

            for key in keys:
                item = {field: columns[field][key] for field in fields}
                if self.part == "train" and self.consistency > 0:
                    loaded_data.append([item, item])
                else:
                    loaded_data.append(item)
        else:
            if data_file in self._legacy_data_cache:
                data_all = self._legacy_data_cache[data_file]
                print(f"Using in-memory legacy cache for {data_file}")
            else:
                data_all = torch.load(data_file, weights_only=False)
                self._legacy_data_cache[data_file] = data_all
            total_num = len(data_all)
            data_num = (
                int(self.percent) if self.percent > 1 else int(total_num * self.percent)
            )
            data_num = min(max(data_num, 0), total_num)

            # 使用全局随机种子生成排列，保存/恢复状态确保train和val一致
            all_keys = list(range(total_num))
            state = random.getstate()
            random.shuffle(all_keys)
            random.setstate(state)

            if self.part == "train":
                    # 过滤掉节点数2-5的小图，只保留节点数>=6的样本
                filtered_keys = [k for k in all_keys if len(data_all[k]["ops"]) >= 6]
                print(f"[random过滤] 总样本: {total_num}, 节点数>=6: {len(filtered_keys)}, "
                      f"过滤掉: {total_num - len(filtered_keys)}")
                keys = sorted(filtered_keys[:data_num])
            elif self.part == "val":
                val_end = min(total_num, data_num + 1024)
                keys = sorted(all_keys[data_num:val_end])
            elif self.part == "test":
                keys = list(range(total_num))  # test all
            else:
                keys = []

            for key in keys:
                if self.part == "train" and self.consistency > 0:
                    loaded_data.append([data_all[key], data_all[key]])
                elif (
                    (self.part == "train" and self.consistency == 0.0)
                    or self.part == "val"
                    or self.part == "test"
                ):
                    loaded_data.append(data_all[key])

        # Cache the processed data
        os.makedirs(self.cache_dir, exist_ok=True)
        torch.save(loaded_data, cache_file)
        print(f"Data cached to {cache_file}")

        return loaded_data

    def _cache_is_compatible(self, cached_data):
        split_meta_path = _get_split_meta_path(self.data_path)
        if not os.path.exists(split_meta_path) or not cached_data:
            return True

        split_meta = torch.load(split_meta_path, weights_only=False)
        required_fields = set(split_meta.get("fields", []))
        sample = cached_data[0]
        if isinstance(sample, (list, tuple)) and sample:
            sample = sample[0]
        if not isinstance(sample, dict):
            return True

        return required_fields.issubset(sample.keys())


    def __getitem__(self, index):
        if self.dataset == "nasbench101":
            return self.preprocess_101(self.data[index])
        elif self.dataset == "nasbench201":
            return self.preprocess_201(self.data[index])

    def preprocess_101(self, data):
        ops = _to_tensor(data["ops"], torch.int32)
        code = _to_tensor(data["code"], torch.float32)
        code_rel_pos = _to_tensor(data["code_rel_pos"], torch.int32)
        code_depth = _to_tensor(data["code_depth"], torch.float32)
        val_acc_avg = _to_tensor([data["validation_accuracy"]], torch.float32)
        test_acc_avg = _to_tensor([data["test_accuracy"]], torch.float32)

        op_depth = _to_tensor(data["op_depth"], torch.float32)
        in_degree = _to_tensor(data["in_degree"], torch.int64)
        out_degree = _to_tensor(data["out_degree"], torch.int64)
        reachability = _get_reachability(data)

        result = {
            "ops": ops,
            "code": code,
            "code_rel_pos": code_rel_pos,
            "code_adj": (code_rel_pos == 1).float(),
            "code_depth": code_depth,
            "val_acc_avg": val_acc_avg,
            "test_acc_avg": test_acc_avg,
            "op_depth": op_depth,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "reachability": reachability,
        }
        for key, value in data.items():
            if key.startswith("dir_pe_"):
                result[key] = _to_tensor(value, torch.float32)
        if "latency" in data:
            result["latency"] = _to_tensor([data["latency"]], torch.float32)
        return result

    def preprocess_201(self, data):
        ops = _to_tensor(data["ops"], torch.int32)
        code = _to_tensor(data["code"], torch.float32)
        code_rel_pos = _to_tensor(data["code_rel_pos"], torch.int32)
        code_depth = _to_tensor(data["code_depth"], torch.float32)
        val_acc_avg = _to_tensor([data["validation_accuracy"]], torch.float32) * 0.01
        test_acc_avg = _to_tensor([data["test_accuracy"]], torch.float32) * 0.01

        op_depth = _to_tensor(data["op_depth"], torch.float32)
        in_degree = _to_tensor(data["in_degree"], torch.int64)
        out_degree = _to_tensor(data["out_degree"], torch.int64)
        reachability = _get_reachability(data)
        result = {
            "ops": ops,
            "code": code,
            "code_rel_pos": code_rel_pos,
            "code_adj": (code_rel_pos == 1).float(),
            "code_depth": code_depth,
            "val_acc_avg": val_acc_avg,
            "test_acc_avg": test_acc_avg,
            "op_depth": op_depth,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "reachability": reachability,
        }
        if "latency" in data:
            result["latency"] = _to_tensor([data["latency"]], torch.float32)

        for key, value in data.items():
            if key.startswith("dir_pe_"):
                result[key] = _to_tensor(value, torch.float32)
        return result

    def __len__(self):
        return len(self.data)
