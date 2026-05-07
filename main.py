"""
Graph Regression 训练入口
支持多轮实验编排，结构化日志输出，best model 保存与测试评估
"""

import os
import shutil
import time
import json
import pickle
import collections
import argparse
import sys
import torch
import numpy as np
from datetime import datetime

from training import (
    Metric,
    auto_load_model,
    init_layers,
    init_optim,
    save_check_point,
)
from datasets import init_dataloader
from utils.paths import METRICS_DIR, PROJECT_RUN_LOG, TAU_DIR, PathManager
from utils.logger import get_logger
from utils.seed import set_seed
from utils.grad_norm import compute_grad_norm
from utils.visualization import plot_training_curves
from utils.summary import (
    write_config_json,
    append_summary_json,
    append_runlog_aggregate,
    print_training_summary,
    print_test_results,
)
from config import parse_args

torch.set_default_dtype(torch.float32)


def merge_config_into_args(
    args, config, *, only_existing=False, skip_none=True, verbose=False
):
    """
    用 config 覆盖 args，并打印：
      - overwritten: args 原来有该字段且值发生变化
      - added: args 原来没有该字段，新添加
      - unchanged: args 原来有该字段，但值相同（可选打印）
      - skipped_none: config 中为 None 被跳过（可选打印）
      - skipped_missing: only_existing=True 且 args 没有该字段被跳过（可选打印）
    """

    def to_dict(x):
        if isinstance(x, dict):
            return x
        if hasattr(x, "__dict__"):
            return vars(x)
        raise TypeError(f"Unsupported type: {type(x)}")

    def has_key(obj, k):
        return (k in obj) if isinstance(obj, dict) else hasattr(obj, k)

    def get_val(obj, k, default=None):
        return obj.get(k, default) if isinstance(obj, dict) else getattr(obj, k, default)

    def set_val(obj, k, v):
        if isinstance(obj, dict):
            obj[k] = v
        else:
            setattr(obj, k, v)

    cdict = to_dict(config)

    overwritten = []  # (k, old, new)
    added = []  # (k, new)
    unchanged = []  # (k, val)
    skipped_none = []  # (k)
    skipped_missing = []  # (k)

    for k, v in cdict.items():
        if skip_none and v is None:
            skipped_none.append(k)
            continue

        exists = has_key(args, k)
        if only_existing and not exists:
            skipped_missing.append(k)
            continue

        if exists:
            old = get_val(args, k)
            if old != v:
                overwritten.append((k, old, v))
            else:
                unchanged.append((k, v))
        else:
            added.append((k, v))

        set_val(args, k, v)

    if verbose:
        if overwritten:
            print(f"[merge] overwritten ({len(overwritten)}):")
            for k, old, new in overwritten:
                print(f"  - {k}: {old} -> {new}")
        else:
            print("[merge] overwritten (0)")

        if added:
            print(f"[merge] added ({len(added)}):")
            for k, new in added:
                print(f"  + {k}: {new}")
        else:
            print("[merge] added (0)")

    return args


def remove_pycache_dirs(project_root=None):
    root = project_root or os.path.dirname(os.path.abspath(__file__))
    removed = []
    for current_root, dirnames, _ in os.walk(root, topdown=True):
        pycache_paths = [
            os.path.join(current_root, name)
            for name in dirnames
            if name == "__pycache__"
        ]
        for path in pycache_paths:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
        if pycache_paths:
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    return removed


def get_experiment_name(config):
    exclude = {"rounds", "track", "debug", "epochs"}  # 不写进文件名的字段

    source = config if isinstance(config, dict) else vars(config)
    detail_fields = {}
    for k, v in source.items():
        if k in exclude:
            continue
        detail_fields[k] = v

    front_keys = ["dataset", "model"]
    front_items = [(k, detail_fields.pop(k)) for k in front_keys if k in detail_fields]
    rest_items = sorted(detail_fields.items(), key=lambda kv: str(kv[0]))
    items = front_items + rest_items

    exper_detail = ", ".join(f"{k} : {v}" for k, v in items)

    prefix_keys = ["dataset", "model", "percent"]
    prefix_parts = []
    for k, v in items:
        if k in prefix_keys:
            clean_val = "".join(ch for ch in str(v) if ch.isalnum())
            prefix_parts.append(clean_val)
            if len(prefix_parts) >= len(prefix_keys):
                break

    log_filename = "-".join(prefix_parts)

    return log_filename, exper_detail


def format_time_minutes(seconds):
    """格式化为分钟字符串，超过60分钟用 Xh Ym 格式"""
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"
    else:
        m = seconds / 60
        return f"{m:.1f}min"


def normalize_save_path(path):
    if path is None:
        return path
    if path.startswith("./output/"):
        return "./results/" + path[len("./output/") :]
    if path.startswith("output/"):
        return "results/" + path[len("output/") :]
    return path


def save_tau_curve(tau_history, save_dir, dataset, model, percent):
    """保存 tau 曲线的遗留兼容函数"""
    if len(tau_history) == 0:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d%H%M%S", time.localtime())
    filename = f"{ts}_{model}_{percent}.pdf"
    path = os.path.join(save_dir, filename)
    epochs = [x[0] for x in tau_history]
    taus = [x[1] for x in tau_history]
    plt.figure()
    plt.plot(epochs, taus)
    plt.xlabel("epoch")
    plt.ylabel("tau")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, format="pdf")
    plt.close()


def parse_experiment_config():
    """解析 main.py 的实验级参数，并兼容透传未知 key-value 参数。"""
    config = parse_args()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--num_runs", type=int, default=None)
    parser.add_argument("--debug", type=int, default=None)
    parser.add_argument("--early_stopping", type=int, default=None)
    parser.add_argument("--min_delta", type=float, default=None)
    parser.add_argument("--print_freq", type=int, default=None)
    extra, unknown = parser.parse_known_args()

    for key, value in vars(extra).items():
        if value is not None:
            setattr(config, key, value)

    if len(unknown) % 2 != 0:
        raise ValueError(f"Unknown args must be key-value pairs, got: {unknown}")

    it = iter(unknown)
    for key in it:
        if not key.startswith("--"):
            raise ValueError(f"Unknown arg key must start with '--', got: {key}")
        raw = next(it)
        low = raw.lower()
        if low in ("true", "false"):
            parsed = low == "true"
        else:
            try:
                parsed = int(raw)
            except ValueError:
                try:
                    parsed = float(raw)
                except ValueError:
                    parsed = raw
        setattr(config, key[2:], parsed)

    if not hasattr(config, "num_runs") or config.num_runs is None:
        config.num_runs = (
            getattr(config, "rounds", 1)
            if getattr(config, "rounds", None) is not None
            else 1
        )
    if not hasattr(config, "rounds") or config.rounds is None:
        config.rounds = config.num_runs
    if not hasattr(config, "debug"):
        config.debug = 0
    if not hasattr(config, "early_stopping"):
        config.early_stopping = True
    if not hasattr(config, "min_delta"):
        config.min_delta = 0.0
    # 统一 debug / early_stopping 为 bool（单一真相源）
    config.debug = bool(int(config.debug)) if str(config.debug).isdigit() else bool(config.debug)
    config.early_stopping = (
        bool(int(config.early_stopping))
        if str(config.early_stopping).isdigit()
        else bool(config.early_stopping)
    )

    # print_freq 自动按 epochs 的 10% 设置（仅在用户未显式传入时）
    if extra.print_freq is None:
        config.print_freq = max(1, int(config.epochs * 0.1))

    return config


def _resolve_runtime_device(device: str) -> str:
    """根据当前环境修正运行设备，避免无 CUDA 时崩溃。"""
    dev = str(device).lower()
    if dev.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _load_denorm_params(config, target_key):
    """加载反归一化参数"""
    data_path = getattr(config, "data_path", None)
    embed_type = getattr(config, "embed_type", "onehot_op")
    if data_path is None:
        return None
    if not os.path.isdir(data_path):
        data_path = os.path.dirname(data_path)
    import glob as _glob

    candidates = _glob.glob(os.path.join(data_path, "all_*.meta.pt"))
    candidates = sorted(
        candidates,
        key=lambda path: (
            os.path.basename(path).endswith(f"_{embed_type}.meta.pt"),
            path,
        ),
    )
    if not candidates:
        return None
    meta = torch.load(candidates[0], weights_only=False)
    norm_params = meta.get("norm_params", {})
    if target_key in norm_params:
        return norm_params[target_key]
    return None


def compute_flops(model, batch_x):
    """计算 FLOPs（优先 torch.profiler，降级为 thop/fvcore）"""
    try:
        with torch.profiler.profile(with_stack=True, record_shapes=True) as prof:
            with torch.no_grad():
                _ = model(batch_x, None)
        # 从 profiler 中提取 FLOPs（近似值）
        flops = prof.key_averages()
        total_flops = sum(
            evt.flop / 1e9 for evt in flops if hasattr(evt, "flop") and evt.flop > 0
        )
        if total_flops > 0:
            return total_flops
    except Exception:
        pass

    # 降级方案：尝试 thop
    try:
        import thop

        dummy = batch_x
        if isinstance(dummy, (list, tuple)):
            dummy = dummy[0]
        if isinstance(dummy, dict):
            dummy = dummy.get("ops", torch.randn(1, 32, 32, 32))[:1]
        flops, _ = thop.profile(model, inputs=(dummy, None), verbose=False)
        return flops / 1e9
    except ImportError:
        pass
    except Exception:
        pass

    return None


def compute_model_stats(model, batch_x):
    """计算模型参数量和 FLOPs"""
    # 总参数量
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = (
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    )

    # FLOPs
    model.eval()
    with torch.no_grad():
        flops = compute_flops(model, batch_x)
    if flops is None:
        print("[Warn] FLOPs calculation failed, skipping...")

    return total_params, trainable_params, flops


def move_batch_to_device(batch_data, device):
    """将 batch 内 tensor 移动到指定 device。"""
    for key, value in batch_data.items():
        if isinstance(value, torch.Tensor):
            batch_data[key] = value.to(device)
    return batch_data


def forward_logits(model, batch_data):
    """统一前向输出格式，兼容 tuple 输出。"""
    output = model(batch_data, None)
    if isinstance(output, tuple):
        return output
    return output, 0.0


def denorm_values(values, denorm):
    """根据归一化参数反归一化。支持 min-max 和 mean-std 两种格式。"""
    if denorm is None:
        return values
    if isinstance(denorm, dict):
        # 新格式: {"method": "minmax", "min": x, "max": y}
        if denorm.get("method") == "minmax" or (
            "min" in denorm and "max" in denorm
        ):
            d_min = float(denorm["min"])
            d_max = float(denorm["max"])
            return [v * (d_max - d_min) + d_min for v in values]
        # 兼容旧格式: {"mean": x, "std": y}
        if "mean" in denorm and "std" in denorm:
            d_mean = float(denorm["mean"])
            d_std = float(denorm["std"])
            return [v * d_std + d_mean for v in values]
    # 兼容更早的 tuple/list 结构: (mean, std)
    if isinstance(denorm, (tuple, list)) and len(denorm) == 2:
        d_mean, d_std = denorm
        return [v * d_std + d_mean for v in values]
    return values


def evaluate_loader(model, data_loader, target_key, denorm, device):
    """在给定数据集上评估并返回 (mape, err, tau, infer_speed, latency_ms)。"""
    metric = Metric()
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for batch_data in data_loader:
            batch_data = move_batch_to_device(batch_data, device)
            gt = batch_data[target_key]
            logits, _ = forward_logits(model, batch_data)

            ps = denorm_values(logits.detach().cpu().numpy()[:, 0].tolist(), denorm)
            gs = denorm_values(gt.detach().cpu().numpy()[:, 0].tolist(), denorm)
            metric.update(ps, gs)

    mape, err, tau = metric.get()
    elapsed = time.time() - t0
    num_samples = len(data_loader.dataset)
    infer_speed = num_samples / elapsed if elapsed > 0 else 0
    latency_ms = (elapsed / num_samples * 1000) if num_samples > 0 else 0
    return mape, err, tau, infer_speed, latency_ms


# ============================================================
# 训练函数
# ============================================================


def train_single_run(config, run_id, paths, logger):
    """
    单轮训练 + 验证 + best model 保存 + 测试评估
    返回: run_summary dict
    """
    # 设备兜底：防止外部参数在传递链路中回退为不可用 CUDA
    config.device = _resolve_runtime_device(getattr(config, "device", "cpu"))

    # 打印实验环境
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    gpu_count = torch.cuda.device_count()
    logger.info("========== Runtime Env ==========")
    logger.info(f"GPU          : {gpu_name} x{gpu_count}")
    logger.info(f"Random Seed  : {config.seed + run_id}")
    logger.info(f"PyTorch      : {torch.__version__}")
    logger.info(f"Python       : {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    logger.info("================================")

    # 初始化 DataLoader
    train_loader, val_loader = init_dataloader(config, logger)
    n_batches = len(train_loader)

    # 获取第一个 batch 用于 FLOPs 计算
    first_batch = move_batch_to_device(next(iter(train_loader)), config.device)

    # 初始化模型和损失
    net, criterion = init_layers(config, logger)
    config.save_path = normalize_save_path(config.save_path)
    os.makedirs(config.save_path, exist_ok=True)

    logger.info(
        f"train_loader samples: {len(train_loader.dataset)} | batch_size: {config.batch_size}"
    )

    # 优化器
    optimizer, scheduler = init_optim(config, net, n_batches, config.warmup_step)

    # 自动恢复
    start_epoch_idx = auto_load_model(config, net, optimizer, scheduler)

    # 确定 target_key
    if "nasbench" in config.dataset:
        predict_target = getattr(config, "predict_target", "accuracy")
        target_key = "latency" if predict_target == "latency" else "val_acc_avg"
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")
    denorm = _load_denorm_params(config, target_key)
    use_mape_early_stop = target_key == "latency"
    if isinstance(denorm, dict):
        if denorm.get("method") == "minmax" or ("min" in denorm and "max" in denorm):
            denorm_str = f"denorm=minmax(min={denorm['min']:.6f}, max={denorm['max']:.6f})"
        elif "mean" in denorm and "std" in denorm:
            denorm_str = f"denorm=zscore(mean={denorm['mean']:.6f}, std={denorm['std']:.6f})"
        else:
            denorm_str = "denorm=unknown"
    elif denorm and isinstance(denorm, (tuple, list)) and len(denorm) == 2:
        denorm_str = f"denorm=zscore(mean={denorm[0]:.6f}, std={denorm[1]:.6f})"
    else:
        denorm_str = "denorm=None"
    logger.info(f"[Target ] predict={target_key} | {denorm_str}")

    # 指标判断
    best_metric_name = "mape" if use_mape_early_stop else "tau"
    best_metric_direction = "min" if use_mape_early_stop else "max"

    # 初始化训练状态
    best_value = -99 if best_metric_direction == "max" else 1e10
    best_epoch = -1
    best_val_tau = None
    best_val_mape = None
    best_val_err = None
    bad_epochs = 0
    stop_training = False

    # 打印模型信息
    total_params, trainable_params, flops = compute_model_stats(net, first_batch)
    logger.info("========== Model Info ==========")
    logger.info(f"Total Params   : {total_params:.2f} M")
    logger.info(f"Trainable      : {trainable_params:.2f} M")
    logger.info(f"FLOPs          : {flops:.2f} G" if flops else "FLOPs          : N/A")
    logger.info(f"Model          : {config.model}")
    run_device = "CUDA" if torch.cuda.is_available() else "CPU"
    logger.info(f"Run in         : {run_device}")
    logger.info("================================")

    # 打印数据集信息
    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    test_samples = 0
    try:
        if hasattr(val_loader, "dataset"):
            test_samples = len(val_loader.dataset)
    except Exception:
        pass

    t0_dl = time.time()
    _ = next(iter(train_loader))
    dl_time = time.time() - t0_dl

    logger.info("========== Dataset Info ==========")
    logger.info(f"Train samples : {train_samples}")
    logger.info(f"Val samples   : {val_samples}")
    logger.info(f"Test samples  : {test_samples}")
    logger.info(f"DataLoader    : {dl_time:.3f}s (first batch)")
    logger.info("=================================")

    # 训练历史
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "tau": [],
        "val_tau": [],
        "mape": [],
        "val_mape": [],
        "err": [],
        "val_err": [],
        "lr": [],
        "grad_norm": [],
        "epoch_time": [],
        "train_speed": [],
        "infer_speed": [],
        "latency_ms": [],
        "gpu_mem_mb": [],
        "overfit_gap": [],
    }

    # 辅助变量
    tau_history = []
    train_start_time = time.time()
    last_v_spd = 0
    last_v_latency = 0

    # 主训练循环
    total_epochs = config.epochs
    for epoch_idx in range(start_epoch_idx, total_epochs):
        # ========== Train ==========
        net.train()
        metric = Metric()
        t0 = time.time()

        for batch_idx, batch_data in enumerate(train_loader):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            optimizer.zero_grad()

            # 数据迁移到设备
            batch_data = move_batch_to_device(batch_data, config.device)
            gt = batch_data[target_key]
            logits, aux_loss = forward_logits(net, batch_data)

            loss_dict = criterion(logits, gt)
            loss = loss_dict["loss"] + 1e-3 * aux_loss
            loss.backward()

            # 计算梯度范数（在 backward 后、step 前）
            gnorm = compute_grad_norm(net)

            optimizer.step()
            scheduler.step()

            # 更新 metric
            ps = denorm_values(logits.detach().cpu().numpy()[:, 0].tolist(), denorm)
            gs = denorm_values(gt.detach().cpu().numpy()[:, 0].tolist(), denorm)
            metric.update(ps, gs)

        # 本 epoch 训练指标
        train_acc, train_err, train_tau = metric.get()
        train_loss_val = loss.item()

        # 梯度范数
        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        epoch_time = time.time() - t0
        train_speed = n_batches * config.batch_size / epoch_time

        # GPU 显存
        gpu_mem_mb = (
            torch.cuda.max_memory_allocated() / 1024 / 1024
            if torch.cuda.is_available()
            else 0
        )

        # ========== Validation ==========
        is_val_epoch = (epoch_idx % config.print_freq == 0) or (
            epoch_idx == total_epochs - 1
        )

        val_loss_val = None
        val_acc = None
        val_err = None
        val_tau = None
        infer_speed = 0
        latency_ms = 0

        if is_val_epoch:
            val_acc, val_err, val_tau, infer_speed, latency_ms = evaluate_loader(
                model=net,
                data_loader=val_loader,
                target_key=target_key,
                denorm=denorm,
                device=config.device,
            )
            val_loss_val = 0.0
            last_v_spd = infer_speed
            last_v_latency = latency_ms

        # ========== Early Stopping & Best Model ==========
        current_metric = val_tau if not use_mape_early_stop else val_acc
        improved = False

        if is_val_epoch and current_metric is not None:
            if best_metric_direction == "max":
                improved = current_metric > best_value + getattr(
                    config, "min_delta", 0.0
                )
            else:
                improved = current_metric < best_value - getattr(
                    config, "min_delta", 0.0
                )

            if improved:
                best_value = current_metric
                best_epoch = epoch_idx + 1
                best_val_tau = val_tau
                best_val_mape = val_acc
                best_val_err = val_err
                bad_epochs = 0

                # 保存 best model
                torch.save(
                    {
                        "epoch": best_epoch,
                        "best_metric_value": best_value,
                        "state_dict": net.state_dict(),
                        "config": (
                            vars(config) if hasattr(config, "__dict__") else config
                        ),
                    },
                    paths.checkpoint_path(run_id),
                )
            elif getattr(config, "early_stopping", True):
                bad_epochs += 1

        # 检查 early stopping
        early_stopping_enabled = (
            getattr(config, "early_stopping", True) and config.patience > 0
        )
        if early_stopping_enabled and bad_epochs >= config.patience:
            stop_training = True

        # ========== History 记录 ==========
        history["epoch"].append(epoch_idx + 1)
        history["train_loss"].append(train_loss_val)
        history["val_loss"].append(val_loss_val)
        history["tau"].append(train_tau)
        history["val_tau"].append(val_tau)
        history["mape"].append(train_acc)
        history["val_mape"].append(val_acc)
        history["err"].append(train_err)
        history["val_err"].append(val_err)
        history["lr"].append(lr)
        history["grad_norm"].append(gnorm)
        history["epoch_time"].append(epoch_time)
        history["train_speed"].append(train_speed)
        history["infer_speed"].append(last_v_spd)
        history["latency_ms"].append(last_v_latency)
        history["gpu_mem_mb"].append(gpu_mem_mb)
        history["overfit_gap"].append(
            train_loss_val - val_loss_val if val_loss_val is not None else None
        )

        # ========== 结构化进度块打印 ==========
        if is_val_epoch:
            elapsed = time.time() - train_start_time
            avg_epoch_time = elapsed / (epoch_idx - start_epoch_idx + 1)
            remaining_epochs = total_epochs - epoch_idx - 1
            eta = (
                format_time_minutes(avg_epoch_time * remaining_epochs)
                if remaining_epochs > 0
                else "0min"
            )

            patience_str = (
                f"{bad_epochs}/{config.patience}" if early_stopping_enabled else "off"
            )
            mem_str = f"{gpu_mem_mb/1024:.2f}GB" if torch.cuda.is_available() else "N/A"

            # 指标格式化
            def fmt(v):
                if v is None:
                    return "N/A"
                if abs(v) < 1e-4:
                    return f"{v:.2e}"
                return f"{v:.6f}"

            logger.info(
                "-" * 30
                + f" Run{run_id}  Epoch {epoch_idx + 1}/{total_epochs} "
                + "-" * 30
            )
            logger.info(
                f"  [Info   ]    dataset={config.dataset}   model={config.model}   run_seed={config.seed + run_id}"
            )
            logger.info(
                f"  [Metrics]    tau={fmt(train_tau)}   mape={fmt(train_acc)}   err={fmt(train_err)}   loss={fmt(train_loss_val)}   (train)"
            )
            logger.info(
                f"  [Metrics]    tau={fmt(val_tau)}   mape={fmt(val_acc)}   err={fmt(val_err)}   loss={fmt(val_loss_val)}   (val)"
            )
            logger.info(f"  [Optim  ]    lr={lr:.2e}   gnorm={gnorm:.4f}")
            logger.info(
                f"  [System ]    t_spd={train_speed:.0f}/s   v_spd={last_v_spd:.0f}/s   mem={mem_str}   patience={patience_str}   elapsed={format_time_minutes(elapsed)}   eta={eta}"
            )
            logger.info(
                f"  [Best   ]    best_tau={fmt(best_val_tau)}   "
                f"best_mape={fmt(best_val_mape)}   best_err={fmt(best_val_err)}"
            )

            if improved:
                logger.info(
                    f"  [Saved  ]    best model updated at epoch {epoch_idx + 1}   "
                    f"{best_metric_name}={fmt(best_value)}   -> {paths.checkpoint_path(run_id)}"
                )

        # 定期保存 checkpoint
        if (epoch_idx + 1) % getattr(config, "save_epoch_freq", 1000) == 0:
            save_check_point(
                epoch_idx + 1,
                batch_idx + 1,
                config,
                net.state_dict(),
                optimizer,
                scheduler,
                False,
                config.dataset + "_checkpoint_Epoch" + str(epoch_idx + 1) + ".pth.tar",
            )

        # 保存 tau curve（遗留兼容）
        tau_history.append((epoch_idx + 1, float(train_tau)))

        if stop_training:
            logger.info(
                f"[EarlyStop] Early stopping triggered at epoch {epoch_idx + 1}"
            )
            break

    # ========== 训练结束 ==========
    total_time = time.time() - train_start_time

    # 保存 tau curve
    save_tau_curve(tau_history, TAU_DIR, config.dataset, config.model, config.percent)

    # 保存 history JSON
    history_path = paths.history_path(run_id)
    with open(history_path, "w") as f:
        json.dump(history, f)
    logger.info(f"[History] Saved to {history_path}")

    # 生成训练可视化
    curve_path = paths.curve_path(run_id)
    plot_training_curves(
        history_path, curve_path, config.dataset, config.model, paths.timestamp
    )

    # 计算统计指标
    valid_infer_speeds = [v for v in history["infer_speed"] if v > 0]
    valid_latencies = [v for v in history["latency_ms"] if v > 0]
    avg_infer_spd = np.mean(valid_infer_speeds) if valid_infer_speeds else 0
    avg_latency_ms = np.mean(valid_latencies) if valid_latencies else 0
    avg_epoch_sec = np.mean(history["epoch_time"]) if history["epoch_time"] else 0

    # 打印训练总结
    print_training_summary(
        logger=logger,
        best_epoch=best_epoch,
        metrics={
            "tau": best_val_tau,
            "mape": best_val_mape,
            "err": best_val_err,
        },
        total_time_sec=total_time,
        early_stopped=stop_training,
        early_stop_epoch=best_epoch,
        params_m=total_params,
        flops_g=flops,
        avg_train_spd=np.mean(history["train_speed"]),
        avg_infer_spd=avg_infer_spd,
        avg_latency_ms=avg_latency_ms,
        peak_gpu_mem_gb=(
            np.max(history["gpu_mem_mb"]) / 1024 if history["gpu_mem_mb"] else 0
        ),
        avg_epoch_sec=avg_epoch_sec,
    )

    # ========== 测试评估 ==========
    # 保存当前 do_train 状态
    original_do_train = config.do_train
    config.do_train = False
    test_loader = init_dataloader(config, logger)
    config.do_train = original_do_train
    best_ckpt_path = paths.checkpoint_path(run_id)

    if os.path.exists(best_ckpt_path):
        logger.info(
            f"[Test   ]    loading best model from {best_ckpt_path}   "
            f"(epoch {best_epoch}, {best_metric_name}={best_value:.6f})"
        )

        checkpoint = torch.load(
            best_ckpt_path, map_location=config.device, weights_only=False
        )
        net.load_state_dict(checkpoint["state_dict"], strict=False)

        # 在测试集上评估
        test_acc, test_err, test_tau, _, _ = evaluate_loader(
            model=net,
            data_loader=test_loader,
            target_key=target_key,
            denorm=denorm,
            device=config.device,
        )

        print_test_results(
            logger=logger,
            run_id=run_id,
            best_epoch=best_epoch,
            metrics={"tau": test_tau, "mape": test_acc, "err": test_err},
            best_ckpt_path=best_ckpt_path,
        )

        test_metrics = {"tau": test_tau, "mape": test_acc, "err": test_err}
    else:
        test_metrics = {"tau": None, "mape": None, "err": None}

    # 构建 run summary
    run_summary = {
        "run_id": run_id,
        "best_epoch": best_epoch,
        "train_time_sec": total_time,
        "tau": best_val_tau,
        "mape": best_val_mape,
        "err": best_val_err,
        "test_tau": test_metrics.get("tau"),
        "test_mape": test_metrics.get("mape"),
        "test_err": test_metrics.get("err"),
    }

    return run_summary


# ============================================================
# 实验入口
# ============================================================


def build_run_args(config, runid, checkpoints_dir=None):
    """从实验配置构建单轮运行参数。"""
    args = parse_args()
    args = merge_config_into_args(args, config)
    args.runid = runid

    dataset_default_paths = {
        "nasbench101": "data/nasbench101/all_nasbench101.pt",
        "nasbench201": "data/nasbench201",
    }
    default_data_path = "data/nasbench101/all_nasbench101.pt"
    if args.data_path == default_data_path and args.dataset in dataset_default_paths:
        args.data_path = dataset_default_paths[args.dataset]

    if checkpoints_dir is not None:
        args.save_path = checkpoints_dir

    return args


def save_result_pickle(metrics, log_filename, config):
    """保存结果到 pickle 文件（遗留兼容）"""
    os.makedirs(METRICS_DIR, exist_ok=True)
    config_copy = {k: v for k, v in config.__dict__.items() if k != "log"}
    result = {
        "config": config_copy,
        "dataset": config.dataset,
        "model": config.model,
        **{k: metrics[k] for k in metrics},
        **{
            f"{k}_mean": (
                np.mean([v for v in metrics[k] if v is not None])
                if any(v is not None for v in metrics[k])
                else None
            )
            for k in metrics
        },
        **{
            f"{k}_std": (
                np.std([v for v in metrics[k] if v is not None])
                if any(v is not None for v in metrics[k])
                else None
            )
            for k in metrics
        },
    }
    with open(os.path.join(METRICS_DIR, f"{log_filename}.pkl"), "wb") as f:
        pickle.dump(result, f)


def run_experiments(config):
    """直接在 main.py 中执行完整实验编排。"""
    if bool(getattr(config, "debug", False)):
        config.num_runs = 2
        config.epochs = 1
        config.print_freq = 1
        print(
            f"[Debug] Debug mode: num_runs={config.num_runs}, epochs={config.epochs}, print_freq={config.print_freq}"
        )

    # 统一修正 device，避免参数为 cuda 但机器无 GPU 导致 run0/run1 中断
    config.device = _resolve_runtime_device(getattr(config, "device", "cpu"))

    timestamp = datetime.now().strftime("%m%d_%H%M")
    paths = PathManager(config.dataset, config.model, timestamp)
    write_config_json(config, paths.config_path)
    get_logger("ProjectLog", PROJECT_RUN_LOG)

    print("")
    print("=" * 60)
    print(f"Experiment: {config.dataset} | {config.model}")
    print(f"Timestamp: {timestamp}")
    print(f"Runs: {config.num_runs} | Debug: {bool(getattr(config, 'debug', False))}")
    print("=" * 60)

    all_run_summaries = []
    metrics = collections.defaultdict(list)

    for run_id in range(config.num_runs):
        run_seed = config.seed + run_id
        set_seed(run_seed)
        remove_pycache_dirs(project_root=os.getcwd())

        print(f"\n{'=' * 60}")
        print(f"Starting Run {run_id}/{config.num_runs} (seed={run_seed})")
        print(f"{'=' * 60}")

        args = build_run_args(config, run_id, checkpoints_dir=paths.checkpoints_dir)
        logger = get_logger("GraphRegression", paths.detail_log_path(run_id))
        args.do_train = True
        run_summary = train_single_run(args, run_id, paths, logger)

        append_summary_json(paths.summary_path, run_summary)
        all_run_summaries.append(run_summary)

        for key, value in run_summary.items():
            metrics[key].append(value)

    log_filename, _ = get_experiment_name(config)

    print("\n" + "=" * 60)
    print("Experiment Results Summary")
    print("=" * 60)

    metrics_order = ["tau", "mape", "err", "test_tau", "test_mape", "test_err"]
    for key in metrics_order:
        vals = [v for v in metrics.get(key, []) if v is not None]
        if vals:
            print(f"{key:15s}: {np.mean(vals):.6f} ± {np.std(vals):.6f}")

    best_epochs = [v for v in metrics.get("best_epoch", []) if v is not None]
    train_times = [v for v in metrics.get("train_time_sec", []) if v is not None]
    if best_epochs:
        print(f"{'best_epoch':15s}: {np.mean(best_epochs):.1f} ± {np.std(best_epochs):.1f}")
    if train_times:
        print(f"{'train_time_sec':15s}: {np.mean(train_times):.1f} ± {np.std(train_times):.1f}")

    print("=" * 60)

    append_runlog_aggregate(
        run_log_path=paths.run_log_path,
        project_runlog_path=PROJECT_RUN_LOG,
        all_run_summaries=all_run_summaries,
        config=config,
        metrics_order=metrics_order,
    )

    save_result_pickle(metrics, log_filename, config)

    print("\nExperiment completed!")
    print(f"Results saved to: {paths.exp_dir}")
    print(f"Project run.log: {os.path.abspath(PROJECT_RUN_LOG)}")

    return metrics


def run_exp(runid, config):
    """兼容旧接口（Experiment.py 调用）"""
    args = build_run_args(config, runid)
    args.device = _resolve_runtime_device(getattr(args, "device", "cpu"))
    timestamp = datetime.now().strftime("%m%d_%H%M")
    paths = PathManager(args.dataset, args.model, timestamp)
    args.save_path = paths.checkpoints_dir

    logger = get_logger("GraphRegression", paths.detail_log_path(runid))
    args.do_train = True
    return train_single_run(args, runid, paths, logger)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    config = parse_experiment_config()
    run_experiments(config)
