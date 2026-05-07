"""训练可视化工具 - 生成训练曲线大图 (3行2列)"""

import json
import os
from datetime import datetime


def plot_training_curves(history_path: str, save_path: str, dataset: str, model: str, timestamp: str):
    """
    根据 history JSON 生成训练曲线可视化大图。

    Args:
        history_path: history_run{run_id}.json 文件路径
        save_path: 输出的 .png 文件路径
        dataset: 数据集名称
        model: 模型名称
        timestamp: 实验时间戳
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"[Warn] matplotlib not available: {e}")
        return False

    if not os.path.exists(history_path):
        print(f"[Warn] history file not found: {history_path}")
        return False

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = history.get("epoch", [])
    if not epochs:
        print(f"[Warn] history is empty: {history_path}")
        return False

    n_epochs = len(epochs)

    train_loss = history.get("train_loss", [None] * n_epochs)
    val_loss = history.get("val_loss", [None] * n_epochs)
    tau = history.get("tau", [None] * n_epochs)
    val_tau = history.get("val_tau", [None] * n_epochs)
    mape = history.get("mape", [None] * n_epochs)
    val_mape = history.get("val_mape", [None] * n_epochs)
    lr = history.get("lr", [None] * n_epochs)
    grad_norm = history.get("grad_norm", [None] * n_epochs)
    gpu_mem = history.get("gpu_mem_mb", [None] * n_epochs)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"{dataset} | {model} | {timestamp}", fontsize=14, fontweight="bold")

    # 子图1: Train Loss vs Val Loss
    ax = axes[0, 0]
    valid_train_loss = [(e, v) for e, v in zip(epochs, train_loss) if v is not None]
    valid_val_loss = [(e, v) for e, v in zip(epochs, val_loss) if v is not None]
    if valid_train_loss:
        es, vs = zip(*valid_train_loss)
        ax.plot(es, vs, label="Train Loss", color="steelblue", linewidth=1.5)
    if valid_val_loss:
        es, vs = zip(*valid_val_loss)
        ax.plot(es, vs, label="Val Loss", color="coral", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train Loss vs Val Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图2: Overfitting Gap
    ax = axes[0, 1]
    gaps = []
    gap_epochs = []
    for e, tl, vl in zip(epochs, train_loss, val_loss):
        if tl is not None and vl is not None:
            gaps.append(tl - vl)
            gap_epochs.append(e)
    if gaps:
        ax.plot(gap_epochs, gaps, color="purple", linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gap (Train - Val)")
    ax.set_title("Overfitting Gap")
    ax.grid(True, alpha=0.3)

    # 子图3: 主监控指标 Tau (train + val)
    ax = axes[1, 0]
    valid_tau = [(e, v) for e, v in zip(epochs, tau) if v is not None]
    valid_val_tau = [(e, v) for e, v in zip(epochs, val_tau) if v is not None]
    if valid_tau:
        es, vs = zip(*valid_tau)
        ax.plot(es, vs, label="Train Tau", color="steelblue", linewidth=1.5)
    if valid_val_tau:
        es, vs = zip(*valid_val_tau)
        ax.plot(es, vs, label="Val Tau", color="coral", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Kendall Tau")
    ax.set_title("Kendall Tau (Train vs Val)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图4: Learning Rate
    ax = axes[1, 1]
    valid_lr = [(e, v) for e, v in zip(epochs, lr) if v is not None]
    if valid_lr:
        es, vs = zip(*valid_lr)
        ax.plot(es, vs, color="green", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # 子图5: Gradient Norm
    ax = axes[2, 0]
    valid_gn = [(e, v) for e, v in zip(epochs, grad_norm) if v is not None]
    if valid_gn:
        es, vs = zip(*valid_gn)
        ax.plot(es, vs, color="orange", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm")
    ax.grid(True, alpha=0.3)

    # 子图6: GPU 显存占用
    ax = axes[2, 1]
    valid_mem = [(e, v) for e, v in zip(epochs, gpu_mem) if v is not None]
    if valid_mem:
        es, vs = zip(*valid_mem)
        ax.plot(es, vs, color="red", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("GPU Memory (MB)")
    ax.set_title("GPU Memory Usage")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Training curves saved to {save_path}")
    return True
