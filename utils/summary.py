"""训练总结工具 - 写入 summary_all_runs.json 和 run.log"""

import json
import os
from datetime import datetime


def format_time(seconds: float) -> str:
    """将秒数格式化为 h:mm:ss 或 mm:ss"""
    if seconds >= 3600:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {mins}m {secs}s"
    elif seconds >= 60:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        return f"{int(seconds)}s"


def write_config_json(config, config_path: str):
    """将配置写入 config.json"""
    import copy
    cfg = copy.deepcopy(vars(config)) if hasattr(config, "__dict__") else dict(config)
    os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)


def append_summary_json(summary_path: str, run_summary: dict):
    """将单轮结果追加到 summary_all_runs.json (JSON array)"""
    summaries = []
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            try:
                summaries = json.load(f)
                if not isinstance(summaries, list):
                    summaries = [summaries]
            except json.JSONDecodeError:
                summaries = []

    summaries.append(run_summary)
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)


def append_runlog_aggregate(run_log_path: str, project_runlog_path: str, 
                            all_run_summaries: list, config, metrics_order: list):
    """
    将多轮汇总统计追加到 run.log（两份：实验目录 + 项目根目录）
    
    Args:
        run_log_path: 实验目录内的 run.log 路径
        project_runlog_path: 项目根目录的 run.log 路径
        all_run_summaries: 每轮的 summary dict 列表
        config: 配置对象
        metrics_order: 指标名称列表（主指标在前）
    """
    import numpy as np
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 计算均值和标准差
    def safe_mean_std(values, key):
        vals = [v for v in values if v is not None]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    lines = []
    lines.append("")
    lines.append(f"[{timestamp}] ******************** Experiment Results ********************")
    
    # 实验详情行（只含模型结构相关超参）
    detail_parts = [f"dataset={config.dataset}", f"model={config.model}"]
    model_params = ["d_model", "dropout", "gcn_layers", "graph_readout", 
                    "embed_type", "graph_n_head", "depths", "tf_layers"]
    for p in model_params:
        if hasattr(config, p):
            val = getattr(config, p)
            if val is not None:
                detail_parts.append(f"{p}={val}")
    lines.append(f"[{timestamp}] Experiment Detail: {'   '.join(detail_parts)}")
    lines.append(f"[{timestamp}] ------------------------------------------------------------")

    # 指标行
    for metric_name in metrics_order:
        vals = [s.get(metric_name) for s in all_run_summaries]
        mean_val, std_val = safe_mean_std(vals, metric_name)
        if mean_val is None:
            continue
        metric_label = metric_name.ljust(12)
        lines.append(f"[{timestamp}] {metric_label}: {mean_val:.6f} ± {std_val:.6f}")

    # BestEpoch 和 TrainTimeSec
    best_epochs = [s.get("best_epoch") for s in all_run_summaries]
    train_times = [s.get("train_time_sec") for s in all_run_summaries]
    be_mean, be_std = safe_mean_std(best_epochs, "best_epoch")
    tt_mean, tt_std = safe_mean_std(train_times, "train_time_sec")
    if be_mean is not None:
        lines.append(f"[{timestamp}] BestEpoch    : {be_mean:.6f} ± {be_std:.6f}")
    if tt_mean is not None:
        lines.append(f"[{timestamp}] TrainTimeSec : {tt_mean:.6f} ± {tt_std:.6f}")

    content = "\n".join(lines) + "\n"

    # 写入两份 run.log
    for path in [run_log_path, project_runlog_path]:
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(content)
    
    # 同时打印到终端
    print("\n" + content)
    return content


def print_training_summary(logger, best_epoch: int, metrics: dict,
                          total_time_sec: float, early_stopped: bool,
                          early_stop_epoch: int, params_m: float, 
                          flops_g: float, avg_train_spd: float,
                          avg_infer_spd: float, avg_latency_ms: float,
                          peak_gpu_mem_gb: float, avg_epoch_sec: float):
    """
    打印结构化训练总结
    """
    early_stop_str = f"Yes (epoch {early_stop_epoch})" if early_stopped else "No"

    logger.info("========== Training Summary ==========")
    logger.info(f"Best Epoch     : {best_epoch}")
    for k, v in metrics.items():
        if v is not None:
            logger.info(f"Best {k.upper():10s}: {v:.6f}")
    logger.info(f"Total Time     : {format_time(total_time_sec)}")
    logger.info(f"Early Stop     : {early_stop_str}")
    logger.info(f"Params         : {params_m:.1f} M")
    logger.info(f"FLOPs          : {flops_g:.2f} G" if flops_g else "FLOPs          : N/A")
    logger.info(f"Avg Train Spd  : {avg_train_spd:.0f} samples/s")
    logger.info(f"Avg Infer Spd  : {avg_infer_spd:.0f} samples/s")
    logger.info(f"Avg Latency    : {avg_latency_ms:.4f} ms/sample")
    logger.info(f"Peak GPU Mem   : {peak_gpu_mem_gb:.2f} GB")
    logger.info(f"Avg Epoch Time : {format_time(avg_epoch_sec)}")
    logger.info("======================================")


def print_test_results(logger, run_id: int, best_epoch: int,
                       metrics: dict, best_ckpt_path: str):
    """打印测试结果"""
    logger.info("")
    logger.info(f"========== Test Results (Run{run_id}) ==========")
    logger.info(f"Source         : {best_ckpt_path}")
    logger.info(f"Best Epoch     : {best_epoch}")
    for k, v in metrics.items():
        if v is not None:
            logger.info(f"{k.upper():10s}: {v:.6f}")
    logger.info("=" * 47)
