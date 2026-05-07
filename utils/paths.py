"""实验路径管理 - 所有路径的唯一来源"""

import os
from datetime import datetime

RESULTS_ROOT = "results"
PROJECT_RUN_LOG = "run.log"
METRICS_DIR = os.path.join(RESULTS_ROOT, "metrics")
TAU_DIR = os.path.join(RESULTS_ROOT, "tau")
NOHUP_TRAIN_LOG = os.path.join(RESULTS_ROOT, "train.log")
NOHUP_TRAIN_PID = os.path.join(RESULTS_ROOT, "train.pid")


class PathManager:
    """集中管理单次实验的所有文件路径"""

    def __init__(self, dataset: str, model: str, timestamp: str = None):
        self.dataset = dataset
        self.model = model
        # 时间戳只生成一次，全程固定，不在循环内重新生成
        self.timestamp = timestamp or datetime.now().strftime("%m%d_%H%M")
        self.exp_dir = os.path.join(RESULTS_ROOT, dataset, model, self.timestamp)
        self._init_dirs()

    def _init_dirs(self):
        """训练开始前一次性创建所有必要子目录"""
        for d in [self.exp_dir, self.details_dir, self.checkpoints_dir]:
            os.makedirs(d, exist_ok=True)

    # ── 子目录 ──────────────────────────────────
    @property
    def details_dir(self) -> str:
        return os.path.join(self.exp_dir, "details")

    @property
    def checkpoints_dir(self) -> str:
        return os.path.join(self.exp_dir, "checkpoints")

    # ── 固定文件（整个实验唯一） ─────────────────
    @property
    def config_path(self) -> str:
        return os.path.join(self.exp_dir, "config.json")

    @property
    def summary_path(self) -> str:
        return os.path.join(self.exp_dir, "summary_all_runs.json")

    @property
    def run_log_path(self) -> str:
        return os.path.join(self.exp_dir, "run.log")

    # ── 按 run_id 变化的文件 ─────────────────────
    def curve_path(self, run_id: int) -> str:
        return os.path.join(self.exp_dir, f"training_curve_run{run_id}.png")

    def history_path(self, run_id: int) -> str:
        return os.path.join(self.exp_dir, f"history_run{run_id}.json")

    def detail_log_path(self, run_id: int) -> str:
        return os.path.join(self.details_dir, f"run{run_id}.log")

    def checkpoint_path(self, run_id: int) -> str:
        return os.path.join(self.checkpoints_dir, f"best_model_run{run_id}.pt")

    def __repr__(self) -> str:
        return f"PathManager(exp_dir={self.exp_dir!r})"
