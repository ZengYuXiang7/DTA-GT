"""全局随机种子管理 - 确保实验可复现"""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """设置所有随机源的全局 seed，适用于 run_id 推导：seed = base_seed + run_id

    Args:
        seed: 当前轮的全局 seed（如 base_seed=42, run_id=0 → seed=42）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 保证 CUDA 卷积/线性算法确定性（非所有硬件支持，可失败后忽略）
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
