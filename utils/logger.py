"""日志工具 - 同时输出到终端和 run 专属日志文件"""

import logging
import sys


def get_logger(name: str, log_path: str = None) -> logging.Logger:
    """
    获取 logger，同时输出到终端和文件（若提供 log_path）。
    每次新 run 调用时传入新的 log_path，FileHandler 自动切换。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # 防止重复添加 handler
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 终端 handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件 handler（每个 run 独立）
    if log_path:
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
