import logging
import os


def setup_logger(filename):
    """
    创建/获取全局 logger。
    为了避免在多次实验（例如 Experiment.py 多轮 RunExperiments）中重复添加 handler，
    这里保证对同一个进程只初始化一次 handler，后续调用直接复用。
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 如果已经初始化过（有 FileHandler 且指向同一个文件），直接复用，避免重复打印
    abs_filename = os.path.abspath(filename)
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and os.path.abspath(
            getattr(h, "baseFilename", "")
        ) == abs_filename:
            return logger

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
