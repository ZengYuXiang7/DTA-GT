"""时间格式化工具"""


def format_elapsed(seconds: float) -> str:
    """将秒数格式化为易读字符串"""
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) / 60)}min"
    return f"{seconds / 60:.1f}min"


def format_duration(seconds: float) -> str:
    """将秒数格式化为 hh:mm:ss 格式"""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s"
