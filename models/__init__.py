"""统一模型包。"""

import importlib
import pkgutil

from models.registry import build_model, get_model, list_models, register_model


def import_all_models() -> None:
    """导入所有可注册模型模块。训练初始化时调用。"""
    skip = {"registry", "layer_init", "model56_utils"}
    for module in pkgutil.iter_modules(__path__):
        if module.name in skip:
            continue
        if module.ispkg:
            if module.name == "ablations":
                importlib.import_module(f"{__name__}.{module.name}")
            continue
        importlib.import_module(f"{__name__}.{module.name}")


__all__ = [
    "register_model",
    "build_model",
    "get_model",
    "list_models",
    "import_all_models",
]
