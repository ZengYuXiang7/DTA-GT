"""消融实验模型包。"""

import importlib
import pkgutil


for module in pkgutil.iter_modules(__path__):
    if module.ispkg:
        continue
    importlib.import_module(f"{__name__}.{module.name}")

__all__ = []
