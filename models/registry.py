"""模型注册表 - 统一管理所有模型的实例化。"""

from __future__ import annotations

from typing import Any, Callable

_MODEL_REGISTRY: dict[str, type] = {}


def register_model(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Duplicate model name: {name}")
        _MODEL_REGISTRY[name] = cls
        return cls

    return deco


def build_model(name: str, args: Any) -> Any:
    cls = _MODEL_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: [{available}]")
    return cls(args)


def get_model(args):
    return build_model(args.model, args)


def list_models() -> list[str]:
    return sorted(_MODEL_REGISTRY.keys())


__all__ = [
    "register_model",
    "build_model",
    "get_model",
    "list_models",
    "_MODEL_REGISTRY",
]
