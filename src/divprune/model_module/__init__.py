"""Model module: model registry and factory."""

from typing import Callable, Dict, Type

import torch.nn as nn

MODEL_FACTORY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """Register a model class under a name."""

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        MODEL_FACTORY[name] = cls
        return cls

    return decorator


def ModelFactory(name: str) -> Type[nn.Module]:
    """Return the model class registered under name.

    Raises:
        KeyError: When name is not registered.
    """
    if name not in MODEL_FACTORY:
        raise KeyError(f"Model '{name}' not registered. Available: {list(MODEL_FACTORY)}")
    return MODEL_FACTORY[name]


__all__ = ["MODEL_FACTORY", "register_model", "ModelFactory"]
