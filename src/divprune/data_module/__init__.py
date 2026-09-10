"""Data module: dataset registry and factory."""

from typing import Callable, Dict, Type

from torch.utils.data import Dataset

DATASET_FACTORY: Dict[str, Type[Dataset]] = {}


def register_dataset(name: str) -> Callable[[Type[Dataset]], Type[Dataset]]:
    """Register a dataset class under a name."""

    def decorator(cls: Type[Dataset]) -> Type[Dataset]:
        DATASET_FACTORY[name] = cls
        return cls

    return decorator


def DatasetFactory(name: str) -> Type[Dataset]:
    """Return the dataset class registered under name.

    Raises:
        KeyError: When name is not registered.
    """
    if name not in DATASET_FACTORY:
        raise KeyError(f"Dataset '{name}' not registered. Available: {list(DATASET_FACTORY)}")
    return DATASET_FACTORY[name]


__all__ = ["DATASET_FACTORY", "register_dataset", "DatasetFactory"]
