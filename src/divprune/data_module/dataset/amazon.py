"""Amazon co-purchase graphs (Computers / Photo) with fixed class-balanced splits.

The raw PyG Amazon dataset ships no split masks, so we follow the standard
protocol of Shchur et al. (2018): 20 training labels per class, 30 validation
labels per class, the rest for testing. Masks are drawn once with a fixed
split RNG so every run of every arm sees the identical split; run-to-run
variance then comes from the initialization seed only, matching the
fixed-split property of the Planetoid wrapper.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon

from divprune.data_module import register_dataset

AMAZON_NAMES = ("computers", "photo")
# Fixed split: class-balanced, deterministic across runs and machines.
SPLIT_SEED = 20260822
TRAIN_PER_CLASS = 20
VAL_PER_CLASS = 30


class AmazonDataset(Dataset):
    """Single-graph wrapper adding fixed class-balanced split masks."""

    def __init__(self, name: str, root: str = "data") -> None:
        if name not in AMAZON_NAMES:
            raise ValueError(
                f"Unknown Amazon dataset: {name}. Available: {AMAZON_NAMES}"
            )
        self.name = name
        pyg_name = "Computers" if name == "computers" else "Photo"
        data: Data = Amazon(root=root, name=pyg_name)[0]
        data.train_mask, data.val_mask, data.test_mask = (
            torch.from_numpy(m) for m in _fixed_class_split(
                data.y.cpu().numpy(),
                train_per_class=TRAIN_PER_CLASS,
                val_per_class=VAL_PER_CLASS,
                seed=SPLIT_SEED,
            )
        )
        self.data = data

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Data:
        return self.data


def _fixed_class_split(
    y: np.ndarray,
    train_per_class: int,
    val_per_class: int,
    seed: int,
) -> tuple:
    """Deterministic class-balanced masks: (train, val, test) boolean arrays."""
    rng = np.random.RandomState(seed)
    n = len(y)
    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        train[idx[:train_per_class]] = True
        val[idx[train_per_class : train_per_class + val_per_class]] = True
    test = ~(train | val)
    return train, val, test


# Register the same class under each dataset name.
for _name in AMAZON_NAMES:
    register_dataset(_name)(AmazonDataset)

__all__ = ["AmazonDataset", "AMAZON_NAMES"]
