"""Planetoid citation-network datasets (Cora / CiteSeer / PubMed)."""

from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

from divprune.data_module import register_dataset

# PyG Planetoid expects the capitalized names (".title()" would mangle CiteSeer).
_PLANETOID_NAME_MAP = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
PLANETOID_NAMES = tuple(_PLANETOID_NAME_MAP)


class PlanetoidDataset(Dataset):
    """Single-graph wrapper so a torch_geometric Data object follows the Dataset API.

    A transductive node-classification dataset is one graph; indexing always
    returns the same graph. Split masks are the public fixed Planetoid split,
    so run-to-run variance comes from random initialization seeds only
    (matching the KSEM 5822 protocol of reporting mean +/- std over seeds).
    """

    def __init__(self, name: str, root: str = "data") -> None:
        if name not in _PLANETOID_NAME_MAP:
            raise ValueError(f"Unknown Planetoid dataset: {name}. Available: {PLANETOID_NAMES}")
        self.name = name
        self.data: Data = Planetoid(root=root, name=_PLANETOID_NAME_MAP[name])[0]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Data:
        return self.data


# Register the same class under each dataset name.
for _name in PLANETOID_NAMES:
    register_dataset(_name)(PlanetoidDataset)

__all__ = ["PlanetoidDataset", "PLANETOID_NAMES"]
