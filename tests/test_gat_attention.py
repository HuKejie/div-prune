"""Shape checks for GAT attention-weight access (guards the Eq. 3 axis bug)."""

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

import divprune.model_module.model  # noqa: F401 (registration side effect)
from divprune.model_module import ModelFactory


def _make_gat():
    cfg = OmegaConf.create(
        {
            "name": "gat",
            "num_layers": 2,
            "num_heads": 8,
            "hidden_per_head": 2,
            "dropout": 0.0,
            "in_channels": 4,
            "num_classes": 2,
            "learning_rate": 0.005,
            "weight_decay": 0.0,
            "epochs": 1,
            "early_stop_patience": 100,
        }
    )
    return ModelFactory("gat")(cfg)


def test_attention_weights_shape_is_edges_by_heads():
    model = _make_gat()
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    alpha = model.attention_weights(x, edge_index)
    # GATConv adds self-loops: 3 edges + 3 self-loops = 6 rows, 8 heads.
    assert alpha.shape == (6, 8)
