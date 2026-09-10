"""Integration test: the adaptive controller activates end-to-end in training."""

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

import divprune.model_module.model  # noqa: F401 (registration side effect)
from divprune.model_module import ModelFactory
from divprune.trainer_module.trainer import train_model


def test_adaptive_lambda_rises_when_target_is_zero():
    """With target=0, any positive similarity must wind the controller up (lambda > 0)."""
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "output_dir": "outputs",
            "model": {
                "name": "gat",
                "num_layers": 2,
                "num_heads": 4,
                "hidden_per_head": 2,
                "dropout": 0.0,
                "in_channels": 4,
                "num_classes": 2,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "epochs": 3,
                "early_stop_patience": 10,
            },
            "experiment": {
                "dataset": "cora",
                "regularizer": "attention",
                "lambda_fixed": 0.0,
                "adaptive_lambda": True,
                "lambda_target": 0.0,
                "lambda_eta": 1.0,
                "lambda_max": 5.0,
            },
        }
    )
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    y = torch.randint(0, 2, (5,))
    mask = torch.zeros(5, dtype=torch.bool)
    mask[:3] = True
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=mask, val_mask=~mask, test_mask=~mask)

    model = ModelFactory("gat")(cfg.model)
    # Make all heads identical so the measured similarity is deterministically ~1,
    # independent of the RNG state left by other tests.
    with torch.no_grad():
        h = model.num_heads
        for p_name in ("weight", "bias"):
            p = getattr(model.conv1.lin, p_name)
            if p is None:  # lin has bias=False; conv1's own bias is separate
                continue
            first = p.view(h, -1)[0].clone()
            p.view(h, -1)[:] = first
        for p_name in ("att_src", "att_dst"):
            p = getattr(model.conv1, p_name)
            first = p[:, 0, :].clone()
            p[:] = first[:, None, :]

    result = train_model(cfg, model, data)
    assert result["lambda_final"] > 0.0
    assert result["div_final"] == result["div_final"]  # not NaN
