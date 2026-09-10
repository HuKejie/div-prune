"""Integration test: training with a KD loss term runs and produces valid metrics."""

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

import divprune.model_module.model  # noqa: F401 (registration side effect)
from divprune.model_module import ModelFactory
from divprune.trainer_module.trainer import train_model


def _toy_cfg() -> OmegaConf:
    return OmegaConf.create(
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
                "regularizer": "none",
                "lambda_fixed": 0.0,
                "adaptive_lambda": False,
                "lambda_target": 0.7,
                "lambda_eta": 0.05,
                "lambda_max": 3.0,
                "distill_alpha": 0.5,
                "distill_temperature": 2.0,
            },
        }
    )


def _toy_data() -> Data:
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    y = torch.randint(0, 2, (5,))
    mask = torch.zeros(5, dtype=torch.bool)
    mask[:3] = True
    return Data(x=x, edge_index=edge_index, y=y, train_mask=mask, val_mask=~mask, test_mask=~mask)


def test_train_with_kd_runs():
    cfg = _toy_cfg()
    data = _toy_data()
    model = ModelFactory("gat")(cfg.model)
    teacher_logits = torch.randn(5, 2)
    result = train_model(cfg, model, data, teacher_logits=teacher_logits)
    assert result["test_acc"] >= 0.0
    assert result["epochs_run"] >= 1.0
