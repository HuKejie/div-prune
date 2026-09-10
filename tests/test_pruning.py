"""Unit tests for head-pruning surgery and criterion shapes."""

import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data

import divprune.model_module.model  # noqa: F401 (registration side effect)
from divprune.model_module import ModelFactory
from divprune.trainer_module.pruning import (
    diversity_scores,
    gradient_scores,
    head_importance,
    magnitude_scores,
    prune_gat_heads,
    select_keep,
)


def _cfg(heads: int = 8) -> DictConfig:
    return OmegaConf.create(
        {
            "name": "gat",
            "num_layers": 2,
            "num_heads": heads,
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


def _graph() -> Data:
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    y = torch.randint(0, 2, (5,))
    mask = torch.zeros(5, dtype=torch.bool)
    mask[:3] = True
    return Data(x=x, edge_index=edge_index, y=y, train_mask=mask, val_mask=~mask, test_mask=~mask)


def test_criterion_shapes():
    model = ModelFactory("gat")(_cfg())
    data = _graph()
    assert diversity_scores(model, data).shape == (8,)
    assert gradient_scores(model, data).shape == (8,)
    assert magnitude_scores(model).shape == (8,)


def test_prune_reduces_params_and_copies_kept_weights():
    model = ModelFactory("gat")(_cfg())
    pruned = prune_gat_heads(model, [0, 2, 4, 6])
    n_full = sum(p.numel() for p in model.parameters())
    n_pruned = sum(p.numel() for p in pruned.parameters())
    assert n_pruned < n_full
    d = model.hidden_per_head
    with torch.no_grad():
        for new_i, old_i in enumerate([0, 2, 4, 6]):
            blk_old = model.conv1.lin.weight.view(8, d, -1)[old_i]
            blk_new = pruned.conv1.lin.weight.view(4, d, -1)[new_i]
            assert torch.equal(blk_new, blk_old)


def test_prune_slices_conv2_input():
    model = ModelFactory("gat")(_cfg())
    pruned = prune_gat_heads(model, [0, 1])
    # conv2 input dim = kept_heads * hidden_per_head
    assert pruned.conv2.lin.weight.shape == (2, 4)


def test_prune_forward_runs():
    model = ModelFactory("gat")(_cfg())
    pruned = prune_gat_heads(model, [0, 1])
    data = _graph()
    out = pruned(data.x, data.edge_index)["logits"]
    assert out.shape == (5, 2)


def test_prune_copies_att_and_bias():
    model = ModelFactory("gat")(_cfg())
    pruned = prune_gat_heads(model, [1, 3, 5])
    keep_t = torch.tensor([1, 3, 5])
    with torch.no_grad():
        for name in ("att_src", "att_dst"):
            assert torch.equal(
                getattr(pruned.conv1, name),
                getattr(model.conv1, name)[:, keep_t, :],
            )
        d = model.hidden_per_head
        assert torch.equal(
            pruned.conv1.bias,
            model.conv1.bias.view(8, d)[keep_t].reshape(-1),
        )
        assert torch.equal(
            pruned.conv2.lin.weight,
            model.conv2.lin.weight.view(2, 8, d)[:, keep_t, :].reshape(2, -1),
        )


def test_prune_kept_heads_are_functionally_identical():
    model = ModelFactory("gat")(_cfg())
    model.eval()
    data = _graph()
    keep = [1, 3, 5]
    pruned = prune_gat_heads(model, keep)
    pruned.eval()
    with torch.no_grad():
        dense_out = model.conv1(data.x, data.edge_index)
        pruned_out = pruned.conv1(data.x, data.edge_index)
        dense_blocks = dense_out.view(-1, 8, 2)[:, keep, :]
        pruned_blocks = pruned_out.view(-1, 3, 2)
        assert torch.allclose(pruned_blocks, dense_blocks, atol=1e-6)


def test_select_keep_direction_per_criterion():
    scores = torch.tensor([0.9, 0.1, 0.5])
    # diversity: lowest redundancy kept -> index 1
    assert select_keep(scores, 1, "diversity") == [1]
    # gradient/magnitude: highest importance kept -> index 0
    assert select_keep(scores, 1, "gradient") == [0]
    assert select_keep(scores, 1, "magnitude") == [0]


def test_prune_raises_on_bad_keep():
    import pytest

    model = ModelFactory("gat")(_cfg())
    with pytest.raises(ValueError):
        prune_gat_heads(model, [])
    with pytest.raises(ValueError):
        prune_gat_heads(model, [0, 0, 1])
    with pytest.raises(ValueError):
        head_importance(model, _graph(), "unknown")


def test_gradient_scores_deterministic():
    model = ModelFactory("gat")(_cfg())
    data = _graph()
    s1 = gradient_scores(model, data)
    s2 = gradient_scores(model, data)
    # allclose (not equal): multi-threaded BLAS reductions can vary in the last bit
    assert torch.allclose(s1, s2, atol=1e-6)


def test_dense_from_checkpoint_rebuilds_metrics(tmp_path):
    from divprune.trainer_module.trainer import dense_metrics_from_checkpoint

    model = ModelFactory("gat")(_cfg())
    ckpt = tmp_path / "teacher_mindiv.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": 5,
            "val_acc": 0.81,
            "div_t": 0.62,
            "lambda_t": 1.7,
        },
        ckpt,
    )
    dense = dense_metrics_from_checkpoint(ckpt, model, _graph(), torch.device("cpu"))
    assert dense["val_acc"] == 0.81
    assert dense["lambda_final"] == 1.7
    assert dense["div_final"] == 0.62
    assert dense["best_epoch"] == 5
    assert dense["epochs_run"] == 6
    assert dense["params"] == sum(p.numel() for p in model.parameters())
    assert 0.0 <= dense["test_acc"] <= 1.0
