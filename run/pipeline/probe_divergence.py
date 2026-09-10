"""Diagnostic probe: does L_div mechanically reduce inter-head similarity at convergence?

Compares the mean off-diagonal cosine similarity of trained multi-head layer
attention coefficients with vs without the regularizer (KSEM 5822 Fig. 2 claim).
"""

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from divprune.data_module.dataset.planetoid import PlanetoidDataset
from divprune.model_module.model.gat import GAT
from divprune.seed import set_seed
from divprune.trainer_module.regularizer import attention_divergence


def probe(use_reg: bool, lam: float = 0.1, seed: int = 0) -> tuple[float, float]:
    """Train one GAT and return (best_test_acc, divergence_at_best_checkpoint)."""
    set_seed(seed)
    data = PlanetoidDataset("cora")[0]
    cfg = OmegaConf.create(
        {
            "name": "gat",
            "num_layers": 2,
            "num_heads": 8,
            "hidden_per_head": 8,
            "dropout": 0.6,
            "in_channels": int(data.x.size(1)),
            "num_classes": int(data.y.max()) + 1,
            "learning_rate": 0.005,
            "weight_decay": 5e-4,
            "epochs": 200,
            "early_stop_patience": 100,
        }
    )
    model = GAT(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_val = 0.0
    best_test = 0.0
    div_at_best = float("nan")
    patience = 100
    for _epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)["logits"]
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        if use_reg:
            alpha = model.attention_weights(data.x, data.edge_index)
            loss = loss + lam * attention_divergence(alpha)
        loss.backward()
        opt.step()

        model.eval()
        pred = model(data.x, data.edge_index)["logits"].argmax(dim=1)
        val = float((pred[data.val_mask] == data.y[data.val_mask]).float().mean())
        if val > best_val:
            best_val = val
            best_test = float((pred[data.test_mask] == data.y[data.test_mask]).float().mean())
            alpha = model.attention_weights(data.x, data.edge_index)
            div_at_best = float(attention_divergence(alpha))
            patience = 100
        else:
            patience -= 1
            if patience <= 0:
                break
    return best_test, div_at_best


if __name__ == "__main__":
    for use_reg in (False, True):
        for seed in range(3):
            test, div = probe(use_reg, seed=seed)
            print(f"reg={use_reg} seed={seed}: test={test * 100:.2f}% div_at_best={div:.4f}")
