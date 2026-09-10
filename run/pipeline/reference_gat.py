"""Canonical PyG GAT reference (verbatim recipe) for baseline calibration.

Diagnostic script: if this reproduces ~83.0 on Cora but our pipeline does not,
the pipeline has a bug. If this also lands near ~81, the environment/data
version shifts the baseline and the M1 gate must be recalibrated.
"""

import statistics

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def run(seed: int) -> float:
    torch.manual_seed(seed)
    data = Planetoid(root="data", name="Cora")[0]
    model = GAT(int(data.x.size(1)), int(data.y.max()) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    best_val = 0.0
    best_test = 0.0
    patience = 100
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        pred = model(data.x, data.edge_index).argmax(dim=1)
        val = float((pred[data.val_mask] == data.y[data.val_mask]).float().mean())
        test = float((pred[data.test_mask] == data.y[data.test_mask]).float().mean())
        if val > best_val:
            best_val = val
            best_test = test
            patience = 100
        else:
            patience -= 1
            if patience <= 0:
                break
    return best_test


if __name__ == "__main__":
    tests = [run(s) for s in range(10)]
    print(
        f"reference GAT: {statistics.mean(tests) * 100:.2f} +- "
        f"{statistics.pstdev(tests) * 100:.2f} (canonical target ~82.8-83.1)"
    )
