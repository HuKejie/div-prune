"""GAT model per KSEM 5822 config: 2 layers, 8 heads, hidden_per_head configurable."""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from divprune.model_module import register_model


@register_model("gat")
class GAT(nn.Module):
    """Two-layer graph attention network with a multi-head first layer.

    Layer 1: multi-head (concat), layer 2: single head averaged to logits,
    following the standard PyG GAT recipe and the KSEM 5822 configuration.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        if cfg.num_layers != 2:
            raise ValueError(f"Only 2-layer GAT is supported, got num_layers={cfg.num_layers}")
        self.cfg = cfg  # kept for pruning surgery (rebuild with fewer heads)
        self.in_channels: int = cfg.in_channels
        self.num_classes: int = cfg.num_classes
        self.num_heads: int = cfg.num_heads
        self.hidden_per_head: int = cfg.hidden_per_head
        self.dropout: float = cfg.dropout
        self.hidden_channels: int = self.num_heads * self.hidden_per_head

        self.conv1 = GATConv(
            self.in_channels,
            self.hidden_per_head,
            heads=self.num_heads,
            dropout=self.dropout,
        )
        self.conv2 = GATConv(
            self.hidden_channels,
            self.num_classes,
            heads=1,
            concat=False,
            dropout=self.dropout,
        )
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict:
        """Node-classification logits.

        Args:
            x: Node features of shape (num_nodes, in_channels).
            edge_index: Graph connectivity of shape (2, num_edges).

        Returns:
            Dict with key "logits" of shape (num_nodes, num_classes).
        """
        x = torch.dropout(x, p=self.dropout, train=self.training)  # input dropout (GAT paper)
        x = self.elu(self.conv1(x, edge_index))
        x = torch.dropout(x, p=self.dropout, train=self.training)
        return {"logits": self.conv2(x, edge_index)}

    def head_outputs(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Per-head outputs of the multi-head layer.

        Returns:
            Tensor of shape (num_nodes, num_heads, hidden_per_head).
        """
        out = self.elu(self.conv1(x, edge_index))
        return out.view(out.size(0), self.num_heads, self.hidden_per_head)

    def attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Attention coefficients of the multi-head layer.

        Returns:
            Tensor of shape (num_edges, num_heads).
        """
        _, (_, alpha) = self.conv1(x, edge_index, return_attention_weights=True)
        # PyG >= 2.4 returns (num_edges, num_heads); only legacy builds returned
        # (num_heads, num_edges), which we transpose to the documented layout.
        alpha = alpha.squeeze(-1) if alpha.dim() == 3 else alpha
        if alpha.size(1) != self.num_heads:
            if alpha.size(0) == self.num_heads:
                alpha = alpha.t()
            else:
                raise ValueError(f"Unexpected attention shape: {tuple(alpha.shape)}")
        return alpha
