"""Inter-head diversity regularizers (KSEM 5822, Eqs. 2-5)."""

from typing import Optional

import torch
import torch.nn as nn


def pairwise_cosine_similarity_mean(matrices: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal cosine similarity among K row vectors.

    Args:
        matrices: Tensor of shape (K, D); row i is the per-head vector of head i.

    Returns:
        Scalar mean of the off-diagonal entries of the K x K cosine similarity matrix.
    """
    k = matrices.size(0)
    if k < 2:
        return torch.zeros((), device=matrices.device)
    norms = matrices / (matrices.norm(dim=1, keepdim=True) + 1e-8)
    sim = norms @ norms.t()
    off_diag = sim[~torch.eye(k, dtype=torch.bool, device=sim.device)]
    return off_diag.mean()


def attention_divergence(alpha: torch.Tensor) -> torch.Tensor:
    """L_div_attention (Eq. 3): mean off-diagonal cosine similarity of per-head attention coefficients.

    Args:
        alpha: Attention coefficients of shape (num_edges, num_heads).
    """
    return pairwise_cosine_similarity_mean(alpha.t())


def output_divergence(head_outputs: torch.Tensor) -> torch.Tensor:
    """L_div_output (Eq. 5): negative mean off-diagonal Euclidean distance of per-head outputs.

    Args:
        head_outputs: Per-head node outputs of shape (num_nodes, num_heads, dim).

    Returns:
        Negative mean pairwise distance, so minimizing the loss pushes heads apart.
    """
    k = head_outputs.size(1)
    if k < 2:
        return torch.zeros((), device=head_outputs.device)
    dist = torch.cdist(head_outputs, head_outputs)  # (num_nodes, num_heads, num_heads)
    off_diag = dist[:, ~torch.eye(k, dtype=torch.bool, device=dist.device)]
    return -off_diag.mean()


def transformer_attention_divergence(attn_mats) -> torch.Tensor:
    """Mean across layers of the per-layer head divergence.

    Args:
        attn_mats: List of attention matrices of shape (B, H, T, T).
    """
    divs = []
    for w in attn_mats:
        b, h, t, _ = w.size()
        flat = w.permute(1, 0, 2, 3).reshape(h, -1)  # (H, B*T*T)
        divs.append(attention_divergence(flat.t()))
    return torch.stack(divs).mean()


class DivergenceRegularizer(nn.Module):
    """Combines attention- and output-level divergence terms per config mode."""

    def __init__(self, mode: str = "attention") -> None:
        super().__init__()
        if mode not in ("attention", "output", "both"):
            raise ValueError(f"Unknown regularizer mode: {mode}")
        self.mode = mode

    def forward(
        self,
        alpha: Optional[torch.Tensor],
        head_outputs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the divergence loss term.

        Args:
            alpha: (num_edges, num_heads) attention coefficients.
            head_outputs: (num_nodes, num_heads, dim) per-head outputs.
        """
        terms: list[torch.Tensor] = []
        if self.mode in ("attention", "both"):
            if alpha is None:
                raise ValueError("attention mode requires attention coefficients")
            terms.append(attention_divergence(alpha))
        if self.mode in ("output", "both"):
            if head_outputs is None:
                raise ValueError("output mode requires head outputs")
            terms.append(output_divergence(head_outputs))
        return torch.stack(terms).sum()
