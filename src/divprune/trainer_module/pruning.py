"""Head pruning: importance criteria and GAT multi-head layer surgery."""

from typing import Dict, List

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch_geometric.data import Data

from divprune.model_module.model.gat import GAT

VALID_CRITERIA = ("diversity", "gradient", "random", "magnitude")


def _head_similarity_matrix(alpha: torch.Tensor) -> torch.Tensor:
    """(num_heads, num_heads) cosine similarity of per-head attention coefficients.

    Args:
        alpha: Attention coefficients of shape (num_edges, num_heads).
    """
    norms = alpha / (alpha.norm(dim=0, keepdim=True) + 1e-8)
    return norms.t() @ norms


@torch.no_grad()
def diversity_scores(model: GAT, data: Data) -> torch.Tensor:
    """Per-head redundancy score: mean similarity to the other heads (higher = more redundant).

    This is the L_div metric itself reused as the pruning criterion.
    """
    model.eval()
    alpha = model.attention_weights(data.x, data.edge_index)
    sim = _head_similarity_matrix(alpha)
    k = sim.size(0)
    if k < 2:
        return torch.zeros(k, device=sim.device)
    off = sim[~torch.eye(k, dtype=torch.bool, device=sim.device)].view(k, k - 1)
    return off.mean(dim=1)


def gradient_scores(model: GAT, data: Data) -> torch.Tensor:
    """Per-head gradient-based importance (Michel et al. 2019 style).

    Runs in eval mode (no dropout) for a deterministic estimate. The hook
    captures the raw pre-ELU conv1 output, so |out * grad| is a consistent
    first-order attribution on that quantity.
    """
    captured: Dict[str, torch.Tensor] = {}

    def fwd_hook(module, inputs, output) -> None:
        captured["out"] = output.detach()

    def bwd_hook(module, grad_input, grad_output) -> None:
        captured["grad"] = grad_output[0].detach()

    prev_training = model.training
    h1 = model.conv1.register_forward_hook(fwd_hook)
    h2 = model.conv1.register_full_backward_hook(bwd_hook)
    model.eval()
    model.zero_grad(set_to_none=True)
    out = model(data.x, data.edge_index)["logits"]
    loss = nn.functional.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    h1.remove()
    h2.remove()
    model.zero_grad(set_to_none=True)
    if prev_training:
        model.train()

    h, d = model.num_heads, model.hidden_per_head
    out_h = captured["out"].view(-1, h, d)
    grad_h = captured["grad"].view(-1, h, d)
    return (out_h * grad_h).abs().mean(dim=(0, 2))


@torch.no_grad()
def magnitude_scores(model: GAT) -> torch.Tensor:
    """Per-head weight-block norms of the multi-head convolution."""
    h, d = model.num_heads, model.hidden_per_head
    blocks = model.conv1.lin.weight.view(h, d, -1)
    return blocks.norm(dim=(1, 2))


def random_scores(model: GAT) -> torch.Tensor:
    """Uniform random scores (control criterion)."""
    return torch.rand(model.num_heads)


def head_importance(model: GAT, data: Data, criterion: str) -> torch.Tensor:
    """Dispatch per-head scores for the multi-head layer.

    Returns:
        Tensor of shape (num_heads,). Semantics differ by criterion:
        - diversity: higher = more redundant = prune first
        - gradient / magnitude: higher = more important = keep first
        - random: uniform scores, selection is arbitrary
    """
    if criterion == "diversity":
        return diversity_scores(model, data)
    if criterion == "gradient":
        return gradient_scores(model, data)
    if criterion == "magnitude":
        return magnitude_scores(model)
    if criterion == "random":
        return random_scores(model)
    raise ValueError(f"Unknown pruning criterion: {criterion}. Available: {VALID_CRITERIA}")


def select_keep(scores: torch.Tensor, n_keep: int, criterion: str) -> List[int]:
    """Indices of heads to keep, ordered ascending.

    diversity prunes the highest (most redundant) scores; gradient and
    magnitude keep the highest (most important) scores.
    """
    largest = criterion in ("gradient", "magnitude")
    return torch.topk(scores, k=n_keep, largest=largest).indices.sort().values.tolist()


def prune_gat_heads(model: GAT, keep: List[int]) -> GAT:
    """Return a new GAT that keeps only the given heads of the multi-head layer.

    Copies per-head blocks of conv1 (weight/bias/attention params) and slices
    conv2's input dimension accordingly. Real parameter reduction, not masking.
    """
    h_old, d = model.num_heads, model.hidden_per_head
    if len(keep) < 1 or len(keep) > h_old:
        raise ValueError(f"keep must contain 1..{h_old} heads, got {len(keep)}")
    if len(set(keep)) != len(keep):
        raise ValueError(f"keep contains duplicates: {keep}")

    new_cfg = OmegaConf.create(
        {**OmegaConf.to_container(model.cfg, resolve=True), "num_heads": len(keep)}
    )
    new = GAT(new_cfg)
    keep_t = torch.tensor(keep, dtype=torch.long)

    with torch.no_grad():
        # conv1 linear weight (H*D, in) and bias (H*D,) -- PyG 2.8: conv.lin
        w1 = model.conv1.lin.weight.view(h_old, d, -1)[keep_t].reshape(-1, model.in_channels)
        new.conv1.lin.weight.copy_(w1)
        if model.conv1.bias is not None:
            b1 = model.conv1.bias.view(h_old, d)[keep_t].reshape(-1)
            new.conv1.bias.copy_(b1)
        # attention params: PyG stores (1, H, D); tolerate other layouts
        for name in ("att_src", "att_dst"):
            src = getattr(model.conv1, name)
            dst = getattr(new.conv1, name)
            if src.dim() == 3 and src.size(1) == h_old:
                dst.copy_(src[:, keep_t, :])
            elif src.size(0) == h_old:
                dst.copy_(src[keep_t])
            else:
                raise ValueError(f"Unexpected {name} shape: {tuple(src.shape)}")
        # conv2 input blocks (num_classes, H*D)
        w2 = model.conv2.lin.weight.view(model.conv2.lin.weight.size(0), h_old, d)[:, keep_t, :]
        new.conv2.lin.weight.copy_(w2.reshape(model.conv2.lin.weight.size(0), -1))
    return new
