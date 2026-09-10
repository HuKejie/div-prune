"""Shared metric helpers."""

import torch


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Classification accuracy as a float in [0, 1]."""
    return float((pred == target).float().mean().item())
