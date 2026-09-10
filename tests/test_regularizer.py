"""Hand-computed unit tests for the divergence math (Eqs. 2-5)."""

import pytest
import torch

from divprune.trainer_module.regularizer import (
    attention_divergence,
    output_divergence,
    pairwise_cosine_similarity_mean,
)


def test_cosine_identical_vectors():
    identical = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    assert pairwise_cosine_similarity_mean(identical) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    orth = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert pairwise_cosine_similarity_mean(orth) == pytest.approx(0.0, abs=1e-6)


def test_cosine_single_vector_is_zero():
    assert pairwise_cosine_similarity_mean(torch.tensor([[1.0, 0.0]])) == 0.0


def test_attention_divergence_computes_over_heads_not_edges():
    # Columns are heads: h0=(1,0,1), h1=(0,1,1) -> cosine = 0.5.
    # (Over edges the mean would be ~0.4714, so this pins the axis.)
    alpha = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert attention_divergence(alpha) == pytest.approx(0.5, abs=1e-6)


def test_output_divergence_negative_distance():
    # One node, two heads: outputs (0,0) and (3,4) -> distance 5 -> -5.
    head_out = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])
    assert output_divergence(head_out) == pytest.approx(-5.0, abs=1e-6)


def test_output_divergence_single_head_is_zero():
    assert output_divergence(torch.zeros(2, 1, 4)) == 0.0
