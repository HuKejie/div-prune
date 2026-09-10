"""Unit tests for the divergence feedback controller."""

import pytest

from divprune.trainer_module.adaptive_lambda import DivergenceController


def test_lambda_rises_above_target():
    c = DivergenceController(target=0.7, eta=0.1, lambda_max=10.0)
    assert c.update(0.9) == pytest.approx(0.02)
    assert c.update(0.9) == pytest.approx(0.04)


def test_lambda_falls_below_target():
    c = DivergenceController(target=0.7, eta=0.1, lambda_max=10.0)
    c.update(0.9)
    assert c.update(0.5) == pytest.approx(0.0)  # clamped at 0


def test_lambda_clamped_at_max():
    c = DivergenceController(target=0.0, eta=1.0, lambda_max=1.0)
    c.update(1.0)
    c.update(1.0)
    assert c.lambda_val == 1.0


def test_invalid_params():
    with pytest.raises(ValueError):
        DivergenceController(target=1.5, eta=0.1, lambda_max=1.0)
    with pytest.raises(ValueError):
        DivergenceController(target=0.5, eta=0.0, lambda_max=1.0)
    with pytest.raises(ValueError):
        DivergenceController(target=0.5, eta=float("nan"), lambda_max=1.0)
    with pytest.raises(ValueError):
        DivergenceController(target=0.5, eta=0.1, lambda_max=-1.0)


def test_nan_div_leaves_lambda_unchanged():
    c = DivergenceController(target=0.7, eta=0.1, lambda_max=10.0)
    c.update(0.9)
    assert c.update(float("nan")) == pytest.approx(0.02)
