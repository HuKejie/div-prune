"""Unit and integration tests for the lambda schedule policies (M1 revision)."""

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch_geometric.data import Data

import divprune.model_module.model  # noqa: F401 (registration side effect)
from divprune.model_module import ModelFactory
from divprune.trainer_module.lambda_schedule import (
    ControllerSchedule,
    FixedSchedule,
    GradNormSchedule,
    KendallSchedule,
    LinearSchedule,
    ScheduleContext,
    build_schedule,
)
from divprune.trainer_module.trainer import train_model


def _ctx(epoch: int = 1, div_t: float = 0.8, model: nn.Module | None = None) -> ScheduleContext:
    if model is None:
        model = nn.Linear(2, 2)
    return ScheduleContext(
        epoch=epoch,
        div_t=div_t,
        model=model,
        device=torch.device("cpu"),
        task_loss_fn=lambda: model(torch.ones(4, 2)).sum(),
        div_loss_fn=lambda: (model(torch.ones(4, 2)) ** 2).mean(),
    )


# --- factory ---

def test_build_schedule_maps_all_policies():
    for name, cls in [
        ("fixed", FixedSchedule),
        ("controller", ControllerSchedule),
        ("linear", LinearSchedule),
        ("kendall", KendallSchedule),
        ("gradnorm", GradNormSchedule),
    ]:
        cfg = OmegaConf.create(
            {
                "experiment": {
                    "regularizer": "attention",
                    "lambda_schedule": name,
                    "lambda_fixed": 0.1,
                    "lambda_target": 0.7,
                    "lambda_eta": 0.05,
                    "lambda_max": 3.0,
                    "adaptive_lambda": False,
                    "schedule_t0": 0,
                    "schedule_ramp": 100,
                }
            }
        )
        assert isinstance(build_schedule(cfg), cls)


def test_build_schedule_adaptive_lambda_alias_forces_controller():
    cfg = OmegaConf.create(
        {
            "experiment": {
                "regularizer": "attention",
                "lambda_schedule": "fixed",
                "lambda_fixed": 0.1,
                "lambda_target": 0.7,
                "lambda_eta": 0.05,
                "lambda_max": 3.0,
                "adaptive_lambda": True,
            }
        }
    )
    assert isinstance(build_schedule(cfg), ControllerSchedule)


def test_build_schedule_unknown_name_raises():
    cfg = OmegaConf.create({"experiment": {"lambda_schedule": "bogus", "adaptive_lambda": False}})
    with pytest.raises(ValueError):
        build_schedule(cfg)


# --- fixed / linear / controller ---

def test_fixed_schedule_returns_constant():
    sched = FixedSchedule(0.5)
    sched.update(_ctx())
    assert sched.lambda_value() == 0.5
    assert sched.combine(torch.tensor(1.0), torch.tensor(2.0)).item() == pytest.approx(2.0)


def test_linear_schedule_ramp_and_cap():
    sched = LinearSchedule(t0=10, ramp=20, lambda_max=2.0)
    sched.update(_ctx(epoch=5))
    assert sched.lambda_value() == 0.0
    sched.update(_ctx(epoch=20))
    assert sched.lambda_value() == pytest.approx(1.0)
    sched.update(_ctx(epoch=50))
    assert sched.lambda_value() == pytest.approx(2.0)


def test_linear_schedule_rejects_bad_params():
    with pytest.raises(ValueError):
        LinearSchedule(t0=-1, ramp=10, lambda_max=1.0)
    with pytest.raises(ValueError):
        LinearSchedule(t0=0, ramp=0, lambda_max=1.0)
    with pytest.raises(ValueError):
        LinearSchedule(t0=0, ramp=10, lambda_max=math.nan)


def test_controller_schedule_rises_above_target():
    sched = ControllerSchedule(target=0.7, eta=0.05, lambda_max=3.0)
    sched.update(_ctx(div_t=0.9))
    assert sched.lambda_value() == pytest.approx(0.01)
    sched.update(_ctx(div_t=0.6))
    assert sched.lambda_value() == pytest.approx(0.005)


# --- kendall ---

def test_kendall_combine_matches_uncertainty_form():
    sched = KendallSchedule()
    task = torch.tensor(2.0, requires_grad=True)
    div = torch.tensor(3.0, requires_grad=True)
    total = sched.combine(task, div)
    expected = 2.0 + 3.0 + 0.0  # s_ce = s_div = 0 at init
    assert total.item() == pytest.approx(expected)
    assert sched.lambda_value() == pytest.approx(1.0)
    assert len(list(sched.extra_parameters())) == 2


def test_kendall_weights_are_learnable():
    sched = KendallSchedule()
    assert all(p.requires_grad for p in sched.extra_parameters())
    with torch.no_grad():
        sched.s_div.fill_(math.log(2.0))  # exp(-ln 2) == 0.5 exactly
    assert sched.lambda_value() == pytest.approx(0.5)


# --- gradnorm ---

def test_gradnorm_epoch0_records_baselines():
    sched = GradNormSchedule()
    sched.update(_ctx(epoch=0))
    assert sched._l0_ce is not None and sched._l0_div is not None
    assert sched.lambda_value() == 1.0


def test_gradnorm_update_rebalances_and_renormalizes():
    sched = GradNormSchedule(eta=0.16, alpha=0.0)
    model = nn.Linear(2, 2)
    sched.update(_ctx(epoch=0, model=model))
    before = (sched.w_ce, sched.w_div)
    sched.update(_ctx(epoch=1, model=model))
    assert sched.w_div != before[1]
    assert sched.w_ce + sched.w_div == pytest.approx(2.0)
    assert sched.w_div >= 0.05 and sched.w_ce >= 0.05


def test_gradnorm_requires_loss_callables():
    sched = GradNormSchedule()
    ctx = ScheduleContext(
        epoch=1, div_t=0.8, model=nn.Linear(2, 2), device=torch.device("cpu")
    )
    with pytest.raises(ValueError):
        sched.update(ctx)


def test_gradnorm_combine_scales_both_terms():
    sched = GradNormSchedule()
    sched.w_ce = 1.5
    sched.w_div = 0.5
    total = sched.combine(torch.tensor(4.0), torch.tensor(2.0))
    assert total.item() == pytest.approx(1.5 * 4.0 + 0.5 * 2.0)


# --- integration ---

def _tiny_cfg(schedule: str, **overrides) -> dict:
    exp = {
        "dataset": "cora",
        "regularizer": "attention",
        "lambda_fixed": 0.0,
        "lambda_schedule": schedule,
        "adaptive_lambda": False,
        "lambda_target": 0.7,
        "lambda_eta": 0.05,
        "lambda_max": 3.0,
        "schedule_t0": 0,
        "schedule_ramp": 5,
        "schedule_lambda_max": 2.0,
        "save_checkpoints": False,
    }
    exp.update(overrides)
    return {
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
        "experiment": exp,
    }


def _tiny_data() -> Data:
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    y = torch.randint(0, 2, (5,))
    mask = torch.zeros(5, dtype=torch.bool)
    mask[:3] = True
    return Data(x=x, edge_index=edge_index, y=y, train_mask=mask, val_mask=~mask, test_mask=~mask)


@pytest.mark.parametrize("schedule", ["linear", "kendall", "gradnorm", "controller"])
def test_train_integration_all_schedules(schedule, tmp_path: Path):
    cfg = OmegaConf.create(_tiny_cfg(schedule))
    model = ModelFactory("gat")(cfg.model)
    trace = tmp_path / "trace.csv"
    result = train_model(cfg, model, _tiny_data(), trace_path=trace)
    assert result["test_acc"] == result["test_acc"]  # finite
    assert trace.exists()
    rows = [line.strip().split(",") for line in trace.read_text().splitlines()]
    assert rows[0] == ["epoch", "div_t", "lambda_t", "val_acc"]
    assert len(rows) == 4  # header + 3 epochs


def test_train_integration_saves_rolling_checkpoints(tmp_path: Path):
    cfg = OmegaConf.create(_tiny_cfg("linear"))
    model = ModelFactory("gat")(cfg.model)
    result = train_model(
        cfg, model, _tiny_data(), ckpt_dir=tmp_path, ckpt_stem="smoke"
    )
    assert result["test_acc"] == result["test_acc"]
    for suffix in ("best", "mindiv", "last"):
        ckpt = tmp_path / f"smoke_{suffix}.pt"
        assert ckpt.exists()
        state = torch.load(ckpt, weights_only=True)
        assert "state_dict" in state and "epoch" in state
        assert "lambda_t" in state  # M2 reuse mode rebuilds dense metrics from this


def test_train_integration_nonfixed_schedule_without_regularizer_raises():
    cfg = OmegaConf.create(_tiny_cfg("linear"))
    cfg.experiment.regularizer = "none"
    model = ModelFactory("gat")(cfg.model)
    with pytest.raises(ValueError):
        train_model(cfg, model, _tiny_data())
