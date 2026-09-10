"""Per-epoch weighting policies for the divergence regularization term.

Policies (``experiment.lambda_schedule``):

- ``fixed``:      constant lambda (paper baseline; default)
- ``controller``: integral feedback controller (core contribution)
- ``linear``:     open-loop ramp from 0 to lambda_max (M1 adaptive baseline)
- ``kendall``:    Kendall & Gal 2018 uncertainty weighting (M1 adaptive baseline)
- ``gradnorm``:   GradNorm gradient balancing (M1 adaptive baseline)

Every policy exposes one scalar per epoch, ``lambda_value()``, recorded in
traces and the results CSV as the effective divergence weight. Kendall and
GradNorm additionally rescale the task loss; see their docstrings.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from divprune.trainer_module.adaptive_lambda import DivergenceController

VALID_SCHEDULES = ("fixed", "controller", "linear", "kendall", "gradnorm")


@dataclass
class ScheduleContext:
    """Per-epoch inputs handed to a schedule's ``update``.

    ``task_loss_fn``/``div_loss_fn`` are only invoked by gradient-based
    policies (gradnorm); other policies ignore them.
    """

    epoch: int
    div_t: float
    model: nn.Module
    device: torch.device
    task_loss_fn: Optional[Callable[[], torch.Tensor]] = None
    div_loss_fn: Optional[Callable[[], torch.Tensor]] = None


class LambdaSchedule(ABC):
    """Base class for per-epoch divergence-weight policies."""

    @abstractmethod
    def update(self, ctx: ScheduleContext) -> None:
        """Update policy state before the epoch's training step."""

    def combine(self, task_loss: torch.Tensor, div_loss: torch.Tensor) -> torch.Tensor:
        """Combine task and divergence losses into the training objective."""
        return task_loss + self.lambda_value() * div_loss

    @abstractmethod
    def lambda_value(self) -> float:
        """Effective weight of the divergence term (for traces and CSV)."""

    def extra_parameters(self) -> Iterator[nn.Parameter]:
        """Learnable policy parameters to hand to the optimizer."""
        return iter(())


class FixedSchedule(LambdaSchedule):
    """Constant lambda (the fixed-lambda baseline)."""

    def __init__(self, lam: float) -> None:
        if not math.isfinite(lam) or lam < 0:
            raise ValueError(f"lam must be finite and >= 0, got {lam}")
        self.lam = lam

    def update(self, ctx: ScheduleContext) -> None:
        pass

    def lambda_value(self) -> float:
        return self.lam


class ControllerSchedule(LambdaSchedule):
    """Wrapper around the integral feedback controller (core contribution)."""

    def __init__(self, target: float, eta: float, lambda_max: float) -> None:
        self.controller = DivergenceController(target, eta, lambda_max)

    def update(self, ctx: ScheduleContext) -> None:
        self.controller.update(ctx.div_t)

    def lambda_value(self) -> float:
        return self.controller.lambda_val


class LinearSchedule(LambdaSchedule):
    """Open-loop ramp: 0 before ``t0``, then linear to ``lambda_max`` over ``ramp`` epochs.

    Calibrated to the controller's lambda slope on Cora (see gat_cora.yaml);
    transferred unchanged to the other datasets.
    """

    def __init__(self, t0: int, ramp: int, lambda_max: float) -> None:
        if t0 < 0:
            raise ValueError(f"t0 must be >= 0, got {t0}")
        if ramp <= 0:
            raise ValueError(f"ramp must be positive, got {ramp}")
        if not math.isfinite(lambda_max) or lambda_max < 0:
            raise ValueError(f"lambda_max must be finite and >= 0, got {lambda_max}")
        self.t0 = t0
        self.ramp = ramp
        self.lambda_max = lambda_max
        self._lam = 0.0

    def update(self, ctx: ScheduleContext) -> None:
        t = ctx.epoch - self.t0
        if t <= 0:
            self._lam = 0.0
        else:
            self._lam = self.lambda_max * min(1.0, t / self.ramp)

    def lambda_value(self) -> float:
        return self._lam


class KendallSchedule(LambdaSchedule):
    """Uncertainty weighting (Kendall & Gal 2018) for a two-term loss.

    Total objective (with ``s_ce``, ``s_div`` learnable):

        L = exp(-s_ce) * L_task + exp(-s_div) * L_div + (s_ce + s_div) / 2

    Both ``s`` start at 0 (weights of 1, i.e. plain summation). The reported
    lambda is the effective divergence weight ``exp(-s_div)``.
    """

    def __init__(self) -> None:
        self.s_ce = nn.Parameter(torch.zeros(()))
        self.s_div = nn.Parameter(torch.zeros(()))

    def update(self, ctx: ScheduleContext) -> None:
        pass  # weights move via gradients on the combined loss

    def combine(self, task_loss: torch.Tensor, div_loss: torch.Tensor) -> torch.Tensor:
        w_ce = torch.exp(-self.s_ce)
        w_div = torch.exp(-self.s_div)
        return w_ce * task_loss + w_div * div_loss + 0.5 * (self.s_ce + self.s_div)

    def lambda_value(self) -> float:
        return float(torch.exp(-self.s_div).detach())

    def extra_parameters(self) -> Iterator[nn.Parameter]:
        return iter((self.s_ce, self.s_div))


class GradNormSchedule(LambdaSchedule):
    """GradNorm (Chen et al. 2018) for two loss terms.

    At each epoch the gradient norms of both terms w.r.t. the shared model
    weights are measured, and the divergence weight ``w_div`` is nudged toward
    equalizing the normalized norms (``alpha=0`` targets equal magnitudes).
    Weights are renormalized so ``w_ce + w_div = 2``. Loss baselines from
    epoch 0 feed the inverse-training-rate terms.
    """

    def __init__(self, eta: float = 0.16, alpha: float = 0.0) -> None:
        if not math.isfinite(eta) or eta <= 0:
            raise ValueError(f"eta must be finite and positive, got {eta}")
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError(f"alpha must be finite and >= 0, got {alpha}")
        self.eta = eta
        self.alpha = alpha
        self.w_ce = 1.0
        self.w_div = 1.0
        self._l0_ce: Optional[float] = None
        self._l0_div: Optional[float] = None

    def _grad_norm(self, model: nn.Module, loss: torch.Tensor) -> float:
        """Full-model gradient norm of ``loss``; zeroes grads afterwards."""
        model.zero_grad(set_to_none=True)
        loss.backward()
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += float((p.grad**2).sum())
        model.zero_grad(set_to_none=True)
        return total**0.5

    def update(self, ctx: ScheduleContext) -> None:
        if ctx.task_loss_fn is None or ctx.div_loss_fn is None:
            raise ValueError("gradnorm requires task_loss_fn and div_loss_fn")
        if ctx.epoch == 0:
            with torch.no_grad():
                self._l0_ce = float(ctx.task_loss_fn().detach())
                self._l0_div = float(ctx.div_loss_fn().detach())
            return

        loss_ce = ctx.task_loss_fn()
        g_ce = self._grad_norm(ctx.model, loss_ce)
        loss_div = ctx.div_loss_fn()
        g_div = self._grad_norm(ctx.model, self.w_div * loss_div)

        # Inverse training rates, normalized to mean 1 (GradNorm Eq. 4).
        r_ce = float(loss_ce.detach()) / float(self._l0_ce)  # type: ignore[arg-type]
        r_div = float(loss_div.detach()) / float(self._l0_div)  # type: ignore[arg-type]
        mean_r = 0.5 * (r_ce + r_div)
        r_ce /= mean_r
        r_div /= mean_r

        g_mean = 0.5 * (g_ce + g_div)
        g_ce_n = g_ce / g_mean
        g_div_n = g_div / g_mean
        self.w_ce = max(0.05, self.w_ce + self.eta * (r_ce**self.alpha - g_ce_n))
        self.w_div = max(0.05, self.w_div + self.eta * (r_div**self.alpha - g_div_n))
        total = self.w_ce + self.w_div
        self.w_ce = 2.0 * self.w_ce / total
        self.w_div = 2.0 * self.w_div / total

    def combine(self, task_loss: torch.Tensor, div_loss: torch.Tensor) -> torch.Tensor:
        return self.w_ce * task_loss + self.w_div * div_loss

    def lambda_value(self) -> float:
        return self.w_div


def build_schedule(cfg: DictConfig) -> LambdaSchedule:
    """Instantiate the lambda policy from the experiment config."""
    exp = cfg.experiment
    name = exp.get("lambda_schedule", "fixed")
    if bool(exp.get("adaptive_lambda", False)):
        name = "controller"  # backward compatibility with pre-schedule configs
    if name == "fixed":
        return FixedSchedule(float(exp.lambda_fixed))
    if name == "controller":
        return ControllerSchedule(
            float(exp.lambda_target),
            float(exp.lambda_eta),
            float(exp.lambda_max),
        )
    if name == "linear":
        return LinearSchedule(
            int(exp.get("schedule_t0", 0)),
            int(exp.get("schedule_ramp", 100)),
            float(exp.get("schedule_lambda_max", exp.lambda_max)),
        )
    if name == "kendall":
        return KendallSchedule()
    if name == "gradnorm":
        return GradNormSchedule(
            float(exp.get("gradnorm_eta", 0.16)),
            float(exp.get("gradnorm_alpha", 0.0)),
        )
    raise ValueError(f"Unknown lambda_schedule: {name}. Available: {VALID_SCHEDULES}")
