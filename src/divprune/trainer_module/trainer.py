"""Training loop with optional divergence regularization and early stopping."""

import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Data

from divprune.data_module.metrics import accuracy
from divprune.trainer_module.lambda_schedule import (
    FixedSchedule,
    LambdaSchedule,
    ScheduleContext,
    build_schedule,
)
from divprune.trainer_module.regularizer import DivergenceRegularizer, attention_divergence

logger = logging.getLogger(__name__)


def resolve_device(cfg: DictConfig) -> torch.device:
    """Map cfg.device to a torch device."""
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


@torch.no_grad()
def evaluate(model: nn.Module, data: Data, mask: torch.Tensor, device: torch.device) -> float:
    """Evaluate node-classification accuracy on a mask."""
    model.eval()
    out = model(data.x, data.edge_index)["logits"]
    return accuracy(out.argmax(dim=1)[mask], data.y[mask])


def _write_trace(trace_path: Path, rows: List[Tuple[int, float, float, float]]) -> None:
    """Persist the per-epoch (epoch, div_t, lambda_t, val_acc) trace of adaptive runs."""
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "div_t", "lambda_t", "val_acc"])
        writer.writerows(rows)


def dense_metrics_from_checkpoint(
    ckpt_path: Path, model: nn.Module, data: Data, device: torch.device
) -> Dict[str, float]:
    """Rebuild the dense-phase metrics dict from a saved rolling checkpoint.

    Mirrors ``train_model``'s return shape so prune's ``reuse_stem`` mode can
    serve one teacher to all criterion/source combinations of the M2 matrix.
    """
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    model.eval()
    with torch.no_grad():
        test_acc = evaluate(model, data, data.test_mask, device)
    return {
        "test_acc": test_acc,
        "val_acc": float(state["val_acc"]),
        "lambda_final": float(state["lambda_t"]),
        "div_final": float(state["div_t"]),
        "best_epoch": float(state["epoch"]),
        "epochs_run": float(state["epoch"] + 1),
        "params": float(sum(p.numel() for p in model.parameters())),
        "wall_time_s": float("nan"),
    }


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    val_acc: float,
    div_t: float,
    lam: float,
) -> None:
    """Persist a model checkpoint (state dict + bookkeeping)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "div_t": div_t,
            "lambda_t": lam,
        },
        path,
    )


def train_model(
    cfg: DictConfig,
    model: nn.Module,
    data: Data,
    trace_path: Optional[Path] = None,
    teacher_logits: Optional[torch.Tensor] = None,
    ckpt_dir: Optional[Path] = None,
    ckpt_stem: Optional[str] = None,
    early_stop: bool = True,
) -> Dict[str, float]:
    """Train one model instance and return best-val checkpoint metrics.

    With a divergence regularizer enabled, conv1 runs three times per step
    (logits, attention weights, head outputs). This is acceptable for the
    small models in scope (GAT-scale, ~90k-740k params depending on config)
    and keeps the forward path simple.

    When both ``ckpt_dir`` and ``ckpt_stem`` are given, three rolling
    checkpoints are written: ``{stem}_best.pt`` (best val accuracy),
    ``{stem}_mindiv.pt`` (lowest measured divergence), and ``{stem}_last.pt``
    (final epoch). ``early_stop=False`` trains the full epoch budget.
    """
    device = resolve_device(cfg)
    model = model.to(device)
    data = data.to(device)

    use_reg = cfg.experiment.regularizer != "none"
    regularizer = DivergenceRegularizer(cfg.experiment.regularizer) if use_reg else None
    schedule = build_schedule(cfg)
    if not isinstance(schedule, FixedSchedule) and not use_reg:
        raise ValueError("a non-fixed lambda schedule requires a divergence regularizer")

    extra_params = list(schedule.extra_parameters())
    param_groups: List[Dict] = [{"params": list(model.parameters())}]
    if extra_params:
        param_groups.append({"params": extra_params, "weight_decay": 0.0})
    optimizer = torch.optim.Adam(
        param_groups,
        lr=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
    )

    distill = teacher_logits is not None
    alpha_kd = float(cfg.experiment.distill_alpha) if distill else 0.0
    temperature = float(cfg.experiment.distill_temperature) if distill else 1.0

    def _task_loss() -> torch.Tensor:
        """Current task loss (CE, optionally KD-mixed). Used by gradient-based schedules."""
        out = model(data.x, data.edge_index)["logits"]
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        if distill:
            kd = F.kl_div(
                F.log_softmax(out / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature * temperature)
            loss = (1 - alpha_kd) * loss + alpha_kd * kd
        return loss

    def _div_loss() -> torch.Tensor:
        """Current raw divergence term (unweighted)."""
        assert regularizer is not None
        alpha = model.attention_weights(data.x, data.edge_index)
        head_out = model.head_outputs(data.x, data.edge_index)
        return regularizer(alpha, head_out)

    save_ckpt = ckpt_dir is not None and ckpt_stem is not None
    ckpt_best = ckpt_dir / f"{ckpt_stem}_best.pt" if save_ckpt else None
    ckpt_mindiv = ckpt_dir / f"{ckpt_stem}_mindiv.pt" if save_ckpt else None
    ckpt_last = ckpt_dir / f"{ckpt_stem}_last.pt" if save_ckpt else None

    best_val_acc = -float("inf")  # epoch 0 always qualifies, so {stem}_best.pt always exists
    best_test_acc = 0.0
    best_epoch = -1
    lam_best = schedule.lambda_value()
    div_best = float("nan")
    min_div_val = float("inf")
    trace_rows: List[Tuple[int, float, float, float]] = []
    patience_left = int(cfg.model.early_stop_patience)
    start = time.time()

    for epoch in range(int(cfg.model.epochs)):
        model.train()
        if use_reg:
            # Deterministic controller input: dropout-masked coefficients would
            # bias the similarity estimate (~0.4x) and distort the feedback signal.
            with torch.no_grad():
                model.eval()
                div_t = float(attention_divergence(model.attention_weights(data.x, data.edge_index)))
                model.train()
        else:
            div_t = float("nan")

        schedule.update(
            ScheduleContext(
                epoch=epoch,
                div_t=div_t,
                model=model,
                device=device,
                task_loss_fn=_task_loss,
                div_loss_fn=_div_loss,
            )
        )
        lam = schedule.lambda_value()

        optimizer.zero_grad()
        loss = _task_loss()
        if use_reg:
            loss = schedule.combine(loss, _div_loss())
        loss.backward()
        optimizer.step()

        val_acc = evaluate(model, data, data.val_mask, device)
        if not isinstance(schedule, FixedSchedule):
            trace_rows.append((epoch, div_t, lam, val_acc))

        if use_reg and div_t < min_div_val:
            min_div_val = div_t
            if save_ckpt:
                assert ckpt_mindiv is not None
                _save_checkpoint(ckpt_mindiv, model, epoch, val_acc, div_t, lam)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = evaluate(model, data, data.test_mask, device)
            best_epoch = epoch
            lam_best = lam
            patience_left = int(cfg.model.early_stop_patience)
            if use_reg:
                # Head similarity at the best checkpoint (deterministic, eval mode).
                with torch.no_grad():
                    model.eval()
                    alpha_b = model.attention_weights(data.x, data.edge_index)
                    div_best = float(attention_divergence(alpha_b))
                    model.train()
            if save_ckpt:
                assert ckpt_best is not None
                _save_checkpoint(ckpt_best, model, epoch, val_acc, div_t, lam)
        elif early_stop:
            patience_left -= 1
            if patience_left <= 0:
                logger.info(f"Early stop at epoch {epoch}")
                break

    if save_ckpt:
        assert ckpt_last is not None
        _save_checkpoint(ckpt_last, model, epoch, val_acc, div_t, lam)

    if trace_path is not None and trace_rows:
        _write_trace(trace_path, trace_rows)

    return {
        "best_epoch": float(best_epoch),
        "epochs_run": float(epoch + 1),
        "val_acc": best_val_acc,
        "test_acc": best_test_acc,
        "params": float(sum(p.numel() for p in model.parameters())),
        "wall_time_s": time.time() - start,
        "lambda_final": lam_best,
        "div_final": div_best,
    }
