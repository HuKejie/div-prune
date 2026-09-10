"""Hydra pruning entry point: dense train -> importance -> prune -> fine-tune.

Usage (from repo root; ``+`` because ``experiment.prune`` is null in the yaml):
    python -m run.pipeline.prune experiment=gat_cora \\
        +experiment.prune='{criterion: diversity, ratio: 0.75, \\
        finetune_epochs: 100, checkpoint_source: late_min_div}'
"""

import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import divprune.data_module.dataset  # noqa: F401 (registration side effects)
import divprune.model_module.model  # noqa: F401
from divprune.data_module import DatasetFactory
from divprune.model_module import ModelFactory
from divprune.seed import set_seed
from divprune.trainer_module.pruning import head_importance, prune_gat_heads, select_keep
from divprune.trainer_module.regularizer import attention_divergence
from divprune.trainer_module.trainer import (
    dense_metrics_from_checkpoint,
    evaluate,
    resolve_device,
    train_model,
)
from run.pipeline.train import RESULT_HEADERS, _append_csv

logger = logging.getLogger(__name__)

# checkpoint_source -> rolling checkpoint file suffix written by the trainer.
_SOURCE_SUFFIX = {"best": "best", "late_min_div": "mindiv", "late_last": "last"}


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    prune_cfg = cfg.experiment.get("prune")
    if prune_cfg is None:
        raise ValueError("experiment.prune must be set for the prune pipeline")
    criterion = prune_cfg.criterion
    ratio = float(prune_cfg.ratio)
    finetune_epochs = int(prune_cfg.finetune_epochs)
    if not 0 <= ratio < 1:
        raise ValueError(f"prune ratio must be in [0, 1), got {ratio}")
    if finetune_epochs < 0:
        raise ValueError(f"finetune_epochs must be >= 0, got {finetune_epochs}")

    data_root = Path(get_original_cwd()) / "data"
    dataset = DatasetFactory(cfg.experiment.dataset)(cfg.experiment.dataset, root=str(data_root))
    data = dataset[0]
    device = resolve_device(cfg)
    data = data.to(device)

    model_cfg = OmegaConf.create(
        {
            **OmegaConf.to_container(cfg.model, resolve=True),
            "in_channels": int(data.x.size(1)),
            "num_classes": int(data.y.max()) + 1,
        }
    )
    # Fine-tuning uses its own epoch budget.
    ft_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    ft_cfg.model.epochs = finetune_epochs

    results_path = Path(get_original_cwd()) / cfg.output_dir / "tables" / "results.csv"
    source = prune_cfg.get("checkpoint_source", "memory")
    if source not in ("memory", "best", "late_min_div", "late_last"):
        raise ValueError(
            f"prune.checkpoint_source must be memory|best|late_min_div|late_last, got {source}"
        )
    extend = bool(prune_cfg.get("extend_to_budget", False))
    reuse_stem = prune_cfg.get("reuse_stem", None)
    if reuse_stem is not None and source == "memory":
        raise ValueError("prune.reuse_stem requires checkpoint_source != memory")
    if source == "memory":
        suffix = None
    else:
        suffix = _SOURCE_SUFFIX[source]
    # Where teacher checkpoints live (defaults to output_dir/checkpoints).
    teacher_dir_cfg = prune_cfg.get("teacher_dir", None)
    teacher_dir = (
        Path(get_original_cwd()) / teacher_dir_cfg
        if teacher_dir_cfg is not None
        else Path(get_original_cwd()) / cfg.output_dir / "checkpoints"
    )
    sched = cfg.experiment.get("lambda_schedule", "fixed")
    if bool(cfg.experiment.adaptive_lambda):
        sched = "controller"
    row_base = {
        "dataset": cfg.experiment.dataset,
        "hidden_per_head": int(cfg.model.hidden_per_head),
        "regularizer": cfg.experiment.regularizer,
        "lambda_fixed": float(cfg.experiment.lambda_fixed),
        "lambda_target": float(cfg.experiment.lambda_target),
        "lambda_eta": float(cfg.experiment.lambda_eta),
        "adaptive_lambda": bool(cfg.experiment.adaptive_lambda),
        "lambda_schedule": sched,
        "prune_criterion": criterion,
        "prune_ratio": ratio,
        "checkpoint_source": source,
        "criterion_div": "",
        "distill": bool(cfg.experiment.distill),
        "distill_alpha": float(cfg.experiment.distill_alpha) if bool(cfg.experiment.distill) else "",
        "distill_temperature": float(cfg.experiment.distill_temperature) if bool(cfg.experiment.distill) else "",
    }

    ckpt_dir = Path(get_original_cwd()) / cfg.output_dir / "checkpoints"

    for run_i in range(int(cfg.experiment.runs)):
        seed = int(cfg.seed) + run_i
        set_seed(seed)
        model = ModelFactory(cfg.model.name)(model_cfg).to(device)
        ckpt_stem = f"{cfg.experiment.dataset}_seed{seed}_sched{sched}" if source != "memory" else None
        if reuse_stem is not None:
            assert suffix is not None
            ckpt_path = teacher_dir / f"{reuse_stem.format(seed=seed)}_{suffix}.pt"
            dense = dense_metrics_from_checkpoint(ckpt_path, model, data, device)
            logger.info(
                f"seed={seed}: reused dense checkpoint {ckpt_path} "
                f"(epoch {dense['best_epoch']:.0f})"
            )
        else:
            dense = train_model(
                cfg,
                model,
                data,
                ckpt_dir=ckpt_dir if source != "memory" else None,
                ckpt_stem=ckpt_stem,
                early_stop=not extend,
            )  # trains dense (with regularizer if configured)

            if source != "memory":
                assert ckpt_stem is not None and suffix is not None
                ckpt_path = ckpt_dir / f"{ckpt_stem}_{suffix}.pt"
                state = torch.load(ckpt_path, map_location=device, weights_only=True)
                model.load_state_dict(state["state_dict"])
                logger.info(
                    f"seed={seed}: criterion model from {ckpt_path} (epoch {state['epoch']})"
                )

        with torch.no_grad():
            model.eval()
            criterion_div = float(
                attention_divergence(model.attention_weights(data.x, data.edge_index))
            )
        row_base["criterion_div"] = criterion_div

        scores = head_importance(model, data, criterion)
        n_keep = max(1, int(model.num_heads * (1 - ratio)))
        keep = select_keep(scores, n_keep, criterion)
        pruned = prune_gat_heads(model, keep).to(device)
        acc_before = evaluate(pruned, data, data.test_mask, device)

        if bool(cfg.experiment.distill):
            with torch.no_grad():
                model.eval()
                teacher_logits = model(data.x, data.edge_index)["logits"].clone()
            ft = train_model(ft_cfg, pruned, data, teacher_logits=teacher_logits)
        else:
            ft = train_model(ft_cfg, pruned, data)
        row = {
            **row_base,
            "seed": seed,
            "pruned_acc_before_ft": acc_before,
            # Dense-phase values (the model the pruning criterion was computed from).
            "dense_test_acc": dense["test_acc"],
            "dense_lambda_final": dense["lambda_final"],
            "dense_div_final": dense["div_final"],
            # Fine-tune-phase values.
            "best_epoch": ft["best_epoch"],
            "epochs_run": ft["epochs_run"],
            "val_acc": ft["val_acc"],
            "test_acc": ft["test_acc"],
            "params": ft["params"],
            "wall_time_s": ft["wall_time_s"],
            "lambda_final": ft["lambda_final"],
            "div_final": ft["div_final"],
        }
        _append_csv(results_path, row)
        logger.info(
            f"seed={seed} {criterion} ratio={ratio}: dense_test={dense['test_acc']:.4f} "
            f"before_ft={acc_before:.4f} after_ft={ft['test_acc']:.4f}"
        )

    logger.info(f"Pruning experiment done ({cfg.experiment.runs} runs). Results: {results_path}")


if __name__ == "__main__":
    main()
