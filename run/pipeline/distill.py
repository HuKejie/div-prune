"""Hydra entry: pure KD baseline (B5) -- dense teacher -> fewer-head student from scratch.

Usage (from repo root):
    python -m run.pipeline.distill experiment=gat_cora \\
        +experiment.prune.ratio=0.5 +experiment.prune.finetune_epochs=200
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
from divprune.trainer_module.trainer import resolve_device, train_model
from run.pipeline.train import RESULT_HEADERS, _append_csv

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    prune_cfg = cfg.experiment.get("prune")
    if prune_cfg is None:
        raise ValueError("experiment.prune.ratio must be set for the distill pipeline")
    ratio = float(prune_cfg.ratio)

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
    # Student: same architecture with fewer heads, trained from scratch with KD.
    student_cfg = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
    student_cfg.num_heads = max(1, int(model_cfg.num_heads * (1 - ratio)))

    results_path = Path(get_original_cwd()) / cfg.output_dir / "tables" / "results.csv"
    sched = cfg.experiment.get("lambda_schedule", "fixed")
    if bool(cfg.experiment.adaptive_lambda):
        sched = "controller"  # backward compatibility with pre-schedule configs
    row_base = {
        "dataset": cfg.experiment.dataset,
        "hidden_per_head": int(cfg.model.hidden_per_head),
        "regularizer": cfg.experiment.regularizer,
        "lambda_fixed": float(cfg.experiment.lambda_fixed),
        "lambda_target": float(cfg.experiment.lambda_target),
        "lambda_eta": float(cfg.experiment.lambda_eta),
        "adaptive_lambda": bool(cfg.experiment.adaptive_lambda),
        "lambda_schedule": sched,
        "prune_criterion": "",  # B5: no pruning, fewer heads from scratch
        "prune_ratio": ratio,
        "checkpoint_source": "",  # B5 teacher is trained in-memory, no reuse
        "criterion_div": "",
        "distill": True,
        "distill_alpha": float(cfg.experiment.distill_alpha),
        "distill_temperature": float(cfg.experiment.distill_temperature),
    }

    for run_i in range(int(cfg.experiment.runs)):
        seed = int(cfg.seed) + run_i
        set_seed(seed)
        teacher = ModelFactory(cfg.model.name)(model_cfg).to(device)
        dense = train_model(cfg, teacher, data)

        with torch.no_grad():
            teacher.eval()
            teacher_logits = teacher(data.x, data.edge_index)["logits"].clone()

        student = ModelFactory(cfg.model.name)(student_cfg).to(device)
        ft_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        ft_cfg.model.epochs = int(prune_cfg.finetune_epochs)
        result = train_model(ft_cfg, student, data, teacher_logits=teacher_logits)

        row = {
            **row_base,
            "seed": seed,
            "dense_test_acc": dense["test_acc"],
            "best_epoch": result["best_epoch"],
            "epochs_run": result["epochs_run"],
            "val_acc": result["val_acc"],
            "test_acc": result["test_acc"],
            "params": result["params"],
            "wall_time_s": result["wall_time_s"],
            "lambda_final": result["lambda_final"],
            "div_final": result["div_final"],
        }
        _append_csv(results_path, row)
        logger.info(
            f"seed={seed} KD-from-scratch ratio={ratio}: dense={dense['test_acc']:.4f} "
            f"student={result['test_acc']:.4f}"
        )

    logger.info(f"Distill experiment done ({cfg.experiment.runs} runs). Results: {results_path}")


if __name__ == "__main__":
    main()
