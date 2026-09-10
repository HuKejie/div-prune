"""Hydra training entry point: run an experiment config over N seeds, one CSV row per run.

Usage (from repo root):
    python -m run.pipeline.train experiment=gat_cora experiment.runs=1 model.epochs=5
"""

import csv
import logging
from pathlib import Path
from typing import Dict

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from divprune.data_module import DatasetFactory
from divprune.model_module import ModelFactory
from divprune.seed import set_seed
from divprune.trainer_module.trainer import train_model

# Subpackage imports trigger auto-registration of datasets and models.
import divprune.data_module.dataset  # noqa: F401
import divprune.model_module.model  # noqa: F401

logger = logging.getLogger(__name__)

RESULT_HEADERS = [
    "dataset", "seed", "hidden_per_head", "regularizer", "lambda_fixed",
    "lambda_target", "lambda_eta", "lambda_final", "div_final", "adaptive_lambda",
    "lambda_schedule", "prune_criterion", "prune_ratio", "pruned_acc_before_ft",
    "checkpoint_source", "criterion_div", "distill",
    "distill_alpha", "distill_temperature",
    "dense_test_acc", "dense_lambda_final", "dense_div_final",
    "best_epoch", "epochs_run", "val_acc", "test_acc", "params", "wall_time_s",
]


def _append_csv(path: Path, row: Dict) -> None:
    """Append one result row, writing the header if the file is missing or empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    dataset_cls = DatasetFactory(cfg.experiment.dataset)
    # Anchor data root to the repo (Hydra chdirs into per-job output dirs).
    data_root = Path(get_original_cwd()) / "data"
    dataset = dataset_cls(cfg.experiment.dataset, root=str(data_root))
    data = dataset[0]

    model_cfg = OmegaConf.create(
        {
            **OmegaConf.to_container(cfg.model, resolve=True),
            # PyG 2.8 Data no longer carries num_features/num_classes; infer from tensors.
            "in_channels": int(data.x.size(1)),
            "num_classes": int(data.y.max()) + 1,
        }
    )

    # Absolute path: Hydra chdirs into the per-job output dir.
    results_path = Path(get_original_cwd()) / cfg.output_dir / "tables" / "results.csv"

    prune_cfg = cfg.experiment.get("prune")
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
        # Tolerate a ratio-only prune cfg (B4 from-scratch small models: no
        # pruning, ratio labels the head budget).
        "prune_criterion": prune_cfg.get("criterion", "") if prune_cfg else "",
        "prune_ratio": float(prune_cfg.ratio) if prune_cfg else "",
        "checkpoint_source": "",
        "criterion_div": "",
        "distill": bool(cfg.experiment.distill),
        "distill_alpha": float(cfg.experiment.distill_alpha) if bool(cfg.experiment.distill) else "",
        "distill_temperature": float(cfg.experiment.distill_temperature) if bool(cfg.experiment.distill) else "",
    }

    trace_dir = Path(get_original_cwd()) / cfg.output_dir / "traces"
    ckpt_dir = Path(get_original_cwd()) / cfg.output_dir / "checkpoints"
    save_ckpt = bool(cfg.experiment.get("save_checkpoints", False))

    for run_i in range(int(cfg.experiment.runs)):
        seed = int(cfg.seed) + run_i
        set_seed(seed)
        model = ModelFactory(cfg.model.name)(model_cfg)
        trace_path = None
        if sched != "fixed":
            if sched == "controller":
                trace_path = trace_dir / (
                    f"{cfg.experiment.dataset}_seed{seed}_t{cfg.experiment.lambda_target}"
                    f"_e{cfg.experiment.lambda_eta}.csv"
                )
            else:
                trace_path = trace_dir / f"{cfg.experiment.dataset}_seed{seed}_sched{sched}.csv"
        result = train_model(
            cfg,
            model,
            data,
            trace_path=trace_path,
            ckpt_dir=ckpt_dir if save_ckpt else None,
            ckpt_stem=f"{cfg.experiment.dataset}_seed{seed}_sched{sched}" if save_ckpt else None,
        )
        _append_csv(results_path, {**row_base, "seed": seed, **result})
        logger.info(
            f"seed={seed}: val={result['val_acc']:.4f} test={result['test_acc']:.4f} "
            f"({result['epochs_run']:.0f} epochs) -> {results_path}"
        )

    logger.info(f"Experiment done ({cfg.experiment.runs} runs). Results: {results_path}")


if __name__ == "__main__":
    main()
