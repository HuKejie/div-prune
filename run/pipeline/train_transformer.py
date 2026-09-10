"""Hydra entry for the Transformer/Multi30k leg.

Usage (from repo root):
    python -m run.pipeline.train_transformer model=transformer experiment=multi30k \\
        experiment.runs=5 model.epochs=50 seed=0
"""

import logging
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import divprune.model_module.model  # noqa: F401 (registration side effects)
from divprune.data_module.dataset.multi30k import load_multi30k
from divprune.model_module import ModelFactory
from divprune.seed import set_seed
from divprune.trainer_module.transformer_trainer import train_transformer
from run.pipeline.train import RESULT_HEADERS, _append_csv

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    data_root = Path(get_original_cwd()) / "data"
    data = load_multi30k(root=str(data_root), vocab_size=int(cfg.model.vocab_size))

    model_cfg = OmegaConf.create(
        {
            **OmegaConf.to_container(cfg.model, resolve=True),
            "src_vocab": int(data["src_vocab"]),
            "tgt_vocab": int(data["tgt_vocab"]),
            "pad_idx": int(data["pad_idx"]),
        }
    )

    results_path = Path(get_original_cwd()) / cfg.output_dir / "tables" / "results.csv"
    adaptive = bool(cfg.experiment.adaptive_lambda)
    trace_dir = Path(get_original_cwd()) / cfg.output_dir / "traces"

    row_base = {
        "dataset": cfg.experiment.dataset,
        "hidden_per_head": int(cfg.model.num_heads),
        "regularizer": cfg.experiment.regularizer,
        "lambda_fixed": float(cfg.experiment.lambda_fixed),
        "lambda_target": float(cfg.experiment.lambda_target),
        "lambda_eta": float(cfg.experiment.lambda_eta),
        "adaptive_lambda": bool(cfg.experiment.adaptive_lambda),
        "prune_criterion": "",
        "prune_ratio": "",
        "distill": False,
        "distill_alpha": "",
        "distill_temperature": "",
    }

    for run_i in range(int(cfg.experiment.runs)):
        seed = int(cfg.seed) + run_i
        set_seed(seed)
        model = ModelFactory(cfg.model.name)(model_cfg)
        result = train_transformer(
            cfg,
            model,
            data,
            trace_path=trace_dir / f"multi30k_seed{seed}_t{cfg.experiment.lambda_target}"
            f"_e{cfg.experiment.lambda_eta}.csv"
            if adaptive
            else None,
        )
        # Multi30k rows: val_acc/test_acc columns carry BLEU scores.
        row = {**row_base, "seed": seed, **result}
        _append_csv(results_path, row)
        logger.info(
            f"seed={seed}: val_bleu={result['val_acc']:.2f} test_bleu={result['test_acc']:.2f} "
            f"-> {results_path}"
        )

    logger.info(f"Transformer experiment done ({cfg.experiment.runs} runs). Results: {results_path}")


if __name__ == "__main__":
    main()
