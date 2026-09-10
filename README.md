# div-prune

Code for the IJPRAI submission *Feedback-Controlled Diversity Regularization for
Multi-Head Attention: Enforcing a Similarity Budget Late in Training* (under review).

Feedback-controlled weighting of an inter-head diversity loss ($L_{div}$) for multi-head
attention: a target similarity budget is treated as a control set point, and the
regularization weight is adjusted per epoch via integral control. The same similarity
metric is then reused as a head-pruning criterion, combined with structured pruning and
knowledge distillation.

## Layout

- `run/pipeline/` — training / pruning / analysis scripts (Hydra entry points)
- `run/conf/` — Hydra configs (model, experiment)
- `src/divprune/` — data_module / model_module / trainer_module (registry + factory pattern)
- `paper/` — manuscript LaTeX sources, figures, and tables (`paper/main.pdf` is the compiled submission)
- `analysis-output/` — statistics appendix and analysis reports
- `outputs/tables/results_clean.csv` — merged experiment table behind the paper numbers
- `outputs_rev2_*/tables/results.csv` — second-revision sensitivity and fixed-λ sweep runs
- `colab_divprune.ipynb`, `colab_divprune_h3ext.ipynb` — Colab notebooks for the revision batches B1/B2/B3 and the H3 extension

## Quick start

```bash
uv sync                       # install deps (Colab: pip install -e . --no-deps + torch-geometric)
uv run python -m run.pipeline.train experiment=gat_cora experiment.runs=1 model.epochs=5
```

## Reproducibility

- GAT 2-layer, 8 heads, hidden 8 per head (92,373 parameters); Planetoid split; early stopping patience 100 (max 200 epochs); 10 seeds per configuration.
- Paired Wilcoxon signed-rank tests over identical seeds; full statistics in `analysis-output/stats-appendix.md`.
- Dense teacher checkpoints (B2, ~43 MB) are excluded from git; regenerate them with `colab_divprune.ipynb` (B2 cell) or the Hydra pipeline.

## License

MIT — see `LICENSE`.
