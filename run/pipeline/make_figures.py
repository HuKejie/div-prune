"""Generate analysis-bundle figures from outputs/tables/results_clean.csv and outputs/traces/.

Run from repo root: uv run python -m run.pipeline.make_figures
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "outputs" / "tables" / "results_clean.csv"
TRACES = ROOT / "outputs" / "traces"
FIGDIR = ROOT / "analysis-output" / "figures"

# Okabe-Ito CVD-safe categorical palette (fixed order: blue -> orange -> green)
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
INK, MUTED = "#1a1a1a", "#6b6b6b"


def _load() -> list[dict]:
    return list(csv.DictReader(CSV.open(encoding="utf-8")))


def _dense(rows: list[dict]) -> list[dict]:
    # Dense training rows only (excludes prune rows and pure-KD B5 rows).
    return [r for r in rows
            if r["prune_criterion"] == "" and r["hidden_per_head"] == "8"
            and r["distill"] == "False"]


def _mean_std(sub: list[dict], key: str = "test_acc") -> tuple[float, float]:
    vals = np.array([float(r[key]) for r in sub if r[key] not in ("", "nan")])
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std())


def _b0(dense, ds):
    return [r for r in dense if r["dataset"] == ds and r["regularizer"] == "none"]


def _fixed20(dense, ds):
    return [r for r in dense if r["dataset"] == ds and r["regularizer"] == "attention"
            and r["lambda_fixed"] == "20.0"]


def _adaptive(dense, ds):
    return [r for r in dense if r["dataset"] == ds and r["adaptive_lambda"] == "True"
            and r["lambda_target"] == "0.7" and r["lambda_eta"] == "0.05"]


def fig1_h1(rows: list[dict]) -> None:
    """Main comparison: test accuracy across 3 datasets x 3 training configs."""
    dense = _dense(rows)
    datasets = ["cora", "citeseer", "pubmed"]
    configs = [("Baseline", _b0, BLUE), ("Fixed λ=20", _fixed20, ORANGE), ("Adaptive (ours)", _adaptive, GREEN)]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(datasets))
    w = 0.26
    for i, (label, sel, color) in enumerate(configs):
        means, stds = [], []
        for ds in datasets:
            m, s = _mean_std(sel(dense, ds))
            means.append(m * 100)
            stds.append(s * 100)
        ax.bar(x + (i - 1) * w, means, w, yerr=stds, label=label, color=color,
               capsize=3, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Cora", "Citeseer", "Pubmed"])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(60, 86)
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_h1_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def fig2_trace(dataset: str = "cora", seed: int = 0) -> None:
    """Adaptive-controller dynamics: div_t and lambda_t over epochs (small multiples)."""
    trace = TRACES / f"{dataset}_seed{seed}_t0.7_e0.05.csv"
    rows = list(csv.DictReader(trace.open(encoding="utf-8")))
    epochs = [int(r["epoch"]) for r in rows]
    div = [float(r["div_t"]) for r in rows]
    lam = [float(r["lambda_t"]) for r in rows]
    val = [float(r["val_acc"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(epochs, div, color=BLUE, linewidth=2)
    axes[0].axhline(0.7, color=MUTED, linestyle="--", linewidth=1)
    axes[0].text(2, 0.72, "target budget", color=MUTED, fontsize=8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Head similarity $d_t$")
    axes[0].set_ylim(0.3, 1.05)
    axes[1].plot(epochs, lam, color=ORANGE, linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Controller λ")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{dataset.capitalize()} seed {seed}: final val acc {val[-1]:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"fig2_trace_{dataset}_seed{seed}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig3_h3(rows: list[dict]) -> None:
    """KD comparison at matched 100-epoch budget: B5 vs B6 vs B8, grouped by prune ratio."""
    ratios = ["0.5", "0.75"]
    groups = [
        ("Pure KD (B5)", lambda r, ratio: r["prune_criterion"] == "" and r["distill"] == "True"
         and r["prune_ratio"] == ratio and float(r["epochs_run"]) <= 100.5, BLUE),
        ("Prune + FT (B6)", lambda r, ratio: r["prune_criterion"] == "diversity"
         and r["distill"] == "False" and r["prune_ratio"] == ratio
         and r["regularizer"] == "attention" and r["adaptive_lambda"] == "True"
         and float(r["epochs_run"]) == 100.0, ORANGE),
        ("Prune + KD (B8)", lambda r, ratio: r["prune_criterion"] == "diversity"
         and r["distill"] == "True" and r["prune_ratio"] == ratio
         and r["regularizer"] == "attention" and r["adaptive_lambda"] == "True", GREEN),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.arange(len(ratios))
    w = 0.26
    bar_tops = []
    for i, (label, sel, color) in enumerate(groups):
        means, stds = [], []
        for ratio in ratios:
            m, s = _mean_std([r for r in rows if sel(r, ratio)])
            means.append(m * 100)
            stds.append(s * 100)
        ax.bar(x + (i - 1) * w, means, w, yerr=stds, label=label, color=color,
               capsize=3, edgecolor="white", linewidth=0.5)
        bar_tops.append([m + s for m, s in zip(means, stds)])
    # significance annotation: B8 vs B6 paired Wilcoxon, computed from the data
    y_annot = max(max(t) for t in bar_tops) + 0.6
    for j, ratio in enumerate(ratios):
        b6 = {int(r["seed"]): float(r["test_acc"]) for r in rows
              if r["prune_criterion"] == "diversity" and r["distill"] == "False"
              and r["prune_ratio"] == ratio and r["regularizer"] == "attention"
              and r["adaptive_lambda"] == "True" and float(r["epochs_run"]) == 100.0}
        b8 = {int(r["seed"]): float(r["test_acc"]) for r in rows
              if r["prune_criterion"] == "diversity" and r["distill"] == "True"
              and r["prune_ratio"] == ratio and r["regularizer"] == "attention"
              and r["adaptive_lambda"] == "True"}
        seeds = sorted(set(b6) & set(b8))
        if len(seeds) >= 2:
            w = stats.wilcoxon([b8[s] for s in seeds], [b6[s] for s in seeds])
            if w.pvalue < 0.05:
                ax.annotate("", xy=(x[j] + 0.26, y_annot), xytext=(x[j], y_annot),
                            arrowprops=dict(arrowstyle="-", color=INK))
                ax.text(x[j] + 0.13, y_annot + 0.1, f"p={w.pvalue:.3f}",
                        ha="center", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["Keep 4/8 heads", "Keep 2/8 heads"])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(70, max(max(t) for t in bar_tops) + 2)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_h3_distillation.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    rows = _load()
    fig1_h1(rows)
    fig2_trace("cora", 3)  # traces: same-protocol runs (see dedup-report.md caveat 2)
    fig2_trace("citeseer", 0)
    fig3_h3(rows)
    print(f"figures written to {FIGDIR}")
