"""Publication-grade figures for the IJPRAI paper (revision of analysis figures).

Reads outputs/tables/results_clean.csv and outputs/traces/; writes paper/figures/.
Run from repo root: uv run python paper/make_figures_pub.py

Upgrades over run/pipeline/make_figures.py:
- honest 0-based y-axes (no-harm claim must not be visually exaggerated)
- grayscale-robust: Okabe-Ito colors + distinct hatch patterns
- value labels for exact lookup; no in-figure titles (captions carry context)
- Fig. 2 gains a validation-accuracy panel so the "accuracy stays flat"
  mechanism claim is visible in the figure itself
- captions written to paper/figures/captions.txt (same-protocol caveat included)
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "outputs" / "tables" / "results_clean.csv"
TRACES = ROOT / "outputs" / "traces"
FIGDIR = ROOT / "paper" / "figures"

# Okabe-Ito CVD-safe palette (fixed order: blue -> orange -> green).
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
INK, MUTED = "#1a1a1a", "#6b6b6b"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


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
    # Original-protocol rows only (revision batches must not pollute the bar).
    return [r for r in dense if r["dataset"] == ds and r["regularizer"] == "none"
            and r["lambda_schedule"] == ""]


def _fixed20(dense, ds):
    # Cora: second-revision sweep (true n=10, recorded similarities; the
    # original batch stored 5 duplicated seeds for this cell). Others:
    # original batch.
    sched = "fixed" if ds == "cora" else ""
    return [r for r in dense if r["dataset"] == ds and r["regularizer"] == "attention"
            and r["lambda_fixed"] == "20.0" and r["lambda_schedule"] == sched]


def _adaptive(dense, ds):
    return [r for r in dense if r["dataset"] == ds and r["regularizer"] == "attention"
            and r["adaptive_lambda"] == "True"
            and r["lambda_target"] == "0.7" and r["lambda_eta"] == "0.05"
            and r["lambda_schedule"] == ""]


def _bar_hatch(fig, ax, x, means, stds, width, label, color, hatch,
               label_dy: float = 2.0) -> list[float]:
    bars = ax.bar(x, means, width, yerr=stds, label=label, color=color,
                  capsize=3, edgecolor=color, linewidth=0.5, hatch=hatch,
                  error_kw={"elinewidth": 0.8, "ecolor": INK})
    # Value labels sit above each bar's error-bar cap (m + s); label_dy is
    # in data units and must clear the 3 pt cap at the axes' unit scale.
    label_tops = []
    for bar, m, s in zip(bars, means, stds):
        if np.isfinite(m):
            y = m + s + label_dy
            ax.text(bar.get_x() + bar.get_width() / 2, y,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
            label_tops.append(y)
    return label_tops


def fig1_h1(rows: list[dict]) -> None:
    """Main comparison: test accuracy across 3 datasets x 3 training configs."""
    dense = _dense(rows)
    datasets = ["cora", "citeseer", "pubmed"]
    configs = [
        ("Baseline", _b0, BLUE, ""),
        ("Fixed $\\lambda{=}20$", _fixed20, ORANGE, "///"),
        ("Adaptive (ours)", _adaptive, GREEN, "..."),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(datasets))
    w = 0.26
    for i, (label, sel, color, hatch) in enumerate(configs):
        means, stds = [], []
        for ds in datasets:
            m, s = _mean_std(sel(dense, ds))
            means.append(m * 100)
            stds.append(s * 100)
        _bar_hatch(fig, ax, x + (i - 1) * w, means, stds, w, label, color, hatch,
                   label_dy=2.2)
    ax.set_xticks(x)
    ax.set_xticklabels(["Cora", "Citeseer", "Pubmed"])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 90)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_h1_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def fig2_trace(dataset: str = "cora", seed: int = 3) -> None:
    """Adaptive-controller dynamics: similarity, lambda, and val acc over epochs."""
    trace = TRACES / f"{dataset}_seed{seed}_t0.7_e0.05.csv"
    rows = list(csv.DictReader(trace.open(encoding="utf-8")))
    epochs = [int(r["epoch"]) for r in rows]
    div = [float(r["div_t"]) for r in rows]
    lam = [float(r["lambda_t"]) for r in rows]
    val = [float(r["val_acc"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.9))
    for ax, letter in zip(axes, "abc"):
        ax.set_title(f"({letter})", loc="left", fontsize=9)

    axes[0].plot(epochs, div, color=BLUE, linewidth=1.6)
    axes[0].axhline(0.7, color=MUTED, linestyle="--", linewidth=1)
    axes[0].text(2, 0.715, "budget $d^*{=}0.7$", color=MUTED, fontsize=7.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Head similarity $d_t$")
    axes[0].set_ylim(0.3, 1.05)

    axes[1].plot(epochs, lam, color=ORANGE, linewidth=1.6)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Controller $\\lambda_t$")

    axes[2].plot(epochs, val, color=GREEN, linewidth=1.6)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Validation accuracy")

    fig.tight_layout()
    fig.savefig(FIGDIR / f"fig2_trace_{dataset}_seed{seed}.pdf", bbox_inches="tight")
    plt.close(fig)


def _b5(rows, ratio):
    return [r for r in rows if r["prune_criterion"] == "" and r["distill"] == "True"
            and r["prune_ratio"] == ratio and float(r["epochs_run"]) <= 100.5]


def _b6(rows, ratio):
    return [r for r in rows if r["prune_criterion"] == "diversity"
            and r["distill"] == "False" and r["prune_ratio"] == ratio
            and r["regularizer"] == "attention" and r["adaptive_lambda"] == "True"
            and float(r["epochs_run"]) == 100.0]


def _b8(rows, ratio):
    return [r for r in rows if r["prune_criterion"] == "diversity"
            and r["distill"] == "True" and r["prune_ratio"] == ratio
            and r["regularizer"] == "attention" and r["adaptive_lambda"] == "True"]


def fig3_h3(rows: list[dict]) -> None:
    """KD comparison at matched 100-epoch budget: B5 vs B6 vs B8 by prune ratio."""
    ratios = ["0.5", "0.75"]  # prune_ratio semantics: fraction pruned (see pruning.py)
    groups = [
        ("Pure KD (B5)", _b5, BLUE, ""),
        ("Prune + FT (B6)", _b6, ORANGE, "///"),
        ("Prune + KD (B8)", _b8, GREEN, "..."),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    x = np.arange(len(ratios))
    w = 0.26
    label_tops = []
    for i, (label, sel, color, hatch) in enumerate(groups):
        means, stds = [], []
        for ratio in ratios:
            m, s = _mean_std(sel(rows, ratio))
            means.append(m * 100)
            stds.append(s * 100)
        label_tops += _bar_hatch(fig, ax, x + (i - 1) * w, means, stds, w,
                                 label, color, hatch, label_dy=2.0)

    def _pairs(sel, ratio):
        return {int(r["seed"]): float(r["test_acc"])
                for r in sel(rows, ratio) if r["test_acc"] not in ("", "nan")}

    # Paired Wilcoxon, appendix convention (final_stats.py): one-sided
    # alternative="greater" with B8 as the first argument.
    y_annot = max(label_tops) + 1.8
    p_b8b6: dict[str, float] = {}
    p_b8b5: dict[str, float] = {}
    for j, ratio in enumerate(ratios):
        b6 = _pairs(_b6, ratio)
        b8 = _pairs(_b8, ratio)
        seeds = sorted(set(b6) & set(b8))
        if len(seeds) >= 2:
            p = stats.wilcoxon([b8[s] for s in seeds], [b6[s] for s in seeds],
                               alternative="greater").pvalue
            p_b8b6[ratio] = p
            if p < 0.05:
                ax.annotate("", xy=(x[j] + 0.26, y_annot), xytext=(x[j], y_annot),
                            arrowprops=dict(arrowstyle="-", color=INK))
                ax.text(x[j] + 0.13, y_annot + 0.3, f"$p={p:.4f}$",
                        ha="center", fontsize=8, color=INK)
        b5 = _pairs(_b5, ratio)
        seeds5 = sorted(set(b5) & set(b8))
        if len(seeds5) >= 2:
            p_b8b5[ratio] = stats.wilcoxon(
                [b8[s] for s in seeds5], [b5[s] for s in seeds5]).pvalue
    ax.set_xticks(x)
    ax.set_xticklabels(["Keep 4/8 heads", "Keep 2/8 heads"])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, y_annot + 4)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_h3_distillation.pdf", bbox_inches="tight")
    plt.close(fig)
    return p_b8b6, p_b8b5


def _controller(dense, ds):
    return [r for r in dense if r["dataset"] == ds and r["regularizer"] == "attention"
            and r["adaptive_lambda"] == "True"
            and r["lambda_target"] == "0.7" and r["lambda_eta"] == "0.05"]


def _b1_sched(dense, ds, sched):
    return [r for r in dense if r["dataset"] == ds and r["lambda_schedule"] == sched]


def fig4_baselines(rows: list[dict]) -> None:
    """M1 revision: controller vs three adaptive baselines, 3 datasets."""
    dense = [r for r in rows
             if r["prune_criterion"] == "" and r["hidden_per_head"] == "8"
             and r["distill"] == "False"]
    datasets = ["cora", "citeseer", "pubmed"]
    configs = [
        ("Controller (ours)", lambda d, ds: _controller(d, ds), BLUE, ""),
        ("Linear schedule", lambda d, ds: _b1_sched(d, ds, "linear"), ORANGE, "///"),
        ("Kendall-style", lambda d, ds: _b1_sched(d, ds, "kendall"), GREEN, "..."),
        ("GradNorm", lambda d, ds: _b1_sched(d, ds, "gradnorm"), "#CC79A7", "xxx"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(datasets))
    w = 0.2
    label_tops = []
    for i, (label, sel, color, hatch) in enumerate(configs):
        means, stds = [], []
        for ds in datasets:
            m, s = _mean_std(sel(dense, ds))
            means.append(m * 100)
            stds.append(s * 100)
        label_tops += _bar_hatch(fig, ax, x + (i - 1.5) * w, means, stds, w,
                                 label, color, hatch, label_dy=1.6)

    def _pairs(sel, ds):
        return {int(r["seed"]): float(r["test_acc"])
                for r in sel(dense, ds) if r["test_acc"] not in ("", "nan")}

    # Significance brackets: controller vs GradNorm (one-sided paired Wilcoxon).
    y_annot = max(label_tops) + 1.8
    ctrl_sel = lambda d, ds: _controller(d, ds)
    gn_sel = lambda d, ds: _b1_sched(d, ds, "gradnorm")
    for j, ds in enumerate(datasets):
        a = _pairs(ctrl_sel, ds)
        b = _pairs(gn_sel, ds)
        seeds = sorted(set(a) & set(b))
        p = stats.wilcoxon([a[s] for s in seeds], [b[s] for s in seeds],
                           alternative="greater").pvalue
        if p < 0.05:
            ax.annotate("", xy=(x[j] + 0.3, y_annot), xytext=(x[j] - 0.3, y_annot),
                        arrowprops=dict(arrowstyle="-", color=INK))
            ax.text(x[j], y_annot + 0.3, f"$p={p:.4f}$", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["Cora", "Citeseer", "Pubmed"])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, y_annot + 4)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_adaptive_baselines.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / "fig4_adaptive_baselines.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


FIG1_CAPTION = """\
Fig. 1 — Test accuracy of a 2-layer GAT on three citation networks under baseline
training, fixed-lambda (lambda=20) diversity regularization, and feedback-controlled
adaptive regularization (target 0.7, eta=0.05). Bars are means over 10 seeds; error
bars are +/-1 standard deviation over seeds; values above bars are means. The
adaptive configuration stays within +/-0.4 points of baseline on all three datasets;
no fixed weight achieves both an effect on similarity and this level of accuracy
(see Table 1).
"""

FIG2_CAPTION = """\
Fig. 2{a} ({dataset}, seed {seed}) — Adaptive-controller dynamics on {dataset}, seed
{seed} (same-protocol run; see Section 5 for the provenance note): (a) head
similarity d_t falls toward the 0.7 budget as (b) the integral controller raises
lambda; (c) validation accuracy remains flat once the budget is enforced late in
training, after the best checkpoint is selected.
"""

FIG3_CAPTION = """\
Fig. 3 — Test accuracy at fixed head budgets: training a smaller student from
scratch with KD (B5), diversity-based pruning followed by fine-tuning (B6), and
pruning followed by KD (B8), at prune ratios 0.5 (4/8 heads) and 0.75 (2/8 heads).
Bars are means over 10 seeds; error bars are +/-1 standard deviation. Brackets:
one-sided paired Wilcoxon signed-rank test, B8 vs. B6, p={p86:.4f} at both ratios.
B8 vs. B5: no significant difference (two-sided Wilcoxon p={p85_50:.4f} at ratio
0.5, p={p85_75:.4f} at ratio 0.75; 95% CIs on the difference are in the text); KD,
not the pruning path, accounts for the recovery.
"""

FIG4_CAPTION = """\
Fig. 4 - Controller vs. adaptive baselines (M1 revision). Test accuracy of a 2-layer
GAT under feedback-controlled regularization and three adaptive-baseline schedules
(linear, Kendall-style, GradNorm). Bars are means over 10 seeds; error bars are
+/-1 standard deviation. Brackets: one-sided paired Wilcoxon signed-rank test,
controller vs. GradNorm; p=0.0098 (Cora) and p=0.0049 (Pubmed); Citeseer is
p=0.0527. Linear and Kendall schedules match the controller on accuracy but do not
enforce the similarity budget; GradNorm enforces it already at the best checkpoint
and pays 1.1-1.8 points. Only the controller enforces the budget late in training
without accuracy loss (adaptive-baselines table).
"""

if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    rows = _load()
    fig1_h1(rows)
    fig2_trace("cora", 3)
    fig2_trace("citeseer", 0)
    p_b8b6, p_b8b5 = fig3_h3(rows)
    fig4_baselines(rows)
    captions = FIG1_CAPTION
    captions += FIG2_CAPTION.format(a="a", dataset="Cora", seed=3)
    captions += "\n"
    captions += FIG2_CAPTION.format(a="b", dataset="Citeseer", seed=0)
    captions += "\n"
    captions += FIG3_CAPTION.format(
        p86=p_b8b6.get("0.5", float("nan")),
        p85_50=p_b8b5.get("0.5", float("nan")),
        p85_75=p_b8b5.get("0.75", float("nan")),
    )
    captions += "\n"
    captions += FIG4_CAPTION
    (FIGDIR / "captions.txt").write_text(captions, encoding="utf-8")
    print(f"figures written to {FIGDIR}")
