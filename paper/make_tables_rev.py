"""Revision tables for the IJPRAI paper (M1/M2 Colab batches, merged 2026-09-05).

Reads outputs/tables/results_clean.csv; writes paper/tables/tab4_adaptive_baselines.tex
and paper/tables/tab5_revival.tex. Numbers must match analysis-output/stats-appendix.md
sections R1/R2 (regenerate with `uv run python -m run.pipeline.final_stats` first).

Run from repo root: uv run python paper/make_tables_rev.py
"""

import csv
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "outputs" / "tables" / "results_clean.csv"
TABDIR = ROOT / "paper" / "tables"

DATASETS = ("cora", "citeseer", "pubmed")
SCHEDS = ("linear", "kendall", "gradnorm")
CRITS = ("diversity", "magnitude", "gradient", "random")
SOURCES = ("best", "late_min_div", "late_last")


def load() -> list[dict]:
    return list(csv.DictReader(CSV.open(encoding="utf-8")))


def sel(rows, **conds):
    out = []
    for r in rows:
        if all((r.get(k, "") == str(v)) if v is not None else (r.get(k, "") == "")
               for k, v in conds.items()):
            out.append(r)
    return out


def ms(sub, key="test_acc") -> str:
    vals = [float(r[key]) for r in sub if r[key] not in ("", "nan")]
    if not vals:
        return "n/a"
    return f"{mean(vals) * 100:.2f} \\pm {stdev(vals) * 100:.2f}"


def mean_f(sub, key) -> float:
    vals = [float(r[key]) for r in sub if r[key] not in ("", "nan")]
    return mean(vals) if vals else float("nan")


def ci_cell(sub, key="test_acc") -> str:
    """'mean [lo, hi]' cell from a t-based 95% CI over seeds."""
    vals = np.array([float(r[key]) for r in sub if r[key] not in ("", "nan")]) * 100
    if len(vals) < 2:
        return "n/a"
    m, s = float(vals.mean()), float(vals.std(ddof=1))
    t = stats.t.ppf(0.975, len(vals) - 1)
    se = t * s / np.sqrt(len(vals))
    return f"{m:.2f} $[{m - se:.2f}, {m + se:.2f}]$"


def tab6(rows) -> None:
    """Appendix CI table for the revision protocol (R1/R2)."""
    b1 = [r for r in rows if r["lambda_schedule"] in SCHEDS]
    b3 = [r for r in rows if r["prune_criterion"]]
    lines = [
        "% Table (number assigned by position): 95% CIs for the revision protocol",
        "% (R1/R2). Generated 2026-09-06 from results_clean.csv; matches",
        "% stats-appendix.md R1/R2.",
        "\\begin{table}[th]",
        "\\tbl{95\\% confidence intervals for the revision protocol (10 seeds per "
        "cell). Top: adaptive-baseline comparison (R1); controller rows are the "
        "submitted dense adaptive block. Bottom: revival test (R2), test accuracy by "
        "checkpoint source and criterion; prune ratio 0.75, fine-tuning 100 epochs.",
        "\\label{tab:revision-cis}}",
        "{\\begin{tabular}{llcccc}",
        "\\toprule",
        "\\multicolumn{6}{c}{\\textbf{R1: adaptive baselines}} \\\\",
        "\\colrule",
        "Dataset & Schedule & Test acc.\\ 95\\% CI & $\\lambda_{\\mathrm{final}}$ & "
        "$d_{\\mathrm{best}}$ & \\\\",
        "\\colrule",
    ]
    for ds in DATASETS:
        ctrl = sel(rows, dataset=ds, adaptive_lambda="True", prune_criterion=None,
                   lambda_schedule=None, distill="False")
        lines.append(f"{ds.title()} & Controller & {ci_cell(ctrl)} & "
                     f"{mean_f(ctrl, 'lambda_final'):.2f} & {mean_f(ctrl, 'div_final'):.3f} & \\\\")
        for sched in SCHEDS:
            x = sel(b1, dataset=ds, lambda_schedule=sched)
            lines.append(f" & {sched.title()} & {ci_cell(x)} & {mean_f(x, 'lambda_final'):.2f} & "
                         f"{mean_f(x, 'div_final'):.3f} & \\\\")
        lines.append("\\colrule")
    lines.extend([
        "\\multicolumn{6}{c}{\\textbf{R2: revival test (test accuracy 95\\% CIs)}} \\\\",
        "\\colrule",
        "Dataset & Source & Diversity & Magnitude & Gradient & Random \\\\",
        "\\colrule",
    ])
    for ds in DATASETS:
        for i, src in enumerate(SOURCES):
            label = {"best": "Best", "late_min_div": "Late (min div.)", "late_last": "Late (last)"}[src]
            cells = " & ".join(ci_cell(sel(b3, dataset=ds, checkpoint_source=src,
                                           prune_criterion=c)) for c in CRITS)
            lead = ds.title() if i == 0 else ""
            lines.append(f"{lead} & {label} & {cells} \\\\")
    lines.extend([
        "\\botrule",
        "\\end{tabular}}",
        "\\end{table}",
        "",
    ])
    (TABDIR / "tab6_revision_cis.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"tab6_revision_cis.tex written ({len(lines)} lines)")


def paired_one_sided(a, b, key="test_acc"):
    """One-sided (a > b) paired tests; returns (diff_pp, wilcoxon_p, paired_t_p, cohen_d, n)."""
    da = {int(r["seed"]): float(r[key]) for r in a if r[key] not in ("", "nan")}
    db = {int(r["seed"]): float(r[key]) for r in b if r[key] not in ("", "nan")}
    seeds = sorted(set(da) & set(db))
    A = np.array([da[s] for s in seeds])
    B = np.array([db[s] for s in seeds])
    w = stats.wilcoxon(A, B, alternative="greater").pvalue
    t = stats.ttest_rel(A, B, alternative="greater").pvalue
    d = A - B
    cohen = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")
    return float((A - B).mean() * 100), float(w), float(t), cohen, len(seeds)


def tab4(rows) -> None:
    b1 = [r for r in rows if r["lambda_schedule"] in SCHEDS]
    lines = [
        "% Table (number assigned by position): adaptive-baseline comparison",
        "% (M1 revision, Colab batch B1). WS-IJPRAI table format. Numbers:",
        "% stats-appendix.md R1 (2026-09-06). Controller rows: submitted dense",
        "% adaptive block (distill=False), n=10 per dataset.",
        "\\begin{table}[th]",
        "\\tbl{Controller vs.\\ adaptive baselines (one-sided paired tests, 10 seeds). "
        "Accuracy is mean~$\\pm$~std per cell; $\\lambda_{\\mathrm{final}}$ and "
        "$d_{\\mathrm{best}}$ are the mean final controller weight and the inter-head "
        "similarity at the best checkpoint. GradNorm is the only baseline that reaches the "
        "similarity budget, and it does so already at the best checkpoint, at a significant "
        "accuracy cost; linear and Kendall schedules stay accuracy-neutral but do not enforce "
        "the budget. The controller is the only scheme that enforces the budget late in "
        "training while matching baseline accuracy.",
        "\\label{tab:adaptive-baselines}}",
        "{\\begin{tabular}{llccccc}",
        "\\toprule",
        "Dataset & Schedule & Test acc.\\ (\\%) & $\\lambda_{\\mathrm{final}}$ & "
        "$d_{\\mathrm{best}}$ & $\\Delta$ vs.\\ ctrl.\\ (pp) & $p$ (paired) \\\\",
        "\\colrule",
    ]
    for ds in DATASETS:
        ctrl = sel(rows, dataset=ds, adaptive_lambda="True", prune_criterion=None,
                   lambda_schedule=None, distill="False")
        ctrl_ms = ms(ctrl)
        ctrl_val = mean([float(r["test_acc"]) for r in ctrl])
        lines.append(f"{ds.title()} & Controller & ${ctrl_ms}$ & "
                     f"{mean_f(ctrl, 'lambda_final'):.2f} & {mean_f(ctrl, 'div_final'):.3f} & --- & --- \\\\")
        for sched in SCHEDS:
            x = sel(b1, dataset=ds, lambda_schedule=sched)
            diff, wp, tp, _, n = paired_one_sided(ctrl, x)
            lines.append(f" & {sched.title()} & ${ms(x)}$ & {mean_f(x, 'lambda_final'):.2f} & "
                         f"{mean_f(x, 'div_final'):.3f} & ${diff:+.2f}$ & {wp:.4f} \\\\")
        lines.append("\\colrule")
    lines.extend([
        "\\botrule",
        "\\end{tabular}}",
        "\\end{table}",
        "",
    ])
    (TABDIR / "tab4_adaptive_baselines.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"tab4_adaptive_baselines.tex written ({len(lines)} lines)")


def tab5(rows) -> None:
    b3 = [r for r in rows if r["prune_criterion"]]
    lines = [
        "% Table (number assigned by position): revival test (M2 revision, Colab",
        "% batches B2/B3). WS-IJPRAI table format. Numbers: stats-appendix.md R2",
        "% (2026-09-06). Prune ratio 0.75, controller teacher, FT 100 epochs, no KD,",
        "% 10 seeds per cell.",
        "\\begin{table}[th]",
        "\\tbl{Revival test: pruning criteria computed from late-training checkpoints. Top: "
        "test accuracy (mean~$\\pm$~std) by checkpoint source and criterion. $d_{src}$ is the "
        "teacher's head similarity at the checkpoint source. The lowest-similarity checkpoint "
        "coincides with the final one in 119/120 (dataset, seed, criterion) combinations "
        "(similarity decreases monotonically), so $d_{src}$ for the late source is the "
        "teacher's final similarity. Bottom: one-sided paired contrasts (diversity vs.\\ "
        "baseline criterion) at the late source, the timing where the signal is actually "
        "differentiated; none is significant, and the nominally largest effect favors random. "
        "The negative criterion result therefore survives the revival test.",
        "\\label{tab:revival}}",
        "{\\begin{tabular}{llcccccc}",
        "\\toprule",
        "Dataset & Source & $d_{src}$ & & Diversity & Magnitude & Gradient & Random \\\\",
        "\\colrule",
    ]
    for ds in DATASETS:
        for i, src in enumerate(SOURCES):
            dsrc = mean_f(sel(b3, dataset=ds, checkpoint_source=src, prune_criterion="diversity"),
                          "criterion_div")
            cells = " & ".join(f"${ms(sel(b3, dataset=ds, checkpoint_source=src, prune_criterion=c))}$"
                               for c in CRITS)
            lead = ds.title() if i == 0 else ""
            label = {"best": "Best", "late_min_div": "Late (min div.)", "late_last": "Late (last)"}[src]
            lines.append(f"{lead} & {label} & {dsrc:.3f} & & {cells} \\\\")
        lines.append("\\colrule")
    lines.extend([
        "\\multicolumn{8}{l}{\\textbf{Paired contrasts at the late source (diversity minus "
        "baseline, 10 pairs)}} \\\\",
        "\\colrule",
        "Dataset & Contrast & $\\Delta$ (pp) & & Wilcoxon $p$ & Paired-$t$ $p$ & $d$ & \\\\",
        "\\colrule",
    ])
    for ds in DATASETS:
        for crit in ("random", "magnitude", "gradient"):
            a = sel(b3, dataset=ds, checkpoint_source="late_min_div", prune_criterion="diversity")
            b = sel(b3, dataset=ds, checkpoint_source="late_min_div", prune_criterion=crit)
            diff, wp, tp, cohen, _ = paired_one_sided(a, b)
            lines.append(f"{ds.title()} & vs.\\ {crit} & ${diff:+.2f}$ & & {wp:.4f} & {tp:.4f} & "
                         f"{cohen:.2f} & \\\\")
    lines.extend([
        "\\botrule",
        "\\end{tabular}}",
        "\\end{table}",
        "",
    ])
    (TABDIR / "tab5_revival.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"tab5_revival.tex written ({len(lines)} lines)")


def tab2(rows) -> None:
    """H2 criteria matrix (regenerated: adaptive-0.75 rows now come from the B3
    `best`-source runs, which superseded the pre-merge rows dropped 2026-09-05)."""
    old = [r for r in rows if r["lambda_schedule"] == "" and r["checkpoint_source"] == ""]
    b3 = [r for r in rows if r["prune_criterion"]]
    lines = [
        "% Table (number assigned by position): pruning-criterion contrasts (H2).",
        "% Regenerated 2026-09-06 from results_clean.csv. Adaptive-teacher ratio-0.75",
        "% rows use the B3 `best`-source runs (supersede the pre-merge rows dropped",
        "% in the 2026-09-05 merge; see stats-appendix.md H2 note).",
        "\\begin{table}[th]",
        "\\tbl{Paired comparison of pruning criteria at GAT scale (fine-tuning 100 "
        "epochs, no KD; identical seeds within each contrast). $\\Delta$ is the accuracy "
        "difference (diversity criterion minus baseline criterion) in percentage points; "
        "Cohen's $d$ is reported for each contrast. Wilcoxon $p$-values are one-sided "
        "(diversity $>$ baseline). No contrast survives multiplicity control across the "
        "12 tests; the single nominally significant pair (B0 teacher, ratio 0.5, "
        "vs.\\ gradient) rests on 5 pairs and is directionally inconsistent with the "
        "main comparison. Tied pairs in the Wilcoxon test fall back to the normal "
        "approximation; paired-$t$ $p$-values are reported alongside.",
        "\\label{tab:criteria}}",
        "{\\begin{tabular}{lllcccc}",
        "\\toprule",
        "Teacher & Ratio & Contrast & $\\Delta$ (pp) & Wilcoxon $p$ & Paired-$t$ $p$ & $d$ \\\\",
        "\\colrule",
    ]

    def contrast_cells(teacher, ratio, rows_src):
        out = []
        for crit in ("random", "gradient", "magnitude"):
            a = sel(rows_src, prune_criterion="diversity")
            b = sel(rows_src, prune_criterion=crit)
            diff, wp, tp, cohen, _ = paired_one_sided(a, b)
            out.append(f"{teacher} & {ratio} & div vs.\\ {crit} & ${diff:+.2f}$ & "
                       f"{wp:.4f} & {tp:.4f} & {cohen:+.2f} \\\\")
        return out

    lines += contrast_cells("Adaptive", "0.5", sel(old, regularizer="attention",
                                                   prune_ratio="0.5", epochs_run="100.0",
                                                   distill="False"))
    lines += contrast_cells("Adaptive", "0.75", sel(b3, dataset="cora",
                                                    checkpoint_source="best",
                                                    prune_ratio="0.75"))
    lines += contrast_cells("B0", "0.5", sel(old, regularizer="none", prune_ratio="0.5",
                                             epochs_run="100.0", distill="False"))
    lines += contrast_cells("B0", "0.75", sel(old, regularizer="none", prune_ratio="0.75",
                                              epochs_run="100.0", distill="False"))
    lines.extend([
        "\\botrule",
        "\\end{tabular}}",
        "\\end{table}",
        "",
    ])
    (TABDIR / "tab2_h2_criteria.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"tab2_h2_criteria.tex written ({len(lines)} lines)")


if __name__ == "__main__":
    TABDIR.mkdir(parents=True, exist_ok=True)
    rows = load()
    tab2(rows)
    tab4(rows)
    tab5(rows)
    tab6(rows)
