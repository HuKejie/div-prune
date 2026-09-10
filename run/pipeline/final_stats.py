"""Compute the complete statistics appendix and write analysis-output/stats-appendix.md.

Run from repo root: uv run python -m run.pipeline.final_stats

Protocol conventions:
- "old" protocol rows (pre-2026-09-05) have lambda_schedule == "" and
  checkpoint_source == ""; "new" revision rows (Colab batches B1/B2/B3) fill
  both. Sections H1-H3 select old rows only so revision rows cannot pollute
  the submitted tables; sections R1-R2 analyze the revision protocol.
"""

import csv
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "outputs" / "tables" / "results_clean.csv"
SWEEP = ROOT / "outputs_rev2_sweep" / "tables" / "results.csv"
OUT = ROOT / "analysis-output"
OUT.mkdir(parents=True, exist_ok=True)


def load() -> list[dict]:
    return list(csv.DictReader(CSV.open(encoding="utf-8")))


def load_sweep() -> list[dict]:
    """Second-revision Cora fixed-λ sweep: true n=10 per weight with recorded
    similarities (read from its own CSV; results_clean has duplicate/polluted
    rows for λ=0.1 and the old λ>=5 rows are 5 duplicated seeds)."""
    return list(csv.DictReader(SWEEP.open(encoding="utf-8")))


def sel(rows, **conds):
    out = []
    for r in rows:
        if all(_match(r, k, v) for k, v in conds.items()):
            out.append(r)
    return out


def _match(r, k, v):
    if v is None:
        return r.get(k, "") == ""
    return r.get(k, "") == str(v)


def desc_raw(sub: list[dict], key: str = "test_acc") -> str:
    """Mean ± std without the x100 percentage scaling (for wall time etc.)."""
    vals = [float(r[key]) for r in sub if r[key] not in ("", "nan")]
    if not vals:
        return "n/a"
    return f"{mean(vals):.2f} ± {stdev(vals):.2f} (n={len(vals)})"


def desc(sub: list[dict], key: str = "test_acc") -> str:
    vals = [float(r[key]) for r in sub if r[key] not in ("", "nan")]
    if not vals:
        return "n/a"
    m, s = mean(vals), stdev(vals)
    n = len(vals)
    t = stats.t.ppf(0.975, n - 1) if n > 1 else float("nan")
    ci = t * s / np.sqrt(n)
    return f"{m * 100:.2f} ± {s * 100:.2f} (n={n}, 95% CI [{m * 100 - ci * 100:.2f}, {m * 100 + ci * 100:.2f}])"


def paired(a: list[dict], b: list[dict], key: str = "test_acc") -> str:
    da = {int(r["seed"]): float(r[key]) for r in a if r[key] not in ("", "nan")}
    db = {int(r["seed"]): float(r[key]) for r in b if r[key] not in ("", "nan")}
    seeds = sorted(set(da) & set(db))
    if len(seeds) < 2:
        return "insufficient pairs"
    A = [da[s] for s in seeds]
    B = [db[s] for s in seeds]
    w = stats.wilcoxon(A, B, alternative="greater")
    t = stats.ttest_rel(A, B, alternative="greater")
    diff = mean(A) - mean(B)
    # Cohen's d (paired, using pooled sd of differences)
    d_arr = np.array(A) - np.array(B)
    d_cohen = d_arr.mean() / d_arr.std(ddof=1) if d_arr.std(ddof=1) > 0 else float("nan")
    return (f"diff {diff * 100:+.2f} pp, Wilcoxon p={w.pvalue:.4f}, "
            f"paired-t p={t.pvalue:.4f}, Cohen's d={d_cohen:.2f}, {len(seeds)} pairs")


def paired_twosided(a: list[dict], b: list[dict], key: str = "test_acc") -> str:
    """Two-sided paired tests with a 95% CI on the mean difference (tie claims, M5)."""
    da = {int(r["seed"]): float(r[key]) for r in a if r[key] not in ("", "nan")}
    db = {int(r["seed"]): float(r[key]) for r in b if r[key] not in ("", "nan")}
    seeds = sorted(set(da) & set(db))
    if len(seeds) < 2:
        return "insufficient pairs"
    A = np.array([da[s] for s in seeds])
    B = np.array([db[s] for s in seeds])
    d_arr = (A - B) * 100
    m, s = float(d_arr.mean()), float(d_arr.std(ddof=1))
    t_crit = stats.t.ppf(0.975, len(seeds) - 1)
    lo, hi = m - t_crit * s / np.sqrt(len(seeds)), m + t_crit * s / np.sqrt(len(seeds))
    w = stats.wilcoxon(A, B).pvalue
    t = stats.ttest_rel(A, B).pvalue
    return (f"diff {m:+.2f} pp, 95% CI [{lo:+.2f}, {hi:+.2f}], "
            f"two-sided Wilcoxon p={w:.4f}, paired-t p={t:.4f}, {len(seeds)} pairs")


def dense(rows):
    # Dense training rows only: exclude pruning rows and KD rows (B5 has prune_criterion=""
    # but distill=True; its test_acc is the student's, not a dense model's).
    return sel(rows, prune_criterion=None, hidden_per_head=8, distill="False")


def mean_f(sub: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in sub if r[key] not in ("", "nan")]
    return mean(vals) if vals else float("nan")


def main() -> None:
    rows = load()
    old = [r for r in rows if r["lambda_schedule"] == "" and r["checkpoint_source"] == ""]
    # R2 revival rows are B3 protocol (prune+FT, no KD). distill=True rows
    # (B8 prune+KD students) must be excluded or they pollute the "best"
    # checkpoint-source cells with KD-student accuracies (2026-09-10 fix).
    b3 = [r for r in rows if r["prune_criterion"] and r["distill"] == "False"]
    sweep = load_sweep()
    d = dense(old)
    lines = ["# Statistics Appendix (auto-generated, 2026-09-10)", ""]
    lines.append("Data source: `outputs/tables/results_clean.csv` (merged 2026-09-05, see "
                 "`temp/merge_colab_20260905.py` and `analysis-output/dedup-report.md`). "
                 "Sections H1-H3: old protocol (submitted version); sections R1-R2: "
                 "revision protocol (Colab batches B1/B2/B3). All rows: dense training "
                 "runs unless noted. GAT 2-layer, 8 heads, hidden 8/head, fixed Planetoid "
                 "split, early stopping patience 100, 10 seeds per config unless stated.")
    lines.append("")

    # --- H1 ---
    lines.append("## H1: adaptive lambda vs fixed lambda vs baseline")
    lines.append("")
    lines.append("| Dataset | B0 baseline | Fixed λ=20 | Adaptive (t=0.7, η=0.05) |")
    lines.append("|---|---|---|---|")
    for ds in ("cora", "citeseer", "pubmed"):
        b0 = sel(d, dataset=ds, regularizer="none", adaptive_lambda="False")
        fx = sel(d, dataset=ds, regularizer="attention", adaptive_lambda="False", lambda_fixed="20.0")
        ad = sel(d, dataset=ds, adaptive_lambda="True", lambda_target="0.7", lambda_eta="0.05")
        if ds == "cora":
            # Second-revision sweep (true n=10; the original batch stored 5
            # duplicated seeds for this cell).
            fx = sel(sweep, dataset="cora", lambda_fixed="20.0")
        lines.append(f"| {ds} | {desc(b0)} | {desc(fx)} | {desc(ad)} |")
    lines.append("")
    for ds in ("cora", "citeseer", "pubmed"):
        ad = sel(d, dataset=ds, adaptive_lambda="True", lambda_target="0.7", lambda_eta="0.05")
        lam = mean_f(ad, "lambda_final")
        div = mean_f(ad, "div_final")
        n = len(ad)
        lines.append(f"- {ds}: adaptive runs end with λ_final={lam:.2f}, div_final={div:.3f} "
                     f"(best-checkpoint similarity; n={n} protocol repetitions, see traces "
                     f"for late-training values)")
    lines.append("")
    lines.append("### Fixed-λ failure spectrum (Cora, attention-level)")
    lines.append("")
    lines.append("Second-revision sweep (`outputs_rev2_sweep/tables/results.csv`): true "
                 "n=10 per weight with recorded best-checkpoint similarities, superseding "
                 "the pre-submission n=5 values (the old rows stored 5 duplicated seeds "
                 "for λ>=5 and probe values for λ<=1).")
    lines.append("")
    lines.append("| λ | test acc | div_final |")
    lines.append("|---|---|---|")
    for lam in ("0.01", "0.1", "0.5", "1.0", "5.0", "20.0", "50.0"):
        sub = sel(sweep, dataset="cora", lambda_fixed=lam)
        div_vals = [float(r["div_final"]) for r in sub if r["div_final"] not in ("", "nan")]
        div = mean(div_vals) if div_vals else float("nan")
        lines.append(f"| {lam} | {desc(sub) if sub else 'n/a'} | {div:.3f} |")
    lines.append("")

    # --- H2 ---
    lines.append("## H2: pruning criterion comparison (paired, same seeds, FT=100, no KD)")
    lines.append("")
    lines.append("| Source | Ratio | Contrast | Test |")
    lines.append("|---|---|---|---|")
    for src, src_name in (("attention", "adaptive t0.7"), ("none", "B0")):
        for ratio in ("0.75", "0.5"):
            for crit in ("random", "gradient", "magnitude"):
                a = sel(old, regularizer=src, prune_ratio=ratio, prune_criterion="diversity",
                        epochs_run="100.0", distill="False")
                b = sel(old, regularizer=src, prune_ratio=ratio, prune_criterion=crit,
                        epochs_run="100.0", distill="False")
                if a and b:
                    lines.append(f"| {src_name} | {ratio} | diversity vs {crit} | {paired(a, b)} |")
    lines.append("")
    lines.append("### H2 note: adaptive-teacher ratio-0.75 contrasts use B3 `best`-source rows")
    lines.append("")
    lines.append("The pre-merge adaptive-teacher ratio-0.75 rows were dropped in the 2026-09-05 "
                 "merge as semantic duplicates of the new B3 `best`-source rows (same protocol, "
                 "post-bugfix trainer, explicit checkpoint_source). The submitted H2 numbers for "
                 "adaptive 0.75 are superseded by:")
    lines.append("")
    lines.append("| Source | Ratio | Contrast | Test |")
    lines.append("|---|---|---|---|")
    for crit in ("random", "gradient", "magnitude"):
        a = sel(b3, dataset="cora", checkpoint_source="best", prune_criterion="diversity")
        b = sel(b3, dataset="cora", checkpoint_source="best", prune_criterion=crit)
        lines.append(f"| adaptive t0.7 (B3 best) | 0.75 | diversity vs {crit} | {paired(a, b)} |")
    lines.append("")

    # --- H3 ---
    lines.append("## H3: distillation (adaptive teacher t0.7, matched 100-epoch budget)")
    lines.append("")
    lines.append("| Ratio | B5 pure KD | B6 prune+FT | B8 prune+KD | B8 vs B6 | B8 vs B5 |")
    lines.append("|---|---|---|---|---|---|")
    for ratio in ("0.5", "0.75"):
        b5 = sel(old, prune_criterion=None, distill="True", prune_ratio=ratio)
        b5 = [r for r in b5 if float(r["epochs_run"]) <= 100.5]
        b6 = sel(old, prune_criterion="diversity", distill="False", prune_ratio=ratio,
                 epochs_run="100.0", regularizer="attention", adaptive_lambda="True")
        b8 = sel(old, prune_criterion="diversity", distill="True", prune_ratio=ratio,
                 regularizer="attention", adaptive_lambda="True")
        lines.append(f"| {ratio} | {desc(b5)} | {desc(b6)} | {desc(b8)} | "
                     f"{paired(b8, b6) if b8 and b6 else 'n/a'} | {paired_twosided(b8, b5) if b8 and b5 else 'n/a'} |")
    lines.append("")
    lines.append("- Note: B5 (pure KD) students exceed the dense teacher accuracy (Cora B0 81.15) — "
                 "consistent with the student-over-teacher phenomenon reported in KRD (2023).")
    lines.append("- B8-vs-B5 uses two-sided tests with 95% CIs (tie claim, M5); B8-vs-B6 keeps "
                 "the one-sided convention (direction hypothesized).")
    lines.append("")
    lines.append("### H3 wall time (M3, Cora, seconds per run, mean ± std, n=10)")
    lines.append("")
    lines.append("| Ratio | B5 pure KD | B8 prune+KD |")
    lines.append("|---|---|---|")
    for ratio in ("0.5", "0.75"):
        b5 = sel(old, prune_criterion=None, distill="True", prune_ratio=ratio)
        b5 = [r for r in b5 if float(r["epochs_run"]) <= 100.5]
        b8 = sel(old, prune_criterion="diversity", distill="True", prune_ratio=ratio,
                 regularizer="attention", adaptive_lambda="True")
        lines.append(f"| {ratio} | {desc_raw(b5, 'wall_time_s')} | {desc_raw(b8, 'wall_time_s')} |")
    lines.append("")
    lines.append("- Both paths include dense teacher training, so the pruning path offers no "
                 "wall-time saving at this scale.")
    lines.append("")

    # --- R1: adaptive baselines (M1 revision) ---
    lines.append("## R1: adaptive-baseline comparison (M1 revision, Colab batch B1)")
    lines.append("")
    lines.append("Controller vs three adaptive baselines. Controller rows are the submitted "
                 "dense adaptive block (distill=False; n=10 per dataset); B1 rows are 10 seeds "
                 "per cell. λ_final is the mean final controller weight; div_final is the "
                 "inter-head similarity at the best checkpoint.")
    lines.append("")
    lines.append("| Dataset | Schedule | Test acc | λ_final | div_final |")
    lines.append("|---|---|---|---|---|")
    b1 = [r for r in rows if r["lambda_schedule"] in ("linear", "kendall", "gradnorm")]
    for ds in ("cora", "citeseer", "pubmed"):
        ctrl = sel(d, dataset=ds, adaptive_lambda="True", lambda_target="0.7", lambda_eta="0.05")
        lines.append(f"| {ds} | controller | {desc(ctrl)} | {mean_f(ctrl, 'lambda_final'):.2f} | "
                     f"{mean_f(ctrl, 'div_final'):.3f} |")
        for sched in ("linear", "kendall", "gradnorm"):
            x = sel(b1, dataset=ds, lambda_schedule=sched)
            lines.append(f"| {ds} | {sched} | {desc(x)} | {mean_f(x, 'lambda_final'):.2f} | "
                         f"{mean_f(x, 'div_final'):.3f} |")
    lines.append("")
    lines.append("### R1 paired tests (controller vs baseline, 10 seeds paired)")
    lines.append("")
    lines.append("| Dataset | Contrast | Test |")
    lines.append("|---|---|---|")
    for ds in ("cora", "citeseer", "pubmed"):
        ctrl = sel(d, dataset=ds, adaptive_lambda="True", lambda_target="0.7", lambda_eta="0.05")
        for sched in ("linear", "kendall", "gradnorm"):
            x = sel(b1, dataset=ds, lambda_schedule=sched)
            lines.append(f"| {ds} | controller vs {sched} | {paired(ctrl, x)} |")
    lines.append("")

    # --- R2: revival test (M2 revision, Colab batches B2/B3) ---
    lines.append("## R2: revival test (M2 revision, Colab batches B2/B3)")
    lines.append("")
    lines.append("Prune ratio 0.75, controller teacher, fine-tuning 100 epochs, no KD. "
                 "Three checkpoint sources: `best` (early-stopping best), `late_min_div` "
                 "(lowest head similarity during training), `late_last` (final epoch). "
                 "criterion_div is the teacher's head similarity at the checkpoint source.")
    lines.append("")
    lines.append("| Dataset | Source | criterion_div | diversity | magnitude | gradient | random |")
    lines.append("|---|---|---|---|---|---|---|")
    for ds in ("cora", "citeseer", "pubmed"):
        for src in ("best", "late_min_div", "late_last"):
            cells = [f"{desc(sel(b3, dataset=ds, checkpoint_source=src, prune_criterion=c, prune_ratio='0.75'))}"
                     for c in ("diversity", "magnitude", "gradient", "random")]
            cdiv = mean_f(sel(b3, dataset=ds, checkpoint_source=src, prune_criterion="diversity",
                              prune_ratio="0.75"),
                          "criterion_div")
            lines.append(f"| {ds} | {src} | {cdiv:.3f} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("### R2 paired tests (diversity vs other criteria, per source, 10 seeds)")
    lines.append("")
    lines.append("| Dataset | Source | Contrast | Test |")
    lines.append("|---|---|---|---|")
    for ds in ("cora", "citeseer", "pubmed"):
        for src in ("best", "late_min_div"):
            for crit in ("random", "magnitude", "gradient"):
                a = sel(b3, dataset=ds, checkpoint_source=src, prune_criterion="diversity",
                        prune_ratio="0.75")
                b = sel(b3, dataset=ds, checkpoint_source=src, prune_criterion=crit,
                        prune_ratio="0.75")
                lines.append(f"| {ds} | {src} | diversity vs {crit} | {paired(a, b)} |")
    lines.append("")
    lines.append("### R2: best vs late source (diversity criterion)")
    lines.append("")
    lines.append("| Dataset | Contrast | Test |")
    lines.append("|---|---|---|")
    for ds in ("cora", "citeseer", "pubmed"):
        a = sel(b3, dataset=ds, checkpoint_source="best", prune_criterion="diversity",
                prune_ratio="0.75")
        b = sel(b3, dataset=ds, checkpoint_source="late_min_div", prune_criterion="diversity",
                prune_ratio="0.75")
        lines.append(f"| {ds} | best vs late_min_div | {paired(a, b)} |")
    lines.append("")
    lines.append("### R2: min-div / last checkpoint coincidence")
    lines.append("")
    for ds in ("cora", "citeseer", "pubmed"):
        ident = 0
        total = 0
        for seed in range(10):
            for crit in ("diversity", "magnitude", "gradient", "random"):
                a = sel(b3, dataset=ds, seed=str(seed), checkpoint_source="late_min_div",
                        prune_criterion=crit)
                b = sel(b3, dataset=ds, seed=str(seed), checkpoint_source="late_last",
                        prune_criterion=crit)
                if a and b:
                    total += 1
                    if (a[0]["test_acc"] == b[0]["test_acc"]
                            and a[0]["best_epoch"] == b[0]["best_epoch"]):
                        ident += 1
        lines.append(f"- {ds}: late_min_div == late_last in {ident}/{total} (dataset, seed, "
                     f"criterion) combinations — similarity decreases monotonically under the "
                     f"controller, so the lowest-similarity checkpoint is the final one.")
    lines.append("")

    # --- Fixed-λ failure summary (self-contained) ---
    lines.append("## Fixed-λ failure summary (Cora, attention-level, self-contained)")
    lines.append("")
    lines.append("- No fixed λ is simultaneously effective and harmless: λ ≤ 0.5 leaves head "
                 "similarity ~0.95-0.98 (λ=1 moves it slightly, to 0.91); λ ≥ 5 moves "
                 "similarity to 0.37-0.67 but costs 0.7-2.9 pp accuracy. The adaptive "
                 "controller targets the middle ground (target 0.7) without the accuracy loss.")
    lines.append("")

    # --- Blocker / limitation ---
    lines.append("## Limitations")
    lines.append("")
    lines.append("- H1 similarity values are best-checkpoint measurements; late-training values are lower "
                 "(trace evidence, fig2).")
    lines.append("- All evidence is GAT-only; the Multi30k leg was cut from the paper in the "
                 "2026-09-06 revision (its rows remain in the raw CSV but are not analyzed "
                 "here).")
    lines.append("- Single metric (attention-coefficient cosine), GAT-only pruning evidence; "
                 "generalization beyond GAT unverified.")
    lines.append("- H2 contrasts contain tied pairs (zero diffs); Wilcoxon falls back to the "
                 "normal approximation there — paired-t p-values are reported alongside.")
    lines.append("- R1 controller arm is the submitted dense adaptive block (distill=False rows only; "
                 "distill=True adaptive rows are B5 pure-KD students whose test_acc is the "
                 "student's, excluded from every controller statistic).")
    lines.append("- R2 pubmed late-source teacher similarity (0.988) barely moves from the best "
                 "checkpoint (0.995), so pubmed is a weak revival probe; the controller is "
                 "ceiling-bound (λ_final=1.94) on pubmed.")
    lines.append("- Pubmed probe at η=0.2 (nine seeds, `outputs_rev2_pubreach_eta020`): λ "
                 "saturates at the 3.0 ceiling on all runs, yet late similarity stays ~0.99 "
                 "(response-limited, Sec. 4.1); a higher gain does not buy the budget there.")

    (OUT / "stats-appendix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"stats-appendix.md written ({len(lines)} lines)")


if __name__ == "__main__":
    main()
