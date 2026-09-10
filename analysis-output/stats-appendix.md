# Statistics Appendix (auto-generated, 2026-09-10)

Data source: `outputs/tables/results_clean.csv` (merged 2026-09-05, see `temp/merge_colab_20260905.py` and `analysis-output/dedup-report.md`). Sections H1-H3: old protocol (submitted version); sections R1-R2: revision protocol (Colab batches B1/B2/B3). All rows: dense training runs unless noted. GAT 2-layer, 8 heads, hidden 8/head, fixed Planetoid split, early stopping patience 100, 10 seeds per config unless stated.

## H1: adaptive lambda vs fixed lambda vs baseline

| Dataset | B0 baseline | Fixed λ=20 | Adaptive (t=0.7, η=0.05) |
|---|---|---|---|
| cora | 81.15 ± 1.02 (n=10, 95% CI [80.42, 81.88]) | 79.71 ± 1.24 (n=10, 95% CI [78.82, 80.60]) | 80.79 ± 0.92 (n=10, 95% CI [80.13, 81.45]) |
| citeseer | 68.27 ± 1.56 (n=10, 95% CI [67.15, 69.39]) | 66.55 ± 2.00 (n=10, 95% CI [65.12, 67.98]) | 68.46 ± 1.35 (n=10, 95% CI [67.49, 69.43]) |
| pubmed | 77.52 ± 0.54 (n=10, 95% CI [77.13, 77.91]) | 76.30 ± 0.68 (n=10, 95% CI [75.81, 76.79]) | 77.44 ± 0.67 (n=10, 95% CI [76.96, 77.92]) |

- cora: adaptive runs end with λ_final=0.84, div_final=0.934 (best-checkpoint similarity; n=10 protocol repetitions, see traces for late-training values)
- citeseer: adaptive runs end with λ_final=0.27, div_final=0.988 (best-checkpoint similarity; n=10 protocol repetitions, see traces for late-training values)
- pubmed: adaptive runs end with λ_final=1.94, div_final=0.996 (best-checkpoint similarity; n=10 protocol repetitions, see traces for late-training values)

### Fixed-λ failure spectrum (Cora, attention-level)

Second-revision sweep (`outputs_rev2_sweep/tables/results.csv`): true n=10 per weight with recorded best-checkpoint similarities, superseding the pre-submission n=5 values (the old rows stored 5 duplicated seeds for λ>=5 and probe values for λ<=1).

| λ | test acc | div_final |
|---|---|---|
| 0.01 | 81.16 ± 1.63 (n=10, 95% CI [79.99, 82.33]) | 0.979 |
| 0.1 | 81.03 ± 1.50 (n=10, 95% CI [79.96, 82.10]) | 0.978 |
| 0.5 | 81.13 ± 1.37 (n=10, 95% CI [80.15, 82.11]) | 0.954 |
| 1.0 | 81.08 ± 1.68 (n=10, 95% CI [79.88, 82.28]) | 0.906 |
| 5.0 | 80.50 ± 1.44 (n=10, 95% CI [79.47, 81.53]) | 0.674 |
| 20.0 | 79.71 ± 1.24 (n=10, 95% CI [78.82, 80.60]) | 0.597 |
| 50.0 | 78.22 ± 1.88 (n=10, 95% CI [76.87, 79.57]) | 0.366 |

## H2: pruning criterion comparison (paired, same seeds, FT=100, no KD)

| Source | Ratio | Contrast | Test |
|---|---|---|---|
| adaptive t0.7 | 0.5 | diversity vs random | diff -1.20 pp, Wilcoxon p=1.0000, paired-t p=0.9828, Cohen's d=-1.41, 5 pairs |
| adaptive t0.7 | 0.5 | diversity vs gradient | diff -0.22 pp, Wilcoxon p=0.7812, paired-t p=0.6893, Cohen's d=-0.24, 5 pairs |
| adaptive t0.7 | 0.5 | diversity vs magnitude | diff -0.26 pp, Wilcoxon p=0.8125, paired-t p=0.7677, Cohen's d=-0.36, 5 pairs |
| B0 | 0.75 | diversity vs random | diff -0.48 pp, Wilcoxon p=0.9062, paired-t p=0.8254, Cohen's d=-0.47, 5 pairs |
| B0 | 0.75 | diversity vs gradient | diff +0.04 pp, Wilcoxon p=0.5000, paired-t p=0.4778, Cohen's d=0.03, 5 pairs |
| B0 | 0.75 | diversity vs magnitude | diff +0.12 pp, Wilcoxon p=0.6875, paired-t p=0.4110, Cohen's d=0.11, 5 pairs |
| B0 | 0.5 | diversity vs random | diff -0.34 pp, Wilcoxon p=0.7812, paired-t p=0.8067, Cohen's d=-0.43, 5 pairs |
| B0 | 0.5 | diversity vs gradient | diff +0.62 pp, Wilcoxon p=0.0938, paired-t p=0.0485, Cohen's d=0.97, 5 pairs |
| B0 | 0.5 | diversity vs magnitude | diff -0.20 pp, Wilcoxon p=0.5938, paired-t p=0.6924, Cohen's d=-0.24, 5 pairs |

### H2 note: adaptive-teacher ratio-0.75 contrasts use B3 `best`-source rows

The pre-merge adaptive-teacher ratio-0.75 rows were dropped in the 2026-09-05 merge as semantic duplicates of the new B3 `best`-source rows (same protocol, post-bugfix trainer, explicit checkpoint_source). The submitted H2 numbers for adaptive 0.75 are superseded by:

| Source | Ratio | Contrast | Test |
|---|---|---|---|
| adaptive t0.7 (B3 best) | 0.75 | diversity vs random | diff +0.05 pp, Wilcoxon p=0.2783, paired-t p=0.4466, Cohen's d=0.04, 10 pairs |
| adaptive t0.7 (B3 best) | 0.75 | diversity vs gradient | diff -0.13 pp, Wilcoxon p=0.6289, paired-t p=0.7593, Cohen's d=-0.23, 10 pairs |
| adaptive t0.7 (B3 best) | 0.75 | diversity vs magnitude | diff +0.01 pp, Wilcoxon p=0.5391, paired-t p=0.4807, Cohen's d=0.02, 10 pairs |

## H3: distillation (adaptive teacher t0.7, matched 100-epoch budget)

| Ratio | B5 pure KD | B6 prune+FT | B8 prune+KD | B8 vs B6 | B8 vs B5 |
|---|---|---|---|---|---|
| 0.5 | 82.48 ± 1.08 (n=10, 95% CI [81.70, 83.26]) | 79.89 ± 0.64 (n=10, 95% CI [79.43, 80.35]) | 82.00 ± 0.69 (n=10, 95% CI [81.50, 82.50]) | diff +2.11 pp, Wilcoxon p=0.0010, paired-t p=0.0001, Cohen's d=2.06, 10 pairs | diff -0.48 pp, 95% CI [-1.12, +0.16], two-sided Wilcoxon p=0.1934, paired-t p=0.1252, 10 pairs |
| 0.75 | 82.88 ± 1.16 (n=10, 95% CI [82.05, 83.71]) | n/a | 82.04 ± 1.46 (n=10, 95% CI [80.99, 83.09]) | n/a | diff -0.84 pp, 95% CI [-1.79, +0.11], two-sided Wilcoxon p=0.1348, paired-t p=0.0774, 10 pairs |

- Note: B5 (pure KD) students exceed the dense teacher accuracy (Cora B0 81.15) — consistent with the student-over-teacher phenomenon reported in KRD (2023).
- B8-vs-B5 uses two-sided tests with 95% CIs (tie claim, M5); B8-vs-B6 keeps the one-sided convention (direction hypothesized).

### H3 wall time (M3, Cora, seconds per run, mean ± std, n=10)

| Ratio | B5 pure KD | B8 prune+KD |
|---|---|---|
| 0.5 | 7.87 ± 1.12 (n=10) | 8.79 ± 0.12 (n=10) |
| 0.75 | 8.40 ± 0.10 (n=10) | 8.27 ± 0.15 (n=10) |

- Both paths include dense teacher training, so the pruning path offers no wall-time saving at this scale.

## R1: adaptive-baseline comparison (M1 revision, Colab batch B1)

Controller vs three adaptive baselines. Controller rows are the submitted dense adaptive block (distill=False; n=10 per dataset); B1 rows are 10 seeds per cell. λ_final is the mean final controller weight; div_final is the inter-head similarity at the best checkpoint.

| Dataset | Schedule | Test acc | λ_final | div_final |
|---|---|---|---|---|
| cora | controller | 80.79 ± 0.92 (n=10, 95% CI [80.13, 81.45]) | 0.84 | 0.934 |
| cora | linear | 80.83 ± 1.05 (n=10, 95% CI [80.08, 81.58]) | 0.87 | 0.924 |
| cora | kendall | 80.83 ± 1.21 (n=10, 95% CI [79.97, 81.69]) | 1.37 | 0.849 |
| cora | gradnorm | 79.04 ± 1.57 (n=10, 95% CI [77.91, 80.17]) | 1.95 | 0.693 |
| citeseer | controller | 68.46 ± 1.35 (n=10, 95% CI [67.49, 69.43]) | 0.27 | 0.988 |
| citeseer | linear | 68.94 ± 1.01 (n=10, 95% CI [68.22, 69.66]) | 0.30 | 0.976 |
| citeseer | kendall | 68.66 ± 0.90 (n=10, 95% CI [68.01, 69.31]) | 1.10 | 0.969 |
| citeseer | gradnorm | 67.36 ± 1.35 (n=10, 95% CI [66.40, 68.32]) | 1.95 | 0.688 |
| pubmed | controller | 77.44 ± 0.67 (n=10, 95% CI [76.96, 77.92]) | 1.94 | 0.996 |
| pubmed | linear | 77.23 ± 0.79 (n=10, 95% CI [76.67, 77.79]) | 1.67 | 0.995 |
| pubmed | kendall | 77.32 ± 0.76 (n=10, 95% CI [76.78, 77.86]) | 1.26 | 0.996 |
| pubmed | gradnorm | 76.07 ± 0.67 (n=10, 95% CI [75.59, 76.55]) | 1.95 | 1.000 |

### R1 paired tests (controller vs baseline, 10 seeds paired)

| Dataset | Contrast | Test |
|---|---|---|
| cora | controller vs linear | diff -0.04 pp, Wilcoxon p=0.4531, paired-t p=0.6571, Cohen's d=-0.13, 10 pairs |
| cora | controller vs kendall | diff -0.04 pp, Wilcoxon p=0.5781, paired-t p=0.5587, Cohen's d=-0.05, 10 pairs |
| cora | controller vs gradnorm | diff +1.75 pp, Wilcoxon p=0.0098, paired-t p=0.0063, Cohen's d=0.98, 10 pairs |
| citeseer | controller vs linear | diff -0.48 pp, Wilcoxon p=0.7764, paired-t p=0.8113, Cohen's d=-0.29, 10 pairs |
| citeseer | controller vs kendall | diff -0.20 pp, Wilcoxon p=0.6055, paired-t p=0.6662, Cohen's d=-0.14, 10 pairs |
| citeseer | controller vs gradnorm | diff +1.10 pp, Wilcoxon p=0.0527, paired-t p=0.0628, Cohen's d=0.53, 10 pairs |
| pubmed | controller vs linear | diff +0.21 pp, Wilcoxon p=0.3477, paired-t p=0.2251, Cohen's d=0.25, 10 pairs |
| pubmed | controller vs kendall | diff +0.12 pp, Wilcoxon p=0.3389, paired-t p=0.3100, Cohen's d=0.16, 10 pairs |
| pubmed | controller vs gradnorm | diff +1.37 pp, Wilcoxon p=0.0049, paired-t p=0.0014, Cohen's d=1.29, 10 pairs |

## R2: revival test (M2 revision, Colab batches B2/B3)

Prune ratio 0.75, controller teacher, fine-tuning 100 epochs, no KD. Three checkpoint sources: `best` (early-stopping best), `late_min_div` (lowest head similarity during training), `late_last` (final epoch). criterion_div is the teacher's head similarity at the checkpoint source.

| Dataset | Source | criterion_div | diversity | magnitude | gradient | random |
|---|---|---|---|---|---|---|
| cora | best | 0.934 | 80.32 ± 0.78 (n=10, 95% CI [79.76, 80.88]) | 80.31 ± 0.59 (n=10, 95% CI [79.89, 80.73]) | 80.45 ± 0.66 (n=10, 95% CI [79.98, 80.92]) | 80.27 ± 0.75 (n=10, 95% CI [79.73, 80.81]) |
| cora | late_min_div | 0.706 | 80.36 ± 1.49 (n=10, 95% CI [79.30, 81.42]) | 79.95 ± 0.86 (n=10, 95% CI [79.34, 80.56]) | 79.88 ± 1.01 (n=10, 95% CI [79.16, 80.60]) | 79.74 ± 1.05 (n=10, 95% CI [78.99, 80.49]) |
| cora | late_last | 0.706 | 80.36 ± 1.49 (n=10, 95% CI [79.30, 81.42]) | 79.95 ± 0.86 (n=10, 95% CI [79.34, 80.56]) | 79.88 ± 1.01 (n=10, 95% CI [79.16, 80.60]) | 79.74 ± 1.05 (n=10, 95% CI [78.99, 80.49]) |
| citeseer | best | 0.972 | 67.95 ± 1.17 (n=10, 95% CI [67.11, 68.79]) | 67.94 ± 1.13 (n=10, 95% CI [67.13, 68.75]) | 67.77 ± 1.36 (n=10, 95% CI [66.80, 68.74]) | 68.35 ± 1.05 (n=10, 95% CI [67.60, 69.10]) |
| citeseer | late_min_div | 0.847 | 66.05 ± 1.35 (n=10, 95% CI [65.08, 67.02]) | 66.21 ± 1.80 (n=10, 95% CI [64.93, 67.49]) | 66.65 ± 1.67 (n=10, 95% CI [65.46, 67.84]) | 66.87 ± 1.19 (n=10, 95% CI [66.02, 67.72]) |
| citeseer | late_last | 0.847 | 66.05 ± 1.35 (n=10, 95% CI [65.08, 67.02]) | 66.18 ± 1.86 (n=10, 95% CI [64.85, 67.51]) | 66.65 ± 1.67 (n=10, 95% CI [65.46, 67.84]) | 66.87 ± 1.19 (n=10, 95% CI [66.02, 67.72]) |
| pubmed | best | 0.995 | 76.35 ± 0.93 (n=10, 95% CI [75.69, 77.01]) | 76.33 ± 1.02 (n=10, 95% CI [75.60, 77.06]) | 77.02 ± 0.72 (n=10, 95% CI [76.51, 77.53]) | 76.77 ± 1.51 (n=10, 95% CI [75.69, 77.85]) |
| pubmed | late_min_div | 0.988 | 76.17 ± 1.25 (n=10, 95% CI [75.27, 77.07]) | 75.98 ± 1.03 (n=10, 95% CI [75.25, 76.71]) | 76.51 ± 0.76 (n=10, 95% CI [75.97, 77.05]) | 77.04 ± 1.31 (n=10, 95% CI [76.11, 77.97]) |
| pubmed | late_last | 0.988 | 76.17 ± 1.25 (n=10, 95% CI [75.27, 77.07]) | 75.98 ± 1.03 (n=10, 95% CI [75.25, 76.71]) | 76.51 ± 0.76 (n=10, 95% CI [75.97, 77.05]) | 77.04 ± 1.31 (n=10, 95% CI [76.11, 77.97]) |

### R2 paired tests (diversity vs other criteria, per source, 10 seeds)

| Dataset | Source | Contrast | Test |
|---|---|---|---|
| cora | best | diversity vs random | diff +0.05 pp, Wilcoxon p=0.2783, paired-t p=0.4466, Cohen's d=0.04, 10 pairs |
| cora | best | diversity vs magnitude | diff +0.01 pp, Wilcoxon p=0.5391, paired-t p=0.4807, Cohen's d=0.02, 10 pairs |
| cora | best | diversity vs gradient | diff -0.13 pp, Wilcoxon p=0.6289, paired-t p=0.7593, Cohen's d=-0.23, 10 pairs |
| cora | late_min_div | diversity vs random | diff +0.62 pp, Wilcoxon p=0.1162, paired-t p=0.1153, Cohen's d=0.41, 10 pairs |
| cora | late_min_div | diversity vs magnitude | diff +0.41 pp, Wilcoxon p=0.0781, paired-t p=0.0779, Cohen's d=0.49, 10 pairs |
| cora | late_min_div | diversity vs gradient | diff +0.48 pp, Wilcoxon p=0.1436, paired-t p=0.1223, Cohen's d=0.39, 10 pairs |
| citeseer | best | diversity vs random | diff -0.40 pp, Wilcoxon p=0.7842, paired-t p=0.7950, Cohen's d=-0.27, 10 pairs |
| citeseer | best | diversity vs magnitude | diff +0.01 pp, Wilcoxon p=0.5000, paired-t p=0.4921, Cohen's d=0.01, 10 pairs |
| citeseer | best | diversity vs gradient | diff +0.18 pp, Wilcoxon p=0.3965, paired-t p=0.3327, Cohen's d=0.14, 10 pairs |
| citeseer | late_min_div | diversity vs random | diff -0.82 pp, Wilcoxon p=0.9502, paired-t p=0.9475, Cohen's d=-0.57, 10 pairs |
| citeseer | late_min_div | diversity vs magnitude | diff -0.16 pp, Wilcoxon p=0.6289, paired-t p=0.6122, Cohen's d=-0.09, 10 pairs |
| citeseer | late_min_div | diversity vs gradient | diff -0.60 pp, Wilcoxon p=0.8691, paired-t p=0.8296, Cohen's d=-0.32, 10 pairs |
| pubmed | best | diversity vs random | diff -0.42 pp, Wilcoxon p=0.7842, paired-t p=0.7917, Cohen's d=-0.27, 10 pairs |
| pubmed | best | diversity vs magnitude | diff +0.02 pp, Wilcoxon p=0.6406, paired-t p=0.4533, Cohen's d=0.04, 10 pairs |
| pubmed | best | diversity vs gradient | diff -0.67 pp, Wilcoxon p=0.9629, paired-t p=0.9588, Cohen's d=-0.62, 10 pairs |
| pubmed | late_min_div | diversity vs random | diff -0.87 pp, Wilcoxon p=0.9756, paired-t p=0.9589, Cohen's d=-0.62, 10 pairs |
| pubmed | late_min_div | diversity vs magnitude | diff +0.19 pp, Wilcoxon p=0.2656, paired-t p=0.1509, Cohen's d=0.35, 10 pairs |
| pubmed | late_min_div | diversity vs gradient | diff -0.34 pp, Wilcoxon p=0.8203, paired-t p=0.8391, Cohen's d=-0.33, 10 pairs |

### R2: best vs late source (diversity criterion)

| Dataset | Contrast | Test |
|---|---|---|
| cora | best vs late_min_div | diff -0.04 pp, Wilcoxon p=0.5098, paired-t p=0.5530, Cohen's d=-0.04, 10 pairs |
| citeseer | best vs late_min_div | diff +1.90 pp, Wilcoxon p=0.0010, paired-t p=0.0000, Cohen's d=2.73, 10 pairs |
| pubmed | best vs late_min_div | diff +0.18 pp, Wilcoxon p=0.1855, paired-t p=0.1843, Cohen's d=0.30, 10 pairs |

### R2: min-div / last checkpoint coincidence

- cora: late_min_div == late_last in 40/40 (dataset, seed, criterion) combinations — similarity decreases monotonically under the controller, so the lowest-similarity checkpoint is the final one.
- citeseer: late_min_div == late_last in 39/40 (dataset, seed, criterion) combinations — similarity decreases monotonically under the controller, so the lowest-similarity checkpoint is the final one.
- pubmed: late_min_div == late_last in 40/40 (dataset, seed, criterion) combinations — similarity decreases monotonically under the controller, so the lowest-similarity checkpoint is the final one.

## Fixed-λ failure summary (Cora, attention-level, self-contained)

- No fixed λ is simultaneously effective and harmless: λ ≤ 0.5 leaves head similarity ~0.95-0.98 (λ=1 moves it slightly, to 0.91); λ ≥ 5 moves similarity to 0.37-0.67 but costs 0.7-2.9 pp accuracy. The adaptive controller targets the middle ground (target 0.7) without the accuracy loss.

## Limitations

- H1 similarity values are best-checkpoint measurements; late-training values are lower (trace evidence, fig2).
- All evidence is GAT-only; the Multi30k leg was cut from the paper in the 2026-09-06 revision (its rows remain in the raw CSV but are not analyzed here).
- Single metric (attention-coefficient cosine), GAT-only pruning evidence; generalization beyond GAT unverified.
- H2 contrasts contain tied pairs (zero diffs); Wilcoxon falls back to the normal approximation there — paired-t p-values are reported alongside.
- R1 controller arm is the submitted dense adaptive block (distill=False rows only; distill=True adaptive rows are B5 pure-KD students whose test_acc is the student's, excluded from every controller statistic).
- R2 pubmed late-source teacher similarity (0.988) barely moves from the best checkpoint (0.995), so pubmed is a weak revival probe; the controller is ceiling-bound (λ_final=1.94) on pubmed.
- Pubmed probe at η=0.2 (nine seeds, `outputs_rev2_pubreach_eta020`): λ saturates at the 3.0 ceiling on all runs, yet late similarity stays ~0.99 (response-limited, Sec. 4.1); a higher gain does not buy the budget there.
