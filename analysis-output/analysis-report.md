# Analysis Report: Diversity-Budget Adaptive Regularization for Multi-Head Attention

Date: 2026-08-20. Data: `outputs/tables/results_clean.csv` (316 rows, deduplicated — see
`analysis-output/dedup-report.md`). Full numbers: `stats-appendix.md` (auto-generated).
Figures: `figures/` (catalog: `figure-catalog.md`).

## Analysis Questions

1. (H1) Can feedback-controlled adaptive λ reduce inter-head similarity without
   accuracy loss, where fixed λ fails (too weak, or too strong)?
2. (H2) Does the L_div similarity signal serve as a better pruning criterion than
   gradient / random / magnitude baselines?
3. (H3) Does KD after pruning significantly improve over fine-tuning alone, and
   how does the full pipeline compare to pure KD at matched budgets?
4. (M5) Does the Transformer/Multi30k leg corroborate the GAT findings?

## Key Findings

### H1 — SUPPORTED (no-harm + divergence control)

| Dataset | Baseline | Fixed λ=20 | Adaptive (t=0.7, η=0.05) |
|---|---|---|---|
| Cora | 81.15 ± 1.02 | 79.98 ± 1.08 | 80.79 ± 0.92 |
| Citeseer | 68.27 ± 1.56 | 66.55 ± 2.00 | 68.46 ± 1.35 |
| Pubmed | 77.52 ± 0.54 | 76.30 ± 0.68 | 77.44 ± 0.67 |

- Adaptive λ keeps accuracy within ±0.4 of baseline on all three datasets
  (10 seeds each; all CIs overlap).
- Fixed λ that actually moves similarity (λ ≥ 5) hurts accuracy (−1.2 to −1.7).
  Fixed λ ≤ 1 leaves similarity untouched (~0.97, probe evidence).
- Traces show the closed loop working: similarity falls from ~0.995 toward the
  0.7 budget as the integral controller raises λ, while val accuracy stays flat.
- Caveat: `div_final` at the **best checkpoint** remains high (Cora 0.934) — the
  divergence happens late in training, after the selected checkpoint.

### H2 — REJECTED (honest negative)

- Diversity criterion vs random / gradient / magnitude, paired over identical
  seeds, FT=100, no KD: all contrasts null.
  - Adaptive teacher, ratio 0.75: diffs −0.11 … −0.04 pp, Wilcoxon p = 0.50–0.62 (10 pairs).
  - Adaptive teacher, ratio 0.5: p = 0.77–1.00 (5 pairs; random numerically +1.20 pp better).
  - B0 teacher: same null pattern (5 pairs).
- Mechanism consistent with the evidence: the criterion is computed at the best
  checkpoint, where head similarity is high for every config (0.93–0.99) — the
  L_div signal does not differentiate heads at the point where pruning happens.
- This matches Michel et al. (2019): at GAT scale, fine-tuning washes out
  criterion choice. Reported as a negative result with strict paired tests.
- Statistical note: some contrasts contain tied pairs; Wilcoxon falls back to
  the normal approximation there (paired-t p reported alongside, same verdict).

### H3 — KD SUPPORTED; pipeline vs pure KD TIED (Risk-3 materialized)

| Ratio | B5 pure KD | B6 prune+FT | B8 prune+KD | B8 vs B6 | B8 vs B5 |
|---|---|---|---|---|---|
| 0.5 | 82.48 ± 1.08 | 79.89 ± 0.64 | 82.00 ± 0.69 | +2.11, p=0.001 | −0.48, p=0.92 |
| 0.75 | 82.88 ± 1.16 | 79.66 ± 1.56 | 82.04 ± 1.46 | +2.38, p=0.001 | −0.84, p=0.95 |

- KD recovers most of the pruning loss: B8 vs B6 +2.1…+2.4 pp, Wilcoxon p=0.001
  (10 pairs), Cohen's d ≈ 2.1.
- B8 vs B5 (pure KD, matched 100-epoch budget): statistically tied, pure KD
  numerically +0.5/+0.8 pp higher. Pure KD alone already captures the recovery.
- Bonus: pure-KD students exceed the dense teacher (82.5–82.9 vs 81.15 Cora),
  consistent with student-over-teacher KD findings (KRD, 2023).

### M5 — NOT MET (exploratory only)

- Multi30k (sacrebleu corpus-BLEU, full val split, 50 epochs, patience 50):
  none 10.49 ± 2.62 (n=5), adaptive 12.81 ± 3.44 (n=6), fixed-λ20 23.67 ± 16.94 (n=5).
- Absolute BLEU is depressed by the tokenizer mismatch and the known
  corpus-BLEU / template-degeneracy traps (see `plan/task_plan.md`); only
  within-study relative comparisons are reported for this leg.
- Relative pattern agrees with GAT (adaptive ≈ baseline); fixed-λ20 is wildly
  unstable (std 16.9). Use only as weak corroboration, never as an absolute claim.

## Claim Candidates

- Claim: Adaptive-λ diversity regularization preserves accuracy while fixed-λ
  regularization either does nothing (λ ≤ 1) or degrades accuracy (λ ≥ 5), on
  three citation networks (10 seeds each).
  - Source evidence: stats-appendix H1 table + fixed-λ spectrum + trace fig2.
  - Allowed wording: "adaptive λ incurs no significant accuracy loss (within
    ±0.4 pp across datasets) and enforces the diversity budget late in training;
    fixed λ either fails to move similarity or costs 1.2–1.7 pp."
  - Forbidden stronger wording: "adaptive λ improves accuracy"; "reduces
    redundancy at the best checkpoint" (div_final stays high, ~0.93).
  - Uncertainty: cora adaptive leg uses the Colab T4 10-seed batch (job logs not
    archived — see dedup-report caveat 1).
  - Next check: re-run cora adaptive dense on T4 with job-log archiving if the
    provenance is needed for the camera-ready.
  - Decision: keep (with the no-improvement wording).

- Claim: KD after diversity-based pruning recovers pruned-model accuracy (+2.1–2.4
  pp over fine-tuning alone) and matches pure KD at the same head budget.
  - Source evidence: stats-appendix H3 table, fig3.
  - Allowed wording: "KD restores most of the pruning loss and matches the
    accuracy of training a small student from scratch with KD; the structured
    (pruned) student reaches this accuracy via a cheaper re-training path."
  - Forbidden stronger wording: "B8 outperforms pure KD"; "pruning+KD beats KD"
    (it does not: −0.5/−0.8 pp, n.s.).
  - Uncertainty: B8 vs B5 power is limited (10 seeds, |d| ≈ 0.5–0.8 pp).
  - Next check: none required for a tied-statement.
  - Decision: keep with "matches, not beats" wording.

- Claim: The L_div similarity criterion selects better heads to keep than
  gradient / random / magnitude.
  - Source evidence: stats-appendix H2 (all contrasts null).
  - Allowed wording: "at GAT scale with 100-epoch fine-tuning, the similarity
    criterion performs on par with strong baselines; we find no evidence that it
    is better (negative result)."
  - Forbidden stronger wording: any statement of criterion superiority.
  - Uncertainty: criterion computed at best checkpoint where similarities are
    high for all configs; a late-training criterion might differ (untested).
  - Next check: optional — compute the criterion from a late-training checkpoint
    and test whether H2 revives; if not, drop the criterion from the paper's
    contribution claims.
  - Decision: discard as a positive claim; keep as an honest negative if useful.

## Limitations / Blockers

- Cora adaptive-dense 10-seed batch has no archived Hydra job logs (Colab T4) —
  provenance caveat for the reproducibility section (dedup-report caveat 1).
- Traces on disk are same-protocol local runs, not the T4 batch (dedup-report
  caveat 2); trace figures describe controller dynamics, not the exact reported rows.
- No timestamp column in results.csv (suggested pipeline improvement).
- Multi30k leg: absolute BLEU not usable (tokenizer/corpus-BLEU issues); treat as exploratory.
- Single metric family (attention-coefficient cosine), GAT-only pruning evidence.

## QA Gate

- [x] primary comparison question explicit (H1–H3, M5)
- [x] sample size / seed count stated (10 seeds main; 5–6 auxiliary)
- [x] inferential tests justified (paired Wilcoxon + paired-t; ties → normal
      approximation, stated)
- [x] effect sizes reported for major contrasts (Cohen's d for H2/H3)
- [x] real figures exist (fig1, fig2 ×2, fig3)
- [x] each figure has an interpretation note (figure-catalog.md)
- [x] limitations and blockers explicit
- [x] claim candidates carry evidence, uncertainty, allowed/forbidden wording
- [x] over-strong wording explicitly blocked (criterion superiority; B8 > B5)
- [x] no manuscript-style Results draft included
