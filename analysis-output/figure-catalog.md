# Figure Catalog

Regenerated 2026-08-20 from `outputs/tables/results_clean.csv` and `outputs/traces/`.

## fig1_h1_comparison.pdf — Main comparison (H1)

- **Filename**: `figures/fig1_h1_comparison.pdf`
- **Purpose**: Show that adaptive λ keeps test accuracy at baseline across three
  datasets while the fixed λ that moves similarity (λ=20) hurts it everywhere.
- **Data source**: `results_clean.csv` dense rows (Baseline / Fixed λ=20 /
  Adaptive t=0.7 η=0.05), 10 seeds per config per dataset.
- **Plotted**: grouped bars, y = test acc %, error bars = ±1 sample std over seeds.
- **Caption draft**: "Test accuracy of a 2-layer GAT on three citation networks
  under baseline training, fixed-λ (λ=20) diversity regularization, and
  feedback-controlled adaptive regularization (target 0.7, η=0.05). Bars are
  means over 10 seeds; error bars are ±1 std."
- **Key observation**: adaptive ≈ baseline on all three datasets (within 0.4);
  fixed λ=20 below both everywhere (−1.2 to −1.7).
- **Interpretation checklist**:
  1. Does the adaptive bar overlap the baseline bar's error on every dataset?
  2. Is the fixed-λ bar consistently the lowest?
  3. Does the figure support "no harm", not "improvement"?
- **Known caveats**: y-axis starts at 60; stds over seeds, not CIs; cora
  adaptive bar = Colab T4 batch (provenance caveat in dedup-report).

## fig2_trace_{dataset}_seed{seed}.pdf — Controller dynamics (H1 mechanism)

- **Filenames**: `figures/fig2_trace_cora_seed3.pdf`, `figures/fig2_trace_citeseer_seed0.pdf`
- **Purpose**: Show the closed-loop dynamics — similarity falls as λ winds up,
  val accuracy stays flat.
- **Data source**: `outputs/traces/{dataset}_seed{seed}_t0.7_e0.05.csv`
  (same-protocol runs; not the exact rows in the main table — see dedup-report caveat 2).
- **Plotted**: two panels — head similarity d_t with the 0.7 budget line; controller λ over epochs.
- **Caption draft**: "Adaptive-controller dynamics: head similarity d_t falls
  toward the 0.7 budget as the integral controller raises λ; validation
  accuracy (panel title) remains flat."
- **Key observation**: similarity decreases 0.2–0.5 while val acc stays flat —
  the controller enforces the budget late in training without hurting accuracy.
- **Interpretation checklist**:
  1. Does λ rise only after d_t exceeds the budget (controller design)?
  2. Does d_t respond to rising λ (falling curve)?
  3. Does val acc stay flat or improve?
- **Known caveats**: one seed per dataset shown; best-checkpoint similarity
  (div_final in tables, ~0.93) is higher than late-training values — the trace
  is the dynamics evidence, the table column is the checkpoint measurement.

## fig3_h3_distillation.pdf — KD comparison (H3)

- **Filename**: `figures/fig3_h3_distillation.pdf`
- **Purpose**: Show that KD recovers pruned-model accuracy and matches pure KD
  at the same head budget (Risk-3 evidence).
- **Data source**: `results_clean.csv` — B5 (pure KD, 100-ep budget), B6
  (diversity prune + FT, adaptive teacher), B8 (prune + KD), ratios 0.5/0.75,
  10 seeds each.
- **Plotted**: grouped bars per keep-ratio (4/8, 2/8 heads), y = test acc %,
  error bars ±1 std; significance bracket = B8 vs B6 paired Wilcoxon
  (p=0.001 at both ratios, computed from the data, drawn only if p<0.05).
- **Caption draft**: "Accuracy at fixed head budgets: training a smaller student
  from scratch with KD (B5), diversity-based pruning followed by fine-tuning
  (B6), and pruning followed by KD (B8). Bracket: paired Wilcoxon, p=0.001."
- **Key observation**: B8 ≈ B5 > B6; KD (either route) recovers +2.1–2.4 pp
  over fine-tuning alone; pure KD is numerically (n.s.) higher than prune+KD.
- **Interpretation checklist**:
  1. Is B6 the lowest bar at both ratios (pruning loss without KD)?
  2. Do B8 and B5 overlap within error (tied)?
  3. Does the figure support "KD recovers / matches", not "B8 wins"?
- **Known caveats**: the bracket shows the KD effect (B8 vs B6), not a B8 vs B5
  comparison (that contrast is n.s.); stds over seeds, not CIs.
