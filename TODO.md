# TODO

- [x] GAT model + Cora/Citeseer/Pubmed datasets
- [x] L_div regularizer (attention-matrix cosine + output Euclidean) + unit tests (7 passing)
- [x] Hydra training pipeline + result CSV appender
- [x] M1 gate: 环境基线标定 81.15±0.97 (hidden=8, 10 seeds, 与 PyG 官方参考完全一致；论文 83.0 系文献值)
- [x] M2 gate: +attention reg 应显著优于 81.15 (lambda sweep 0.01-1.0)
- [x] Head-pruning module (Michel gradient / random / magnitude / diversity criteria)
- [x] Adaptive lambda (per-layer learnable + scheduled)
- [x] KD module (dense teacher -> pruned student)
- [x] 3-layer Transformer + Multi30k leg (M5 gate)
- [x] Analysis scripts (accuracy-vs-params frontier, Spearman criterion check)
- [x] Paper submitted to IJPR-AI (2026-08-22)
- [x] Revision batches B1/B2/B3 on Colab (2026-09-05): M1 adaptive baselines
      (linear/kendall/gradnorm) + M2 revival matrix (4 criteria × 3 checkpoint
      sources × 10 seeds); results merged, checkpoints archived
- [x] S4: merge + dedup + final_stats; paper updated with R1/R2
      (see analysis-output/revision-20260906.md)

## Remaining (revision)

- [x] Compile main.tex (TeX Live 2026 scheme-basic installed at
      C:\Users\hukej\texlive\2026; main.pdf builds clean: 0 errors, 0 overfull,
      15 pages, 2026-09-06)
- [x] M3: B5 vs B8 wall-time reporting (equal cost; no computational-saving claim)
- [x] M4: Multi30k leg cut (2026-09-06)
- [x] M5: two-sided tests + 95% CIs for the B8-vs-B5 tie claim
- [x] M6: code URL / artifact archive (decision 2026-09-07: keep "released upon
      acceptance" placeholder)
- [x] B4 + H3-extension batch on Colab (612-row Drive CSV, 2026-09-07)
- [x] Merge Colab results (merge_h3ext_colab_20260907.py, 905->1067 rows), fill
      H3 paragraph + tab8_h3ext, recompile (0 errors, 18 pages)
- [x] Refresh submission zip: ijprai-submission-20260907.zip (16 files)
- [x] R3 revision round (2026-09-08, desktop reviews: 审稿报告 +
      reviewer-simulation-20260908.md):
      - R3-M1: Tab 7 column semantics fixed (d_best, not d_late) + epochs axis
      - R3-M2: Tab 1 fully regenerated from the n=10 recorded sweep
        (outputs_rev2_sweep; old λ rows were 5 duplicated seeds, probe cells
        replaced); Appendix A Cora λ=20 CI updated
      - R3-M3: Tab 8 paired Wilcoxon + 95% CIs + Holm in Appendix C text
      - R3-M4: Sec 4.4 limited to Cora + Citeseer/Pubmed reverse evidence
      - R3-M5: Pubmed η=0.2 probe (outputs_rev2_pubreach_eta020) replaces the
        "higher gain" promise: response-limited, λ saturates at 3.0, d stays ~0.99
      - R3-M6: head-fusion reworded as future work; Conclusion adds Pubmed
        prune+KD vs B4 +3 pp data point
      - 审稿报告 items: abstract opening generalized, em-dash purge (~30
        places), Holm p-value wording, nonbreaking spaces before \cite,
        Appendix C guiding text, tab2 caption
      - Fig 1 regenerated (Cora λ=20 = second-revision sweep), final_stats.py
        + make_figures_pub.py made batch-aware
- [x] Recompile (0 errors, 0 overfull, 19 pages) + repack
      ijprai-submission-20260908.zip (16 files)
- [x] Title fixed per review MAJOR 1 (2026-09-09): subtitle
      "Controlling Head Redundancy without Accuracy Loss" ->
      "Enforcing a Similarity Budget Late in Training"; recompiled (0 errors,
      0 overfull, 19 pages) + repacked ijprai-submission-20260909.zip (16 files)
- [x] Self-review round (2026-09-10, paper/self-review-20260910.md):
      AI-trace scan 43/50 (low); P0 fix — R1 "only scheme that enforces the
      budget late" narrowed to best-checkpoint evidence (baseline late-phase
      similarity was never logged; now stated in Sec 4.2, fig4/tab4 captions,
      Conclusion, Limitations); abstract 227->199 words with the "on the
      datasets we test" qualifier restored; 5 style edits; fig4 p-value
      precision unified; 2 uncited Multi30k-era bib entries removed
      (elliott2016, post2018)
- [x] final_stats.py R2 fixes (2026-09-10): distill=False + prune_ratio=0.75
      filters (B8 and B6@0.5 rows were polluting the revival best-source
      cells); stats-appendix.md regenerated, now matches tab5 and the paper
      text (+1.90 pp citeseer best-vs-late, p=0.001)
- [x] Recompile (0 errors, 0 overfull, 19 pages) + repack
      ijprai-submission-20260910.zip (16 files)
- [x] R4 style review (2026-09-10, Desktop/reviewer-simulation-20260910.md,
      verdict Accept/Minor): 必修 fixed — abstract double "from" removed
      ("students trained from scratch with KD"); 建议改 fixed — H2 "those two
      blocks" now names the nine cells explicitly; optional applied — "for
      free" -> "at no accuracy cost", "by our evidence" -> "based on our
      evidence", "exactly" weakened at 2 of 3 sites, "The main practical
      takeaway" -> "The practical lesson". "natural" duplication was already
      gone in the 09-10 draft. Kept per M6 decision: "released upon
      acceptance" placeholder. Recompiled (0 errors, 0 overfull, 19 pages) +
      zip rebuilt.
- [x] Code release prep (2026-09-10, repo public at
      https://github.com/HuKejie/div-prune): LICENSE (MIT) + README rewrite
      (paper title/link, reproducibility notes); .gitignore reworked so the
      public repo carries code + paper sources + the 13 evidence CSVs
      (results_clean.csv + outputs_rev2_*/tables/results.csv) + Colab
      notebooks, while excluding datasets (118M), checkpoints (43M),
      submission zips, internal review artifacts
      (paper/reviewer-simulation-*.md, paper/self-review-*.md), and drafts.
      Paper updated: contribution 4 + limitations (6) now cite the repo URL;
      SUBMISSION-README synced (25 refs, 20260910 zip); recompiled
      (0 errors, 0 overfull, 19 pages) + zip rebuilt.
      Commit 5ad376d (initial, branch main, 128 files). R3/R4 simulation
      files archived to paper/ (gitignored from the public repo).
- [x] Push to GitHub (2026-09-10, via Clash proxy 127.0.0.1:7890):
      `main -> main` on origin, 128 files. Repo live at
      https://github.com/HuKejie/div-prune
- [x] Commit rewritten without the Co-Authored-By trailer
      (user request): amended to 69fa104 and force-pushed
      (`5ad376d...69fa104 main -> main`). Future commits in this repo
      carry no Claude attribution line.
- [ ] Set repo visibility to Public (user action: repo Settings ->
      General -> Danger Zone -> Change repository visibility -> Public)
- [x] Resubmit (2026-09-10: repo set to Public +
      ijprai-submission-20260910.zip uploaded to the journal system)
- [ ] Await journal decision (watch submission system / email; on a new
      review round, route to review-response)
