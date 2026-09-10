# IJPRAI Submission Package

Manuscript: `main.tex` — **official `ws-ijprai` class applied (2026-08-22)**,
converted from `draft-v2-full.md`.

## Files to upload (Overleaf recommended) — revised 2026-09-10

| File | Role |
|---|---|
| `main.tex` | Manuscript (ws-ijprai class), revision R1/R2/R3 |
| `ws-ijprai.cls` | Official journal class (required) |
| `ws-ijprai.bst` | Official bibliography style (required) |
| `refs.bib` | 25 verified references (arXiv API / CrossRef) |
| `figures/fig1_h1_comparison.pdf` | H1 main comparison (Cora λ=20 = revised sweep) |
| `figures/fig2_trace_cora_seed3.pdf` | Controller dynamics (Cora) |
| `figures/fig3_h3_distillation.pdf` | KD comparison |
| `figures/fig4_adaptive_baselines.pdf` | Adaptive baselines (R1) |
| `tables/tab1_fixed_lambda_h1.tex` | Table 1: fixed-λ sweep (n=10 recorded) + main comparison |
| `tables/tab2_h2_criteria.tex` | Table 2: pruning criteria matrix |
| `tables/tab3_h3_distill.tex` | Table 3: KD study (Cora) |
| `tables/tab4_adaptive_baselines.tex` | Table 4: adaptive baselines (R1) |
| `tables/tab5_revival.tex` | Table 5: revival test (R2) |
| `tables/tab6_revision_cis.tex` | Table 6: main 95% CIs (revision) |
| `tables/tab7_sensitivity.tex` | Table 7: controller sensitivity grid (η×d*, d_best semantics) |
| `tables/tab8_h3ext.tex` | Table 8: H3 extension to Citeseer/Pubmed |

The zip `ijprai-submission-20260910.zip` (repo root) contains exactly these
files. `fig2_trace_citeseer_seed0.pdf` was removed in the revision (no longer
referenced). The code repository is public at
https://github.com/HuKejie/div-prune (cited in the manuscript, contributions
and limitations).

Not needed for upload: `ws-ijprai.tex`, `ws-ijprai_bib.tex`, `ws-ijprai.pdf`,
`ws-ijprai_bib.pdf`, `ijpraif1.*`, `ws-ijprai/` (official samples), `*.md`
drafts, `*.py` utilities, PNG previews.

## TODO before submission

- [ ] Real author names, affiliations, email (placeholders in `main.tex`)
- [ ] Funding info or remove the Acknowledgments section (`main.tex`)
- [ ] History block dates (`\received/\revised/\accepted/\published` — fill
      only the received date at submission if required)
- [ ] Compile on Overleaf (pdflatex + bibtex; class expects bibtex, not biblatex)
- [ ] Final `paper-self-review` pass after any content edits
