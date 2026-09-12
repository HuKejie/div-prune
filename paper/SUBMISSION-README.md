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

The zip `ijprai-submission-20260912.zip` (repo root) contains exactly these
files (2026-09-12 layout overhaul: full-width tables, uncluttered Fig. 3/4,
ordered appendices). `fig2_trace_citeseer_seed0.pdf` was removed in the revision (no longer
referenced). The code repository is public at
https://github.com/HuKejie/div-prune (cited in the manuscript, contributions
and limitations).

Not needed for upload: `ws-ijprai.tex`, `ws-ijprai_bib.tex`, `ws-ijprai.pdf`,
`ws-ijprai_bib.pdf`, `ijpraif1.*`, `ws-ijprai/` (official samples), `*.md`
drafts, `*.py` utilities, PNG previews.

## TODO before submission

- [x] Author names, affiliations, email (2026-09-10: Kejie Hu, City
      University of Wu Han, xshu1999@gmail.com; co-authors pending)
- [x] Funding: none — Acknowledgments section removed (2026-09-10)
- [x] History block: `\received{(10 September 2026)}` only; the journal
      fills the rest (2026-09-10)
- [x] Local compile verified (2026-09-12, TeXLive 2026 at
      `C:\Users\hukej\texlive\2026\bin\windows`: pdflatex + bibtex + 2×pdflatex,
      0 errors, 0 overfull, 20 pages)
- [x] Final verification pass (2026-09-12: all 8 polish items + self-review
      P0/P1/P2 confirmed applied; 25 cites ↔ 25 bib entries; check_main_tex.py
      passed; stats-appendix.md regenerated and consistent with the paper)
- [x] Layout overhaul (2026-09-12): all tables full text width; KD-table
      statistics and revision-CI cells stacked to stay inside the 30pc text
      block; appendices anchored with \clearpage so titles are never orphaned;
      Fig. 3/4 significance labels cleared of value labels; Fig. 4 now plots
      all three B5/B6/B8 bars at both ratios and matches Table 3 exactly
- [ ] Compile on Overleaf at upload time (pdflatex + bibtex; class expects
      bibtex, not biblatex)
