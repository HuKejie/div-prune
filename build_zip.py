# -*- coding: utf-8 -*-
"""Rebuild the IJPRAI submission zip with the canonical 16-file layout."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = ROOT / "paper"
FILES = [
    "main.tex", "ws-ijprai.cls", "ws-ijprai.bst", "refs.bib",
    "figures/fig1_h1_comparison.pdf",
    "figures/fig2_trace_cora_seed3.pdf",
    "figures/fig3_h3_distillation.pdf",
    "figures/fig4_adaptive_baselines.pdf",
    "tables/tab1_fixed_lambda_h1.tex",
    "tables/tab2_h2_criteria.tex",
    "tables/tab3_h3_distill.tex",
    "tables/tab4_adaptive_baselines.tex",
    "tables/tab5_revival.tex",
    "tables/tab6_revision_cis.tex",
    "tables/tab7_sensitivity.tex",
    "tables/tab8_h3ext.tex",
]
OUT = ROOT / "ijprai-submission-20260914.zip"
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(P / f, f)
print(f"wrote {OUT.name}: {len(FILES)} files")
