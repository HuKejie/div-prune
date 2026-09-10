"""Build the revised submission zip (ijprai-submission-20260910.zip).

Contents: main.tex + journal class/style files + refs.bib + all figures and
tables referenced by main.tex (verified against the compiled PDF).
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
ZIP = ROOT / "ijprai-submission-20260910.zip"

FILES = [
    "main.tex",
    "ws-ijprai.cls",
    "ws-ijprai.bst",
    "refs.bib",
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

missing = [f for f in FILES if not (PAPER / f).exists()]
if missing:
    raise SystemExit(f"missing files: {missing}")

with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(PAPER / f, f)
print(f"wrote {ZIP.name}: {len(FILES)} files")
