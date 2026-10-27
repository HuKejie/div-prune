"""Generate the capacity-boundary table (tab9) from the probe results.

Reads outputs_cap_full/tables/results.csv (4 total widths x 2 arms x 10 seeds)
and writes paper/tables/tab9_capacity.tex.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "outputs_cap_full/tables/results.csv"
DST = REPO / "paper/tables/tab9_capacity.tex"

cells = defaultdict(lambda: {"baseline": {}, "controller": {}})
params = {}
heads = {}
for r in csv.DictReader(SRC.open(encoding="utf-8")):
    hp = int(r["hidden_per_head"])
    nh = int(r["num_heads"]) if "num_heads" in r else 8
    total = nh * hp
    arm = "controller" if r["lambda_schedule"] == "controller" else "baseline"
    cells[total][arm][int(r["seed"])] = (float(r["test_acc"]) * 100, float(r["div_last"]))
    params[total] = int(float(r["params"]))
    heads[total] = (nh, hp)

rows = []
for total in sorted(cells, reverse=True):
    b = cells[total]["baseline"]
    c = cells[total]["controller"]
    common = sorted(set(b) & set(c))
    bv = np.array([b[s][0] for s in common])
    cv = np.array([c[s][0] for s in common])
    diff = cv - bv
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    lo, hi = diff.mean() - 1.96 * se, diff.mean() + 1.96 * se
    p = wilcoxon(cv, bv).pvalue
    nh, hp = heads[total]
    rows.append(
        {
            "total": total,
            "cfg": f"{nh} $\\times$ {hp}",
            "params": f"{params[total]:,}",
            "b": f"{bv.mean():.2f} $\\pm$ {bv.std(ddof=1):.2f}",
            "c": f"{cv.mean():.2f} $\\pm$ {cv.std(ddof=1):.2f}",
            "d": f"{diff.mean():+.2f}",
            "ci": f"$[{lo:+.2f}, {hi:+.2f}]$",
            "p": f"{p:.3f}",
            "db": f"{np.mean([b[s][1] for s in common]):.3f}",
            "dc": f"{np.mean([c[s][1] for s in common]):.3f}",
        }
    )

lines = [
    "% Table (number assigned by position): capacity boundary of the controller.",
    "% Generated from outputs_cap_full/tables/results.csv (10 paired seeds per",
    "% cell) by make_table_capacity.py. Cora; 8 heads throughout; total hidden",
    "% width = num_heads x hidden_per_head is varied.",
    "\\begin{table}[ht]",
    "\\tbl{The controller's effect depends on model capacity (Cora, 10 paired "
    "seeds, 2-layer GAT, 8 heads throughout; total hidden width is varied while "
    "the head count is held fixed). $\\Delta$ is controller minus baseline test "
    "accuracy in percentage points, with the 95\\% CI of the paired difference "
    "and a two-sided Wilcoxon signed-rank $p$-value. The last two columns give "
    "inter-head similarity at the final epoch. The controller is indistinguishable "
    "from baseline when the model is over-parameterised and significantly harmful "
    "at the narrowest width.",
    "\\label{tab:capacity}}",
    "{\\setlength{\\tabcolsep}{3pt}\\scriptsize"
    "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llccccccc}",
    "\\toprule",
    "Total width & Heads $\\times$ dim & Params & Baseline & Controller "
    "& $\\Delta$ (pp) & 95\\% CI & $p$ & $d_{\\mathrm{last}}$ base $\\to$ ctrl \\\\",
    "\\colrule",
]
for r in rows:
    lines.append(
        f"{r['total']} & {r['cfg']} & {r['params']} & {r['b']} & {r['c']} "
        f"& {r['d']} & {r['ci']} & {r['p']} & {r['db']} $\\to$ {r['dc']} \\\\"
    )
lines += [
    "\\botrule",
    "\\end{tabular*}}",
    "\\end{table}",
    "",
]

DST.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {DST}")
for r in rows:
    print(f"  width {r['total']:>3}: delta {r['d']} pp  CI {r['ci']}  p={r['p']}")
