"""Confirmatory analysis: capacity probe, 4 widths x 2 arms x 10 seeds."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

CSV = Path(r"C:/Users/hukej/Desktop/Claude/div-prune/outputs_cap_full/tables/results.csv")
rows = list(csv.DictReader(CSV.open(encoding="utf-8")))

cells = defaultdict(lambda: {"baseline": {}, "controller": {}})
params = {}
for r in rows:
    hp = int(r["hidden_per_head"])
    total = 8 * hp
    arm = "controller" if r["lambda_schedule"] == "controller" else "baseline"
    cells[total][arm][int(r["seed"])] = (
        float(r["test_acc"]) * 100,
        float(r["div_last"]),
    )
    params[total] = float(r["params"])

print(f"{'width':>5} {'params':>7} {'n':>3} {'baseline':>14} {'controller':>14} "
      f"{'delta':>7} {'p':>7} {'div_b':>6} {'div_c':>6}")
print("-" * 82)

deltas = {}
for total in sorted(cells, reverse=True):
    b = cells[total]["baseline"]
    c = cells[total]["controller"]
    common = sorted(set(b) & set(c))
    if len(common) < 3:
        continue
    bv = np.array([b[s][0] for s in common])
    cv = np.array([c[s][0] for s in common])
    diff = cv - bv
    try:
        p = wilcoxon(cv, bv).pvalue
    except Exception:
        p = float("nan")
    deltas[total] = diff.mean()
    bd = np.mean([b[s][1] for s in common])
    cd = np.mean([c[s][1] for s in common])
    print(
        f"{total:>5} {params[total]:>7.0f} {len(common):>3} "
        f"{bv.mean():>7.2f}±{bv.std(ddof=1):<6.2f} "
        f"{cv.mean():>7.2f}±{cv.std(ddof=1):<6.2f} "
        f"{diff.mean():>+7.2f} {p:>7.3f} {bd:>6.3f} {cd:>6.3f}"
    )

print()
for w in (16, 8):
    if w in deltas and 64 in deltas:
        print(f"interaction delta({w}) - delta(64) = {deltas[w] - deltas[64]:+.2f} pp")

order = [w for w in (64, 32, 16, 8) if w in deltas]
print("delta by width (wide -> narrow):", [f"{w}:{deltas[w]:+.2f}" for w in order])
mono = all(deltas[order[i]] <= deltas[order[i + 1]] + 1e-9 for i in range(len(order) - 1))
print("monotone increase as capacity shrinks:", mono)
