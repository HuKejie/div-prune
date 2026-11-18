"""Analyze the P1 capacity-boundary cross-dataset sweep.

Reads the Drive mirror of outputs_p1/tables/results.csv, pairs controller vs
baseline by seed within each (dataset, width) cell, and reports mean diff in
percentage points, a 95% CI on the paired diffs (normal approximation), and a
two-sided Wilcoxon signed-rank p-value.

Usage:
  python analyze_p1.py [csv_path]
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

DEFAULT_CSV = Path(r"G:/我的云端硬盘/colab-divprune/work/outputs_mirror/outputs_p1/tables/results.csv")


def load_cells(path: Path):
    cells = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            arm = "ctrl" if r["lambda_schedule"] == "controller" else "base"
            cells[(r["dataset"], int(r["hidden_per_head"]))][arm][int(r["seed"])] = r
    return cells


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    cells = load_cells(path)
    print(f"source: {path}")
    print(f"{'dataset':<11} {'w':>3} {'n':>3} | {'base':>6} {'ctrl':>6} {'diff':>7} "
          f"{'95% CI':>16} {'p(wilx)':>8} {'div b→c':>15}")
    print("-" * 84)
    for (ds, hp) in sorted(cells.keys(), key=lambda k: (k[0], -k[1])):
        c = cells[(ds, hp)]
        b, t = c.get("base", {}), c.get("ctrl", {})
        seeds = sorted(set(b) & set(t))
        if not seeds:
            print(f"{ds:<11} {hp*8:>3}   0 | MISSING ARM")
            continue
        base = np.array([float(b[s]["test_acc"]) for s in seeds]) * 100
        ctrl = np.array([float(t[s]["test_acc"]) for s in seeds]) * 100
        diff = ctrl - base
        mean = diff.mean()
        half = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
        if np.allclose(diff, 0):
            p = 1.0
        else:
            p = stats.wilcoxon(ctrl, b_acc := base, zero_method="wilcox").pvalue
        db = np.nanmean([float(b[s]["div_last"]) for s in seeds])
        dt = np.nanmean([float(t[s]["div_last"]) for s in seeds])
        flag = " *" if p < 0.05 else ""
        print(f"{ds:<11} {hp*8:>3} {len(seeds):>3} | {base.mean():>6.2f} {ctrl.mean():>6.2f} "
              f"{mean:>+7.2f} [{mean-half:>+6.2f},{mean+half:>+6.2f}] {p:>8.3f}{flag} "
              f"{db:>6.3f}→{dt:<6.3f}")


if __name__ == "__main__":
    main()
