"""Append the B4 + H3-extension (Citeseer/Pubmed) batch cells to colab_divprune.ipynb.

Batch plan (160 runs):
- B4: from-scratch GAT, 4/2 heads, no KD, no regularizer, 100 epochs,
  3 datasets x 2 ratios x 10 seeds (60 runs).
- B5: pure-KD student from scratch, Citeseer/Pubmed only (Cora already in the
  paper CSV), controller teacher trained in-memory, 100 epochs (40 runs).
- B6: diversity prune + FT 100 epochs, Citeseer/Pubmed, ratio 0.5 only —
  ratio 0.75 is already covered by the B3 `best` rows (20 runs).
- B8: diversity prune + KD 100 epochs, Citeseer/Pubmed, teacher = archived B2
  checkpoint (best source) via reuse_stem (40 runs).

Requires syncing the patched run/pipeline/train.py and run/pipeline/distill.py
to the Drive copy (see the markdown cell).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "colab_divprune.ipynb"


def cell(cell_type, source):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").split("\n")],
    }


MD = cell("markdown", """\
# Second revision (2026-09-07): B4 + H3 extension (Citeseer/Pubmed)

**Before running the cells below**, sync these two patched files from the
local repo to this Drive copy:

- `run/pipeline/train.py` — tolerates a ratio-only `experiment.prune` cfg
  (B4 from-scratch small models label the head budget without pruning).
- `run/pipeline/distill.py` — B5 rows now write `lambda_schedule` /
  `checkpoint_source` / `criterion_div` (new-protocol fields).

Batch plan (160 runs; each cell is resumable via per-seed skip on
`outputs/tables/results.csv`):
- B4: from-scratch GAT, 4/2 heads, no KD, no regularizer, 100 epochs,
  3 datasets x 2 ratios x 10 seeds.
- B5: pure-KD student from scratch (Citeseer/Pubmed only; Cora is already in
  the paper CSV), controller teacher trained in-memory, 100 epochs.
- B6: diversity prune + FT 100 epochs (Citeseer/Pubmed, ratio 0.5 only;
  ratio 0.75 is already covered by the B3 `best` rows).
- B8: diversity prune + KD 100 epochs (Citeseer/Pubmed), teacher = archived
  B2 checkpoint (`best` source) via `reuse_stem`.
""")

B4 = cell("code", """\
# === B4: from-scratch small models (4/2 heads, no KD), 10 seeds x 3 datasets ===
# num_heads = 8*(1-ratio): ratio 0.5 -> 4 heads, 0.75 -> 2 heads (matches the
# pruned-student structure). regularizer=none, distill=false, 100 epochs.
# Per-seed skip via outputs/tables/results.csv makes this cell resumable.
import os
import subprocess
from datetime import datetime

import pandas as pd

def run(cmd):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=False).returncode

def plog(msg):
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with open("progress_log.md", "a", encoding="utf-8") as f:
        f.write(line + "\\n")
    print(f"[log] {line}", flush=True)

csv_path = "outputs/tables/results.csv"

def have_run(ds, ratio, seed):
    if not os.path.exists(csv_path):
        return False
    have = pd.read_csv(csv_path)
    m = (
        (have.dataset == ds)
        & (have.regularizer == "none")
        & (have.prune_ratio == ratio)
        & (have.distill == False)
        & (have.seed == seed)
    )
    return m.any()

plog("START B4")
for ds in ["cora", "citeseer", "pubmed"]:
    for ratio, heads in [(0.5, 4), (0.75, 2)]:
        for seed in range(10):
            if have_run(ds, ratio, seed):
                continue
            rc = run(
                f"python -m run.pipeline.train experiment=gat_cora "
                f"experiment.dataset={ds} experiment.regularizer=none "
                f"experiment.lambda_schedule=fixed experiment.runs=1 seed={seed} "
                f"model.num_heads={heads} model.epochs=100 "
                f"+experiment.prune.ratio={ratio}"
            )
            plog(f"done rc={rc} B4 {ds} ratio{ratio} seed{seed}")
plog("END B4")
""")

B5 = cell("code", """\
# === B5: pure-KD students from scratch, Citeseer + Pubmed, 10 seeds ===
# Teacher = adaptive dense controller trained in-memory (matches the Cora B5
# protocol in the paper). 100 epochs. Rows append to outputs/tables/results.csv.
import os
import subprocess
from datetime import datetime

import pandas as pd

def run(cmd):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=False).returncode

def plog(msg):
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with open("progress_log.md", "a", encoding="utf-8") as f:
        f.write(line + "\\n")
    print(f"[log] {line}", flush=True)

csv_path = "outputs/tables/results.csv"

def have_run(ds, ratio, seed):
    if not os.path.exists(csv_path):
        return False
    have = pd.read_csv(csv_path)
    m = (
        (have.dataset == ds)
        & (have.distill == True)
        & (have.prune_criterion.isna())
        & (have.prune_ratio == ratio)
        & (have.seed == seed)
    )
    return m.any()

plog("START B5-h3ext")
for ds in ["citeseer", "pubmed"]:
    for ratio in [0.5, 0.75]:
        for seed in range(10):
            if have_run(ds, ratio, seed):
                continue
            rc = run(
                f"python -m run.pipeline.distill experiment=gat_cora "
                f"experiment.dataset={ds} experiment.lambda_schedule=controller "
                f"experiment.regularizer=attention experiment.runs=1 seed={seed} "
                f"+experiment.prune.ratio={ratio} +experiment.prune.finetune_epochs=100"
            )
            plog(f"done rc={rc} B5 {ds} ratio{ratio} seed{seed}")
plog("END B5-h3ext")
""")

B6B8 = cell("code", """\
# === B6 + B8: diversity prune + FT / + KD, Citeseer + Pubmed, 10 seeds ===
# Teacher: archived B2 controller checkpoint (best source) via reuse_stem, so no
# dense retraining. 100 FT epochs. Ratio 0.75 B6 rows are already covered by the
# B3 `best` rows (same protocol) and get skipped by have_run.
import os
import subprocess
from datetime import datetime

import pandas as pd

def run(cmd):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=False).returncode

def plog(msg):
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with open("progress_log.md", "a", encoding="utf-8") as f:
        f.write(line + "\\n")
    print(f"[log] {line}", flush=True)

csv_path = "outputs/tables/results.csv"

def have_run(ds, ratio, kd, seed):
    if not os.path.exists(csv_path):
        return False
    have = pd.read_csv(csv_path)
    m = (
        (have.dataset == ds)
        & (have.prune_criterion == "diversity")
        & (have.prune_ratio == ratio)
        & (have.checkpoint_source == "best")
        & (have.distill == kd)
        & (have.seed == seed)
    )
    return m.any()

plog("START B6B8-h3ext")
for kd, label in [(False, "B6"), (True, "B8")]:
    for ds in ["citeseer", "pubmed"]:
        for ratio in [0.5, 0.75]:
            for seed in range(10):
                if have_run(ds, ratio, kd, seed):
                    continue
                prune_cfg = (
                    "{criterion: diversity, ratio: " + str(ratio) + ", finetune_epochs: 100, "
                    "checkpoint_source: best, "
                    "reuse_stem: '" + ds + "_seed{seed}_schedcontroller', "
                    "teacher_dir: outputs_ckpt/checkpoints}"
                )
                kd_flag = " experiment.distill=true" if kd else ""
                rc = run(
                    f"python -m run.pipeline.prune experiment=gat_cora "
                    f"experiment.dataset={ds} experiment.lambda_schedule=controller "
                    f"experiment.regularizer=attention experiment.runs=1 seed={seed}"
                    f"{kd_flag} +experiment.prune=\\"{prune_cfg}\\""
                )
                plog(f"done rc={rc} {label} {ds} ratio{ratio} seed{seed}")
plog("END B6B8-h3ext")
""")

COVERAGE = cell("code", """\
# Coverage check: B4/B5/B6/B8 for the H3 extension (tab8_h3ext), 10 seeds each
import pandas as pd

df = pd.read_csv("outputs/tables/results.csv")
missing = []
for ds in ["cora", "citeseer", "pubmed"]:
    for ratio in [0.5, 0.75]:
        b4 = df[(df.dataset == ds) & (df.prune_ratio == ratio) &
                (df.regularizer == "none") & (df.distill == False)]
        b5 = df[(df.dataset == ds) & (df.prune_ratio == ratio) &
                (df.distill == True) & (df.prune_criterion.isna())]
        b6 = df[(df.dataset == ds) & (df.prune_ratio == ratio) &
                (df.distill == False) & (df.prune_criterion == "diversity")]
        b8 = df[(df.dataset == ds) & (df.prune_ratio == ratio) &
                (df.distill == True) & (df.prune_criterion == "diversity")]
        for label, sub in [("B4", b4), ("B5", b5), ("B6", b6), ("B8", b8)]:
            n_seeds = sub.seed.nunique()
            print(f"{ds} r{ratio} {label}: {n_seeds}/10")
            if n_seeds < 10:
                missing.append((ds, ratio, label))
print("MISSING:", missing if missing else "none - batch complete")
""")


def main() -> None:
    with NB.open(encoding="utf-8") as f:
        nb = json.load(f)
    nb["cells"].extend([MD, B4, B5, B6B8, COVERAGE])
    with NB.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"appended {len([MD, B4, B5, B6B8, COVERAGE])} cells -> {len(nb['cells'])} total")


if __name__ == "__main__":
    main()
