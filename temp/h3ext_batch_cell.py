# === B4 + B5 + B6/B8: H3-extension batch (self-contained, resumable) ===
# 160 runs. If the session drops, just rerun this cell; finished combos are
# skipped via (dataset, ratio, seed) checks against outputs/tables/results.csv.
# Works with both patched and unpatched train.py/distill.py.
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
        f.write(line + "\n")
    print(f"[log] {line}", flush=True)

CSV = "outputs/tables/results.csv"

def have():
    return pd.read_csv(CSV) if os.path.exists(CSV) else None

def have_b4(ds, ratio, seed):
    h = have()
    if h is None: return False
    return ((h.dataset == ds) & (h.regularizer == "none") &
            (h.prune_ratio == ratio) & (h.distill == False) &
            (h.seed == seed)).any()

def have_b5(ds, ratio, seed):
    h = have()
    if h is None: return False
    return ((h.dataset == ds) & (h.distill == True) &
            (h.prune_criterion.isna()) & (h.prune_ratio == ratio) &
            (h.seed == seed)).any()

def have_b6b8(ds, ratio, kd, seed):
    h = have()
    if h is None: return False
    return ((h.dataset == ds) & (h.prune_criterion == "diversity") &
            (h.prune_ratio == ratio) & (h.checkpoint_source == "best") &
            (h.distill == kd) & (h.seed == seed)).any()

plog("START h3ext")
# --- B4: from-scratch small models (4/2 heads, no KD), 60 runs ---
for ds in ["cora", "citeseer", "pubmed"]:
    for ratio, heads in [(0.5, 4), (0.75, 2)]:
        for seed in range(10):
            if have_b4(ds, ratio, seed):
                continue
            prune_cfg = "{criterion: , ratio: " + str(ratio) + "}"
            rc = run(
                f"python -m run.pipeline.train experiment=gat_cora "
                f"experiment.dataset={ds} experiment.regularizer=none "
                f"experiment.lambda_schedule=fixed experiment.runs=1 seed={seed} "
                f"model.num_heads={heads} model.epochs=100 "
                f'+experiment.prune="{prune_cfg}"'
            )
            plog(f"done rc={rc} B4 {ds} ratio{ratio} seed{seed}")

# --- B5: pure-KD students from scratch, Citeseer + Pubmed, 40 runs ---
for ds in ["citeseer", "pubmed"]:
    for ratio in [0.5, 0.75]:
        for seed in range(10):
            if have_b5(ds, ratio, seed):
                continue
            rc = run(
                f"python -m run.pipeline.distill experiment=gat_cora "
                f"experiment.dataset={ds} experiment.lambda_schedule=controller "
                f"experiment.regularizer=attention experiment.runs=1 seed={seed} "
                f"+experiment.prune.ratio={ratio} +experiment.prune.finetune_epochs=100"
            )
            plog(f"done rc={rc} B5 {ds} ratio{ratio} seed{seed}")

# --- B6 + B8: diversity prune + FT / + KD, Citeseer + Pubmed, 60 runs ---
# Teacher = archived B2 controller checkpoint (best source) via reuse_stem.
for ds in ["citeseer", "pubmed"]:
    for seed in range(10):
        stem = f"{ds}_seed{seed}_schedcontroller_best.pt"
        if not os.path.exists(f"outputs_ckpt/checkpoints/{stem}"):
            print(f"WARNING: missing B2 teacher checkpoint {stem} (run the B2 cell of the old notebook first)")
for kd, label in [(False, "B6"), (True, "B8")]:
    for ds in ["citeseer", "pubmed"]:
        for ratio in [0.5, 0.75]:
            for seed in range(10):
                if have_b6b8(ds, ratio, kd, seed):
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
                    f'{kd_flag} +experiment.prune="{prune_cfg}"'
                )
                plog(f"done rc={rc} {label} {ds} ratio{ratio} seed{seed}")
plog("END h3ext")

# --- coverage check (expect: none - batch complete) ---
df = have()
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
