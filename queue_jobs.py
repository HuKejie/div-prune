"""Generate job JSONs into the Drive queue folder.

Batches:
  p2_smoke : 1 job, transformer leg 2-epoch smoke (validates the Multi30k arm)
  p1_pilot : 16 jobs, capacity pilot on 4 datasets x 2 widths x 2 arms x 2 seeds
  p1_full  : 32 jobs, full 4-width x 2-arm x 10-seed sweep (stage after pilot OK)

Usage:
  python queue_jobs.py p2_smoke p1_pilot   # write into queue/
  python queue_jobs.py p1_full --stage     # write into work/staging/ only
"""

import argparse
import json
from pathlib import Path

DRIVE_ROOT = Path(r"G:/我的云端硬盘/colab-divprune")
QUEUE = DRIVE_ROOT / "queue"
STAGING = DRIVE_ROOT / "work" / "staging"

P1_DATASETS = ["citeseer", "pubmed", "computers", "photo"]
# (label, hidden_per_head); num_heads fixed at 8
P1_WIDTHS_PILOT = {"w64": 8, "w8": 1}
P1_WIDTHS_FULL = {"w64": 8, "w32": 4, "w16": 2, "w8": 1}
ARMS = {
    "base": "experiment.regularizer=none experiment.lambda_schedule=fixed",
    "ctrl": "experiment.regularizer=attention experiment.lambda_schedule=controller",
}


def job(job_id, cmd, output_dirs, results_patterns, timeout_s=1800, label=""):
    return {
        "job_id": job_id,
        "cmd": cmd,
        "cwd": "/content/div-prune",
        "output_dirs": output_dirs,
        "results_patterns": results_patterns,
        "timeout_s": timeout_s,
        "label": label,
    }


def p1_jobs(runs, widths, prefix):
    jobs = []
    for ds in P1_DATASETS:
        for wlab, hp in widths.items():
            for arm, arm_overrides in ARMS.items():
                jid = f"{prefix}-{ds}-{wlab}-{arm}"
                cmd = (
                    "python -m run.pipeline.train experiment=gat_cora "
                    f"experiment.dataset={ds} {arm_overrides} "
                    f"model.num_heads=8 model.hidden_per_head={hp} "
                    f"experiment.runs={runs} seed=0 output_dir=outputs_p1"
                )
                jobs.append(job(jid, cmd, ["outputs_p1"],
                                 ["outputs_p1/tables/results.csv"],
                                 timeout_s=3600, label=f"{prefix} {ds} {wlab} {arm}"))
    return jobs


def p2_smoke_jobs():
    cmd = (
        "pip install -q sacrebleu && python -m run.pipeline.train_transformer model=transformer "
        "experiment=multi30k experiment.runs=1 model.epochs=2 seed=0 "
        "output_dir=outputs_p2_smoke"
    )
    return [job("000-p2-smoke", cmd, ["outputs_p2_smoke"],
                ["outputs_p2_smoke/tables/results.csv"], timeout_s=2400,
                label="p2 smoke")]



def p1_firm_jobs():
    """Firm up the computers w16/w8 finding and the photo w8 variance: seeds 10-19."""
    jobs = []
    plan = [("computers", 2), ("computers", 1), ("photo", 1)]
    for ds, hp in plan:
        wlab = f"w{hp*8}"
        for arm, arm_overrides in ARMS.items():
            jid = f"030-p1firm-{ds}-{wlab}-{arm}"
            cmd = (
                "python -m run.pipeline.train experiment=gat_cora "
                f"experiment.dataset={ds} {arm_overrides} "
                f"model.num_heads=8 model.hidden_per_head={hp} "
                f"experiment.runs=10 seed=10 output_dir=outputs_p1"
            )
            jobs.append(job(jid, cmd, ["outputs_p1"],
                             ["outputs_p1/tables/results.csv"],
                             timeout_s=3600, label=f"p1 firm {ds} {wlab} {arm} s10-19"))
    return jobs


def p2_pilot_jobs():
    """Transformer capacity pilot: d_model 256 vs 16, 2 arms, 2 seeds."""
    jobs = []
    for d_model in (256, 16):
        wlab = f"d{d_model}"
        for arm, (reg, adaptive) in {
            "base": ("none", "false"),
            "ctrl": ("attention", "true"),
        }.items():
            jid = f"040-p2pilot-{wlab}-{arm}"
            cmd = (
                "python -m run.pipeline.train_transformer model=transformer "
                f"experiment=multi30k model.d_model={d_model} "
                f"experiment.regularizer={reg} experiment.adaptive_lambda={adaptive} "
                "experiment.runs=2 model.epochs=30 seed=0 "
                "output_dir=outputs_p2"
            )
            jobs.append(job(jid, cmd, ["outputs_p2"],
                             ["outputs_p2/tables/results.csv"],
                             timeout_s=3600, label=f"p2 pilot {wlab} {arm}"))
    return jobs



def p2_full_jobs():
    """Transformer capacity sweep, 30 epochs x 10 seeds, two extreme widths.

    Uses a fresh output_dir so the 2-seed pilot rows (30-epoch, old code
    without div_last) do not mix in. The unzip prefix refreshes code from the
    Drive zip in an already-running Colab session.
    """
    jobs = []
    refresh = (
        "unzip -oq /content/drive/MyDrive/colab-divprune/work/div-prune-src.zip "
        "-d /content/div-prune && "
    )
    for d_model in (256, 16):
        wlab = f"d{d_model}"
        for arm, (reg, adaptive) in {
            "base": ("none", "false"),
            "ctrl": ("attention", "true"),
        }.items():
            jid = f"050-p2full-{wlab}-{arm}"
            cmd = (
                refresh + "python -m run.pipeline.train_transformer model=transformer "
                f"experiment=multi30k model.d_model={d_model} "
                f"experiment.regularizer={reg} experiment.adaptive_lambda={adaptive} "
                "experiment.runs=10 model.epochs=30 seed=0 "
                "output_dir=outputs_p2_full"
            )
            jobs.append(job(jid, cmd, ["outputs_p2_full"],
                             ["outputs_p2_full/tables/results.csv"],
                             timeout_s=14400, label=f"p2 full {wlab} {arm} 10 seeds"))
    return jobs



def p2_mid_jobs():
    """Transformer intermediate widths d64/d32, 30 epochs x 10 seeds.

    Fills the gap between d256 (baseline similarity already below budget) and
    d16 (controller forced a large early drop, +2.68 BLEU). Same protocol as
    the existing two widths so all cells are directly comparable.
    """
    jobs = []
    refresh = (
        "unzip -oq /content/drive/MyDrive/colab-divprune/work/div-prune-src.zip "
        "-d /content/div-prune && "
    )
    for d_model in (64, 32):
        wlab = f"d{d_model}"
        for arm, (reg, adaptive) in {
            "base": ("none", "false"),
            "ctrl": ("attention", "true"),
        }.items():
            jid = f"055-p2mid-{wlab}-{arm}"
            cmd = (
                refresh + "python -m run.pipeline.train_transformer model=transformer "
                f"experiment=multi30k model.d_model={d_model} "
                f"experiment.regularizer={reg} experiment.adaptive_lambda={adaptive} "
                "experiment.runs=10 model.epochs=30 seed=0 "
                "output_dir=outputs_p2_full"
            )
            jobs.append(job(jid, cmd, ["outputs_p2_full"],
                             ["outputs_p2_full/tables/results.csv"],
                             timeout_s=14400, label=f"p2 mid {wlab} {arm} 10 seeds"))
    return jobs


BATCHES = {
    "p2_smoke": p2_smoke_jobs,
    "p1_pilot": lambda: p1_jobs(2, P1_WIDTHS_PILOT, "010-p1pilot"),
    "p1_full": lambda: p1_jobs(10, P1_WIDTHS_FULL, "020-p1full"),
    "p1_firm": p1_firm_jobs,
    "p2_pilot": p2_pilot_jobs,
    "p2_full": p2_full_jobs,
    "p2_mid": p2_mid_jobs,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batches", nargs="+", choices=sorted(BATCHES))
    ap.add_argument("--stage", action="store_true",
                    help="write into work/staging/ instead of queue/")
    args = ap.parse_args()

    target = STAGING if args.stage else QUEUE
    target.mkdir(parents=True, exist_ok=True)
    for name in args.batches:
        for j in BATCHES[name]():
            path = target / f"{j['job_id']}.json"
            path.write_text(json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8")
            print("wrote", path)
    print(f"\ntarget dir: {target}")


if __name__ == "__main__":
    main()
