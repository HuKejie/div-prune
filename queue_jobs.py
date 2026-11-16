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
        "python -m run.pipeline.train_transformer model=transformer "
        "experiment=multi30k experiment.runs=1 model.epochs=2 seed=0 "
        "output_dir=outputs_p2_smoke"
    )
    return [job("000-p2-smoke", cmd, ["outputs_p2_smoke"],
                ["outputs_p2_smoke/tables/results.csv"], timeout_s=2400,
                label="p2 smoke")]


BATCHES = {
    "p2_smoke": p2_smoke_jobs,
    "p1_pilot": lambda: p1_jobs(2, P1_WIDTHS_PILOT, "010-p1pilot"),
    "p1_full": lambda: p1_jobs(10, P1_WIDTHS_FULL, "020-p1full"),
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
