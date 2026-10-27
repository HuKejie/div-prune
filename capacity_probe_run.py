"""Capacity probe, full: 4 widths x 2 arms x 10 seeds.

The 2-seed pilot showed a monotone interaction (+1.05 pp from width 64 to 16),
but with 2 seeds the error bar is roughly the size of the effect. This run is
the confirmatory version, extended to width 8.
"""

import subprocess
from pathlib import Path

REPO = Path(r"C:/Users/hukej/Desktop/Claude/div-prune")
PY = REPO / ".venv/Scripts/python.exe"
OUT = "outputs_cap_full"

# (num_heads, hidden_per_head) -> total width 64 / 32 / 16 / 8
CONFIGS = [(8, 8), (8, 4), (8, 2), (8, 1)]
ARMS = [
    ("baseline", "none", "fixed"),
    ("controller", "attention", "controller"),
]

for num_heads, hp in CONFIGS:
    for label, reg, sched in ARMS:
        cmd = (
            f'"{PY}" -m run.pipeline.train experiment=gat_cora '
            f"experiment.regularizer={reg} experiment.lambda_schedule={sched} "
            f"model.num_heads={num_heads} model.hidden_per_head={hp} "
            f"experiment.runs=10 seed=0 output_dir={OUT}"
        )
        print(f"$ total={num_heads * hp} {label}", flush=True)
        rc = subprocess.run(cmd, shell=True, check=False, cwd=str(REPO)).returncode
        print(f"  rc={rc}", flush=True)

print("DONE")
