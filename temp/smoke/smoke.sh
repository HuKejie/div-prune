#!/usr/bin/env bash
# S2 local smoke: M1 (3 datasets x 5 schedules x 1 seed x 20 epochs) + M2 single config.
# Outputs isolated under outputs_smoke/ so the paper CSV is untouched.
set +e
cd /c/Users/hukej/Desktop/div-prune
mkdir -p temp/smoke
LOG=temp/smoke/smoke.log
: > "$LOG"

echo "### M1 loop start $(date +%H:%M:%S)" >> "$LOG"
for ds in cora citeseer pubmed; do
  for s in fixed controller linear kendall gradnorm; do
    echo "== $ds $s $(date +%H:%M:%S)" >> "$LOG"
    uv run python -m run.pipeline.train experiment=gat_cora experiment.dataset=$ds \
      experiment.lambda_schedule=$s experiment.regularizer=attention \
      experiment.runs=1 model.epochs=20 \
      output_dir=outputs_smoke seed=7 >> "$LOG" 2>&1
    echo "exit=$? $ds $s" >> "$LOG"
  done
done
echo "### M1 loop done $(date +%H:%M:%S)" >> "$LOG"

echo "== M2 cora controller diversity 0.75 late_last $(date +%H:%M:%S)" >> "$LOG"
uv run python -m run.pipeline.prune experiment=gat_cora experiment.dataset=cora \
  experiment.lambda_schedule=controller experiment.regularizer=attention \
  +experiment.prune='{criterion: diversity, ratio: 0.75, finetune_epochs: 10, checkpoint_source: late_last}' \
  experiment.runs=1 model.epochs=20 \
  output_dir=outputs_smoke seed=7 >> "$LOG" 2>&1
echo "exit=$? M2" >> "$LOG"
echo "### SMOKE DONE $(date +%H:%M:%S)" >> "$LOG"
