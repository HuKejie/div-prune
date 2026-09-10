"""Merge second-revision local batches (2026-09-06, outputs_rev2_*) into
results_clean.csv.

Sources (12 CSVs, 179 rows total):
- outputs_rev2_e4_e0xx_t0xx  (8 x 10 rows): controller sensitivity grid on Cora
  (eta x target combos; the 9th cell eta=0.05/t=0.7 is the old-protocol main
  controller arm already in results_clean).
- outputs_rev2_pubreach_eta020 (9 rows): Pubmed eta=0.2/t=0.7 (seed 9 missing).
- outputs_rev2_sweep (70 rows): fixed-lambda re-run sweep on Cora, 10 seeds
  (old submission rows had n=5 for lambda<=1; numbers agree, both kept).
- outputs_rev2_e1_l5_{citeseer,pubmed} (2 x 10 rows): fixed lambda=5 runs for
  the R1 comparison prose.

Rules:
- Config-level dedup: skip a new row if (dataset, seed, config columns) already
  exists in results_clean.csv.
- No rows are dropped from results_clean.csv.
- Backup written before modification.

Run from repo root: uv run python temp/merge_outputs_rev2_20260907.py
"""

import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "outputs" / "tables" / "results_clean.csv"
BACKUP = ROOT / "outputs" / "tables" / "results_clean_premerge_20260907.csv"

SOURCES = [
    "outputs_rev2_e4_e001_t050",
    "outputs_rev2_e4_e001_t070",
    "outputs_rev2_e4_e001_t090",
    "outputs_rev2_e4_e005_t050",
    "outputs_rev2_e4_e005_t090",
    "outputs_rev2_e4_e020_t050",
    "outputs_rev2_e4_e020_t070",
    "outputs_rev2_e4_e020_t090",
    "outputs_rev2_pubreach_eta020",
    "outputs_rev2_sweep",
    "outputs_rev2_e1_l5_citeseer",
    "outputs_rev2_e1_l5_pubmed",
]

CONFIG_COLS = [
    "dataset", "seed", "hidden_per_head", "regularizer", "lambda_fixed",
    "lambda_target", "lambda_eta", "adaptive_lambda", "lambda_schedule",
    "prune_criterion", "prune_ratio", "checkpoint_source", "distill",
    "distill_alpha", "distill_temperature",
]


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def config_key(r: dict) -> tuple:
    return tuple(r.get(c, "") or "" for c in CONFIG_COLS)


def main() -> None:
    existing = load(CLEAN)
    existing_keys = {config_key(r) for r in existing}
    columns = list(existing[0].keys())

    new_rows, skipped = [], 0
    for src in SOURCES:
        rows = load(ROOT / src / "tables" / "results.csv")
        for r in rows:
            # Drop the empty "criterion_div" style padding: normalize None->""
            r = {k: (v if v is not None else "") for k, v in r.items()}
            # Align to the clean schema (sources share the same 28 columns).
            if config_key(r) in existing_keys:
                skipped += 1
                continue
            existing_keys.add(config_key(r))
            new_rows.append({k: r.get(k, "") for k in columns})

    shutil.copyfile(CLEAN, BACKUP)
    with CLEAN.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(existing + new_rows)

    print(f"backup: {BACKUP.name} ({len(existing)} rows)")
    print(f"new rows merged: {len(new_rows)}, config-dupes skipped: {skipped}")
    print(f"results_clean.csv now: {len(existing) + len(new_rows)} rows")


if __name__ == "__main__":
    main()
