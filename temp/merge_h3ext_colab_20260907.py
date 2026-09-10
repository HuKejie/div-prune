"""Merge the B4/H3-extension Colab batch into results_clean.csv.

Usage (from repo root, after downloading the Drive CSV):
    uv run python temp/merge_h3ext_colab_20260907.py \
        outputs/tables/results_colab_h3ext_20260907.csv

Rules:
- Config-level dedup: skip a Colab row if (dataset, seed, config columns)
  already exists in results_clean.csv (makes the merge idempotent and lets the
  whole Drive CSV be passed in, including already-merged B1-B3 rows).
- No rows are dropped from results_clean.csv.
- Backup written before modification.
"""

import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "outputs" / "tables" / "results_clean.csv"
BACKUP = ROOT / "outputs" / "tables" / "results_clean_premerge_h3ext.csv"

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
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    colab_path = ROOT / sys.argv[1]
    if not colab_path.exists():
        raise SystemExit(f"not found: {colab_path}")

    existing = load(CLEAN)
    existing_keys = {config_key(r) for r in existing}
    columns = list(existing[0].keys())

    new_rows, skipped = [], 0
    for r in load(colab_path):
        r = {k: (v if v is not None else "") for k, v in r.items()}
        # Post-tag B5 rows written by an UNPATCHED distill.py (no lambda_schedule
        # column): pure-KD students (distill, no prune criterion) trained with a
        # controller teacher -> new-protocol lambda_schedule=controller.
        if (
            r.get("distill") == "True"
            and not r.get("prune_criterion")
            and not r.get("lambda_schedule")
        ):
            r["lambda_schedule"] = "controller"
        if config_key(r) in existing_keys:
            skipped += 1
            continue
        existing_keys.add(config_key(r))
        new_rows.append({k: r.get(k, "") for k in columns})

    if not new_rows:
        print("nothing new to merge (all rows already present)")
        return

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
