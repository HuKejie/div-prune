"""Merge Colab B1/B3 rows (2026-09-05) into results_clean.csv.

Rules applied:
- Drop 2 smoke rows (epochs_run=5.0 / regularizer=none & schedule=fixed).
- Drop 40 old cora rows: attention+adaptive (controller) teacher, ratio 0.75,
  non-KD, 4 criteria x 10 seeds — semantic duplicates of new B3 `best` source
  rows (schema rename: old adaptive_lambda=True == new lambda_schedule=controller).
  New rows are the authoritative M2 evidence (explicit checkpoint_source,
  post-bugfix trainer).
- Keep everything else untouched; old rows are padded with '' for the 3 new
  columns (lambda_schedule, checkpoint_source, criterion_div).

Run from repo root: uv run python temp/merge_colab_20260905.py
"""

import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "outputs" / "tables" / "results_clean.csv"
COLAB = ROOT / "outputs" / "tables" / "results_colab_20260905.csv"
BACKUP = ROOT / "outputs" / "tables" / "results_clean_premerge_20260820.csv"

CRITERIA = ("diversity", "magnitude", "gradient", "random")


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_smoke(r: dict) -> bool:
    return r["epochs_run"] == "5.0" or (
        r["regularizer"] == "none" and r["lambda_schedule"] == "fixed"
    )


def collides_with_b3_best(r: dict) -> bool:
    """Old cora row that is a semantic duplicate of a new B3 `best`-source row."""
    return (
        r["dataset"] == "cora"
        and r["regularizer"] == "attention"
        and r["adaptive_lambda"] == "True"
        and r["prune_criterion"] in CRITERIA
        and r["prune_ratio"] == "0.75"
        and r["distill"] == "False"
        and r["lambda_target"] == "0.7"
        and r["lambda_eta"] == "0.05"
    )


def main() -> None:
    old = load(CLEAN)
    new = load(COLAB)
    new_columns = list(new[0].keys())
    old_columns = set(old[0].keys())

    shutil.copyfile(CLEAN, BACKUP)

    dropped_smoke = [r for r in new if is_smoke(r)]
    kept_new = [r for r in new if not is_smoke(r)]
    dropped_old = [r for r in old if collides_with_b3_best(r)]
    kept_old = [r for r in old if not collides_with_b3_best(r)]

    # Pad old rows to the new schema (old lacks lambda_schedule/checkpoint_source/criterion_div).
    merged_rows = []
    for r in kept_old:
        merged_rows.append({k: (r.get(k, "") if k in old_columns else "") for k in new_columns})
    merged_rows.extend(kept_new)

    with CLEAN.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_columns)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"backup: {BACKUP.name} ({len(old)} rows)")
    print(f"dropped smoke rows: {len(dropped_smoke)}")
    print(f"dropped old semantic-duplicate rows: {len(dropped_old)}")
    print(f"kept old rows: {len(kept_old)}, new protocol rows: {len(kept_new)}")
    print(f"results_clean.csv now: {len(merged_rows)} rows, {len(new_columns)} columns")


if __name__ == "__main__":
    main()
