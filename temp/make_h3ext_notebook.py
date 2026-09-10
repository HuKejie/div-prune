"""Build colab_divprune_h3ext.ipynb: setup cells (copied from the existing
notebook) + the self-contained B4/B5/B6-B8 batch cell."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "colab_divprune.ipynb"
NEW = ROOT / "colab_divprune_h3ext.ipynb"

with OLD.open(encoding="utf-8") as f:
    old = json.load(f)

# Setup cells from the old notebook: GPU check, deps, mount+chdir+install.
setup_cells = [old["cells"][i] for i in (1, 2, 3)]

with (ROOT / "temp" / "h3ext_batch_cell.py").open(encoding="utf-8") as f:
    batch_source = [line + "\n" for line in f.read().split("\n")]

md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# div-prune H3-extension batch (2026-09-07)\n",
        "\n",
        "Setup + one self-contained batch cell for the H3 extension:\n",
        "\n",
        "- B4: from-scratch GAT, 4/2 heads, no KD (3 datasets x 2 ratios x 10 seeds)\n",
        "- B5: pure-KD students from scratch (Citeseer/Pubmed)\n",
        "- B6: diversity prune + FT (Citeseer/Pubmed; ratio 0.75 skipped if B3 rows exist)\n",
        "- B8: diversity prune + KD (Citeseer/Pubmed; teacher = archived B2 best checkpoint)\n",
        "\n",
        "Run cells 1-3 (GPU, deps, mount+install), then the batch cell. Resumable:\n",
        "rerunning the batch cell skips finished (dataset, ratio, seed) combos.\n",
    ],
}

batch_cell = {
    "cell_type": "code",
    "metadata": {},
    "source": batch_source,
}

nb = {
    "cells": [md, *setup_cells, batch_cell],
    "metadata": old.get("metadata", {}),
    "nbformat": old.get("nbformat", 4),
    "nbformat_minor": old.get("nbformat_minor", 5),
}

with NEW.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")

print(f"wrote {NEW.name}: {len(nb['cells'])} cells")
# Sanity: the batch cell must compile as Python.
compile("".join(batch_source), "batch_cell", "exec")
print("batch cell: syntax OK")
