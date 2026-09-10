"""Auto-import helper so registered components are discovered by side effect."""

import importlib
from pathlib import Path


def import_modules(module_dir: str, package_prefix: str) -> None:
    """Import every non-private Python module under module_dir.

    Args:
        module_dir: Directory containing the module files.
        package_prefix: Dotted package path used for import (e.g. "src.divprune.model_module.model").
    """
    root = Path(module_dir)
    for path in sorted(root.glob("*.py")):
        if path.name.startswith("_"):
            continue
        importlib.import_module(f"{package_prefix}.{path.stem}")
