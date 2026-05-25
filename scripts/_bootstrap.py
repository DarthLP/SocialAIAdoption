"""
Script summary:
Centralize repository root resolution and sys.path setup for CLI scripts under
`scripts/` (including nested `scripts/archive/` paths).

Functionality:
- `setup_project_path(caller_file)` walks upward from the invoking script until it
  finds `scripts/_project_root.py`, loads `project_root()`, and prepends the repo
  root to `sys.path`.

How to apply/run:
- Imported via importlib from domain scripts; not executed standalone.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def setup_project_path(caller_file: str | Path) -> Path:
    """Function summary: resolve repo root from any script under `scripts/` and ensure imports work.

    Parameters:
    - caller_file: ``__file__`` of the invoking script.

    Returns:
    - Absolute repository root Path.

    Raises:
    - RuntimeError: If `scripts/_project_root.py` cannot be located.
    """
    caller = Path(caller_file).resolve()
    scripts_dir: Path | None = None
    for parent in caller.parents:
        if parent.name == "scripts" and (parent / "_project_root.py").is_file():
            scripts_dir = parent
            break
    if scripts_dir is None:
        raise RuntimeError(
            f"Could not locate scripts/_project_root.py walking up from {caller}"
        )
    spec = importlib.util.spec_from_file_location(
        "_socialai_scripts_project_root_mod",
        scripts_dir / "_project_root.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {scripts_dir / '_project_root.py'}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root = mod.project_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
