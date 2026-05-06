"""
Script summary:
Shared helper for CLI scripts under `scripts/<domain>/`. Resolves the repository
root by locating `MasterSystemPrompt.md`, so scripts work when nested one level
below `scripts/` (unlike a fixed `Path(__file__).parents[1]`).

Functionality:
- `project_root()` walks parents of this file's directory (`scripts/`) upward until
  it finds `MasterSystemPrompt.md` at the repository root.
- `scripts_dir()` returns the absolute `scripts/` path (parent of this file).

How to apply/run:
- Imported via `importlib` from sibling domain scripts; not executed standalone.
"""

from __future__ import annotations

from pathlib import Path


def scripts_dir() -> Path:
    """Return the absolute path to the `scripts/` directory containing this module."""
    return Path(__file__).resolve().parent


def project_root() -> Path:
    """
    Return the repository root (directory containing `MasterSystemPrompt.md`).

    Raises:
        RuntimeError: If the marker file is not found walking upward from `scripts/`.
    """
    for parent in scripts_dir().parents:
        if (parent / "MasterSystemPrompt.md").is_file():
            return parent
    raise RuntimeError(
        "Could not locate project root (missing MasterSystemPrompt.md). "
        "Run scripts from the SocialAIAdoption repository checkout."
    )
