"""
Script summary:
Archive DiD estimate tables and figures produced before the 2026-06-10 post-window contamination fix.

Functionality:
- Copies estimates/, estimates_exbantopic/, and matching figure subtrees into dated archive folders.
- Writes README.txt documenting superseded spec ids and the contamination reason.

How to apply/run:
  .venv/bin/python scripts/diagnostics/archive_did_precontamination.py
  .venv/bin/python scripts/diagnostics/archive_did_precontamination.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from datetime import date
from pathlib import Path


ARCHIVE_TAG = "2026-06-10"
SUPERSEDED_SPECS = (
    "early_ban_7d",
    "early_ban_14d",
    "post_short_3d",
    "post_medium_7d",
    "post_long_tail",
    "post_first_2bd",
    "full_ban",
)


def _setup_project_root() -> Path:
    """Function summary: resolve repository root via scripts bootstrap."""
    caller = Path(__file__).resolve()
    for parent in caller.parents:
        if parent.name == "scripts" and (parent / "_bootstrap.py").is_file():
            spec = importlib.util.spec_from_file_location("_mod", parent / "_bootstrap.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.setup_project_path(caller)
    raise RuntimeError("bootstrap missing")


PROJECT_ROOT = _setup_project_root()

from src.config_utils import figures_subdir, load_config, tables_subdir  # noqa: E402


def _archive_readme_text() -> str:
    """Function summary: README body for archive folders."""
    return (
        f"Archive date: {ARCHIVE_TAG}\n"
        "Reason: post-window baseline contamination + post-lift treatment-window blend "
        "(subreddit-day static TWFE). See thesis analysis memo 2026-06-10.\n"
        "Superseded spec ids (subreddit-day / comment static TWFE):\n"
        + "\n".join(f"  - {s}" for s in SUPERSEDED_SPECS)
        + "\n"
        "Not archived: event studies, user-week, within-author outputs (clean).\n"
    )


def _ignore_archive_dirs(_dir: str, names: list[str]) -> set[str]:
    """Function summary: skip nested _archived_* directories during copy."""
    return {n for n in names if n.startswith("_archived")}


def _copy_tree(src: Path, dst: Path, *, dry_run: bool) -> None:
    """Function summary: copy directory tree if source exists."""
    if not src.is_dir():
        print(f"[archive] skip missing {src}", flush=True)
        return
    if dry_run:
        print(f"[archive] would copy {src} -> {dst}", flush=True)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=_ignore_archive_dirs)
    print(f"[archive] copied {src.name} -> {dst}", flush=True)


def main() -> None:
    """Function summary: CLI entry for DiD pre-contamination archive."""
    parser = argparse.ArgumentParser(description="Archive pre-contamination DiD artifacts.")
    parser.add_argument("--config", default="config/italy_polarization_setup.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    tables_root = tables_subdir(config, "did")
    figures_root = figures_subdir(config, "did")
    archive_name = f"_archived_precontamination_{ARCHIVE_TAG}"

    table_archive = tables_root / archive_name
    fig_archive = figures_root / archive_name

    for sub in ("estimates", "estimates_exbantopic", "estimates_weighted"):
        _copy_tree(tables_root / sub, table_archive / sub, dry_run=args.dry_run)

    _copy_tree(figures_root, fig_archive / "figures", dry_run=args.dry_run)

    readme = _archive_readme_text()
    if args.dry_run:
        print(readme, flush=True)
        return
    for folder in (table_archive, fig_archive):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "README.txt").write_text(readme, encoding="utf-8")
    print(f"[archive] wrote README to {table_archive} and {fig_archive}", flush=True)


if __name__ == "__main__":
    main()
