"""
Script summary:
One-time migration of flat DiD tables under results/tables/.../did/ into nested panels/ and estimates/.

Functionality:
- Moves panel CSVs into did/panels/{country,semantic,subreddit}/.
- Moves did_summary*.csv into estimates/summary/.
- Renames did_coefficients_*, robustness_*, eventstudy_* into family subfolders.
- Idempotent: skips when destination already exists.
- Reports unknown files left at did/ root.

How to apply/run:
  .venv/bin/python scripts/devtools/migrate_did_table_layout.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/devtools/migrate_did_table_layout.py --config config/italy_polarization_setup.yaml --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def _setup_project_root() -> Path:
    """Function summary: resolve repo root via scripts/_bootstrap.py."""
    caller = Path(__file__).resolve()
    for parent in caller.parents:
        if parent.name == "scripts" and (parent / "_bootstrap.py").is_file():
            spec = importlib.util.spec_from_file_location(
                "_socialai_bootstrap_mod", parent / "_bootstrap.py"
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("Failed to load scripts/_bootstrap.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.setup_project_path(caller)
    raise RuntimeError("Could not locate scripts/_bootstrap.py")


PROJECT_ROOT = _setup_project_root()

from src.config_utils import load_config  # noqa: E402
from src.did.outcomes import outcome_family_map  # noqa: E402
from src.did.paths import (  # noqa: E402
    did_outcome_table_path,
    did_panels_dir,
    did_root,
    did_summary_paths,
)

PANEL_COUNTRY_PREFIXES = (
    "did_country_panel_",
    "did_country_panel_by_universe_slice_",
)
PANEL_SEMANTIC_PREFIX = "did_semantic_"
PANEL_SUBREDDIT_NAMES = (
    "did_subreddit_panel_1d.csv",
    "did_subreddit_panel_by_universe_slice_1d.csv",
)

COEF_RE = re.compile(r"^did_coefficients_(.+)\.csv$")
ROB_RE = re.compile(r"^robustness_(.+)\.csv$")
ES_RE = re.compile(r"^eventstudy_(.+)\.csv$")


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for DiD table layout migration."""
    parser = argparse.ArgumentParser(description="Migrate flat did/ tables to nested layout.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print moves without writing files.",
    )
    return parser.parse_args()


def _panel_kind_for_name(name: str) -> str | None:
    """Function summary: classify flat panel filename into panels/ subgroup."""
    if name in PANEL_SUBREDDIT_NAMES:
        return "subreddit"
    if name.startswith(PANEL_SEMANTIC_PREFIX):
        return "semantic"
    for prefix in PANEL_COUNTRY_PREFIXES:
        if name.startswith(prefix):
            return "country"
    return None


def _planned_moves(root: Path, config: dict) -> List[Tuple[Path, Path]]:
    """Function summary: build list of (src, dst) moves for flat did/ files."""
    family_map = outcome_family_map()
    moves: List[Tuple[Path, Path]] = []

    summary_csv, summary_labeled = did_summary_paths(config)
    for flat_name in ("did_summary.csv", "did_summary_labeled.csv"):
        src = root / flat_name
        dst = summary_csv if flat_name == "did_summary.csv" else summary_labeled
        if src.is_file() and src != dst:
            moves.append((src, dst))

    for path in sorted(root.glob("*.csv")):
        name = path.name
        kind = _panel_kind_for_name(name)
        if kind:
            dst = did_panels_dir(config, kind) / name  # type: ignore[arg-type]
            if path != dst:
                moves.append((path, dst))
            continue

        m = COEF_RE.match(name)
        if m:
            oid = m.group(1)
            fam = family_map.get(oid)
            if fam:
                dst = did_outcome_table_path(config, fam, "coefficients", oid)
                moves.append((path, dst))
            continue

        m = ROB_RE.match(name)
        if m:
            oid = m.group(1)
            fam = family_map.get(oid, "lexical")
            dst = did_outcome_table_path(config, fam, "robustness", oid)
            moves.append((path, dst))
            continue

        m = ES_RE.match(name)
        if m:
            oid = m.group(1)
            fam = family_map.get(oid)
            if fam:
                dst = did_outcome_table_path(config, fam, "event_study", oid)
                moves.append((path, dst))
            continue

    return moves


def _execute_move(src: Path, dst: Path, did_root_path: Path, dry_run: bool) -> str:
    """Function summary: move or skip one file; return status label."""
    if dst.is_file():
        return "skip_exists"
    if dry_run:
        print(f"[dry-run] {src} -> {dst}", flush=True)
        return "dry_run"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    try:
        rel = dst.relative_to(did_root_path)
    except ValueError:
        rel = dst
    print(f"[migrate] {src.name} -> {rel}", flush=True)
    return "moved"


def main() -> None:
    """Function summary: run migration from flat did/ to nested layout."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    root = did_root(config)
    if not root.is_dir():
        print(f"[migrate_did_table_layout] missing {root}", flush=True)
        return

    moves = _planned_moves(root, config)
    stats: Dict[str, int] = {}
    for src, dst in moves:
        status = _execute_move(src, dst, root, args.dry_run)
        stats[status] = stats.get(status, 0) + 1

    remaining = sorted(p.name for p in root.glob("*.csv"))
    print(
        f"[migrate_did_table_layout] done: {stats} | remaining flat csv: {remaining or '(none)'}",
        flush=True,
    )
    if remaining and not args.dry_run:
        print(
            "[migrate_did_table_layout] re-run did_event_study.py to generate summary txt exports.",
            flush=True,
        )


if __name__ == "__main__":
    main()
