"""
Script summary:
This script removes duplicate Reddit comments from filtered daily chunk NDJSON files
under the project raw data directory. It is designed for safe cleanup after
interrupt/restart scenarios where a tail segment may have been appended twice.

Functionality:
- Recursively scans `daily_chunks/*/*.ndjson` files.
- Deduplicates records by comment `id` while preserving first-seen row order.
- Supports dry-run reporting and apply mode with in-place replacement.
- Writes a CSV report with per-file before/after counts and duplicates removed.

How to run:
- Dry run (no file changes):
  `.venv/bin/python scripts/dedupe_daily_chunks.py --config config/political_forums_setup.yaml`
- Apply dedupe in place:
  `.venv/bin/python scripts/dedupe_daily_chunks.py --config config/political_forums_setup.yaml --apply`
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_config


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI args for dedupe mode, config path, and reporting output."""
    parser = argparse.ArgumentParser(description="Deduplicate daily chunk NDJSON files by comment id.")
    parser.add_argument("--config", type=str, default="config/political_forums_setup.yaml")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite files in place. Without this flag, script runs in dry-run mode.",
    )
    parser.add_argument(
        "--report_csv",
        type=str,
        default="results/tables/dedupe_daily_chunks_report.csv",
        help="CSV output path with per-file dedupe metrics.",
    )
    return parser.parse_args()


def dedupe_file(path: Path, apply_changes: bool) -> Dict[str, Any]:
    """Function summary: deduplicate one NDJSON file by `id` and optionally write cleaned output in place."""
    rows_total = 0
    rows_kept = 0
    rows_removed = 0
    missing_id_rows = 0
    invalid_json_rows = 0
    seen_ids: set[str] = set()

    if apply_changes:
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tmp:
            tmp_path = Path(tmp.name)
            with path.open("r", encoding="utf-8") as src:
                for raw_line in src:
                    rows_total += 1
                    line = raw_line.rstrip("\n")
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        invalid_json_rows += 1
                        rows_kept += 1
                        tmp.write(raw_line)
                        continue

                    rec_id = rec.get("id")
                    if isinstance(rec_id, str) and rec_id:
                        if rec_id in seen_ids:
                            rows_removed += 1
                            continue
                        seen_ids.add(rec_id)
                        rows_kept += 1
                        tmp.write(raw_line)
                    else:
                        missing_id_rows += 1
                        rows_kept += 1
                        tmp.write(raw_line)

        tmp_path.replace(path)
    else:
        with path.open("r", encoding="utf-8") as src:
            for raw_line in src:
                rows_total += 1
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json_rows += 1
                    rows_kept += 1
                    continue

                rec_id = rec.get("id")
                if isinstance(rec_id, str) and rec_id:
                    if rec_id in seen_ids:
                        rows_removed += 1
                        continue
                    seen_ids.add(rec_id)
                    rows_kept += 1
                else:
                    missing_id_rows += 1
                    rows_kept += 1

    return {
        "file": str(path),
        "rows_total": rows_total,
        "rows_kept": rows_kept,
        "rows_removed": rows_removed,
        "missing_id_rows": missing_id_rows,
        "invalid_json_rows": invalid_json_rows,
    }


def iter_daily_chunk_files(base_raw_dir: Path) -> list[Path]:
    """Function summary: return sorted list of project daily chunk NDJSON files."""
    root = base_raw_dir / "daily_chunks"
    if not root.exists():
        return []
    return sorted(root.glob("*/*.ndjson"))


def write_report(path: Path, rows: list[Dict[str, Any]]) -> None:
    """Function summary: write per-file dedupe metrics to CSV for reproducibility and auditing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file",
                "rows_total",
                "rows_kept",
                "rows_removed",
                "missing_id_rows",
                "invalid_json_rows",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Function summary: orchestrate dry-run or apply dedupe over all daily chunk NDJSON files and emit report."""
    args = parse_args()
    config = load_config(args.config)
    base_raw_dir = Path(config["paths"]["raw_dir"])
    report_path = Path(args.report_csv)

    files = iter_daily_chunk_files(base_raw_dir)
    print(f"found_files={len(files)} apply={bool(args.apply)}")

    results: list[Dict[str, Any]] = []
    for path in files:
        stats = dedupe_file(path, apply_changes=bool(args.apply))
        results.append(stats)
        if stats["rows_removed"] > 0:
            print(f"deduped file={path} removed={stats['rows_removed']} total={stats['rows_total']}")

    write_report(report_path, results)
    total_rows = sum(int(r["rows_total"]) for r in results)
    total_removed = sum(int(r["rows_removed"]) for r in results)
    print(f"done files={len(results)} rows_total={total_rows} rows_removed={total_removed} report={report_path}")


if __name__ == "__main__":
    main()
