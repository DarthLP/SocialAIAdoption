"""
Script summary:
This script applies deterministic cleaning rules to the filtered Reddit daily
chunk corpus and writes cleaned NDJSON outputs to the interim data layer. The
raw layer is preserved unchanged. The script also adds analysis flags to kept
rows and produces daily/subreddit cleaning audit tables for transparent row
accounting.

Functionality:
- Reads raw daily chunks from `data/raw/political_forums/daily_chunks/`.
- Drops rows for configured moderation/deletion placeholders:
  - body == "[removed]"
  - body == "[deleted]"
  - author == "AutoModerator"
  - stickied == true
  - distinguished == "moderator"
- Keeps URL-only text and keeps author == "[deleted]" rows.
- Adds flags on kept rows:
  - is_deleted_author
  - is_bot_name_heuristic
  - is_url_only
  - is_short_text (body length < 20 chars)
- Writes cleaned daily chunks to
  `data/interim/political_forums/cleaned_daily_chunks/<subreddit>/<YYYY-MM-DD>.ndjson`.
- Writes audit outputs to `results/tables/cleaning/` with day/subreddit totals.

How to apply/run:
- `.venv/bin/python scripts/clean_daily_chunks.py --config config/political_forums_setup.yaml`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_config

URL_ONLY_PATTERN = re.compile(r"^\s*https?://\S+\s*$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments and return runtime options."""
    parser = argparse.ArgumentParser(description="Clean filtered daily chunk NDJSON files.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/political_forums_setup.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def is_url_only_text(body: str) -> bool:
    """Function summary: return true when body contains only a single URL token."""
    return bool(URL_ONLY_PATTERN.match(body or ""))


def build_flags(record: Dict[str, Any]) -> Dict[str, bool]:
    """Function summary: compute non-dropping analysis flags for a retained record."""
    body = (record.get("body") or "").strip()
    author = (record.get("author") or "").strip()
    return {
        "is_deleted_author": author == "[deleted]",
        "is_bot_name_heuristic": "bot" in author.lower(),
        "is_url_only": is_url_only_text(body),
        "is_short_text": len(body) < 20,
    }


def evaluate_drop_rules(record: Dict[str, Any]) -> Dict[str, bool]:
    """Function summary: evaluate all configured drop conditions for one record."""
    body = (record.get("body") or "").strip()
    author = (record.get("author") or "").strip()
    return {
        "drop_body_removed": body == "[removed]",
        "drop_body_deleted": body == "[deleted]",
        "drop_author_automoderator": author == "AutoModerator",
        "drop_stickied_true": bool(record.get("stickied")),
        "drop_distinguished_moderator": (record.get("distinguished") == "moderator"),
    }


def clean_one_file(in_path: Path, out_path: Path, subreddit: str, date_utc: str) -> Dict[str, Any]:
    """Function summary: clean one daily NDJSON file and return audit counters."""
    counters: Dict[str, Any] = {
        "subreddit": subreddit,
        "date_utc": date_utc,
        "rows_input": 0,
        "rows_kept": 0,
        "rows_dropped_any": 0,
        "drop_body_removed": 0,
        "drop_body_deleted": 0,
        "drop_author_automoderator": 0,
        "drop_stickied_true": 0,
        "drop_distinguished_moderator": 0,
        "drop_overlap_automod_and_distinguished": 0,
        "invalid_json_rows": 0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open("r", encoding="utf-8") as in_handle, out_path.open("w", encoding="utf-8") as out_handle:
        for raw_line in in_handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counters["invalid_json_rows"] += 1
                continue
            counters["rows_input"] += 1
            rule_hits = evaluate_drop_rules(record)
            for rule_name, is_hit in rule_hits.items():
                if is_hit:
                    counters[rule_name] += 1
            if rule_hits["drop_author_automoderator"] and rule_hits["drop_distinguished_moderator"]:
                counters["drop_overlap_automod_and_distinguished"] += 1
            if any(rule_hits.values()):
                counters["rows_dropped_any"] += 1
                continue
            record.update(build_flags(record))
            out_handle.write(json.dumps(record, ensure_ascii=True))
            out_handle.write("\n")
            counters["rows_kept"] += 1
    counters["row_balance_ok"] = (
        counters["rows_input"] == (counters["rows_kept"] + counters["rows_dropped_any"])
    )
    return counters


def list_daily_chunk_files(raw_daily_dir: Path, subreddits: list[str]) -> list[tuple[str, Path]]:
    """Function summary: gather sorted raw daily files for all configured subreddits."""
    files: list[tuple[str, Path]] = []
    for subreddit in sorted(subreddits):
        sub_dir = raw_daily_dir / subreddit
        if not sub_dir.exists():
            continue
        for in_path in sorted(sub_dir.glob("*.ndjson")):
            files.append((subreddit, in_path))
    return files


def write_audit_outputs(audit_df: pd.DataFrame, tables_dir: Path) -> None:
    """Function summary: write day-level, subreddit-level, and run-note cleaning audit outputs."""
    cleaning_tables_dir = tables_dir / "cleaning"
    cleaning_tables_dir.mkdir(parents=True, exist_ok=True)
    by_day_path = cleaning_tables_dir / "clean_daily_chunks_audit_by_day.csv"
    by_subreddit_path = cleaning_tables_dir / "clean_daily_chunks_audit_by_subreddit.csv"
    note_path = cleaning_tables_dir / "clean_daily_chunks_notes.txt"

    audit_df = audit_df.sort_values(["subreddit", "date_utc"]).reset_index(drop=True)
    audit_df.to_csv(by_day_path, index=False)

    sum_columns = [
        "rows_input",
        "rows_kept",
        "rows_dropped_any",
        "drop_body_removed",
        "drop_body_deleted",
        "drop_author_automoderator",
        "drop_stickied_true",
        "drop_distinguished_moderator",
        "drop_overlap_automod_and_distinguished",
        "invalid_json_rows",
    ]
    by_subreddit = audit_df.groupby("subreddit", as_index=False)[sum_columns].sum()
    by_subreddit["kept_rate_pct"] = (by_subreddit["rows_kept"] / by_subreddit["rows_input"].replace(0, pd.NA)) * 100.0
    by_subreddit.to_csv(by_subreddit_path, index=False)

    automod_drop = int(audit_df["drop_author_automoderator"].sum())
    mod_drop = int(audit_df["drop_distinguished_moderator"].sum())
    overlap = int(audit_df["drop_overlap_automod_and_distinguished"].sum())
    automod_only = automod_drop - overlap
    mod_only = mod_drop - overlap
    note_lines = [
        "Clean Daily Chunks Notes",
        "========================",
        "",
        "Drop rules applied:",
        "- body == [removed]",
        "- body == [deleted]",
        "- author == AutoModerator",
        "- stickied == true",
        '- distinguished == "moderator"',
        "",
        "Keep policy reminders:",
        "- URL-only text is kept.",
        "- author == [deleted] is kept.",
        "",
        (
            f"Run totals: drop_author_automoderator={automod_drop}, "
            f"drop_distinguished_moderator={mod_drop}, overlap={overlap}"
        ),
        (
            f"Unique counts after overlap split: automod_only={automod_only}, "
            f"distinguished_only={mod_only}"
        ),
        (
            "Interpretation: on this full event window, distinguished moderator rows are not "
            "equivalent to AutoModerator rows and include additional non-AutoModerator moderator content."
        ),
    ]
    note_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")


def main() -> None:
    """Function summary: run full cleaning workflow and emit cleaned files plus audit artifacts."""
    args = parse_args()
    config = load_config(args.config)
    raw_daily_dir = Path(config["paths"]["raw_dir"]) / "daily_chunks"
    interim_dir = Path(config["paths"]["interim_dir"]) / "cleaned_daily_chunks"
    tables_dir = Path(config["paths"]["tables_dir"])
    subreddits = list(config["subreddits"]["primary"])

    files = list_daily_chunk_files(raw_daily_dir, subreddits)
    if not files:
        raise FileNotFoundError(f"No daily chunk files found under: {raw_daily_dir}")

    audits: list[Dict[str, Any]] = []
    for subreddit, in_path in files:
        date_utc = in_path.stem
        out_path = interim_dir / subreddit / in_path.name
        audits.append(clean_one_file(in_path=in_path, out_path=out_path, subreddit=subreddit, date_utc=date_utc))

    audit_df = pd.DataFrame(audits)
    write_audit_outputs(audit_df, tables_dir)


if __name__ == "__main__":
    main()
