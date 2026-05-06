"""
Script summary:
This script applies deterministic cleaning rules to the filtered Reddit daily
chunk corpus and writes cleaned monthly Parquet outputs to the interim data
layer. The raw layer is preserved unchanged. The script adds analysis flags to
kept rows, enforces a fixed interim schema, reports schema coercion mismatches,
and produces daily/subreddit cleaning audit tables for transparent row
accounting plus type-quality diagnostics.

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
- Writes cleaned monthly chunks to
  `data/interim/political_forums/cleaned_monthly_chunks/<subreddit>/<YYYY-MM>.parquet`.
- Uses snappy-compressed Parquet via pandas/pyarrow.
- Enforces a canonical schema for interim data and emits schema mismatch audits.
- Writes audit outputs to `results/tables/cleaning/` with day/subreddit totals
  and schema coercion diagnostics.

How to apply/run:
- `.venv/bin/python scripts/cleaning/clean_daily_chunks.py --config config/political_forums_setup.yaml`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import importlib.util
import re
import sys
from collections import defaultdict
import time
from typing import Any, Dict

import pandas as pd

def _resolve_project_root() -> Path:
    """Load scripts/_project_root.py and return the repository root Path."""
    _scripts_dir = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_socialai_scripts_project_root_mod",
        _scripts_dir / "_project_root.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/_project_root.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.project_root()


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_config

URL_ONLY_PATTERN = re.compile(r"^\s*https?://\S+\s*$", re.IGNORECASE)
SCHEMA_SAMPLE_LIMIT = 200

INTERIM_SCHEMA: dict[str, str] = {
    "id": "string",
    "author": "string",
    "subreddit": "string",
    "created_utc": "Int64",
    "body": "string",
    "score": "Int64",
    "parent_id": "string",
    "link_id": "string",
    "permalink": "string",
    "edited": "string",
    "controversiality": "Int64",
    "distinguished": "string",
    "stickied": "boolean",
    "is_deleted_author": "boolean",
    "is_bot_name_heuristic": "boolean",
    "is_url_only": "boolean",
    "is_short_text": "boolean",
    "date_utc": "string",
    "year_month": "string",
}
REQUIRED_NON_NULL_COLUMNS = [
    "id",
    "author",
    "subreddit",
    "created_utc",
    "body",
    "date_utc",
    "year_month",
]
BOOLEAN_COLUMNS = {
    "stickied",
    "is_deleted_author",
    "is_bot_name_heuristic",
    "is_url_only",
    "is_short_text",
}
INTEGER_COLUMNS = {"created_utc", "score", "controversiality"}


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


def clean_one_file(in_path: Path, subreddit: str, date_utc: str) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """Function summary: clean one daily NDJSON file and return audit counters plus kept records."""
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
    kept_records: list[Dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as in_handle:
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
            record["date_utc"] = date_utc
            record["year_month"] = date_utc[:7]
            kept_records.append(record)
            counters["rows_kept"] += 1
    counters["row_balance_ok"] = (
        counters["rows_input"] == (counters["rows_kept"] + counters["rows_dropped_any"])
    )
    return counters, kept_records


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


def coerce_boolean(value: Any) -> bool | None:
    """Function summary: coerce mixed boolean-like values to bool/None for schema enforcement."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "1", "yes", "y"}:
            return True
        if normalized in {"false", "f", "0", "no", "n"}:
            return False
        if normalized in {"", "none", "null", "na"}:
            return None
    return None


def apply_schema_and_collect_issues(
    records: list[Dict[str, Any]],
    subreddit: str,
    year_month: str,
) -> tuple[pd.DataFrame, dict[str, Any], list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Function summary: enforce canonical interim schema and return typed frame plus issue diagnostics."""
    if not records:
        empty_df = pd.DataFrame(columns=list(INTERIM_SCHEMA))
        for col, dtype in INTERIM_SCHEMA.items():
            empty_df[col] = empty_df[col].astype(dtype)
        summary = {
            "subreddit": subreddit,
            "year_month": year_month,
            "rows_before_schema": 0,
            "rows_after_schema": 0,
            "rows_dropped_required_null": 0,
        }
        return empty_df, summary, [], []

    df = pd.DataFrame(records)
    for col in INTERIM_SCHEMA:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[list(INTERIM_SCHEMA)]
    row_count = len(df)
    field_issues: list[Dict[str, Any]] = []
    invalid_row_samples: list[Dict[str, Any]] = []

    for col in df.columns:
        before_null = int(df[col].isna().sum())
        if col in BOOLEAN_COLUMNS:
            non_null_before = int(df[col].notna().sum())
            coerced = df[col].map(coerce_boolean)
            non_null_after = int(pd.Series(coerced).notna().sum())
            invalid_count = max(0, non_null_before - non_null_after)
            df[col] = pd.Series(coerced, dtype="boolean")
            field_issues.append(
                {
                    "subreddit": subreddit,
                    "year_month": year_month,
                    "column": col,
                    "expected_dtype": INTERIM_SCHEMA[col],
                    "rows_total": row_count,
                    "null_before": before_null,
                    "null_after": int(df[col].isna().sum()),
                    "invalid_cast_count": invalid_count,
                }
            )
        elif col in INTEGER_COLUMNS:
            non_null_before = int(df[col].notna().sum())
            numeric = pd.to_numeric(df[col], errors="coerce")
            non_null_after = int(numeric.notna().sum())
            invalid_count = max(0, non_null_before - non_null_after)
            df[col] = numeric.astype("Int64")
            field_issues.append(
                {
                    "subreddit": subreddit,
                    "year_month": year_month,
                    "column": col,
                    "expected_dtype": INTERIM_SCHEMA[col],
                    "rows_total": row_count,
                    "null_before": before_null,
                    "null_after": int(df[col].isna().sum()),
                    "invalid_cast_count": invalid_count,
                }
            )
        else:
            non_string_mask = df[col].notna() & (~df[col].map(lambda v: isinstance(v, str)))
            invalid_count = int(non_string_mask.sum())
            df[col] = df[col].astype("string")
            field_issues.append(
                {
                    "subreddit": subreddit,
                    "year_month": year_month,
                    "column": col,
                    "expected_dtype": INTERIM_SCHEMA[col],
                    "rows_total": row_count,
                    "null_before": before_null,
                    "null_after": int(df[col].isna().sum()),
                    "invalid_cast_count": invalid_count,
                }
            )

    required_ok = df[REQUIRED_NON_NULL_COLUMNS].notna().all(axis=1)
    dropped_required = int((~required_ok).sum())
    if dropped_required > 0:
        sample = df.loc[~required_ok, REQUIRED_NON_NULL_COLUMNS].head(SCHEMA_SAMPLE_LIMIT)
        for _, row in sample.iterrows():
            invalid_row_samples.append(
                {
                    "subreddit": subreddit,
                    "year_month": year_month,
                    **{col: row.get(col) for col in REQUIRED_NON_NULL_COLUMNS},
                }
            )
    df = df.loc[required_ok].reset_index(drop=True)

    summary = {
        "subreddit": subreddit,
        "year_month": year_month,
        "rows_before_schema": row_count,
        "rows_after_schema": int(len(df)),
        "rows_dropped_required_null": dropped_required,
    }
    return df, summary, field_issues, invalid_row_samples


def write_monthly_parquet(cleaned_df: pd.DataFrame, interim_dir: Path, subreddit: str, year_month: str) -> Path:
    """Function summary: write one subreddit-month cleaned dataset to snappy-compressed Parquet."""
    target_dir = interim_dir / "cleaned_monthly_chunks" / subreddit
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{year_month}.parquet"
    cleaned_df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    return out_path


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


def write_schema_outputs(
    summary_rows: list[Dict[str, Any]],
    issue_rows: list[Dict[str, Any]],
    invalid_row_samples: list[Dict[str, Any]],
    tables_dir: Path,
) -> None:
    """Function summary: write schema coercion summaries, field issues, and invalid row samples."""
    cleaning_tables_dir = tables_dir / "cleaning"
    cleaning_tables_dir.mkdir(parents=True, exist_ok=True)
    by_month_path = cleaning_tables_dir / "clean_daily_chunks_schema_coercion_by_month.csv"
    field_issue_path = cleaning_tables_dir / "clean_daily_chunks_schema_coercion_field_issues.csv"
    invalid_samples_path = cleaning_tables_dir / "clean_daily_chunks_schema_invalid_row_samples.csv"

    pd.DataFrame(summary_rows).sort_values(["subreddit", "year_month"]).to_csv(by_month_path, index=False)
    pd.DataFrame(issue_rows).sort_values(["subreddit", "year_month", "column"]).to_csv(field_issue_path, index=False)
    pd.DataFrame(invalid_row_samples[:SCHEMA_SAMPLE_LIMIT]).to_csv(invalid_samples_path, index=False)


def main() -> None:
    """Function summary: run full cleaning workflow and emit cleaned monthly Parquet plus audit artifacts."""
    args = parse_args()
    config = load_config(args.config)
    raw_daily_dir = Path(config["paths"]["raw_dir"]) / "daily_chunks"
    interim_dir = Path(config["paths"]["interim_dir"])
    tables_dir = Path(config["paths"]["tables_dir"])
    subreddits = list(config["subreddits"]["primary"])

    files = list_daily_chunk_files(raw_daily_dir, subreddits)
    if not files:
        raise FileNotFoundError(f"No daily chunk files found under: {raw_daily_dir}")
    print(f"[clean_daily_chunks] discovered_daily_files={len(files)}", flush=True)

    audits: list[Dict[str, Any]] = []
    grouped_files: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for subreddit, in_path in files:
        grouped_files[(subreddit, in_path.stem[:7])].append(in_path)

    schema_summary_rows: list[Dict[str, Any]] = []
    schema_issue_rows: list[Dict[str, Any]] = []
    schema_invalid_row_samples: list[Dict[str, Any]] = []
    grouped_items = sorted(grouped_files.items())
    started_at = time.perf_counter()
    for idx, ((subreddit, year_month), month_files) in enumerate(grouped_items, start=1):
        print(
            f"[clean_daily_chunks] month_start {idx}/{len(grouped_items)} subreddit={subreddit} month={year_month} files={len(month_files)}",
            flush=True,
        )
        monthly_records: list[Dict[str, Any]] = []
        for file_idx, in_path in enumerate(sorted(month_files), start=1):
            date_utc = in_path.stem
            print(
                f"[clean_daily_chunks] file_start subreddit={subreddit} month={year_month} file={file_idx}/{len(month_files)} date={date_utc}",
                flush=True,
            )
            counters, kept_records = clean_one_file(in_path=in_path, subreddit=subreddit, date_utc=date_utc)
            audits.append(counters)
            monthly_records.extend(kept_records)
            print(
                f"[clean_daily_chunks] file_done subreddit={subreddit} date={date_utc} rows_input={counters['rows_input']} rows_kept={counters['rows_kept']}",
                flush=True,
            )
        cleaned_df, summary, issues, invalid_samples = apply_schema_and_collect_issues(
            records=monthly_records,
            subreddit=subreddit,
            year_month=year_month,
        )
        write_monthly_parquet(
            cleaned_df=cleaned_df,
            interim_dir=interim_dir,
            subreddit=subreddit,
            year_month=year_month,
        )
        schema_summary_rows.append(summary)
        schema_issue_rows.extend(issues)
        schema_invalid_row_samples.extend(invalid_samples)
        elapsed = time.perf_counter() - started_at
        print(
            f"[clean_daily_chunks] month_done subreddit={subreddit} month={year_month} rows_after_schema={summary['rows_after_schema']} elapsed_s={elapsed:.1f}",
            flush=True,
        )

    audit_df = pd.DataFrame(audits)
    write_audit_outputs(audit_df, tables_dir)
    write_schema_outputs(schema_summary_rows, schema_issue_rows, schema_invalid_row_samples, tables_dir)
    print("[clean_daily_chunks] done wrote_audits_and_schema_outputs=true", flush=True)


if __name__ == "__main__":
    main()
