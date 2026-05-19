"""
Script summary:
Build human-readable exclusion summaries from Stage-2 screening outputs for the
Italy polarization study.

Functionality:
- Reads subreddit_screening_pooled.csv and optional Stage-1 audit aggregates.
- Writes subreddit_exclusion_summary.csv (one row per primary subreddit) and
  exclusion_summary_by_code.csv (counts by exclusion code).

How to apply/run:
  Called automatically from screen_subreddits.py, or:
  .venv/bin/python scripts/cleaning/write_exclusion_summary.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _resolve_project_root() -> Path:
    """Function summary: load scripts/_project_root.py and return repository root Path."""
    scripts_dir = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_socialai_scripts_project_root_mod", scripts_dir / "_project_root.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/_project_root.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.project_root()


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import (  # noqa: E402
    infer_subreddit_topic,
    load_config,
    load_subreddit_metadata,
    resolve_primary_subreddits,
    subreddit_arm_map,
    subreddit_family_map,
    subreddit_topic_map,
    topic_family_map,
)

CODE_PRIORITY = [
    "PROFILE_USER",
    "HIGH_URL_ONLY_SHARE",
    "LOW_ITALIAN_POOLED",
    "LOW_VOLUME_WINDOW",
]

CODE_MESSAGES = {
    "PROFILE_USER": "Excluded: subreddit is a user profile (name matches ^u_).",
    "HIGH_URL_ONLY_SHARE": "Excluded: URL-only spam share at Stage 1 exceeds threshold.",
    "LOW_ITALIAN_POOLED": "Excluded: pooled Italian langid share below 70% threshold.",
    "LOW_VOLUME_WINDOW": "Low volume: fewer than 100 kept comments in the study window.",
}


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Write human-readable exclusion summary CSVs.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def primary_reason(codes: str) -> str:
    """Function summary: pick dominant exclusion code by fixed priority.

    Parameters:
    - codes: pipe-separated exclusion codes.

    Returns:
    - Single code string or empty.
    """
    if not codes:
        return ""
    parts = [c.strip() for c in codes.split("|") if c.strip()]
    for code in CODE_PRIORITY:
        if code in parts:
            return code
    return parts[0]


def plain_english_summary(row: Dict[str, Any], reason: str) -> str:
    """Function summary: one-sentence summary for a screened subreddit.

    Parameters:
    - row: pooled screening row dict.
    - reason: primary exclusion code.

    Returns:
    - Human-readable sentence.
    """
    action = str(row.get("action", ""))
    if action == "large_volume":
        return "Included: passes screening gates with ≥100 kept comments (large_volume)."
    if action == "low_volume":
        kept = row.get("n_kept_window", 0)
        return f"Low volume: {kept} kept comments in window; use as supplementary sample only."
    if reason == "LOW_ITALIAN_POOLED":
        share = row.get("italian_share_pooled", 0)
        return f"Excluded: pooled Italian share {share:.0%} below 70% threshold."
    if reason == "HIGH_URL_ONLY_SHARE":
        share = row.get("url_only_share_stage1", 0)
        return f"Excluded: URL-only share at Stage 1 is {share:.0%} (above forum threshold)."
    if reason == "PROFILE_USER":
        return "Excluded: user-profile subreddit (not a community forum)."
    if reason:
        return CODE_MESSAGES.get(reason, f"Excluded or flagged: {reason}.")
    return "Excluded from main analysis."


def recommended_use(action: str) -> str:
    """Function summary: map screening action to recommended_use label.

    Parameters:
    - action: large_volume | low_volume | excluded.

    Returns:
    - recommended_use string.
    """
    if action == "large_volume":
        return "large_volume"
    if action == "low_volume":
        return "low_volume"
    return "do_not_use"


def write_exclusion_summaries(
    config: Dict[str, Any],
    project_root: Path,
    tables_dir: Path,
) -> None:
    """Function summary: write exclusion summary CSVs under tables_dir/screening/.

    Parameters:
    - config: loaded study YAML.
    - project_root: repository root.
    - tables_dir: study tables directory.
    """
    pooled_path = tables_dir / "screening" / "subreddit_screening_pooled.csv"
    if not pooled_path.is_file():
        raise FileNotFoundError(f"Missing {pooled_path}; run screen_subreddits.py first.")

    pooled = pd.read_csv(pooled_path)
    pooled_by_sub = {str(r["subreddit"]): r for r in pooled.to_dict(orient="records")}

    audit_path = tables_dir / "cleaning" / "clean_daily_chunks_audit_by_subreddit.csv"
    audit_by_sub: Dict[str, Dict[str, Any]] = {}
    if audit_path.is_file():
        audit_df = pd.read_csv(audit_path)
        audit_by_sub = {str(r["subreddit"]): r for r in audit_df.to_dict(orient="records")}

    metadata = load_subreddit_metadata(config, project_root=project_root)
    sub_to_topic = subreddit_topic_map(config, include_topic_aliases=False)
    sub_to_family = subreddit_family_map(config, include_family_aliases=False)
    topic_to_family = topic_family_map(config, include_family_aliases=False)
    arms = subreddit_arm_map(config)

    rows: List[Dict[str, Any]] = []
    for subreddit in resolve_primary_subreddits(config):
        screen = pooled_by_sub.get(subreddit, {})
        audit = audit_by_sub.get(subreddit, {})
        topic = sub_to_topic.get(subreddit) or infer_subreddit_topic(
            config, subreddit, metadata=metadata
        )
        family = topic_to_family.get(topic, sub_to_family.get(subreddit, "italian"))
        codes = str(screen.get("exclusion_codes", "") or "")
        reason = primary_reason(codes)
        action = str(screen.get("action", ""))
        rows_input = int(audit.get("rows_input", screen.get("rows_input_stage1", 0)) or 0)
        rows_kept = int(audit.get("rows_kept", screen.get("n_kept_window", 0)) or 0)
        kept_pct = round(100.0 * rows_kept / rows_input, 2) if rows_input > 0 else 0.0
        rows.append(
            {
                "subreddit": subreddit,
                "arm": arms.get(subreddit, screen.get("arm", "")),
                "topic": topic,
                "topic_family": family,
                "volume_band": screen.get("volume_band", ""),
                "action": action,
                "exclusion_codes": codes,
                "n_kept_window": screen.get("n_kept_window", 0),
                "italian_share_pooled": screen.get("italian_share_pooled", ""),
                "url_only_share_stage1": screen.get("url_only_share_stage1", ""),
                "primary_reason": reason,
                "plain_english_summary": plain_english_summary(screen, reason),
                "stage1_kept_rate_pct": kept_pct,
                "drop_url_only_stage1": audit.get("drop_url_only", screen.get("drop_url_only_stage1", 0)),
                "drop_body_removed_stage1": audit.get("drop_body_removed", 0),
                "drop_body_deleted_stage1": audit.get("drop_body_deleted", 0),
                "recommended_use": recommended_use(action),
            }
        )

    out_dir = tables_dir / "screening"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "subreddit_exclusion_summary.csv", index=False)

    by_code_rows: List[Dict[str, Any]] = []
    for code in CODE_PRIORITY:
        affected = summary_df[summary_df["exclusion_codes"].str.contains(code, na=False)]
        if affected.empty:
            continue
        by_code_rows.append(
            {
                "code": code,
                "n_subreddits": int(affected["subreddit"].nunique()),
                "n_kept_window_total": int(affected["n_kept_window"].sum()),
            }
        )
    pd.DataFrame(by_code_rows).to_csv(out_dir / "exclusion_summary_by_code.csv", index=False)


def main() -> None:
    """Function summary: CLI entrypoint for exclusion summary CSVs."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    tables_dir = Path(config["paths"]["tables_dir"])
    write_exclusion_summaries(config=config, project_root=PROJECT_ROOT, tables_dir=tables_dir)
    print(f"[write_exclusion_summary] wrote under {tables_dir / 'screening'}", flush=True)


if __name__ == "__main__":
    main()
