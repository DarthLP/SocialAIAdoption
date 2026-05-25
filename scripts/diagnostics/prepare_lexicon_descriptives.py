"""
Script summary:
Aggregate dominant ideology, pair-framing, and v4 metadata columns into descriptives tables.

Functionality:
- Daily and rolling-ready series for Italian forums; symmetric ban windows (launch primary, lift appendix).
- Pre-registered primary outcome table (net_ideology + pair_framing_net_strict, W0 launch).
- Tie-break robustness summary using weighted ideology columns.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_lexicon_descriptives.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd



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

from src.config_utils import (  # noqa: E402
    load_config,
    load_polarization_config,
    require_dominant_v1_ideology_scoring,
    resolve_primary_subreddits,
    subreddit_topic_map,
)

from scripts.diagnostics import descriptives_util as du  # noqa: E402

ITALIAN_FAMILIES = frozenset({"it_political", "it_pure_political", "it_others"})

READ_COLUMNS = [
    "id",
    "subreddit",
    "date_utc",
    "n_words",
    "topic_family",
    "lang_comment",
    "primary_lexicon",
    "net_ideology",
    "left_hits",
    "center_hits",
    "right_hits",
    "left_rate_100w",
    "center_rate_100w",
    "right_rate_100w",
    "net_ideology_weighted",
    "pair_framing_net_strict",
    "pair_framing_net_all",
    "pair_framing_rate_100w_strict",
    "pair_active_strict",
    "stance_pro_rate_100w",
    "stance_contra_rate_100w",
    "valence_negative_rate_100w",
    "valence_positive_rate_100w",
    "polarized_yes_rate_100w",
    "relevance_weighted_contra_rate_100w",
]


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare lexicon descriptives tables.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def load_comment_frame(shard_root: Path, subreddits: List[str]) -> pd.DataFrame:
    """Function summary: load enriched parquet columns for listed subreddits.

    Parameters:
    - shard_root: cleaned_monthly_chunks root.
    - subreddits: subreddit names.

    Returns:
    - Combined dataframe.
    """
    parts: List[pd.DataFrame] = []
    for sub in subreddits:
        shard_dir = shard_root / sub
        if not shard_dir.is_dir():
            continue
        for shard in sorted(shard_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(shard, columns=READ_COLUMNS)
            except Exception:
                try:
                    df = pd.read_parquet(shard)
                except Exception:
                    continue
            if df.empty:
                continue
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _italian_mask(df: pd.DataFrame) -> pd.Series:
    """Function summary: boolean mask for Italian-primary comments."""
    if "lang_comment" in df.columns:
        lang = df["lang_comment"].astype(str).str.lower() == "it"
    else:
        lang = pd.Series(True, index=df.index)
    if "topic_family" in df.columns:
        fam = df["topic_family"].isin(ITALIAN_FAMILIES)
        return lang & fam
    if "primary_lexicon" in df.columns:
        return df["primary_lexicon"].astype(str).str.lower() == "it"
    return lang


def daily_by_family(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """Function summary: comment-weighted daily means by topic_family.

    Parameters:
    - df: comment-level frame.
    - metrics: columns to average.

    Returns:
    - Daily family aggregates.
    """
    rows: List[Dict[str, Any]] = []
    for (fam, day), grp in df.groupby(["topic_family", "date_utc"], sort=True):
        nw = grp["n_words"].astype(float)
        row: Dict[str, Any] = {
            "topic_family": fam,
            "date_utc": day,
            "n_comments": len(grp),
        }
        for m in metrics:
            if m in grp.columns:
                row[f"{m}_mean"] = du.weighted_mean(grp[m], nw)
        rows.append(row)
    return pd.DataFrame(rows)


def ban_window_table(
    df: pd.DataFrame,
    anchor: str,
    metrics: List[str],
    label: str,
) -> pd.DataFrame:
    """Function summary: pool pre/post symmetric windows W0–W2 around anchor.

    Parameters:
    - df: Italian comment frame with date_utc.
    - anchor: YYYY-MM-DD.
    - metrics: outcome columns.
    - label: anchor label for output.

    Returns:
    - Window summary table.
    """
    rows: List[Dict[str, Any]] = []
    for wk in (0, 1, 2):
        pre_m, post_m = du.ban_window_masks(df["date_utc"], anchor, wk)
        for phase, mask in (("pre", pre_m), ("post", post_m)):
            grp = df[mask]
            if grp.empty:
                continue
            nw = grp["n_words"].astype(float)
            row: Dict[str, Any] = {
                "anchor": label,
                "window": f"W{wk}",
                "phase": phase,
                "n_comments": len(grp),
            }
            for m in metrics:
                if m in grp.columns:
                    row[f"{m}_mean"] = du.weighted_mean(grp[m], nw)
            rows.append(row)
    return pd.DataFrame(rows)


def primary_outcomes_w0(df: pd.DataFrame, launch: str) -> pd.DataFrame:
    """Function summary: pre-registered W0 contrasts for primary outcomes on it_political.

    Parameters:
    - df: Italian comments.
    - launch: launch date string.

    Returns:
    - One-row-per-metric primary table.
    """
    core = df[df["topic_family"] == "it_political"].copy()
    pre_m, post_m = du.ban_window_masks(core["date_utc"], launch, 0)
    pre, post = core[pre_m], core[post_m]
    rows = []
    for metric, label in (
        ("net_ideology", "primary_A_net_ideology"),
        ("pair_framing_net_strict", "primary_B_pair_framing_net_strict"),
    ):
        if metric not in core.columns:
            continue
        rows.append(
            {
                "outcome": label,
                "pre_mean": du.weighted_mean(pre[metric], pre["n_words"]),
                "post_mean": du.weighted_mean(post[metric], post["n_words"]),
                "delta_post_minus_pre": du.weighted_mean(post[metric], post["n_words"])
                - du.weighted_mean(pre[metric], pre["n_words"]),
                "n_pre": len(pre),
                "n_post": len(post),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Function summary: write lexicon descriptives CSV tables."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    require_dominant_v1_ideology_scoring(config)
    pol_cfg = load_polarization_config(config)
    start, end_excl, launch, lift = du.event_dates_from_config(config)

    shard_root = Path(config["paths"]["interim_dir"]) / "cleaned_monthly_chunks"
    tables_dir = Path(config["paths"]["tables_dir"]) / "descriptives"
    tables_dir.mkdir(parents=True, exist_ok=True)

    du.stamp_metrics_notes(tables_dir / "polarization_metrics_notes.txt", config)

    subs = resolve_primary_subreddits(config)
    topic_map = subreddit_topic_map(config, include_topic_aliases=False)
    df = load_comment_frame(shard_root, subs)
    if df.empty:
        print("[prepare_lexicon_descriptives] no data", flush=True)
        return
    if "date_utc" not in df.columns and "created_utc" in df.columns:
        df["date_utc"] = pd.to_datetime(df["created_utc"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    df = df[(df["date_utc"] >= start) & (df["date_utc"] < end_excl)].copy()
    if "topic_family" not in df.columns:
        from src.config_utils import subreddit_family_map

        df["topic_family"] = df["subreddit"].map(subreddit_family_map(config))
    it_df = df[_italian_mask(df)].copy()

    ideology_metrics = [
        "net_ideology",
        "left_rate_100w",
        "center_rate_100w",
        "right_rate_100w",
        "net_ideology_weighted",
    ]
    pair_metrics = [
        "pair_framing_net_strict",
        "pair_framing_net_all",
        "pair_framing_rate_100w_strict",
        "pair_active_strict",
    ]
    meta_metrics = [
        "stance_pro_rate_100w",
        "stance_contra_rate_100w",
        "valence_negative_rate_100w",
        "valence_positive_rate_100w",
        "polarized_yes_rate_100w",
        "relevance_weighted_contra_rate_100w",
    ]

    daily_ideology = daily_by_family(it_df, ideology_metrics)
    daily_ideology.to_csv(tables_dir / "daily_ideology_dominant_by_topic_family.csv", index=False)

    daily_pairs = daily_by_family(it_df, pair_metrics)
    daily_pairs.to_csv(tables_dir / "daily_pair_framing_by_topic_family.csv", index=False)

    daily_meta = daily_by_family(it_df, meta_metrics)
    daily_meta.to_csv(tables_dir / "daily_v4_metadata_by_topic_family.csv", index=False)

    roll_days = int(pol_cfg.get("primary_outcomes", {}).get("rolling_days", 7))
    du.grouped_trailing_daily_rolling(daily_ideology, "topic_family", roll_days).to_csv(
        tables_dir / "rolling_ideology_dominant_by_topic_family.csv", index=False
    )
    du.grouped_trailing_daily_rolling(daily_pairs, "topic_family", roll_days).to_csv(
        tables_dir / "rolling_pair_framing_by_topic_family.csv", index=False
    )
    du.grouped_trailing_daily_rolling(daily_meta, "topic_family", roll_days).to_csv(
        tables_dir / "rolling_v4_metadata_by_topic_family.csv", index=False
    )

    ban_window_table(it_df, launch, ideology_metrics + pair_metrics, "launch").to_csv(
        tables_dir / "ban_windows_launch_primary.csv", index=False
    )
    ban_window_table(it_df, lift, ideology_metrics + pair_metrics, "lift").to_csv(
        tables_dir / "ban_windows_lift_secondary.csv", index=False
    )

    primary_outcomes_w0(it_df, launch).to_csv(tables_dir / "primary_outcomes_launch_w0.csv", index=False)

    if "net_ideology" in it_df.columns and "net_ideology_weighted" in it_df.columns:
        pd.DataFrame(
            [
                {
                    "metric": "net_ideology",
                    "overall_mean": du.weighted_mean(it_df["net_ideology"], it_df["n_words"]),
                },
                {
                    "metric": "net_ideology_weighted",
                    "overall_mean": du.weighted_mean(it_df["net_ideology_weighted"], it_df["n_words"]),
                },
            ]
        ).to_csv(tables_dir / "tie_break_robustness_net_ideology.csv", index=False)

    print(f"[prepare_lexicon_descriptives] wrote tables to {tables_dir}", flush=True)


if __name__ == "__main__":
    main()
