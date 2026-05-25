"""
Script summary:
Aggregate semantic-axis features into DiD-ready panels and validation tables.

Functionality:
- Scans enriched shards for sem_axis_* columns; builds subreddit-day-family panel.
- Writes validation correlations vs net_ideology, seed OOV coverage, held-out axis sanity, and example comments.
- Optionally restricts to comment_in_political_universe when polarization config says so.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_semantic_axis_descriptives.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

READ_COLUMNS = [
    "id",
    "author",
    "subreddit",
    "date_utc",
    "body",
    "n_words",
    "topic_family",
    "primary_lexicon",
    "lang_comment",
    "comment_in_political_universe",
    "net_ideology",
    "pair_framing_net_strict",
    "sem_axis_ideology",
    "sem_axis_emotion",
    "sem_axis_aggression",
    "sem_axis_coverage",
    "has_sem_axis",
]

REQUIRED_FEATURE_COLUMNS = (
    "sem_axis_ideology",
    "sem_axis_emotion",
    "has_sem_axis",
)


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

from scripts.diagnostics.descriptives_util import (  # noqa: E402
    event_dates_from_config,
    weighted_mean,
)
from scripts.features._enriched_shard_runner import read_parquet_shard_safe  # noqa: E402

from src.config_utils import (  # noqa: E402
    load_config,
    load_polarization_config,
    load_semantic_axis_config,
    resolve_primary_subreddits,
    tables_subdir,
)


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare semantic-axis descriptives tables.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--subreddit", type=str, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    return parser.parse_args()


def _in_event_window(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: filter rows to study event window by date_utc."""
    start, end_excl, _, _ = event_dates_from_config(config)
    dates = df["date_utc"].astype(str)
    return df[(dates >= start) & (dates < end_excl)].copy()


def _load_shard_frames(
    interim_dir: Path,
    subs: List[str],
    max_shards: int | None,
) -> pd.DataFrame:
    """Function summary: concatenate selected columns from all shards."""
    chunks: List[pd.DataFrame] = []
    for sub in subs:
        shard_dir = interim_dir / "cleaned_monthly_chunks" / sub
        if not shard_dir.is_dir():
            continue
        shards = sorted(shard_dir.glob("*.parquet"))
        if max_shards:
            shards = shards[:max_shards]
        for shard in shards:
            df = read_parquet_shard_safe(shard)
            if df is None or df.empty:
                continue
            cols = [c for c in READ_COLUMNS if c in df.columns]
            chunks.append(df[cols].copy())
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _panel_aggregate(
    df: pd.DataFrame,
    launch: str,
    pole_thresholds: Dict[str, float],
) -> pd.DataFrame:
    """Function summary: build subreddit x date x topic_family panel with distributional stats."""
    thresholds = {
        "ideology": float(pole_thresholds.get("ideology", 0.25)),
        "emotion": float(pole_thresholds.get("emotion", 0.25)),
        "aggression": float(pole_thresholds.get("aggression", 0.25)),
    }

    def agg_fn(grp: pd.DataFrame) -> pd.Series:
        w = grp["n_words"].astype(float)
        scored = grp[grp["has_sem_axis"].astype(float) > 0]
        out = {
            "n_comments": len(grp),
            "n_scored": len(scored),
            "sem_axis_ideology_mean": weighted_mean(grp["sem_axis_ideology"], w),
            "sem_axis_emotion_mean": weighted_mean(grp["sem_axis_emotion"], w),
            "sem_axis_aggression_mean": weighted_mean(grp["sem_axis_aggression"], w),
            "sem_axis_coverage_mean": weighted_mean(grp["sem_axis_coverage"], w),
            "net_ideology_mean": weighted_mean(grp["net_ideology"], w),
        }
        if len(scored) >= 2:
            out["sem_axis_ideology_var"] = float(scored["sem_axis_ideology"].astype(float).var())
            out["sem_axis_emotion_var"] = float(scored["sem_axis_emotion"].astype(float).var())
        else:
            out["sem_axis_ideology_var"] = float("nan")
            out["sem_axis_emotion_var"] = float("nan")
        for axis, tau in thresholds.items():
            col = f"sem_axis_{axis}"
            if col not in grp.columns:
                continue
            vals = grp[col].astype(float)
            out[f"{col}_share_above_pos"] = float((vals > tau).mean())
            out[f"{col}_share_below_neg"] = float((vals < -tau).mean())
        return pd.Series(out)

    records: List[Dict[str, Any]] = []
    for keys, grp in df.groupby(["subreddit", "date_utc", "topic_family"], sort=True):
        row = agg_fn(grp).to_dict()
        row["subreddit"], row["date_utc"], row["topic_family"] = keys
        records.append(row)
    panel = pd.DataFrame(records)
    panel["post"] = (panel["date_utc"].astype(str) >= launch).astype(int)
    return panel


def _write_seed_validation_tables(
    config: Dict[str, Any],
    sem_cfg: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Function summary: seed OOV coverage and held-out axis sanity (requires fastText models)."""
    from src.embeddings import held_out_axis_sanity_report, seed_coverage_report

    try:
        coverage_rows: List[Dict[str, Any]] = []
        sanity_rows: List[Dict[str, Any]] = []
        for lang in ("it", "en", "de"):
            coverage_rows.extend(
                seed_coverage_report(lang, PROJECT_ROOT, sem_cfg, config=config)
            )
            sanity_rows.extend(
                held_out_axis_sanity_report(lang, PROJECT_ROOT, sem_cfg, config=config)
            )
        pd.DataFrame(coverage_rows).to_csv(out_dir / "semantic_axis_seed_coverage.csv", index=False)
        pd.DataFrame(sanity_rows).to_csv(out_dir / "semantic_axis_axis_sanity.csv", index=False)
        print(
            f"[prepare_semantic_axis_descriptives] seed validation rows="
            f"{len(coverage_rows)} sanity={len(sanity_rows)}",
            flush=True,
        )
    except FileNotFoundError as exc:
        print(
            f"[prepare_semantic_axis_descriptives] skip seed validation (models missing): {exc}",
            flush=True,
        )


def _validation_tables(
    df: pd.DataFrame,
    config: Dict[str, Any],
    sem_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Function summary: correlation and coverage validation by primary_lexicon language."""
    from src.embeddings import seed_oov_summary_by_lang

    rows: List[Dict[str, Any]] = []
    for lang, grp in df.groupby(df["primary_lexicon"].astype(str)):
        scored = grp[grp["has_sem_axis"].astype(float) > 0]
        if len(scored) < 10:
            continue
        r_comment = float(scored["sem_axis_ideology"].corr(scored["net_ideology"]))
        r_pair = float("nan")
        if "pair_framing_net_strict" in scored.columns and lang.lower() == "it":
            r_pair = float(scored["sem_axis_ideology"].corr(scored["pair_framing_net_strict"]))
        daily = (
            scored.groupby(["subreddit", "date_utc"], as_index=False)
            .agg(
                sem_axis_ideology_mean=("sem_axis_ideology", "mean"),
                net_ideology_mean=("net_ideology", "mean"),
            )
        )
        r_day = float(daily["sem_axis_ideology_mean"].corr(daily["net_ideology_mean"])) if len(daily) >= 5 else float("nan")
        row: Dict[str, Any] = {
            "lang": lang,
            "n_comments": len(scored),
            "corr_ideology_comment": r_comment,
            "corr_ideology_subreddit_day": r_day,
            "corr_pair_framing_strict_it": r_pair,
            "coverage_mean": float(scored["sem_axis_coverage"].mean()),
            "coverage_p10": float(scored["sem_axis_coverage"].quantile(0.1)),
            "coverage_p50": float(scored["sem_axis_coverage"].quantile(0.5)),
        }
        try:
            row.update(seed_oov_summary_by_lang(str(lang), PROJECT_ROOT, sem_cfg, config=config))
        except FileNotFoundError:
            row["seed_oov_share_ideology"] = float("nan")
            row["seed_oov_share_emotion"] = float("nan")
            row["seed_oov_share_aggression"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _examples_table(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Function summary: top/bottom comments per axis per language for eyeball validation."""
    rows: List[Dict[str, Any]] = []
    axes = ("sem_axis_ideology", "sem_axis_emotion", "sem_axis_aggression")
    for lang, lang_df in df.groupby(df["primary_lexicon"].astype(str)):
        scored = lang_df[lang_df["has_sem_axis"].astype(float) > 0]
        if scored.empty:
            continue
        for axis in axes:
            if axis not in scored.columns:
                continue
            for label, part in (("high", scored.nlargest(n, axis)), ("low", scored.nsmallest(n, axis))):
                for _, r in part.iterrows():
                    rows.append(
                        {
                            "lang": lang,
                            "axis": axis,
                            "extreme": label,
                            "score": float(r[axis]),
                            "net_ideology": float(r.get("net_ideology", 0)),
                            "subreddit": r["subreddit"],
                            "date_utc": r["date_utc"],
                            "body": str(r.get("body", ""))[:500],
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    """Function summary: CLI entry for semantic-axis descriptives preparation."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    pol_cfg = load_polarization_config(config)
    sem_cfg = load_semantic_axis_config(config)
    pole_thresholds = sem_cfg.get("pole_thresholds") or {}
    interim_dir = Path(config["paths"]["interim_dir"])
    if not interim_dir.is_absolute():
        interim_dir = PROJECT_ROOT / interim_dir
    out_dir = tables_subdir(config, "semantic_axis")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_seed_validation_tables(config, sem_cfg, out_dir)

    subs = [args.subreddit] if args.subreddit else resolve_primary_subreddits(config)
    df = _load_shard_frames(interim_dir, subs, args.max_shards)
    if df.empty:
        print("[prepare_semantic_axis_descriptives] no shard data; seed tables only", flush=True)
        return
    missing = [c for c in REQUIRED_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing semantic-axis columns {missing}; run compute_semantic_axis_features.py first."
        )
    df = _in_event_window(df, config)
    if pol_cfg.get("restrict_to_political_comments") and "comment_in_political_universe" in df.columns:
        df = df[df["comment_in_political_universe"].astype(bool)].copy()

    _, _, launch, _ = event_dates_from_config(config)
    panel = _panel_aggregate(df, launch, pole_thresholds)
    panel.to_csv(out_dir / "semantic_axis_panel.csv", index=False)

    val_df = _validation_tables(df, config, sem_cfg)
    val_df.to_csv(out_dir / "semantic_axis_validation.csv", index=False)
    _examples_table(df).to_csv(out_dir / "semantic_axis_examples.csv", index=False)
    print(f"[prepare_semantic_axis_descriptives] wrote tables to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
