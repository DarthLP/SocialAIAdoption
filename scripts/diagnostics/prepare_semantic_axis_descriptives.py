"""
Script summary:
Aggregate semantic-axis features into DiD-ready panels and validation tables.

Functionality:
- Scans enriched shards for sem_axis_* columns; builds panels at forum, topic_family,
  topic, language, and language×political-universe levels.
- Pole shares: per-lexicon absolute cutoffs and percentile buckets (p10/p90); share_unscored.
- DiD inference should use sem_axis_*_mean (within-language); not cross-language pole levels.
- Time bins: 1d calendar days; 3d and 7d launch-aligned blocks from launch_day_utc.
- Shard streaming with parquet column projection (avoids loading body / full schema).

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_semantic_axis_descriptives.py --config config/italy_polarization_setup.yaml

Fast path (low RAM, no fastText):
  .venv/bin/python scripts/diagnostics/prepare_semantic_axis_descriptives.py \\
    --config config/italy_polarization_setup.yaml --panels-only

Subset of time bins:
  ... --panels-only --bin-days 1
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd

# Columns for panel build + optional validation (no comment body).
PANEL_COLUMNS: Tuple[str, ...] = (
    "id",
    "subreddit",
    "date_utc",
    "n_words",
    "topic",
    "topic_family",
    "primary_lexicon",
    "comment_in_political_universe",
    "net_ideology",
    "sem_axis_ideology",
    "sem_axis_emotion",
    "sem_axis_aggression",
    "sem_axis_coverage",
    "has_sem_axis",
)

VALIDATION_EXTRA_COLUMNS: Tuple[str, ...] = ("pair_framing_net_strict",)

EXAMPLE_EXTRA_COLUMNS: Tuple[str, ...] = ("body",)

READ_COLUMNS = list(PANEL_COLUMNS) + list(VALIDATION_EXTRA_COLUMNS) + list(EXAMPLE_EXTRA_COLUMNS)

REQUIRED_FEATURE_COLUMNS = (
    "sem_axis_ideology",
    "sem_axis_emotion",
    "has_sem_axis",
)

UNIVERSE_SLICE_IN = "in_political_tree"
UNIVERSE_SLICE_OUT = "out_political_tree"

SEMANTIC_AXES = ("ideology", "emotion", "aggression")

WEIGHTED_MEAN_COLS = (
    "sem_axis_ideology",
    "sem_axis_emotion",
    "sem_axis_aggression",
    "sem_axis_coverage",
    "net_ideology",
)

# panel_level -> (group_keys, file_slug, carry_first_columns)
PANEL_SPECS: Dict[str, Tuple[List[str], str, Tuple[str, ...]]] = {
    "forum": (["subreddit"], "by_forum", ("topic", "topic_family", "primary_lexicon")),
    "topic_family": (["topic_family"], "by_topic_family", ()),
    "topic": (["topic"], "by_topic", ()),
    "language": (["primary_lexicon"], "by_language", ()),
    "language_universe": (
        ["primary_lexicon", "universe_slice"],
        "by_language_universe",
        (),
    ),
}


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
    assign_period_start,
    event_dates_from_config,
    weighted_mean,
)
from src.circumvention import (  # noqa: E402
    attach_italy_circumvention_columns,
    build_circumvention_geo_panel,
    enrich_daily_with_transforms,
    italy_circumvention_by_period,
    load_circumvention_daily,
)
from src.config_utils import (  # noqa: E402
    load_config,
    load_semantic_axis_config,
    resolve_primary_subreddits,
    subreddit_topic_map,
    tables_subdir,
)
from src.political_lexicon import esteban_ray_index  # noqa: E402
from src.semantic_axis_stats import (  # noqa: E402
    PoleBucketSpec,
    absolute_threshold,
    build_pole_bucket_specs,
    calibrate_lexicon_percentiles,
    group_primary_lexicon,
    ideology_orientation_report,
    percentile_lookup_from_csv,
    pole_column_prefix,
)

POLE_SHARE_EPS = 1e-9


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare semantic-axis descriptives tables.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--subreddit", type=str, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument(
        "--skip-seed-validation",
        action="store_true",
        help="Skip fastText seed tables (use validate_semantic_axis_seeds.py).",
    )
    parser.add_argument(
        "--skip-examples",
        action="store_true",
        help="Skip semantic_axis_examples.csv (avoids loading comment body).",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip semantic_axis_validation.csv.",
    )
    parser.add_argument(
        "--panels-only",
        action="store_true",
        help="Only write panel CSVs (implies skip seed, validation, examples).",
    )
    parser.add_argument(
        "--bin-days",
        type=str,
        default=None,
        help="Comma-separated bin sizes (e.g. 1,3,7); default from semantic_axis.panel_bin_days.",
    )
    return parser.parse_args()


def _resolve_panel_bin_days(args: argparse.Namespace, sem_cfg: Dict[str, Any]) -> List[int]:
    """Function summary: parse --bin-days or fall back to config list."""
    if args.bin_days:
        return [int(x.strip()) for x in str(args.bin_days).split(",") if x.strip()]
    return [int(x) for x in (sem_cfg.get("panel_bin_days") or [1, 3, 7])]


def _columns_to_read(
    *,
    include_validation: bool,
    include_examples: bool,
) -> Tuple[str, ...]:
    """Function summary: build parquet column list for shard reads."""
    cols = list(PANEL_COLUMNS)
    if include_validation:
        cols.extend(VALIDATION_EXTRA_COLUMNS)
    if include_examples:
        cols.extend(EXAMPLE_EXTRA_COLUMNS)
    return tuple(dict.fromkeys(cols))


def _read_shard_projected(shard: Path, columns: Sequence[str]) -> pd.DataFrame | None:
    """Function summary: read parquet with column projection; skip corrupt/empty files.

    Parameters:
    - shard: path to monthly parquet.
    - columns: desired columns (intersected with file schema).

    Returns:
    - DataFrame or None.
    """
    if not shard.is_file() or shard.stat().st_size < 8:
        return None
    try:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(shard).schema.names)
        use_cols = [c for c in columns if c in available]
        if not use_cols:
            return None
        return pd.read_parquet(shard, columns=use_cols)
    except Exception:
        try:
            df = pd.read_parquet(shard)
            if df is None or df.empty:
                return None
            use_cols = [c for c in columns if c in df.columns]
            return df[use_cols].copy()
        except Exception:
            return None


def _universe_slice_label(in_universe: bool) -> str:
    """Function summary: map comment_in_political_universe to slice id."""
    return UNIVERSE_SLICE_IN if in_universe else UNIVERSE_SLICE_OUT


def _enrich_comment_frame(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: ensure topic and universe_slice columns on comment-level frame."""
    out = df.copy()
    if "topic" not in out.columns or out["topic"].isna().all():
        topic_map = subreddit_topic_map(config, include_topic_aliases=False)
        out["topic"] = out["subreddit"].astype(str).map(topic_map)
    if "comment_in_political_universe" in out.columns:
        out["universe_slice"] = out["comment_in_political_universe"].astype(bool).map(_universe_slice_label)
    else:
        out["universe_slice"] = UNIVERSE_SLICE_OUT
    return out


def _init_pole_counters(bucket_specs: Sequence[PoleBucketSpec]) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    """Function summary: empty pole bucket counters keyed by (axis, label, suffix)."""
    pole: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for spec in bucket_specs:
        if spec.kind == "symmetric_absolute":
            pole[(spec.axis, spec.high_label, spec.suffix)] = {"n_comments": 0, "n_words": 0.0}
            pole[(spec.axis, spec.low_label, spec.suffix)] = {"n_comments": 0, "n_words": 0.0}
        elif spec.kind == "high_percentile":
            pole[(spec.axis, spec.high_label, spec.suffix)] = {"n_comments": 0, "n_words": 0.0}
        elif spec.kind == "low_percentile":
            pole[(spec.axis, spec.low_label, spec.suffix)] = {"n_comments": 0, "n_words": 0.0}
    return pole


def _new_agg_accumulator(bucket_specs: Sequence[PoleBucketSpec]) -> Dict[str, Any]:
    """Function summary: empty mutable state for merging shard partial aggregates."""
    return {
        "n_comments": 0,
        "n_scored": 0,
        "n_words_total": 0.0,
        "carry_first": {},
        "w_sums": {c: 0.0 for c in WEIGHTED_MEAN_COLS},
        "wx_sums": {c: 0.0 for c in WEIGHTED_MEAN_COLS},
        "ideology_vals": [],
        "emotion_vals": [],
        "pole": _init_pole_counters(bucket_specs),
        "calendar_days": set(),
    }


def _accumulate_group(
    acc: Dict[str, Any],
    grp: pd.DataFrame,
    bucket_specs: Sequence[PoleBucketSpec],
    sem_cfg: Dict[str, Any],
    percentile_lookup: Mapping[Tuple[str, str, int], float],
) -> None:
    """Function summary: merge one comment group into accumulator state."""
    w = grp["n_words"].astype(float)
    n_comments = len(grp)
    scored = grp[grp["has_sem_axis"].astype(float) > 0]
    acc["n_comments"] += n_comments
    acc["n_scored"] += len(scored)
    acc["n_words_total"] += float(w.sum())
    for col in WEIGHTED_MEAN_COLS:
        if col not in grp.columns:
            continue
        vals = grp[col].astype(float)
        acc["w_sums"][col] += float(w.sum())
        acc["wx_sums"][col] += float((vals * w).sum())
    if len(scored) >= 1:
        acc["ideology_vals"].extend(scored["sem_axis_ideology"].astype(float).tolist())
        acc["emotion_vals"].extend(scored["sem_axis_emotion"].astype(float).tolist())
    if "date_utc" in grp.columns:
        acc["calendar_days"].update(grp["date_utc"].astype(str).unique())

    lex = group_primary_lexicon(grp)
    for spec in bucket_specs:
        col = pole_column_prefix(spec.axis)
        if col not in grp.columns:
            continue
        vals = grp[col].astype(float)
        nw = w
        if spec.kind == "symmetric_absolute":
            thr = absolute_threshold(sem_cfg, lex, spec.axis)
            high_mask = vals > thr
            low_mask = vals < -thr
            hk = (spec.axis, spec.high_label, spec.suffix)
            lk = (spec.axis, spec.low_label, spec.suffix)
            acc["pole"][hk]["n_comments"] += int(high_mask.sum())
            acc["pole"][lk]["n_comments"] += int(low_mask.sum())
            acc["pole"][hk]["n_words"] += float(nw[high_mask].sum())
            acc["pole"][lk]["n_words"] += float(nw[low_mask].sum())
        elif spec.kind == "high_percentile":
            pct = int(spec.suffix.replace("above_p", ""))
            thr = percentile_lookup.get((str(lex).lower(), spec.axis, pct), float("nan"))
            if thr != thr:
                continue
            high_mask = vals > thr
            hk = (spec.axis, spec.high_label, spec.suffix)
            acc["pole"][hk]["n_comments"] += int(high_mask.sum())
            acc["pole"][hk]["n_words"] += float(nw[high_mask].sum())
        elif spec.kind == "low_percentile":
            pct = int(spec.suffix.replace("below_p", ""))
            thr = percentile_lookup.get((str(lex).lower(), spec.axis, pct), float("nan"))
            if thr != thr:
                continue
            low_mask = vals < thr
            lk = (spec.axis, spec.low_label, spec.suffix)
            acc["pole"][lk]["n_comments"] += int(low_mask.sum())
            acc["pole"][lk]["n_words"] += float(nw[low_mask].sum())


def _ideology_bucket_metric(out: Dict[str, Any], label: str, field: str) -> float | None:
    """Function summary: read ideology pole share or n_comments; prefer abs then tau25 columns."""
    for suffix in ("abs", "tau25"):
        if field == "share":
            key = f"sem_axis_ideology_share_{label}_{suffix}"
        else:
            key = f"sem_axis_ideology_n_comments_{label}_{suffix}"
        if key not in out:
            continue
        val = out[key]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        return float(val)
    return None


def _append_ideology_derived_metrics(out: Dict[str, Any]) -> None:
    """Function summary: add sem_axis_ideology_pole_share and sem_axis_ideology_esteban_ray to panel row."""
    n_comments = int(out.get("n_comments") or 0)
    share_unscored = float(out.get("share_unscored") or 0.0)
    if np.isnan(share_unscored):
        share_unscored = 0.0
    left_s = _ideology_bucket_metric(out, "left", "share")
    right_s = _ideology_bucket_metric(out, "right", "share")
    if left_s is not None and right_s is not None:
        center_s = max(0.0, 1.0 - left_s - right_s - share_unscored)
        denom = left_s + right_s + center_s + POLE_SHARE_EPS
        out["sem_axis_ideology_pole_share"] = float((left_s + right_s) / denom) if denom > 0 else float("nan")
    else:
        out["sem_axis_ideology_pole_share"] = float("nan")
    n_left = _ideology_bucket_metric(out, "left", "n_comments")
    n_right = _ideology_bucket_metric(out, "right", "n_comments")
    if n_left is not None and n_right is not None:
        n_scored = int(round(n_comments * (1.0 - share_unscored))) if n_comments else 0
        n_center = max(0, n_scored - int(n_left) - int(n_right))
        out["sem_axis_ideology_esteban_ray"] = float(
            esteban_ray_index(float(n_left), float(n_center), float(n_right))
        )
    else:
        out["sem_axis_ideology_esteban_ray"] = float("nan")


def _finalize_accumulator(
    acc: Dict[str, Any],
    bucket_specs: Sequence[PoleBucketSpec],
    bin_days: int = 1,
) -> Dict[str, Any]:
    """Function summary: convert accumulator state to panel row metrics dict."""
    n_comments = acc["n_comments"]
    n_scored = acc["n_scored"]
    out: Dict[str, Any] = {
        "n_comments": n_comments,
        "n_scored": n_scored,
        "n_words_total": acc["n_words_total"],
        "share_unscored": (
            float(n_comments - n_scored) / float(n_comments) if n_comments else float("nan")
        ),
    }
    mean_names = {
        "sem_axis_ideology": "sem_axis_ideology_mean",
        "sem_axis_emotion": "sem_axis_emotion_mean",
        "sem_axis_aggression": "sem_axis_aggression_mean",
        "sem_axis_coverage": "sem_axis_coverage_mean",
        "net_ideology": "net_ideology_mean",
    }
    for col, out_name in mean_names.items():
        ws = acc["w_sums"].get(col, 0.0)
        out[out_name] = acc["wx_sums"][col] / ws if ws > 0 else float("nan")
    ideo = acc["ideology_vals"]
    emo = acc["emotion_vals"]
    out["sem_axis_ideology_var"] = float(np.var(ideo)) if len(ideo) >= 2 else float("nan")
    out["sem_axis_emotion_var"] = float(np.var(emo)) if len(emo) >= 2 else float("nan")
    for spec in bucket_specs:
        axis = spec.axis
        if spec.kind == "symmetric_absolute":
            for label in (spec.high_label, spec.low_label):
                key = (axis, label, spec.suffix)
                if key not in acc["pole"]:
                    continue
                cnt = int(acc["pole"][key]["n_comments"])
                out[f"sem_axis_{axis}_share_{label}_{spec.suffix}"] = (
                    float(cnt) / float(n_comments) if n_comments else float("nan")
                )
                out[f"sem_axis_{axis}_n_comments_{label}_{spec.suffix}"] = cnt
                out[f"sem_axis_{axis}_n_words_{label}_{spec.suffix}"] = acc["pole"][key]["n_words"]
        elif spec.kind == "high_percentile":
            key = (axis, spec.high_label, spec.suffix)
            if key in acc["pole"]:
                cnt = int(acc["pole"][key]["n_comments"])
                out[f"sem_axis_{axis}_share_{spec.high_label}_{spec.suffix}"] = (
                    float(cnt) / float(n_comments) if n_comments else float("nan")
                )
                out[f"sem_axis_{axis}_n_comments_{spec.high_label}_{spec.suffix}"] = cnt
                out[f"sem_axis_{axis}_n_words_{spec.high_label}_{spec.suffix}"] = acc["pole"][key][
                    "n_words"
                ]
        elif spec.kind == "low_percentile":
            key = (axis, spec.low_label, spec.suffix)
            if key in acc["pole"]:
                cnt = int(acc["pole"][key]["n_comments"])
                out[f"sem_axis_{axis}_share_{spec.low_label}_{spec.suffix}"] = (
                    float(cnt) / float(n_comments) if n_comments else float("nan")
                )
                out[f"sem_axis_{axis}_n_comments_{spec.low_label}_{spec.suffix}"] = cnt
                out[f"sem_axis_{axis}_n_words_{spec.low_label}_{spec.suffix}"] = acc["pole"][key][
                    "n_words"
                ]
    n_days = len(acc.get("calendar_days") or ())
    if n_days <= 0:
        n_days = 1
    out["n_days_in_bin"] = int(n_days)
    out["is_partial_bin"] = bool(int(bin_days) > 1 and n_days < int(bin_days))
    _append_ideology_derived_metrics(out)
    return out


def _semantic_axis_agg(
    grp: pd.DataFrame,
    bucket_specs: Sequence[PoleBucketSpec],
    sem_cfg: Dict[str, Any],
    percentile_lookup: Mapping[Tuple[str, str, int], float],
    bin_days: int = 1,
) -> Dict[str, Any]:
    """Function summary: aggregate semantic-axis means and pole-bucket shares/counts for one group."""
    acc = _new_agg_accumulator(bucket_specs)
    _accumulate_group(acc, grp, bucket_specs, sem_cfg, percentile_lookup)
    return _finalize_accumulator(acc, bucket_specs, bin_days=int(bin_days))


def _build_panel(
    df: pd.DataFrame,
    panel_level: str,
    group_keys: List[str],
    carry_first: Tuple[str, ...],
    bin_days: int,
    launch: str,
    bucket_specs: Sequence[PoleBucketSpec],
    sem_cfg: Dict[str, Any],
    percentile_lookup: Mapping[Tuple[str, str, int], float],
    parent_counts: Dict[Tuple[str, ...], int] | None = None,
) -> pd.DataFrame:
    """Function summary: aggregate comment frame to one panel table."""
    keys = group_keys + ["period_start"]
    records: List[Dict[str, Any]] = []
    for key_vals, grp in df.groupby(keys, sort=True):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        row = _semantic_axis_agg(grp, bucket_specs, sem_cfg, percentile_lookup, bin_days=int(bin_days))
        row["panel_level"] = panel_level
        row["bin_days"] = int(bin_days)
        for col, val in zip(keys, key_vals):
            row[col] = val
        for col in carry_first:
            if col in grp.columns:
                row[col] = grp[col].iloc[0]
        row["post"] = int(str(row["period_start"]) >= launch)
        if parent_counts is not None and panel_level == "language_universe":
            parent_key = (str(row["primary_lexicon"]), str(row["period_start"]))
            denom = parent_counts.get(parent_key, 0)
            row["share_of_cell_comments"] = (
                float(row["n_comments"]) / float(denom) if denom > 0 else float("nan")
            )
        records.append(row)
    return pd.DataFrame(records)


def _parent_counts_language(df: pd.DataFrame) -> Dict[Tuple[str, ...], int]:
    """Function summary: comment counts per (primary_lexicon, period_start) for universe shares."""
    counts: Dict[Tuple[str, ...], int] = {}
    for key, grp in df.groupby(["primary_lexicon", "period_start"], sort=True):
        counts[key if isinstance(key, tuple) else (key,)] = len(grp)
    return counts


def _prepare_binned_frame(df: pd.DataFrame, bin_days: int, launch: str) -> pd.DataFrame:
    """Function summary: attach period_start for a given bin size."""
    out = df.copy()
    out["period_start"] = assign_period_start(out["date_utc"], bin_days, launch)
    return out


def _italy_circumvention_lookup(
    config: Dict[str, Any],
    launch: str,
    panel_bin_days: Sequence[int],
) -> Dict[int, pd.DataFrame]:
    """Function summary: load circumvention data and build IT period lookup per bin size."""
    start, end_excl, _, _ = event_dates_from_config(config)
    try:
        daily = load_circumvention_daily(PROJECT_ROOT, config, start=start, end_exclusive=end_excl)
        daily = enrich_daily_with_transforms(daily)
    except FileNotFoundError as exc:
        print(
            f"[prepare_semantic_axis_descriptives] skip IT VPN join (circumvention missing): {exc}",
            flush=True,
        )
        return {}
    lookup: Dict[int, pd.DataFrame] = {}
    for bin_days in panel_bin_days:
        geo_panel = build_circumvention_geo_panel(
            daily, launch, int(bin_days), assign_period_start=assign_period_start
        )
        lookup[int(bin_days)] = italy_circumvention_by_period(geo_panel)
    return lookup


def _panel_accumulators_finalize(
    accum: MutableMapping[int, MutableMapping[str, MutableMapping[Any, Dict[str, Any]]]],
    bucket_specs: Sequence[PoleBucketSpec],
    launch: str,
    panel_bin_days: Sequence[int],
    parent_lang_counts: Dict[int, Dict[Tuple[str, ...], int]],
) -> Dict[Tuple[int, str], pd.DataFrame]:
    """Function summary: turn nested accumulators into panel DataFrames per bin_days and level."""
    out: Dict[Tuple[int, str], pd.DataFrame] = {}
    for bin_days in panel_bin_days:
        parent_counts = parent_lang_counts.get(int(bin_days), {})
        for panel_level, (group_keys, slug, carry_first) in PANEL_SPECS.items():
            records: List[Dict[str, Any]] = []
            level_acc = accum.get(int(bin_days), {}).get(panel_level, {})
            for key_vals, acc in sorted(level_acc.items(), key=lambda x: str(x[0])):
                if not isinstance(key_vals, tuple):
                    key_vals = (key_vals,)
                row = _finalize_accumulator(acc, bucket_specs, bin_days=int(bin_days))
                row["panel_level"] = panel_level
                row["bin_days"] = int(bin_days)
                for col, val in zip(group_keys + ["period_start"], key_vals):
                    row[col] = val
                for col in carry_first:
                    if col in acc.get("carry_first", {}):
                        row[col] = acc["carry_first"][col]
                row["post"] = int(str(row["period_start"]) >= launch)
                if panel_level == "language_universe":
                    parent_key = (str(row["primary_lexicon"]), str(row["period_start"]))
                    denom = parent_counts.get(parent_key, 0)
                    row["share_of_cell_comments"] = (
                        float(row["n_comments"]) / float(denom) if denom > 0 else float("nan")
                    )
                records.append(row)
            out[(int(bin_days), slug)] = pd.DataFrame(records)
    return out


def _accumulate_shard_into_panels(
    chunk: pd.DataFrame,
    accum: MutableMapping[int, MutableMapping[str, MutableMapping[Any, Dict[str, Any]]]],
    bucket_specs: Sequence[PoleBucketSpec],
    sem_cfg: Dict[str, Any],
    percentile_lookup: Mapping[Tuple[str, str, int], float],
    panel_bin_days: Sequence[int],
    launch: str,
    parent_lang_counts: Dict[int, Dict[Tuple[str, ...], int]],
) -> None:
    """Function summary: update panel accumulators from one in-window shard chunk."""
    for bin_days in panel_bin_days:
        binned = _prepare_binned_frame(chunk, int(bin_days), launch)
        for pl, pc in binned.groupby(["primary_lexicon", "period_start"], sort=False):
            parent_lang_counts.setdefault(int(bin_days), {})
            key = pl if isinstance(pl, tuple) else (pl,)
            parent_lang_counts[int(bin_days)][key] = parent_lang_counts[int(bin_days)].get(key, 0) + len(pc)

        for panel_level, (group_keys, _slug, carry_first) in PANEL_SPECS.items():
            keys = group_keys + ["period_start"]
            level_store = accum.setdefault(int(bin_days), {}).setdefault(panel_level, {})
            for key_vals, grp in binned.groupby(keys, sort=False):
                if not isinstance(key_vals, tuple):
                    key_vals = (key_vals,)
                if key_vals not in level_store:
                    level_store[key_vals] = _new_agg_accumulator(bucket_specs)
                    for col in carry_first:
                        if col in grp.columns:
                            level_store[key_vals]["carry_first"][col] = grp[col].iloc[0]
                _accumulate_group(
                    level_store[key_vals],
                    grp,
                    bucket_specs,
                    sem_cfg,
                    percentile_lookup,
                )


def _write_panels_from_accumulators(
    accum: MutableMapping[int, MutableMapping[str, MutableMapping[Any, Dict[str, Any]]]],
    launch: str,
    bucket_specs: Sequence[PoleBucketSpec],
    panel_bin_days: Sequence[int],
    out_dir: Path,
    config: Dict[str, Any],
    parent_lang_counts: Dict[int, Dict[Tuple[str, ...], int]],
) -> None:
    """Function summary: finalize accumulators, attach VPN, write panel CSVs."""
    italy_lookup = _italy_circumvention_lookup(config, launch, panel_bin_days)
    panels = _panel_accumulators_finalize(
        accum, bucket_specs, launch, panel_bin_days, parent_lang_counts
    )
    for bin_days in panel_bin_days:
        for panel_level, (_group_keys, slug, _carry) in PANEL_SPECS.items():
            panel = panels.get((int(bin_days), slug), pd.DataFrame())
            if panel.empty:
                continue
            italy_period = italy_lookup.get(int(bin_days))
            if italy_period is not None and not italy_period.empty:
                panel = attach_italy_circumvention_columns(
                    panel, italy_period, panel_level=panel_level
                )
            path = out_dir / f"semantic_axis_panel_{slug}_{int(bin_days)}d.csv"
            panel.to_csv(path, index=False)
            if slug == "by_forum" and int(bin_days) == 1:
                alias = out_dir / "semantic_axis_panel.csv"
                panel.to_csv(alias, index=False)
            print(
                f"[prepare_semantic_axis_descriptives] panel_level={panel_level} "
                f"bin_days={bin_days} rows={len(panel)} -> {path.name}",
                flush=True,
            )
    return None


def _iter_shards(interim_dir: Path, subs: List[str], max_shards: int | None) -> List[Path]:
    """Function summary: list parquet shard paths for primary subreddits."""
    paths: List[Path] = []
    for sub in subs:
        shard_dir = interim_dir / "cleaned_monthly_chunks" / sub
        if not shard_dir.is_dir():
            continue
        shards = sorted(shard_dir.glob("*.parquet"))
        if max_shards:
            shards = shards[: max_shards]
        paths.extend(shards)
    return paths


def _in_event_window(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: filter rows to study event window by date_utc."""
    start, end_excl, _, _ = event_dates_from_config(config)
    dates = df["date_utc"].astype(str)
    return df[(dates >= start) & (dates < end_excl)].copy()


def _stream_shards_for_panels(
    interim_dir: Path,
    subs: List[str],
    max_shards: int | None,
    read_columns: Sequence[str],
    config: Dict[str, Any],
    bucket_specs: Sequence[PoleBucketSpec],
    sem_cfg: Dict[str, Any],
    percentile_lookup: Mapping[Tuple[str, str, int], float],
    panel_bin_days: Sequence[int],
    launch: str,
) -> Tuple[
    MutableMapping[int, MutableMapping[str, MutableMapping[Any, Dict[str, Any]]]],
    Dict[int, Dict[Tuple[str, ...], int]],
    bool,
]:
    """Function summary: stream shards into panel accumulators without full concat.

    Returns:
    - accumulators, parent_lang_counts, found_any_rows
    """
    accum: Dict[int, Dict[str, Dict[Any, Dict[str, Any]]]] = {}
    parent_lang_counts: Dict[int, Dict[Tuple[str, ...], int]] = {}
    found = False
    skipped_no_semaxis = 0
    shards = _iter_shards(interim_dir, subs, max_shards)
    n_shards = len(shards)
    for i, shard in enumerate(shards, start=1):
        df = _read_shard_projected(shard, read_columns)
        if df is None or df.empty:
            continue
        missing = [c for c in REQUIRED_FEATURE_COLUMNS if c not in df.columns]
        if missing:
            skipped_no_semaxis += 1
            continue
        df = _in_event_window(df, config)
        if df.empty:
            continue
        df = _enrich_comment_frame(df, config)
        _accumulate_shard_into_panels(
            df,
            accum,
            bucket_specs,
            sem_cfg,
            percentile_lookup,
            panel_bin_days,
            launch,
            parent_lang_counts,
        )
        found = True
        if i % 20 == 0 or i == n_shards:
            print(
                f"[prepare_semantic_axis_descriptives] shards {i}/{n_shards} "
                f"used={found} skipped_no_semaxis={skipped_no_semaxis}",
                flush=True,
            )
    if skipped_no_semaxis:
        print(
            f"[prepare_semantic_axis_descriptives] skipped {skipped_no_semaxis} shards "
            "without sem_axis_* (not enriched for semantic axis; expected for some forums)",
            flush=True,
        )
    if not found:
        raise RuntimeError(
            "No shards with sem_axis_* columns in the event window; "
            "run compute_semantic_axis_features.py on enriched shards first."
        )
    return accum, parent_lang_counts, found


def _load_shard_frames_concat(
    interim_dir: Path,
    subs: List[str],
    max_shards: int | None,
    read_columns: Sequence[str],
) -> pd.DataFrame:
    """Function summary: concatenate projected columns from all shards (for validation/examples)."""
    chunks: List[pd.DataFrame] = []
    for shard in _iter_shards(interim_dir, subs, max_shards):
        df = _read_shard_projected(shard, read_columns)
        if df is None or df.empty:
            continue
        missing = [c for c in REQUIRED_FEATURE_COLUMNS if c not in df.columns]
        if missing:
            continue
        chunks.append(df)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _write_seed_validation_tables(
    config: Dict[str, Any],
    sem_cfg: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Function summary: seed OOV coverage and held-out axis sanity (requires fastText models)."""
    from src.embeddings import (
        ensure_exclusive_vector_lang,
        held_out_axis_sanity_report,
        seed_coverage_report,
    )

    try:
        coverage_rows: List[Dict[str, Any]] = []
        sanity_rows: List[Dict[str, Any]] = []
        for lang in ("it", "en", "de"):
            ensure_exclusive_vector_lang(lang, sem_cfg)
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
    *,
    include_seed_oov: bool,
) -> pd.DataFrame:
    """Function summary: correlation and coverage validation by primary_lexicon language."""
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
        r_day = (
            float(daily["sem_axis_ideology_mean"].corr(daily["net_ideology_mean"]))
            if len(daily) >= 5
            else float("nan")
        )
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
        if include_seed_oov:
            from src.embeddings import seed_oov_summary_by_lang

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
    if args.panels_only:
        args.skip_seed_validation = True
        args.skip_examples = True
        args.skip_validation = True

    config = load_config(PROJECT_ROOT / args.config)
    sem_cfg = load_semantic_axis_config(config)
    bucket_specs = build_pole_bucket_specs(sem_cfg)
    panel_bin_days = _resolve_panel_bin_days(args, sem_cfg)
    interim_dir = Path(config["paths"]["interim_dir"])
    if not interim_dir.is_absolute():
        interim_dir = PROJECT_ROOT / interim_dir
    out_dir = tables_subdir(config, "semantic_axis")
    out_dir.mkdir(parents=True, exist_ok=True)

    subs = [args.subreddit] if args.subreddit else resolve_primary_subreddits(config)
    _, _, launch, _ = event_dates_from_config(config)

    panel_read_cols = _columns_to_read(include_validation=False, include_examples=False)
    cal_path = out_dir / "semantic_axis_lexicon_percentile_thresholds.csv"
    shard_paths = _iter_shards(interim_dir, subs, args.max_shards)
    cal_df = calibrate_lexicon_percentiles(
        shard_paths,
        panel_read_cols,
        sem_cfg,
        read_shard_fn=lambda p, c: _read_shard_projected(p, c),
    )
    if not cal_df.empty:
        cal_df.to_csv(cal_path, index=False)
        print(
            f"[prepare_semantic_axis_descriptives] wrote {cal_path.name} rows={len(cal_df)}",
            flush=True,
        )
    percentile_lookup = percentile_lookup_from_csv(cal_path)

    accum, parent_lang_counts, found = _stream_shards_for_panels(
        interim_dir,
        subs,
        args.max_shards,
        panel_read_cols,
        config,
        bucket_specs,
        sem_cfg,
        percentile_lookup,
        panel_bin_days,
        launch,
    )
    if not found:
        print("[prepare_semantic_axis_descriptives] no shard data in event window", flush=True)
    else:
        _write_panels_from_accumulators(
            accum, launch, bucket_specs, panel_bin_days, out_dir, config, parent_lang_counts
        )

    if not args.skip_validation or not args.skip_examples:
        val_cols = _columns_to_read(
            include_validation=not args.skip_validation,
            include_examples=not args.skip_examples,
        )
        df = _load_shard_frames_concat(interim_dir, subs, args.max_shards, val_cols)
        if not df.empty:
            df = _in_event_window(df, config)
            df = _enrich_comment_frame(df, config)
            if not args.skip_validation:
                val_df = _validation_tables(df, config, sem_cfg, include_seed_oov=False)
                val_df.to_csv(out_dir / "semantic_axis_validation.csv", index=False)
                orient_df = ideology_orientation_report(df)
                orient_df.to_csv(out_dir / "ideology_axis_orientation_report.csv", index=False)
                print(
                    "[prepare_semantic_axis_descriptives] wrote ideology_axis_orientation_report.csv",
                    flush=True,
                )
            if not args.skip_examples:
                _examples_table(df).to_csv(out_dir / "semantic_axis_examples.csv", index=False)

    if not args.skip_seed_validation:
        _write_seed_validation_tables(config, sem_cfg, out_dir)

    print(f"[prepare_semantic_axis_descriptives] wrote tables to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
