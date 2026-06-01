"""
Aggregated DiD panels (topic_family, language, language_universe) for event studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from scripts.diagnostics.descriptives_util import (
    bin_lexical_daily_panel,
    event_dates_from_config,
    pooled_variance,
    weighted_mean_nan,
)
from src.config_utils import tables_subdir
from src.did.outcomes import OUTCOME_REGISTRY, OutcomeSpec
from src.did.panels import load_subreddit_panel, merge_wordfish_forum, wordfish_forum_v2_available
from src.did.paths import did_panels_dir
from src.did.specs import CONTROL_FAMILIES, ITALY_FAMILIES, assign_language_hub_series, rel_day_from_date

AGGREGATED_PANEL_LEVELS: Tuple[str, ...] = ("topic_family", "language", "language_universe")
AGGREGATED_BIN_DAYS: Tuple[int, ...] = (1, 3)

PANEL_OUTCOME_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "topic_family": ("lexical", "semantic_axis", "wordfish_forum", "wordfish_forum_v2"),
    "language": ("lexical", "semantic_axis", "wordfish_forum", "wordfish_forum_v2"),
    "language_universe": ("lexical", "semantic_axis", "wordfish_forum", "wordfish_forum_v2"),
}


@dataclass(frozen=True)
class AggregatedPanelKey:
    """Function summary: identifies one loaded analysis panel."""

    panel_level: str
    bin_days: int


@dataclass
class AggregatedPanels:
    """Function summary: container for topic_family / language / language_universe panels."""

    topic_family_1d: pd.DataFrame
    topic_family_3d: pd.DataFrame
    language_1d: pd.DataFrame
    language_3d: pd.DataFrame
    language_universe_1d: pd.DataFrame
    language_universe_3d: pd.DataFrame

    def get(self, key: AggregatedPanelKey) -> pd.DataFrame:
        """Function summary: return panel DataFrame for level and bin."""
        return getattr(self, f"{key.panel_level}_{key.bin_days}d")


def _annotate_did_panel(
    df: pd.DataFrame,
    launch: str,
    end_excl: str,
    entity_col: str,
    time_col: str,
    bin_days: int,
) -> pd.DataFrame:
    """Function summary: add rel_day, rel_period, IT, treat, entity_id, time_id."""
    if df.empty:
        return df.copy()
    out = df.copy()
    date_col = "period_start" if "period_start" in out.columns else "date_utc"
    out[date_col] = out[date_col].astype(str)
    out = out[out[date_col] < end_excl]
    out["rel_day"] = rel_day_from_date(out[date_col], launch)
    bd = int(bin_days)
    out["rel_period"] = (out["rel_day"] // bd).astype(int)
    out["post"] = (out[date_col] >= launch).astype(int)
    if "topic_family" in out.columns:
        out["IT"] = out["topic_family"].astype(str).isin(ITALY_FAMILIES).astype(int)
    elif "language_hub" in out.columns:
        out["IT"] = (out["language_hub"].astype(str) == "it").astype(int)
    elif "primary_lexicon" in out.columns:
        out["IT"] = (out["primary_lexicon"].astype(str) == "it").astype(int)
    else:
        out["IT"] = 0
    out["treat"] = out["IT"]
    out["entity_id"] = out[entity_col].astype(str)
    out["time_id"] = out[time_col].astype(str)
    out["bin_days"] = bd
    return out


def _subreddit_language_hub_meta(config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: subreddit metadata with language_hub (it/de/eu/us/uk)."""
    desc_dir = tables_subdir(config, "descriptives")
    fam_path = desc_dir / "daily_by_subreddit.csv"
    lex = _subreddit_lexicon_meta(config)
    if fam_path.is_file():
        fam = pd.read_csv(fam_path, usecols=["subreddit", "topic_family"]).drop_duplicates("subreddit")
        meta = fam.merge(lex, on="subreddit", how="inner")
    else:
        meta = lex.copy()
        meta["topic_family"] = ""
    meta["language_hub"] = assign_language_hub_series(meta)
    meta["primary_lexicon"] = meta["primary_lexicon"].astype(str)
    return meta


def _hub_primary_lexicon(hub: str) -> str:
    """Function summary: map language_hub back to primary_lexicon for legacy columns."""
    return "en" if hub in CONTROL_FAMILIES and hub != "de" else hub


def _rollup_weighted_outcomes(
    grp: pd.DataFrame,
    num_cols: List[str],
    weights: pd.Series | None,
) -> Dict[str, float]:
    """Function summary: aggregate numeric outcome columns with NaN-safe weights.

    Parameters:
    - grp: one hub-day (or similar) group of subreddit rows.
    - num_cols: numeric metric columns to roll up.
    - weights: comment-count weights aligned with grp; uniform if None.

    Returns:
    - Dict mapping column name to aggregated value.
    """
    out: Dict[str, float] = {}
    w = weights if weights is not None else pd.Series(1.0, index=grp.index)
    for col in num_cols:
        if col.endswith("_var"):
            mean_col = col.replace("_var", "_mean")
            if mean_col in grp.columns:
                out[col] = pooled_variance(grp[mean_col], grp[col], w)
                continue
        if w.sum() > 0:
            out[col] = weighted_mean_nan(grp[col].astype(float), w)
        else:
            out[col] = float(grp[col].astype(float).mean(skipna=True))
    return out


def _rollup_subreddit_lexical_by_hub(
    desc_dir: Path,
    hub_meta: pd.DataFrame,
    bin_days: int,
    launch: str,
    group_cols: Tuple[str, ...],
) -> pd.DataFrame:
    """Function summary: aggregate subreddit-day lexical metrics by language_hub (and optional slice)."""
    sub_path = desc_dir / "daily_by_subreddit.csv"
    if not sub_path.is_file() or hub_meta.empty:
        return pd.DataFrame()
    sub = pd.read_csv(sub_path)
    if "universe_slice" in group_cols:
        slice_path = desc_dir / "daily_by_subreddit_universe_slice.csv"
        if not slice_path.is_file():
            return pd.DataFrame()
        sl = pd.read_csv(slice_path, usecols=["subreddit", "date_utc", "universe_slice"])
        sub = sub.merge(sl, on=["subreddit", "date_utc"], how="inner")
    sub = sub.merge(
        hub_meta[["subreddit", "language_hub", "primary_lexicon"]].drop_duplicates("subreddit"),
        on="subreddit",
        how="inner",
    )
    skip = {
        "subreddit",
        "date_utc",
        "topic",
        "topic_family",
        "primary_lexicon",
        "language_hub",
        "universe_slice",
        "n_comments",
    }
    num_cols = [c for c in sub.columns if c not in skip and pd.api.types.is_numeric_dtype(sub[c])]
    rows: List[Dict[str, Any]] = []
    for keys, grp in sub.groupby(list(group_cols) + ["date_utc"], sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(list(group_cols) + ["date_utc"], keys))
        w = grp["n_comments"].astype(float) if "n_comments" in grp.columns else None
        row: Dict[str, Any] = {**key_map, "n_comments": len(grp)}
        row.update(_rollup_weighted_outcomes(grp, num_cols, w))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    daily = pd.DataFrame(rows)
    return bin_lexical_daily_panel(daily, list(group_cols), bin_days, launch)


def _rollup_semantic_to_language_hub(
    config: Dict[str, Any],
    bin_days: int,
    launch: str,
    group_cols: Tuple[str, ...],
) -> pd.DataFrame:
    """Function summary: roll up subreddit semantic-axis panel to language_hub × period (optional slice)."""
    sem_path = tables_subdir(config, "semantic_axis") / "semantic_axis_panel.csv"
    hub_meta = _subreddit_language_hub_meta(config)
    if not sem_path.is_file() or hub_meta.empty:
        return pd.DataFrame()
    sem = pd.read_csv(sem_path)
    date_col = "date_utc" if "date_utc" in sem.columns else "period_start"
    if date_col == "period_start":
        sem = sem.rename(columns={"period_start": "date_utc"})
        date_col = "date_utc"
    if "universe_slice" in group_cols:
        desc_dir = tables_subdir(config, "descriptives")
        slice_path = desc_dir / "daily_by_subreddit_universe_slice.csv"
        if not slice_path.is_file():
            return pd.DataFrame()
        sl = pd.read_csv(slice_path, usecols=["subreddit", "date_utc", "universe_slice"])
        sem = sem.merge(sl, on=["subreddit", "date_utc"], how="inner")
    sem = sem.merge(
        hub_meta[["subreddit", "language_hub", "primary_lexicon"]].drop_duplicates("subreddit"),
        on="subreddit",
        how="inner",
    )
    skip = {
        "subreddit",
        date_col,
        "topic",
        "topic_family",
        "primary_lexicon",
        "language_hub",
        "universe_slice",
        "panel_level",
        "bin_days",
        "post",
        "n_comments",
        "n_scored",
        "n_words_total",
    }
    skip |= {c for c in sem.columns if c.startswith("vpn_") or c.startswith("tor_") or c.startswith("log1p_tor")}
    num_cols = [c for c in sem.columns if c not in skip and pd.api.types.is_numeric_dtype(sem[c])]
    if not num_cols:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    gcols = list(group_cols) + [date_col]
    for keys, grp in sem.groupby(gcols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(gcols, keys))
        w = grp["n_comments"].astype(float) if "n_comments" in grp.columns else None
        row = dict(key_map)
        row["n_comments"] = int(grp["n_comments"].sum()) if "n_comments" in grp.columns else len(grp)
        row.update(_rollup_weighted_outcomes(grp, num_cols, w))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    daily = pd.DataFrame(rows)
    if "date_utc" not in daily.columns and "period_start" in daily.columns:
        daily = daily.rename(columns={"period_start": "date_utc"})
    return bin_lexical_daily_panel(daily, list(group_cols), bin_days, launch)


def _subreddit_lexicon_meta(config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: subreddit -> primary_lexicon from semantic forum panel."""
    sem_path = tables_subdir(config, "semantic_axis") / "semantic_axis_panel.csv"
    if not sem_path.is_file():
        return pd.DataFrame(columns=["subreddit", "primary_lexicon"])
    return pd.read_csv(sem_path, usecols=["subreddit", "primary_lexicon"]).drop_duplicates("subreddit")


def _merge_semantic_topic_family(panel: pd.DataFrame, config: Dict[str, Any], bin_days: int) -> pd.DataFrame:
    """Function summary: left-merge semantic topic_family panel onto lexical panel."""
    sem_path = tables_subdir(config, "semantic_axis") / f"semantic_axis_panel_by_topic_family_{bin_days}d.csv"
    if not sem_path.is_file():
        return panel
    sem = pd.read_csv(sem_path)
    date_col = "period_start" if "period_start" in sem.columns else "date_utc"
    sem = sem.rename(columns={date_col: "period_start"})
    drop = [c for c in sem.columns if c in panel.columns and c not in ("topic_family", "period_start")]
    sem = sem.drop(columns=drop, errors="ignore")
    return panel.merge(sem, on=["topic_family", "period_start"], how="left")


def _subreddit_panel_with_wordfish(config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: subreddit-day panel with forum Wordfish v1 and optional v2 columns."""
    try:
        sub = load_subreddit_panel(config)
    except FileNotFoundError:
        return pd.DataFrame()
    desc_dir = tables_subdir(config, "descriptives")
    fam_path = desc_dir / "daily_by_subreddit.csv"
    if fam_path.is_file():
        meta = pd.read_csv(fam_path, usecols=["subreddit", "topic_family"]).drop_duplicates()
        if "topic_family" not in sub.columns:
            sub = sub.merge(meta, on="subreddit", how="left")
    sub = merge_wordfish_forum(sub, config, tables_subdir_name="wordfish")
    if wordfish_forum_v2_available(config):
        wf_v2 = merge_wordfish_forum(sub[["subreddit", "date_utc"]].drop_duplicates(), config, tables_subdir_name="wordfish_forum_v2")
        rename = {
            c: f"{c}_v2"
            for c in ("extremity", "extremity_z", "change", "change_z")
            if c in wf_v2.columns
        }
        if rename:
            wf_v2 = wf_v2[["subreddit", "date_utc"] + list(rename.keys())].rename(columns=rename)
            sub = sub.merge(wf_v2, on=["subreddit", "date_utc"], how="left")
    return sub.loc[:, ~sub.columns.duplicated()].copy()


def _rollup_wordfish_grouped(
    config: Dict[str, Any],
    bin_days: int,
    launch: str,
    group_cols: Tuple[str, ...],
    base: pd.DataFrame,
) -> pd.DataFrame:
    """Function summary: bin subreddit Wordfish panel and mean by group_cols × period_start."""
    if base.empty:
        return pd.DataFrame()
    sub = base.copy()
    if "primary_lexicon" not in sub.columns:
        sub = sub.merge(_subreddit_lexicon_meta(config), on="subreddit", how="left")
    keep_cols = list(group_cols) + ["subreddit"]
    extra = [c for c in ("topic_family", "universe_slice") if c in sub.columns and c not in keep_cols]
    fam_map = sub[keep_cols + extra].drop_duplicates(subset=["subreddit"])
    if int(bin_days) <= 1:
        if "period_start" not in sub.columns:
            sub = sub.rename(columns={"date_utc": "period_start"})
    else:
        sub = bin_lexical_daily_panel(sub, ["subreddit"], bin_days, launch)
        sub = sub.merge(fam_map, on="subreddit", how="left")
    sub = sub.loc[:, ~sub.columns.duplicated()].copy()
    if "period_start" in sub.columns:
        sub["period_start"] = sub["period_start"].astype(str)
    wf_cols = [
        c
        for c in sub.columns
        if c in ("extremity", "extremity_z", "change", "change_z")
        or c.endswith("_v2")
    ]
    gcols = [c for c in group_cols if c in sub.columns]
    if not wf_cols or not gcols or "period_start" not in sub.columns:
        return pd.DataFrame()
    return sub.groupby(gcols + ["period_start"], as_index=False)[wf_cols].mean(numeric_only=True)


def _rollup_wordfish_to_topic_family(
    config: Dict[str, Any],
    bin_days: int,
    launch: str,
) -> pd.DataFrame:
    """Function summary: mean forum Wordfish outcomes by topic_family × period."""
    sub = _subreddit_panel_with_wordfish(config)
    if sub.empty or "topic_family" not in sub.columns:
        return pd.DataFrame()
    return _rollup_wordfish_grouped(config, bin_days, launch, ("topic_family",), sub)


def _rollup_wordfish_to_language_hub(
    config: Dict[str, Any],
    bin_days: int,
    launch: str,
    group_cols: Tuple[str, ...] = ("language_hub",),
) -> pd.DataFrame:
    """Function summary: mean forum Wordfish by language_hub (and optional universe_slice)."""
    sub = _subreddit_panel_with_wordfish(config)
    if sub.empty:
        return pd.DataFrame()
    hub_meta = _subreddit_language_hub_meta(config)
    sub = sub.merge(hub_meta[["subreddit", "language_hub"]].drop_duplicates("subreddit"), on="subreddit", how="left")
    sub["language_hub"] = sub["language_hub"].fillna(
        assign_language_hub_series(sub) if "primary_lexicon" in sub.columns else "en"
    )
    if "universe_slice" in group_cols:
        desc_dir = tables_subdir(config, "descriptives")
        slice_path = desc_dir / "daily_by_subreddit_universe_slice.csv"
        if slice_path.is_file():
            sl = pd.read_csv(slice_path, usecols=["subreddit", "date_utc", "universe_slice"])
            sub = sub.merge(sl, on=["subreddit", "date_utc"], how="inner")
    return _rollup_wordfish_grouped(config, bin_days, launch, group_cols, sub)


def _rollup_wordfish_to_language_universe(
    config: Dict[str, Any],
    desc_dir: Path,
    bin_days: int,
    launch: str,
) -> pd.DataFrame:
    """Function summary: mean forum Wordfish by language_hub × universe_slice × period."""
    return _rollup_wordfish_to_language_hub(
        config, bin_days, launch, group_cols=("language_hub", "universe_slice")
    )


def _build_topic_family_panel(
    desc_dir: Path,
    config: Dict[str, Any],
    bin_days: int,
    launch: str,
    end_excl: str,
) -> pd.DataFrame:
    """Function summary: topic_family panel with lexical + semantic + forum Wordfish."""
    fam_path = desc_dir / "daily_by_topic_family.csv"
    if not fam_path.is_file():
        return pd.DataFrame()
    panel = bin_lexical_daily_panel(pd.read_csv(fam_path), ["topic_family"], bin_days, launch)
    panel = _merge_semantic_topic_family(panel, config, bin_days)
    wf = _rollup_wordfish_to_topic_family(config, bin_days, launch)
    if not wf.empty:
        panel = panel.merge(wf, on=["topic_family", "period_start"], how="left")
    if "date_utc" not in panel.columns and "period_start" in panel.columns:
        panel["date_utc"] = panel["period_start"]
    return _annotate_did_panel(panel, launch, end_excl, "topic_family", "period_start", bin_days)


def _build_language_panel(
    config: Dict[str, Any],
    desc_dir: Path,
    bin_days: int,
    launch: str,
    end_excl: str,
) -> pd.DataFrame:
    """Function summary: language-hub panel (it/de/eu/us/uk) from semantic + lexical + Wordfish rollups."""
    hub_meta = _subreddit_language_hub_meta(config)
    panel = _rollup_semantic_to_language_hub(config, bin_days, launch, ("language_hub",))
    lex = _rollup_subreddit_lexical_by_hub(desc_dir, hub_meta, bin_days, launch, ("language_hub",))
    if not lex.empty:
        if panel.empty:
            panel = lex
        else:
            panel = panel.merge(lex, on=["language_hub", "period_start"], how="outer", suffixes=("", "_lex"))
    wf = _rollup_wordfish_to_language_hub(config, bin_days, launch, ("language_hub",))
    if not wf.empty:
        panel = panel.merge(wf, on=["language_hub", "period_start"], how="left") if not panel.empty else wf
    if panel.empty:
        return panel
    panel["primary_lexicon"] = panel["language_hub"].astype(str).map(_hub_primary_lexicon)
    return _annotate_did_panel(panel, launch, end_excl, "language_hub", "period_start", bin_days)


def _rollup_lexical_language_universe(
    desc_dir: Path,
    hub_meta: pd.DataFrame,
    bin_days: int,
    launch: str,
) -> pd.DataFrame:
    """Function summary: lexical daily metrics by language_hub × universe_slice."""
    return _rollup_subreddit_lexical_by_hub(
        desc_dir, hub_meta, bin_days, launch, ("language_hub", "universe_slice")
    )


def _build_language_universe_panel(
    config: Dict[str, Any],
    desc_dir: Path,
    bin_days: int,
    launch: str,
    end_excl: str,
) -> pd.DataFrame:
    """Function summary: language_hub × universe_slice panel with lexical + Wordfish rollups."""
    hub_meta = _subreddit_language_hub_meta(config)
    panel = _rollup_semantic_to_language_hub(
        config, bin_days, launch, ("language_hub", "universe_slice")
    )
    lex = _rollup_lexical_language_universe(desc_dir, hub_meta, bin_days, launch)
    if not lex.empty:
        if panel.empty:
            panel = lex
        else:
            panel = panel.merge(
                lex,
                on=["language_hub", "universe_slice", "period_start"],
                how="outer",
                suffixes=("", "_lex"),
            )
    wf = _rollup_wordfish_to_language_universe(config, desc_dir, bin_days, launch)
    if not wf.empty and not panel.empty:
        panel = panel.merge(
            wf,
            on=["language_hub", "universe_slice", "period_start"],
            how="left",
        )
    elif not wf.empty:
        panel = wf
    if panel.empty:
        return panel
    panel["primary_lexicon"] = panel["language_hub"].astype(str).map(_hub_primary_lexicon)
    panel["entity_id"] = panel["language_hub"].astype(str) + "|" + panel["universe_slice"].astype(str)
    out = _annotate_did_panel(panel, launch, end_excl, "entity_id", "period_start", bin_days)
    out["IT"] = out["language_hub"].astype(str) == "it"
    out["treat"] = out["IT"].astype(int)
    return out


def build_aggregated_panels(config: Dict[str, Any]) -> AggregatedPanels:
    """Function summary: load or build all aggregated DiD panels (1d and 3d)."""
    _, end_excl, launch, _ = event_dates_from_config(config)
    desc_dir = tables_subdir(config, "descriptives")
    agg_dir = did_panels_dir(config, "aggregated")
    agg_dir.mkdir(parents=True, exist_ok=True)

    def _cached(name: str, builder) -> Tuple[pd.DataFrame, pd.DataFrame]:
        p1, p3 = agg_dir / f"did_{name}_1d.csv", agg_dir / f"did_{name}_3d.csv"
        if p1.is_file() and p3.is_file():
            return pd.read_csv(p1), pd.read_csv(p3)
        d1, d3 = builder(1), builder(3)
        if not d1.empty:
            d1.to_csv(p1, index=False)
        if not d3.empty:
            d3.to_csv(p3, index=False)
        return d1, d3

    tf_1d, tf_3d = _cached(
        "topic_family",
        lambda bd: _build_topic_family_panel(desc_dir, config, bd, launch, end_excl),
    )
    lang_1d, lang_3d = _cached(
        "language",
        lambda bd: _build_language_panel(config, desc_dir, bd, launch, end_excl),
    )
    lu_1d, lu_3d = _cached(
        "language_universe",
        lambda bd: _build_language_universe_panel(config, desc_dir, bd, launch, end_excl),
    )
    return AggregatedPanels(
        topic_family_1d=tf_1d,
        topic_family_3d=tf_3d,
        language_1d=lang_1d,
        language_3d=lang_3d,
        language_universe_1d=lu_1d,
        language_universe_3d=lu_3d,
    )


def outcomes_for_panel_level(panel_level: str) -> Tuple[OutcomeSpec, ...]:
    """Function summary: outcome registry rows applicable to an aggregated panel level."""
    fams = set(PANEL_OUTCOME_FAMILIES.get(panel_level, ()))
    return tuple(o for o in OUTCOME_REGISTRY if o.family in fams)


def rel_col_for_bin(bin_days: int) -> str:
    """Function summary: event-time column for estimation and plots."""
    return "rel_day" if int(bin_days) <= 1 else "rel_period"
