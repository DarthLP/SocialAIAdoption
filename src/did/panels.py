"""
Load and merge DiD analysis panels from config-resolved paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from scripts.diagnostics.descriptives_util import event_dates_from_config
from src.config_utils import tables_subdir
from src.did.specs import rel_day_from_date


@dataclass
class AnalysisPanels:
    """Function summary: container for all DiD estimation panels."""

    sub_v1: pd.DataFrame
    sub_v2: pd.DataFrame
    slice_panel: pd.DataFrame
    auth_v1: pd.DataFrame
    auth_v2: pd.DataFrame


def load_subreddit_panel(config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: load did_subreddit_panel_1d.csv with calendar fields."""
    path = tables_subdir(config, "did") / "did_subreddit_panel_1d.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run prepare_did_subreddit_panel.py")
    return pd.read_csv(path)


def load_subreddit_slice_panel(config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: load long-format subreddit × universe_slice panel."""
    path = tables_subdir(config, "did") / "did_subreddit_panel_by_universe_slice_1d.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run prepare_did_subreddit_panel.py")
    return pd.read_csv(path)


def slice_panel_for_ddd(sl: pd.DataFrame) -> pd.DataFrame:
    """Function summary: slice panel with subreddit entity_id for triple-diff estimation."""
    out = sl.copy()
    out["entity_id"] = out["subreddit"].astype(str)
    out["time_id"] = out["date_utc"].astype(str)
    return out


def merge_semantic_axis(sub: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: left-merge forum semantic-axis outcomes onto subreddit-day panel."""
    sem_path = tables_subdir(config, "semantic_axis") / "semantic_axis_panel.csv"
    if not sem_path.is_file():
        return sub
    sem = pd.read_csv(sem_path)
    if "period_start" in sem.columns:
        sem = sem.rename(columns={"period_start": "date_utc"})
    keep = [
        "subreddit",
        "date_utc",
        "sem_axis_ideology_mean",
        "sem_axis_emotion_mean",
        "sem_axis_aggression_mean",
        "sem_axis_ideology_var",
        "sem_axis_emotion_var",
    ]
    keep = [c for c in keep if c in sem.columns]
    sem = sem[keep].drop_duplicates(["subreddit", "date_utc"])
    return sub.merge(sem, on=["subreddit", "date_utc"], how="left")


def merge_wordfish_forum(
    sub: pd.DataFrame,
    config: Dict[str, Any],
    tables_subdir_name: str = "wordfish",
) -> pd.DataFrame:
    """Function summary: merge day-level forum Wordfish onto subreddit panel.

    Parameters:
    - sub: subreddit-day panel.
    - config: loaded YAML.
    - tables_subdir_name: wordfish or wordfish_forum_v2.

    Returns:
    - Panel with extremity columns merged.
    """
    wf_path = tables_subdir(config, tables_subdir_name) / "wordfish_extremity_panel.csv"
    if not wf_path.is_file():
        return sub
    wf = pd.read_csv(wf_path)
    wf = wf[wf["time_bin"].astype(str) == "day"] if "time_bin" in wf.columns else wf
    date_col = "date_utc" if "date_utc" in wf.columns else "bin_start"
    cols = [
        "subreddit",
        date_col,
        "extremity",
        "extremity_z",
        "change",
        "change_z",
        "dispersion_var",
        "topic_family",
    ]
    cols = [c for c in cols if c in wf.columns]
    wf = wf[cols].rename(columns={date_col: "date_utc"})
    wf = wf.drop_duplicates(["subreddit", "date_utc"])
    drop = [c for c in ("topic_family",) if c in sub.columns and c in wf.columns]
    if drop:
        wf = wf.drop(columns=drop)
    return sub.merge(wf, on=["subreddit", "date_utc"], how="left", suffixes=("", "_wf"))


def wordfish_forum_v2_available(config: Dict[str, Any]) -> bool:
    """Function summary: True if forum v2 extremity panel exists."""
    path = tables_subdir(config, "wordfish_forum_v2") / "wordfish_extremity_panel.csv"
    return path.is_file()


def wordfish_authors_v2_available(config: Dict[str, Any]) -> bool:
    """Function summary: True if author v2 extremity panel exists."""
    path = tables_subdir(config, "wordfish_authors_v2") / "wordfish_authors_extremity_panel.csv"
    return path.is_file()


def _annotate_author_panel(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: add DiD calendar and entity columns to author panel."""
    start, end_excl, launch, _ = event_dates_from_config(config)
    out = df.copy()
    out["date_utc"] = out["bin_start"].astype(str)
    out["rel_day"] = rel_day_from_date(out["bin_start"], launch)
    out["post"] = (out["bin_start"].astype(str) >= launch).astype(int)
    if "IT" not in out.columns:
        out["IT"] = (out["primary_lexicon"].astype(str) == "it").astype(int)
    out["entity_id"] = out["author"].astype(str)
    out["time_id"] = out["bin_start"].astype(str)
    return out[(out["date_utc"] >= start) & (out["date_utc"] < end_excl)]


def load_author_wordfish_panel(
    config: Dict[str, Any],
    tables_subdir_name: str = "wordfish_authors",
) -> pd.DataFrame:
    """Function summary: load headline author Wordfish extremity panel from a tables subdir."""
    tab = tables_subdir(config, tables_subdir_name)
    path = tab / "wordfish_authors_extremity_panel.csv"
    if not path.is_file():
        path = tab / "wordfish_authors_extremity_panel_balanced_week7.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing wordfish authors panel under {tab}")
    return _annotate_author_panel(pd.read_csv(path), config)


def stack_author_wordfish_panels(
    config: Dict[str, Any],
    tables_subdir_name: str = "wordfish_authors",
) -> pd.DataFrame:
    """Function summary: stack per-language author panels when present for cross-country."""
    tab = tables_subdir(config, tables_subdir_name)
    frames = []
    for lang in ("it", "en", "de"):
        loaded = False
        for name in (
            f"wordfish_authors_extremity_panel_balanced_week7_{lang}.csv",
            f"wordfish_authors_extremity_panel_{lang}_balanced_week7.csv",
        ):
            p = tab / name
            if p.is_file():
                frames.append(pd.read_csv(p))
                loaded = True
                break
        if lang == "it" and not loaded:
            generic = tab / "wordfish_authors_extremity_panel_balanced_week7.csv"
            if generic.is_file():
                frames.append(pd.read_csv(generic))
    if not frames:
        try:
            return load_author_wordfish_panel(config, tables_subdir_name)
        except FileNotFoundError:
            return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["author", "bin_start"], keep="last"
    )
    return _annotate_author_panel(df, config)


def author_panel_has_multi_lang(auth: pd.DataFrame) -> Tuple[bool, bool]:
    """Function summary: detect en/de author panels for cross-language strategies.

    Returns:
    - (has_en, has_de) booleans.
    """
    if auth.empty or "primary_lexicon" not in auth.columns:
        return False, False
    lex = auth["primary_lexicon"].astype(str)
    return bool((lex == "en").any()), bool((lex == "de").any())


def _build_subreddit_panel(config: Dict[str, Any], wf_subdir: str) -> pd.DataFrame:
    """Function summary: subreddit-day panel with semantic axis and forum Wordfish merge."""
    sub = load_subreddit_panel(config)
    sub = merge_semantic_axis(sub, config)
    sub = merge_wordfish_forum(sub, config, tables_subdir_name=wf_subdir)
    sub["entity_id"] = sub["subreddit"].astype(str)
    sub["time_id"] = sub["date_utc"].astype(str)
    return sub


def build_analysis_panels(config: Dict[str, Any]) -> AnalysisPanels:
    """Function summary: all panels for DiD estimation (v1/v2 forum and author)."""
    sub_v1 = _build_subreddit_panel(config, "wordfish")
    sub_v2 = _build_subreddit_panel(config, "wordfish_forum_v2") if wordfish_forum_v2_available(config) else pd.DataFrame()

    try:
        sl = load_subreddit_slice_panel(config)
        sl["entity_id"] = sl["subreddit"].astype(str) + "|" + sl["universe_slice"].astype(str)
        sl["time_id"] = sl["date_utc"].astype(str)
    except FileNotFoundError:
        sl = pd.DataFrame()

    try:
        auth_v1 = stack_author_wordfish_panels(config, "wordfish_authors")
    except FileNotFoundError:
        auth_v1 = pd.DataFrame()

    if wordfish_authors_v2_available(config):
        auth_v2 = stack_author_wordfish_panels(config, "wordfish_authors_v2")
    else:
        auth_v2 = pd.DataFrame()

    return AnalysisPanels(
        sub_v1=sub_v1,
        sub_v2=sub_v2,
        slice_panel=sl,
        auth_v1=auth_v1,
        auth_v2=auth_v2,
    )


def build_analysis_panels_legacy(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Function summary: legacy 3-tuple API for backward compatibility."""
    p = build_analysis_panels(config)
    return p.sub_v1, p.slice_panel, p.auth_v1
