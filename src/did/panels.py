"""
Load and merge DiD analysis panels from config-resolved paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd

from scripts.diagnostics.descriptives_util import event_dates_from_config
from src.config_utils import tables_subdir
from src.did.paths import did_panels_dir, resolve_panel_path
from src.did.specs import rel_day_from_date


@dataclass
class AnalysisPanels:
    """Function summary: container for all DiD estimation panels."""

    sub_v1: pd.DataFrame
    sub_v2: pd.DataFrame
    slice_panel: pd.DataFrame
    auth_v1: pd.DataFrame
    auth_v2: pd.DataFrame
    auth_semantic: pd.DataFrame
    comment_1d: pd.DataFrame
    author_day_1d: pd.DataFrame


def load_subreddit_panel(config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: load did_subreddit_panel_1d.csv with calendar fields."""
    path = resolve_panel_path(config, "subreddit", "did_subreddit_panel_1d.csv")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run prepare_did_subreddit_panel.py")
    return pd.read_csv(path)


def load_subreddit_slice_panel(config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: load long-format subreddit × universe_slice panel."""
    path = resolve_panel_path(
        config, "subreddit", "did_subreddit_panel_by_universe_slice_1d.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run prepare_did_subreddit_panel.py")
    return pd.read_csv(path)


def load_subreddit_event_study_panel(config: Dict[str, Any], bin_days: int) -> pd.DataFrame:
    """Function summary: subreddit panel for aggregated language event studies (3d bins outcomes).

    Parameters:
    - config: project config dict.
    - bin_days: 1 (daily) or 3 (launch-aligned outcome bins via bin_lexical_daily_panel).

    Returns:
    - Subreddit panel with entity_id, time_id, rel_day, rel_period.
    """
    from src.did.event_study_panels import load_subreddit_event_study_panel_binned

    return load_subreddit_event_study_panel_binned(config, bin_days)


def load_subreddit_slice_event_study_panel(
    config: Dict[str, Any], bin_days: int
) -> pd.DataFrame:
    """Function summary: subreddit×universe_slice panel for in/out political-tree event studies.

    Parameters:
    - config: project config dict.
    - bin_days: 1 or 3 (3d bins outcomes at subreddit×slice).

    Returns:
    - Slice panel with entity_id=subreddit|universe_slice, treat, universe_slice, rel columns.
    """
    from src.did.event_study_panels import load_subreddit_slice_event_study_panel_binned

    return load_subreddit_slice_event_study_panel_binned(config, bin_days)


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
        "sem_axis_ideology_pole_share",
        "sem_axis_ideology_esteban_ray",
        "sem_axis_ideology_share_left_below_p10",
        "sem_axis_ideology_share_right_above_p90",
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


def resolve_author_wordfish_spec(
    config: Dict[str, Any],
    override: Optional[str] = None,
) -> str:
    """Function summary: Wordfish author time-bin spec for DiD (week7, week3, etc.).

    Parameters:
    - config: loaded YAML.
    - override: CLI or test override; wins over config.

    Returns:
    - Spec name string (e.g. week7).
    """
    if override:
        return str(override).strip()
    did_cfg = config.get("did") or {}
    if did_cfg.get("author_wordfish_spec"):
        return str(did_cfg["author_wordfish_spec"]).strip()
    wfa = config.get("wordfish_authors") or {}
    return str(wfa.get("headline_spec", "week7")).strip()


def _author_extremity_panel_candidates(
    tab: Path,
    spec: str,
    panel_mode: str = "balanced",
    lang: Optional[str] = None,
) -> Tuple[str, ...]:
    """Function summary: ordered CSV paths to try for one author Wordfish extremity panel."""
    tag = f"{panel_mode}_{spec}"
    names: list[str] = []
    if lang:
        names.extend(
            [
                f"wordfish_authors_extremity_panel_{tag}_{lang}.csv",
                f"wordfish_authors_extremity_panel_{lang}_{tag}.csv",
            ]
        )
    names.extend(
        [
            f"wordfish_authors_extremity_panel_{tag}.csv",
            "wordfish_authors_extremity_panel.csv",
            "wordfish_authors_extremity_panel_balanced_week7.csv",
        ]
    )
    return tuple(str(tab / n) for n in names)


def _read_first_existing_csv(candidates: Tuple[str, ...]) -> pd.DataFrame:
    """Function summary: load first existing path from candidates or raise FileNotFoundError."""
    for path in candidates:
        p = Path(path)
        if p.is_file():
            return pd.read_csv(p)
    raise FileNotFoundError(f"No panel found among: {candidates[:3]}...")


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
    spec: Optional[str] = None,
) -> pd.DataFrame:
    """Function summary: load author Wordfish extremity panel from a tables subdir.

    Parameters:
    - config: loaded YAML.
    - tables_subdir_name: wordfish_authors or wordfish_authors_v2.
    - spec: time-bin tag (week7, week3); default from resolve_author_wordfish_spec.

    Returns:
    - Annotated author×bin panel.
    """
    tab = tables_subdir(config, tables_subdir_name)
    spec_name = resolve_author_wordfish_spec(config, override=spec)
    df = _read_first_existing_csv(_author_extremity_panel_candidates(tab, spec_name))
    return _annotate_author_panel(df, config)


def stack_author_wordfish_panels(
    config: Dict[str, Any],
    tables_subdir_name: str = "wordfish_authors",
    spec: Optional[str] = None,
) -> pd.DataFrame:
    """Function summary: stack per-language author panels when present for cross-country.

    Parameters:
    - config: loaded YAML.
    - tables_subdir_name: wordfish_authors or wordfish_authors_v2.
    - spec: time-bin tag (week7, week3); default from resolve_author_wordfish_spec.

    Returns:
    - Annotated stacked author panel.
    """
    tab = tables_subdir(config, tables_subdir_name)
    spec_name = resolve_author_wordfish_spec(config, override=spec)
    frames = []
    for lang in ("it", "en", "de"):
        try:
            frames.append(
                _read_first_existing_csv(
                    _author_extremity_panel_candidates(tab, spec_name, lang=lang)
                )
            )
        except FileNotFoundError:
            continue
    if not frames:
        try:
            return load_author_wordfish_panel(config, tables_subdir_name, spec=spec_name)
        except FileNotFoundError:
            return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["author", "bin_start"], keep="last"
    )
    return _annotate_author_panel(df, config)


def author_semantic_week_panel_available(config: Dict[str, Any]) -> bool:
    """Function summary: True if author×week semantic DiD panel CSV exists."""
    path = resolve_panel_path(config, "author", "did_author_semantic_week_panel.csv")
    return path.is_file()


def load_author_semantic_week_panel(config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: load author×ISO-week semantic panel for cross-country DiD."""
    path = resolve_panel_path(config, "author", "did_author_semantic_week_panel.csv")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run prepare_did_author_semantic_week_panel.py"
        )
    return pd.read_csv(path)


def comment_panel_available(config: Dict[str, Any], bin_days: int = 1) -> bool:
    """Function summary: True if partitioned comment DiD panel exists."""
    tag = f"{int(bin_days)}d"
    panel_dir = did_panels_dir(config, "comment") / f"did_comment_panel_{tag}"
    return panel_dir.is_dir() and bool(list(panel_dir.glob("month=*.parquet")))


def load_comment_panel(
    config: Dict[str, Any],
    bin_days: int = 1,
    sample_frac: Optional[float] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Function summary: load partitioned comment-level DiD panel.

    Parameters:
    - config: loaded YAML.
    - bin_days: 1 or 3 calendar bin width.
    - sample_frac: optional random subsample for dev runs.
    - max_rows: optional row cap after sampling.

    Returns:
    - Concatenated comment panel.
    """
    tag = f"{int(bin_days)}d"
    panel_dir = did_panels_dir(config, "comment") / f"did_comment_panel_{tag}"
    parts = sorted(panel_dir.glob("month=*.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"Missing comment panel under {panel_dir}; run prepare_did_comment_panel.py"
        )
    frames = [pd.read_parquet(p) for p in parts]
    df = pd.concat(frames, ignore_index=True)
    if sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=float(sample_frac), random_state=42)
    if max_rows is not None and max_rows > 0:
        df = df.head(int(max_rows))
    return df


def load_author_day_panel(config: Dict[str, Any], bin_days: int = 1) -> pd.DataFrame:
    """Function summary: load author×day aggregated DiD panel CSV.

    Parameters:
    - config: loaded YAML.
    - bin_days: 1 or 3.

    Returns:
    - Author×day panel with entity_id and time_id.
    """
    tag = f"{int(bin_days)}d"
    path = did_panels_dir(config, "comment") / f"did_author_day_panel_{tag}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run prepare_did_comment_panel.py")
    out = pd.read_csv(path)
    if "entity_id" not in out.columns and "author" in out.columns:
        out["entity_id"] = out["author"].astype(str)
    if "time_id" not in out.columns and "date_utc" in out.columns:
        out["time_id"] = out["date_utc"].astype(str)
    return out


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


def _families_need_subreddit(families: Optional[Sequence[str]]) -> bool:
    """Function summary: True when estimation needs subreddit-day panels."""
    if not families:
        return True
    fam = set(families)
    if fam & {"lexical", "semantic_axis"}:
        return True
    return any(f.startswith("wordfish_forum") for f in fam)


def _families_need_slice(families: Optional[Sequence[str]]) -> bool:
    """Function summary: True when estimation may use universe-slice panel."""
    if not families:
        return True
    return bool(set(families) & {"lexical", "semantic_axis"})


def _families_need_author_wordfish(families: Optional[Sequence[str]]) -> bool:
    """Function summary: True when estimation needs Wordfish author panels."""
    if not families:
        return True
    return any(f in ("wordfish_author", "wordfish_author_v2") for f in families)


def _families_need_author_semantic(families: Optional[Sequence[str]]) -> bool:
    """Function summary: True when estimation needs author×week semantic panel."""
    if not families:
        return True
    return "semantic_axis_author_week" in set(families)


def _families_need_comment(families: Optional[Sequence[str]]) -> bool:
    """Function summary: True when estimation needs comment-level panel."""
    if not families:
        return False
    fam = set(families)
    return bool(fam & {"lexical_comment", "semantic_axis_comment"})


def _families_need_author_day(families: Optional[Sequence[str]]) -> bool:
    """Function summary: True when estimation needs author×day panel."""
    if not families:
        return False
    fam = set(families)
    return bool(fam & {"lexical_author_day", "semantic_axis_author_day"})


def build_analysis_panels(
    config: Dict[str, Any],
    families: Optional[Sequence[str]] = None,
    author_wordfish_spec: Optional[str] = None,
    comment_sample_frac: Optional[float] = None,
    comment_max_rows: Optional[int] = None,
) -> AnalysisPanels:
    """Function summary: panels for DiD estimation (v1/v2 forum and author).

    Parameters:
    - config: loaded study YAML.
    - families: when set, skip loading panels not required by these outcome families.
    - author_wordfish_spec: override did.author_wordfish_spec (e.g. week3 robustness).
    """
    wfa_spec = resolve_author_wordfish_spec(config, override=author_wordfish_spec)
    if _families_need_subreddit(families):
        sub_v1 = _build_subreddit_panel(config, "wordfish")
        need_v2 = not families or "wordfish_forum_v2" in set(families)
        sub_v2 = (
            _build_subreddit_panel(config, "wordfish_forum_v2")
            if need_v2 and wordfish_forum_v2_available(config)
            else pd.DataFrame()
        )
    else:
        sub_v1 = pd.DataFrame()
        sub_v2 = pd.DataFrame()

    if _families_need_slice(families):
        try:
            sl = load_subreddit_slice_panel(config)
            sl["entity_id"] = sl["subreddit"].astype(str) + "|" + sl["universe_slice"].astype(str)
            sl["time_id"] = sl["date_utc"].astype(str)
        except FileNotFoundError:
            sl = pd.DataFrame()
    else:
        sl = pd.DataFrame()

    if _families_need_author_wordfish(families):
        try:
            auth_v1 = stack_author_wordfish_panels(
                config, "wordfish_authors", spec=wfa_spec
            )
        except FileNotFoundError:
            auth_v1 = pd.DataFrame()
        if wordfish_authors_v2_available(config):
            auth_v2 = stack_author_wordfish_panels(
                config, "wordfish_authors_v2", spec=wfa_spec
            )
        else:
            auth_v2 = pd.DataFrame()
    else:
        auth_v1 = pd.DataFrame()
        auth_v2 = pd.DataFrame()

    if _families_need_author_semantic(families) and author_semantic_week_panel_available(config):
        try:
            auth_semantic = load_author_semantic_week_panel(config)
        except FileNotFoundError:
            auth_semantic = pd.DataFrame()
    else:
        auth_semantic = pd.DataFrame()

    if _families_need_comment(families) and comment_panel_available(config, bin_days=1):
        try:
            comment_1d = load_comment_panel(
                config,
                bin_days=1,
                sample_frac=comment_sample_frac,
                max_rows=comment_max_rows,
            )
        except FileNotFoundError:
            comment_1d = pd.DataFrame()
    else:
        comment_1d = pd.DataFrame()

    if _families_need_author_day(families):
        try:
            author_day_1d = load_author_day_panel(config, bin_days=1)
        except FileNotFoundError:
            author_day_1d = pd.DataFrame()
    else:
        author_day_1d = pd.DataFrame()

    return AnalysisPanels(
        sub_v1=sub_v1,
        sub_v2=sub_v2,
        slice_panel=sl,
        auth_v1=auth_v1,
        auth_v2=auth_v2,
        auth_semantic=auth_semantic,
        comment_1d=comment_1d,
        author_day_1d=author_day_1d,
    )


def build_analysis_panels_legacy(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Function summary: legacy 3-tuple API for backward compatibility."""
    p = build_analysis_panels(config)
    return p.sub_v1, p.slice_panel, p.auth_v1
