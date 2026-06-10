"""
Within-author English writing quality DiD helpers (Italy ChatGPT ban).

Roster classification, cohort gates, language-based standardization, and
comment-level estimation specs for native-control and cross-language designs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.config_utils import infer_subreddit_primary_lexicon, subreddit_arm_map
from src.did.bucket_estimate import (
    _es_rows_from_fit,
    _feols_fit,
    estimate_comment_it_event_study,
)
from src.did.specs import rel_day_from_date

ITALIAN_ARMS = frozenset({"discovered_italian", "discovery_seed_italian"})
EN_CONTROL_ARM = "control_english_political"
EUROPE_HUB_ARM = "control_europe_hub"
EN_POLITICAL_ARM = "control_europe_political"

# Control arms whose English-ness is decided by the subreddit primary lexicon
# (so ukpolitics/europe/unitedkingdom count as English, while de does not).
EN_CONTROL_ARMS = frozenset({EN_CONTROL_ARM, EUROPE_HUB_ARM, EN_POLITICAL_ARM})

DEFAULT_BOT_AUTHORS = frozenset(
    {
        "AutoModerator",
        "PoliticsModeratorBot",
        "politicsmoderatorbot",
        "ModeratorOfPolitics",
        "SnapshillBot",
        "WikiTextBot",
        "RemindMeBot",
        "sneakpeekbot",
        "converter-bot",
    }
)

ENGLISH_QUALITY_OUTCOMES: Tuple[str, ...] = (
    "readability",
    "ttr_50w",
    "avg_words_per_sentence_comment",
    "log_len",
    "ai_sentence_length_variance",
    "style_index_llm",
)

HEADLINE_OUTCOMES: Tuple[str, ...] = (
    "readability",
    "ttr_50w",
    "avg_words_per_sentence_comment",
    "log_len",
)

CROSS_LANGUAGE_HEADLINE_OUTCOMES: Tuple[str, ...] = (
    "log_len",
    "avg_words_per_sentence_comment",
    "ttr_50w",
)

OUTCOME_CAVEATS: Dict[str, str] = {
    "ttr_50w": (
        "50-word minimum induces endogenous selection if the ban shortens comments"
    ),
}

# Polarization / aggression / semantic outcomes added on top of the writing-quality
# set. Lexical and semantic levels are not comparable across languages, so these are
# always z-scored within lang_comment (see MasterSystemPrompt sign_only_cross_country).
POLARIZATION_OUTCOMES: Tuple[str, ...] = (
    "net_ideology",
    "aggression_rate_100w",
    "extremity",
    "sem_axis_ideology",
    "sem_axis_emotion",
    "sem_axis_aggression",
)

ROSTER_WINDOW_CHOICES: Tuple[str, ...] = ("pre_ban", "full")
CROSS_LANGUAGE_MIN_STRICT_AUTHORS = 50

PANEL_COLUMNS: Tuple[str, ...] = (
    "id",
    "author",
    "subreddit",
    "date_utc",
    "n_words",
    "topic_family",
    "primary_lexicon",
    "lang_comment",
    "comment_in_political_universe",
    "readability",
    "ttr_50w",
    "log_len",
    "avg_words_per_sentence_comment",
    "ai_sentence_length_variance",
    "style_index_llm",
    "net_ideology",
    "aggression_rate_100w",
    "negative_rate_100w",
    "extremity",
    "sem_axis_ideology",
    "sem_axis_emotion",
    "sem_axis_aggression",
    "sem_axis_coverage",
    "has_sem_axis",
)


@dataclass(frozen=True)
class EnglishQualityCohortThresholds:
    """Function summary: per-author comment/word gates for English-quality panels."""

    label: str
    min_comments: int
    min_words: int
    min_comment_words: int


def default_cohort_thresholds() -> List[EnglishQualityCohortThresholds]:
    """Function summary: strict and loose cohort definitions for panel gating.

    Returns:
    - List with strict then loose threshold bundles.
    """
    return [
        EnglishQualityCohortThresholds("strict", 4, 200, 5),
        EnglishQualityCohortThresholds("loose", 2, 100, 3),
    ]


def cohort_thresholds_by_label(label: str) -> EnglishQualityCohortThresholds:
    """Function summary: resolve one cohort label to thresholds.

    Parameters:
    - label: strict or loose.

    Returns:
    - Matching EnglishQualityCohortThresholds.

    Raises:
    - ValueError when label is unknown.
    """
    for th in default_cohort_thresholds():
        if th.label == label:
            return th
    raise ValueError(f"Unknown cohort label: {label}")


def english_quality_tables_dir(config: Dict[str, Any]) -> Path:
    """Function summary: tables root for English-quality DiD artifacts.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - Path under results/tables/<study>/did/english_quality/.
    """
    from src.config_utils import tables_subdir

    return tables_subdir(config, "did") / "english_quality"


def english_quality_figures_dir(config: Dict[str, Any]) -> Path:
    """Function summary: figures root for English-quality event studies.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - Path under results/figures/<study>/did/english_quality/.
    """
    from src.config_utils import figures_subdir

    return figures_subdir(config, "did") / "english_quality"


def roster_window_subdir(roster_window: str) -> str:
    """Function summary: subdirectory tag for a roster-window run.

    Parameters:
    - roster_window: pre_ban or full.

    Returns:
    - Subdir name like roster_window=pre_ban.

    Raises:
    - ValueError when roster_window is unknown.
    """
    if roster_window not in ROSTER_WINDOW_CHOICES:
        raise ValueError(
            f"Unknown roster_window {roster_window!r}; expected one of {ROSTER_WINDOW_CHOICES}"
        )
    return f"roster_window={roster_window}"


def english_quality_run_tables_dir(
    config: Dict[str, Any],
    roster_window: str = "pre_ban",
) -> Path:
    """Function summary: tables root for one roster-window English-quality run.

    Parameters:
    - config: loaded study YAML.
    - roster_window: pre_ban (default) or full.

    Returns:
    - Path under results/tables/<study>/did/english_quality/roster_window=<mode>/.
    """
    return english_quality_tables_dir(config) / roster_window_subdir(roster_window)


def english_quality_run_figures_dir(
    config: Dict[str, Any],
    roster_window: str = "pre_ban",
) -> Path:
    """Function summary: figures root for one roster-window English-quality run.

    Parameters:
    - config: loaded study YAML.
    - roster_window: pre_ban (default) or full.

    Returns:
    - Path under results/figures/<study>/did/english_quality/roster_window=<mode>/.
    """
    return english_quality_figures_dir(config) / roster_window_subdir(roster_window)


def is_italian_arm_subreddit(subreddit: str, arm_map: Dict[str, str]) -> bool:
    """Function summary: True when subreddit belongs to an Italian discovery arm.

    Parameters:
    - subreddit: forum name.
    - arm_map: subreddit -> arm from config.

    Returns:
    - Boolean Italian-arm indicator.
    """
    return arm_map.get(subreddit, "") in ITALIAN_ARMS


def is_en_control_subreddit(subreddit: str, config: Dict[str, Any], arm_map: Dict[str, str]) -> bool:
    """Function summary: True for English-control forums used in native-control contrast.

    Parameters:
    - subreddit: forum name.
    - config: study YAML.
    - arm_map: subreddit -> arm map.

    Returns:
    - Boolean English-control indicator.

    Notes:
    - All recognised control arms (English-political, Europe hubs, and the
      UK-political arm that holds ukpolitics) are admitted only when the
      subreddit primary lexicon resolves to English. This keeps ukpolitics and
      europe/unitedkingdom in, while excluding German hubs such as de.
    """
    arm = arm_map.get(subreddit, "")
    if arm in EN_CONTROL_ARMS:
        return infer_subreddit_primary_lexicon(config, subreddit) == "en"
    return False


LANG_BILINGUAL_MIN_COMMENTS = 2


def dominant_pre_language(pre_en: int, pre_it: int, pre_total: int) -> str:
    """Function summary: dominant pre-ban language label from comment counts.

    Parameters:
    - pre_en: pre-ban English comments.
    - pre_it: pre-ban Italian comments.
    - pre_total: pre-ban comments in any language.

    Returns:
    - One of "it", "en", "other" (the majority bucket; ties favour the larger
      explicit language, then "other"). Returns "other" when pre_total == 0.
    """
    if pre_total <= 0:
        return "other"
    pre_other = max(pre_total - pre_en - pre_it, 0)
    shares = {"it": pre_it, "en": pre_en, "other": pre_other}
    return max(shares, key=lambda k: (shares[k], k == "it", k == "en"))


def classify_author_roster(
    author_forums: Dict[str, Set[str]],
    config: Dict[str, Any],
    arm_map: Optional[Dict[str, str]] = None,
    author_lang: Optional[Dict[str, Dict[str, float]]] = None,
    *,
    lang_bilingual_min: int = LANG_BILINGUAL_MIN_COMMENTS,
) -> pd.DataFrame:
    """Function summary: assign forum group and merge native-language proxy attrs.

    Parameters:
    - author_forums: author -> set of subreddits commented in.
    - config: study YAML.
    - arm_map: optional precomputed arm map.
    - author_lang: optional author -> language activity dict with keys
      pre_en, pre_it, pre_total, tot_en, tot_it (see collect_author_activity).
      When the roster is built with roster_window=pre_ban, tot_en and tot_it count
      only pre-launch comments (same window as forum membership).
    - lang_bilingual_min: min total EN and IT comments for the forum-agnostic
      lang_bilingual flag.

    Returns:
    - DataFrame with author, author_group, forum counts plus native-language
      proxy columns (italian_share_pre, dominant_pre_lang, pre_en_comments,
      pre_it_comments, pre_total_comments, lang_bilingual).
    """
    arm_map = arm_map or subreddit_arm_map(config)
    author_lang = author_lang or {}
    rows: List[Dict[str, Any]] = []
    for author, forums in sorted(author_forums.items()):
        it_forums = sorted(f for f in forums if is_italian_arm_subreddit(f, arm_map))
        en_forums = sorted(f for f in forums if is_en_control_subreddit(f, config, arm_map))
        has_it = bool(it_forums)
        has_en = bool(en_forums)
        if has_it and has_en:
            group = "italian_bilingual"
        elif has_en and not has_it:
            group = "native_control"
        else:
            group = "other"
        stats = author_lang.get(str(author), {})
        pre_en = int(stats.get("pre_en", 0))
        pre_it = int(stats.get("pre_it", 0))
        pre_total = int(stats.get("pre_total", 0))
        tot_en = int(stats.get("tot_en", 0))
        tot_it = int(stats.get("tot_it", 0))
        italian_share_pre = (pre_it / pre_total) if pre_total > 0 else 0.0
        rows.append(
            {
                "author": str(author),
                "author_group": group,
                "n_forums": len(forums),
                "italian_forums": ";".join(it_forums),
                "en_control_forums": ";".join(en_forums),
                "has_italian_forum": int(has_it),
                "has_en_control_forum": int(has_en),
                "pre_en_comments": pre_en,
                "pre_it_comments": pre_it,
                "pre_total_comments": pre_total,
                "italian_share_pre": round(float(italian_share_pre), 6),
                "dominant_pre_lang": dominant_pre_language(pre_en, pre_it, pre_total),
                "lang_bilingual": int(tot_en >= lang_bilingual_min and tot_it >= lang_bilingual_min),
            }
        )
    return pd.DataFrame(rows)


def filter_roster_authors(roster: pd.DataFrame, groups: Sequence[str]) -> Set[str]:
    """Function summary: author ids for selected roster groups.

    Parameters:
    - roster: classify_author_roster output.
    - groups: author_group values to keep.

    Returns:
    - Set of author id strings.
    """
    if roster.empty:
        return set()
    keep = roster["author_group"].astype(str).isin([str(g) for g in groups])
    return set(roster.loc[keep, "author"].astype(str))


def annotate_english_quality_comments(
    df: pd.DataFrame,
    launch: str,
    roster: pd.DataFrame,
) -> pd.DataFrame:
    """Function summary: add DiD calendar fields and author/language treatment flags.

    Parameters:
    - df: comment rows with date_utc, author, lang_comment, primary_lexicon.
    - launch: ban launch day UTC (YYYY-MM-DD).
    - roster: author roster with author_group.

    Returns:
    - Annotated comment DataFrame.
    """
    out = df.copy()
    out["date_utc"] = out["date_utc"].astype(str)
    out["author"] = out["author"].astype(str)
    out["rel_day"] = rel_day_from_date(out["date_utc"], launch)
    out["post"] = (out["date_utc"] >= launch).astype(int)
    out["time_id"] = out["date_utc"].astype(str)
    out["lang_comment"] = out["lang_comment"].astype(str).str.lower()
    out["is_english"] = (out["lang_comment"] == "en").astype(int)
    if not roster.empty:
        idx = roster.set_index("author")
        grp = idx["author_group"].to_dict()
        out["author_group"] = out["author"].map(grp).fillna("other")
        for col, default in (
            ("italian_share_pre", 0.0),
            ("dominant_pre_lang", "other"),
            ("lang_bilingual", 0),
        ):
            if col in roster.columns:
                out[col] = out["author"].map(idx[col].to_dict())
                out[col] = out[col].fillna(default)
            else:
                out[col] = default
    else:
        out["author_group"] = "other"
        out["italian_share_pre"] = 0.0
        out["dominant_pre_lang"] = "other"
        out["lang_bilingual"] = 0
    out["italian_author"] = (out["author_group"] == "italian_bilingual").astype(int)
    return out


def apply_3d_bins(df: pd.DataFrame, launch: str, bin_days: int) -> pd.DataFrame:
    """Function summary: map rel_day to launch-aligned 3-day bins and time_id.

    Parameters:
    - df: rows with rel_day.
    - launch: ban anchor date.
    - bin_days: bin width (1 leaves daily time_id).

    Returns:
    - Copy with rel_period and updated time_id when bin_days > 1.
    """
    if bin_days <= 1:
        out = df.copy()
        out["rel_period"] = out["rel_day"].astype(int)
        return out
    out = df.copy()
    out["rel_period"] = (out["rel_day"] // bin_days).astype(int)
    launch_dt = pd.Timestamp(launch)
    out["period_start"] = (
        launch_dt + pd.to_timedelta(out["rel_period"] * bin_days, unit="D")
    ).dt.strftime("%Y-%m-%d")
    out["time_id"] = out["period_start"].astype(str)
    return out


def author_language_activity(
    panel: pd.DataFrame,
    *,
    lang_value: str,
    english_forum_only: bool = False,
) -> pd.DataFrame:
    """Function summary: per-author pre/post comment and word counts for one language.

    Parameters:
    - panel: annotated comment panel.
    - lang_value: lang_comment value to count (en or it).
    - english_forum_only: when True, also require primary_lexicon == en.

    Returns:
    - DataFrame indexed by author with n_pre, n_post, words_pre, words_post.
    """
    work = panel.copy()
    work["lang_comment"] = work["lang_comment"].astype(str).str.lower()
    mask = work["lang_comment"] == lang_value
    if english_forum_only:
        mask &= work["primary_lexicon"].astype(str) == "en"
    work = work.loc[mask]
    if work.empty:
        return pd.DataFrame(
            columns=["author", "n_pre", "n_post", "words_pre", "words_post"]
        )
    nw = pd.to_numeric(work["n_words"], errors="coerce").fillna(0)
    work["n_words_num"] = nw
    rows: List[Dict[str, Any]] = []
    for author, grp in work.groupby("author", observed=True):
        pre_mask = grp["post"].astype(int) == 0
        post_mask = grp["post"].astype(int) == 1
        rows.append(
            {
                "author": str(author),
                "n_pre": int(pre_mask.sum()),
                "n_post": int(post_mask.sum()),
                "words_pre": float(grp.loc[pre_mask, "n_words_num"].sum()),
                "words_post": float(grp.loc[post_mask, "n_words_num"].sum()),
            }
        )
    return pd.DataFrame(rows)


def authors_passing_cohort(
    activity: pd.DataFrame,
    thresholds: EnglishQualityCohortThresholds,
) -> Set[str]:
    """Function summary: authors meeting min comment/word gates pre and post.

    Parameters:
    - activity: author_language_activity output.
    - thresholds: cohort gates.

    Returns:
    - Set of passing author ids.
    """
    if activity.empty:
        return set()
    ok = (
        (activity["n_pre"] >= thresholds.min_comments)
        & (activity["n_post"] >= thresholds.min_comments)
        & (activity["words_pre"] >= thresholds.min_words)
        & (activity["words_post"] >= thresholds.min_words)
    )
    return set(activity.loc[ok, "author"].astype(str))


CROSS_LANGUAGE_DESIGNS = frozenset(
    {"cross_language", "cross_language_native_it", "cross_language_langmix"}
)
NATIVE_IT_SHARE_MIN = 0.60


def _authors_with_attr(panel: pd.DataFrame, predicate) -> Set[str]:
    """Function summary: author ids whose first-seen attribute row satisfies predicate.

    Parameters:
    - panel: annotated comment panel (author-invariant attribute columns).
    - predicate: callable taking the per-author attribute row (Series) -> bool.

    Returns:
    - Set of author id strings.
    """
    if panel.empty:
        return set()
    first = panel.groupby("author", observed=True).first()
    keep = first.apply(predicate, axis=1)
    return set(first.index[keep].astype(str))


def cohort_authors_for_design(
    panel: pd.DataFrame,
    design: str,
    thresholds: EnglishQualityCohortThresholds,
) -> Set[str]:
    """Function summary: authors passing design-specific language cohort gates.

    Parameters:
    - panel: annotated comment panel (carries author_group, italian_share_pre,
      lang_bilingual when built by annotate_english_quality_comments).
    - design: native_control, cross_language, cross_language_native_it, or
      cross_language_langmix.
    - thresholds: cohort gates.

    Returns:
    - Set of author ids in the estimation cohort.
    """
    if design == "native_control":
        en_act = author_language_activity(panel, lang_value="en", english_forum_only=True)
        return authors_passing_cohort(en_act, thresholds)
    if design in CROSS_LANGUAGE_DESIGNS:
        en_act = author_language_activity(panel, lang_value="en", english_forum_only=False)
        it_act = author_language_activity(panel, lang_value="it", english_forum_only=False)
        base = authors_passing_cohort(en_act, thresholds) & authors_passing_cohort(it_act, thresholds)
        if design == "cross_language":
            forum_authors = _authors_with_attr(
                panel, lambda r: str(r.get("author_group", "")) == "italian_bilingual"
            )
            return base & forum_authors
        if design == "cross_language_native_it":
            native_it = _authors_with_attr(
                panel,
                lambda r: str(r.get("author_group", "")) == "italian_bilingual"
                and float(r.get("italian_share_pre", 0.0)) >= NATIVE_IT_SHARE_MIN,
            )
            return base & native_it
        lang_mix = _authors_with_attr(
            panel, lambda r: int(r.get("lang_bilingual", 0) or 0) == 1
        )
        return base & lang_mix
    raise ValueError(f"Unknown design: {design}")


def march_standardization_moments_by_lang(
    df: pd.DataFrame,
    outcome_col: str,
    *,
    rel_day_col: str = "rel_day",
    lang_col: str = "lang_comment",
) -> pd.DataFrame:
    """Function summary: per-language mean/std from pre-ban rows for z-scoring.

    Parameters:
    - df: comment panel with rel_day and lang_comment.
    - outcome_col: raw outcome column.
    - rel_day_col: event-time column (<0 is pre-ban).
    - lang_col: language grouping column.

    Returns:
    - DataFrame with lang_col, mu, sigma, n_comments.
    """
    work = df.loc[df[rel_day_col].astype(int) < 0].copy()
    rows: List[Dict[str, Any]] = []
    for lang, grp in work.groupby(work[lang_col].astype(str), observed=True):
        y = pd.to_numeric(grp[outcome_col], errors="coerce").dropna()
        if len(y) < 2:
            rows.append({lang_col: lang, "mu": 0.0, "sigma": 1.0, "n_comments": len(y)})
            continue
        sigma = float(y.std(ddof=0))
        rows.append(
            {
                lang_col: lang,
                "mu": float(y.mean()),
                "sigma": sigma if sigma > 1e-9 else 1.0,
                "n_comments": int(len(y)),
            }
        )
    return pd.DataFrame(rows)


def march_standardization_moments_pooled(
    df: pd.DataFrame,
    outcome_col: str,
    *,
    rel_day_col: str = "rel_day",
) -> pd.DataFrame:
    """Function summary: single pooled mean/std from pre-ban rows (Design 1 English-only).

    Parameters:
    - df: comment panel.
    - outcome_col: raw outcome.
    - rel_day_col: event-time column.

    Returns:
    - One-row moments DataFrame with mu, sigma, n_comments.
    """
    work = df.loc[df[rel_day_col].astype(int) < 0]
    y = pd.to_numeric(work[outcome_col], errors="coerce").dropna()
    if len(y) < 2:
        return pd.DataFrame([{"mu": 0.0, "sigma": 1.0, "n_comments": len(y)}])
    sigma = float(y.std(ddof=0))
    return pd.DataFrame(
        [{"mu": float(y.mean()), "sigma": sigma if sigma > 1e-9 else 1.0, "n_comments": int(len(y))}]
    )


def apply_standardized_outcome(
    df: pd.DataFrame,
    outcome_col: str,
    moments: pd.DataFrame,
    *,
    group_col: Optional[str] = None,
    out_col: str = "y",
) -> pd.DataFrame:
    """Function summary: z-score outcome using pre-ban moments (pooled or by language).

    Parameters:
    - df: comment panel.
    - outcome_col: raw outcome.
    - moments: standardization table from march_standardization_moments_*.
    - group_col: when set, map mu/sigma by this column; else pooled moments.
    - out_col: output column name.

    Returns:
    - Copy with standardized y column.
    """
    out = df.copy()
    raw = pd.to_numeric(out[outcome_col], errors="coerce")
    if group_col and group_col in moments.columns:
        mu_map = moments.set_index(group_col)["mu"].to_dict()
        sig_map = moments.set_index(group_col)["sigma"].to_dict()
        grp = out[group_col].astype(str)
        mu = grp.map(mu_map).fillna(0.0).astype(float)
        sig = grp.map(sig_map).fillna(1.0).astype(float).clip(lower=1e-9)
    else:
        mu = float(moments["mu"].iloc[0])
        sig = max(float(moments["sigma"].iloc[0]), 1e-9)
        mu = pd.Series(mu, index=out.index, dtype=float)
        sig = pd.Series(sig, index=out.index, dtype=float)
    out[out_col] = ((raw - mu) / sig).astype("float32")
    return out


def filter_native_control_sample(panel: pd.DataFrame) -> pd.DataFrame:
    """Function summary: English comments on English forums by bilingual or native authors.

    Parameters:
    - panel: annotated panel with roster flags.

    Returns:
    - Filtered comment DataFrame for Design 1.
    """
    work = panel.copy()
    work["lang_comment"] = work["lang_comment"].astype(str).str.lower()
    mask = (
        work["lang_comment"].eq("en")
        & work["primary_lexicon"].astype(str).eq("en")
        & work["author_group"].isin(["italian_bilingual", "native_control"])
    )
    return work.loc[mask].copy()


def filter_cross_language_sample(panel: pd.DataFrame) -> pd.DataFrame:
    """Function summary: EN/IT comments by italian_bilingual authors.

    Parameters:
    - panel: annotated panel.

    Returns:
    - Filtered comment DataFrame for Design 2.
    """
    work = panel.copy()
    work["lang_comment"] = work["lang_comment"].astype(str).str.lower()
    mask = work["author_group"].eq("italian_bilingual") & work["lang_comment"].isin(["en", "it"])
    return work.loc[mask].copy()


def filter_language_pair_sample(panel: pd.DataFrame) -> pd.DataFrame:
    """Function summary: keep EN/IT comments regardless of forum group.

    Used by all cross_language-family designs once author selection has been
    handled by the design cohort gate (which encodes any forum/native-language
    membership), so this only restricts to the EN/IT language pair.

    Parameters:
    - panel: annotated panel (already cohort-filtered to design authors).

    Returns:
    - Filtered comment DataFrame with lang_comment in {en, it}.
    """
    work = panel.copy()
    work["lang_comment"] = work["lang_comment"].astype(str).str.lower()
    return work.loc[work["lang_comment"].isin(["en", "it"])].copy()


def estimate_within_language_post(
    df: pd.DataFrame,
    *,
    lang_value: str = "it",
    y_col: str = "y",
    cluster_col: str = "author",
) -> Dict[str, Any]:
    """Function summary: Italian-language placebo — raw post effect within one language.

    Estimates y ~ post | author on the subset of comments in a single language.
    Under the ChatGPT-helps-English story, the Italian-only post effect should be
    near zero (no AI assistance for Italian), so a large coefficient flags a
    general time confound rather than a writing-assistance channel.

    Parameters:
    - df: comment panel with lang_comment, post, author, y.
    - lang_value: language to restrict to (default it).
    - y_col: standardized outcome column.
    - cluster_col: cluster id for SEs.

    Returns:
    - Result dict for the post coefficient (beta/se/pvalue/n_obs/...).
    """
    work = df.copy()
    work["lang_comment"] = work["lang_comment"].astype(str).str.lower()
    work = work.loc[work["lang_comment"] == lang_value].copy()
    work["y"] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=["y", "author", "time_id"])
    work["author"] = work["author"].astype(str)
    work["time_id"] = work["time_id"].astype(str)
    work["post"] = pd.to_numeric(work["post"], errors="coerce").fillna(0).astype(float)
    res = _feols_fit("y ~ post | author", work, "post", cluster_col)
    res["coef_name"] = "post"
    res["placebo_lang"] = lang_value
    return res


def prep_treat_design(
    df: pd.DataFrame,
    treat_col: str,
    y_col: str = "y",
    cluster_col: str = "author",
) -> pd.DataFrame:
    """Function summary: build static/event-study design matrix with generic treatment.

    Parameters:
    - df: comment panel with post, treat_col, author, time_id.
    - treat_col: treatment indicator column name.
    - y_col: outcome column.
    - cluster_col: cluster id for SEs.

    Returns:
    - Design DataFrame with y, post, IT (treatment), post_IT interaction.
    """
    work = df.copy()
    work["y"] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=["y", "author", "time_id"])
    work["author"] = work["author"].astype(str)
    work["time_id"] = work["time_id"].astype(str)
    work["post"] = work["post"].astype(float)
    work["IT"] = pd.to_numeric(work[treat_col], errors="coerce").fillna(0).astype(float)
    work["post_IT"] = work["post"] * work["IT"]
    keep = ["y", "post", "IT", "post_IT", "author", "time_id"]
    for extra in (cluster_col, "rel_day", "rel_period", "subreddit"):
        if extra in work.columns and extra not in keep:
            keep.append(extra)
    return work[keep]


def headline_outcomes_for_design(design: str) -> Tuple[str, ...]:
    """Function summary: design-specific headline outcome columns.

    Parameters:
    - design: native_control or a cross_language-family design.

    Returns:
    - Tuple of outcome column names for estimation.
    """
    if design in CROSS_LANGUAGE_DESIGNS:
        return CROSS_LANGUAGE_HEADLINE_OUTCOMES
    if design == "native_control":
        return HEADLINE_OUTCOMES
    raise ValueError(f"Unknown design: {design}")


def outcome_caveat(outcome_id: str) -> str:
    """Function summary: optional caveat text for an outcome (empty if none).

    Parameters:
    - outcome_id: outcome column name.

    Returns:
    - Caveat string or empty string.
    """
    return OUTCOME_CAVEATS.get(outcome_id, "")


def estimate_static_post_treat(
    df: pd.DataFrame,
    treat_col: str,
    y_col: str = "y",
    cluster_col: str = "author",
    *,
    include_post_main: bool = True,
    include_treat_main: bool = False,
) -> Dict[str, Any]:
    """Function summary: static DiD y ~ post [+ treat] + post:treat | author + time_id.

    Parameters:
    - df: comment panel.
    - treat_col: treatment indicator.
    - y_col: outcome column.
    - cluster_col: cluster for CRV1 SEs.
    - include_post_main: when False, absorb post into time_id only (post_IT variant).
    - include_treat_main: when True, add treat main effect (needed when treat varies
      within author, e.g. cross_language is_english).

    Returns:
    - Result dict for post:treat coefficient.
    """
    work = prep_treat_design(df, treat_col, y_col, cluster_col)
    if include_treat_main:
        if include_post_main:
            formula = "y ~ post + IT + post_IT | author + time_id"
        else:
            formula = "y ~ IT + post_IT | author + time_id"
    elif include_post_main:
        formula = "y ~ post + post_IT | author + time_id"
    else:
        formula = "y ~ post_IT | author + time_id"
    res = _feols_fit(formula, work, "post_IT", cluster_col)
    res["static_variant"] = "post_treat_time_fe"
    res["coef_name"] = "post:treat"
    res["treat_col"] = treat_col
    return res


def static_es_post_avg(es_df: pd.DataFrame) -> Dict[str, Any]:
    """Function summary: weighted average of post-ban event-study bin coefficients.

    Parameters:
    - es_df: event-study frame with rel_period, gamma, se columns.

    Returns:
    - Dict with beta (weighted mean gamma), se (sqrt of inverse-variance weights),
      n_bins, and estimation_note.
    """
    if es_df.empty or "gamma" not in es_df.columns:
        return {"beta": float("nan"), "se": float("nan"), "n_bins": 0, "estimation_note": "empty"}
    post = es_df.loc[es_df["rel_period"].astype(int) >= 0].copy()
    if post.empty:
        return {"beta": float("nan"), "se": float("nan"), "n_bins": 0, "estimation_note": "no_post_bins"}
    g = post["gamma"].astype(float)
    se = post["se"].astype(float).replace(0, np.nan)
    w = 1.0 / (se**2)
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0)
    if w.sum() <= 0:
        beta = float(g.mean())
        se_out = float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else float("nan")
    else:
        beta = float((w * g).sum() / w.sum())
        se_out = float(np.sqrt(1.0 / w.sum()))
    return {
        "beta": beta,
        "se": se_out,
        "n_bins": int(len(post)),
        "estimation_note": "ok",
    }


def _within_author_diff_cells(
    df: pd.DataFrame,
    *,
    y_col: str = "y",
    rel_col: str = "rel_period",
    bin_days: int = 3,
    window: int = 30,
) -> pd.DataFrame:
    """Function summary: author x bin EN-vs-IT difference cells with pair counts.

    Parameters:
    - df: comment panel with author, lang_comment or is_english, rel_period/rel_day.
    - y_col: standardized outcome column.
    - rel_col: event-time bin column.
    - bin_days: bin width for rel_period fallback.
    - window: trim rel_col to [-window, window].

    Returns:
    - DataFrame with author, rel_period, d, w, n_en, n_it, post, subreddit (if present).
    """
    work = df.copy()
    work["y"] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=["y", "author"])
    work["author"] = work["author"].astype(str)
    if rel_col not in work.columns:
        work["rel_period"] = (work["rel_day"] // bin_days).astype(int)
        rel_col = "rel_period"
    work = work[work[rel_col].between(-window, window)]
    if work.empty:
        return pd.DataFrame()
    if "is_english" in work.columns:
        work["is_en"] = work["is_english"].astype(int)
    else:
        work["is_en"] = (work["lang_comment"].astype(str).str.lower() == "en").astype(int)
    rows: List[Dict[str, Any]] = []
    for (author, rel), grp in work.groupby(["author", rel_col], observed=True):
        en = grp.loc[grp["is_en"] == 1, "y"]
        it = grp.loc[grp["is_en"] == 0, "y"]
        n_en = int(len(en))
        n_it = int(len(it))
        if n_en < 1 or n_it < 1:
            continue
        post_val = int(grp["post"].astype(int).max()) if "post" in grp.columns else int(rel) >= 0
        rows.append(
            {
                "author": str(author),
                rel_col: int(rel),
                "rel_day": int(rel) * bin_days if rel_col == "rel_period" else int(rel),
                "d": float(en.mean() - it.mean()),
                "w": float(min(n_en, n_it)),
                "n_en": n_en,
                "n_it": n_it,
                "n_pairs": int(min(n_en, n_it)),
                "post": float(post_val),
            }
        )
    return pd.DataFrame(rows)


def estimate_within_author_diff_event_study(
    df: pd.DataFrame,
    y_col: str = "y",
    rel_col: str = "rel_period",
    ref_period: int = -1,
    window: int = 30,
    cluster_col: str = "author",
    bin_days: int = 3,
    *,
    weighted: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Function summary: within-author EN-vs-IT difference event study (headline Design 2).

    Per author x bin: d = mean(y|EN) - mean(y|IT), requiring >=1 comment each language.
    Estimates d ~ i(rel_period, ref) | author with weights min(n_EN, n_IT).

    Parameters:
    - df: comment panel with language and outcome.
    - y_col: outcome column.
    - rel_col: event-time bin column.
    - ref_period: omitted reference bin.
    - window: trim rel_col to [-window, window].
    - cluster_col: cluster for CRV1 SEs (author).
    - bin_days: rel_day display multiplier.
    - weighted: when True, weight by min(n_EN, n_IT); else unweighted.

    Returns:
    - Tuple (meta dict, es_df with rel_period, gamma, se, n_authors, n_pairs).
    """
    cell = _within_author_diff_cells(
        df, y_col=y_col, rel_col=rel_col, bin_days=bin_days, window=window
    )
    meta: Dict[str, Any] = {
        "n_obs": int(len(cell)),
        "n_authors": int(cell["author"].nunique()) if not cell.empty else 0,
        "n_cells": int(len(cell)),
    }
    if cell.empty or cell["author"].nunique() < 3 or cell[rel_col].nunique() < 2:
        return {**meta, "estimation_note": "insufficient_cells"}, pd.DataFrame()
    try:
        from pyfixest.estimation import feols
    except ImportError:
        return {**meta, "estimation_note": "pyfixest_missing"}, pd.DataFrame()
    vcov: Any = {"CRV1": cluster_col} if cluster_col in cell.columns else "iid"
    weights_col = "w" if weighted else None
    if weights_col:
        cell = cell.copy()
        cell["w"] = cell["w"].astype(float).clip(lower=1e-9)
    formula = f"d ~ i({rel_col}, ref={ref_period}) | author"
    try:
        feols_kw: Dict[str, Any] = {"vcov": vcov}
        if weights_col:
            feols_kw["weights"] = weights_col
        fit = feols(formula, data=cell, **feols_kw)
    except Exception:
        return {**meta, "estimation_note": "estimation_error"}, pd.DataFrame()
    rows = _es_rows_from_fit(fit, rel_col, ref_period, bin_days=bin_days, gamma_col="gamma")
    es_df = pd.DataFrame(rows).sort_values(rel_col) if rows else pd.DataFrame()
    if not es_df.empty:
        counts = (
            cell.groupby(rel_col, observed=True)
            .agg(n_authors=("author", "nunique"), n_pairs=("n_pairs", "sum"))
            .reset_index()
        )
        es_df = es_df.merge(counts, on=rel_col, how="left")
    meta["estimation_note"] = "ok"
    meta["weighted"] = weighted
    return meta, es_df


def estimate_within_author_diff_static(
    df: pd.DataFrame,
    y_col: str = "y",
    cluster_col: str = "author",
    *,
    rel_col: str = "rel_period",
    bin_days: int = 3,
    window: int = 30,
    weighted: bool = True,
) -> Dict[str, Any]:
    """Function summary: static within-author diff d ~ post | author (mean post d - pre d).

    Parameters:
    - df: comment panel.
    - y_col: outcome column.
    - cluster_col: cluster for CRV1 SEs.
    - rel_col: event-time bin column.
    - bin_days: bin width for rel_period fallback.
    - window: trim rel_col.
    - weighted: when True, weight by min(n_EN, n_IT).

    Returns:
    - Result dict for post coefficient on d cells.
    """
    cell = _within_author_diff_cells(
        df, y_col=y_col, rel_col=rel_col, bin_days=bin_days, window=window
    )
    if cell.empty:
        return {
            "beta": float("nan"),
            "se": float("nan"),
            "pvalue": float("nan"),
            "n_obs": 0,
            "n_clusters": 0,
            "estimation_note": "insufficient_cells",
        }
    weights_col = "w" if weighted else None
    res = _feols_fit("d ~ post | author", cell, "post", cluster_col, weights_col=weights_col)
    res["static_variant"] = "within_author_diff"
    res["coef_name"] = "post"
    res["n_cells"] = int(len(cell))
    res["n_authors"] = int(cell["author"].nunique())
    res["weighted"] = weighted
    return res


def estimate_treat_event_study(
    df: pd.DataFrame,
    treat_col: str,
    y_col: str = "y",
    rel_col: str = "rel_period",
    ref_period: int = -1,
    window: int = 30,
    cluster_col: str = "author",
    bin_days: int = 3,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Function summary: y ~ i(rel_period, treat, ref) | author + time_id event study.

    Parameters:
    - df: comment panel.
    - treat_col: treatment indicator (must vary within author for clean dynamics).
    - y_col: outcome.
    - rel_col: event-time bin column.
    - ref_period: omitted reference bin.
    - window: trim rel_col to [-window, window].
    - cluster_col: cluster for SEs.
    - bin_days: rel_day display multiplier.

    Returns:
    - Tuple (summary dict, coefficient DataFrame).
    """
    work = prep_treat_design(df, treat_col, y_col, cluster_col)
    work = work.rename(columns={"IT": "TREAT_PLACEHOLDER"})
    work["IT"] = work["TREAT_PLACEHOLDER"]
    work = work.drop(columns=["TREAT_PLACEHOLDER"])
    return estimate_comment_it_event_study(
        work,
        y_col="y",
        rel_col=rel_col,
        ref_period=ref_period,
        window=window,
        cluster_col=cluster_col,
        bin_days=bin_days,
    )


def estimate_fd_event_study(
    df: pd.DataFrame,
    treat_col: str,
    y_col: str = "y",
    ref_period: int = -1,
    window: int = 30,
    bin_days: int = 3,
    baseline: str = "ref",
    cluster_col: str = "author",
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Function summary: first-difference event study for author-invariant treatment.

    Parameters:
    - df: comment panel.
    - treat_col: treatment indicator (author-invariant).
    - y_col: outcome.
    - ref_period: reference bin for FD_ref.
    - window: event window trim.
    - bin_days: bin width.
    - baseline: ref or preban_mean.
    - cluster_col: cluster id.

    Returns:
    - Tuple (meta dict, event-study DataFrame).
    """
    from scripts.diagnostics.event_study_level_robustness import (
        estimate_first_difference_event_study,
    )

    work = prep_treat_design(df, treat_col, y_col, cluster_col)
    pool = work.copy()
    pool["y"] = work["y"]
    return estimate_first_difference_event_study(
        pool,
        ref_period=ref_period,
        window=window,
        bin_days=bin_days,
        baseline=baseline,
        y_col="y",
        cluster_col=cluster_col,
    )


def outcome_label(outcome_id: str) -> str:
    """Function summary: human-readable outcome label for plots and tables.

    Parameters:
    - outcome_id: column name.

    Returns:
    - Display label string.
    """
    labels = {
        "readability": "Readability",
        "ttr_50w": "Type-token ratio (50w)",
        "avg_words_per_sentence_comment": "Avg words per sentence",
        "log_len": "Log comment length",
        "ai_sentence_length_variance": "Sentence length variance",
        "style_index_llm": "Style index (LLM)",
        "net_ideology": "Net ideology",
        "aggression_rate_100w": "Aggression rate (per 100w)",
        "negative_rate_100w": "Negative rate (per 100w)",
        "extremity": "Ideological extremity",
        "sem_axis_ideology": "Semantic axis: ideology",
        "sem_axis_emotion": "Semantic axis: emotion",
        "sem_axis_aggression": "Semantic axis: aggression",
    }
    return labels.get(outcome_id, outcome_id)


def roster_summary(roster: pd.DataFrame) -> Dict[str, int]:
    """Function summary: count authors by roster group.

    Parameters:
    - roster: classify_author_roster output.

    Returns:
    - Dict of group -> count.
    """
    if roster.empty:
        return {}
    return roster["author_group"].value_counts().to_dict()
