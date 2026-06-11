"""
Write DiD tables and event-study figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
import seaborn as sns

from src.did.figure_readmes import write_overview_readme
from src.did.outcomes import (
    FAMILY_FIGURE_DIRS,
    FIRST_STAGE_OUTCOMES,
    HEADLINE_FOREST_STRATEGIES,
    HEADLINE_OUTCOMES,
    SUMMARY_THEMES,
    outcome_label,
)
from src.did.specs import (
    PHASE_JOINT_SPECS,
    PLOT_STRATEGY_GROUPS,
    spec_label_parenthetical,
    spec_label_short,
    strategy_label,
)
from src.plotting.thesis_theme import (
    THESIS_COEF_MARKER,
    THESIS_ITALY,
    THESIS_CONTROL,
    shade_ban_window,
    thesis_title_for_outcome,
    xlabel_event_study,
    ylabel_italy_bin_coefficient,
)

HEADLINE_TWFE_SPEC = "ban_in_effect"

HEADLINE_STRATEGY_ORDER = list(PLOT_STRATEGY_GROUPS["headline"])

FOREST_XLIM_BY_OUTCOME: dict[str, tuple[float, float]] = {
    "sem_axis_ideology": (-0.02, 0.02),
    "sem_axis_aggression": (-0.02, 0.02),
    "ai_style_rate": (-0.05, 0.05),
    "em_dash_rate": (-0.05, 0.05),
    "wf_extremity_z": (-0.3, 0.3),
}

DDD_LEFT_OUTCOMES = (
    "em_dash_rate",
    "ai_style_rate",
    "salience_rate",
    "aggression_rate",
    "avg_wps",
)

DDD_RIGHT_OUTCOMES = ("sentence_len_var", "exclamation_rate")


def coefplot_strategies_for_family(family: str) -> List[str]:
    """Function summary: strategy_ids for family headline coefplots.

    Parameters:
    - family: outcome family id.

    Returns:
    - List of strategy_id strings for plot_coef_comparison.
    """
    if family in ("wordfish_author", "wordfish_author_v2"):
        return list(PLOT_STRATEGY_GROUPS["author_it"]) + list(PLOT_STRATEGY_GROUPS["author_cross"])
    if family in ("lexical_comment", "semantic_axis_comment", "lexical_author_day", "semantic_axis_author_day"):
        return list(PLOT_STRATEGY_GROUPS["headline"]) + list(PLOT_STRATEGY_GROUPS["italy_only"])
    return list(PLOT_STRATEGY_GROUPS["headline"]) + list(PLOT_STRATEGY_GROUPS["italy_only"])

PHASE_JOINT_PLOT_COLORS: dict[str, str] = {
    "phase_joint_short": "#1d3557",
    "phase_joint_medium": "#457b9d",
    "phase_joint_long": "#89c2d9",
    "phase_joint_lift": "#e76f51",
}


def _drop_degenerate_lift_rows(sub: pd.DataFrame) -> pd.DataFrame:
    """Function summary: exclude phase_joint_lift / post_lift rows marked degenerate_collinear_lift."""
    if sub.empty or "estimation_note" not in sub.columns:
        return sub
    mask = sub["estimation_note"].astype(str) == "degenerate_collinear_lift"
    return sub.loc[~mask].copy()


def _filter_headline_spec(sub: pd.DataFrame, *, fallback: str = "full_ban") -> pd.DataFrame:
    """Function summary: keep ban_in_effect rows, falling back to full_ban when absent."""
    sub = _ensure_spec_column(sub)
    primary = sub[sub["spec"].astype(str) == HEADLINE_TWFE_SPEC]
    if not primary.empty:
        return primary
    return sub[sub["spec"].astype(str) == fallback]


def _ensure_spec_column(sub: pd.DataFrame) -> pd.DataFrame:
    """Function summary: guarantee a spec column (default full_ban) for plotting.

    Parameters:
    - sub: summary slice.

    Returns:
    - Copy with spec column.
    """
    out = sub.copy()
    if "spec" not in out.columns:
        out["spec"] = "full_ban"
    return out


def disambiguate_with_spec(base_labels: pd.Series, specs: pd.Series) -> pd.Series:
    """Function summary: append (full ban) / (early ban) when a base label repeats.

    Parameters:
    - base_labels: strategy or outcome display names.
    - specs: post-window spec per row.

    Returns:
    - Labels with parenthetical spec suffix only on duplicated base names.
    """
    base = base_labels.astype(str)
    spec = specs.astype(str)
    dup = base.duplicated(keep=False)
    suffixed = base + " " + spec.map(spec_label_parenthetical)
    return base.where(~dup, suffixed)


def _strategy_plot_labels(sub: pd.DataFrame, *, short: bool = True) -> pd.Series:
    """Function summary: strategy y-axis labels with post-window disambiguation."""
    work = _ensure_spec_column(sub)
    base = work["strategy_id"].astype(str).map(lambda s: strategy_label(s, short=short))
    return disambiguate_with_spec(base, work["spec"])


def _outcome_plot_labels(sub: pd.DataFrame, *, short: bool = True) -> pd.Series:
    """Function summary: outcome y-axis labels with post-window disambiguation."""
    work = _ensure_spec_column(sub)
    base = work["outcome_id"].astype(str).map(lambda o: outcome_label(o, short=short))
    return disambiguate_with_spec(base, work["spec"])


def figure_path(fig_dir: Path, family: str, plot_type: str, outcome_id: str, suffix: str = "png") -> Path:
    """Function summary: nested figure path under family/plot_type/.

    Parameters:
    - fig_dir: did figures root.
    - family: outcome family id.
    - plot_type: coefplots_headline, event_study, etc.
    - outcome_id: outcome slug.
    - suffix: file extension.

    Returns:
    - Full output path.
    """
    sub = FAMILY_FIGURE_DIRS.get(family, family)
    return fig_dir / sub / plot_type / f"{outcome_id}.{suffix}"


def write_summary_rows(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Function summary: append or write did_summary.csv."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if out_path.is_file():
        old = pd.read_csv(out_path)
        combined = pd.concat([old, new], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=[c for c in ("outcome_id", "strategy_id", "spec") if c in combined.columns],
            keep="last",
        )
        combined.to_csv(out_path, index=False)
    else:
        new.to_csv(out_path, index=False)


def add_strategy_labels(summary: pd.DataFrame) -> pd.DataFrame:
    """Function summary: add strategy_label column for reporting."""
    out = summary.copy()
    out["strategy_label"] = out["strategy_id"].astype(str).map(strategy_label)
    return out


def _strategy_sort_key(strategy_id: str) -> tuple:
    """Function summary: sort headline strategies first, then alphabetically."""
    sid = str(strategy_id)
    if sid in HEADLINE_STRATEGY_ORDER:
        return (0, HEADLINE_STRATEGY_ORDER.index(sid))
    return (1, sid)


def headline_pvalue(row: pd.Series) -> float:
    """Function summary: primary inference p (placebo-space or WCB), else cluster p for display."""
    role = str(row.get("inference_role", "") or "")
    if role == "descriptive":
        p = row.get("p_placebo_space", row.get("perm_p", float("nan")))
        if pd.notna(p):
            return float(p)
    if role == "primary":
        p = row.get("wild_p", float("nan"))
        if pd.notna(p):
            return float(p)
    try:
        return float(row.get("pvalue", float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def _format_beta_line(row: pd.Series) -> str:
    """Function summary: one-line TWFE result for text summaries."""
    label = row.get("strategy_label") or strategy_label(str(row.get("strategy_id", "")))
    note = str(row.get("estimation_note", "ok"))
    beta = row.get("beta")
    if pd.isna(beta):
        return f"  {label}: (no estimate) [{note}]"
    se = row.get("se", float("nan"))
    p_cluster = row.get("pvalue", float("nan"))
    pval = headline_pvalue(row)
    role = str(row.get("inference_role", "") or "")
    n_obs = row.get("n_obs", "")
    n_cl = row.get("n_clusters", "")
    flags: List[str] = []
    if note and note != "ok":
        flags.append(note)
    if int(row.get("sign_only_cross_country", 0) or 0):
        flags.append("sign-only cross-country")
    sid = str(row.get("strategy_id", ""))
    if sid in HEADLINE_STRATEGY_ORDER:
        flags.append("headline")
    pq = str(row.get("pretrend_quality", "") or "")
    if pq and pq != "ok":
        flags.append(f"pretrend:{pq}")
    if role == "descriptive" and pd.notna(p_cluster):
        flags.append(f"p_cluster={float(p_cluster):.4g}")
    if role == "primary" and pd.notna(row.get("wild_p")):
        flags.append("p_wcb")
    flag_s = f" [{', '.join(flags)}]" if flags else ""
    return (
        f"  {label}: β={beta:.4f} (SE={se:.4f}), p={pval:.4g}, "
        f"N={n_obs}, clusters={n_cl}{flag_s}"
    )


def _write_summary_txt(sub: pd.DataFrame, path: Path, title: str, launch_date: str) -> None:
    """Function summary: write human-readable DiD summary text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        title,
        "=" * min(72, len(title)),
        f"Ban date: {launch_date}",
        "",
    ]
    if sub.empty:
        lines.append("(no estimates)")
    else:
        for outcome_id, grp in sub.groupby("outcome_id", sort=True):
            fam = grp["outcome_family"].iloc[0] if "outcome_family" in grp.columns else ""
            lines.append(f"Outcome: {outcome_id} ({fam})")
            ordered = grp.copy()
            ordered["_sort"] = ordered["strategy_id"].map(_strategy_sort_key)
            ordered = ordered.sort_values("_sort").drop(columns="_sort", errors="ignore")
            for _, row in ordered.iterrows():
                lines.append(_format_beta_line(row))
            lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


SUMMARY_DEDUPE_KEYS = ("outcome_id", "strategy_id", "spec", "weights")


def normalize_summary_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: coerce weights column to empty string (CSV NaN round-trip safe).

    Parameters:
    - df: did_summary-style frame.

    Returns:
    - Copy with weights normalized for stable dedupe keys.
    """
    out = df.copy()
    if "weights" in out.columns:
        out["weights"] = out["weights"].fillna("").astype(str)
    return out


def dedupe_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: drop duplicate summary rows after normalizing weights.

    Parameters:
    - df: did_summary-style frame.

    Returns:
    - Deduped copy (last row wins per outcome/strategy/spec/weights).
    """
    out = normalize_summary_weights(df)
    keys = [c for c in SUMMARY_DEDUPE_KEYS if c in out.columns]
    return out.drop_duplicates(subset=keys, keep="last")


def write_summary_exports(
    summary_df: pd.DataFrame,
    summary_dir: Path,
    launch_date: str,
) -> None:
    """Function summary: write sliced CSV and TXT summaries under estimates/summary/.

    Parameters:
    - summary_df: full did_summary table.
    - summary_dir: estimates/summary/ path.
    - launch_date: ban launch YYYY-MM-DD for headers.
    """
    if summary_df.empty:
        return
    summary_df = dedupe_summary_rows(summary_df)
    labeled = add_strategy_labels(summary_df)
    summary_dir.mkdir(parents=True, exist_ok=True)

    master_csv, master_labeled = (
        summary_dir / "did_summary.csv",
        summary_dir / "did_summary_labeled.csv",
    )
    summary_df.to_csv(master_csv, index=False)
    labeled.to_csv(master_labeled, index=False)
    _write_summary_txt(
        labeled,
        summary_dir / "did_summary_all.txt",
        "DiD summary — all outcomes",
        launch_date,
    )

    by_family = summary_dir / "by_family"
    by_outcome = summary_dir / "by_outcome"
    by_theme = summary_dir / "by_theme"
    for sub in (by_family, by_outcome, by_theme):
        sub.mkdir(parents=True, exist_ok=True)
    for family, grp in summary_df.groupby("outcome_family", sort=True):
        grp.to_csv(by_family / f"{family}.csv", index=False)
        _write_summary_txt(
            add_strategy_labels(grp),
            by_family / f"{family}.txt",
            f"DiD summary — {family}",
            launch_date,
        )

    for outcome_id, grp in summary_df.groupby("outcome_id", sort=True):
        grp.to_csv(by_outcome / f"{outcome_id}.csv", index=False)
        fam = grp["outcome_family"].iloc[0] if "outcome_family" in grp.columns else ""
        _write_summary_txt(
            add_strategy_labels(grp),
            by_outcome / f"{outcome_id}.txt",
            f"DiD summary — {outcome_id} ({fam})",
            launch_date,
        )

    for theme, outcome_ids in SUMMARY_THEMES.items():
        if theme == "all":
            sub = summary_df
        elif theme == "wordfish":
            sub = summary_df[summary_df["outcome_family"].astype(str).str.startswith("wordfish")]
        elif theme == "lexical":
            sub = summary_df[summary_df["outcome_family"] == "lexical"]
        else:
            sub = summary_df[summary_df["outcome_id"].isin(outcome_ids)]
        if sub.empty:
            continue
        sub.to_csv(by_theme / f"{theme}.csv", index=False)
        _write_summary_txt(
            add_strategy_labels(sub),
            by_theme / f"{theme}.txt",
            f"DiD summary — theme: {theme}",
            launch_date,
        )


def _subtitle_for_family(family: str) -> str:
    """Function summary: family-specific plot subtitle."""
    if family == "wordfish_forum":
        return "Cross-language: sign/direction only"
    if family == "wordfish_forum_v2":
        return "Forum θ (v2); cross-language sign/direction only; gate may fail"
    if family == "wordfish_author":
        return "Italian-writing authors; week bins"
    if family == "wordfish_author_v2":
        return "Author θ (v2); validation-gated ideology interpretation"
    return ""


EVENT_STUDY_OVERLAY_STYLES: tuple[dict[str, Any], ...] = (
    {"color": "black", "marker": "o", "mfc": "white", "label": ""},
    {"color": "#1f4e79", "marker": "D", "mfc": "white", "label": ""},
    {"color": "#555555", "marker": "s", "mfc": "white", "label": ""},
    {"color": "#8b0000", "marker": "P", "mfc": "white", "label": ""},
    {"color": "#2d6a4f", "marker": "^", "mfc": "white", "label": ""},
)

EventStudyMarker = Tuple[int, str]

ITALY_THESIS_POLITICAL_EVENT_MARKERS: tuple[EventStudyMarker, ...] = (
    (2, "Friuli vote"),
    (5, "Berlusconi ICU"),
    (11, "migration crisis"),
    (25, "Liberation Day"),
    (28, "ban lifted"),
)


@dataclass(frozen=True)
class EventStudySeries:
    """Function summary: one strategy's event-study coefficients for overlay plotting."""

    label: str
    es_df: pd.DataFrame
    rel_col: str = "event_time"


def _prepare_event_study_plot_df(
    es_df: pd.DataFrame,
    rel_col: str,
    ref_time: int = -1,
) -> pd.DataFrame:
    """Function summary: sort ES rows and append normalized reference period at coef=0."""
    if es_df.empty:
        return es_df
    time_col = rel_col if rel_col in es_df.columns else "rel_day"
    work = es_df.sort_values(time_col).copy()
    work = work.rename(columns={time_col: "event_time"})
    if ref_time not in set(work["event_time"].astype(int)):
        ref_row = {
            "event_time": ref_time,
            "gamma": 0.0,
            "se": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
        }
        work = pd.concat([work, pd.DataFrame([ref_row])], ignore_index=True)
    return work.sort_values("event_time")


def apply_event_study_axes_style(
    ax: plt.Axes,
    *,
    bin_days: int = 1,
    x_scale: str = "days",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """Function summary: thesis event-study axes (zero line, ban window, standardized labels).

    Parameters:
    - ax: matplotlib axes.
    - bin_days: 1 or 3 for x-label parenthetical and ban-boundary math.
    - x_scale: 'days' or 'period' for ban guide positions on the x-axis.
    - xlabel: optional override; default from xlabel_event_study(bin_days).
    - ylabel: optional override; default Italy × bin coefficient.

    Returns:
    - None; mutates ax in place.
    """
    ax.axhline(0, color="black", linewidth=0.9, zorder=4)
    shade_ban_window(
        ax,
        mode="event_study",
        bin_days=int(bin_days),
        x_scale="period" if x_scale == "period" else "days",
        zorder=0,
    )
    ax.set_xlabel(xlabel if xlabel is not None else xlabel_event_study(int(bin_days)))
    ax.set_ylabel(ylabel if ylabel is not None else ylabel_italy_bin_coefficient())
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")


def apply_event_study_markers(ax: plt.Axes, markers: Sequence[EventStudyMarker]) -> None:
    """Function summary: overlay thin vertical dotted lines and rotated labels on an event-study axes.

    Parameters:
    - ax: matplotlib axes with coefficients already drawn.
    - markers: sequence of (rel_day, label) pairs (days relative to 2023-03-31 ban launch).

    Returns:
    - None; mutates ax in place; labels are drawn inside the plot area.
    """
    if not markers:
        return
    label_trans = blended_transform_factory(ax.transData, ax.transAxes)
    y_levels = (0.96, 0.78)
    for idx, (rel_day, label) in enumerate(markers):
        ax.axvline(rel_day, color="0.55", linestyle=":", linewidth=0.8, zorder=1)
        ax.text(
            rel_day,
            y_levels[idx % len(y_levels)],
            label,
            transform=label_trans,
            rotation=90,
            va="top",
            ha="center",
            fontsize=6,
            color="0.45",
            clip_on=True,
            zorder=4,
        )


def plot_event_study(
    es_df: pd.DataFrame,
    outcome_id: str,
    out_path: Path,
    launch_label: str = "2023-03-31",
    title: Optional[str] = None,
    subtitle: str = "",
    rel_col: str = "rel_day",
    event_markers: Sequence[EventStudyMarker] | None = None,
    bin_days: int = 1,
) -> None:
    """Function summary: single-series classic event-study plot.

    Parameters:
    - es_df: event-study coefficient table.
    - outcome_id: outcome slug for title/labels.
    - out_path: PNG output path.
    - launch_label: unused (kept for API compatibility).
    - title: optional plot title override.
    - subtitle: unused (kept for API compatibility).
    - rel_col: relative-day column name in es_df.
    - event_markers: optional (rel_day, label) pairs for thesis-style political-event overlays.
    - bin_days: calendar bin width for axis label and ban guides (1 or 3).

    Returns:
    - None; writes PNG to out_path.
    """
    del launch_label, subtitle
    if es_df.empty:
        return
    plot = _prepare_event_study_plot_df(es_df, rel_col)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    x_scale = "period" if rel_col == "rel_period" else "days"
    mask = plot["se"].fillna(0) > 0
    ax.errorbar(
        plot.loc[mask, "event_time"],
        plot.loc[mask, "gamma"],
        yerr=1.96 * plot.loc[mask, "se"],
        fmt="none",
        ecolor=THESIS_COEF_MARKER,
        capsize=3,
        zorder=5,
    )
    ax.plot(
        plot["event_time"],
        plot["gamma"],
        linestyle="none",
        marker="o",
        markerfacecolor="white",
        markeredgecolor=THESIS_COEF_MARKER,
        markeredgewidth=1.0,
        zorder=6,
    )
    apply_event_study_axes_style(ax, bin_days=bin_days, x_scale=x_scale)
    if event_markers:
        apply_event_study_markers(ax, event_markers)
    resolved_title = title
    if resolved_title is None:
        resolved_title = thesis_title_for_outcome(
            outcome_id,
            fallback=outcome_label(outcome_id, short=True),
        )
    ax.set_title(resolved_title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_event_study_overlay(
    series_list: Sequence[EventStudySeries],
    outcome_id: str,
    out_path: Path,
    title: Optional[str] = None,
    max_series: int = 5,
    bin_days: int = 1,
) -> None:
    """Function summary: overlay up to max_series event studies with horizontal dodge."""
    usable = []
    for s in series_list:
        if s.es_df is None or s.es_df.empty:
            continue
        se = pd.to_numeric(s.es_df.get("se"), errors="coerce")
        if int((se.notna() & (se > 1e-12)).sum()) < 2:
            continue
        usable.append(s)
    if not usable:
        return
    usable = usable[:max_series]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    all_times = sorted(
        {
            int(v)
            for s in usable
            for v in _prepare_event_study_plot_df(s.es_df, s.rel_col)["event_time"].astype(int)
        }
    )
    if not all_times:
        plt.close(fig)
        return
    n = len(usable)
    dodge_span = 0.85
    offsets = np.linspace(-dodge_span / 2, dodge_span / 2, n) if n > 1 else [0.0]
    for idx, (series, style) in enumerate(zip(usable, EVENT_STUDY_OVERLAY_STYLES)):
        plot = _prepare_event_study_plot_df(series.es_df, series.rel_col)
        off = offsets[idx]
        x = plot["event_time"].astype(float) + off
        se = plot["se"].fillna(0)
        mask = se > 0
        ax.errorbar(
            x[mask],
            plot.loc[mask, "gamma"],
            yerr=1.96 * se[mask],
            fmt="none",
            ecolor=style["color"],
            capsize=3,
            elinewidth=1.0,
            zorder=2 + idx,
        )
        ax.plot(
            x,
            plot["gamma"],
            linestyle="none",
            marker=style["marker"],
            markerfacecolor=style.get("mfc", "white"),
            markeredgecolor=style["color"],
            markeredgewidth=1.0,
            label=series.label,
            zorder=3 + idx,
        )
    x_scale = "period" if any(s.rel_col == "rel_period" for s in usable) else "days"
    bd = 3 if x_scale == "period" else int(bin_days)
    apply_event_study_axes_style(ax, bin_days=bd, x_scale=x_scale)
    ax.set_xticks(all_times)
    resolved_title = title
    if resolved_title is None:
        resolved_title = thesis_title_for_outcome(
            outcome_id,
            fallback=outcome_label(outcome_id, short=True),
        )
    ax.set_title(resolved_title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(n, 3), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _post_ban_gamma_summary(es_df: pd.DataFrame, rel_col: str) -> tuple[float, float]:
    """Function summary: mean post-ban treat×time γ and two-sided p from normal SE."""
    if es_df.empty:
        return float("nan"), float("nan")
    time_col = rel_col if rel_col in es_df.columns else "rel_day"
    post = es_df[es_df[time_col].astype(int) >= 0]
    if post.empty:
        return float("nan"), float("nan")
    beta = float(post["gamma"].mean())
    se_vals = post["se"].replace(0, np.nan).dropna()
    if se_vals.empty:
        return beta, float("nan")
    se_mean = float(se_vals.mean())
    if se_mean <= 0 or np.isnan(se_mean):
        return beta, float("nan")
    from scipy import stats

    z = abs(beta / se_mean)
    p = float(2 * (1 - stats.norm.cdf(z)))
    return beta, p


def _format_p_value(p: float) -> str:
    """Function summary: compact p-value string for figure titles."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "p=—"
    if p < 0.001:
        return "p<.001"
    return f"p={p:.3f}".lstrip("0") if p < 1 else f"p={p:.2f}"


def tail_shift_interpretation_title(
    left_beta: float,
    left_p: float,
    right_beta: float,
    right_p: float,
) -> str:
    """Function summary: one-line title for dual-tail semantic ideology event study."""
    def _part(label: str, beta: float, p: float, direction: str) -> str:
        if beta is None or (isinstance(beta, float) and np.isnan(beta)):
            return f"{label} (n/a)"
        sign = "+" if beta >= 0 else ""
        return f"{label} {direction} ({sign}{beta:.3f}, {_format_p_value(p)})"

    left_dir = "up" if (not np.isnan(left_beta) and left_beta > 0) else "down"
    right_dir = "up" if (not np.isnan(right_beta) and right_beta > 0) else "down"
    left_part = _part("extreme-LEFT share", left_beta, left_p, left_dir)
    right_part = _part("extreme-RIGHT share", right_beta, right_p, right_dir)
    if (
        not np.isnan(left_beta)
        and not np.isnan(right_beta)
        and left_beta > 0
        and right_beta < 0
    ):
        interp = "leftward location shift, not symmetric tail-widening"
    elif (
        not np.isnan(left_beta)
        and not np.isnan(right_beta)
        and left_beta > 0
        and right_beta > 0
    ):
        interp = "both tails up (dispersion / tail-widening)"
    elif (
        not np.isnan(left_beta)
        and not np.isnan(right_beta)
        and left_beta < 0
        and right_beta < 0
    ):
        interp = "both tails down"
    else:
        interp = "mixed tail movement"
    return f"Semantic axis: {left_part}, {right_part} → {interp}"


def plot_sem_axis_ideology_tail_shift_event_study(
    left: EventStudySeries,
    right: EventStudySeries,
    out_path: Path,
    *,
    bin_days: int = 1,
    strategy_label_text: str = "",
) -> None:
    """Function summary: dual-tail event study (p10 left vs p90 right) on one axes.

    Parameters:
    - left: extreme-left tail EventStudySeries.
    - right: extreme-right tail EventStudySeries.
    - out_path: PNG destination.
    - bin_days: 1 or 3 (controls x-axis label).
    - strategy_label_text: optional subtitle context (unused in title if empty).

    Returns:
    - None; skips if both series empty.
    """
    del strategy_label_text
    if (left.es_df is None or left.es_df.empty) and (right.es_df is None or right.es_df.empty):
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    styles = (
        {
            "color": THESIS_ITALY,
            "marker": "o",
            "mfc": "white",
            "label": "extreme-LEFT tail share (sem axis, <p10)",
        },
        {
            "color": THESIS_CONTROL,
            "marker": "s",
            "mfc": "white",
            "label": "extreme-RIGHT tail share (sem axis, >p90)",
        },
    )
    left_plot = _prepare_event_study_plot_df(left.es_df, left.rel_col)
    right_plot = _prepare_event_study_plot_df(right.es_df, right.rel_col)
    for plot_df, style in ((left_plot, styles[0]), (right_plot, styles[1])):
        if plot_df.empty:
            continue
        mask = plot_df["se"].fillna(0) > 0
        ax.errorbar(
            plot_df.loc[mask, "event_time"],
            plot_df.loc[mask, "gamma"],
            yerr=1.96 * plot_df.loc[mask, "se"],
            fmt="none",
            ecolor=style["color"],
            capsize=3,
            zorder=5,
        )
        ax.plot(
            plot_df["event_time"],
            plot_df["gamma"],
            linestyle="none",
            marker=style["marker"],
            markerfacecolor=style["mfc"],
            markeredgecolor=style["color"],
            markeredgewidth=1.0,
            label=style["label"],
            zorder=6,
        )
    x_scale = "period" if left.rel_col == "rel_period" or right.rel_col == "rel_period" else "days"
    apply_event_study_axes_style(ax, bin_days=int(bin_days), x_scale=x_scale)
    all_times = sorted(
        set(left_plot["event_time"].astype(int).tolist())
        | set(right_plot["event_time"].astype(int).tolist())
    )
    if all_times:
        ax.set_xticks(all_times)
    ax.set_title("Extreme-left and extreme-right tail shares", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_coef_post_phases(
    summary: pd.DataFrame,
    outcome_id: str,
    out_path: Path,
    strategies: Optional[Sequence[str]] = None,
    phase_specs: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
) -> None:
    """Function summary: horizontal coef plot with short/medium/long post-phase β per headline strategy.

    Parameters:
    - summary: did_summary-like frame with strategy_id, spec, beta, se.
    - outcome_id: outcome slug.
    - out_path: PNG path.
    - strategies: headline strategy_ids; default PLOT_STRATEGY_GROUPS['headline'].
    - phase_specs: phase_joint_* spec ids; default PHASE_JOINT_SPECS.
    - title: optional plot title.
    """
    del title  # use outcome_label below
    phase_specs = tuple(phase_specs or PHASE_JOINT_SPECS)
    strategies = list(strategies or PLOT_STRATEGY_GROUPS["headline"])
    sub = summary[(summary["outcome_id"] == outcome_id) & (summary["strategy_id"].isin(strategies))].copy()
    sub = sub[sub["spec"].astype(str).isin(phase_specs)]
    sub = _drop_degenerate_lift_rows(_ensure_spec_column(sub))
    sub = sub.dropna(subset=["beta"], how="all")
    if sub.empty or sub["beta"].notna().sum() == 0:
        return
    sub = sub[sub["beta"].notna()]
    if sub.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(strategies)
    n_phase = len(phase_specs)
    offsets = np.linspace(-(n_phase - 1) * 0.11 / 2, (n_phase - 1) * 0.11 / 2, n_phase) if n_phase > 1 else [0.0]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * n)))
    handles = []
    labels = []
    for j, spec in enumerate(phase_specs):
        color = PHASE_JOINT_PLOT_COLORS.get(str(spec), "#333333")
        for i, sid in enumerate(strategies):
            rows = sub[(sub["strategy_id"] == sid) & (sub["spec"].astype(str) == spec)]
            if rows.empty:
                continue
            row = rows.iloc[0]
            beta = float(row["beta"])
            se = float(row.get("se", float("nan")) or float("nan"))
            y = i + offsets[j]
            if np.isfinite(se) and se > 0:
                ax.errorbar(
                    beta,
                    y,
                    xerr=1.96 * se,
                    fmt="none",
                    ecolor=color,
                    capsize=2,
                    elinewidth=1.0,
                    zorder=2,
                )
            ax.scatter(
                [beta],
                [y],
                s=42,
                color="white",
                edgecolors=color,
                linewidths=1.2,
                zorder=3,
            )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="white",
                markeredgecolor=color,
                markersize=8,
            )
        )
        labels.append(spec_label_short(str(spec)))
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([strategy_label(s, short=True) for s in strategies], fontsize=8)
    ax.set_xlabel("β")
    ax.set_title(f"Post phases: {outcome_label(outcome_id, short=True)}")
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_post_phase_comparison(summary: pd.DataFrame, outcome_id: str, out_path: Path) -> None:
    """Function summary: vertical bar chart of short/medium/long post-phase β (cross_country_all)."""
    sub = summary[
        (summary["outcome_id"] == outcome_id)
        & (summary["strategy_id"] == "cross_country_all")
        & (summary["spec"].astype(str).isin(PHASE_JOINT_SPECS))
    ].copy()
    sub = _drop_degenerate_lift_rows(_ensure_spec_column(sub))
    if sub.empty:
        return
    records: List[Dict[str, Any]] = []
    for spec in PHASE_JOINT_SPECS:
        r = sub[sub["spec"].astype(str) == spec]
        if r.empty or pd.isna(r["beta"].iloc[0]):
            continue
        records.append(
            {
                "phase": spec_label_short(str(spec)),
                "beta": float(r["beta"].iloc[0]),
                "se": float(r["se"].iloc[0]) if pd.notna(r["se"].iloc[0]) else float("nan"),
            }
        )
    if not records:
        return
    plot_df = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(plot_df))
    yerr = plot_df["se"].fillna(0).astype(float) * 1.96
    yerr = yerr.where(yerr > 0, 0)
    ax.bar(x, plot_df["beta"], yerr=yerr, color="#457b9d", capsize=4, error_kw={"linewidth": 1.0})
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["phase"].astype(str), rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("β")
    ax.set_title(f"Post phases — {strategy_label('cross_country_all', short=True)}: {outcome_label(outcome_id, short=True)}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_coef_comparison(
    summary: pd.DataFrame,
    outcome_id: str,
    out_path: Path,
    strategies: Optional[Sequence[str]] = None,
    label_col: str = "strategy_label",
    title: Optional[str] = None,
    subtitle: str = "",
    outcome_family: Optional[str] = None,
) -> None:
    """Function summary: coefficient plot across strategies for one outcome."""
    sub = summary[summary["outcome_id"] == outcome_id].copy()
    if strategies:
        sub = sub[sub["strategy_id"].isin(strategies)]
    sub = _filter_headline_spec(sub)
    absorbed = sub[sub["estimation_note"].astype(str) == "no_treat_variation"]
    sub = sub[sub["estimation_note"].astype(str) != "no_treat_variation"]
    sub["plot_label"] = _strategy_plot_labels(sub)
    sub = sub.dropna(subset=["beta"], how="all")
    if sub.empty or sub["beta"].notna().sum() == 0:
        return
    sub = sub[sub["beta"].notna()]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(sub))))
    y_pos = range(len(sub))
    ax.errorbar(
        sub["beta"],
        list(y_pos),
        xerr=1.96 * sub["se"],
        fmt="o",
        color="#2d6a4f",
        capsize=3,
    )
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sub["plot_label"].astype(str), fontsize=8)
    ax.set_xlabel("β")
    ax.set_title(title or f"DiD: {outcome_label(outcome_id, short=True)}")
    if outcome_family in ("wordfish_author_v2",) and not absorbed.empty:
        note_txt = (
            f"n strategies plotted: {len(sub)} "
            "(author_it_vs_de absorbed by author FE)"
        )
        ax.text(0.02, 0.98, note_txt, transform=ax.transAxes, fontsize=8, va="top", color="#555")
    elif subtitle:
        ax.text(0.02, 0.98, subtitle, transform=ax.transAxes, fontsize=8, va="top", color="#555")
    if "estimation_note" in sub.columns:
        for i, (_, row) in enumerate(sub.iterrows()):
            if str(row.get("estimation_note", "ok")) != "ok":
                ax.annotate(
                    str(row["estimation_note"]),
                    (row["beta"], i),
                    fontsize=6,
                    color="#888",
                )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_placebo_robustness(rob_df: pd.DataFrame, outcome_id: str, out_path: Path) -> None:
    """Function summary: placebo vs baseline bar comparison."""
    if rob_df.empty:
        return
    plot_df = rob_df.copy()
    if "estimation_note" in plot_df.columns:
        plot_df = plot_df[plot_df["estimation_note"].astype(str) != "skipped_insufficient_pre"]
    plot_df = plot_df[plot_df["beta"].map(lambda v: pd.notna(v) and abs(float(v)) < 1e6)]
    if plot_df.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=plot_df, x="check", y="beta", ax=ax, color="#457b9d")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(
        f"Robustness: {outcome_label(outcome_id, short=True)}\n"
        "(descriptive placebo dates; unequal windows — not a permutation test)"
    )
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_first_stage(summary: pd.DataFrame, out_path: Path) -> None:
    """Function summary: AI-writing first-stage coefs across strategies."""
    fs = _ensure_spec_column(summary[summary["outcome_id"].isin(FIRST_STAGE_OUTCOMES)])
    fs = fs[fs["spec"].astype(str) == "full_ban"]
    if fs.empty:
        return
    agg = fs.groupby("outcome_id", as_index=False)["beta"].mean()
    agg["plot_label"] = agg["outcome_id"].map(lambda o: outcome_label(str(o), short=True))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=agg, x="plot_label", y="beta", ax=ax, color="#e76f51")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("First stage (AI stylometrics)")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_significance_heatmap(summary: pd.DataFrame, out_path: Path) -> None:
    """Function summary: outcome × strategy heatmap of sign and significance."""
    headline = list(PLOT_STRATEGY_GROUPS["headline"]) + list(PLOT_STRATEGY_GROUPS["by_topic"])
    raw = _filter_headline_spec(summary[summary["strategy_id"].isin(headline)].copy())
    if raw.empty:
        return
    sub = raw[
        (raw["estimation_note"].astype(str) == "ok")
        & (raw["beta"].abs() <= 1e3)
        & raw["pvalue"].notna()
    ].copy()
    if sub.empty:
        return
    sub["plot_label"] = _strategy_plot_labels(sub)
    sub["outcome_plot"] = _outcome_plot_labels(sub)
    pivot_beta = sub.pivot_table(index="outcome_plot", columns="plot_label", values="beta", aggfunc="first")
    sub["p_headline"] = sub.apply(headline_pvalue, axis=1)
    pivot_p = sub.pivot_table(index="outcome_plot", columns="plot_label", values="p_headline", aggfunc="first")
    pivot_n = sub.pivot_table(index="outcome_plot", columns="plot_label", values="n_clusters", aggfunc="first")
    if pivot_beta.empty:
        return
    plot_beta = pivot_beta.copy()
    oid_by_plot = sub.drop_duplicates("outcome_plot").set_index("outcome_plot")["outcome_id"].astype(str)
    for idx in plot_beta.index:
        oid = oid_by_plot.get(idx, "")
        if str(oid).startswith(("wf_", "wf2_", "wfa_", "wfa2_")):
            row = plot_beta.loc[idx]
            mx = row.abs().max(skipna=True)
            if pd.notna(mx) and mx > 0:
                plot_beta.loc[idx] = row / mx * 0.2
    annot = pivot_p.map(lambda p: "*" if pd.notna(p) and p < 0.05 else "").astype(object)
    for _, row in sub.iterrows():
        ridx = row["outcome_plot"]
        cidx = row["plot_label"]
        p_star = headline_pvalue(row)
        star = "*" if pd.notna(p_star) and p_star < 0.05 else ""
        ncl = row.get("n_clusters", float("nan"))
        txt = f"{star}\nN={int(ncl)}".strip() if pd.notna(ncl) else star
        if ridx in annot.index and cidx in annot.columns:
            annot.loc[ridx, cidx] = txt
    for _, row in raw.iterrows():
        if str(row.get("estimation_note")) != "degenerate_collinear":
            continue
        rlab = outcome_label(str(row["outcome_id"]), short=True)
        slab = strategy_label(str(row["strategy_id"]), short=True)
        matches = sub[(sub["outcome_plot"] == rlab) & (sub["plot_label"] == slab)]
        if matches.empty and rlab in annot.index:
            for cidx in annot.columns:
                if slab in str(cidx) or strategy_label(str(row["strategy_id"])) in str(cidx):
                    annot.loc[rlab, cidx] = "deg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, pivot_beta.shape[1] * 0.9), max(5, pivot_beta.shape[0] * 0.35)))
    sns.heatmap(
        plot_beta,
        cmap="RdBu_r",
        center=0,
        vmin=-0.2,
        vmax=0.2,
        annot=annot,
        fmt="",
        ax=ax,
        cbar_kws={"label": "β (row-norm WF)"},
    )
    ax.set_title("DiD β (* p<0.05; N=clusters)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _strategy_suffix(strategy_id: str) -> str:
    """Function summary: short strategy label suffix for forest annotations."""
    if strategy_id == "cross_country_all":
        return "pooled"
    if strategy_id == "cross_country_it_political":
        return "it_pol"
    if strategy_id == "cross_country_it_others":
        return "it_oth"
    return strategy_id.replace("cross_country_", "")


def plot_headline_forest(summary: pd.DataFrame, out_path: Path) -> None:
    """Function summary: faceted forest for headline outcomes × three cross-country strategies."""
    sub = summary[
        (summary["outcome_id"].isin(HEADLINE_OUTCOMES))
        & (summary["strategy_id"].isin(HEADLINE_FOREST_STRATEGIES))
    ].copy()
    sub = _filter_headline_spec(sub)
    sub = sub[sub["estimation_note"].astype(str) == "ok"]
    if sub.empty:
        return
    outcomes = [o for o in HEADLINE_OUTCOMES if o in set(sub["outcome_id"])]
    if not outcomes:
        return
    ncols = len(HEADLINE_FOREST_STRATEGIES)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        nrows=len(outcomes),
        ncols=ncols,
        figsize=(3.2 * ncols, max(2.5, 1.1 * len(outcomes))),
        squeeze=False,
    )
    off_scale_notes: List[str] = []
    for i, oid in enumerate(outcomes):
        xlim = FOREST_XLIM_BY_OUTCOME.get(oid, (-0.05, 0.05))
        for j, sid in enumerate(HEADLINE_FOREST_STRATEGIES):
            ax = axes[i, j]
            row = sub[(sub["outcome_id"] == oid) & (sub["strategy_id"] == sid)]
            if row.empty or pd.isna(row["beta"].iloc[0]):
                ax.set_visible(False)
                continue
            beta = float(row["beta"].iloc[0])
            se = float(row["se"].iloc[0]) if pd.notna(row["se"].iloc[0]) else float("nan")
            if np.isfinite(beta) and (beta < xlim[0] or beta > xlim[1]):
                off_scale_notes.append(f"{oid}/{_strategy_suffix(sid)}")
                ax.axvline(0, color="gray", linewidth=0.6)
                ax.set_xlim(xlim)
                ax.set_yticks([0])
                ax.set_yticklabels([outcome_label(oid, short=True) if j == 0 else ""])
                ax.text(0.5, 0.5, "(off-scale)", transform=ax.transAxes, ha="center", color="red", fontsize=7)
            else:
                if np.isfinite(se) and se > 0:
                    ax.errorbar([beta], [0], xerr=1.96 * se, fmt="o", color="#1d3557", capsize=3)
                else:
                    ax.scatter([beta], [0], color="#1d3557")
                ax.axvline(0, color="gray", linewidth=0.6)
                ax.set_xlim(xlim)
                ax.set_yticks([0])
                ax.set_yticklabels([outcome_label(oid, short=True) if j == 0 else ""])
                ax.text(
                    beta,
                    0.35,
                    _strategy_suffix(sid),
                    fontsize=6,
                    ha="center",
                    va="bottom",
                )
            if i == 0:
                ax.set_title(strategy_label(sid, short=True), fontsize=8)
    if off_scale_notes:
        fig.text(0.5, 0.01, "Off-scale: " + ", ".join(off_scale_notes), ha="center", color="red", fontsize=7)
    fig.suptitle("Headline outcomes (cross-country strategies)", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_early_ban_comparison(summary: pd.DataFrame, outcome_id: str, out_path: Path) -> None:
    """Function summary: compare ban-in-effect vs 7d vs 14d early-ban estimates."""
    pairs = [
        ("cross_country_all", HEADLINE_TWFE_SPEC),
        ("cross_country_all", "early_ban_7d"),
        ("cross_country_it_political", "early_ban_7d"),
        ("cross_country_it_others", "early_ban_7d"),
        ("cross_country_all_14d", "early_ban_14d"),
    ]
    sub = summary[summary["outcome_id"] == outcome_id].copy()
    if sub.empty:
        return
    sub = _ensure_spec_column(sub)
    picked = []
    for sid, spec in pairs:
        rows = sub[(sub["strategy_id"] == sid) & (sub["spec"].astype(str) == spec)]
        if not rows.empty:
            picked.append(rows.iloc[-1])
    if not picked:
        return
    sub = pd.DataFrame(picked)
    sub = sub.drop_duplicates(subset=["strategy_id", "spec"], keep="last")
    sub["plot_label"] = sub.apply(
        lambda r: (
            f"{strategy_label(str(r['strategy_id']), short=True)} "
            f"{spec_label_parenthetical(str(r['spec']))}"
        ),
        axis=1,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=sub, x="plot_label", y="beta", ax=ax, color="#457b9d")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(f"Early-ban: {outcome_label(outcome_id, short=True)}")
    plt.xticks(rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_ddd_subpanel(ax: plt.Axes, sub: pd.DataFrame, outcome_ids: Sequence[str], xlim: Optional[tuple] = None) -> None:
    """Function summary: draw one DDD forest subpanel for selected outcomes."""
    rows = sub[sub["outcome_id"].isin(outcome_ids)].dropna(subset=["beta"])
    rows = rows.drop_duplicates(subset=["outcome_id"], keep="last")
    order = [o for o in outcome_ids if o in set(rows["outcome_id"])]
    if not order:
        ax.set_visible(False)
        return
    rows = rows.set_index("outcome_id").loc[order].reset_index()
    y_pos = range(len(rows))
    ax.errorbar(
        rows["beta"],
        list(y_pos),
        xerr=1.96 * rows["se"],
        fmt="o",
        color="#6a4c93",
        capsize=3,
    )
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(rows["outcome_id"].map(lambda o: outcome_label(o, short=True)), fontsize=8)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_xlabel("β")


def plot_ddd_panel(summary: pd.DataFrame, out_path: Path) -> None:
    """Function summary: within-Italy triple-diff — rates vs variance outcomes side by side."""
    sub = _ensure_spec_column(summary[summary["strategy_id"] == "within_italy_ddd"].copy())
    sub = sub[sub["spec"].astype(str) == "full_ban"]
    sub = sub[sub["beta"].notna()]
    if sub.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, max(4, 0.35 * len(DDD_LEFT_OUTCOMES))))
    _plot_ddd_subpanel(ax_left, sub, DDD_LEFT_OUTCOMES, xlim=(-1.0, 1.0))
    ax_left.set_title("Rates / levels (0–1 scale)")
    _plot_ddd_subpanel(ax_right, sub, DDD_RIGHT_OUTCOMES, xlim=None)
    ax_right.set_title("Variance-style outcomes")
    fig.suptitle("Within-IT triple-diff", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pretrend_summary(summary: pd.DataFrame, out_path: Path) -> None:
    """Function summary: bar chart of event-study pretrend F p-values for headline outcomes."""
    sub = summary[
        (summary["outcome_id"].isin(HEADLINE_OUTCOMES))
        & (summary["strategy_id"] == "cross_country_all")
    ].copy()
    sub = _filter_headline_spec(sub)
    if "pretrend_quality" in sub.columns:
        sub = sub[sub["pretrend_quality"].astype(str) == "ok"]
    else:
        sub = sub[sub["estimation_note"].astype(str) == "ok"]
        sub = sub[sub["beta"].notna()]
    if "pretrend_F_p" not in sub.columns or sub["pretrend_F_p"].notna().sum() == 0:
        return
    sub = sub.dropna(subset=["pretrend_F_p"])
    if sub.empty:
        return
    sub = _ensure_spec_column(sub.copy())
    sub["plot_label"] = _outcome_plot_labels(sub)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=sub, x="plot_label", y="pretrend_F_p", ax=ax, color="#a8dadc")
    ax.axhline(0.05, color="red", linestyle="--", linewidth=0.8, label="α=0.05")
    ax.set_title("Pre-trend F p-values")
    plt.xticks(rotation=25, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_overview_figures(summary: pd.DataFrame, fig_dir: Path) -> None:
    """Function summary: write all overview/ diagnostic figures and overview/README.md."""
    overview = fig_dir / "overview"
    plot_significance_heatmap(summary, overview / "significance_heatmap.png")
    plot_headline_forest(summary, overview / "headline_forest_lexical_semantic.png")
    plot_ddd_panel(summary, overview / "ddd_political_specificity.png")
    plot_first_stage(summary, overview / "first_stage_aiwriting.png")
    plot_pretrend_summary(summary, overview / "pretrend_summary.png")
    overview_outcomes = (
        "net_ideology",
        "pole_share",
        "esteban_ray",
        "sem_axis_ideology",
        "sem_axis_ideology_pole_share",
        "sem_axis_ideology_esteban_ray",
        "ai_style_rate",
    )
    for oid in overview_outcomes:
        if (summary["outcome_id"] == oid).any():
            plot_early_ban_comparison(summary, oid, overview / f"early_ban_{oid}.png")
            plot_post_phase_comparison(summary, oid, overview / f"post_phase_{oid}.png")
    write_overview_readme(overview)


def regenerate_did_figures(
    summary_df: pd.DataFrame,
    fig_dir: Path,
    *,
    families: Optional[Sequence[str]] = None,
    full_coefplots: bool = False,
) -> Dict[str, List[str]]:
    """Function summary: rebuild DiD coefplots and overview figures from did_summary.csv.

    Parameters:
    - summary_df: master summary table.
    - fig_dir: did figures root.
    - families: optional subset of outcome_family ids; default all in summary.
    - full_coefplots: also write coefplots_full/ per outcome.

    Returns:
    - outcome_ids_by_family map used for README generation.
    """
    labeled = add_strategy_labels(summary_df)
    fam_filter = set(families) if families else None
    outcome_ids_by_family: Dict[str, List[str]] = {}

    for oid in summary_df["outcome_id"].unique():
        rows = summary_df[summary_df["outcome_id"] == oid]
        fam = str(rows["outcome_family"].iloc[0])
        if fam_filter is not None and fam not in fam_filter:
            continue
        outcome_ids_by_family.setdefault(fam, []).append(str(oid))
        fam_strats = coefplot_strategies_for_family(fam)
        if fam in ("wordfish_author", "wordfish_author_v2"):
            plot_coef_comparison(
                labeled,
                str(oid),
                figure_path(fig_dir, fam, "coefplots_headline", str(oid)),
                strategies=fam_strats,
                outcome_family=fam,
            )
        else:
            plot_coef_post_phases(
                labeled,
                str(oid),
                figure_path(fig_dir, fam, "coefplots_headline", str(oid)),
                strategies=list(PLOT_STRATEGY_GROUPS["headline"]),
            )
        if full_coefplots:
            plot_coef_comparison(
                labeled,
                str(oid),
                figure_path(fig_dir, fam, "coefplots_full", str(oid)),
                strategies=None,
            )

    generate_overview_figures(labeled, fig_dir)
    return outcome_ids_by_family
