"""
Script summary:
Estimate within-author English-quality DiD and event studies (Italy ChatGPT ban).

Functionality:
- Loads English-quality comment panel parquet.
- Design 1 (native_control): English comments on English forums; treat = italian_author.
- Design 2 (cross_language family): bilingual authors; treat = is_english vs Italian.
- Design 2 headline: within-author diff d_ab = mean(y|EN) - mean(y|IT) per author x bin.
- Design 2 robustness: clean_TWFE comment-level i(rel,is_english) | author + time_id.
- Static cross_language: y ~ post + is_english + post:is_english | author + time_id.
- Subreddit wild cluster bootstrap (adaptive enumeration when G<=12) for static post:EN.
- Writes coefficient tables and event-study figures with ban and lift markers.

How to apply/run:
  .venv/bin/python scripts/analysis/estimate_english_quality_did.py \\
    --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/analysis/estimate_english_quality_did.py --bin-days 3 --cohort strict
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _setup_project_root() -> Path:
    """Function summary: resolve repo root via scripts/_bootstrap.py.

    Returns:
    - Absolute repo root Path.
    """
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

from src.config_utils import load_config, plot_reference_dates_calendar_utc  # noqa: E402
from src.did.english_quality import (  # noqa: E402
    CROSS_LANGUAGE_DESIGNS,
    HEADLINE_OUTCOMES,
    POLARIZATION_OUTCOMES,
    ROSTER_WINDOW_CHOICES,
    apply_standardized_outcome,
    cohort_authors_for_design,
    cohort_thresholds_by_label,
    english_quality_run_figures_dir,
    english_quality_run_tables_dir,
    estimate_fd_event_study,
    estimate_static_post_treat,
    estimate_treat_event_study,
    estimate_within_author_diff_event_study,
    estimate_within_author_diff_static,
    estimate_within_language_post,
    filter_language_pair_sample,
    filter_native_control_sample,
    headline_outcomes_for_design,
    march_standardization_moments_by_lang,
    march_standardization_moments_pooled,
    outcome_caveat,
    outcome_label,
    static_es_post_avg,
)
from src.did.inference import wild_cluster_bootstrap_cross_language_static  # noqa: E402
from src.did.outputs import apply_event_study_axes_style  # noqa: E402

DesignId = Literal[
    "native_control",
    "cross_language",
    "cross_language_native_it",
    "cross_language_langmix",
]

ALL_DESIGNS: Tuple[DesignId, ...] = (
    "native_control",
    "cross_language",
    "cross_language_native_it",
    "cross_language_langmix",
)
DOMINANT_LANG_GROUPS: Tuple[str, ...] = ("it", "en", "other")


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI for English-quality DiD estimation.

    Returns:
    - Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description="Estimate English-quality within-author DiD.")
    parser.add_argument("--config", default="config/italy_polarization_setup.yaml")
    parser.add_argument("--bin-days", type=int, default=3, choices=(1, 3))
    parser.add_argument("--cohort", default="strict", choices=("strict", "loose"))
    parser.add_argument("--outcomes", default=",".join(HEADLINE_OUTCOMES))
    parser.add_argument(
        "--polarization-outcomes",
        default=",".join(POLARIZATION_OUTCOMES),
        help="Comma-separated polarization/semantic outcomes (always z-scored within language).",
    )
    parser.add_argument(
        "--no-polarization",
        action="store_true",
        help="Skip the polarization/semantic outcome set.",
    )
    parser.add_argument("--cluster", default="author", choices=("author", "subreddit"))
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--political-only",
        action="store_true",
        help="Robustness: restrict to comment_in_political_universe rows (output tag _politicalonly).",
    )
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument(
        "--wcb-draws",
        type=int,
        default=999,
        help="Subreddit wild-cluster bootstrap draws when G>12 (enumerated exactly when G<=12).",
    )
    parser.add_argument(
        "--roster-window",
        default="pre_ban",
        choices=ROSTER_WINDOW_CHOICES,
        help="Roster classification window used to build the panel (pre_ban default).",
    )
    return parser.parse_args()


def load_panel(
    config: Dict[str, Any],
    bin_days: int,
    *,
    roster_window: str = "pre_ban",
) -> pd.DataFrame:
    """Function summary: load partitioned English-quality panel parquet.

    Parameters:
    - config: study YAML.
    - bin_days: panel bin tag (1d or 3d).
    - roster_window: pre_ban or full run subdir.

    Returns:
    - Concatenated comment panel DataFrame.
    """
    panel_dir = english_quality_run_tables_dir(config, roster_window) / "panel" / f"{int(bin_days)}d"
    parts = sorted(panel_dir.glob("month=*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No panel parquet under {panel_dir}")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def apply_cohort_filter(panel: pd.DataFrame, design: DesignId, cohort_label: str) -> pd.DataFrame:
    """Function summary: keep rows for authors passing design cohort gates.

    Parameters:
    - panel: full comment panel.
    - design: native_control or cross_language.
    - cohort_label: strict or loose threshold bundle.

    Returns:
    - Filtered panel.
    """
    authors = cohort_authors_for_design(
        panel, design, cohort_thresholds_by_label(cohort_label)
    )
    return panel.loc[panel["author"].astype(str).isin(authors)].copy()


def prepare_design_sample(
    panel: pd.DataFrame,
    design: DesignId,
    outcome: str,
    cohort_label: str,
) -> pd.DataFrame:
    """Function summary: filter design sample and attach standardized outcome y.

    Parameters:
    - panel: full panel.
    - design: native_control or a cross_language-family design.
    - outcome: raw outcome column.
    - cohort_label: cohort gate label.

    Returns:
    - Estimation-ready DataFrame with y column. Native-control uses pooled
      (English-only) moments; cross-language designs always z-score within
      lang_comment so EN vs IT levels are comparable (required for the
      polarization/semantic outcomes).
    """
    work = apply_cohort_filter(panel, design, cohort_label)
    if design == "native_control":
        work = filter_native_control_sample(work)
        moments = march_standardization_moments_pooled(work, outcome)
        return apply_standardized_outcome(work, outcome, moments, out_col="y")
    work = filter_language_pair_sample(work)
    if outcome not in work.columns:
        return work.iloc[0:0].copy()
    moments = march_standardization_moments_by_lang(work, outcome)
    if moments.empty:
        out = work.copy()
        out["y"] = float("nan")
        return out
    return apply_standardized_outcome(work, outcome, moments, group_col="lang_comment", out_col="y")


def plot_english_quality_event_study(
    es_df: pd.DataFrame,
    outcome_id: str,
    design: DesignId,
    out_path: Path,
    *,
    bin_days: int = 3,
    lift_date: str = "2023-04-28",
    launch_date: str = "2023-03-31",
    spec_label: str = "",
) -> None:
    """Function summary: event-study plot with ban onset and lift markers.

    Parameters:
    - es_df: coefficients with rel_period/rel_day, gamma, se.
    - outcome_id: outcome column name.
    - design: design id for title.
    - out_path: PNG output path.
    - bin_days: rel_day multiplier when rel_col is rel_period.
    - lift_date: ban lift UTC date for vertical marker.
    - launch_date: ban onset date (for rel_day conversion).
    - spec_label: optional spec suffix in title.
    """
    if es_df.empty:
        return
    plot = es_df.copy()
    if "rel_period" in plot.columns and "rel_day" not in plot.columns:
        plot["rel_day"] = plot["rel_period"].astype(int) * bin_days
    plot["event_time"] = plot["rel_day"].astype(int)
    if -1 not in plot["event_time"].values:
        plot = pd.concat(
            [
                plot,
                pd.DataFrame([{"event_time": -1, "gamma": 0.0, "se": 0.0}]),
            ],
            ignore_index=True,
        )
    plot = plot.sort_values("event_time")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    mask = plot["se"].fillna(0) > 0
    ax.errorbar(
        plot.loc[mask, "event_time"],
        plot.loc[mask, "gamma"],
        yerr=1.96 * plot.loc[mask, "se"],
        fmt="none",
        ecolor="black",
        capsize=3,
        zorder=2,
    )
    ax.plot(
        plot["event_time"],
        plot["gamma"],
        linestyle="none",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.0,
        zorder=3,
    )
    launch_dt = pd.Timestamp(launch_date)
    lift_rel = int((pd.Timestamp(lift_date) - launch_dt).days)
    ax.axvline(-0.5, color="0.35", linestyle="--", linewidth=1.0, zorder=1)
    ax.axvline(lift_rel + 0.5, color="0.55", linestyle=":", linewidth=1.0, zorder=1)
    apply_event_study_axes_style(ax)
    title = f"{outcome_label(outcome_id)} — {design.replace('_', ' ')}"
    if spec_label:
        title = f"{title} ({spec_label})"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _static_row(
    *,
    design: DesignId,
    outcome: str,
    spec: str,
    res: Dict[str, Any],
    cluster_col: str,
    cohort_label: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Function summary: build one static results row with optional caveat metadata."""
    row: Dict[str, Any] = {
        "design": design,
        "outcome": outcome,
        "spec": spec,
        "beta": res.get("beta"),
        "se": res.get("se"),
        "pvalue": res.get("pvalue"),
        "n_obs": res.get("n_obs"),
        "n_clusters": res.get("n_clusters"),
        "estimation_note": res.get("estimation_note"),
        "cluster_col": cluster_col,
        "cohort": cohort_label,
        "outcome_caveat": outcome_caveat(outcome),
    }
    if extra:
        row.update(extra)
    return row


def _append_es_rows(
    es_rows: List[Dict[str, Any]],
    *,
    es_df: pd.DataFrame,
    design: DesignId,
    outcome: str,
    spec: str,
    cohort_label: str,
    cluster_col: str,
) -> None:
    """Function summary: append event-study coefficient rows from an es_df."""
    if es_df.empty:
        return
    for _, row in es_df.iterrows():
        es_rows.append(
            {
                "design": design,
                "outcome": outcome,
                "spec": spec,
                "rel_period": row.get("rel_period"),
                "rel_day": row.get("rel_day"),
                "gamma": row.get("gamma"),
                "se": row.get("se"),
                "pvalue": row.get("pvalue"),
                "n_authors": row.get("n_authors"),
                "n_pairs": row.get("n_pairs"),
                "cohort": cohort_label,
                "cluster_col": cluster_col,
            }
        )


def run_design_estimates(
    panel: pd.DataFrame,
    design: DesignId,
    outcomes: Sequence[str],
    *,
    cohort_label: str,
    bin_days: int,
    cluster_col: str,
    window: int,
    config: Dict[str, Any],
    write_figures: bool = True,
    file_tag: Optional[str] = None,
    wcb_draws: int = 999,
    roster_window: str = "pre_ban",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Function summary: run static and event-study estimates for one design.

    Parameters:
    - panel: full comment panel.
    - design: native_control or a cross_language-family design.
    - outcomes: outcome column names.
    - cohort_label: strict or loose.
    - bin_days: event bin width.
    - cluster_col: cluster for SEs.
    - window: event-study trim window.
    - config: study YAML.
    - write_figures: when True, save PNG event studies.
    - file_tag: output filename tag (defaults to cohort_label; carries the
      _politicalonly suffix in the political-only robustness run).
    - wcb_draws: bootstrap replications for subreddit WCB when G>12.
    - roster_window: pre_ban or full run subdir for outputs.

    Returns:
    - Tuple (static_df, es_df_all, within_author_static_df, within_author_es_df).
    """
    file_tag = file_tag or cohort_label
    treat_col = "italian_author" if design == "native_control" else "is_english"
    is_cross = design in CROSS_LANGUAGE_DESIGNS
    static_rows: List[Dict[str, Any]] = []
    es_rows: List[Dict[str, Any]] = []
    wa_static_rows: List[Dict[str, Any]] = []
    wa_es_rows: List[Dict[str, Any]] = []
    tables_dir = english_quality_run_tables_dir(config, roster_window)
    figures_dir = english_quality_run_figures_dir(config, roster_window)

    for outcome in outcomes:
        sample = prepare_design_sample(panel, design, outcome, cohort_label)
        if sample.empty:
            continue
        static = estimate_static_post_treat(
            sample,
            treat_col=treat_col,
            y_col="y",
            cluster_col=cluster_col,
            include_treat_main=is_cross,
        )
        static_rows.append(
            _static_row(
                design=design,
                outcome=outcome,
                spec="static_post_treat",
                res=static,
                cluster_col=cluster_col,
                cohort_label=cohort_label,
            )
        )

        if design == "cross_language":
            wcb = wild_cluster_bootstrap_cross_language_static(
                sample,
                treat_col=treat_col,
                y_col="y",
                cluster_col="subreddit",
                n_draws=wcb_draws,
            )
            static_rows.append(
                _static_row(
                    design=design,
                    outcome=outcome,
                    spec="static_post_treat_wcb_subreddit",
                    res={
                        "beta": wcb.get("beta"),
                        "se": float("nan"),
                        "pvalue": wcb.get("pvalue"),
                        "n_obs": static.get("n_obs"),
                        "n_clusters": wcb.get("n_clusters"),
                        "estimation_note": wcb.get("estimation_note"),
                    },
                    cluster_col="subreddit",
                    cohort_label=cohort_label,
                    extra={
                        "wcb_enumerated": wcb.get("enumerated"),
                        "wcb_p_floor": wcb.get("p_floor"),
                        "wcb_n_draws": wcb.get("n_draws_used"),
                    },
                )
            )

        if design == "native_control":
            for spec, baseline in (("FD_ref", "ref"), ("FD_mean", "preban_mean"), ("baseline_B", "ref")):
                if spec == "baseline_B":
                    _, es_df = estimate_treat_event_study(
                        sample,
                        treat_col=treat_col,
                        y_col="y",
                        cluster_col=cluster_col,
                        bin_days=bin_days,
                        window=window,
                    )
                    spec_name = "baseline_TWFE"
                else:
                    _, es_df = estimate_fd_event_study(
                        sample,
                        treat_col=treat_col,
                        y_col="y",
                        bin_days=bin_days,
                        window=window,
                        baseline=baseline,
                        cluster_col=cluster_col,
                    )
                    spec_name = spec
                _append_es_rows(
                    es_rows,
                    es_df=es_df,
                    design=design,
                    outcome=outcome,
                    spec=spec_name,
                    cohort_label=cohort_label,
                    cluster_col=cluster_col,
                )
                if write_figures and spec_name in ("FD_ref", "baseline_TWFE"):
                    plot_english_quality_event_study(
                        es_df,
                        outcome,
                        design,
                        figures_dir / design / outcome / f"es_{spec_name}.png",
                        bin_days=bin_days,
                        spec_label=spec_name,
                    )
        else:
            _, es_wa = estimate_within_author_diff_event_study(
                sample,
                y_col="y",
                cluster_col=cluster_col,
                bin_days=bin_days,
                window=window,
                weighted=True,
            )
            _append_es_rows(
                wa_es_rows,
                es_df=es_wa,
                design=design,
                outcome=outcome,
                spec="within_author_diff",
                cohort_label=cohort_label,
                cluster_col=cluster_col,
            )
            _append_es_rows(
                es_rows,
                es_df=es_wa,
                design=design,
                outcome=outcome,
                spec="within_author_diff",
                cohort_label=cohort_label,
                cluster_col=cluster_col,
            )
            if write_figures and design == "cross_language":
                plot_english_quality_event_study(
                    es_wa,
                    outcome,
                    design,
                    figures_dir / design / outcome / "es_within_author_diff.png",
                    bin_days=bin_days,
                    spec_label="within_author_diff",
                )

            wa_static = estimate_within_author_diff_static(
                sample,
                y_col="y",
                cluster_col=cluster_col,
                bin_days=bin_days,
                window=window,
                weighted=True,
            )
            wa_static_rows.append(
                _static_row(
                    design=design,
                    outcome=outcome,
                    spec="within_author_diff",
                    res=wa_static,
                    cluster_col=cluster_col,
                    cohort_label=cohort_label,
                    extra={"n_cells": wa_static.get("n_cells"), "n_authors": wa_static.get("n_authors")},
                )
            )
            wa_static_uw = estimate_within_author_diff_static(
                sample,
                y_col="y",
                cluster_col=cluster_col,
                bin_days=bin_days,
                window=window,
                weighted=False,
            )
            wa_static_rows.append(
                _static_row(
                    design=design,
                    outcome=outcome,
                    spec="within_author_diff_unweighted",
                    res=wa_static_uw,
                    cluster_col=cluster_col,
                    cohort_label=cohort_label,
                    extra={
                        "n_cells": wa_static_uw.get("n_cells"),
                        "n_authors": wa_static_uw.get("n_authors"),
                    },
                )
            )

            _, es_df = estimate_treat_event_study(
                sample,
                treat_col=treat_col,
                y_col="y",
                cluster_col=cluster_col,
                bin_days=bin_days,
                window=window,
            )
            _append_es_rows(
                es_rows,
                es_df=es_df,
                design=design,
                outcome=outcome,
                spec="clean_TWFE",
                cohort_label=cohort_label,
                cluster_col=cluster_col,
            )
            if not es_df.empty:
                es_avg = static_es_post_avg(es_df)
                static_rows.append(
                    _static_row(
                        design=design,
                        outcome=outcome,
                        spec="static_es_post_avg",
                        res={
                            "beta": es_avg.get("beta"),
                            "se": es_avg.get("se"),
                            "pvalue": float("nan"),
                            "n_obs": static.get("n_obs"),
                            "n_clusters": static.get("n_clusters"),
                            "estimation_note": es_avg.get("estimation_note"),
                        },
                        cluster_col=cluster_col,
                        cohort_label=cohort_label,
                        extra={"n_bins": es_avg.get("n_bins")},
                    )
                )
            if write_figures and design == "cross_language":
                plot_english_quality_event_study(
                    es_df,
                    outcome,
                    design,
                    figures_dir / design / outcome / "es_clean_TWFE.png",
                    bin_days=bin_days,
                    spec_label="clean_TWFE",
                )

    static_df = pd.DataFrame(static_rows)
    es_df_all = pd.DataFrame(es_rows)
    wa_static_df = pd.DataFrame(wa_static_rows)
    wa_es_df = pd.DataFrame(wa_es_rows)
    static_df.to_csv(tables_dir / f"static_{design}_{file_tag}.csv", index=False)
    es_df_all.to_csv(tables_dir / f"event_study_{design}_{file_tag}.csv", index=False)
    if design == "cross_language":
        wa_static_df.to_csv(tables_dir / f"static_within_author_diff_{file_tag}.csv", index=False)
        wa_es_df.to_csv(tables_dir / f"event_study_within_author_diff_{file_tag}.csv", index=False)
    return static_df, es_df_all, wa_static_df, wa_es_df


def run_cross_language_heterogeneity(
    panel: pd.DataFrame,
    outcomes: Sequence[str],
    *,
    cohort_label: str,
    cluster_col: str,
    config: Dict[str, Any],
    file_tag: str,
    roster_window: str = "pre_ban",
) -> pd.DataFrame:
    """Function summary: cross-language static effect split by dominant pre-ban language.

    Splits the headline cross_language cohort by dominant_pre_lang (it/en/other)
    and re-estimates the within-language-standardized static EN-vs-IT effect,
    using moments computed on the full cross sample for comparability.

    Parameters:
    - panel: full comment panel.
    - outcomes: outcome column names.
    - cohort_label: strict or loose cohort gate.
    - cluster_col: cluster for SEs.
    - config: study YAML.
    - file_tag: output filename tag.
    - roster_window: pre_ban or full run subdir for outputs.

    Returns:
    - Heterogeneity static results DataFrame.
    """
    work_all = apply_cohort_filter(panel, "cross_language", cohort_label)
    work_all = filter_language_pair_sample(work_all)
    rows: List[Dict[str, Any]] = []
    if not work_all.empty:
        for outcome in outcomes:
            if outcome not in work_all.columns:
                continue
            moments = march_standardization_moments_by_lang(work_all, outcome)
            std = apply_standardized_outcome(
                work_all, outcome, moments, group_col="lang_comment", out_col="y"
            )
            for lang_grp in DOMINANT_LANG_GROUPS:
                sub = std.loc[std["dominant_pre_lang"].astype(str) == lang_grp]
                if sub.empty:
                    continue
                res = estimate_static_post_treat(
                    sub,
                    treat_col="is_english",
                    y_col="y",
                    cluster_col=cluster_col,
                    include_treat_main=True,
                )
                rows.append(
                    {
                        "design": "cross_language",
                        "outcome": outcome,
                        "dominant_pre_lang": lang_grp,
                        "spec": "static_post_treat",
                        "beta": res.get("beta"),
                        "se": res.get("se"),
                        "pvalue": res.get("pvalue"),
                        "n_obs": res.get("n_obs"),
                        "n_clusters": res.get("n_clusters"),
                        "n_authors": int(sub["author"].nunique()),
                        "estimation_note": res.get("estimation_note"),
                        "cluster_col": cluster_col,
                        "cohort": cohort_label,
                    }
                )
    out = pd.DataFrame(rows)
    tables_dir = english_quality_run_tables_dir(config, roster_window)
    out.to_csv(tables_dir / f"static_cross_language_by_dominant_lang_{file_tag}.csv", index=False)
    return out


def run_italian_placebo(
    panel: pd.DataFrame,
    outcomes: Sequence[str],
    *,
    designs: Sequence[DesignId],
    cohort_label: str,
    cluster_col: str,
    config: Dict[str, Any],
    file_tag: str,
    roster_window: str = "pre_ban",
) -> pd.DataFrame:
    """Function summary: Italian-language placebo (within-IT post effect) per cross design.

    Under the ChatGPT-helps-English story, the post effect estimated on Italian
    comments only should be near zero; a large coefficient flags a general time
    confound rather than an English-writing-assistance channel.

    Parameters:
    - panel: full comment panel.
    - outcomes: outcome column names.
    - designs: cross_language-family designs to evaluate.
    - cohort_label: strict or loose cohort gate.
    - cluster_col: cluster for SEs.
    - config: study YAML.
    - file_tag: output filename tag.
    - roster_window: pre_ban or full run subdir for outputs.

    Returns:
    - Placebo results DataFrame.
    """
    rows: List[Dict[str, Any]] = []
    for design in designs:
        for outcome in outcomes:
            sample = prepare_design_sample(panel, design, outcome, cohort_label)
            if sample.empty:
                continue
            res = estimate_within_language_post(
                sample, lang_value="it", y_col="y", cluster_col=cluster_col
            )
            rows.append(
                {
                    "design": design,
                    "outcome": outcome,
                    "placebo_lang": "it",
                    "spec": "within_it_post",
                    "beta": res.get("beta"),
                    "se": res.get("se"),
                    "pvalue": res.get("pvalue"),
                    "n_obs": res.get("n_obs"),
                    "n_clusters": res.get("n_clusters"),
                    "estimation_note": res.get("estimation_note"),
                    "cluster_col": cluster_col,
                    "cohort": cohort_label,
                }
            )
    out = pd.DataFrame(rows)
    tables_dir = english_quality_run_tables_dir(config, roster_window)
    out.to_csv(tables_dir / f"placebo_italian_{file_tag}.csv", index=False)
    return out


def outcomes_for_design(
    design: DesignId,
    *,
    cli_outcomes: Sequence[str],
    polarization: Sequence[str],
    include_polarization: bool,
) -> List[str]:
    """Function summary: merge design-specific headline outcomes with optional polarization set.

    Parameters:
    - design: estimation design id.
    - cli_outcomes: outcomes from --outcomes (used for native_control override).
    - polarization: polarization outcome columns.
    - include_polarization: when True, append polarization outcomes.

    Returns:
    - Deduplicated outcome list for one design run.
    """
    if design == "native_control":
        base = list(cli_outcomes) if cli_outcomes else list(HEADLINE_OUTCOMES)
    else:
        base = list(headline_outcomes_for_design(design))
    if include_polarization:
        base = list(dict.fromkeys([*base, *polarization]))
    return base


def main() -> None:
    """Function summary: CLI entry — estimate all English-quality DiD designs."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    cli_outcomes = [o.strip() for o in args.outcomes.split(",") if o.strip()]
    polarization = [o.strip() for o in args.polarization_outcomes.split(",") if o.strip()]
    panel = load_panel(config, args.bin_days, roster_window=args.roster_window)

    file_tag = args.cohort
    if args.political_only:
        if "comment_in_political_universe" in panel.columns:
            panel = panel.loc[panel["comment_in_political_universe"].astype(bool)].copy()
        file_tag = f"{args.cohort}_politicalonly"

    print(
        f"[estimate_english_quality_did] roster_window={args.roster_window} "
        f"panel rows={len(panel):,} authors={panel['author'].nunique():,} tag={file_tag}",
        flush=True,
    )
    for design in ALL_DESIGNS:
        design_outcomes = outcomes_for_design(
            design,
            cli_outcomes=cli_outcomes,
            polarization=polarization,
            include_polarization=not args.no_polarization,
        )
        static_df, es_df, wa_static_df, wa_es_df = run_design_estimates(
            panel,
            design,
            design_outcomes,
            cohort_label=args.cohort,
            bin_days=args.bin_days,
            cluster_col=args.cluster,
            window=args.window,
            config=config,
            write_figures=not args.no_figures,
            file_tag=file_tag,
            wcb_draws=args.wcb_draws,
            roster_window=args.roster_window,
        )
        print(
            f"  {design}: outcomes={len(design_outcomes)} static={len(static_df)} "
            f"event_study={len(es_df)} within_author_diff={len(wa_es_df)}",
            flush=True,
        )

    cross_outcomes = outcomes_for_design(
        "cross_language",
        cli_outcomes=cli_outcomes,
        polarization=polarization,
        include_polarization=not args.no_polarization,
    )
    het = run_cross_language_heterogeneity(
        panel,
        cross_outcomes,
        cohort_label=args.cohort,
        cluster_col=args.cluster,
        config=config,
        file_tag=file_tag,
        roster_window=args.roster_window,
    )
    cross_designs: Tuple[DesignId, ...] = tuple(d for d in ALL_DESIGNS if d in CROSS_LANGUAGE_DESIGNS)
    placebo = run_italian_placebo(
        panel,
        cross_outcomes,
        designs=cross_designs,
        cohort_label=args.cohort,
        cluster_col=args.cluster,
        config=config,
        file_tag=file_tag,
        roster_window=args.roster_window,
    )
    print(
        f"  heterogeneity rows={len(het)} italian_placebo rows={len(placebo)}",
        flush=True,
    )

    ref_dates = plot_reference_dates_calendar_utc(config)
    if len(ref_dates) >= 2:
        print(f"Ban window: {ref_dates[0]} – {ref_dates[1]} (post-lift window is short; see docs)", flush=True)


if __name__ == "__main__":
    main()
