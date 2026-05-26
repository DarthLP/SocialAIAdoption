"""
Write DiD tables and event-study figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.did.outcomes import FAMILY_FIGURE_DIRS, FIRST_STAGE_OUTCOMES, HEADLINE_OUTCOMES
from src.did.specs import PLOT_STRATEGY_GROUPS, strategy_label


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


def plot_event_study(
    es_df: pd.DataFrame,
    outcome_id: str,
    out_path: Path,
    launch_label: str = "2023-03-31",
    title: Optional[str] = None,
    subtitle: str = "",
) -> None:
    """Function summary: leads/lags plot with ban vertical line at k=0."""
    if es_df.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.errorbar(
        es_df["rel_day"],
        es_df["gamma"],
        yerr=1.96 * es_df["se"],
        fmt="o-",
        capsize=3,
        color="#1f4e79",
    )
    ax.axvline(0, color="red", linestyle=":", linewidth=1.0, label=f"Ban {launch_label}")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Days relative to ban (ref = -1)")
    ax.set_ylabel("γ_k (treated × day)")
    ax.set_title(title or f"Event study: {outcome_id}")
    if subtitle:
        ax.text(0.02, 0.02, subtitle, transform=ax.transAxes, fontsize=8, color="#555")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_coef_comparison(
    summary: pd.DataFrame,
    outcome_id: str,
    out_path: Path,
    strategies: Optional[Sequence[str]] = None,
    label_col: str = "strategy_label",
    title: Optional[str] = None,
    subtitle: str = "",
) -> None:
    """Function summary: coefficient plot across strategies for one outcome."""
    sub = summary[summary["outcome_id"] == outcome_id].copy()
    if strategies:
        sub = sub[sub["strategy_id"].isin(strategies)]
    if label_col not in sub.columns:
        sub[label_col] = sub["strategy_id"].astype(str).map(strategy_label)
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
    ax.set_yticklabels(sub[label_col].astype(str), fontsize=8)
    ax.set_xlabel("β (treat × post)")
    ax.set_title(title or f"DiD estimates: {outcome_id}")
    if subtitle:
        ax.text(0.02, 0.98, subtitle, transform=ax.transAxes, fontsize=8, va="top", color="#555")
    note = sub.get("estimation_note", pd.Series(dtype=str))
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=rob_df, x="check", y="beta", ax=ax, color="#457b9d")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(f"Robustness: {outcome_id}")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_first_stage(summary: pd.DataFrame, out_path: Path) -> None:
    """Function summary: AI-writing first-stage coefs across strategies."""
    fs = summary[summary["outcome_id"].isin(FIRST_STAGE_OUTCOMES)]
    if fs.empty:
        return
    agg = fs.groupby("outcome_id", as_index=False)["beta"].mean()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=agg, x="outcome_id", y="beta", ax=ax, color="#e76f51")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("First stage: AI-writing stylometrics (mean β across strategies)")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_significance_heatmap(summary: pd.DataFrame, out_path: Path) -> None:
    """Function summary: outcome × strategy heatmap of sign and significance."""
    headline = list(PLOT_STRATEGY_GROUPS["headline"]) + list(PLOT_STRATEGY_GROUPS["by_topic"])
    sub = summary[summary["strategy_id"].isin(headline)].copy()
    if sub.empty:
        return
    if "strategy_label" not in sub.columns:
        sub["strategy_label"] = sub["strategy_id"].map(strategy_label)
    pivot_beta = sub.pivot_table(index="outcome_id", columns="strategy_label", values="beta", aggfunc="first")
    pivot_p = sub.pivot_table(index="outcome_id", columns="strategy_label", values="pvalue", aggfunc="first")
    if pivot_beta.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, pivot_beta.shape[1] * 0.9), max(5, pivot_beta.shape[0] * 0.35)))
    sns.heatmap(
        pivot_beta,
        cmap="RdBu_r",
        center=0,
        annot=pivot_p.map(lambda p: "*" if pd.notna(p) and p < 0.05 else ""),
        fmt="",
        ax=ax,
        cbar_kws={"label": "β"},
    )
    ax.set_title("DiD coefficients (* p<0.05)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_headline_forest(summary: pd.DataFrame, out_path: Path) -> None:
    """Function summary: forest plot for pre-registered headline outcomes."""
    sub = summary[
        (summary["outcome_id"].isin(HEADLINE_OUTCOMES))
        & (summary["strategy_id"] == "cross_country_all")
    ].copy()
    if sub.empty or sub["beta"].notna().sum() == 0:
        return
    sub = sub[sub["beta"].notna()]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(sub))))
    y_pos = range(len(sub))
    ax.errorbar(sub["beta"], list(y_pos), xerr=1.96 * sub["se"], fmt="o", color="#1d3557", capsize=3)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sub["outcome_id"].astype(str))
    ax.set_xlabel("β (Italian forums vs controls, full ban)")
    ax.set_title("Headline outcomes: cross-country DiD")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_early_ban_comparison(summary: pd.DataFrame, outcome_id: str, out_path: Path) -> None:
    """Function summary: compare full-ban vs 7d vs 14d early-ban estimates."""
    ids = (
        "cross_country_all",
        "cross_country_it_political",
        "cross_country_it_others",
        "cross_country_all_14d",
    )
    sub = summary[(summary["outcome_id"] == outcome_id) & (summary["strategy_id"].isin(ids))].copy()
    if sub.empty:
        return
    if "strategy_label" not in sub.columns:
        sub["strategy_label"] = sub["strategy_id"].map(strategy_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=sub, x="strategy_label", y="beta", ax=ax, color="#457b9d")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(f"Early-ban vs full ban: {outcome_id}")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ddd_panel(summary: pd.DataFrame, out_path: Path) -> None:
    """Function summary: within-Italy triple-diff coefficients across outcomes."""
    sub = summary[summary["strategy_id"] == "within_italy_ddd"].copy()
    sub = sub[sub["beta"].notna()]
    if sub.empty:
        return
    if "strategy_label" not in sub.columns:
        sub["strategy_label"] = sub["strategy_id"].map(strategy_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(sub))))
    y_pos = range(len(sub))
    ax.errorbar(sub["beta"], list(y_pos), xerr=1.96 * sub["se"], fmt="o", color="#6a4c93", capsize=3)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sub["outcome_id"].astype(str), fontsize=8)
    ax.set_xlabel("β (IT × post × political tree)")
    ax.set_title("Within-Italy triple-difference")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pretrend_summary(summary: pd.DataFrame, out_path: Path) -> None:
    """Function summary: bar chart of event-study pretrend F p-values for headline outcomes."""
    sub = summary[
        (summary["outcome_id"].isin(HEADLINE_OUTCOMES))
        & (summary["strategy_id"] == "cross_country_all")
    ].copy()
    if "pretrend_F_p" not in sub.columns or sub["pretrend_F_p"].notna().sum() == 0:
        return
    sub = sub.dropna(subset=["pretrend_F_p"])
    if sub.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=sub, x="outcome_id", y="pretrend_F_p", ax=ax, color="#a8dadc")
    ax.axhline(0.05, color="red", linestyle="--", linewidth=0.8, label="α=0.05")
    ax.set_title("Pre-trend joint F-test p-values (cross_country_all)")
    plt.xticks(rotation=25, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_overview_figures(summary: pd.DataFrame, fig_dir: Path) -> None:
    """Function summary: write all overview/ diagnostic figures."""
    overview = fig_dir / "overview"
    plot_significance_heatmap(summary, overview / "significance_heatmap.png")
    plot_headline_forest(summary, overview / "headline_forest_lexical_semantic.png")
    plot_ddd_panel(summary, overview / "ddd_political_specificity.png")
    plot_first_stage(summary, overview / "first_stage_aiwriting.png")
    plot_pretrend_summary(summary, overview / "pretrend_summary.png")
    for oid in ("net_ideology", "ai_style_rate", "sem_axis_ideology"):
        if (summary["outcome_id"] == oid).any():
            plot_early_ban_comparison(summary, oid, overview / f"early_ban_{oid}.png")
