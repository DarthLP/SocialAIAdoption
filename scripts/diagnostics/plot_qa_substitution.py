"""
Script summary:
Ban-window shaded plots for the Italian Q&A substitution test (volume, question share, event study).

Functionality:
- Daily volume indexed to pre-ban mean by group (Q&A, non-Q&A, hubs).
- Question-share time series with ban shading.
- 3-day event-study coefficient plot from qa_event_study_3d.csv.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_qa_volume_panel.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/analysis/qa_volume_did.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/plot_qa_substitution.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

BAN_START = "2023-03-31"
BAN_END = "2023-04-28"
ROLLING_WINDOW = 7


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

from scripts.diagnostics.descriptives_util import grouped_trailing_daily_rolling  # noqa: E402
from src.config_utils import figures_subdir, load_config, tables_subdir  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot Q&A substitution ban-window series.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--rolling-window", type=int, default=ROLLING_WINDOW)
    return parser.parse_args()


def _shade_ban(ax: plt.Axes, ban_start: str, ban_end: str) -> None:
    """Function summary: add grey ban window shading and vertical onset line.

    Parameters:
    - ax: matplotlib axes.
    - ban_start: ban onset date string.
    - ban_end: lift date string.
    """
    start = pd.Timestamp(ban_start)
    end = pd.Timestamp(ban_end)
    ax.axvspan(start, end, color="0.85", alpha=0.5, zorder=0)
    ax.axvline(start, color="0.35", linestyle="--", linewidth=1.0, zorder=1)


def _pool_daily(panel: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Function summary: sum or mean a metric by date and group column.

    Parameters:
    - panel: subreddit-day panel.
    - group_col: qa / IT / is_hub derived group id column name in output.
    - value_col: column to aggregate.

    Returns:
    - date_utc × group daily series.
    """
    work = panel.copy()
    if group_col == "group":
        work["group"] = "other"
        work.loc[work["qa"].astype(int) == 1, "group"] = "IT_QA"
        work.loc[(work["IT"].astype(int) == 1) & (work["qa"].astype(int) == 0), "group"] = "IT_nonQA"
        work.loc[work["is_hub"].astype(int) == 1, "group"] = "hubs"
    rows: List[Dict[str, object]] = []
    for (day, grp_name), grp in work.groupby(["date_utc", group_col], sort=True):
        if value_col in {"question_share", "qmark_rate_100w"}:
            w = pd.to_numeric(grp["n_comments"], errors="coerce").fillna(1).clip(lower=1)
            v = pd.to_numeric(grp[value_col], errors="coerce")
            ok = v.notna()
            val = float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")
        else:
            val = float(pd.to_numeric(grp[value_col], errors="coerce").sum())
        rows.append({"date_utc": str(day), group_col: grp_name, value_col: val})
    return pd.DataFrame(rows)


def _index_to_preban(daily: pd.DataFrame, group_col: str, value_col: str, launch: str) -> pd.DataFrame:
    """Function summary: index daily series to pre-ban mean (=100) by group.

    Parameters:
    - daily: pooled daily series.
    - group_col: group identifier column.
    - value_col: value column.
    - launch: ban onset date.

    Returns:
    - Copy with indexed_value column.
    """
    out = daily.copy()
    out["date_utc"] = out["date_utc"].astype(str)
    indexed: List[pd.DataFrame] = []
    for name, grp in out.groupby(group_col):
        pre = grp[grp["date_utc"] < launch][value_col]
        base = float(pre.mean()) if len(pre) and pre.notna().any() else float("nan")
        g = grp.copy()
        g["indexed_value"] = 100.0 * g[value_col] / base if base and base > 0 else float("nan")
        g["group"] = name
        indexed.append(g)
    return pd.concat(indexed, ignore_index=True) if indexed else pd.DataFrame()


def plot_volume_indexed(panel: pd.DataFrame, launch: str, out_path: Path, rolling: int) -> None:
    """Function summary: plot pre-ban-indexed comment volume by group with ban shading.

    Parameters:
    - panel: 1d panel.
    - launch: ban onset.
    - out_path: PNG output path.
    - rolling: trailing rolling window days.
    """
    daily = _pool_daily(panel, "group", "n_comments")
    indexed = _index_to_preban(daily, "group", "n_comments", launch)
    if indexed.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"IT_QA": "#c0392b", "IT_nonQA": "#2980b9", "hubs": "#7f8c8d", "other": "#bdc3c7"}
    roll = grouped_trailing_daily_rolling(indexed, "group", rolling, date_col="date_utc")
    for name, grp in roll.groupby("group"):
        ax.plot(
            pd.to_datetime(grp["date_utc"]),
            grp["indexed_value"],
            label=str(name),
            color=colors.get(str(name), None),
            linewidth=1.6,
        )
    _shade_ban(ax, BAN_START, BAN_END)
    ax.set_title("Comment volume (pre-ban mean = 100, 7d rolling)")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Indexed volume")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_question_share(panel: pd.DataFrame, out_path: Path, rolling: int) -> None:
    """Function summary: plot question_share time series for Q&A vs non-Q&A.

    Parameters:
    - panel: 1d panel restricted to Italian forums.
    - out_path: PNG output path.
    - rolling: trailing rolling window.
    """
    it = panel[panel["IT"].astype(int) == 1].copy()
    daily = _pool_daily(it, "group", "question_share")
    daily = daily[daily["group"].isin(["IT_QA", "IT_nonQA"])]
    if daily.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"IT_QA": "#c0392b", "IT_nonQA": "#2980b9"}
    roll = grouped_trailing_daily_rolling(daily, "group", rolling, date_col="date_utc")
    for name, grp in roll.groupby("group"):
        ax.plot(
            pd.to_datetime(grp["date_utc"]),
            100.0 * grp["question_share"],
            label=str(name),
            color=colors.get(str(name), None),
            linewidth=1.6,
        )
    _shade_ban(ax, BAN_START, BAN_END)
    ax.set_title("Share of comments with '?' (7d rolling)")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Question share (%)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_event_study(es: pd.DataFrame, outcome: str, out_path: Path) -> None:
    """Function summary: plot 3-day event-study coefficients for one outcome.

    Parameters:
    - es: event study coefficient table.
    - outcome: outcome id filter.
    - out_path: PNG path.
    """
    sub = es[es["outcome"] == outcome].sort_values("rel_period")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    x = sub["rel_period"].astype(int)
    y = sub["beta"].astype(float)
    se = sub["se"].astype(float)
    ax.errorbar(x, y, yerr=1.96 * se, fmt="o-", capsize=3, color="#2c3e50")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.axvline(-0.5, color="0.35", linestyle="--", linewidth=1.0)
    y_label = "Coefficient (log1p scale)" if outcome in {"n_comments", "n_questions", "n_authors"} else "Coefficient"
    ax.set_title(f"Event study (3d bins): {outcome} (Q&A differential)")
    ax.set_xlabel("rel_period (3-day bins; ref = -1)")
    ax.set_ylabel(y_label)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Function summary: write Q&A substitution figures."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    launch = BAN_START
    tables = tables_subdir(config, "qa_substitution")
    figures = figures_subdir(config, "qa_substitution")
    figures.mkdir(parents=True, exist_ok=True)

    panel_path = tables / "qa_volume_panel_1d.csv"
    if not panel_path.is_file():
        raise FileNotFoundError(f"Missing {panel_path}; run prepare_qa_volume_panel.py first.")
    panel = pd.read_csv(panel_path)

    plot_volume_indexed(panel, launch, figures / "volume_indexed_ban_shaded.png", args.rolling_window)
    plot_question_share(panel, figures / "question_share_ban_shaded.png", args.rolling_window)

    es_path = tables / "qa_event_study_3d.csv"
    if es_path.is_file():
        es = pd.read_csv(es_path)
        for outcome in ("n_comments", "question_share"):
            plot_event_study(es, outcome, figures / f"event_study_{outcome}_3d.png")

    print(f"[plot_qa_substitution] wrote figures to {figures}", flush=True)


if __name__ == "__main__":
    main()
