"""
Script summary:
Daily country time series for headline outcomes with ChatGPT ban window shaded.

Functionality:
- Plots Italy vs pooled controls (separate panels per control country group).
- 7-day trailing rolling means from descriptives / semantic / Wordfish tables.
- Grey axvspan 2023-03-31–2023-04-28 and vertical line at ban onset.

How to apply/run:
  .venv/bin/python scripts/diagnostics/plot_descriptives_ban_shaded.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


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
from scripts.diagnostics.prepare_polarization_descriptives import COUNTRY_PANEL_FAMILIES  # noqa: E402
from src.config_utils import figures_subdir, load_config, tables_subdir  # noqa: E402
from src.did.outcomes import HEADLINE_EVENT_STUDY_OUTCOMES, outcome_label  # noqa: E402

BAN_START = "2023-03-31"
BAN_END = "2023-04-28"
CONTROLS = ("Germany", "EU_hub_en", "UK", "US")
ITALY_PANELS = ("Italy_political", "Italy_others")

OUTCOME_METRIC: Dict[str, tuple[str, str]] = {
    "ai_style_rate": ("descriptives", "ai_style_rate_100w_mean"),
    "em_dash_rate": ("descriptives", "em_dash_rate_100w"),
    "sem_axis_ideology": ("semantic", "sem_axis_ideology_mean"),
    "sem_axis_aggression": ("semantic", "sem_axis_aggression_mean"),
    "wf_extremity_z": ("wordfish", "extremity_z"),
}


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for ban-window descriptives plots."""
    parser = argparse.ArgumentParser(description="Ban-window shaded descriptives.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--rolling-window", type=int, default=7)
    return parser.parse_args()


def _daily_lexical(config) -> pd.DataFrame:
    """Function summary: load daily_country_panel descriptives."""
    path = tables_subdir(config, "descriptives") / "daily_country_panel.csv"
    return pd.read_csv(path)


def _daily_semantic(config, metric: str) -> pd.DataFrame:
    """Function summary: aggregate semantic_axis_panel to country_panel × date."""
    path = tables_subdir(config, "semantic_axis") / "semantic_axis_panel.csv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "period_start" in df.columns:
        df["date_utc"] = df["period_start"].astype(str).str[:10]
    elif "date_utc" not in df.columns:
        return pd.DataFrame()
    df["country_panel"] = df["topic_family"].astype(str).map(COUNTRY_PANEL_FAMILIES)
    df = df[df["country_panel"].notna()]
    if metric not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby(["country_panel", "date_utc"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "value"})
    )


def _daily_wordfish(config) -> pd.DataFrame:
    """Function summary: aggregate forum Wordfish extremity_z to country × date."""
    path = tables_subdir(config, "wordfish") / "wordfish_extremity_panel.csv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    date_col = "date_utc" if "date_utc" in df.columns else "bin_start"
    df["date_utc"] = df[date_col].astype(str).str[:10]
    if "topic_family" in df.columns:
        df["country_panel"] = df["topic_family"].astype(str).map(COUNTRY_PANEL_FAMILIES)
    else:
        df["country_panel"] = df.get("primary_lexicon", "").astype(str)
    df = df[df["country_panel"].notna()]
    return (
        df.groupby(["country_panel", "date_utc"], as_index=False)["extremity_z"]
        .mean()
        .rename(columns={"extremity_z": "value"})
    )


def _series_for_outcome(config, outcome_id: str) -> pd.DataFrame:
    """Function summary: long daily series with country_panel and value column."""
    source, metric = OUTCOME_METRIC[outcome_id]
    if source == "descriptives":
        df = _daily_lexical(config)
        if metric not in df.columns:
            return pd.DataFrame()
        out = df[["country_panel", "date_utc", metric]].rename(columns={metric: "value"})
    elif source == "semantic":
        out = _daily_semantic(config, metric)
    else:
        out = _daily_wordfish(config)
    out["date_utc"] = pd.to_datetime(out["date_utc"])
    return out.sort_values(["country_panel", "date_utc"])


def _italy_daily(roll: pd.DataFrame) -> pd.DataFrame:
    """Function summary: mean across Italian topic-family panels per date."""
    it = roll[roll["country_panel"].isin(ITALY_PANELS)]
    if it.empty:
        return pd.DataFrame()
    return it.groupby("date_utc", as_index=False)["value"].mean()


def _plot_outcome(series: pd.DataFrame, outcome_id: str, out_path: Path, rolling_window: int) -> None:
    """Function summary: four-panel IT vs control country lines with ban shading."""
    if series.empty:
        return
    roll = grouped_trailing_daily_rolling(
        series.rename(columns={"value": outcome_id}),
        group_col="country_panel",
        rolling_window_days=rolling_window,
        date_col="date_utc",
    )
    roll = roll.rename(columns={outcome_id: "value"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    it_daily = _italy_daily(roll)
    for ax, ctrl in zip(axes.flatten(), CONTROLS):
        ct = roll[roll["country_panel"] == ctrl]
        if not it_daily.empty:
            ax.plot(it_daily["date_utc"], it_daily["value"], color="#c1121f", linewidth=2, label="Italy")
        if not ct.empty:
            ax.plot(ct["date_utc"], ct["value"], color="#457b9d", linewidth=1.5, label=ctrl)
        ax.axvspan(pd.Timestamp(BAN_START), pd.Timestamp(BAN_END), color="0.85", alpha=0.5, zorder=0)
        ax.axvline(pd.Timestamp(BAN_START), color="0.4", linestyle="--", linewidth=0.9)
        ax.set_title(f"IT vs {ctrl}")
        ax.legend(fontsize=8)
    fig.suptitle(outcome_label(outcome_id, short=True), fontsize=12)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Function summary: write ban_window/*.png for headline outcomes."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    out_dir = figures_subdir(config, "descriptives") / "ban_window"
    for oid in HEADLINE_EVENT_STUDY_OUTCOMES:
        series = _series_for_outcome(config, oid)
        path = out_dir / f"{oid}.png"
        _plot_outcome(series, oid, path, args.rolling_window)
        print(f"[plot_descriptives_ban_shaded] {oid} -> {path.name}", flush=True)


if __name__ == "__main__":
    main()
