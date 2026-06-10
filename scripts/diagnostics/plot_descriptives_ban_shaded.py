"""
Script summary:
Daily country time series for cross-country descriptives with ChatGPT ban window shaded.

Functionality:
- Plots Italy vs pooled controls (separate 2x2 panels per control country group).
- 7-day trailing rolling means from descriptives / semantic / Wordfish tables.
- Grey axvspan 2023-03-31–2023-04-28 and vertical line at ban onset.
- Outcomes from BAN_WINDOW_DESCRIPTIVE_OUTCOMES (lexical, semantic, wordfish subdirs).

How to apply/run:
  .venv/bin/python scripts/diagnostics/plot_descriptives_ban_shaded.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Dict, Iterable, Tuple

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
from src.did.outcomes import (  # noqa: E402
    BAN_WINDOW_DESCRIPTIVE_OUTCOMES,
    BAN_WINDOW_LEXICAL_EXTRA_COLUMNS,
    outcome_label,
    outcome_spec,
)

BAN_START = "2023-03-31"
BAN_END = "2023-04-28"
CONTROL_PANELS = ("Germany", "EU_hub_en", "UK", "US_political")
ITALY_PANELS = ("Italy_political", "Italy_others")

CONTROL_PANEL_DISPLAY: Dict[str, str] = {
    "Germany": "Germany",
    "EU_hub_en": "EU_hub_en",
    "UK": "UK",
    "US_political": "US",
}


def _build_outcome_metric() -> Dict[str, Tuple[str, str]]:
    """Function summary: map outcome_id to (source, column) for ban-window plots.

    Returns:
    - Dict outcome_id -> (descriptives|semantic|wordfish, column name).
    """
    mapping: Dict[str, Tuple[str, str]] = {}
    for oid in BAN_WINDOW_DESCRIPTIVE_OUTCOMES:
        spec = outcome_spec(oid)
        if spec is None:
            continue
        if spec.family == "lexical":
            mapping[oid] = ("descriptives", spec.column)
        elif spec.family == "semantic_axis":
            mapping[oid] = ("semantic", spec.column)
        elif spec.family == "wordfish_forum":
            mapping[oid] = ("wordfish", spec.column)
    for oid, col in BAN_WINDOW_LEXICAL_EXTRA_COLUMNS:
        mapping[oid] = ("descriptives", col)
    return mapping


OUTCOME_METRIC = _build_outcome_metric()


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for ban-window descriptives plots."""
    parser = argparse.ArgumentParser(description="Ban-window shaded descriptives.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--rolling-window", type=int, default=7)
    return parser.parse_args()


def _control_display(panel_id: str) -> str:
    """Function summary: human-readable control label for titles and legends.

    Parameters:
    - panel_id: canonical country_panel id from prepared tables.

    Returns:
    - Display label string.
    """
    return CONTROL_PANEL_DISPLAY.get(panel_id, panel_id)


def _empty_panel_note(panel_id: str, source: str) -> str:
    """Function summary: annotation when a control panel has no plotted series.

    Parameters:
    - panel_id: canonical country_panel id.
    - source: data source family (descriptives, semantic, wordfish).

    Returns:
    - Short note for the empty subplot.
    """
    if panel_id == "Germany" and source == "wordfish":
        return "No data (DE not in Wordfish fit)"
    return "No data"


def _outcome_subdir(source: str) -> str:
    """Function summary: figure subfolder for a data source family.

    Parameters:
    - source: descriptives, semantic, or wordfish.

    Returns:
    - Subdirectory name under ban_window/.
    """
    if source == "descriptives":
        return "lexical"
    if source == "semantic":
        return "semantic"
    return "wordfish"


def _daily_lexical(config) -> pd.DataFrame:
    """Function summary: load daily_country_panel descriptives.

    Parameters:
    - config: study YAML dict.

    Returns:
    - Country-panel daily table, or empty DataFrame if missing.
    """
    path = tables_subdir(config, "descriptives") / "daily_country_panel.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _daily_semantic(config, metric: str) -> pd.DataFrame:
    """Function summary: aggregate semantic_axis_panel to country_panel × date.

    Parameters:
    - config: study YAML dict.
    - metric: semantic axis mean column.

    Returns:
    - Long table with country_panel, date_utc, value.
    """
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


def _daily_wordfish(config, metric: str) -> pd.DataFrame:
    """Function summary: aggregate forum Wordfish panel to country × date.

    Parameters:
    - config: study YAML dict.
    - metric: Wordfish column (e.g. extremity_z, change_z).

    Returns:
    - Long table with country_panel, date_utc, value.
    """
    path = tables_subdir(config, "wordfish") / "wordfish_extremity_panel.csv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if metric not in df.columns:
        return pd.DataFrame()
    date_col = "date_utc" if "date_utc" in df.columns else "bin_start"
    df["date_utc"] = df[date_col].astype(str).str[:10]
    if "topic_family" in df.columns:
        df["country_panel"] = df["topic_family"].astype(str).map(COUNTRY_PANEL_FAMILIES)
    else:
        df["country_panel"] = df.get("primary_lexicon", "").astype(str)
    df = df[df["country_panel"].notna()]
    return (
        df.groupby(["country_panel", "date_utc"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "value"})
    )


def _series_for_outcome(config, outcome_id: str) -> Tuple[pd.DataFrame, str]:
    """Function summary: long daily series with country_panel and value column.

    Parameters:
    - config: study YAML dict.
    - outcome_id: outcome key from BAN_WINDOW_DESCRIPTIVE_OUTCOMES.

    Returns:
    - Tuple of (series DataFrame, source family string).
    """
    if outcome_id not in OUTCOME_METRIC:
        return pd.DataFrame(), "descriptives"
    source, metric = OUTCOME_METRIC[outcome_id]
    if source == "descriptives":
        df = _daily_lexical(config)
        if metric not in df.columns:
            return pd.DataFrame(), source
        out = df[["country_panel", "date_utc", metric]].rename(columns={metric: "value"})
    elif source == "semantic":
        out = _daily_semantic(config, metric)
    else:
        out = _daily_wordfish(config, metric)
    if out.empty:
        return out, source
    out["date_utc"] = pd.to_datetime(out["date_utc"])
    return out.sort_values(["country_panel", "date_utc"]), source


def _italy_daily(roll: pd.DataFrame) -> pd.DataFrame:
    """Function summary: mean across Italian topic-family panels per date.

    Parameters:
    - roll: rolled long series with country_panel and value.

    Returns:
    - Italy pooled daily means.
    """
    it = roll[roll["country_panel"].isin(ITALY_PANELS)]
    if it.empty:
        return pd.DataFrame()
    return it.groupby("date_utc", as_index=False)["value"].mean()


def _plot_outcome(
    series: pd.DataFrame,
    outcome_id: str,
    out_path: Path,
    rolling_window: int,
    source: str,
) -> None:
    """Function summary: four-panel IT vs control country lines with ban shading.

    Parameters:
    - series: long daily series (country_panel, date_utc, value).
    - outcome_id: outcome key for title.
    - out_path: PNG output path.
    - rolling_window: trailing rolling window in days.
    - source: data source family for empty-panel notes.
    """
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
    for ax, ctrl in zip(axes.flatten(), CONTROL_PANELS):
        ct = roll[roll["country_panel"] == ctrl]
        ctrl_label = _control_display(ctrl)
        if not it_daily.empty:
            ax.plot(
                it_daily["date_utc"],
                it_daily["value"],
                color="#c1121f",
                linewidth=2,
                label="Italy",
            )
        if not ct.empty:
            ax.plot(
                ct["date_utc"],
                ct["value"],
                color="#457b9d",
                linewidth=1.5,
                label=ctrl_label,
            )
        else:
            ax.text(
                0.5,
                0.5,
                _empty_panel_note(ctrl, source),
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="0.45",
            )
            print(
                f"[plot_descriptives_ban_shaded] {outcome_id}: {_control_display(ctrl)} — no series",
                flush=True,
            )
        ax.axvspan(pd.Timestamp(BAN_START), pd.Timestamp(BAN_END), color="0.85", alpha=0.5, zorder=0)
        ax.axvline(pd.Timestamp(BAN_START), color="0.4", linestyle="--", linewidth=0.9)
        ax.set_title(f"IT vs {ctrl_label}")
        ax.legend(fontsize=8)
    fig.suptitle(outcome_label(outcome_id, short=True), fontsize=12)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _all_ban_window_outcomes() -> Iterable[str]:
    """Function summary: ordered outcome ids for ban-window figure export.

    Returns:
    - Iterable of outcome_id strings (registry + extra lexical columns).
    """
    seen: set[str] = set()
    for oid in BAN_WINDOW_DESCRIPTIVE_OUTCOMES:
        if oid not in seen:
            seen.add(oid)
            yield oid
    for oid, _ in BAN_WINDOW_LEXICAL_EXTRA_COLUMNS:
        if oid not in seen:
            seen.add(oid)
            yield oid


def main() -> None:
    """Function summary: write ban_window/{lexical,semantic,wordfish}/*.png."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    out_root = figures_subdir(config, "descriptives") / "ban_window"
    for oid in _all_ban_window_outcomes():
        if oid not in OUTCOME_METRIC:
            print(f"[plot_descriptives_ban_shaded] skip {oid}: no metric mapping", flush=True)
            continue
        series, source = _series_for_outcome(config, oid)
        subdir = _outcome_subdir(source)
        path = out_root / subdir / f"{oid}.png"
        if series.empty:
            print(f"[plot_descriptives_ban_shaded] skip {oid}: empty series", flush=True)
            continue
        _plot_outcome(series, oid, path, args.rolling_window, source)
        print(f"[plot_descriptives_ban_shaded] {oid} -> {subdir}/{path.name}", flush=True)


if __name__ == "__main__":
    main()
