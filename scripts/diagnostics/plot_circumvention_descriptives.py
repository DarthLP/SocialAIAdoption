"""
Script summary:
Plot VPN/Tor circumvention time series for the Reddit study window and full download span.

Functionality:
- Four figures: Google Trends VPN and Tor bridge users, each for Mar–Apr 2023 (event_window)
  and for the full Jan–Jun 2023 download window.
- Optional twin-axis overlay: semantic ideology vs Italy VPN (study window).
- DiD audit QC: geo-matched VPN levels vs Italy broadcast on 7d bins.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_circumvention_descriptives.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/plot_circumvention_descriptives.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

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

from scripts.diagnostics.descriptives_util import event_dates_from_config  # noqa: E402
from src.circumvention import load_circumvention_daily  # noqa: E402
from src.config_utils import (  # noqa: E402
    figures_subdir,
    load_config,
    plot_reference_dates_calendar_utc,
    tables_subdir,
)

FULL_DOWNLOAD_START = "2023-01-01"
FULL_DOWNLOAD_END_EXCLUSIVE = "2023-07-01"
_DATE_AXIS_LABEL_7D = "Period start (UTC, launch-aligned 7-day bins)"


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot circumvention descriptives figures.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def _add_reference_lines(ax: plt.Axes, ref_dates: list[str]) -> None:
    """Function summary: draw vertical lines at plot_reference_dates_utc calendar dates."""
    for d in ref_dates:
        ax.axvline(pd.Timestamp(d), color="0.4", linestyle="--", linewidth=0.9, alpha=0.8)


def _clip_study_window(daily: pd.DataFrame, start: str, end_exclusive: str) -> pd.DataFrame:
    """Function summary: restrict daily circumvention frame to event_window dates."""
    if daily.empty:
        return daily.copy()
    dates = daily["date_utc"].astype(str)
    return daily[(dates >= start) & (dates < end_exclusive)].copy()


def _plot_metric_by_geo(
    daily: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
    ref_dates: list[str],
) -> None:
    """Function summary: one line per geo for a single circumvention metric.

    Parameters:
    - daily: circumvention_daily_by_geo-style frame.
    - metric_col: column to plot (vpn_interest or tor_bridge_users).
    - ylabel: y-axis label.
    - title: figure suptitle.
    - out_path: PNG output path.
    - ref_dates: ban/lift reference dates for vertical lines.

    Returns:
    - None.
    """
    if daily.empty or metric_col not in daily.columns:
        return
    geos = sorted(daily["geo"].astype(str).unique())
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for geo in geos:
        sub = daily[daily["geo"] == geo].sort_values("date_utc")
        if sub[metric_col].notna().sum() == 0:
            continue
        ax.plot(
            pd.to_datetime(sub["date_utc"]),
            sub[metric_col].astype(float),
            label=geo,
            linewidth=1.2,
            marker="o",
            markersize=2.5,
        )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date (UTC)")
    _add_reference_lines(ax, ref_dates)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_four_circumvention_panels(
    daily_full: pd.DataFrame,
    daily_study: pd.DataFrame,
    daily_dir: Path,
    ref_dates: list[str],
    study_label: str,
) -> None:
    """Function summary: write VPN and Tor figures for study window and full download.

    Parameters:
    - daily_full: Jan–Jun 2023 (download window).
    - daily_study: Mar–Apr 2023 (event_window).
    - daily_dir: figures/daily output directory.
    - ref_dates: reference vertical lines.
    - study_label: human-readable study span for titles.

    Returns:
    - None.
    """
    specs = [
        (
            "vpn_interest",
            "Google Trends VPN interest (topic)",
            daily_study,
            f"Google Trends VPN — study window ({study_label})",
            daily_dir / "google_trends_vpn_study_window.png",
        ),
        (
            "vpn_interest",
            "Google Trends VPN interest (topic)",
            daily_full,
            "Google Trends VPN — full download (2023-01-01 to 2023-06-30)",
            daily_dir / "google_trends_vpn_full_download.png",
        ),
        (
            "tor_bridge_users",
            "Tor bridge users (estimated)",
            daily_study,
            f"Tor bridge users — study window ({study_label})",
            daily_dir / "tor_bridge_study_window.png",
        ),
        (
            "tor_bridge_users",
            "Tor bridge users (estimated)",
            daily_full,
            "Tor bridge users — full download (2023-01-01 to 2023-06-30)",
            daily_dir / "tor_bridge_full_download.png",
        ),
    ]
    for metric_col, ylabel, frame, title, path in specs:
        _plot_metric_by_geo(frame, metric_col, ylabel, title, path, ref_dates)
        print(f"[plot_circumvention_descriptives] wrote {path.name}", flush=True)


def _plot_semantic_vpn_overlay(
    circum_daily: pd.DataFrame,
    semantic_path: Path,
    out_dir: Path,
    ref_dates: list[str],
) -> None:
    """Function summary: twin-axis Italy ideology mean vs IT VPN interest (study-window VPN)."""
    if not semantic_path.is_file():
        return
    sem = pd.read_csv(semantic_path)
    it_sem = sem[sem["topic_family"].astype(str).isin(("it_political", "it_pure_political"))].copy()
    de_sem = sem[sem["topic_family"].astype(str) == "de"].copy()
    if it_sem.empty or de_sem.empty:
        return
    it_vpn = circum_daily[circum_daily["geo"] == "IT"].sort_values("date_utc")
    if it_vpn.empty:
        return

    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    x_it = pd.to_datetime(it_sem["period_start"])
    x_de = pd.to_datetime(de_sem["period_start"])
    ax1.plot(x_it, it_sem["sem_axis_ideology_mean"], label="IT political sem_axis_ideology", color="C0")
    ax1.plot(x_de, de_sem["sem_axis_ideology_mean"], label="DE sem_axis_ideology", color="C1")
    ax1.set_ylabel("Semantic ideology (mean)")
    ax2 = ax1.twinx()
    ax2.plot(
        pd.to_datetime(it_vpn["date_utc"]),
        it_vpn["vpn_interest"],
        label="IT VPN interest",
        color="C3",
        alpha=0.7,
        linewidth=1.2,
    )
    ax2.set_ylabel("IT VPN Trends index")
    _add_reference_lines(ax1, ref_dates)
    ax1.grid(True, alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    fig.suptitle("Semantic ideology (IT/DE families) vs Italy VPN interest")
    fig.tight_layout()
    fig.savefig(out_dir / "semantic_ideology_vs_vpn_it.png", dpi=150)
    plt.close(fig)


def _plot_vpn_geo_vs_it_broadcast(
    circum_path: Path,
    semantic_path: Path,
    out_path: Path,
    ref_dates: list[str],
    bin_days: int,
) -> None:
    """Function summary: QC chart — geo-matched VPN levels are not comparable; IT broadcast is pooled intensity.

    Parameters:
    - circum_path: circumvention_panel_by_geo_{bd}d.csv.
    - semantic_path: semantic_axis_panel_by_topic_family_{bd}d.csv.
    - out_path: output PNG path.
    - ref_dates: vertical reference dates.
    - bin_days: 1, 3, or 7 for axis label.

    Returns:
    - None.
    """
    bd = int(bin_days)
    x_label = (
        _DATE_AXIS_LABEL_7D
        if bd > 1
        else "Period start (UTC, daily)"
    )
    if not circum_path.is_file() or not semantic_path.is_file():
        return
    circ = pd.read_csv(circum_path)
    sem = pd.read_csv(semantic_path)
    if circ.empty or "vpn_interest" not in circ.columns or "geo" not in circ.columns:
        return
    if sem.empty or "vpn_interest_it" not in sem.columns:
        return

    fig, (ax_geo, ax_it) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=False)
    for geo_val, grp in circ.groupby("geo", sort=True):
        grp = grp.sort_values("period_start")
        ax_geo.plot(
            pd.to_datetime(grp["period_start"]),
            grp["vpn_interest"].astype(float),
            label=str(geo_val),
            marker="o",
            alpha=0.85,
        )
    _add_reference_lines(ax_geo, ref_dates)
    ax_geo.set_title("Geo-matched VPN (within-geo Trends 0–100; levels not comparable)")
    ax_geo.set_ylabel("vpn_interest by geo")
    ax_geo.set_xlabel(x_label)
    ax_geo.legend(loc="best", fontsize=7)

    it_ser = sem.drop_duplicates(subset=["period_start"]).sort_values("period_start")
    ax_it.plot(
        pd.to_datetime(it_ser["period_start"]),
        it_ser["vpn_interest_it"].astype(float),
        color="C0",
        marker="o",
        label="vpn_interest_it",
    )
    _add_reference_lines(ax_it, ref_dates)
    ax_it.set_title("Italy broadcast (use for pooled semantic DiD intensity)")
    ax_it.set_ylabel("vpn_interest_it")
    ax_it.set_xlabel(x_label)
    ax_it.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Circumvention intensity: do not pool geo-matched VPN across countries",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Function summary: CLI entry for circumvention figures."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    ref_dates = [
        d.strftime("%Y-%m-%d") for d in plot_reference_dates_calendar_utc(config)
    ]
    start, end_excl, _, _ = event_dates_from_config(config)
    study_label = f"{start} to {end_excl} (exclusive end)"

    daily_full = load_circumvention_daily(
        PROJECT_ROOT,
        config,
        start=FULL_DOWNLOAD_START,
        end_exclusive=FULL_DOWNLOAD_END_EXCLUSIVE,
    )
    daily_study = _clip_study_window(daily_full, start, end_excl)

    out_dir = figures_subdir(config, "circumvention")
    daily_dir = out_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    _plot_four_circumvention_panels(daily_full, daily_study, daily_dir, ref_dates, study_label)

    semantic_dir = tables_subdir(config, "semantic_axis")
    semantic_path = semantic_dir / "semantic_axis_panel_by_topic_family_1d.csv"
    _plot_semantic_vpn_overlay(daily_study, semantic_path, out_dir, ref_dates)

    circum_tables = tables_subdir(config, "circumvention")
    for bin_days in (1, 3, 7):
        bin_dir = out_dir / f"bins_{bin_days}d"
        bin_dir.mkdir(parents=True, exist_ok=True)
        _plot_vpn_geo_vs_it_broadcast(
            circum_tables / f"circumvention_panel_by_geo_{bin_days}d.csv",
            semantic_dir / f"semantic_axis_panel_by_topic_family_{bin_days}d.csv",
            bin_dir / "vpn_geo_levels_vs_it_broadcast.png",
            ref_dates,
            bin_days=int(bin_days),
        )
    print(f"[plot_circumvention_descriptives] wrote figures to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
