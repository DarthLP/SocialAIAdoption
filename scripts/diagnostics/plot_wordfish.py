"""
Script summary:
Plot Wordfish robustness figures from prepared CSV tables.

Functionality:
- Dispersion timeseries per (language, time_bin) — no cross-fit level comparison.
- Mean extremity by topic_family (day-primary; split IT vs EN panels).
- Top +/- axis words per fit (plus day-primary alias PNGs).
- Subreddit-mean theta vs net_ideology and sem_axis_ideology scatter.

How to apply/run:
  .venv/bin/python scripts/diagnostics/plot_wordfish.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

from src.config_utils import (  # noqa: E402
    figures_subdir,
    load_config,
    load_wordfish_config,
    plot_reference_dates_calendar_utc,
    tables_subdir,
)


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot Wordfish descriptives.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def add_ref_lines(ax: plt.Axes, config: dict) -> None:
    """Function summary: vertical ban reference lines."""
    for dt in plot_reference_dates_calendar_utc(config):
        ax.axvline(pd.Timestamp(dt), color="red", linestyle=":", linewidth=1.0, alpha=0.8)


def plot_dispersion_timeseries(
    disp: pd.DataFrame,
    language: str,
    time_bin: str,
    fig_dir: Path,
    config: dict,
) -> None:
    """Function summary: family dispersion trajectories for one fit.

    Parameters:
    - disp: dispersion panel subset.
    - language: it or en.
    - time_bin: day or week.
    - fig_dir: output directory.
    - config: study config for reference dates.
    """
    sub = disp[(disp["primary_lexicon"] == language) & (disp["time_bin"] == time_bin)].copy()
    if sub.empty:
        return
    sub["bin_ts"] = pd.to_datetime(sub["bin_start"])
    fig, ax = plt.subplots(figsize=(11, 5))
    for family, grp in sub.groupby("topic_family"):
        grp = grp.sort_values("bin_ts")
        ax.plot(
            grp["bin_ts"],
            grp["dispersion_var"],
            marker="o",
            markersize=3,
            label=family,
            alpha=0.85,
        )
    add_ref_lines(ax, config)
    ax.set_title(
        f"Wordfish θ dispersion by topic_family ({language}, {time_bin})\n"
        "Levels comparable only within family; not across it vs en fits."
    )
    ax.set_xlabel("bin_start")
    ax.set_ylabel("dispersion_var (NaN if n_subreddits < 2)")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = fig_dir / f"dispersion_timeseries_{language}_{time_bin}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_axis_words(
    axis_path: Path,
    language: str,
    time_bin: str,
    fig_dir: Path,
) -> None:
    """Function summary: horizontal bar chart of top +/- beta words.

    Parameters:
    - axis_path: wordfish_axis_words CSV path.
    - language: lexicon code.
    - time_bin: day or week.
    - fig_dir: output directory.
    """
    if not axis_path.is_file():
        return
    df = pd.read_csv(axis_path)
    if df.empty:
        return
    colors = df["sign"].map({"pos": "#2166ac", "neg": "#b2182b"}).fillna("#888888")
    fig, ax = plt.subplots(figsize=(8, max(6, len(df) * 0.22)))
    y = range(len(df))
    ax.barh(y, df["beta"], color=colors)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["word"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("beta (word weight)")
    ax.set_title(f"Wordfish axis words ({language}, {time_bin})")
    fig.tight_layout()
    out = fig_dir / f"axis_words_{language}_{time_bin}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


IT_TOPIC_FAMILIES = frozenset({"it_political", "it_others"})
EN_TOPIC_FAMILIES = frozenset({"us", "uk", "eu"})


def plot_extremity_timeseries_by_family(
    ext: pd.DataFrame,
    primary_time_bin: str,
    fig_dir: Path,
    config: dict,
) -> None:
    """Function summary: mean extremity by topic_family; IT and EN on separate axes.

    Parameters:
    - ext: wordfish_extremity_panel.
    - primary_time_bin: headline bin (typically day).
    - fig_dir: output directory.
    - config: study config for reference dates.
    """
    sub = ext[ext["time_bin"] == primary_time_bin].copy()
    if sub.empty:
        return
    sub["bin_ts"] = pd.to_datetime(sub["bin_start"])
    agg = (
        sub.groupby(["topic_family", "bin_ts", "primary_lexicon"], as_index=False)["extremity"]
        .mean()
        .sort_values("bin_ts")
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    panels = [
        (axes[0], IT_TOPIC_FAMILIES, "Italian topic families (it fit)"),
        (axes[1], EN_TOPIC_FAMILIES, "English control families (en fit)"),
    ]
    for ax, families, title in panels:
        panel = agg[agg["topic_family"].isin(families)]
        for family, grp in panel.groupby("topic_family"):
            grp = grp.sort_values("bin_ts")
            ax.plot(
                grp["bin_ts"],
                grp["extremity"],
                marker="o",
                markersize=3,
                label=family,
                alpha=0.85,
            )
        add_ref_lines(ax, config)
        ax.set_title(title)
        ax.set_ylabel("mean extremity (within-fit)")
        ax.legend(loc="best", fontsize=8)
    axes[1].set_xlabel("bin_start (date_utc for day bins)")
    fig.suptitle(
        f"Wordfish extremity by topic_family ({primary_time_bin} bins only)\n"
        "Within-fit matched bins; levels not comparable across it vs en panels.",
        fontsize=10,
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / "extremity_timeseries_by_family.png", dpi=150)
    plt.close(fig)


def plot_theta_validation_scatter(
    merged: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """Function summary: scatter subreddit means: theta vs net_ideology and sem_axis.

    Parameters:
    - merged: wordfish_subreddit_theta_ideology frame.
    - fig_dir: output directory.
    """
    if merged.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, ycol, title in zip(
        axes,
        ("net_ideology_mean", "sem_axis_ideology_mean"),
        ("theta vs net_ideology", "theta vs sem_axis_ideology"),
    ):
        if ycol not in merged.columns:
            continue
        for (lang, tbin), grp in merged.groupby(["primary_lexicon", "time_bin"]):
            ax.scatter(
                grp[ycol],
                grp["theta"],
                alpha=0.55,
                s=25,
                label=f"{lang}/{tbin}",
            )
        ax.set_xlabel(ycol)
        ax.set_ylabel("subreddit-mean theta")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.suptitle("Sign matters for theta interpretation (anchor convention)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "theta_vs_lexicon_scatter.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Function summary: render all Wordfish figures."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    wf_cfg = load_wordfish_config(config)
    tbl = tables_subdir(config, "wordfish")
    fig_dir = figures_subdir(config, "wordfish")
    fig_dir.mkdir(parents=True, exist_ok=True)

    disp_path = tbl / "wordfish_dispersion_panel.csv"
    pos_path = tbl / "wordfish_positions.csv"
    ext_path = tbl / "wordfish_extremity_panel.csv"
    if not disp_path.is_file() or not pos_path.is_file():
        print("[plot_wordfish] missing prepared tables — run prepare_wordfish.py first", flush=True)
        return

    disp = pd.read_csv(disp_path)
    positions = pd.read_csv(pos_path)
    primary_tbin = str(wf_cfg.get("primary_time_bin", "day"))

    if ext_path.is_file():
        plot_extremity_timeseries_by_family(
            pd.read_csv(ext_path), primary_tbin, fig_dir, config
        )

    for lang in wf_cfg.get("languages", ["it", "en"]):
        for tbin in wf_cfg.get("time_bins", ["day", "week"]):
            plot_dispersion_timeseries(disp, lang, tbin, fig_dir, config)
            axis_csv = tbl / f"wordfish_axis_words_{lang}_{tbin}.csv"
            plot_axis_words(axis_csv, lang, tbin, fig_dir)
            if tbin == primary_tbin:
                src = fig_dir / f"axis_words_{lang}_{primary_tbin}.png"
                if src.is_file():
                    shutil.copy2(src, fig_dir / f"axis_words_{lang}.png")

    sub_path = tbl / "wordfish_subreddit_theta_ideology.csv"
    if sub_path.is_file():
        plot_theta_validation_scatter(pd.read_csv(sub_path), fig_dir)

    print(f"[plot_wordfish] wrote figures to {fig_dir}", flush=True)


if __name__ == "__main__":
    main()
