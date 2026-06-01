"""
Script summary:
Plot author-level Wordfish descriptives from prepared CSV tables.

Functionality:
- Mean extremity / extremity_z / change / change_z by language (headline balanced week7).
- Dispersion, coverage, axis words, and theta–lexicon figures per language.
- Robustness timeseries for week3 and full RCS when CSVs exist.
- X-axis spans full event window; ban/lift reference lines.

How to apply/run:
  .venv/bin/python scripts/diagnostics/plot_wordfish_authors_v2.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/plot_wordfish_authors.py --spec week7 --panel-mode balanced
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
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

from src.config_utils import (  # noqa: E402
    figures_subdir,
    load_config,
    load_wordfish_authors_v2_config,
    plot_reference_dates_calendar_utc,
    tables_subdir,
)


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot author Wordfish descriptives.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--spec", type=str, default=None, help="week7, week3, window, or all headline variants")
    parser.add_argument("--panel-mode", type=str, default=None, choices=("full", "balanced", None))
    return parser.parse_args()


def _event_xlim(config: dict) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Function summary: study window xlim from event_window config.

    Parameters:
    - config: loaded YAML.

    Returns:
    - Tuple (start, end inclusive display through last day before end_exclusive).
    """
    start, end_excl, _, _ = event_dates_from_config(config)
    return pd.Timestamp(start), pd.Timestamp(end_excl) - pd.Timedelta(days=1)


def _add_ref_lines(ax: plt.Axes, config: dict) -> None:
    """Function summary: vertical ban/lift reference lines (red dotted).

    Parameters:
    - ax: matplotlib axes.
    - config: study config.
    """
    for dt in plot_reference_dates_calendar_utc(config):
        ax.axvline(pd.Timestamp(dt), color="red", linestyle=":", linewidth=1.0, alpha=0.85)


def _resolve_panel_path(
    tab_dir: Path,
    panel_mode: str,
    spec: str,
    kind: str,
) -> Optional[Path]:
    """Function summary: find prepared CSV for a panel kind and tag.

    Parameters:
    - tab_dir: tables root.
    - panel_mode: full or balanced.
    - spec: week7, week3, window.
    - kind: extremity_panel, dispersion_panel, coverage, positions.

    Returns:
    - Path if file exists, else None.
    """
    tag = f"{panel_mode}_{spec}"
    name = f"wordfish_authors_{kind}_{tag}.csv"
    path = tab_dir / name
    return path if path.is_file() else None


def _plot_timeseries(
    ext: pd.DataFrame,
    value_col: str,
    out_path: Path,
    config: dict,
    title: str,
    xlim: Tuple[pd.Timestamp, pd.Timestamp],
) -> None:
    """Function summary: language-stratified mean time series over full event window.

    Parameters:
    - ext: extremity panel.
    - value_col: column to aggregate.
    - out_path: PNG path.
    - config: study config for reference lines.
    - title: figure title.
    - xlim: (start, end) for x-axis.
    """
    if ext.empty or value_col not in ext.columns:
        return
    work = ext.dropna(subset=[value_col])
    if work.empty:
        return
    daily = (
        work.groupby(["primary_lexicon", "bin_start"], as_index=False)[value_col]
        .mean()
        .sort_values("bin_start")
    )
    daily["bin_dt"] = pd.to_datetime(daily["bin_start"])
    fig, ax = plt.subplots(figsize=(12, 5))
    for lang, grp in daily.groupby("primary_lexicon"):
        ax.plot(grp["bin_dt"], grp[value_col], marker="o", label=lang, alpha=0.9)
    _add_ref_lines(ax, config)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_title(
        f"{title}\n"
        "Levels not comparable across languages; use extremity_z / change_z for cross-language sign."
    )
    ax.set_xlabel("bin_start (ban-anchored week)")
    ax.set_ylabel(f"mean {value_col} (within-fit)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_dispersion_timeseries(
    disp: pd.DataFrame,
    language: str,
    out_path: Path,
    config: dict,
    xlim: Tuple[pd.Timestamp, pd.Timestamp],
    spec: str,
    panel_mode: str,
) -> None:
    """Function summary: author-position dispersion (var) over bins for one language.

    Parameters:
    - disp: dispersion panel.
    - language: it, en, or de.
    - out_path: PNG path.
    - config: study config.
    - xlim: x-axis limits.
    - spec: time spec name.
    - panel_mode: full or balanced.
    """
    sub = disp[disp["primary_lexicon"] == language].copy()
    if sub.empty:
        return
    sub["bin_dt"] = pd.to_datetime(sub["bin_start"])
    sub = sub.sort_values("bin_dt")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        sub["bin_dt"],
        sub["dispersion_var"],
        marker="o",
        color="#4c72b0",
        alpha=0.85,
    )
    _add_ref_lines(ax, config)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_title(
        f"Author θ dispersion ({language}, {panel_mode} {spec})\n"
        "Variance across authors per bin; NaN if n_authors < 2."
    )
    ax.set_xlabel("bin_start")
    ax.set_ylabel("dispersion_var")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_coverage_by_bin(
    cov: pd.DataFrame,
    language: str,
    out_path: Path,
    config: dict,
    xlim: Tuple[pd.Timestamp, pd.Timestamp],
    spec: str,
    panel_mode: str,
) -> None:
    """Function summary: bar chart of kept author-documents per bin.

    Parameters:
    - cov: coverage audit (doc_kept rows only aggregated).
    - language: lexicon code.
    - out_path: PNG path.
    - config: study config.
    - xlim: x-axis limits.
    - spec: time spec.
    - panel_mode: panel mode label.
    """
    sub = cov[(cov["primary_lexicon"] == language) & (cov["doc_kept"] == True)].copy()  # noqa: E712
    if sub.empty:
        sub = cov[cov["primary_lexicon"] == language].copy()
    if sub.empty:
        return
    agg = (
        sub.groupby("bin_start", as_index=False)
        .agg(n_docs=("author", "count"))
        .sort_values("bin_start")
    )
    agg["bin_dt"] = pd.to_datetime(agg["bin_start"])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(agg["bin_dt"], agg["n_docs"], width=5, alpha=0.75, color="#55a868")
    _add_ref_lines(ax, config)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_title(f"Kept author-documents per bin ({language}, {panel_mode} {spec})")
    ax.set_xlabel("bin_start")
    ax.set_ylabel("n author-docs (kept)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_axis_words(axis_path: Path, out_path: Path, lang: str, tag: str) -> None:
    """Function summary: horizontal bar chart of top +/- beta words.

    Parameters:
    - axis_path: axis words CSV.
    - out_path: PNG path.
    - lang: language code.
    - tag: panel_mode_spec label.
    """
    if not axis_path.is_file():
        return
    df = pd.read_csv(axis_path)
    if df.empty:
        return
    colors = df["sign"].map({"pos": "#2166ac", "neg": "#b2182b"}).fillna("#888888")
    fig, ax = plt.subplots(figsize=(8, max(6, len(df) * 0.2)))
    y = range(len(df))
    ax.barh(list(y), df["beta"], color=colors)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["word"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("beta (word weight)")
    ax.set_title(f"Wordfish author axis words ({lang}, {tag})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_theta_scatter(
    merged: pd.DataFrame,
    out_path: Path,
    lang: str,
) -> None:
    """Function summary: author-mean theta vs lexicon ideology scatter for one language.

    Parameters:
    - merged: rows with theta, net_ideology_mean, sem_axis_ideology_mean.
    - out_path: PNG path.
    - lang: language label.
    """
    if len(merged) < 5:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(merged["net_ideology_mean"], merged["theta"], alpha=0.35, s=10)
    axes[0].set_xlabel("mean net_ideology")
    axes[0].set_ylabel("mean theta")
    axes[0].set_title(f"{lang}: theta vs net_ideology")
    if "sem_axis_ideology_mean" in merged.columns:
        axes[1].scatter(
            merged["sem_axis_ideology_mean"], merged["theta"], alpha=0.35, s=10
        )
        axes[1].set_xlabel("mean sem_axis_ideology")
        axes[1].set_ylabel("mean theta")
        axes[1].set_title(f"{lang}: theta vs sem_axis")
    fig.suptitle(f"Author-mean validation ({lang}); within-fit positions only", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_panel_bundle(
    tab_dir: Path,
    fig_dir: Path,
    config: dict,
    wfa: dict,
    panel_mode: str,
    spec: str,
    xlim: Tuple[pd.Timestamp, pd.Timestamp],
    *,
    write_headline_aliases: bool = False,
) -> int:
    """Function summary: generate all figure types for one panel_mode × spec tag.

    Parameters:
    - tab_dir: tables directory.
    - fig_dir: figures directory.
    - config: study config.
    - wfa: wordfish_authors config.
    - panel_mode: full or balanced.
    - spec: week7, week3, window.
    - xlim: event window limits.
    - write_headline_aliases: copy week7 balanced outputs to unsuffixed headline names.

    Returns:
    - Count of PNG files written.
    """
    tag = f"{panel_mode}_{spec}"
    n_written = 0
    ext_path = _resolve_panel_path(tab_dir, panel_mode, spec, "extremity_panel")
    if ext_path is None:
        print(f"[plot_wordfish_authors] skip {tag}: no extremity panel", flush=True)
        return 0

    ext = pd.read_csv(ext_path)

    series_outputs = [
        ("extremity", f"authors_extremity_timeseries_{tag}.png"),
        ("extremity_z", f"authors_extremity_z_timeseries_{tag}.png"),
        ("change", f"authors_change_timeseries_{tag}.png"),
        ("change_z", f"authors_change_z_timeseries_{tag}.png"),
    ]
    headline_aliases = {
        "extremity": "authors_extremity_timeseries_by_language.png",
        "extremity_z": "authors_extremity_z_timeseries_by_language.png",
        "change": "authors_change_timeseries_by_language.png",
        "change_z": "authors_change_z_timeseries_by_language.png",
    }
    for col, fname in series_outputs:
        if col not in ext.columns:
            continue
        _plot_timeseries(
            ext,
            col,
            fig_dir / fname,
            config,
            f"Mean author {col} by language ({tag})",
            xlim,
        )
        n_written += 1
        if write_headline_aliases and col in headline_aliases:
            _plot_timeseries(
                ext,
                col,
                fig_dir / headline_aliases[col],
                config,
                f"Mean author {col} by language ({tag})",
                xlim,
            )
            n_written += 1

    disp_path = _resolve_panel_path(tab_dir, panel_mode, spec, "dispersion_panel")
    if disp_path is not None:
        disp = pd.read_csv(disp_path)
        for lang in wfa.get("languages", ["it", "en", "de"]):
            out = fig_dir / f"authors_dispersion_timeseries_{lang}_{tag}.png"
            _plot_dispersion_timeseries(disp, lang, out, config, xlim, spec, panel_mode)
            n_written += 1

    cov_path = _resolve_panel_path(tab_dir, panel_mode, spec, "coverage")
    if cov_path is not None:
        cov = pd.read_csv(cov_path)
        for lang in wfa.get("languages", ["it", "en", "de"]):
            out = fig_dir / f"authors_coverage_by_bin_{lang}_{tag}.png"
            _plot_coverage_by_bin(cov, lang, out, config, xlim, spec, panel_mode)
            n_written += 1

    for lang in wfa.get("languages", ["it", "en", "de"]):
        axis_p = tab_dir / f"wordfish_authors_axis_words_{lang}_{tag}.csv"
        if axis_p.is_file():
            out = fig_dir / f"authors_axis_words_{lang}_{tag}.png"
            _plot_axis_words(axis_p, out, lang, tag)
            if write_headline_aliases:
                _plot_axis_words(axis_p, fig_dir / f"authors_axis_words_{lang}.png", lang, tag)
            n_written += 1

    pos_path = _resolve_panel_path(tab_dir, panel_mode, spec, "positions")
    ideol_path = tab_dir / "wordfish_authors_theta_ideology.csv"
    if pos_path is not None and ideol_path.is_file():
        pos = pd.read_csv(pos_path)
        ideol = pd.read_csv(ideol_path)
        auth_theta = (
            pos.groupby(["author", "primary_lexicon"], as_index=False)["theta"]
            .mean()
        )
        # ideol CSV also has theta (balanced week7 author mean); suffix so we keep positions theta.
        merged_all = auth_theta.merge(
            ideol, on="author", how="inner", suffixes=("", "_ideol")
        )
        if "theta_ideol" in merged_all.columns:
            merged_all = merged_all.drop(columns=["theta_ideol"])
        for lang in wfa.get("languages", ["it", "en", "de"]):
            sub_pos = pos[pos["primary_lexicon"] == lang]
            if sub_pos.empty:
                continue
            sub = merged_all[merged_all["primary_lexicon"] == lang].copy()
            if sub.empty:
                sub = (
                    sub_pos.groupby("author", as_index=False)["theta"]
                    .mean()
                    .merge(ideol, on="author", how="inner", suffixes=("", "_ideol"))
                )
                if "theta_ideol" in sub.columns:
                    sub = sub.drop(columns=["theta_ideol"])
            if sub.empty:
                continue
            out = fig_dir / f"authors_theta_vs_lexicon_scatter_{lang}_{tag}.png"
            _plot_theta_scatter(sub, out, lang)
            n_written += 1
        if write_headline_aliases and not merged_all.empty:
            _plot_theta_scatter(merged_all, fig_dir / "authors_theta_vs_lexicon_scatter.png", "all")
            n_written += 1

    return n_written


def main() -> None:
    """Function summary: generate author Wordfish figures from prepared tables."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    wfa = load_wordfish_authors_v2_config(config)
    subdir = str(wfa.get("output_figures_subdir", "wordfish_authors_v2"))
    tab_dir = tables_subdir(config, subdir)
    fig_dir = figures_subdir(config, subdir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    xlim = _event_xlim(config)
    headline_spec = str(wfa.get("headline_spec", "week7"))
    headline_mode = str(wfa.get("headline_mode", "balanced"))

    specs_to_plot: Sequence[Tuple[str, str]] = []
    if args.spec or args.panel_mode:
        spec = args.spec or headline_spec
        mode = args.panel_mode or headline_mode
        specs_to_plot = [(mode, spec)]
    else:
        specs_to_plot = [
            (headline_mode, headline_spec),
            ("balanced", "week3"),
            ("full", headline_spec),
        ]

    total = 0
    for panel_mode, spec in specs_to_plot:
        is_headline = panel_mode == headline_mode and spec == headline_spec
        total += _plot_panel_bundle(
            tab_dir,
            fig_dir,
            config,
            wfa,
            panel_mode,
            spec,
            xlim,
            write_headline_aliases=is_headline,
        )

    print(f"[plot_wordfish_authors] wrote {total} figure(s) to {fig_dir}", flush=True)


if __name__ == "__main__":
    main()
