"""
Script summary:
Plot semantic-axis descriptives from prepared CSV tables.

Functionality:
- Organized output under results/figures/.../semantic_axis/bins_{1,3,7}d/{topic_family,topic,language,language_universe}/.
- Per level: timeseries (ideology, emotion, aggression, share_unscored), pole_shares_abs, pole_percentiles.
- bins_{bd}d/audit/ (bin completeness, Italy circumvention IT) and lexical_country/ (from did/).
- _global/: seed OOV, score histograms, forum scatter vs lexical ideology.

How to apply/run:
  .venv/bin/python scripts/diagnostics/plot_semantic_axis_descriptives.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

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

from src.config_utils import (  # noqa: E402
    figures_subdir,
    load_config,
    plot_reference_dates_calendar_utc,
    tables_subdir,
)
from src.did.paths import resolve_panel_path  # noqa: E402
from src.embeddings import ALL_AXIS_NAMES, AXIS_SCORE_COLUMNS  # noqa: E402

SCORE_COLS: tuple[str, ...] = tuple(AXIS_SCORE_COLUMNS[axis] for axis in ALL_AXIS_NAMES)
STALE_POLE_SUFFIXES = ("tau50", "tau75", "tau25")

_DATE_AXIS_LABEL = "Date (UTC, daily period start)"
PANEL_BIN_DAYS: tuple[int, ...] = (1, 3, 7)
# level_key -> (panel slug, group column, build series_id from lang+universe)
PLOT_LEVELS: dict[str, tuple[str, str, bool]] = {
    "topic_family": ("by_topic_family", "topic_family", False),
    "topic": ("by_topic", "topic", False),
    "language": ("by_language", "primary_lexicon", False),
    "language_universe": ("by_language_universe", "series_id", True),
}
LEVEL_DISPLAY: dict[str, str] = {
    "topic_family": "topic family",
    "topic": "topic",
    "language": "language",
    "language_universe": "language × political universe",
}
_AXIS_POLE_CONFIG: dict[str, dict[str, str]] = {
    "ideology": {"high": "right", "low": "left", "title": "Ideology"},
    "emotion": {"high": "pos", "low": "neg", "title": "Emotion (affect vs cognition)"},
    "aggression": {"high": "pos", "low": "neg", "title": "Aggression (incivility vs neutral)"},
    "economic": {"high": "pos", "low": "neg", "title": "Economic (market vs equality)"},
    "cultural": {"high": "pos", "low": "neg", "title": "Cultural (traditional vs progressive)"},
    "nationalism": {"high": "pos", "low": "neg", "title": "Nationalism (nationalist vs cosmopolitan)"},
    "anti_establishment": {
        "high": "pos",
        "low": "neg",
        "title": "Anti-establishment (anti-institution vs pro-institution)",
    },
}
_TIMESERIES_AXIS_TITLES: dict[str, str] = {
    "ideology": "ideology",
    "emotion": "emotion",
    "aggression": "aggression",
    "economic": "economic",
    "cultural": "cultural",
    "nationalism": "nationalism",
    "anti_establishment": "anti_establishment",
}
TIMESERIES_METRICS: tuple[tuple[str, str], ...] = tuple(
    [(f"sem_axis_{axis}_mean", _TIMESERIES_AXIS_TITLES[axis]) for axis in ALL_AXIS_NAMES]
    + [("share_unscored", "share_unscored")]
)
_PANEL_METRIC_YLABELS: dict[str, str] = {
    "sem_axis_ideology_mean": (
        "Mean comment score on ideology axis\n"
        "(projection toward right pole minus left pole; higher = more right-leaning)"
    ),
    "sem_axis_emotion_mean": (
        "Mean comment score on emotion axis\n"
        "(affect pole minus cognition pole; higher = more emotional/affective)"
    ),
    "sem_axis_aggression_mean": (
        "Mean comment score on aggression axis\n"
        "(insult/incivility pole minus neutral; higher = more aggressive)"
    ),
    "sem_axis_economic_mean": (
        "Mean comment score on economic axis\n"
        "(market pole minus equality pole; higher = more market-oriented)"
    ),
    "sem_axis_cultural_mean": (
        "Mean comment score on cultural axis\n"
        "(traditional pole minus progressive pole; higher = more traditional)"
    ),
    "sem_axis_nationalism_mean": (
        "Mean comment score on nationalism axis\n"
        "(nationalist pole minus cosmopolitan pole; higher = more nationalist)"
    ),
    "sem_axis_anti_establishment_mean": (
        "Mean comment score on anti-establishment axis\n"
        "(anti-institution pole minus pro-institution; higher = more anti-establishment)"
    ),
    "share_unscored": (
        "Share of comments without a semantic-axis vector\n"
        "(has_sem_axis=0; use instead of saturated FastText coverage)"
    ),
}
_COMMENT_SCORE_YLABELS: dict[str, str] = {
    AXIS_SCORE_COLUMNS[axis]: f"Density of comments by {axis.replace('_', ' ')} axis score"
    for axis in ALL_AXIS_NAMES
}
_POLE_SHARE_YLABEL = "Share of scored comments in pole bucket (0–1)"
_HISTOGRAM_XLABEL = "Comment-level semantic axis score"


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot semantic-axis descriptives.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def add_ref_lines(ax: plt.Axes, config: dict) -> None:
    """Function summary: draw vertical ban reference dates."""
    for dt in plot_reference_dates_calendar_utc(config):
        ax.axvline(pd.Timestamp(dt), color="red", linestyle=":", linewidth=1.0, alpha=0.8)


def _date_axis_label(date_col: str, *, bin_days: int | None = None) -> str:
    """Function summary: map panel date column name to a readable x-axis label."""
    if date_col == "period_start" and bin_days is not None and int(bin_days) > 1:
        return f"Period start (UTC, launch-aligned {int(bin_days)}-day bins)"
    if date_col == "period_start":
        return _DATE_AXIS_LABEL
    if date_col == "date_utc":
        return "Date (UTC)"
    return date_col


def _level_out_dir(fig_root: Path, bin_days: int, level_key: str, chart_type: str) -> Path:
    """Function summary: path for one chart type under bins_{bd}d/{level}/."""
    return fig_root / f"bins_{int(bin_days)}d" / level_key / chart_type


def _prepare_panel_for_level(panel: pd.DataFrame, use_series_id: bool) -> pd.DataFrame:
    """Function summary: add series_id column for language_universe level when needed."""
    if not use_series_id:
        return panel
    out = panel.copy()
    if not {"primary_lexicon", "universe_slice"}.issubset(out.columns):
        return out
    out["series_id"] = (
        out["primary_lexicon"].astype(str) + " (" + out["universe_slice"].astype(str) + ")"
    )
    return out


def _cleanup_legacy_flat_figures(fig_root: Path) -> None:
    """Function summary: remove obsolete flat PNGs at semantic_axis/ root after folder migration."""
    _remove_stale_pole_figures(fig_root)
    for path in fig_root.glob("*.png"):
        if path.is_file():
            path.unlink()


def _panel_metric_ylabel(metric: str) -> str:
    """Function summary: map aggregated panel metric column to a readable y-axis label."""
    return _PANEL_METRIC_YLABELS.get(metric, metric)


def _comment_score_ylabel(col: str) -> str:
    """Function summary: map comment-level score column to a readable y-axis label."""
    return _COMMENT_SCORE_YLABELS.get(col, col)


def _pole_cutoff_label(tau: str) -> str:
    """Function summary: turn tau25-style suffix into a readable cutoff (e.g. ±0.25)."""
    try:
        value = int(tau.replace("tau", "")) / 100.0
    except ValueError:
        return tau
    return f"±{value:g}"


def _date_col(panel: pd.DataFrame) -> str:
    """Function summary: prefer period_start for panel time axis."""
    if "period_start" in panel.columns:
        return "period_start"
    if "date_utc" in panel.columns:
        return "date_utc"
    raise KeyError("panel missing period_start or date_utc")


def _remove_stale_pole_figures(fig_root: Path) -> None:
    """Function summary: delete obsolete pole-share PNGs no longer produced."""
    stale_names = [
        *(f"{axis}_pole_shares_{suffix}_by_topic_family.png" for axis in _AXIS_POLE_CONFIG for suffix in STALE_POLE_SUFFIXES),
        *(f"{axis}_pole_shares_{suffix}_by_family.png" for axis in _AXIS_POLE_CONFIG for suffix in STALE_POLE_SUFFIXES),
        "coverage_axis_timeseries_by_family.png",
        "ideology_pole_shares_tau25_by_family.png",
    ]
    for name in stale_names:
        path = fig_root / name
        if path.is_file():
            path.unlink()


def _plot_group_timeseries(
    panel: pd.DataFrame,
    group_col: str,
    metric: str,
    title: str,
    out_path: Path,
    config: dict,
    *,
    bin_days: int | None = None,
) -> None:
    """Function summary: daily group means for one semantic-axis panel metric."""
    if metric not in panel.columns or group_col not in panel.columns:
        return
    date_col = _date_col(panel)
    if bin_days is None and "bin_days" in panel.columns:
        try:
            bin_days = int(panel["bin_days"].dropna().iloc[0])
        except (IndexError, ValueError, TypeError):
            bin_days = None
    fam = panel.groupby([date_col, group_col], as_index=False)[metric].mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    for group_val, grp in fam.groupby(group_col):
        grp = grp.sort_values(date_col)
        ax.plot(pd.to_datetime(grp[date_col]), grp[metric], label=str(group_val), alpha=0.85)
    add_ref_lines(ax, config)
    ax.set_title(title)
    ax.set_xlabel(_date_axis_label(date_col, bin_days=bin_days))
    ax.set_ylabel(_panel_metric_ylabel(metric))
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _resolve_pole_share_columns(panel: pd.DataFrame, axis: str) -> tuple[str, str, str] | None:
    """Function summary: pick high/low share columns (abs_symmetric or legacy tau25)."""
    cfg = _AXIS_POLE_CONFIG[axis]
    high_label = cfg["high"]
    low_label = cfg["low"]
    for suffix in ("abs", "tau25"):
        high_col = f"sem_axis_{axis}_share_{high_label}_{suffix}"
        low_col = f"sem_axis_{axis}_share_{low_label}_{suffix}"
        if high_col in panel.columns and low_col in panel.columns:
            return high_col, low_col, suffix
    return None


def _plot_axis_pole_shares(
    panel: pd.DataFrame,
    axis: str,
    group_col: str,
    config: dict,
    out_path: Path,
    *,
    bin_days: int | None = None,
    level_label: str = "",
) -> None:
    """Function summary: plot high/low pole shares (lexicon-calibrated abs buckets) to out_path."""
    resolved = _resolve_pole_share_columns(panel, axis)
    if resolved is None or group_col not in panel.columns:
        return
    right_col, left_col, suffix = resolved
    cfg = _AXIS_POLE_CONFIG[axis]
    date_col = _date_col(panel)
    if bin_days is None and "bin_days" in panel.columns:
        try:
            bin_days = int(panel["bin_days"].dropna().iloc[0])
        except (IndexError, ValueError, TypeError):
            bin_days = None
    fam = panel.groupby([date_col, group_col], as_index=False).agg(
        high=(right_col, "mean"),
        low=(left_col, "mean"),
    )
    title_suffix = "per-lexicon abs" if suffix == "abs" else suffix
    fig, ax = plt.subplots(figsize=(11, 5))
    for group_val, grp in fam.groupby(group_col):
        grp = grp.sort_values(date_col)
        x = pd.to_datetime(grp[date_col])
        ax.plot(x, grp["high"], label=f"{group_val} {cfg['high']}", alpha=0.85)
        ax.plot(x, grp["low"], linestyle="--", label=f"{group_val} {cfg['low']}", alpha=0.65)
    add_ref_lines(ax, config)
    by = level_label or group_col
    ax.set_title(f"{cfg['title']} pole shares ({title_suffix}) — {by}")
    ax.set_xlabel(_date_axis_label(date_col, bin_days=bin_days))
    ax.set_ylabel(_POLE_SHARE_YLABEL)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_percentile_pole_shares(
    panel: pd.DataFrame,
    axis: str,
    group_col: str,
    config: dict,
    out_path: Path,
    *,
    bin_days: int | None = None,
    level_label: str = "",
) -> None:
    """Function summary: plot above_p90 / below_p10 percentile pole shares when present."""
    cfg = _AXIS_POLE_CONFIG[axis]
    high_col = f"sem_axis_{axis}_share_{cfg['high']}_above_p90"
    low_col = f"sem_axis_{axis}_share_{cfg['low']}_below_p10"
    if high_col not in panel.columns or low_col not in panel.columns or group_col not in panel.columns:
        return
    date_col = _date_col(panel)
    if bin_days is None and "bin_days" in panel.columns:
        try:
            bin_days = int(panel["bin_days"].dropna().iloc[0])
        except (IndexError, ValueError, TypeError):
            bin_days = None
    fam = panel.groupby([date_col, group_col], as_index=False).agg(
        high=(high_col, "mean"),
        low=(low_col, "mean"),
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    for group_val, grp in fam.groupby(group_col):
        grp = grp.sort_values(date_col)
        x = pd.to_datetime(grp[date_col])
        ax.plot(x, grp["high"], label=f"{group_val} above p90", alpha=0.85)
        ax.plot(x, grp["low"], linestyle="--", label=f"{group_val} below p10", alpha=0.65)
    add_ref_lines(ax, config)
    by = level_label or group_col
    ax.set_title(f"{cfg['title']} percentile pole shares — {by}")
    ax.set_xlabel(_date_axis_label(date_col, bin_days=bin_days))
    ax.set_ylabel(_POLE_SHARE_YLABEL)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_seed_oov_bars(seed_path: Path, fig_root: Path) -> None:
    """Function summary: bar chart of seed in-vocab share by language and axis."""
    if not seed_path.is_file():
        return
    df = pd.read_csv(seed_path)
    if df.empty or "share_in_vocab" not in df.columns:
        return
    df["label"] = df["lang"].astype(str) + "/" + df["axis"].astype(str) + "/" + df["pole"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["label"], df["share_in_vocab"].astype(float), color="steelblue", alpha=0.85)
    ax.set_ylabel("Share of seed terms in FastText vocab")
    ax.set_title("Semantic-axis seed in-vocabulary coverage")
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    fig.tight_layout()
    fig.savefig(fig_root / "seed_in_vocab_coverage.png", dpi=150)
    plt.close(fig)


def _load_stratified_comment_sample(
    shard_paths: list[Path],
    target_families: list[str],
    per_family_cap: int = 8000,
    max_shards: int = 2000,
) -> pd.DataFrame | None:
    """Function summary: sample scored comments until each topic_family hits a cap.

    Parameters:
    - shard_paths: enriched Parquet shard paths.
    - target_families: topic_family values to include.
    - per_family_cap: max comments per family.
    - max_shards: stop after scanning this many shards.

    Returns:
    - DataFrame with score columns and topic_family, or None.
    """
    from scripts.features._enriched_shard_runner import read_parquet_shard_safe

    need = set(target_families)
    counts = {f: 0 for f in target_families}
    chunks: list[pd.DataFrame] = []
    cols = list(SCORE_COLS) + ["topic_family", "has_sem_axis"]

    for i, path in enumerate(shard_paths):
        if not need or i >= max_shards:
            break
        d = read_parquet_shard_safe(path)
        if d is None:
            continue
        use_cols = [c for c in cols if c in d.columns]
        if "sem_axis_ideology" not in use_cols or "topic_family" not in use_cols:
            continue
        scored = d[use_cols].copy()
        if "has_sem_axis" in scored.columns:
            scored = scored[scored["has_sem_axis"].astype(float) > 0]
        else:
            scored = scored[scored["sem_axis_ideology"].notna()]
        if scored.empty:
            continue
        for fam in scored["topic_family"].astype(str).unique():
            if fam not in counts or counts[fam] >= per_family_cap:
                if fam in need and counts[fam] >= per_family_cap:
                    need.discard(fam)
                continue
            remain = per_family_cap - counts[fam]
            fam_rows = scored[scored["topic_family"].astype(str) == fam]
            if len(fam_rows) > remain:
                fam_rows = fam_rows.sample(n=remain, random_state=42)
            counts[fam] += len(fam_rows)
            chunks.append(fam_rows)
            if counts[fam] >= per_family_cap:
                need.discard(fam)

    if not chunks:
        return None
    return pd.concat(chunks, ignore_index=True)


def _plot_score_distributions(
    scored: pd.DataFrame,
    fig_root: Path,
) -> None:
    """Function summary: 1×3 histograms of comment scores by topic_family for all axes."""
    hist_titles = {
        "sem_axis_ideology": "Ideology axis (comment scores)",
        "sem_axis_emotion": "Emotion axis (comment scores)",
        "sem_axis_aggression": "Aggression axis (comment scores)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, SCORE_COLS):
        if col not in scored.columns:
            ax.set_visible(False)
            continue
        for fam, grp in scored.groupby("topic_family"):
            ax.hist(
                grp[col].astype(float),
                bins=40,
                alpha=0.35,
                density=True,
                label=str(fam),
            )
        ax.set_title(hist_titles[col])
        ax.set_xlabel(_HISTOGRAM_XLABEL)
        ax.set_ylabel(_comment_score_ylabel(col))
        ax.legend(title="Topic family", fontsize=6)
    families = ", ".join(sorted(scored["topic_family"].astype(str).unique()))
    fig.suptitle(
        f"Semantic axis score distributions by topic family\n"
        f"(stratified sample; families: {families})"
    )
    fig.tight_layout()
    fig.savefig(fig_root / "score_distributions_by_family.png", dpi=150)
    plt.close(fig)


def _plot_level_bundle(
    panel: pd.DataFrame,
    level_key: str,
    group_col: str,
    bin_days: int,
    fig_root: Path,
    config: dict,
) -> None:
    """Function summary: timeseries + pole abs + pole percentiles for one panel level and bin size."""
    level_label = LEVEL_DISPLAY.get(level_key, level_key)
    bd = int(bin_days)
    for metric, short_name in TIMESERIES_METRICS:
        title = f"{_panel_metric_ylabel(metric).split(chr(10))[0]} — {level_label} ({bd}d bins)"
        _plot_group_timeseries(
            panel,
            group_col,
            metric,
            title,
            _level_out_dir(fig_root, bd, level_key, "timeseries") / f"{short_name}.png",
            config,
            bin_days=bd,
        )
    for axis in _AXIS_POLE_CONFIG:
        _plot_axis_pole_shares(
            panel,
            axis,
            group_col,
            config,
            _level_out_dir(fig_root, bd, level_key, "pole_shares_abs") / f"{axis}.png",
            bin_days=bd,
            level_label=level_label,
        )
        _plot_percentile_pole_shares(
            panel,
            axis,
            group_col,
            config,
            _level_out_dir(fig_root, bd, level_key, "pole_percentiles") / f"{axis}.png",
            bin_days=bd,
            level_label=level_label,
        )


def _plot_bin_completeness_and_volume(
    panel: pd.DataFrame,
    group_col: str,
    out_path: Path,
    config: dict,
    bin_days: int,
) -> None:
    """Function summary: bin calendar coverage and comment volume by group (DiD audit)."""
    need = {group_col, "n_days_in_bin", "is_partial_bin", "n_comments"}
    if not need.issubset(panel.columns):
        return
    date_col = _date_col(panel)
    bd = int(bin_days)
    fig, (ax_days, ax_vol) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for group_val, grp in panel.groupby(group_col, sort=True):
        grp = grp.sort_values(date_col)
        x = pd.to_datetime(grp[date_col])
        label = str(group_val)
        ax_days.plot(x, grp["n_days_in_bin"].astype(float), label=label, marker="o", alpha=0.85)
        partial = grp["is_partial_bin"].astype(bool)
        if partial.any():
            ax_days.scatter(
                x[partial],
                grp.loc[partial, "n_days_in_bin"].astype(float),
                facecolors="none",
                edgecolors="red",
                linewidths=1.5,
                s=50,
                zorder=5,
            )
        ax_vol.plot(x, grp["n_comments"].astype(float), label=label, marker="o", alpha=0.85)
    for ax in (ax_days, ax_vol):
        add_ref_lines(ax, config)
    ax_days.axhline(float(bd), color="0.5", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_days.set_ylabel(f"Calendar days in bin (max {bd})")
    ax_days.set_title(f"{bd}d panel bins: calendar coverage (red ring = partial bin)")
    ax_vol.set_ylabel("Comments in bin")
    ax_vol.set_title(f"{bd}d panel bins: comment volume")
    ax_vol.set_xlabel(_date_axis_label(date_col, bin_days=bd))
    ax_days.legend(loc="upper left", fontsize=7, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_italy_intensity(panel: pd.DataFrame, out_path: Path, config: dict, bin_days: int) -> None:
    """Function summary: Italy broadcast VPN/Tor on period_start (one national series)."""
    if "vpn_interest_it" not in panel.columns or "period_start" not in panel.columns:
        return
    date_col = "period_start"
    bd = int(bin_days)
    work = panel.drop_duplicates(subset=[date_col]).sort_values(date_col)
    if work.empty:
        return
    x = pd.to_datetime(work[date_col])
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.plot(x, work["vpn_interest_it"].astype(float), color="C0", marker="o", label="vpn_interest_it")
    ax1.set_ylabel("Italy VPN Trends index (0–100, within-IT scale)")
    ax1.set_title(f"Italy circumvention proxies on {bd}d bins (broadcast to all semantic arms)")
    tor_col = "log1p_tor_bridge_users_it"
    if tor_col in work.columns and work[tor_col].notna().any():
        ax2 = ax1.twinx()
        ax2.plot(
            x,
            work[tor_col].astype(float),
            color="C1",
            marker="s",
            alpha=0.75,
            label=tor_col,
        )
        ax2.set_ylabel("log1p Tor bridge users (Italy)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    else:
        ax1.legend(loc="upper left", fontsize=8)
    add_ref_lines(ax1, config)
    ax1.set_xlabel(_date_axis_label(date_col, bin_days=bd))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_country_lexical(did_panel: pd.DataFrame, out_path: Path, config: dict, bin_days: int) -> None:
    """Function summary: lexical country-panel outcomes on launch-aligned bins."""
    if did_panel.empty or "country_panel" not in did_panel.columns:
        return
    date_col = _date_col(did_panel)
    bd = int(bin_days)
    metrics = (
        ("net_ideology_mean", "Lexical net ideology (comment-weighted mean)"),
        ("ai_style_rate_100w_mean", "AI style rate per 100 words (mean)"),
    )
    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 3.5 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    partial_col = "is_partial_bin" if "is_partial_bin" in did_panel.columns else None
    for ax, (metric, ylabel) in zip(axes, metrics):
        if metric not in did_panel.columns:
            ax.set_visible(False)
            continue
        for panel_name, grp in did_panel.groupby("country_panel", sort=True):
            grp = grp.sort_values(date_col)
            x = pd.to_datetime(grp[date_col])
            ax.plot(x, grp[metric].astype(float), label=str(panel_name), marker="o", alpha=0.85)
            if partial_col:
                partial = grp[partial_col].astype(bool)
                if partial.any():
                    ax.scatter(
                        x[partial],
                        grp.loc[partial, metric].astype(float),
                        facecolors="none",
                        edgecolors="red",
                        linewidths=1.2,
                        s=40,
                        zorder=5,
                    )
        add_ref_lines(ax, config)
        ax.set_ylabel(ylabel)
    axes[0].set_title(f"Lexical outcomes by country panel ({bd}d launch-aligned bins)")
    axes[-1].set_xlabel(_date_axis_label(date_col, bin_days=bd))
    axes[0].legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_bin_audit_and_lexical(
    fig_root: Path,
    tables: Path,
    did_country_path: Path,
    bin_days: int,
    config: dict,
) -> None:
    """Function summary: per-bin audit figures and lexical country panel from did tables."""
    bd = int(bin_days)
    bin_root = fig_root / f"bins_{bd}d"
    family_path = tables / f"semantic_axis_panel_by_topic_family_{bd}d.csv"
    if family_path.is_file():
        family = pd.read_csv(family_path)
        _plot_bin_completeness_and_volume(
            family,
            "topic_family",
            bin_root / "audit" / "panel_bin_completeness_and_volume.png",
            config,
            bin_days=bd,
        )
        _plot_italy_intensity(
            family,
            bin_root / "audit" / "italy_circumvention_it.png",
            config,
            bin_days=bd,
        )
    if did_country_path.is_file():
        _plot_country_lexical(
            pd.read_csv(did_country_path),
            bin_root / "lexical_country" / "net_ideology_and_ai_style.png",
            config,
            bin_days=bd,
        )


def main() -> None:
    """Function summary: CLI entry for semantic-axis figures."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    tables = tables_subdir(config, "semantic_axis")
    fig_root = figures_subdir(config, "semantic_axis")
    fig_root.mkdir(parents=True, exist_ok=True)
    global_dir = fig_root / "_global"
    global_dir.mkdir(parents=True, exist_ok=True)

    probe = tables / "semantic_axis_panel_by_topic_family_1d.csv"
    if not probe.is_file():
        raise FileNotFoundError(
            f"Run prepare_semantic_axis_descriptives.py first: missing {probe.name}"
        )

    _cleanup_legacy_flat_figures(fig_root)

    n_plotted = 0
    for bin_days in PANEL_BIN_DAYS:
        _plot_bin_audit_and_lexical(
            fig_root,
            tables,
            resolve_panel_path(config, "country", f"did_country_panel_{int(bin_days)}d.csv"),
            int(bin_days),
            config,
        )
        for level_key, (slug, group_col, use_series_id) in PLOT_LEVELS.items():
            panel_path = tables / f"semantic_axis_panel_{slug}_{int(bin_days)}d.csv"
            if not panel_path.is_file():
                print(
                    f"[plot_semantic_axis_descriptives] skip missing {panel_path.name}",
                    flush=True,
                )
                continue
            panel = _prepare_panel_for_level(pd.read_csv(panel_path), use_series_id)
            if group_col not in panel.columns:
                print(
                    f"[plot_semantic_axis_descriptives] skip {panel_path.name}: no {group_col}",
                    flush=True,
                )
                continue
            _plot_level_bundle(panel, level_key, group_col, int(bin_days), fig_root, config)
            n_plotted += 1
            print(
                f"[plot_semantic_axis_descriptives] bins_{bin_days}d/{level_key}",
                flush=True,
            )

    forum_panel_path = tables / "semantic_axis_panel_by_forum_1d.csv"
    if not forum_panel_path.is_file():
        legacy_forum = tables / "semantic_axis_panel.csv"
        if legacy_forum.is_file():
            forum_panel_path = legacy_forum
    if forum_panel_path.is_file():
        forum_panel = pd.read_csv(forum_panel_path)
        date_col = _date_col(forum_panel)
        if {"sem_axis_ideology_mean", "net_ideology_mean"}.issubset(forum_panel.columns):
            sub_period = forum_panel.groupby(["subreddit", date_col], as_index=False).agg(
                sem_axis_ideology_mean=("sem_axis_ideology_mean", "mean"),
                net_ideology_mean=("net_ideology_mean", "mean"),
            )
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(
                sub_period["net_ideology_mean"],
                sub_period["sem_axis_ideology_mean"],
                alpha=0.35,
                s=12,
            )
            ax.set_xlabel(
                "Lexical net ideology (subreddit × day mean)\n"
                "(right-minus-left lexicon hits per word; higher = more right-leaning)"
            )
            ax.set_ylabel(
                "Embedding ideology score (subreddit × day mean)\n"
                "(FastText axis projection; higher = more right-leaning)"
            )
            orient_note = ""
            orient_path = tables / "ideology_axis_orientation_report.csv"
            if orient_path.is_file():
                orient = pd.read_csv(orient_path)
                pooled = orient[orient["lang"] == "_pooled"]
                if not pooled.empty:
                    flag = str(pooled.iloc[0].get("orientation_flag", ""))
                    r = pooled.iloc[0].get("corr_ideology_comment_pearson", float("nan"))
                    orient_note = f"\n(pooled r={r:.2f}, {flag}; see orientation report)"
            ax.set_title(f"Semantic axis vs lexicon ideology{orient_note}")
            fig.tight_layout()
            fig.savefig(global_dir / "axis_vs_lexicon_scatter.png", dpi=150)
            plt.close(fig)

    _plot_seed_oov_bars(tables / "semantic_axis_seed_coverage.csv", global_dir)

    family_1d = tables / "semantic_axis_panel_by_topic_family_1d.csv"
    if family_1d.is_file():
        target_families = sorted(
            pd.read_csv(family_1d)["topic_family"].dropna().astype(str).unique()
        )
        shard_glob = sorted(
            (PROJECT_ROOT / config["paths"]["interim_dir"] / "cleaned_monthly_chunks").glob(
                "*/*.parquet"
            )
        )
        if shard_glob and target_families:
            scored = _load_stratified_comment_sample(shard_glob, target_families)
            if scored is not None and not scored.empty:
                _plot_score_distributions(scored, global_dir)

    print(
        f"[plot_semantic_axis_descriptives] wrote figures under {fig_root} "
        f"({n_plotted} level×bin bundles)",
        flush=True,
    )


if __name__ == "__main__":
    main()
