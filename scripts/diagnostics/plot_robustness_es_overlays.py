"""
Script summary:
Overlay event-study figures comparing baseline DiD estimates to robustness variants.

Functionality:
- Task A: pole_share baseline (all authors) vs fixed pre-ban author set.
- Task B: sem_axis_emotion baseline vs leakage-pruned cognition pole (sem_axis_emotion_pruned).
- Fixed ±0.3 rel-day horizontal dodge, ban/lift vertical markers, static subtitles with p-values.
- Read-only: never modifies robustness job tables; writes new PNGs only.

How to apply/run:
  .venv/bin/python scripts/diagnostics/plot_robustness_es_overlays.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/plot_robustness_es_overlays.py --task pole_share
  .venv/bin/python scripts/diagnostics/plot_robustness_es_overlays.py --task emotion_pruned
"""

from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DODGE_OFFSET = 0.2
BIN_DAYS = 3


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
from src.config_utils import figures_subdir, load_config, tables_subdir  # noqa: E402
from src.did.outputs import _prepare_event_study_plot_df  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for robustness overlay figures."""
    parser = argparse.ArgumentParser(description="Robustness event-study overlay figures.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--task",
        type=str,
        choices=("all", "pole_share", "emotion_pruned"),
        default="all",
        help="Which overlay figure to build.",
    )
    return parser.parse_args()


def _rel_period_markers(config: Dict[str, Any], bin_days: int = BIN_DAYS) -> Tuple[int, int, int]:
    """Function summary: rel-period positions for ban launch, Apr 25, and lift on binned x-axis.

    Parameters:
    - config: study YAML.
    - bin_days: calendar days per event-time period (3 for headline ES).

    Returns:
    - Tuple (period0, period25, period28) in rel_period units.
    """
    _, _, launch, lift = event_dates_from_config(config)
    launch_dt = datetime.strptime(launch, "%Y-%m-%d")
    day25 = (datetime.strptime("2023-04-25", "%Y-%m-%d") - launch_dt).days
    day28 = (datetime.strptime(lift, "%Y-%m-%d") - launch_dt).days
    bd = max(1, int(bin_days))
    return 0, day25 // bd, day28 // bd


def _apply_overlay_axes_style(ax: plt.Axes, *, xlabel: str = "Event Time") -> None:
    """Function summary: overlay axes style without the default ref-bin line at x=-0.5."""
    ax.axhline(0, color="black", linewidth=0.9, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Coefficient")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")


def _add_ban_markers(ax: plt.Axes, period0: int, period25: int, period28: int) -> None:
    """Function summary: draw vertical lines at ban onset, Apr 25, and lift.

    Parameters:
    - ax: matplotlib axes.
    - period0: rel-period 0 (solid).
    - period25: Apr 25 (dotted).
    - period28: lift date (dashed).

    Returns:
    - None; mutates ax in place.
    """
    ax.axvline(period0, color="black", linestyle="-", linewidth=0.9, zorder=1)
    ax.axvline(period25, color="0.45", linestyle=":", linewidth=0.9, zorder=1)
    ax.axvline(period28, color="0.45", linestyle="--", linewidth=0.9, zorder=1)


def _format_p(p: float) -> str:
    """Function summary: format p-value for subtitle display."""
    if not np.isfinite(p):
        return "p=NA"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def _load_es_csv(path: Path, rel_col: str = "rel_period") -> pd.DataFrame:
    """Function summary: load and normalize event-study coefficient CSV."""
    df = pd.read_csv(path)
    time_col = rel_col if rel_col in df.columns else "rel_day"
    if time_col not in df.columns:
        raise ValueError(f"{path} missing {rel_col} or rel_day")
    return df


def _sanitize_es_df(df: pd.DataFrame, *, max_abs_gamma: float = 1.0, max_se: float = 0.5) -> pd.DataFrame:
    """Function summary: drop degenerate event-study rows with nonsensical scale."""
    if df.empty:
        return df
    out = df.copy()
    gamma = pd.to_numeric(out.get("gamma"), errors="coerce")
    se = pd.to_numeric(out.get("se"), errors="coerce")
    ok = gamma.notna() & se.notna() & (gamma.abs() <= max_abs_gamma) & (se <= max_se)
    return out[ok]


def _plot_overlay(
    series: Sequence[Tuple[str, pd.DataFrame, float, str]],
    *,
    title: str,
    subtitle: str,
    out_path: Path,
    rel_col: str = "rel_period",
    xlabel: str = "Event time (3-day periods)",
    ylabel: str = "Coefficient",
    ban_markers: Optional[Tuple[int, int, int]] = None,
    legend_fontsize: float = 8.0,
    legend_bbox_y: float = -0.14,
) -> None:
    """Function summary: draw dodged event-study overlay with error bars.

    Parameters:
    - series: list of (label, es_df, x_offset, color).
    - title: figure title.
    - subtitle: figure subtitle (statics with p-values).
    - out_path: PNG destination.
    - rel_col: event-time column.
    - xlabel, ylabel: axis labels.

    Returns:
    - None; writes PNG.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    all_times: List[int] = []
    for _, es_df, _, _ in series:
        plot = _prepare_event_study_plot_df(_sanitize_es_df(es_df), rel_col, ref_time=-1)
        all_times.extend(plot["event_time"].astype(int).tolist())
    all_times = sorted(set(all_times))

    for label, es_df, offset, color in series:
        plot = _prepare_event_study_plot_df(_sanitize_es_df(es_df), rel_col, ref_time=-1)
        x = plot["event_time"].astype(float) + offset
        se = plot["se"].fillna(0)
        mask = se > 0
        ax.errorbar(
            x[mask],
            plot.loc[mask, "gamma"],
            yerr=1.96 * se[mask],
            fmt="none",
            ecolor=color,
            alpha=0.75,
            capsize=2.5,
            elinewidth=0.9,
            zorder=2,
        )
        ax.plot(
            x,
            plot["gamma"],
            linestyle="none",
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.2,
            label=label,
            zorder=3,
        )

    _apply_overlay_axes_style(ax, xlabel=xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(all_times)
    if all_times:
        yvals = []
        for _, es_df, _, _ in series:
            plot = _prepare_event_study_plot_df(_sanitize_es_df(es_df), rel_col, ref_time=-1)
            yvals.extend(plot["gamma"].tolist())
            se = plot["se"].fillna(0)
            yvals.extend((plot["gamma"] + 1.96 * se).tolist())
            yvals.extend((plot["gamma"] - 1.96 * se).tolist())
        if yvals:
            pad = 0.05 * (max(yvals) - min(yvals) if max(yvals) > min(yvals) else 0.1)
            ax.set_ylim(min(yvals) - pad, max(yvals) + pad)
    if ban_markers is not None:
        _add_ban_markers(ax, *ban_markers)
    ax.set_title(None)
    fig.subplots_adjust(top=0.78, bottom=0.22)
    fig.suptitle(title, fontsize=11, y=0.96)
    if subtitle:
        fig.text(0.5, 0.88, subtitle, ha="center", va="top", fontsize=9, color="0.25")
        fig.subplots_adjust(bottom=0.18)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, legend_bbox_y),
        ncol=2,
        frameon=False,
        fontsize=legend_fontsize,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _summary_row(
    df: pd.DataFrame,
    *,
    sample: Optional[str] = None,
    spec: str,
    strategy: str = "cross_country_all",
) -> Dict[str, Any]:
    """Function summary: extract one static summary row."""
    sub = df.copy()
    if "strategy_id" in sub.columns:
        sub = sub[sub["strategy_id"].astype(str) == strategy]
    if sample is not None and "sample" in sub.columns:
        sub = sub[sub["sample"].astype(str) == sample]
    if "spec" in sub.columns:
        sub = sub[sub["spec"].astype(str) == spec]
    if sub.empty:
        return {}
    row = sub.iloc[0]
    return {
        "beta": float(row.get("beta", np.nan)),
        "pvalue": float(row.get("pvalue", np.nan)),
    }


def _find_pruned_es_csv(did_root: Path) -> Path:
    """Function summary: locate sem_axis_emotion_pruned event-study CSV under did estimates."""
    candidates = list(did_root.rglob("*sem_axis_emotion_pruned*.csv"))
    preferred = [
        p
        for p in candidates
        if "event_study" in str(p)
        and "3d" in str(p)
        and "cross_country_all" in str(p)
        and p.name == "sem_axis_emotion_pruned.csv"
    ]
    if preferred:
        return preferred[0]
    es_paths = [p for p in candidates if "event_study" in str(p) and "cross_country_all" in str(p)]
    if es_paths:
        return es_paths[0]
    legacy = did_root / "did_coefficients_sem_axis_emotion_pruned.csv"
    if legacy.is_file():
        raise FileNotFoundError(
            f"Found static coefficients only at {legacy}; run did_event_study for sem_axis_emotion_pruned first."
        )
    raise FileNotFoundError("sem_axis_emotion_pruned event-study CSV not found under did/")


def plot_pole_share_overlay(config: Dict[str, Any]) -> str:
    """Function summary: build pole_share fixed-author overlay and return interpretation line.

    Parameters:
    - config: study YAML.

    Returns:
    - One-line confirms/kills interpretation string.
    """
    did_root = tables_subdir(config, "did")
    fixed_dir = did_root / "pole_share_fixed_authors"
    if not fixed_dir.is_dir():
        raise FileNotFoundError(f"Missing {fixed_dir}")

    es_all = _load_es_csv(fixed_dir / "event_study.csv")
    es_all = es_all[es_all["sample"].astype(str) == "all_authors"].copy()
    es_fixed = _load_es_csv(fixed_dir / "event_study.csv")
    es_fixed = es_fixed[es_fixed["sample"].astype(str) == "fixed_authors"].copy()

    summary = pd.read_csv(fixed_dir / "summary.csv")
    all_static = _summary_row(summary, sample="all_authors", spec="full_ban")
    fixed_static = _summary_row(summary, sample="fixed_authors", spec="full_ban")
    beta_a = float(all_static.get("beta", np.nan))
    p_a = float(all_static.get("pvalue", np.nan))
    beta_f = float(fixed_static.get("beta", np.nan))
    p_f = float(fixed_static.get("pvalue", np.nan))
    subtitle = (
        f"Full-ban statics: all authors β={beta_a:+.3f} ({_format_p(p_a)}); "
        f"pre-ban author set β={beta_f:+.3f} ({_format_p(p_f)})"
    )

    markers = _rel_period_markers(config)
    out = figures_subdir(config, "did") / "pole_share_fixed_authors" / "pole_share_es_overlay.png"
    _plot_overlay(
        [
            ("all authors", es_all, -DODGE_OFFSET, "#1d3557"),
            ("pre-ban author set", es_fixed, DODGE_OFFSET, "#e76f51"),
        ],
        title="pole_share event study (cross_country_all, 3-day bins, ref = −1)",
        subtitle=subtitle,
        out_path=out,
        ban_markers=markers,
    )

    if np.isfinite(beta_f) and beta_f >= 0.04:
        verdict = f"pole_share overlay: CONFIRMS baseline (fixed-author full-ban β={beta_f:+.3f})"
    else:
        verdict = f"pole_share overlay: KILLS baseline (fixed-author full-ban β={beta_f:+.3f})"
    print(verdict, flush=True)
    print(f"[plot_robustness_es_overlays] wrote {out}", flush=True)
    return verdict


def plot_emotion_pruned_overlay(config: Dict[str, Any]) -> str:
    """Function summary: build sem_axis_emotion pruned overlay and return interpretation line."""
    did_root = tables_subdir(config, "did")
    baseline_es = _load_es_csv(
        did_root
        / "estimates/semantic_axis/event_study/language_universe/3d/cross_country_all/sem_axis_emotion.csv"
    )
    pruned_es_path = _find_pruned_es_csv(did_root)
    pruned_es = _load_es_csv(pruned_es_path)

    base_coef = pd.read_csv(did_root / "did_coefficients_sem_axis_emotion.csv")
    pruned_coef_path = did_root / "did_coefficients_sem_axis_emotion_pruned.csv"
    if not pruned_coef_path.is_file():
        pruned_summary = did_root / "estimates/semantic_axis/coefficients/sem_axis_emotion_pruned.csv"
        if pruned_summary.is_file():
            pruned_coef = pd.read_csv(pruned_summary)
        else:
            raise FileNotFoundError("sem_axis_emotion_pruned static coefficients not found")
    else:
        pruned_coef = pd.read_csv(pruned_coef_path)

    base_static = base_coef[
        (base_coef["strategy_id"] == "cross_country_all") & (base_coef["spec"] == "early_ban_7d")
    ].iloc[0]
    pruned_static = pruned_coef[
        (pruned_coef["strategy_id"] == "cross_country_all") & (pruned_coef["spec"] == "early_ban_7d")
    ].iloc[0]

    beta_b = float(base_static["beta"])
    p_b = float(base_static["pvalue"])
    beta_p = float(pruned_static["beta"])
    p_p = float(pruned_static["pvalue"])
    subtitle = (
        f"Early-ban (7d) statics: baseline cognition pole β={beta_b:+.4f} ({_format_p(p_b)}); "
        f"leakage-pruned pole β={beta_p:+.4f} ({_format_p(p_p)})"
    )

    out = (
        figures_subdir(config, "did")
        / "semantic_axis/event_study/sem_axis_emotion_pruned_overlay.png"
    )
    pruned_label = (
        "leakage-pruned pole (dati, analisi, evidenza, prova, statistica, metodo, stima removed)"
    )
    _plot_overlay(
        [
            ("baseline cognition pole", baseline_es, -DODGE_OFFSET, "#1d3557"),
            (pruned_label, pruned_es, DODGE_OFFSET, "#e76f51"),
        ],
        title="sem_axis_emotion event study (cross_country_all, 3-day bins, ref = −1)",
        subtitle=subtitle,
        out_path=out,
        ban_markers=_rel_period_markers(config),
        legend_fontsize=7.5,
        legend_bbox_y=-0.22,
    )

    pruned_kills_leakage = np.isfinite(beta_p) and abs(beta_p) < 0.003 and (not np.isfinite(p_p) or p_p > 0.10)
    if pruned_kills_leakage:
        verdict = (
            f"sem_axis_emotion_pruned overlay: CONFIRMS leakage/attention story "
            f"(pruned early-ban β={beta_p:+.4f}, {_format_p(p_p)})"
        )
    elif np.isfinite(beta_p) and beta_p < -0.005 and np.isfinite(p_p) and p_p < 0.05:
        verdict = (
            f"sem_axis_emotion_pruned overlay: KILLS leakage story — genuine discourse "
            f"(pruned early-ban β={beta_p:+.4f}, {_format_p(p_p)})"
        )
    else:
        verdict = (
            f"sem_axis_emotion_pruned overlay: MIXED — pruned early-ban β={beta_p:+.4f} ({_format_p(p_p)})"
        )
    print(verdict, flush=True)
    print(f"[plot_robustness_es_overlays] wrote {out}", flush=True)
    return verdict


def main() -> None:
    """Function summary: generate requested robustness overlay figures."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    if args.task in ("all", "pole_share"):
        plot_pole_share_overlay(config)
    if args.task in ("all", "emotion_pruned"):
        plot_emotion_pruned_overlay(config)


if __name__ == "__main__":
    main()
