"""
Script summary:
This script renders the figures for the within-user pre/post style shift
analysis. It consumes the per-user shift CSVs produced by
`scripts/user_week/analyze_user_pre_post_shift.py` and the user-week panel produced by
`scripts/user_week/prepare_user_week_style_panel.py`, and writes a small set of figures
that communicate the headline result and its sanity checks.

Outputs (per cohort and composite under `paths.figures_dir/user_week/<cohort>/<polarization|style>/`):
- dist_std_delta_composite.png: histogram of weekly std_delta_composite with
  vertical lines at +/-1 and +/-2 SD; tail shares in legend.
- dist_t_user_pooled_composite.png: same idea on the precision-aware pooled t.
- weekly_vs_pooled_scatter.png: per-user std_delta_weekly vs t_user_pooled,
  color by post-period word count (sparse vs dense users).
- spaghetti_sample.png: weekly composite trajectories for a deterministic
  sample of panel users with ban reference markers from `plot_reference_dates_utc`.
- mirror_top_movers.png: top-10 surge and top-10 drop users by
  t_user_pooled_composite shown side-by-side.
- components_grid.png: small multiples of std_delta_weekly distributions for
  each composite component plus the composite itself.

How to apply/run:
- Default (renders both strict and loose cohorts when present):
  `.venv/bin/python scripts/user_week/plot_user_pre_post_shift.py --config config/archive/ai_adoption_political_forums_setup.yaml`
- One cohort only:
  `.venv/bin/python scripts/user_week/plot_user_pre_post_shift.py --config config/archive/ai_adoption_political_forums_setup.yaml --cohort strict`
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import importlib.util
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
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

from src.config_utils import infer_user_week_input_mode, load_config, plot_reference_dates_calendar_utc, user_week_composites


def composite_file_slug(composite_name: str) -> str:
    """Function summary: short filesystem slug for per-composite output files."""
    if composite_name.startswith("polarization"):
        return "polarization"
    if "style" in composite_name:
        return "style"
    return composite_name.replace("_user_week", "").replace("_composite", "")


def composites_for_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Function summary: resolve composite list from YAML or Italy defaults."""
    comps = user_week_composites(config)
    if comps:
        return comps
    if infer_user_week_input_mode(config) == "enriched_shards":
        return [
            {
                "name": "polarization_composite_user_week",
                "components": [
                    ("extremity", 1),
                    ("net_ideology", 1),
                    ("other_side_salience_rate_100w", 1),
                    ("aggression_rate_100w", 1),
                ],
            },
            {
                "name": "ai_style_composite_user_week",
                "components": [
                    ("ai_style_rate_100w", 1),
                    ("semicolon_rate_100w", 1),
                    ("em_dash_rate_100w", 1),
                    ("hedging_phrase_rate_100w", 1),
                ],
            },
        ]
    return [
        {
            "name": "ai_likeness_user_week",
            "components": [
                ("ai_word_rate_100w", 1),
                ("formality_balance_100w", 1),
                ("assistant_tone_rate_100w", 1),
                ("list_structure_intensity", 1),
                ("contraction_rate_100w", -1),
            ],
        }
    ]


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI options including cohort selection and figure-tuning controls."""
    parser = argparse.ArgumentParser(description="Plot user-level pre/post style shift figures.")
    parser.add_argument("--config", type=str, default="config/archive/ai_adoption_political_forums_setup.yaml")
    parser.add_argument(
        "--cohort",
        type=str,
        default="both",
        choices=["both", "strict", "loose"],
        help="Cohort(s) to plot. Files must already exist from analyze_user_pre_post_shift.py.",
    )
    parser.add_argument(
        "--spaghetti_n",
        type=int,
        default=50,
        help="Number of panel users to sample for the spaghetti plot.",
    )
    parser.add_argument(
        "--top_movers_n",
        type=int,
        default=10,
        help="Top-N surge and top-N drop users in the mirror plot.",
    )
    parser.add_argument("--seed", type=int, default=20240502)
    return parser.parse_args()


def add_ban_reference_markers(ax: plt.Axes, config: Dict[str, Any]) -> None:
    """Function summary: draw red dotted vertical lines at plot_reference_dates_utc from the study config."""
    for d in plot_reference_dates_calendar_utc(config):
        ax.axvline(x=d, color="red", linestyle=":", linewidth=1.2)


def format_month_start_axis(ax: plt.Axes) -> None:
    """Function summary: align x-axis ticks to the first day of each month for date-based plots."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def figures_root(config: Dict[str, Any]) -> Path:
    """Function summary: resolve the user-week figures root and ensure it exists."""
    root = Path(config["paths"]["figures_dir"]) / "user_week"
    root.mkdir(parents=True, exist_ok=True)
    return root


def cohort_paths(config: Dict[str, Any], cohort: str, composite_slug: str) -> Dict[str, Path]:
    """Function summary: resolve all input/output paths needed to render figures for one cohort and composite."""
    tables = Path(config["paths"]["tables_dir"]) / "user_week"
    figs = figures_root(config) / cohort / composite_slug
    figs.mkdir(parents=True, exist_ok=True)
    per_user = tables / f"shift_per_user_{cohort}_{composite_slug}.csv"
    if not per_user.exists():
        per_user = tables / f"shift_per_user_{cohort}.csv"
    return {
        "per_user_csv": per_user,
        "panel_parquet": tables / "user_week_panel.parquet",
        "figures_dir": figs,
        "scales_json": tables / f"composite_zscale_pre_{cohort}_{composite_slug}.json",
    }


def tail_share_label(values: pd.Series, threshold: float, direction: str) -> str:
    """Function summary: format a tail-share legend entry for vertical-line annotations on histograms."""
    v = values.dropna()
    if v.empty:
        return f"{direction}{threshold}: n/a"
    if direction == ">":
        share = float((v > threshold).mean())
    else:
        share = float((v < threshold).mean())
    return f"share {direction}{threshold}: {share:.1%}"


def plot_histogram_with_thresholds(
    values: pd.Series,
    title: str,
    out_path: Path,
    thresholds: tuple[float, float] = (1.0, 2.0),
    bins: int = 60,
) -> None:
    """Function summary: render one histogram with +/-1 and +/-2 SD/t reference lines and tail-share legend."""
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return
    plt.figure(figsize=(9, 5))
    plt.hist(v.values, bins=bins, color="#4C72B0", alpha=0.85, edgecolor="white")
    legend_entries: List[str] = []
    for thr, color in zip(thresholds, ("#f0a040", "#d43f3f")):
        for direction in (">", "<"):
            anchor = thr if direction == ">" else -thr
            plt.axvline(x=anchor, color=color, linestyle="--", linewidth=1.0)
            legend_entries.append(tail_share_label(v, threshold=anchor, direction=direction))
    plt.title(title)
    plt.xlabel("Standardized shift")
    plt.ylabel("Number of users")
    plt.text(
        0.99,
        0.99,
        "\n".join(legend_entries),
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="#aaaaaa"),
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_weekly_vs_pooled_scatter(per_user: pd.DataFrame, composite_name: str, out_path: Path) -> None:
    """Function summary: scatter weekly std_delta vs pooled t per user, colored by post words (sparse vs dense)."""
    weekly_col = f"std_delta_weekly_{composite_name}"
    pooled_col = f"t_user_pooled_{composite_name}"
    if weekly_col not in per_user.columns or pooled_col not in per_user.columns:
        return
    color_col = f"post_total_words_{composite_name}"
    if color_col not in per_user.columns:
        color_col = "post_words_total_good"
    df = per_user[[weekly_col, pooled_col, color_col]].dropna()
    if df.empty:
        return
    plt.figure(figsize=(7.5, 6))
    sc = plt.scatter(
        df[weekly_col].values,
        df[pooled_col].values,
        c=np.log10(np.clip(df[color_col].astype(float).values, 1.0, None)),
        cmap="viridis",
        s=12,
        alpha=0.75,
    )
    plt.colorbar(sc, label=f"log10 {color_col}")
    plt.axhline(0, color="#888888", linewidth=0.6)
    plt.axvline(0, color="#888888", linewidth=0.6)
    lim = float(max(abs(df[weekly_col].abs().quantile(0.99)), abs(df[pooled_col].abs().quantile(0.99)), 1.0))
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.plot([-lim, lim], [-lim, lim], color="#aaaaaa", linestyle="--", linewidth=0.8)
    same_sign_share = float((np.sign(df[weekly_col]) == np.sign(df[pooled_col])).mean())
    corr = float(df[weekly_col].corr(df[pooled_col]))
    plt.title(
        f"Weekly std_delta vs Pooled t (composite)\n"
        f"sign-agree {same_sign_share:.1%}, corr {corr:.2f}, n={len(df)}"
    )
    plt.xlabel(f"Weekly view: std_delta_weekly_{composite_name}")
    plt.ylabel(f"Pooled view: t_user_pooled_{composite_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def add_composite_to_panel_with_scales(
    panel: pd.DataFrame,
    scales: Dict[str, Dict[str, float]],
    composite_name: str,
    composite_components: List[tuple[str, int]],
) -> pd.DataFrame:
    """Function summary: apply frozen-pre z-scales to compute composite per user-week (mirrors analysis script logic)."""
    if panel.empty or not scales:
        out = panel.copy()
        out[composite_name] = float("nan")
        return out
    out = panel.copy()
    composite = pd.Series(0.0, index=out.index)
    has_any = False
    for component, sign in composite_components:
        if component not in out.columns or component not in scales:
            continue
        sd = float(scales[component].get("sd", float("nan")))
        mean = float(scales[component].get("mean", 0.0))
        if not np.isfinite(sd) or sd == 0:
            continue
        v = pd.to_numeric(out[component], errors="coerce").fillna(0.0)
        composite = composite + int(sign) * (v - mean) / sd
        has_any = True
    out[composite_name] = composite if has_any else float("nan")
    return out


def plot_spaghetti_sample(
    panel: pd.DataFrame,
    per_user: pd.DataFrame,
    sample_n: int,
    seed: int,
    out_path: Path,
    config: Dict[str, Any],
    composite_name: str,
) -> None:
    """Function summary: render a random sample of panel users' weekly composite trajectories with ban reference markers."""
    if panel.empty or per_user.empty:
        return
    panel_authors = per_user["author"].astype(str).unique()
    if panel_authors.size == 0:
        return
    rng = np.random.default_rng(seed)
    n = int(min(sample_n, panel_authors.size))
    chosen = rng.choice(panel_authors, size=n, replace=False)
    sub = panel[panel["author"].astype(str).isin(chosen)].copy()
    if sub.empty:
        return
    sub["date"] = pd.to_datetime(sub["iso_week_start"], errors="coerce")
    sub = sub.dropna(subset=["date"]).sort_values(["author", "date"])

    plt.figure(figsize=(11, 6))
    for author, grp in sub.groupby("author"):
        plt.plot(grp["date"].values, grp[composite_name].values, color="#4C72B0", alpha=0.20, linewidth=0.8)

    pooled = sub.groupby("date")[composite_name].mean().reset_index()
    plt.plot(pooled["date"].values, pooled[composite_name].values, color="black", linewidth=2.0, label="Sample mean")

    add_ban_reference_markers(plt.gca(), config)
    format_month_start_axis(plt.gca())
    plt.title(f"Composite weekly trajectories (random sample n={n})")
    plt.xlabel("Date (UTC)")
    plt.ylabel(f"{composite_name} (frozen-pre z scaled)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_mirror_top_movers(
    panel: pd.DataFrame,
    per_user: pd.DataFrame,
    top_n: int,
    out_path: Path,
    config: Dict[str, Any],
    composite_name: str,
) -> None:
    """Function summary: render top-N surge and top-N drop users (by pooled t) side by side as paired weekly trajectories."""
    pooled_col = f"t_user_pooled_{composite_name}"
    if pooled_col not in per_user.columns:
        return
    valid = per_user[["author", pooled_col]].dropna()
    if valid.empty:
        return
    surge = valid.sort_values(pooled_col, ascending=False).head(top_n)["author"].astype(str).tolist()
    drop = valid.sort_values(pooled_col, ascending=True).head(top_n)["author"].astype(str).tolist()
    if not surge and not drop:
        return

    sub = panel[panel["author"].astype(str).isin(surge + drop)].copy()
    sub["date"] = pd.to_datetime(sub["iso_week_start"], errors="coerce")
    sub = sub.dropna(subset=["date"]).sort_values(["author", "date"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, group, title, color in (
        (axes[0], surge, f"Top {top_n} surge users (highest pooled t)", "#2ca02c"),
        (axes[1], drop, f"Top {top_n} drop users (lowest pooled t)", "#d62728"),
    ):
        for author in group:
            g = sub[sub["author"].astype(str) == author]
            if g.empty:
                continue
            ax.plot(g["date"].values, g[composite_name].values, color=color, alpha=0.7, linewidth=1.0, label=author)
        add_ban_reference_markers(ax, config)
        format_month_start_axis(ax)
        ax.set_title(title)
        ax.set_xlabel("Date (UTC)")
        ax.set_ylabel(f"{composite_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_components_grid(
    per_user: pd.DataFrame,
    out_path: Path,
    composite_name: str,
    component_feats: List[str],
) -> None:
    """Function summary: small multiples of weekly std_delta distributions for each composite component plus the composite itself."""
    feats = list(component_feats) + [composite_name]
    feats = [f for f in feats if f"std_delta_weekly_{f}" in per_user.columns]
    if not feats:
        return
    n = len(feats)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.8 * rows), squeeze=False)
    for i, feat in enumerate(feats):
        ax = axes[i // cols][i % cols]
        col = f"std_delta_weekly_{feat}"
        v = pd.to_numeric(per_user[col], errors="coerce").dropna()
        if v.empty:
            ax.set_visible(False)
            continue
        ax.hist(v.values, bins=40, color="#4C72B0", alpha=0.85, edgecolor="white")
        ax.axvline(0, color="#888888", linewidth=0.6)
        ax.axvline(1, color="#f0a040", linestyle="--", linewidth=0.7)
        ax.axvline(-1, color="#f0a040", linestyle="--", linewidth=0.7)
        ax.axvline(2, color="#d43f3f", linestyle="--", linewidth=0.7)
        ax.axvline(-2, color="#d43f3f", linestyle="--", linewidth=0.7)
        share_pos = float((v > 1).mean())
        share_neg = float((v < -1).mean())
        ax.set_title(f"{feat}\nshare>+1: {share_pos:.1%}  share<-1: {share_neg:.1%}", fontsize=9)
        ax.set_xlabel("std_delta_weekly")
    for j in range(len(feats), rows * cols):
        axes[j // cols][j % cols].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def render_one_cohort(
    config: Dict[str, Any],
    cohort: str,
    composite_name: str,
    composite_slug: str,
    composite_components: List[tuple[str, int]],
    component_feats: List[str],
    args: argparse.Namespace,
) -> None:
    """Function summary: render the full figure set for one cohort and composite if analysis outputs exist."""
    paths = cohort_paths(config, cohort=cohort, composite_slug=composite_slug)
    csv_path = paths["per_user_csv"]
    if not csv_path.exists():
        print(f"[plot_user_pre_post_shift] cohort={cohort} skip: per-user CSV not found at {csv_path}", flush=True)
        return
    per_user = pd.read_csv(csv_path)
    if per_user.empty:
        print(f"[plot_user_pre_post_shift] cohort={cohort} skip: per-user CSV is empty", flush=True)
        return

    print(f"[plot_user_pre_post_shift] cohort={cohort} n_users={len(per_user)}", flush=True)
    figures_dir = paths["figures_dir"]

    weekly_col = f"std_delta_weekly_{composite_name}"
    pooled_col = f"t_user_pooled_{composite_name}"

    if weekly_col in per_user.columns:
        plot_histogram_with_thresholds(
            per_user[weekly_col],
            title=f"Distribution of weekly std_delta_{composite_name} ({cohort}, n={len(per_user)})",
            out_path=figures_dir / "dist_std_delta_composite.png",
        )
    if pooled_col in per_user.columns:
        plot_histogram_with_thresholds(
            per_user[pooled_col],
            title=f"Distribution of pooled t_user_{composite_name} ({cohort}, n={len(per_user)})",
            out_path=figures_dir / "dist_t_user_pooled_composite.png",
        )

    plot_weekly_vs_pooled_scatter(per_user, composite_name=composite_name, out_path=figures_dir / "weekly_vs_pooled_scatter.png")
    plot_components_grid(
        per_user,
        out_path=figures_dir / "components_grid.png",
        composite_name=composite_name,
        component_feats=component_feats,
    )

    panel_path = paths["panel_parquet"]
    if not panel_path.exists():
        print(f"[plot_user_pre_post_shift] cohort={cohort} skip: panel parquet missing at {panel_path}", flush=True)
        return
    panel = pd.read_parquet(panel_path)
    if panel.empty:
        return

    scales_path = paths["scales_json"]
    scales: Dict[str, Dict[str, float]] = {}
    if scales_path.exists():
        try:
            import json

            scales = json.loads(scales_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[plot_user_pre_post_shift] cohort={cohort} composite_zscale_pre load failed: {exc}", flush=True)
            scales = {}
    panel_with_composite = add_composite_to_panel_with_scales(
        panel, scales, composite_name=composite_name, composite_components=composite_components
    )

    plot_spaghetti_sample(
        panel_with_composite,
        per_user=per_user,
        sample_n=int(args.spaghetti_n),
        seed=int(args.seed),
        out_path=figures_dir / "spaghetti_sample.png",
        config=config,
        composite_name=composite_name,
    )
    plot_mirror_top_movers(
        panel_with_composite,
        per_user=per_user,
        top_n=int(args.top_movers_n),
        out_path=figures_dir / "mirror_top_movers.png",
        config=config,
        composite_name=composite_name,
    )
    print(f"[plot_user_pre_post_shift] cohort={cohort} figures written to {figures_dir}", flush=True)


def main() -> None:
    """Function summary: render figures for the requested cohort(s) using existing analysis outputs and the user-week panel."""
    args = parse_args()
    config = load_config(args.config)
    composites = composites_for_config(config)
    cohorts: List[str] = ["strict", "loose"] if args.cohort == "both" else [args.cohort]
    for comp in composites:
        composite_name = str(comp["name"])
        composite_components = [(str(f), int(s)) for f, s in comp["components"]]
        component_feats = [f for f, _ in composite_components]
        slug = composite_file_slug(composite_name)
        for cohort in cohorts:
            render_one_cohort(
                config,
                cohort=cohort,
                composite_name=composite_name,
                composite_slug=slug,
                composite_components=composite_components,
                component_feats=component_feats,
                args=args,
            )


if __name__ == "__main__":
    main()
