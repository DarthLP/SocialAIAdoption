"""
Script summary:
Headline multi-strategy event-study estimation and plots (5 outcomes × 3 strategies).

Functionality:
- TWFE event study with k ∈ {-3,-2,0,1,2,3,4} (k=-1 omitted as reference).
- Strategies: cross_country_all, cross_country_it_political, cross_country_it_others.
- Writes flat tables did/event_study_{outcome}.csv and figures did/event_study/{outcome}.png.

How to apply/run:
  .venv/bin/python scripts/analysis/did_event_study_plots.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List

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

from src.config_utils import figures_subdir, load_config, tables_subdir  # noqa: E402
from src.did.estimate import estimate_event_study  # noqa: E402
from src.did.outcomes import (  # noqa: E402
    HEADLINE_EVENT_STUDY_OUTCOMES,
    OUTCOME_REGISTRY,
    OutcomeSpec,
    outcome_label,
)
from src.did.panels import build_analysis_panels  # noqa: E402
from src.did.specs import StrategySpec, filter_strategy_sample, strategy_label  # noqa: E402

HEADLINE_ES_STRATEGIES = (
    StrategySpec("cross_country_all"),
    StrategySpec("cross_country_it_political", treated_family="it_political"),
    StrategySpec("cross_country_it_others", treated_family="it_others"),
)

ES_WINDOW = 4
ES_COLORS = ("#1d3557", "#457b9d", "#e76f51")


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for headline event-study plots."""
    parser = argparse.ArgumentParser(description="Headline event-study plots.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def _outcome_spec(outcome_id: str) -> OutcomeSpec:
    """Function summary: lookup OutcomeSpec by outcome_id."""
    for oc in OUTCOME_REGISTRY:
        if oc.outcome_id == outcome_id:
            return oc
    raise KeyError(outcome_id)


def _resolve_column(panel: pd.DataFrame, oc: OutcomeSpec) -> str | None:
    """Function summary: return outcome column name if present."""
    if oc.column in panel.columns:
        return oc.column
    alt = oc.column.replace("_mean", "")
    return alt if alt in panel.columns else None


def _panel_for_outcome(panels, oc: OutcomeSpec) -> pd.DataFrame:
    """Function summary: subreddit panel for headline outcomes."""
    if oc.outcome_id.startswith("wf_"):
        return panels.sub_v1
    if oc.outcome_id.startswith("wf2_"):
        return panels.sub_v2
    return panels.sub_v1


def _run_event_study_rows(
    config: Dict[str, Any],
    outcome_id: str,
) -> pd.DataFrame:
    """Function summary: estimate event-study coefficients for all headline strategies.

    Returns:
    - Long DataFrame with outcome_id, strategy_id, k, beta, se, ci_low, ci_high.
    """
    oc = _outcome_spec(outcome_id)
    panels = build_analysis_panels(config)
    panel = _panel_for_outcome(panels, oc)
    y_col = _resolve_column(panel, oc)
    if panel.empty or y_col is None:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for strat in HEADLINE_ES_STRATEGIES:
        sample = filter_strategy_sample(panel, strat, window_days=ES_WINDOW)
        if sample.empty:
            continue
        _, es_df = estimate_event_study(
            sample,
            y_col,
            ref_day=-1,
            window=ES_WINDOW,
            entity_col="subreddit" if "subreddit" in sample.columns else "entity_id",
            time_col="time_id",
        )
        if es_df.empty:
            continue
        rel_col = "rel_day" if "rel_day" in es_df.columns else "event_time"
        for _, r in es_df.iterrows():
            k = int(r[rel_col])
            if k < -ES_WINDOW or k > ES_WINDOW or k == -1:
                continue
            rows.append(
                {
                    "outcome_id": outcome_id,
                    "strategy_id": strat.strategy_id,
                    "k": k,
                    "beta": float(r["gamma"]),
                    "se": float(r["se"]),
                    "ci_low": float(r["ci_low"]),
                    "ci_high": float(r["ci_high"]),
                }
            )
    return pd.DataFrame(rows)


def _plot_headline_event_study(es_long: pd.DataFrame, outcome_id: str, out_path: Path) -> None:
    """Function summary: one PNG with three strategy series."""
    if es_long.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for idx, (sid, grp) in enumerate(es_long.groupby("strategy_id")):
        grp = grp.sort_values("k")
        color = ES_COLORS[idx % len(ES_COLORS)]
        ax.errorbar(
            grp["k"],
            grp["beta"],
            yerr=1.96 * grp["se"],
            fmt="o-",
            color=color,
            capsize=3,
            label=strategy_label(sid, short=True),
        )
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Event time (days relative to ban)")
    ax.set_ylabel("Coefficient (treat × event time)")
    ax.set_title(outcome_label(outcome_id, short=True))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Function summary: write headline event-study tables and figures."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    did_dir = tables_subdir(config, "did")
    fig_dir = figures_subdir(config, "did") / "event_study"
    did_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    for oid in HEADLINE_EVENT_STUDY_OUTCOMES:
        es_long = _run_event_study_rows(config, oid)
        if es_long.empty:
            print(f"[did_event_study_plots] skip {oid}: no coefficients", flush=True)
            continue
        table_path = did_dir / f"event_study_{oid}.csv"
        es_long.to_csv(table_path, index=False)
        _plot_headline_event_study(es_long, oid, fig_dir / f"{oid}.png")
        print(f"[did_event_study_plots] {oid} -> {table_path.name}", flush=True)

    print(f"[did_event_study_plots] figures under {fig_dir}", flush=True)


if __name__ == "__main__":
    main()
