"""
Script summary:
Estimate and plot aggregated-panel DiD event studies with explicit figure bundles.

Functionality:
- topic_family: overlay_pooled (5 strategies), it_political, it_others (single-series).
- language: subreddit (TWFE on subreddit, 3d outcome binning) and hub_pooled (language_hub panel).
- language_universe: in_out_slice (in-tree vs out-of-tree, two-line overlay on slice panel).
- Writes CSVs under estimates/{family}/event_study/{panel}/{bundle}/{bin}d/{strategy}/.
- PNGs under figures/did/event_study/{panel}/{bundle}/{bin}d/{outcome}.png.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_did_aggregated_panels.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/prepare_did_subreddit_panel.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/analysis/did_aggregated_event_study.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/analysis/did_aggregated_event_study.py --config config/italy_polarization_setup.yaml --figures-only
  .venv/bin/python scripts/analysis/did_aggregated_event_study.py --bundle subreddit --panel-level language
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import numpy as np


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

from src.config_utils import figures_subdir, load_config  # noqa: E402
from src.did.aggregated import (  # noqa: E402
    AGGREGATED_BIN_DAYS,
    AGGREGATED_PANEL_LEVELS,
    AggregatedPanelKey,
    AggregatedPanels,
    build_aggregated_panels,
    outcomes_for_panel_level,
    rel_col_for_bin,
)
from src.did.estimate import estimate_event_study  # noqa: E402
from src.did.outcomes import OutcomeSpec  # noqa: E402
from src.did.panels import (  # noqa: E402
    load_subreddit_event_study_panel,
    load_subreddit_slice_event_study_panel,
    wordfish_authors_v2_available,
    wordfish_forum_v2_available,
)
from src.did.outcomes import (  # noqa: E402
    SEM_AXIS_IDEOLOGY_EXTREME_LEFT_COL,
    SEM_AXIS_IDEOLOGY_EXTREME_RIGHT_COL,
)
from src.did.paths import (  # noqa: E402
    aggregated_event_study_figure_path,
    aggregated_tail_shift_figure_path,
    did_aggregated_event_study_path,
)
from src.did.outputs import (  # noqa: E402
    EventStudySeries,
    plot_event_study,
    plot_event_study_overlay,
    plot_sem_axis_ideology_tail_shift_event_study,
)
from src.did.specs import (  # noqa: E402
    EVENT_WINDOW_DAYS_BY_BIN,
    StrategySpec,
    default_strategies,
    event_study_language_universe_slice_strategies,
    event_study_overlay_strategies,
    event_study_topic_family_it_others_strategy,
    event_study_topic_family_it_political_strategy,
    filter_strategy_sample,
    strategy_label,
)


@dataclass(frozen=True)
class AggregatedEsJob:
    """Function summary: one event-study figure bundle (panel level × estimand)."""

    panel_level: str
    bundle: str
    panel_source: str  # aggregated | subreddit | slice
    overlay: bool
    strategies_fn: Callable[[], Tuple[StrategySpec, ...]]


STALE_AGGREGATED_ES_REL_DIRS: Tuple[str, ...] = (
    "event_study/language/1d",
    "event_study/language/3d",
    "event_study/language_universe/1d",
    "event_study/language_universe/3d",
    "event_study/topic_family/1d",
    "event_study/topic_family/3d",
)


def _tail_shift_strategy_specs(job: AggregatedEsJob) -> List[Tuple[StrategySpec, str]]:
    """Function summary: primary TWFE strategy (and filename suffix) per bundle for tail-shift figures."""
    if job.panel_level == "language_universe" and job.bundle == "in_out_slice":
        return [
            (event_study_language_universe_slice_strategies()[0], "in_tree"),
            (event_study_language_universe_slice_strategies()[1], "out_tree"),
        ]
    key = (job.panel_level, job.bundle)
    by_key: Dict[Tuple[str, str], Tuple[str, str]] = {
        ("topic_family", "overlay_pooled"): ("cross_country_all", ""),
        ("topic_family", "it_political"): ("cross_country_it_political", ""),
        ("topic_family", "it_others"): ("cross_country_it_others", ""),
        ("language", "subreddit"): ("cross_country_all", ""),
        ("language", "hub_pooled"): ("cross_country_all", ""),
    }
    sid, suffix = by_key[key]
    by_id = {s.strategy_id: s for s in default_strategies()}
    return [(by_id[sid], suffix)]


def remove_stale_aggregated_event_study_dirs(fig_dir: Path) -> None:
    """Function summary: delete pre-bundle aggregated event-study PNG directories."""
    for rel in STALE_AGGREGATED_ES_REL_DIRS:
        path = fig_dir / rel
        if path.is_dir():
            shutil.rmtree(path)
            print(f"[did_aggregated_event_study] removed stale {rel}", flush=True)


AGGREGATED_ES_JOBS: Tuple[AggregatedEsJob, ...] = (
    AggregatedEsJob(
        "topic_family",
        "overlay_pooled",
        "aggregated",
        True,
        event_study_overlay_strategies,
    ),
    AggregatedEsJob(
        "topic_family",
        "it_political",
        "aggregated",
        False,
        lambda: (event_study_topic_family_it_political_strategy(),),
    ),
    AggregatedEsJob(
        "topic_family",
        "it_others",
        "aggregated",
        False,
        lambda: (event_study_topic_family_it_others_strategy(),),
    ),
    AggregatedEsJob(
        "language",
        "subreddit",
        "subreddit",
        True,
        event_study_overlay_strategies,
    ),
    AggregatedEsJob(
        "language",
        "hub_pooled",
        "aggregated",
        True,
        event_study_overlay_strategies,
    ),
    AggregatedEsJob(
        "language_universe",
        "in_out_slice",
        "slice",
        True,
        event_study_language_universe_slice_strategies,
    ),
)


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for aggregated event studies."""
    parser = argparse.ArgumentParser(description="Aggregated-panel event studies.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--outcome", type=str, default=None, help="Single outcome_id.")
    parser.add_argument("--panel-level", type=str, default=None, help="topic_family|language|language_universe")
    parser.add_argument("--bundle", type=str, default=None, help="Figure bundle slug (e.g. subreddit, hub_pooled).")
    parser.add_argument("--bin-days", type=int, default=None, choices=(1, 3))
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--figures-only", action="store_true", help="Replot from saved CSVs.")
    parser.add_argument("--refresh-panels", action="store_true", help="Rebuild aggregated panels first.")
    return parser.parse_args()


def _resolve_column(panel: pd.DataFrame, spec: OutcomeSpec) -> Optional[str]:
    """Function summary: return outcome column if present."""
    if spec.family == "wordfish_forum_v2":
        v2 = f"{spec.column}_v2"
        if v2 in panel.columns:
            return v2
    if spec.column in panel.columns:
        return spec.column
    alt = spec.column.replace("_mean", "")
    if alt in panel.columns:
        return alt
    return None


def _skip_outcome(spec: OutcomeSpec, config: Dict[str, Any]) -> bool:
    """Function summary: skip v2 outcomes when panels unavailable."""
    if spec.family == "wordfish_forum_v2" and not wordfish_forum_v2_available(config):
        return True
    if spec.family == "wordfish_author_v2" and not wordfish_authors_v2_available(config):
        return True
    return False


def _estimation_panel(
    config: Dict[str, Any],
    panels: AggregatedPanels,
    job: AggregatedEsJob,
    bin_days: int,
) -> pd.DataFrame:
    """Function summary: load the panel rows used for TWFE for this bundle."""
    if job.panel_source == "aggregated":
        return panels.get(AggregatedPanelKey(job.panel_level, bin_days))
    if job.panel_source == "subreddit":
        return load_subreddit_event_study_panel(config, bin_days)
    if job.panel_source == "slice":
        return load_subreddit_slice_event_study_panel(config, bin_days)
    raise ValueError(f"unknown panel_source {job.panel_source}")


def _entity_time_cols(job: AggregatedEsJob, panel: pd.DataFrame) -> Tuple[str, str]:
    """Function summary: TWFE cluster entity and time column names for this bundle."""
    if job.panel_source == "subreddit":
        return "subreddit", "time_id"
    if job.panel_source == "slice":
        return "entity_id", "time_id"
    if job.panel_level == "language" and job.bundle == "hub_pooled":
        col = "language_hub" if "language_hub" in panel.columns else "entity_id"
        return col, "time_id"
    return "entity_id", "time_id"


def _filter_panel_for_strategy(
    panel: pd.DataFrame,
    strat: StrategySpec,
    window: int,
    job: AggregatedEsJob,
) -> pd.DataFrame:
    """Function summary: apply strategy sample filter on the estimation panel."""
    return filter_strategy_sample(panel, strat, window_days=window)


def _estimate_event_study_bundle(
    panel: pd.DataFrame,
    spec: OutcomeSpec,
    bin_days: int,
    y_col: str,
    strategies: Sequence[StrategySpec],
    *,
    job: AggregatedEsJob,
) -> List[EventStudySeries]:
    """Function summary: run event study per strategy on the bundle's estimation panel."""
    window = EVENT_WINDOW_DAYS_BY_BIN[int(bin_days)]
    rel_col = rel_col_for_bin(bin_days)
    entity_col, time_col = _entity_time_cols(job, panel)
    series: List[EventStudySeries] = []
    for strat in strategies:
        work = _filter_panel_for_strategy(panel, strat, window, job)
        if work.empty or y_col not in work.columns:
            continue
        _, es_df = estimate_event_study(
            work,
            y_col,
            rel_col=rel_col,
            window=window,
            entity_col=entity_col,
            time_col=time_col,
        )
        if es_df.empty:
            continue
        if not _event_study_series_usable(es_df):
            print(
                f"[did_aggregated_event_study] skip degenerate ES {spec.outcome_id} "
                f"{strat.strategy_id} {job.panel_level}/{job.bundle}",
                flush=True,
            )
            continue
        es_out = es_df.copy()
        es_out["strategy_id"] = strat.strategy_id
        if rel_col not in es_out.columns:
            es_out[rel_col] = es_out["rel_day"]
        series.append(
            EventStudySeries(
                label=strategy_label(strat.strategy_id, short=True),
                es_df=es_out,
                rel_col=rel_col,
            )
        )
    return series


def _event_study_series_usable(es_df: pd.DataFrame, min_finite_se: int = 2) -> bool:
    """Function summary: True when event-study table has enough identified coefficients."""
    if es_df.empty or "se" not in es_df.columns:
        return False
    finite = es_df["se"].apply(lambda v: np.isfinite(v) and float(v) > 1e-12)
    return int(finite.sum()) >= min_finite_se


def _write_cell(
    config: Dict[str, Any],
    fig_dir: Path,
    job: AggregatedEsJob,
    bin_days: int,
    spec: OutcomeSpec,
    series: List[EventStudySeries],
    write_figures: bool,
) -> None:
    """Function summary: persist CSVs per strategy and optional PNG for one bundle."""
    for s in series:
        sid = s.es_df["strategy_id"].iloc[0] if "strategy_id" in s.es_df.columns else "unknown"
        path = did_aggregated_event_study_path(
            config,
            spec.family,
            job.panel_level,
            job.bundle,
            bin_days,
            str(sid),
            spec.outcome_id,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        s.es_df.to_csv(path, index=False)
    if not write_figures or not series:
        return
    out = aggregated_event_study_figure_path(
        fig_dir, job.panel_level, job.bundle, bin_days, spec.outcome_id
    )
    if job.overlay or len(series) > 1:
        plot_event_study_overlay(series, spec.outcome_id, out)
    else:
        plot_event_study(series[0].es_df, spec.outcome_id, out, rel_col=series[0].rel_col)


def _jobs_for_filters(
    panel_filter: Optional[str],
    bundle_filter: Optional[str],
) -> Tuple[AggregatedEsJob, ...]:
    """Function summary: subset AGGREGATED_ES_JOBS by CLI filters."""
    jobs = AGGREGATED_ES_JOBS
    if panel_filter:
        jobs = tuple(j for j in jobs if j.panel_level == panel_filter)
    if bundle_filter:
        jobs = tuple(j for j in jobs if j.bundle == bundle_filter)
    return jobs


def run_estimation(
    config: Dict[str, Any],
    panels: AggregatedPanels,
    *,
    single_outcome: Optional[str],
    panel_filter: Optional[str],
    bundle_filter: Optional[str],
    bin_filter: Optional[int],
    write_figures: bool,
) -> Dict[str, int]:
    """Function summary: main loop; returns counts of PNGs and CSVs written."""
    fig_dir = figures_subdir(config, "did")
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_png = 0
    n_csv = 0
    jobs = _jobs_for_filters(panel_filter, bundle_filter)
    bins = [bin_filter] if bin_filter else list(AGGREGATED_BIN_DAYS)

    for job in jobs:
        specs = outcomes_for_panel_level(job.panel_level)
        if single_outcome:
            specs = tuple(o for o in specs if o.outcome_id == single_outcome)
        for bin_days in bins:
            try:
                panel = _estimation_panel(config, panels, job, bin_days)
            except FileNotFoundError as exc:
                print(
                    f"[did_aggregated_event_study] skip {job.panel_level}/{job.bundle} "
                    f"{bin_days}d: {exc}",
                    flush=True,
                )
                continue
            if panel.empty:
                print(
                    f"[did_aggregated_event_study] skip empty {job.panel_level}/"
                    f"{job.bundle} {bin_days}d",
                    flush=True,
                )
                continue
            strategies = job.strategies_fn()
            for oc in specs:
                if _skip_outcome(oc, config):
                    continue
                y_col = _resolve_column(panel, oc)
                if y_col is None:
                    continue
                series = _estimate_event_study_bundle(
                    panel, oc, bin_days, y_col, strategies, job=job
                )
                if not series:
                    print(
                        f"[did_aggregated_event_study] no ES {oc.outcome_id} "
                        f"{job.panel_level}/{job.bundle} {bin_days}d",
                        flush=True,
                    )
                    continue
                _write_cell(config, fig_dir, job, bin_days, oc, series, write_figures)
                n_csv += len(series)
                if write_figures:
                    n_png += 1
                print(
                    f"[did_aggregated_event_study] {oc.outcome_id} "
                    f"{job.panel_level}/{job.bundle} {bin_days}d ({len(series)} strategies)",
                    flush=True,
                )
    return {"png": n_png, "csv": n_csv}


def run_tail_shift_figures(
    config: Dict[str, Any],
    panels: AggregatedPanels,
    *,
    panel_filter: Optional[str],
    bundle_filter: Optional[str],
    bin_filter: Optional[int],
    write_figures: bool,
) -> int:
    """Function summary: dual-tail (p10/p90) ideology event studies for each bundle × bin."""
    if not write_figures:
        return 0
    fig_dir = figures_subdir(config, "did")
    jobs = _jobs_for_filters(panel_filter, bundle_filter)
    bins = [bin_filter] if bin_filter else list(AGGREGATED_BIN_DAYS)
    n_png = 0
    for job in jobs:
        for bin_days in bins:
            try:
                panel = _estimation_panel(config, panels, job, bin_days)
            except FileNotFoundError:
                continue
            if panel.empty:
                continue
            if SEM_AXIS_IDEOLOGY_EXTREME_LEFT_COL not in panel.columns or (
                SEM_AXIS_IDEOLOGY_EXTREME_RIGHT_COL not in panel.columns
            ):
                print(
                    f"[did_aggregated_event_study] skip tail_shift "
                    f"{job.panel_level}/{job.bundle} {bin_days}d: missing p10/p90 cols",
                    flush=True,
                )
                continue
            window = EVENT_WINDOW_DAYS_BY_BIN[int(bin_days)]
            rel_col = rel_col_for_bin(bin_days)
            entity_col, time_col = _entity_time_cols(job, panel)
            for strat, suffix in _tail_shift_strategy_specs(job):
                work = _filter_panel_for_strategy(panel, strat, window, job)
                if work.empty:
                    continue
                series_pair: List[EventStudySeries] = []
                for y_col, oid in (
                    (SEM_AXIS_IDEOLOGY_EXTREME_LEFT_COL, "sem_axis_ideology_extreme_left"),
                    (SEM_AXIS_IDEOLOGY_EXTREME_RIGHT_COL, "sem_axis_ideology_extreme_right"),
                ):
                    if y_col not in work.columns:
                        continue
                    _, es_df = estimate_event_study(
                        work,
                        y_col,
                        rel_col=rel_col,
                        window=window,
                        entity_col=entity_col,
                        time_col=time_col,
                    )
                    if es_df.empty:
                        continue
                    es_out = es_df.copy()
                    es_out["strategy_id"] = strat.strategy_id
                    path = did_aggregated_event_study_path(
                        config,
                        "semantic_axis",
                        job.panel_level,
                        job.bundle,
                        bin_days,
                        strat.strategy_id,
                        oid,
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    es_out.to_csv(path, index=False)
                    label = (
                        "extreme-LEFT tail share (sem axis, <p10)"
                        if "left" in y_col
                        else "extreme-RIGHT tail share (sem axis, >p90)"
                    )
                    series_pair.append(
                        EventStudySeries(label=label, es_df=es_out, rel_col=rel_col)
                    )
                if len(series_pair) != 2:
                    continue
                out = aggregated_tail_shift_figure_path(
                    fig_dir,
                    job.panel_level,
                    job.bundle,
                    bin_days,
                    suffix=suffix,
                )
                plot_sem_axis_ideology_tail_shift_event_study(
                    series_pair[0],
                    series_pair[1],
                    out,
                    bin_days=bin_days,
                )
                n_png += 1
                print(
                    f"[did_aggregated_event_study] tail_shift "
                    f"{job.panel_level}/{job.bundle}/{bin_days}d"
                    f"{('/' + suffix) if suffix else ''}",
                    flush=True,
                )
    return n_png


def run_figures_only(
    config: Dict[str, Any],
    *,
    single_outcome: Optional[str],
    panel_filter: Optional[str],
    bundle_filter: Optional[str],
    bin_filter: Optional[int],
) -> int:
    """Function summary: rebuild PNGs from saved strategy CSVs under bundle paths."""
    fig_dir = figures_subdir(config, "did")
    n_png = 0
    jobs = _jobs_for_filters(panel_filter, bundle_filter)
    bins = [bin_filter] if bin_filter else list(AGGREGATED_BIN_DAYS)
    for job in jobs:
        specs = outcomes_for_panel_level(job.panel_level)
        if single_outcome:
            specs = tuple(o for o in specs if o.outcome_id == single_outcome)
        strategies = job.strategies_fn()
        for bin_days in bins:
            rel_col = rel_col_for_bin(bin_days)
            for oc in specs:
                series: List[EventStudySeries] = []
                for strat in strategies:
                    p = did_aggregated_event_study_path(
                        config,
                        oc.family,
                        job.panel_level,
                        job.bundle,
                        bin_days,
                        strat.strategy_id,
                        oc.outcome_id,
                    )
                    if not p.is_file():
                        continue
                    es_df = pd.read_csv(p)
                    if es_df.empty:
                        continue
                    series.append(
                        EventStudySeries(
                            label=strategy_label(strat.strategy_id, short=True),
                            es_df=es_df,
                            rel_col=rel_col,
                        )
                    )
                if not series:
                    continue
                out = aggregated_event_study_figure_path(
                    fig_dir, job.panel_level, job.bundle, bin_days, oc.outcome_id
                )
                if job.overlay or len(series) > 1:
                    plot_event_study_overlay(series, oc.outcome_id, out)
                else:
                    plot_event_study(series[0].es_df, oc.outcome_id, out, rel_col=rel_col)
                n_png += 1
    return n_png


def main() -> None:
    """Function summary: CLI entry."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    if args.refresh_panels and not args.figures_only:
        from scripts.diagnostics import prepare_did_aggregated_panels as prep

        prep.main()
    if args.figures_only:
        n = run_figures_only(
            config,
            single_outcome=args.outcome,
            panel_filter=args.panel_level,
            bundle_filter=args.bundle,
            bin_filter=args.bin_days,
        )
        print(f"[did_aggregated_event_study] rebuilt {n} PNGs", flush=True)
        return
    panels = build_aggregated_panels(config)
    write_figures = not args.no_figures
    counts = run_estimation(
        config,
        panels,
        single_outcome=args.outcome,
        panel_filter=args.panel_level,
        bundle_filter=args.bundle,
        bin_filter=args.bin_days,
        write_figures=write_figures,
    )
    n_tail = run_tail_shift_figures(
        config,
        panels,
        panel_filter=args.panel_level,
        bundle_filter=args.bundle,
        bin_filter=args.bin_days,
        write_figures=write_figures,
    )
    if write_figures:
        remove_stale_aggregated_event_study_dirs(figures_subdir(config, "did"))
    print(
        f"[did_aggregated_event_study] wrote {counts['png']} PNGs, {counts['csv']} CSVs, "
        f"{n_tail} tail-shift PNGs",
        flush=True,
    )


if __name__ == "__main__":
    main()
