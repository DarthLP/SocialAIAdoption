"""
Script summary:
Estimate Italy ChatGPT-ban DiD, event studies, and triple-differences on prepared panels.

Functionality:
- TWFE DiD (subreddit-day and author-bin panels) with multi-strategy summary table.
- Event-study dynamics, early-ban windows, wild-cluster bootstrap, permutation tests.
- Robustness: placebo 2023-03-16, symmetric window trims.
- Outcome families: lexical (00), semantic axis (01), forum/author Wordfish v1 and v2.
- Nested figure folders with human-readable strategy labels.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_did_subreddit_panel.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/analysis/did_event_study.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/analysis/did_event_study.py --config config/italy_polarization_setup.yaml --families lexical
  .venv/bin/python scripts/analysis/did_event_study.py --config config/italy_polarization_setup.yaml --outcome net_ideology --no-figures
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from src.config_utils import figures_subdir, load_config, tables_subdir  # noqa: E402
from src.did.estimate import estimate_event_study, run_strategy_twfe  # noqa: E402
from src.did.inference import permutation_test_p, wild_cluster_bootstrap_p  # noqa: E402
from src.did.outcomes import (  # noqa: E402
    DEFAULT_FAMILIES,
    FIRST_STAGE_OUTCOMES,
    OutcomeSpec,
    outcomes_for_families,
)
from src.did.outputs import (  # noqa: E402
    add_strategy_labels,
    figure_path,
    generate_overview_figures,
    plot_coef_comparison,
    plot_event_study,
    plot_placebo_robustness,
)
from src.did.panels import (  # noqa: E402
    AnalysisPanels,
    author_panel_has_multi_lang,
    build_analysis_panels,
    slice_panel_for_ddd,
    wordfish_authors_v2_available,
    wordfish_forum_v2_available,
)
from src.did.robustness import run_robustness_grid  # noqa: E402
from src.did.specs import (  # noqa: E402
    PLOT_STRATEGY_GROUPS,
    StrategySpec,
    author_strategies,
    default_strategies,
    filter_strategy_sample,
    is_author_strategy,
    is_cross_country_strategy,
    strategy_label,
)


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for DiD estimation."""
    parser = argparse.ArgumentParser(description="DiD and event-study estimation.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--families",
        type=str,
        default=",".join(DEFAULT_FAMILIES),
        help="Comma-separated outcome families to run.",
    )
    parser.add_argument("--outcome", type=str, default=None, help="Run single outcome_id.")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=199)
    parser.add_argument("--event-window", type=int, default=30)
    parser.add_argument(
        "--full-coefplots",
        action="store_true",
        help="Also write coefplots_full/ with all strategies.",
    )
    return parser.parse_args()


def _filter_families(families: List[str], config: Dict[str, Any]) -> List[str]:
    """Function summary: drop v2 families when panels are not yet built."""
    out: List[str] = []
    for fam in families:
        if fam == "wordfish_forum_v2" and not wordfish_forum_v2_available(config):
            print(
                "[did_event_study] skip wordfish_forum_v2: missing wordfish_forum_v2/"
                "wordfish_extremity_panel.csv",
                flush=True,
            )
            continue
        if fam == "wordfish_author_v2" and not wordfish_authors_v2_available(config):
            print(
                "[did_event_study] skip wordfish_author_v2: missing wordfish_authors_v2/"
                "wordfish_authors_extremity_panel.csv",
                flush=True,
            )
            continue
        out.append(fam)
    return out


def _panel_for_outcome(panels: AnalysisPanels, spec: OutcomeSpec) -> pd.DataFrame:
    """Function summary: pick subreddit, slice, or author panel for outcome family."""
    if spec.family == "wordfish_author_v2":
        return panels.auth_v2
    if spec.family == "wordfish_author":
        return panels.auth_v1
    if spec.family == "wordfish_forum_v2":
        return panels.sub_v2
    if spec.panel_kind == "author_bin":
        return panels.auth_v1
    if spec.outcome_id == "pair_framing":
        sub = panels.sub_v1
        if sub.empty:
            return sub
        return sub[sub["topic_family"].astype(str).isin({"it_political", "it_others"})]
    return panels.sub_v1


def _resolve_column(panel: pd.DataFrame, spec: OutcomeSpec) -> Optional[str]:
    """Function summary: return outcome column if present."""
    if spec.column in panel.columns:
        return spec.column
    alt = spec.column.replace("_mean", "")
    if alt in panel.columns:
        return alt
    return None


def _strategies_for_outcome(
    oc: OutcomeSpec,
    panel: pd.DataFrame,
    has_en: bool,
    has_de: bool,
) -> tuple:
    """Function summary: strategy list for this outcome and panel."""
    if oc.family in ("wordfish_author", "wordfish_author_v2"):
        return author_strategies(has_en=has_en, has_de=has_de)
    return default_strategies()


def _entity_col_for_strategy(strat: StrategySpec, oc: OutcomeSpec, default_entity: str) -> str:
    """Function summary: entity column; DDD uses subreddit on slice panel."""
    if strat.strategy_id == "within_italy_ddd":
        return "subreddit"
    return default_entity


def run_estimation(
    config: Dict[str, Any],
    families: List[str],
    single_outcome: Optional[str],
    write_figures: bool,
    do_bootstrap: bool,
    event_window: int,
    bootstrap_draws: int = 199,
    full_coefplots: bool = False,
) -> None:
    """Function summary: main estimation loop."""
    families = _filter_families(families, config)
    panels = build_analysis_panels(config)
    if panels.sub_v1.empty:
        print("[did_event_study] empty subreddit panel", flush=True)
        return
    _, _, launch, _ = event_dates_from_config(config)
    did_dir = tables_subdir(config, "did")
    fig_dir = figures_subdir(config, "did")
    did_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    specs = outcomes_for_families(families)
    if single_outcome:
        specs = tuple(o for o in specs if o.outcome_id == single_outcome)

    summary_rows: List[Dict[str, Any]] = []
    summary_path = did_dir / "did_summary.csv"
    if summary_path.is_file():
        summary_path.unlink()

    auth_has_en_v1, auth_has_de_v1 = author_panel_has_multi_lang(panels.auth_v1)
    auth_has_en_v2, auth_has_de_v2 = author_panel_has_multi_lang(panels.auth_v2)

    for oc in specs:
        panel = _panel_for_outcome(panels, oc)
        if panel.empty:
            print(f"[did_event_study] skip {oc.outcome_id}: empty panel", flush=True)
            continue
        y_col = _resolve_column(panel, oc)
        if y_col is None:
            print(f"[did_event_study] skip {oc.outcome_id}: missing {oc.column}", flush=True)
            continue

        if oc.family in ("wordfish_author", "wordfish_author_v2"):
            has_en, has_de = (
                (auth_has_en_v2, auth_has_de_v2)
                if oc.family == "wordfish_author_v2"
                else (auth_has_en_v1, auth_has_de_v1)
            )
        else:
            has_en, has_de = False, False

        strategies = _strategies_for_outcome(oc, panel, has_en, has_de)
        default_entity = "entity_id" if "entity_id" in panel.columns else "subreddit"
        subtitle = _subtitle_for_family(oc.family)
        coef_detail: List[Dict[str, Any]] = []

        for strat in strategies:
            if strat.strategy_id == "within_italy_ddd":
                if panels.slice_panel.empty or not oc.ddd_allowed:
                    continue
                work_panel = slice_panel_for_ddd(panels.slice_panel)
                entity_col = "subreddit"
            elif strat.universe_slice:
                if panels.slice_panel.empty:
                    continue
                work_panel = panels.slice_panel
                entity_col = default_entity
            else:
                work_panel = panel
                entity_col = _entity_col_for_strategy(strat, oc, default_entity)

            if is_cross_country_strategy(strat.strategy_id) and oc.family in (
                "wordfish_author",
                "wordfish_author_v2",
            ):
                if "topic_family" not in work_panel.columns and not has_en and not has_de:
                    continue

            if y_col not in work_panel.columns:
                continue

            res = run_strategy_twfe(
                work_panel,
                strat,
                y_col,
                entity_col=entity_col,
                time_col="time_id",
            )
            wild_p = perm_p = float("nan")
            if do_bootstrap and strat.strategy_id != "within_italy_ddd" and not is_author_strategy(
                strat.strategy_id
            ):
                try:
                    wild_p = wild_cluster_bootstrap_p(
                        work_panel,
                        strat,
                        y_col,
                        n_draws=bootstrap_draws,
                        entity_col=entity_col,
                        time_col="time_id",
                    )
                    perm_p = permutation_test_p(
                        work_panel,
                        strat,
                        y_col,
                        n_draws=bootstrap_draws,
                        entity_col=entity_col,
                        time_col="time_id",
                    )
                except Exception:
                    pass

            row = {
                "outcome_id": oc.outcome_id,
                "outcome_family": oc.family,
                "column": y_col,
                "strategy_id": strat.strategy_id,
                "strategy_label": strategy_label(strat.strategy_id),
                "spec": strat.post_mode,
                "wordfish_tier": oc.tier or "",
                "sign_only_cross_country": int(oc.sign_only_cross_country),
                **res,
                "wild_p": wild_p,
                "perm_p": perm_p,
            }
            summary_rows.append(row)
            coef_detail.append({**row, "detail": "twfe"})

        es_strat = StrategySpec("cross_country_all")
        if oc.family in ("wordfish_author", "wordfish_author_v2"):
            es_strat = StrategySpec("author_it_ban")
        es_panel = filter_strategy_sample(panel, es_strat, window_days=event_window)
        if not es_panel.empty and y_col in es_panel.columns:
            es_entity = default_entity
            if is_author_strategy(es_strat.strategy_id):
                es_entity = default_entity
            _, es_df = estimate_event_study(
                es_panel,
                y_col,
                window=event_window,
                entity_col=es_entity,
                time_col="time_id",
            )
            if not es_df.empty:
                es_df["outcome_id"] = oc.outcome_id
                es_df.to_csv(did_dir / f"eventstudy_{oc.outcome_id}.csv", index=False)
                if write_figures:
                    plot_event_study(
                        es_df,
                        oc.outcome_id,
                        figure_path(fig_dir, oc.family, "event_study", oc.outcome_id),
                        title=f"Event study: {oc.outcome_id}",
                        subtitle=subtitle,
                    )

        if oc.family == "lexical":
            rob = run_robustness_grid(panel, StrategySpec("cross_country_all"), y_col, launch)
            rob_df = pd.DataFrame(rob)
            rob_df["outcome_id"] = oc.outcome_id
            rob_df.to_csv(did_dir / f"robustness_{oc.outcome_id}.csv", index=False)
            if write_figures:
                plot_placebo_robustness(
                    rob_df,
                    oc.outcome_id,
                    figure_path(fig_dir, oc.family, "robustness", f"placebo_{oc.outcome_id}"),
                )

        if coef_detail:
            pd.DataFrame(coef_detail).to_csv(
                did_dir / f"did_coefficients_{oc.outcome_id}.csv",
                index=False,
            )

        print(f"[did_event_study] finished {oc.outcome_id}", flush=True)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_path, index=False)
        labeled = add_strategy_labels(summary_df)
        labeled.to_csv(did_dir / "did_summary_labeled.csv", index=False)

        if write_figures:
            headline_strats = list(PLOT_STRATEGY_GROUPS["headline"])
            for oid in summary_df["outcome_id"].unique():
                fam = summary_df.loc[summary_df["outcome_id"] == oid, "outcome_family"].iloc[0]
                sub = _subtitle_for_family(str(fam))
                plot_coef_comparison(
                    labeled,
                    oid,
                    figure_path(fig_dir, str(fam), "coefplots_headline", oid),
                    strategies=headline_strats,
                    subtitle=sub,
                )
                if full_coefplots:
                    plot_coef_comparison(
                        labeled,
                        oid,
                        figure_path(fig_dir, str(fam), "coefplots_full", oid),
                        strategies=None,
                        subtitle=sub,
                    )
            generate_overview_figures(labeled, fig_dir)
            fs = summary_df[summary_df["outcome_id"].isin(FIRST_STAGE_OUTCOMES)]
            if not fs.empty and fs["beta"].abs().max() < 1e-6:
                print(
                    "[did_event_study] WARNING: AI first-stage β near zero — flag in write-up",
                    flush=True,
                )

    print(f"[did_event_study] wrote {summary_path}", flush=True)


def _subtitle_for_family(family: str) -> str:
    """Function summary: re-export family subtitle for event-study titles."""
    from src.did.outputs import _subtitle_for_family as _sub

    return _sub(family)


def main() -> None:
    """Function summary: CLI entry."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    run_estimation(
        config,
        families,
        args.outcome,
        write_figures=not args.no_figures,
        do_bootstrap=not args.no_bootstrap,
        event_window=args.event_window,
        bootstrap_draws=args.bootstrap_draws,
        full_coefplots=args.full_coefplots,
    )


if __name__ == "__main__":
    main()
