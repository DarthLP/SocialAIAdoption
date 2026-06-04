"""
Script summary:
Estimate Italy ChatGPT-ban DiD, event studies, and triple-differences on prepared panels.

Functionality:
- TWFE DiD (subreddit-day and author-bin panels) with multi-strategy summary table (full ban, early 7/14d, post-phase short/medium/long windows).
- Event-study dynamics, early-ban windows, wild-cluster bootstrap, permutation tests.
- Robustness: placebo 2023-03-16, symmetric window trims.
- Outcome families: lexical (00), semantic axis (01), forum/author Wordfish v1 and v2.
- Nested table folders under did/estimates/ (mirrors figure families) plus summary CSV/txt slices.
- Nested figure folders with human-readable strategy labels.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_did_subreddit_panel.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/analysis/did_event_study.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/analysis/did_event_study.py --config config/italy_polarization_setup.yaml --families lexical
  .venv/bin/python scripts/analysis/did_event_study.py --config config/italy_polarization_setup.yaml --outcome net_ideology --no-figures
  .venv/bin/python scripts/analysis/did_event_study.py --config config/italy_polarization_setup.yaml --figures-only
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from scripts.diagnostics.descriptives_util import event_dates_from_config  # noqa: E402
from src.config_utils import figures_subdir, load_config  # noqa: E402
from src.did.paths import (  # noqa: E402
    did_estimates_dir,
    did_event_study_path,
    did_legacy_coefficient_path,
    did_outcome_table_path,
    did_summary_dir,
    did_summary_paths,
)
from src.did.estimate import (  # noqa: E402
    annotate_pretrend_quality,
    apply_degeneracy_guard,
    estimate_event_study,
    estimate_pretrend_f,
    run_strategy_twfe,
)
from src.did.inference import placebo_in_space_p, wild_cluster_bootstrap_p  # noqa: E402
from src.did.outcomes import (  # noqa: E402
    DEFAULT_FAMILIES,
    FIRST_STAGE_OUTCOMES,
    SEM_AXIS_IDEOLOGY_EXTREME_LEFT_COL,
    SEM_AXIS_IDEOLOGY_EXTREME_RIGHT_COL,
    OutcomeSpec,
    outcomes_for_families,
)
from src.did.figure_readmes import write_all_family_readmes  # noqa: E402
from src.did.outputs import (  # noqa: E402
    EventStudySeries,
    figure_path,
    plot_event_study,
    plot_placebo_robustness,
    plot_sem_axis_ideology_tail_shift_event_study,
    regenerate_did_figures,
    write_summary_exports,
)
from src.did.panels import (  # noqa: E402
    AnalysisPanels,
    author_panel_has_multi_lang,
    author_semantic_week_panel_available,
    build_analysis_panels,
    comment_panel_available,
    slice_panel_for_ddd,
    wordfish_authors_v2_available,
    wordfish_forum_v2_available,
)
from src.did.robustness import run_robustness_grid  # noqa: E402
from src.did.specs import (  # noqa: E402
    PLOT_STRATEGY_GROUPS,
    StrategySpec,
    activate_post_phases_from_config,
    author_strategies,
    filter_strategy_sample,
    is_author_strategy,
    inference_role_for_strategy,
    is_cross_country_strategy,
    is_entity_fe_only_strategy,
    is_placebo_in_space_eligible_strategy,
    is_wcb_eligible_strategy,
    strategies_for_outcome,
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
    parser.add_argument(
        "--outcomes",
        type=str,
        default=None,
        help="Comma-separated outcome_ids (filters within selected families).",
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=9999)
    parser.add_argument("--event-window", type=int, default=30)
    parser.add_argument(
        "--full-coefplots",
        action="store_true",
        help="Also write coefplots_full/ with all strategies.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate figures and READMEs from existing did_summary.csv (no estimation).",
    )
    parser.add_argument(
        "--author-spec",
        type=str,
        default=None,
        help="Wordfish author bin spec (week7, week3); overrides did.author_wordfish_spec.",
    )
    parser.add_argument(
        "--comment-sample-frac",
        type=float,
        default=None,
        help="Random fraction of comment panel rows (dev/smoke runs).",
    )
    parser.add_argument(
        "--comment-max-rows",
        type=int,
        default=None,
        help="Cap comment panel rows after sampling.",
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
        if fam == "semantic_axis_author_week" and not author_semantic_week_panel_available(config):
            print(
                "[did_event_study] skip semantic_axis_author_week: missing "
                "did/panels/author/did_author_semantic_week_panel.csv",
                flush=True,
            )
            continue
        if fam in ("lexical_comment", "semantic_axis_comment") and not comment_panel_available(
            config
        ):
            print(
                "[did_event_study] skip "
                f"{fam}: missing did/panels/comment/did_comment_panel_1d/; "
                "run prepare_did_comment_panel.py",
                flush=True,
            )
            continue
        out.append(fam)
    return out


def _panel_for_outcome(panels: AnalysisPanels, spec: OutcomeSpec) -> pd.DataFrame:
    """Function summary: pick subreddit, slice, or author panel for outcome family."""
    if spec.panel_kind == "comment":
        return panels.comment_1d
    if spec.panel_kind == "author_day":
        return panels.author_day_1d
    if spec.family == "semantic_axis_author_week" or spec.panel_kind == "author_semantic_week":
        return panels.auth_semantic
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
    if oc.family in ("wordfish_author", "wordfish_author_v2", "semantic_axis_author_week"):
        return author_strategies(has_en=has_en, has_de=has_de)
    if oc.panel_kind == "author_semantic_week":
        return author_strategies(has_en=has_en, has_de=has_de)
    return strategies_for_outcome(
        oc.family,
        panel_has_topic_family="topic_family" in panel.columns,
        author_has_multi_lang=has_en or has_de,
        panel_kind=oc.panel_kind,
    )


def _entity_col_for_strategy(strat: StrategySpec, oc: OutcomeSpec, default_entity: str) -> str:
    """Function summary: entity column; DDD uses subreddit on slice panel."""
    if strat.strategy_id == "within_italy_ddd":
        return "subreddit"
    return default_entity


def run_figures_only(
    config: Dict[str, Any],
    families: List[str],
    full_coefplots: bool = False,
) -> None:
    """Function summary: rebuild DiD figures from saved did_summary.csv."""
    summary_path, _ = did_summary_paths(config)
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing {summary_path}; run estimation first or pass a valid --config."
        )
    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        print("[did_event_study] empty did_summary.csv", flush=True)
        return
    fig_dir = figures_subdir(config, "did")
    fig_dir.mkdir(parents=True, exist_ok=True)
    outcome_ids_by_family = regenerate_did_figures(
        summary_df,
        fig_dir,
        families=families or None,
        full_coefplots=full_coefplots,
    )
    write_all_family_readmes(
        fig_dir,
        list(outcome_ids_by_family.keys()),
        outcome_ids_by_family,
        full_coefplots=full_coefplots,
    )
    print(f"[did_event_study] regenerated figures under {fig_dir}", flush=True)


def _median_abs_beta_ok(rows: List[Dict[str, Any]]) -> float:
    """Function summary: median |β| among successful TWFE rows for degeneracy scaling."""
    vals = [
        abs(float(r["beta"]))
        for r in rows
        if r.get("estimation_note") in ("ok", "ok_entity_fe_only")
        and np.isfinite(r.get("beta", np.nan))
        and abs(float(r["beta"])) < 1e6
    ]
    if not vals:
        return 1.0
    return float(np.median(vals))


def _apply_outcome_degeneracy_guards(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Function summary: apply median-scaled degeneracy guard and pretrend quality tags per strategy row."""
    median_b = _median_abs_beta_ok(rows)
    out: List[Dict[str, Any]] = []
    for r in rows:
        cond = float(r.pop("design_cond", np.nan)) if "design_cond" in r else float("nan")
        if not np.isfinite(cond):
            cond = float("inf")
        cleaned = {k: v for k, v in r.items() if k != "design_cond"}
        guarded = apply_degeneracy_guard(cleaned, cond, median_b)
        out.append(annotate_pretrend_quality(guarded))
    return out


def _plot_subreddit_sem_axis_tail_shift(
    panel: pd.DataFrame,
    fig_dir: Path,
    event_window: int,
) -> None:
    """Function summary: dual-tail ideology event study on subreddit panel (cross_country_all)."""
    if panel.empty:
        return
    if SEM_AXIS_IDEOLOGY_EXTREME_LEFT_COL not in panel.columns or (
        SEM_AXIS_IDEOLOGY_EXTREME_RIGHT_COL not in panel.columns
    ):
        return
    strat = StrategySpec("cross_country_all")
    sample = filter_strategy_sample(panel, strat, window_days=event_window)
    if sample.empty:
        return
    entity = "subreddit" if "subreddit" in sample.columns else "entity_id"
    series: List[EventStudySeries] = []
    for y_col, label in (
        (SEM_AXIS_IDEOLOGY_EXTREME_LEFT_COL, "extreme-LEFT tail share (sem axis, <p10)"),
        (SEM_AXIS_IDEOLOGY_EXTREME_RIGHT_COL, "extreme-RIGHT tail share (sem axis, >p90)"),
    ):
        _, es_df = estimate_event_study(
            sample,
            y_col,
            window=event_window,
            entity_col=entity,
            time_col="time_id",
        )
        if es_df.empty:
            return
        series.append(EventStudySeries(label=label, es_df=es_df, rel_col="rel_day"))
    if len(series) != 2:
        return
    out = figure_path(fig_dir, "semantic_axis", "event_study", "sem_axis_ideology_tail_shift")
    plot_sem_axis_ideology_tail_shift_event_study(
        series[0], series[1], out, bin_days=1
    )


def _write_legacy_coefficient_alias(config: Dict[str, Any], oc: OutcomeSpec, coef_df: pd.DataFrame) -> None:
    """Function summary: write flat did_coefficients_{outcome_id}.csv at did/ root (grep / handoff)."""
    if coef_df.empty:
        return
    export = coef_df.drop(columns=["design_cond"], errors="ignore")
    path = did_legacy_coefficient_path(config, oc.outcome_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(path, index=False)


def run_estimation(
    config: Dict[str, Any],
    families: List[str],
    single_outcome: Optional[str],
    outcome_ids: Optional[List[str]],
    write_figures: bool,
    do_bootstrap: bool,
    event_window: int,
    bootstrap_draws: int = 199,
    full_coefplots: bool = False,
    author_wordfish_spec: Optional[str] = None,
    comment_sample_frac: Optional[float] = None,
    comment_max_rows: Optional[int] = None,
) -> None:
    """Function summary: main estimation loop."""
    activate_post_phases_from_config(config)
    families = _filter_families(families, config)
    panels = build_analysis_panels(
        config,
        families=families,
        author_wordfish_spec=author_wordfish_spec,
        comment_sample_frac=comment_sample_frac,
        comment_max_rows=comment_max_rows,
    )
    if (
        panels.sub_v1.empty
        and panels.auth_v1.empty
        and panels.auth_v2.empty
        and panels.auth_semantic.empty
        and panels.comment_1d.empty
        and panels.author_day_1d.empty
    ):
        print("[did_event_study] no estimation panels loaded", flush=True)
        return
    _, _, launch, _ = event_dates_from_config(config)
    estimates_dir = did_estimates_dir(config)
    summary_dir = did_summary_dir(config)
    fig_dir = figures_subdir(config, "did")
    estimates_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    specs = outcomes_for_families(families)
    if single_outcome:
        specs = tuple(o for o in specs if o.outcome_id == single_outcome)
    if outcome_ids:
        oid_set = set(outcome_ids)
        specs = tuple(o for o in specs if o.outcome_id in oid_set)

    summary_rows: List[Dict[str, Any]] = []
    outcome_ids_by_family: Dict[str, List[str]] = {}
    summary_path, _labeled_path = did_summary_paths(config)

    auth_has_en_v1, auth_has_de_v1 = author_panel_has_multi_lang(panels.auth_v1)
    auth_has_en_v2, auth_has_de_v2 = author_panel_has_multi_lang(panels.auth_v2)
    auth_sem_has_en, auth_sem_has_de = author_panel_has_multi_lang(panels.auth_semantic)

    for oc in specs:
        panel = _panel_for_outcome(panels, oc)
        if panel.empty:
            print(f"[did_event_study] skip {oc.outcome_id}: empty panel", flush=True)
            continue
        y_col = _resolve_column(panel, oc)
        if y_col is None:
            print(f"[did_event_study] skip {oc.outcome_id}: missing {oc.column}", flush=True)
            continue

        if oc.family == "semantic_axis_author_week":
            has_en, has_de = auth_sem_has_en, auth_sem_has_de
        elif oc.family in ("wordfish_author", "wordfish_author_v2"):
            has_en, has_de = (
                (auth_has_en_v2, auth_has_de_v2)
                if oc.family == "wordfish_author_v2"
                else (auth_has_en_v1, auth_has_de_v1)
            )
        else:
            has_en, has_de = False, False

        strategies = _strategies_for_outcome(oc, panel, has_en, has_de)
        default_entity = "entity_id" if "entity_id" in panel.columns else "subreddit"
        coef_detail: List[Dict[str, Any]] = []
        strategy_rows: List[Dict[str, Any]] = []

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
                "semantic_axis_author_week",
            ):
                if "topic_family" not in work_panel.columns and not has_en and not has_de:
                    continue

            if y_col not in work_panel.columns:
                continue

            cluster_col = entity_col
            if oc.panel_kind in ("comment", "author_day") and "author" in work_panel.columns:
                entity_col = "author"
                cluster_col = "author"
            res = run_strategy_twfe(
                work_panel,
                strat,
                y_col,
                entity_col=entity_col,
                time_col="time_id",
                cluster_col=cluster_col,
                panel_kind=oc.panel_kind,
            )
            if strat.strategy_id != "within_italy_ddd" and not is_entity_fe_only_strategy(
                strat.strategy_id
            ):
                pretrend_p, pretrend_note = estimate_pretrend_f(
                    filter_strategy_sample(work_panel, strat, window_days=event_window),
                    y_col,
                    entity_col=entity_col,
                    time_col="time_id",
                )
                res["pretrend_F_p"] = pretrend_p
                if pretrend_note and res.get("estimation_note") in ("ok", "ok_entity_fe_only"):
                    note = str(res.get("estimation_note", "ok"))
                    res["estimation_note"] = (
                        pretrend_note if note == "ok" else f"{note};{pretrend_note}"
                    )
            wild_p = float("nan")
            p_placebo_space = float("nan")
            placebo_p_floor = float("nan")
            placebo_note = np.nan
            role = inference_role_for_strategy(strat.strategy_id)
            if do_bootstrap:
                try:
                    if is_wcb_eligible_strategy(strat.strategy_id) and oc.panel_kind != "comment":
                        wcb_entity = entity_col
                        if is_author_strategy(strat.strategy_id) and "author" in work_panel.columns:
                            wcb_entity = "author"
                        wild_p = wild_cluster_bootstrap_p(
                            work_panel,
                            strat,
                            y_col,
                            n_draws=bootstrap_draws,
                            entity_col=wcb_entity,
                            time_col="time_id",
                        )
                    if is_placebo_in_space_eligible_strategy(strat.strategy_id):
                        pis = placebo_in_space_p(
                            work_panel,
                            strat,
                            y_col,
                            entity_col=entity_col,
                            time_col="time_id",
                        )
                        p_placebo_space = pis.p
                        placebo_p_floor = pis.p_floor
                    elif is_cross_country_strategy(strat.strategy_id):
                        placebo_note = "not_applicable_single_country_contrast"
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
                "inference_role": role,
                **res,
                "wild_p": wild_p,
                "perm_p": p_placebo_space,
                "p_placebo_space": p_placebo_space,
                "placebo_p_floor": placebo_p_floor,
                "placebo_note": placebo_note,
            }
            strategy_rows.append(row)

        strategy_rows = _apply_outcome_degeneracy_guards(strategy_rows)
        for row in strategy_rows:
            summary_rows.append(row)
            coef_detail.append({**row, "detail": "twfe"})

        es_strat = StrategySpec("cross_country_all")
        if oc.family in ("wordfish_author", "wordfish_author_v2", "semantic_axis_author_week"):
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
                es_path = did_event_study_path(config, oc.family, oc.outcome_id)
                es_path.parent.mkdir(parents=True, exist_ok=True)
                es_df.to_csv(
                    es_path,
                    index=False,
                )
                if write_figures:
                    plot_event_study(
                        es_df,
                        oc.outcome_id,
                        figure_path(fig_dir, oc.family, "event_study", oc.outcome_id),
                    )

        if oc.family == "lexical":
            rob = run_robustness_grid(panel, StrategySpec("cross_country_all"), y_col, launch)
            rob_df = pd.DataFrame(rob)
            rob_df["outcome_id"] = oc.outcome_id
            rob_df.to_csv(
                did_outcome_table_path(config, oc.family, "robustness", oc.outcome_id),
                index=False,
            )
            if write_figures:
                plot_placebo_robustness(
                    rob_df,
                    oc.outcome_id,
                    figure_path(fig_dir, oc.family, "robustness", f"placebo_{oc.outcome_id}"),
                )

        if coef_detail:
            coef_df = pd.DataFrame(coef_detail)
            coef_path = did_outcome_table_path(
                config, oc.family, "coefficients", oc.outcome_id
            )
            coef_path.parent.mkdir(parents=True, exist_ok=True)
            coef_df.to_csv(coef_path, index=False)
            _write_legacy_coefficient_alias(config, oc, coef_df)

        outcome_ids_by_family.setdefault(oc.family, []).append(oc.outcome_id)
        print(f"[did_event_study] finished {oc.outcome_id}", flush=True)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        if summary_path.is_file():
            old = pd.read_csv(summary_path)
            summary_df = pd.concat([old, summary_df], ignore_index=True)
            summary_df = summary_df.drop_duplicates(
                subset=[
                    c
                    for c in ("outcome_id", "strategy_id", "spec")
                    if c in summary_df.columns
                ],
                keep="last",
            )
        write_summary_exports(summary_df, summary_dir, launch)

        if write_figures:
            regenerate_did_figures(
                summary_df,
                fig_dir,
                families=families,
                full_coefplots=full_coefplots,
            )
            write_all_family_readmes(
                fig_dir,
                list(families),
                outcome_ids_by_family,
                full_coefplots=full_coefplots,
            )
            if "semantic_axis" in families:
                _plot_subreddit_sem_axis_tail_shift(panels.sub_v1, fig_dir, event_window)
                outcome_ids_by_family.setdefault("semantic_axis", []).append(
                    "sem_axis_ideology_tail_shift"
                )
            fs = summary_df[summary_df["outcome_id"].isin(FIRST_STAGE_OUTCOMES)]
            if not fs.empty and fs["beta"].abs().max() < 1e-6:
                print(
                    "[did_event_study] WARNING: AI first-stage β near zero — flag in write-up",
                    flush=True,
                )

    print(f"[did_event_study] wrote {summary_path}", flush=True)


def main() -> None:
    """Function summary: CLI entry."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    if args.figures_only:
        run_figures_only(config, families, full_coefplots=args.full_coefplots)
        return
    outcome_ids = None
    if args.outcomes:
        outcome_ids = [o.strip() for o in args.outcomes.split(",") if o.strip()]
    run_estimation(
        config,
        families,
        args.outcome,
        outcome_ids,
        write_figures=not args.no_figures,
        do_bootstrap=not args.no_bootstrap,
        event_window=args.event_window,
        bootstrap_draws=args.bootstrap_draws,
        full_coefplots=args.full_coefplots,
        author_wordfish_spec=args.author_spec,
        comment_sample_frac=args.comment_sample_frac,
        comment_max_rows=args.comment_max_rows,
    )


if __name__ == "__main__":
    main()
