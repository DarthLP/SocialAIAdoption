"""
Script summary:
Within-Italy DiD estimation for Q&A substitution (comment volume + question-mark rate) around the ChatGPT ban.

Functionality:
- Headline: qa:post TWFE on Italian Q&A vs non-Q&A forums (zero-filled panel; fepois counts + log1p OLS).
- Leave-one-out: drop ItaliaPersonalFinance, Universitaly, or both from headline volume spec.
- 3-day event study on question_share and n_comments.
- Phase contrasts (pre vs ban; suggestive post-lift check).
- Hub placebo: Italian Q&A vs DE/EU/UK hubs; IT non-Q&A vs hubs pre-trend diagnostic.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_qa_volume_panel.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/analysis/qa_volume_did.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REF_REL_PERIOD = -1

LOO_DROP_SPECS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("within_italy_static_loo_drop_ipf", ("ItaliaPersonalFinance",)),
    ("within_italy_static_loo_drop_universitaly", ("Universitaly",)),
    ("within_italy_static_loo_drop_both", ("ItaliaPersonalFinance", "Universitaly")),
)

DOMANDE_DAREDDIT_RAW_COMMENTS = 42
DOMANDE_DAREDDIT_RAW_DAYS = 21


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
from src.config_utils import load_config, tables_subdir  # noqa: E402
from src.did.estimate import _pack_result  # noqa: E402
from src.qa_substitution import phase_contrast_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Q&A substitution volume/rate DiD.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--panel", type=str, default="1d", choices=["1d", "3d"])
    return parser.parse_args()


def _empty_est(n_obs: int = 0, note: str = "insufficient_obs") -> Dict[str, Any]:
    """Function summary: return empty estimation result dict."""
    return _pack_result(float("nan"), float("nan"), n_obs, 0, note)


def _feols_fit(
    formula: str,
    data: pd.DataFrame,
    coef_name: str,
    cluster_col: str = "subreddit",
    weights_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Function summary: pyfixest feols with CRV1 cluster SE.

    Parameters:
    - formula: pyfixest formula string.
    - data: estimation sample.
    - coef_name: coefficient to extract.
    - cluster_col: cluster column for vcov.
    - weights_col: optional weights column name.

    Returns:
    - Result dict via _pack_result.
    """
    try:
        from pyfixest.estimation import feols
    except ImportError:
        return _empty_est(len(data), "pyfixest_missing")
    if len(data) < 20 or data["subreddit"].nunique() < 2:
        return _empty_est(len(data), "insufficient_obs")
    n_cl = int(data[cluster_col].nunique()) if cluster_col in data.columns else 0
    vcov: Any = {"CRV1": cluster_col} if cluster_col in data.columns else "iid"
    try:
        kw: Dict[str, Any] = {"vcov": vcov}
        fit_data = data
        if weights_col and weights_col in data.columns:
            fit_data = data.copy()
            fit_data[weights_col] = (
                pd.to_numeric(fit_data[weights_col], errors="coerce").astype(float).fillna(1.0).clip(lower=1e-9)
            )
            kw["weights"] = weights_col
        fit = feols(formula, data=fit_data, **kw)
        coefs = fit.coef()
        beta = float(coefs.loc[coef_name]) if coef_name in coefs.index else float("nan")
        se_frame = fit.se()
        se = float(se_frame.loc[coef_name]) if coef_name in se_frame.index else float("nan")
        return _pack_result(beta, se, len(data), n_cl)
    except Exception:
        return _empty_est(len(data), "estimation_error")


def _fepois_fit(
    formula: str,
    data: pd.DataFrame,
    coef_name: str,
    cluster_col: str = "subreddit",
    count_col: str = "n_comments",
) -> Dict[str, Any]:
    """Function summary: pyfixest Poisson FE for count outcomes.

    Parameters:
    - formula: pyfixest formula string.
    - data: estimation sample with non-negative counts.
    - coef_name: coefficient to extract.
    - cluster_col: cluster column for vcov.
    - count_col: outcome count column used for non-negativity filter.

    Returns:
    - Result dict via _pack_result.
    """
    try:
        from pyfixest.estimation import fepois
    except ImportError:
        return _empty_est(len(data), "pyfixest_missing")
    work = data.copy()
    if count_col in work.columns:
        work = work[pd.to_numeric(work[count_col], errors="coerce").fillna(0) >= 0]
    if len(work) < 20 or work["subreddit"].nunique() < 2:
        return _empty_est(len(work), "insufficient_obs")
    n_cl = int(work[cluster_col].nunique()) if cluster_col in work.columns else 0
    vcov: Any = {"CRV1": cluster_col} if cluster_col in work.columns else "iid"
    try:
        fit = fepois(formula, data=work, vcov=vcov)
        coefs = fit.coef()
        beta = float(coefs.loc[coef_name]) if coef_name in coefs.index else float("nan")
        se_frame = fit.se()
        se = float(se_frame.loc[coef_name]) if coef_name in se_frame.index else float("nan")
        return _pack_result(beta, se, len(work), n_cl)
    except Exception:
        return _empty_est(len(work), "estimation_error")


def _within_italy_sample(panel: pd.DataFrame) -> pd.DataFrame:
    """Function summary: restrict to Italian Q&A and non-Q&A forums."""
    return panel[panel["IT"].astype(int) == 1].copy()


def _qa_vs_hub_sample(panel: pd.DataFrame) -> pd.DataFrame:
    """Function summary: Italian Q&A forums plus DE/EU/UK hub controls."""
    mask = (panel["qa"].astype(int) == 1) | (panel["is_hub"].astype(int) == 1)
    return panel[mask].copy()


def _nonqa_vs_hub_sample(panel: pd.DataFrame) -> pd.DataFrame:
    """Function summary: Italian non-Q&A forums plus hubs for parallel-trends placebo."""
    mask = ((panel["IT"].astype(int) == 1) & (panel["qa"].astype(int) == 0)) | (
        panel["is_hub"].astype(int) == 1
    )
    return panel[mask].copy()


def _prep_outcome(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    """Function summary: coerce outcome and add log1p volume helper."""
    work = df.copy()
    work["y"] = pd.to_numeric(work[y_col], errors="coerce")
    if y_col == "n_comments":
        work["log1p_n_comments"] = np.log1p(work["y"].clip(lower=0))
    work["qa"] = work["qa"].astype(float)
    work["post"] = work["post"].astype(float)
    work["qa_post"] = work["qa"] * work["post"]
    work["subreddit"] = work["subreddit"].astype(str)
    work["time_id"] = work["time_id"].astype(str)
    return work.dropna(subset=["y", "subreddit", "time_id"])


def _summary_row(
    spec_id: str,
    outcome: str,
    estimator: str,
    sample: str,
    coef_name: str,
    res: Dict[str, Any],
) -> Dict[str, Any]:
    """Function summary: flatten one estimation result into a summary CSV row."""
    return {
        "spec_id": spec_id,
        "outcome": outcome,
        "estimator": estimator,
        "sample": sample,
        "coef_name": coef_name,
        "beta": res.get("beta"),
        "se": res.get("se"),
        "ci_low": res.get("ci_low"),
        "ci_high": res.get("ci_high"),
        "pvalue": res.get("pvalue"),
        "n_obs": res.get("n_obs"),
        "n_clusters": res.get("n_clusters"),
        "estimation_note": res.get("estimation_note"),
    }


def estimate_static_specs(panel: pd.DataFrame) -> List[Dict[str, Any]]:
    """Function summary: run headline within-Italy and placebo static DiD specs.

    Parameters:
    - panel: annotated subreddit-day or 3d panel.

    Returns:
    - List of summary row dicts.
    """
    rows: List[Dict[str, Any]] = []
    within = _within_italy_sample(panel)

    static_outcomes: Tuple[Tuple[str, str, str], ...] = (
        ("n_comments", "fepois", "n_comments ~ qa:post | subreddit + time_id"),
        ("n_comments", "feols_log1p", "log1p_n_comments ~ qa:post | subreddit + time_id"),
        ("n_questions", "fepois", "n_questions ~ qa:post | subreddit + time_id"),
        ("n_authors", "fepois", "n_authors ~ qa:post | subreddit + time_id"),
        ("question_share", "feols_wtd", "question_share ~ qa:post | subreddit + time_id"),
        ("qmark_rate_100w", "feols_wtd", "qmark_rate_100w ~ qa:post | subreddit + time_id"),
    )

    for outcome, estimator, formula in static_outcomes:
        y_src = "n_comments" if estimator == "feols_log1p" else outcome
        work = _prep_outcome(within, y_src)
        if estimator == "fepois":
            fit_df = work.drop(columns=["y"], errors="ignore")
            fit_df[outcome] = work["y"]
            res = _fepois_fit(formula, fit_df, "qa:post", count_col=outcome)
        elif estimator == "feols_log1p":
            work["log1p_n_comments"] = work["y"] if "log1p_n_comments" not in work.columns else work["log1p_n_comments"]
            res = _feols_fit(formula, work, "qa:post")
        else:
            wcol = "n_comments" if outcome in {"question_share", "qmark_rate_100w"} else None
            res = _feols_fit(formula, work.assign(**{outcome: work["y"]}), "qa:post", weights_col=wcol)
        rows.append(
            _summary_row(
                "within_italy_static",
                outcome,
                estimator,
                "IT_qa_vs_nonqa",
                "qa:post",
                res,
            )
        )

    qa_hub = _qa_vs_hub_sample(panel)
    for outcome in ("n_comments", "question_share"):
        work = _prep_outcome(qa_hub, outcome)
        if outcome == "n_comments":
            fit_df = work.drop(columns=["y"], errors="ignore")
            fit_df["n_comments"] = work["y"]
            res = _fepois_fit("n_comments ~ qa:post | subreddit + time_id", fit_df, "qa:post")
            est = "fepois"
        else:
            res = _feols_fit(
                "question_share ~ qa:post | subreddit + time_id",
                work.assign(question_share=work["y"]),
                "qa:post",
                weights_col="n_comments",
            )
            est = "feols_wtd"
        rows.append(
            _summary_row(
                "placebo_qa_vs_hubs",
                outcome,
                est,
                "IT_qa_vs_hubs",
                "qa:post",
                res,
            )
        )

    nonqa_hub = _nonqa_vs_hub_sample(panel)
    nonqa_hub = nonqa_hub.copy()
    nonqa_hub["it_nonqa"] = ((nonqa_hub["IT"] == 1) & (nonqa_hub["qa"] == 0)).astype(float)
    nonqa_hub["it_nonqa_post"] = nonqa_hub["it_nonqa"] * nonqa_hub["post"]
    for outcome in ("n_comments",):
        work = _prep_outcome(nonqa_hub, outcome)
        fit_df = work.drop(columns=["y"], errors="ignore")
        fit_df["n_comments"] = work["y"]
        fit_df["it_nonqa"] = nonqa_hub.loc[work.index, "it_nonqa"].astype(float)
        res = _fepois_fit("n_comments ~ it_nonqa:post | subreddit + time_id", fit_df, "it_nonqa:post")
        rows.append(
            _summary_row(
                "placebo_nonqa_vs_hubs_pretrend",
                outcome,
                "fepois",
                "IT_nonqa_vs_hubs",
                "it_nonqa:post",
                res,
            )
        )

    return rows


def _estimate_within_italy_n_comments(
    panel: pd.DataFrame,
    spec_id: str,
    sample: str = "IT_qa_vs_nonqa",
) -> List[Dict[str, Any]]:
    """Function summary: run headline n_comments fepois + log1p OLS within Italy for one sample.

    Parameters:
    - panel: annotated panel (possibly LOO-filtered).
    - spec_id: row label for qa_did_summary.csv.
    - sample: sample description string.

    Returns:
    - Two summary row dicts (fepois and feols_log1p).
    """
    rows: List[Dict[str, Any]] = []
    within = _within_italy_sample(panel)
    specs: Tuple[Tuple[str, str, str], ...] = (
        ("n_comments", "fepois", "n_comments ~ qa:post | subreddit + time_id"),
        ("n_comments", "feols_log1p", "log1p_n_comments ~ qa:post | subreddit + time_id"),
    )
    for outcome, estimator, formula in specs:
        y_src = "n_comments" if estimator == "feols_log1p" else outcome
        work = _prep_outcome(within, y_src)
        if estimator == "fepois":
            fit_df = work.drop(columns=["y"], errors="ignore")
            fit_df[outcome] = work["y"]
            res = _fepois_fit(formula, fit_df, "qa:post", count_col=outcome)
        else:
            work["log1p_n_comments"] = work["y"] if "log1p_n_comments" not in work.columns else work["log1p_n_comments"]
            res = _feols_fit(formula, work, "qa:post")
        rows.append(_summary_row(spec_id, outcome, estimator, sample, "qa:post", res))
    return rows


def estimate_loo_specs(panel: pd.DataFrame) -> List[Dict[str, Any]]:
    """Function summary: leave-one-out (or both) robustness for dominant treated Q&A forums.

    Parameters:
    - panel: zero-filled annotated subreddit-day panel.

    Returns:
    - List of summary row dicts for n_comments fepois and feols_log1p.
    """
    rows: List[Dict[str, Any]] = []
    for spec_id, drop_subs in LOO_DROP_SPECS:
        filtered = panel[~panel["subreddit"].astype(str).isin(drop_subs)].copy()
        sample = f"IT_qa_vs_nonqa_excl_{'_'.join(s.lower() for s in drop_subs)}"
        rows.extend(_estimate_within_italy_n_comments(filtered, spec_id, sample=sample))
    return rows


def _parse_es_coef_name(name: str) -> Optional[int]:
    """Function summary: parse rel_period k from pyfixest i(rel_period, qa) coefficient name."""
    m = re.search(r"rel_period::(-?\d+):qa", str(name))
    if m:
        return int(m.group(1))
    return None


def estimate_event_study(panel_3d: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Function summary: 3-day bin event study for qa differential effects.

    Parameters:
    - panel_3d: 3-day aggregated panel.
    - outcome: outcome column name.

    Returns:
    - DataFrame with rel_period, beta, se for plotting.
    """
    within = _within_italy_sample(panel_3d)
    work = _prep_outcome(within, outcome)
    work["rel_period"] = pd.to_numeric(within.loc[work.index, "rel_period"], errors="coerce")
    work = work.dropna(subset=["rel_period"])
    work["time_id"] = within.loc[work.index, "time_id_bin3"].astype(str)
    if len(work) < 30:
        return pd.DataFrame()
    try:
        from pyfixest.estimation import feols
    except ImportError:
        return pd.DataFrame()
    wcol = "n_comments" if outcome in {"question_share", "qmark_rate_100w"} else None
    y_col = outcome
    if outcome in {"n_comments", "n_questions", "n_authors"}:
        y_col = f"log1p_{outcome}"
        work = work.copy()
        work[y_col] = np.log1p(pd.to_numeric(work["y"], errors="coerce").fillna(0).clip(lower=0))
    formula = f"{y_col} ~ i(rel_period, qa, ref={REF_REL_PERIOD}) | subreddit + time_id"
    try:
        kw: Dict[str, Any] = {"vcov": {"CRV1": "subreddit"}}
        fit_work = work
        if wcol:
            fit_work = work.copy()
            fit_work[wcol] = (
                pd.to_numeric(fit_work[wcol], errors="coerce").astype(float).fillna(1.0).clip(lower=1e-9)
            )
            kw["weights"] = wcol
        fit = feols(formula, data=fit_work, **kw)
        coefs = fit.coef()
        ses = fit.se()
    except Exception:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for name in coefs.index:
        k = _parse_es_coef_name(str(name))
        if k is None:
            continue
        rows.append(
            {
                "outcome": outcome,
                "rel_period": k,
                "beta": float(coefs.loc[name]),
                "se": float(ses.loc[name]) if name in ses.index else float("nan"),
                "spec_id": "within_italy_event_study_3d",
            }
        )
    ref_row = {"outcome": outcome, "rel_period": REF_REL_PERIOD, "beta": 0.0, "se": 0.0, "spec_id": "within_italy_event_study_3d"}
    out = pd.DataFrame(rows)
    if out.empty or REF_REL_PERIOD not in set(out["rel_period"]):
        out = pd.concat([pd.DataFrame([ref_row]), out], ignore_index=True)
    return out.sort_values("rel_period")


def _verdict_from_summary(summary: pd.DataFrame) -> str:
    """Function summary: data-driven one-line verdict from zero-filled headline and LOO rows.

    Parameters:
    - summary: qa_did_summary DataFrame.

    Returns:
    - Short verdict string for notes/README.
    """
    headline = summary[
        (summary["spec_id"] == "within_italy_static")
        & (summary["outcome"] == "n_comments")
        & (summary["estimator"] == "fepois")
    ]
    if headline.empty:
        return "Verdict: inconclusive (headline fepois missing)."
    beta = float(headline.iloc[0]["beta"])
    pval = float(headline.iloc[0]["pvalue"])
    pct = (np.exp(beta) - 1.0) * 100.0 if np.isfinite(beta) else float("nan")

    loo = summary[
        summary["spec_id"].astype(str).str.startswith("within_italy_static_loo")
        & (summary["outcome"] == "n_comments")
        & (summary["estimator"] == "fepois")
    ]
    loo_signs = []
    loo_pvals = []
    for _, row in loo.iterrows():
        b = float(row["beta"])
        if np.isfinite(b):
            loo_signs.append(b > 0)
            loo_pvals.append(float(row["pvalue"]))

    pos_headline = beta > 0
    sig_headline = pval < 0.10
    loo_all_same_sign = len(loo_signs) > 0 and all(s == pos_headline for s in loo_signs)
    loo_any_sig = any(p < 0.10 for p in loo_pvals) if loo_pvals else False

    if not pos_headline and sig_headline and not loo_all_same_sign:
        return (
            f"Verdict: KILL — zero-filled headline shows wrong-signed volume drop "
            f"({pct:.1f}%, p={pval:.3f}); effect attenuates or reverses when dropping "
            f"ItaliaPersonalFinance and/or Universitaly (see LOO rows)."
        )
    if pos_headline and loo_all_same_sign and (sig_headline or loo_any_sig):
        return (
            f"Verdict: survives — zero-filled headline qa:post volume lift "
            f"({pct:+.1f}%, p={pval:.3f}); LOO specs same sign."
        )
    if not sig_headline and not loo_any_sig:
        return (
            f"Verdict: inconclusive — zero-filled headline n_comments fepois "
            f"({pct:+.1f}%, p={pval:.3f}); no LOO row significant at 10%."
        )
    return (
        f"Verdict: mixed — zero-filled headline ({pct:+.1f}%, p={pval:.3f}); "
        f"inspect LOO rows in qa_did_summary.csv."
    )


def write_notes(
    path: Path,
    launch: str,
    lift: str,
    end_excl: str,
    summary: pd.DataFrame,
    phase: pd.DataFrame,
) -> None:
    """Function summary: write short interpretation notes for the Q&A DiD module."""
    lines = [
        "Q&A substitution DiD notes",
        f"Ban onset (launch): {launch}",
        f"Ban lift (first post day): {lift}",
        f"Corpus end (exclusive): {end_excl}",
        "",
        "Headline estimand: qa:post within Italian Q&A vs non-Q&A forums (TWFE subreddit + time).",
        "Panel is zero-filled to full subreddit x day grid (61 days); fepois on counts, log1p OLS check.",
        "Post-lift window is ~3 days (Apr 28-30); treat post-vs-ban contrasts as underpowered.",
        "Volume uses comment counts (RC dumps); not submission-level question posts.",
        "",
        "DATA LIMITATION — DomandeDaReddit:",
        f"  Raw daily_chunks and filter audit show only {DOMANDE_DAREDDIT_RAW_COMMENTS} comments across "
        f"~{DOMANDE_DAREDDIT_RAW_DAYS} active days in Mar-Apr 2023 (forum was active; severe under-collection).",
        "  Existing records match exact subreddit name (case/encoding filter bug ruled out).",
        "  Likely collection/coverage gap in RC dump pass, not name mismatch. Treated Q&A panel biased.",
        "  Re-extraction deferred (external dump not mounted).",
        "",
    ]
    headline = summary[summary["spec_id"] == "within_italy_static"]
    if not headline.empty:
        lines.append("Within-Italy static qa:post coefficients (zero-filled panel):")
        for _, row in headline.iterrows():
            lines.append(
                f"  {row['outcome']} ({row['estimator']}): beta={row['beta']:.4f} se={row['se']:.4f} p={row['pvalue']:.4f}"
            )
    loo = summary[summary["spec_id"].astype(str).str.startswith("within_italy_static_loo")]
    if not loo.empty:
        lines.append("")
        lines.append("Leave-one-out n_comments (drop ItaliaPersonalFinance / Universitaly / both):")
        for _, row in loo.iterrows():
            lines.append(
                f"  {row['spec_id']} ({row['estimator']}): beta={row['beta']:.4f} se={row['se']:.4f} p={row['pvalue']:.4f}"
            )
    lines.append("")
    lines.append(_verdict_from_summary(summary))
    if not phase.empty:
        lines.append("")
        lines.append("Phase means available in qa_phase_contrasts.csv (ban vs pre headline).")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Function summary: run Q&A substitution DiD and write summary tables."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    start, end_excl, launch, lift = event_dates_from_config(config)
    out_dir = tables_subdir(config, "qa_substitution")
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_name = "qa_volume_panel_1d.csv" if args.panel == "1d" else "qa_volume_panel_3d.csv"
    panel_path = out_dir / panel_name
    if not panel_path.is_file():
        raise FileNotFoundError(f"Missing {panel_path}; run prepare_qa_volume_panel.py first.")

    panel = pd.read_csv(panel_path)
    panel = panel[(panel["date_utc"].astype(str) >= start) & (panel["date_utc"].astype(str) < end_excl)]

    summary_rows = estimate_static_specs(panel)
    summary_rows.extend(estimate_loo_specs(panel))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "qa_did_summary.csv", index=False)

    panel_3d_path = out_dir / "qa_volume_panel_3d.csv"
    es_frames: List[pd.DataFrame] = []
    if panel_3d_path.is_file():
        panel_3d = pd.read_csv(panel_3d_path)
        panel_3d = panel_3d[
            (panel_3d["date_utc"].astype(str) >= start) & (panel_3d["date_utc"].astype(str) < end_excl)
        ]
        for outcome in ("n_comments", "question_share", "qmark_rate_100w"):
            es = estimate_event_study(panel_3d, outcome)
            if not es.empty:
                es_frames.append(es)
    if es_frames:
        pd.concat(es_frames, ignore_index=True).to_csv(out_dir / "qa_event_study_3d.csv", index=False)

    within = _within_italy_sample(panel)
    phase = phase_contrast_table(within, group_col="qa")
    phase.to_csv(out_dir / "qa_phase_contrasts.csv", index=False)

    write_notes(out_dir / "qa_did_notes.txt", launch, lift, end_excl, summary, phase)
    print(f"[qa_volume_did] wrote summary to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
