"""
Script summary:
Thesis first-stage event-study figure: ±30-day window, 3-day bins, cross_country_all sample.

Functionality:
- Loads canonical did_subreddit_panel_1d.csv (117 forums, not _exbantopic).
- Validates static full-ban TWFE (treat×post | subreddit + time_id) against probed targets.
- Estimates pyfixest event studies: y ~ i(rel_bin, IT, ref=-1) | subreddit + time_id.
- Writes two-panel figure (ai_style_rate, style_index_llm) with ATT and pre-trend annotations.

How to apply/run:
  .venv/bin/python scripts/analysis/plot_first_stage_eventstudy.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BIN_DAYS = 3
ES_WINDOW = 30
REF_BIN = -1
LIFT_DAY = 28
EXPECTED_N_SUBREDDITS = 117
CONTROL_FAMILIES = {"de", "eu", "uk", "us"}

VALIDATION_TARGETS: Dict[str, Dict[str, float]] = {
    "ai_style_rate_100w_mean": {"beta": 0.0013, "p": 0.89},
    "style_index_llm_mean": {"beta": 0.0037, "p": 0.77},
}
BETA_TOL = 0.001
P_TOL = 0.10

OUTCOMES: Tuple[Tuple[str, str], ...] = (
    ("ai_style_rate", "ai_style_rate_100w_mean"),
    ("style_index_llm", "style_index_llm_mean"),
)


def _setup_project_root() -> Path:
    """Function summary: resolve repo root via scripts/_bootstrap.py."""
    caller = Path(__file__).resolve()
    for parent in caller.parents:
        if parent.name == "scripts" and (parent / "_bootstrap.py").is_file():
            spec = importlib.util.spec_from_file_location("_mod", parent / "_bootstrap.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.setup_project_path(caller)
    raise RuntimeError("bootstrap missing")


PROJECT_ROOT = _setup_project_root()

from src.config_utils import figures_subdir, load_config  # noqa: E402
from src.did.bucket_estimate import _rel_period_from_coef_name  # noqa: E402
from src.did.panels import load_subreddit_panel  # noqa: E402
from src.did.paths import did_panels_dir  # noqa: E402
from src.did.specs import StrategySpec, filter_strategy_sample  # noqa: E402
from src.plotting.thesis_theme import (  # noqa: E402
    THESIS_COEF_MARKER,
    shade_ban_window,
    thesis_title_for_outcome,
    xlabel_event_study,
    ylabel_italy_bin_coefficient,
)


@dataclass
class OutcomeResult:
    """Function summary: static ATT, event-study coefs, and pretrend p for one outcome."""

    outcome_id: str
    y_col: str
    static_beta: float
    static_p: float
    pretrend_p: float
    es_df: pd.DataFrame


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for first-stage event-study figure."""
    parser = argparse.ArgumentParser(
        description="Thesis first-stage ±30d / 3d-bin event study figure."
    )
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Override output PNG path (default: did/first_stage/first_stage_eventstudy_3d.png).",
    )
    return parser.parse_args()


def _load_cross_country_sample(config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: load canonical panel and filter to cross_country_all sample.

    Parameters:
    - config: project YAML dict.

    Returns:
    - Filtered subreddit-day panel with time_id set.

    Raises:
    - ValueError: if panel path looks exbantopic or forum count differs from expected.
    """
    panel_path = did_panels_dir(config, "subreddit") / "did_subreddit_panel_1d.csv"
    if "exbantopic" in str(panel_path):
        raise ValueError(f"Refusing exbantopic panel path: {panel_path}")
    panel = load_subreddit_panel(config)
    panel["time_id"] = panel["date_utc"].astype(str)
    sample = filter_strategy_sample(panel, StrategySpec("cross_country_all"), window_days=None)
    n_sub = int(sample["subreddit"].nunique())
    if n_sub != EXPECTED_N_SUBREDDITS:
        raise ValueError(
            f"Expected {EXPECTED_N_SUBREDDITS} subreddits on canonical panel, got {n_sub}"
        )
    fams = set(sample["topic_family"].astype(str).unique())
    if not CONTROL_FAMILIES.issubset(fams):
        missing = CONTROL_FAMILIES - fams
        raise ValueError(f"Missing control families in sample: {missing}")
    return sample


def _prep_regression_frame(sample: pd.DataFrame, y_col: str) -> pd.DataFrame:
    """Function summary: build estimation frame with y, treat_post, IT, rel_day.

    Parameters:
    - sample: cross_country_all filtered panel.
    - y_col: outcome column name.

    Returns:
    - Copy with numeric y and interaction terms; rows with missing y dropped.
    """
    work = sample.copy()
    work["y"] = pd.to_numeric(work[y_col], errors="coerce")
    work["treat_post"] = work["treat"].astype(float) * work["post"].astype(float)
    work["IT"] = work["IT"].astype(float)
    work["rel_day"] = pd.to_numeric(work["rel_day"], errors="coerce")
    return work.dropna(subset=["y"])


def _estimate_static_att(sample: pd.DataFrame, y_col: str) -> Tuple[float, float]:
    """Function summary: full-ban TWFE ATT via pyfixest on full sample.

    Parameters:
    - sample: cross_country_all panel.
    - y_col: outcome column.

    Returns:
    - Tuple (beta, p-value) for treat_post coefficient.
    """
    try:
        from pyfixest.estimation import feols
    except ImportError as exc:
        raise RuntimeError("pyfixest is required for first-stage event study") from exc
    work = _prep_regression_frame(sample, y_col)
    if len(work) < 30:
        return float("nan"), float("nan")
    fit = feols(
        "y ~ treat_post | subreddit + time_id",
        data=work,
        vcov={"CRV1": "subreddit"},
    )
    beta = float(fit.coef()["treat_post"])
    se = float(fit.se()["treat_post"])
    p = float(2 * (1 - stats.norm.cdf(abs(beta / se)))) if se > 0 else float("nan")
    return beta, p


def _pretrend_wald_p(fit: Any, rel_col: str = "rel_bin", ref: int = REF_BIN) -> float:
    """Function summary: joint Wald p-value that all pre-reference bin coefs equal zero.

    Parameters:
    - fit: pyfixest Feols fit from binned event study.
    - rel_col: event-time bin column name in coef labels.
    - ref: omitted reference bin.

    Returns:
    - Wald test p-value, or NaN if no pre-period coefs or test fails.
    """
    coefs = fit.coef()
    pre: List[str] = []
    for name in coefs.index:
        k = _rel_period_from_coef_name(str(name), rel_col)
        if k is not None and k < ref:
            pre.append(str(name))
    if not pre:
        return float("nan")
    all_coef = list(coefs.index)
    r_mat = np.zeros((len(pre), len(all_coef)))
    for i, c in enumerate(pre):
        r_mat[i, all_coef.index(c)] = 1.0
    try:
        wt = fit.wald_test(R=r_mat)
        return float(wt["pvalue"]) if hasattr(wt, "__getitem__") else float(wt.pvalue)
    except Exception:
        return float("nan")


def _coefs_to_plot_df(
    fit: Any,
    rel_col: str = "rel_bin",
    ref: int = REF_BIN,
    bin_days: int = BIN_DAYS,
) -> pd.DataFrame:
    """Function summary: parse pyfixest ES coefs into plot-ready DataFrame.

    Parameters:
    - fit: pyfixest Feols fit.
    - rel_col: bin column name in coef labels.
    - ref: reference bin (coef forced to 0).
    - bin_days: days per bin for midpoint calculation.

    Returns:
    - DataFrame with rel_bin, rel_day_mid, beta, se columns sorted by rel_bin.
    """
    rows: List[Dict[str, float]] = []
    coefs = fit.coef()
    se_frame = fit.se()
    for name in coefs.index:
        k = _rel_period_from_coef_name(str(name), rel_col)
        if k is None or k == ref:
            continue
        b = float(coefs.loc[name])
        se = float(se_frame.loc[name]) if name in se_frame.index else float("nan")
        rows.append(
            {
                "rel_bin": float(k),
                "rel_day_mid": float(k * bin_days + 1),
                "beta": b,
                "se": se,
            }
        )
    rows.append(
        {
            "rel_bin": float(ref),
            "rel_day_mid": float(ref * bin_days + 1),
            "beta": 0.0,
            "se": 0.0,
        }
    )
    return pd.DataFrame(rows).sort_values("rel_bin").reset_index(drop=True)


def _estimate_binned_es(sample: pd.DataFrame, y_col: str) -> Tuple[Any, pd.DataFrame, float]:
    """Function summary: 3-day binned event study on ±30 rel_day window.

    Parameters:
    - sample: cross_country_all panel.
    - y_col: outcome column.

    Returns:
    - Tuple (fit object, plot DataFrame, pretrend Wald p-value).
    """
    try:
        from pyfixest.estimation import feols
    except ImportError as exc:
        raise RuntimeError("pyfixest is required for first-stage event study") from exc
    work = _prep_regression_frame(sample, y_col)
    work = work[work["rel_day"].between(-ES_WINDOW, ES_WINDOW)].copy()
    work["rel_bin"] = (work["rel_day"] // BIN_DAYS).astype(int)
    if len(work) < 30 or work["IT"].nunique() < 2:
        return None, pd.DataFrame(), float("nan")
    fit = feols(
        f"y ~ i(rel_bin, IT, ref={REF_BIN}) | subreddit + time_id",
        data=work,
        vcov={"CRV1": "subreddit"},
    )
    pretrend_p = _pretrend_wald_p(fit, rel_col="rel_bin", ref=REF_BIN)
    es_df = _coefs_to_plot_df(fit, rel_col="rel_bin", ref=REF_BIN, bin_days=BIN_DAYS)
    return fit, es_df, pretrend_p


def _validate_static(results: List[OutcomeResult]) -> Tuple[bool, str]:
    """Function summary: check static ATT estimates against probed guardrail targets.

    Parameters:
    - results: per-outcome estimation results.

    Returns:
    - Tuple (passed, message). Message lists deviations when failed.
    """
    lines: List[str] = []
    ok = True
    for res in results:
        target = VALIDATION_TARGETS.get(res.y_col)
        if target is None:
            continue
        d_beta = abs(res.static_beta - target["beta"])
        d_p = abs(res.static_p - target["p"])
        if d_beta > BETA_TOL or d_p > P_TOL:
            ok = False
            lines.append(
                f"  {res.y_col}: beta={res.static_beta:.6f} (target {target['beta']:.4f}, "
                f"Δ={d_beta:.6f}); p={res.static_p:.4f} (target {target['p']:.2f}, Δ={d_p:.4f})"
            )
    if ok:
        return True, "Static validation passed."
    header = "Static validation FAILED — recompute deviates from canonical-panel targets:\n"
    return False, header + "\n".join(lines)


def _plot_two_panel(results: List[OutcomeResult], out_path: Path) -> None:
    """Function summary: save two-panel event-study figure with ATT subtitles.

    Parameters:
    - results: per-outcome estimation results with es_df populated.
    - out_path: destination PNG path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, res in zip(axes, results):
        plot_df = res.es_df.sort_values("rel_day_mid")
        mask = plot_df["se"] > 0
        if mask.any():
            ax.errorbar(
                plot_df.loc[mask, "rel_day_mid"],
                plot_df.loc[mask, "beta"],
                yerr=1.96 * plot_df.loc[mask, "se"],
                fmt="o",
                color=THESIS_COEF_MARKER,
                ecolor=THESIS_COEF_MARKER,
                elinewidth=0.9,
                capsize=3,
                markersize=5,
                markerfacecolor="white",
                markeredgecolor=THESIS_COEF_MARKER,
                zorder=6,
            )
        ref = plot_df[plot_df["rel_bin"] == REF_BIN]
        if not ref.empty:
            ax.plot(
                ref["rel_day_mid"],
                ref["beta"],
                "o",
                color=THESIS_COEF_MARKER,
                markersize=5,
                markerfacecolor="white",
                markeredgecolor=THESIS_COEF_MARKER,
                zorder=6,
            )
        ax.axhline(0, color="gray", linewidth=0.8, zorder=4)
        shade_ban_window(ax, mode="event_study", bin_days=BIN_DAYS, x_scale="days", zorder=0)
        ax.set_title(thesis_title_for_outcome(res.outcome_id))
        ax.set_xlabel(xlabel_event_study(BIN_DAYS))
        ax.set_ylabel(ylabel_italy_bin_coefficient())
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(config: Dict[str, Any], out_path: Path) -> List[OutcomeResult]:
    """Function summary: estimate outcomes, validate static ATT, write figure.

    Parameters:
    - config: project YAML dict.
    - out_path: destination PNG path.

    Returns:
    - List of OutcomeResult for each panel.

    Raises:
    - SystemExit: if static validation fails (exit code 1).
    """
    sample = _load_cross_country_sample(config)
    results: List[OutcomeResult] = []
    for outcome_id, y_col in OUTCOMES:
        static_beta, static_p = _estimate_static_att(sample, y_col)
        _, es_df, pretrend_p = _estimate_binned_es(sample, y_col)
        results.append(
            OutcomeResult(
                outcome_id=outcome_id,
                y_col=y_col,
                static_beta=static_beta,
                static_p=static_p,
                pretrend_p=pretrend_p,
                es_df=es_df,
            )
        )
    passed, msg = _validate_static(results)
    print(msg)
    for res in results:
        print(
            f"  {res.outcome_id}: static beta={res.static_beta:.6f} p={res.static_p:.4f}; "
            f"pretrend p={res.pretrend_p:.4f}"
        )
    if not passed:
        sys.exit(1)
    _plot_two_panel(results, out_path)
    print(f"Wrote {out_path}")
    return results


def main() -> None:
    """Function summary: CLI entry point."""
    args = parse_args()
    config = load_config(args.config)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = figures_subdir(config, "did") / "first_stage" / "first_stage_eventstudy_3d.png"
    run(config, out_path)


if __name__ == "__main__":
    main()
