"""
Script summary:
Task 5 validation gates for the formula style index (construct validity, length checks, pretrend).

Functionality:
- Samples all primary-subreddit shards (no forum exclusions).
- Writes gates_summary.csv plus detailed test tables under did/style_index_validation/.
- Does not modify SIGNS, weights, or index scoring (read-only validation).

How to apply/run:
  .venv/bin/python scripts/diagnostics/fit_style_index_stats.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/features/compute_style_index_on_shards.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/validate_style_index_gates.py --config config/italy_polarization_setup.yaml
  .venv/bin/python -m pytest tests/test_style_index_validation.py -q
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def _setup_project_root() -> Path:
    """Function summary: resolve repo root."""
    caller = Path(__file__).resolve()
    for parent in caller.parents:
        if parent.name == "scripts" and (parent / "_bootstrap.py").is_file():
            spec = importlib.util.spec_from_file_location("_mod", parent / "_bootstrap.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.setup_project_path(caller)
    raise RuntimeError("bootstrap missing")


PROJECT_ROOT = _setup_project_root()

from scripts.diagnostics.descriptives_util import event_dates_from_config  # noqa: E402
from src.config_utils import load_config, resolve_primary_subreddits, tables_subdir  # noqa: E402
from src.did.estimate import estimate_pretrend_f  # noqa: E402
from src.did.outcomes import outcome_spec  # noqa: E402
from src.did.panels import load_subreddit_panel  # noqa: E402
from src.did.specs import HEADLINE_BASE_STRATEGIES, filter_strategy_sample  # noqa: E402
from src.style_index_validation import (  # noqa: E402
    INDEX_COL_LLM,
    build_ai_rate_review_sample,
    build_joint_lex_em_review_sample,
    by_language_correlation_rows,
    by_subreddit_summary_rows,
    compare_indices_rows,
    convergence_correlation_rows,
    gate_status_from_metrics,
    joint_signal_bucket_rows,
    length_stratified_ai_rows,
    length_stratified_delta,
    prepare_validation_frame,
    spearman_corr,
)

_ITALY_FAMILIES = frozenset({"it_political", "it_others"})
_CONTROL_FAMILIES = frozenset({"de", "eu", "us", "uk"})


def _annotate_daily_subreddit_for_pretrend(
    panel: pd.DataFrame, launch: str, end_excl: str
) -> pd.DataFrame:
    """Function summary: add rel_day, post, and IT flags needed for pretrend F-test on descriptives CSV.

    Parameters:
    - panel: daily_by_subreddit rows with date_utc and topic_family.
    - launch: ban onset YYYY-MM-DD.
    - end_excl: corpus end (exclusive) YYYY-MM-DD.

    Returns:
    - Annotated panel copy suitable for filter_strategy_sample.
    """
    out = panel.copy()
    launch_dt = pd.Timestamp(launch)
    out["date_utc"] = out["date_utc"].astype(str)
    out["rel_day"] = (pd.to_datetime(out["date_utc"]) - launch_dt).dt.days.astype(int)
    out["post"] = (out["date_utc"].astype(str) >= launch).astype(int)
    fam = out["topic_family"].astype(str)
    out["IT"] = fam.isin(_ITALY_FAMILIES).astype(int)
    out["is_control"] = fam.isin(_CONTROL_FAMILIES).astype(int)
    out = out[(out["date_utc"].astype(str) >= out["date_utc"].min()) & (out["date_utc"].astype(str) < end_excl)]
    return out


def _load_subreddit_panel_for_pretrend(config: Dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """Function summary: load DiD subreddit panel for pretrend, with descriptives fallback.

    Parameters:
    - config: project YAML dict.

    Returns:
    - Tuple of (panel DataFrame, source note: panel|daily_by_subreddit).
    """
    try:
        panel = load_subreddit_panel(config)
        return panel, "panel"
    except FileNotFoundError:
        pass
    start, end_excl, launch, _ = event_dates_from_config(config)
    desc_path = tables_subdir(config, "descriptives") / "daily_by_subreddit.csv"
    if not desc_path.is_file():
        raise FileNotFoundError(f"Missing subreddit panel and {desc_path}")
    sub = pd.read_csv(desc_path)
    sub = sub[(sub["date_utc"].astype(str) >= start) & (sub["date_utc"].astype(str) < end_excl)]
    return _annotate_daily_subreddit_for_pretrend(sub, launch, end_excl), "daily_by_subreddit"


SAMPLE_COLUMNS = [
    "body",
    "date_utc",
    "primary_lexicon",
    "subreddit",
    "topic_family",
    "n_words",
    "style_index_llm",
    "style_index_llm_no_ai_style",
    "style_index_llm_no_em_dash",
    "style_index_llm_no_semicolon_colon",
    "ai_style_rate_100w",
    "log_len",
    "hedging_phrase_hits",
    "em_dash_count",
    "em_dash_extended_count",
    "em_dash_rate_100w",
    "em_dash_any",
    "author",
    "id",
]


def parse_args() -> argparse.Namespace:
    """Function summary: CLI."""
    p = argparse.ArgumentParser(description="Style index validation gates and construct tests.")
    p.add_argument("--config", default="config/italy_polarization_setup.yaml")
    p.add_argument("--max-shards", type=int, default=30, help="Cap shards per subreddit (None=all if 0).")
    p.add_argument("--review-n", type=int, default=20)
    return p.parse_args()


def _read_shard(path: Path, columns: List[str]) -> Optional[pd.DataFrame]:
    """Function summary: read available columns from one Parquet shard."""
    try:
        import pyarrow.parquet as pq  # noqa: WPS433

        avail = [c for c in columns if c in pq.read_schema(path).names]
        if not avail:
            return None
        return pd.read_parquet(path, columns=avail)
    except Exception:
        try:
            df = pd.read_parquet(path)
            keep = [c for c in columns if c in df.columns]
            return df[keep] if keep else None
        except Exception:
            return None


def _sample_comments(config: Dict[str, Any], max_shards: int) -> pd.DataFrame:
    """Function summary: load comment rows from all primary subreddits (no exclusions)."""
    shard_root = Path(config["paths"]["interim_dir"]) / "cleaned_monthly_chunks"
    cap: Optional[int] = max_shards if max_shards > 0 else None
    rows: List[pd.DataFrame] = []
    n = 0
    for sub in resolve_primary_subreddits(config):
        for shard in sorted((shard_root / sub).glob("*.parquet")):
            if cap is not None and n >= cap:
                break
            df = _read_shard(shard, SAMPLE_COLUMNS)
            if df is None or df.empty:
                continue
            if INDEX_COL_LLM not in df.columns:
                continue
            rows.append(df)
            n += 1
        if cap is not None and n >= cap:
            break
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _write_convergence_reports(prep: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Function summary: write convergence_correlations.csv and return combined metric rows."""
    all_rows: List[Dict[str, Any]] = []
    all_rows.extend(convergence_correlation_rows(prep, subset="all"))
    ge20 = prep[prep["n_words"].fillna(0) >= 20]
    all_rows.extend(convergence_correlation_rows(ge20, subset="n_words_ge_20"))
    all_rows.extend(by_language_correlation_rows(prep))
    conv = pd.DataFrame(all_rows)
    conv.to_csv(out_dir / "convergence_correlations.csv", index=False)
    return conv


def _write_extra_reports(prep: pd.DataFrame, out_dir: Path, gate_rows: List[Dict[str, Any]]) -> None:
    """Function summary: joint buckets, ablation compare, and IT subset gate rows."""
    if INDEX_COL_LLM not in prep.columns:
        return
    pd.DataFrame(joint_signal_bucket_rows(prep)).to_csv(
        out_dir / "joint_signal_buckets.csv", index=False
    )
    pd.DataFrame(compare_indices_rows(prep)).to_csv(out_dir / "compare_indices.csv", index=False)

    it = prep[prep["primary_lexicon"].astype(str).str.lower() == "it"] if "primary_lexicon" in prep.columns else prep
    rho_it, n_it = spearman_corr(it[INDEX_COL_LLM], it["ai_style_rate_100w"])
    rho_len_it, _ = spearman_corr(it[INDEX_COL_LLM], it["log_len"])
    delta_it = length_stratified_delta(prep, "20_49", index_col=INDEX_COL_LLM)
    gate_rows.append(
        {
            "gate": "it_spearman_style_index_llm_vs_ai_style_rate",
            "status": "pass" if np.isfinite(rho_it) and rho_it > 0.15 else "review",
            "value": rho_it,
            "n": n_it,
            "note": "IT subset; pass if rho>0.15",
        }
    )
    gate_rows.append(
        {
            "gate": "it_spearman_style_index_llm_vs_log_len",
            "status": "pass" if np.isfinite(rho_len_it) and abs(rho_len_it) < 0.5 else "review",
            "value": rho_len_it,
            "note": "IT subset; pass if |rho|<0.5",
        }
    )
    gate_rows.append(
        {
            "gate": "it_length_stratified_ai_20_49",
            "status": "pass" if np.isfinite(delta_it) and delta_it > 0 else "review",
            "value": delta_it,
        }
    )

    joint_rev = build_joint_lex_em_review_sample(prep, n_each=20)
    if not joint_rev.empty:
        joint_rev.to_csv(out_dir / "review_20plus20_joint_high_lex_em.csv", index=False)
        gate_rows.append(
            {
                "gate": "review_joint_high_lex_em",
                "status": "pass",
                "path": str(out_dir / "review_20plus20_joint_high_lex_em.csv"),
            }
        )


def main() -> None:
    """Function summary: run gates and write validation artifacts."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    out_dir = tables_subdir(config, "did") / "style_index_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = _sample_comments(config, args.max_shards)
    gate_rows: List[Dict[str, Any]] = []

    if raw.empty:
        gate_rows.append({"gate": "data", "status": "fail", "note": "no shards with style_index_llm"})
    else:
        prep = prepare_validation_frame(raw)
        prep.to_csv(out_dir / "validation_sample_comments.csv", index=False)

        conv = _write_convergence_reports(prep, out_dir)
        pd.DataFrame(length_stratified_ai_rows(prep)).to_csv(
            out_dir / "length_stratified_ai.csv", index=False
        )
        pd.DataFrame(by_subreddit_summary_rows(prep)).to_csv(
            out_dir / "subreddit_index_summary.csv", index=False
        )

        rho_all, n_all = spearman_corr(prep[INDEX_COL_LLM], prep["ai_style_rate_100w"])
        ge20 = prep[prep["n_words"].fillna(0) >= 20]
        rho_ge20, n_ge20 = spearman_corr(ge20[INDEX_COL_LLM], ge20["ai_style_rate_100w"])
        rho_len, n_len = spearman_corr(prep[INDEX_COL_LLM], prep["log_len"])
        delta_2049 = length_stratified_delta(prep, "20_49")

        for name, status in gate_status_from_metrics(rho_all, rho_len, delta_2049).items():
            row: Dict[str, Any] = {"gate": name, "status": status}
            if name == "spearman_vs_ai_style_rate_100w":
                row["value"] = rho_all
                row["n"] = n_all
                row["note"] = "heuristic pass if rho>0.3; not required if index is broader than lexicon"
            elif name == "spearman_vs_log_len":
                row["value"] = rho_len
                row["n"] = n_len
                row["note"] = "review if |rho|>0.7 (length-dominated index)"
            elif name == "length_stratified_ai_20_49":
                row["value"] = delta_2049
                row["note"] = "mean index(ai_hit=1) - mean index(ai_hit=0) in 20-49 word bin"
            gate_rows.append(row)

        gate_rows.append(
            {
                "gate": "spearman_vs_ai_style_rate_100w_n_words_ge_20",
                "status": "info",
                "value": rho_ge20,
                "n": n_ge20,
                "note": "same heuristic 0.3 optional",
            }
        )

        if plt is not None:
            si = prep[INDEX_COL_LLM].dropna()
            if len(si) >= 50:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(si, bins=50, color="steelblue", edgecolor="white")
                ax.set_title("style_index_llm (sampled comments, all forums)")
                fig.savefig(out_dir / "hist_style_index_llm.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
                gate_rows.append(
                    {
                        "gate": "histogram",
                        "status": "pass",
                        "path": str(out_dir / "hist_style_index_llm.png"),
                    }
                )
            ai_pos = prep[(prep["ai_style_rate_100w"] > 0) & prep[INDEX_COL_LLM].notna()]
            if len(ai_pos) >= 30 and len(si) >= 30:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.scatter(
                    prep.loc[ai_pos.index, "ai_style_rate_100w"],
                    prep.loc[ai_pos.index, INDEX_COL_LLM],
                    alpha=0.15,
                    s=8,
                )
                ax.set_xlabel("ai_style_rate_100w")
                ax.set_ylabel("style_index_llm")
                ax.set_title("Index vs lexicon AI rate (all forums)")
                fig.savefig(out_dir / "scatter_index_vs_ai_style.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
                gate_rows.append(
                    {
                        "gate": "scatter_index_vs_ai_style",
                        "status": "pass",
                        "path": str(out_dir / "scatter_index_vs_ai_style.png"),
                    }
                )

        review = prep.dropna(subset=[INDEX_COL_LLM]).sample(
            n=min(args.review_n * 2, len(prep)), random_state=42
        )
        hi = review.nlargest(args.review_n, INDEX_COL_LLM)
        lo = review.nsmallest(args.review_n, INDEX_COL_LLM)
        pd.concat([hi.assign(review_bucket="high_index"), lo.assign(review_bucket="low_index")]).to_csv(
            out_dir / "review_20plus20_by_index.csv", index=False
        )
        build_ai_rate_review_sample(prep, n_each=args.review_n).to_csv(
            out_dir / "review_20plus20_by_ai_rate.csv", index=False
        )
        _write_extra_reports(prep, out_dir, gate_rows)
        gate_rows.append(
            {
                "gate": "review_by_index",
                "status": "pass",
                "path": str(out_dir / "review_20plus20_by_index.csv"),
            }
        )
        gate_rows.append(
            {
                "gate": "review_by_ai_rate",
                "status": "pass",
                "path": str(out_dir / "review_20plus20_by_ai_rate.csv"),
                "note": "preferred manual read for lexicon alignment",
            }
        )

        readme = out_dir / "README_validation_tests.txt"
        readme.write_text(
            "\n".join(
                [
                    "Style index validation outputs (read-only; SIGNS unchanged).",
                    "",
                    "gates_summary.csv — headline pass/review flags.",
                    "convergence_correlations.csv — Spearman/Pearson vs ai_rate, log_len, partial ai|length; by subset and language.",
                    "length_stratified_ai.csv — mean index by ai_hit within length bins (all forums).",
                    "subreddit_index_summary.csv — descriptive means by subreddit (not an exclusion list).",
                    "review_20plus20_by_index.csv — global top/bottom index (can be length-driven).",
                    "review_20plus20_by_ai_rate.csv — high vs low lexicon ai_style_rate (>=20 words).",
                    "joint_signal_buckets.csv — mean style_index_llm by ai_hit x em_dash_hit x length_bin.",
                    "compare_indices.csv — pairwise Spearman across primary and LOO ablations.",
                    "review_20plus20_joint_high_lex_em.csv — high lex + em dash in 20-49 word bin.",
                    "validation_sample_comments.csv — sampled comments used for metrics.",
                    "",
                    "Interpretation:",
                    "- ai_style_rate_100w = 100 * ai_style lexicon hits / n_words (not word count).",
                    "- rho>0.3 vs ai_rate is a heuristic only if index should track lexicon hits.",
                    "- Use length_stratified_ai and partial ai|log_len to detect length confounds.",
                ]
            ),
            encoding="utf-8",
        )

    try:
        panel, panel_source = _load_subreddit_panel_for_pretrend(config)
        strat = next(s for s in HEADLINE_BASE_STRATEGIES if s.strategy_id == "cross_country_all")
        work = filter_strategy_sample(panel, strat)
        if "entity_id" not in work.columns:
            work["entity_id"] = work["subreddit"].astype(str)
        if "time_id" not in work.columns:
            work["time_id"] = work["date_utc"].astype(str)
        oc = outcome_spec("style_index_llm")
        if oc is None:
            gate_rows.append({"gate": "pretrend_F", "status": "skip", "note": "unknown outcome style_index_llm"})
        elif oc.column not in work.columns:
            if panel_source == "panel":
                start, end_excl, launch, _ = event_dates_from_config(config)
                desc_path = tables_subdir(config, "descriptives") / "daily_by_subreddit.csv"
                if desc_path.is_file():
                    sub = pd.read_csv(desc_path)
                    sub = sub[
                        (sub["date_utc"].astype(str) >= start) & (sub["date_utc"].astype(str) < end_excl)
                    ]
                    panel_fb = _annotate_daily_subreddit_for_pretrend(sub, launch, end_excl)
                    work = filter_strategy_sample(panel_fb, strat)
                    if "entity_id" not in work.columns:
                        work["entity_id"] = work["subreddit"].astype(str)
                    if "time_id" not in work.columns:
                        work["time_id"] = work["date_utc"].astype(str)
                    panel_source = "daily_by_subreddit"
            if oc.column not in work.columns:
                gate_rows.append(
                    {
                        "gate": "pretrend_F_style_index_llm",
                        "status": "skip",
                        "note": f"missing {oc.column} — rebuild descriptives/panel",
                    }
                )
            else:
                fp, _note = estimate_pretrend_f(work, oc.column)
                gate_rows.append(
                    {
                        "gate": "pretrend_F_style_index_llm",
                        "status": "pass" if np.isfinite(fp) and fp > 0.05 else "review",
                        "pretrend_F_p": fp,
                        "panel_source": panel_source,
                    }
                )
        else:
            fp, _note = estimate_pretrend_f(work, oc.column)
            gate_rows.append(
                {
                    "gate": "pretrend_F_style_index_llm",
                    "status": "pass" if np.isfinite(fp) and fp > 0.05 else "review",
                    "pretrend_F_p": fp,
                    "panel_source": panel_source,
                }
            )
    except FileNotFoundError as exc:
        gate_rows.append({"gate": "pretrend_F", "status": "skip", "note": str(exc)})

    summary = pd.DataFrame(gate_rows)
    summary.to_csv(out_dir / "gates_summary.csv", index=False)
    print(f"[validate_style_index_gates] wrote {out_dir}", flush=True)
    print(summary.to_string(index=False), flush=True)
    if not raw.empty:
        print(
            f"[validate_style_index_gates] detail tables: convergence_correlations, "
            f"length_stratified_ai, review_20plus20_by_ai_rate",
            flush=True,
        )


if __name__ == "__main__":
    main()
