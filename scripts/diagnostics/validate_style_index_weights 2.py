"""
Script summary:
Pick frozen style_index_llm weights via leave-one-out ablation checks on a shard sample.

Functionality:
- Scores theory/interaction candidates (no ρ-tuned side indices).
- Writes style_index_stats.json, ablation_report.csv, candidate_comparison.csv.

How to apply/run:
  .venv/bin/python scripts/diagnostics/validate_style_index_weights.py \
    --config config/italy_polarization_setup.yaml --max-shards 30
  .venv/bin/python scripts/features/compute_style_index_on_shards.py \
    --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

PRE_START = "2023-03-01"
PRE_END = "2023-03-30"
TUNE_LANG = "it"


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

from src.config_utils import load_config, resolve_primary_subreddits, tables_subdir  # noqa: E402
from src.style_index import (  # noqa: E402
    STATS_VERSION,
    comment_feature_dict,
    fit_preperiod_stats,
    save_style_index_stats,
    style_index_stats_filename,
)
from src.style_index_ablation import (  # noqa: E402
    ablation_metric_rows,
    feature_rate_vs_all_ablations_rows,
    feature_rate_vs_own_ablation_rows,
    marginal_influence_rows,
    score_candidate,
)
from src.style_index import SIGNS_V3  # noqa: E402
from src.style_index_llm import (  # noqa: E402
    BUNDLE_LLM,
    LLM_CANDIDATES,
    ONLY_FEATURES,
    PRIMARY_COL,
    ablation_column_name,
    compute_llm_index,
    enrich_interaction_features,
    only_column_name,
)
from src.style_index_validation import prepare_validation_frame  # noqa: E402


def _read_shard(path: Path, columns: Sequence[str]) -> Optional[pd.DataFrame]:
    """Function summary: read projected columns from one Parquet shard."""
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


def _load_frame(config: Dict[str, Any], max_shards: int) -> pd.DataFrame:
    """Function summary: build feature-level frame from shard sample."""
    shard_root = Path(config["paths"]["interim_dir"]) / "cleaned_monthly_chunks"
    rows: List[Dict[str, Any]] = []
    n = 0
    for sub in resolve_primary_subreddits(config):
        for shard in sorted((shard_root / sub).glob("*.parquet")):
            if n >= max_shards:
                break
            df = _read_shard(shard, ("body", "date_utc", "primary_lexicon"))
            if df is None or df.empty:
                continue
            for _, r in df.iterrows():
                feats = comment_feature_dict(
                    str(r.get("body", "")), str(r.get("primary_lexicon", "it")), PROJECT_ROOT
                )
                feats["date_utc"] = str(r.get("date_utc", ""))[:10]
                feats["lang"] = str(r.get("primary_lexicon", "it"))
                rows.append(feats)
            n += 1
        if n >= max_shards:
            break
    if not rows:
        raise FileNotFoundError(f"No shards under {shard_root}")
    return pd.DataFrame(rows)


def _frame_with_interactions(frame: pd.DataFrame, candidate_id: str) -> pd.DataFrame:
    """Function summary: add interaction feature columns for one candidate."""
    cand = LLM_CANDIDATES[candidate_id]
    interactions = list(cand.get("interactions", []))
    out_rows: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        base = {k: row[k] for k in row.index}
        out_rows.append(enrich_interaction_features(base, interactions))
    return pd.DataFrame(out_rows)


def _score_candidate_on_sample(
    frame: pd.DataFrame,
    stats: Dict[str, Any],
    candidate_id: str,
) -> pd.DataFrame:
    """Function summary: compute primary + ablation columns for one candidate."""
    cand = LLM_CANDIDATES[candidate_id]
    weights = dict(cand["weights"])
    interactions = list(cand.get("interactions", []))
    signs = dict(cand.get("signs", SIGNS_V3))
    lang_stats = stats["languages"][TUNE_LANG]
    rows: List[Dict[str, float]] = []
    for _, row in frame.iterrows():
        feats = {k: row[k] for k in row.index if k not in ("date_utc", "lang")}
        feats = enrich_interaction_features(feats, interactions)
        rec: Dict[str, float] = {}
        rec[PRIMARY_COL] = compute_llm_index(
            feats, lang_stats, weights, interactions=interactions, drop=(), signs=signs
        )
        from src.style_index_llm import ABLATION_DROP_FEATURES, ONLY_FEATURES  # noqa: WPS433

        for drop in ABLATION_DROP_FEATURES:
            if drop not in weights and drop not in interactions:
                continue
            rec[ablation_column_name(drop)] = compute_llm_index(
                feats, lang_stats, weights, interactions=interactions, drop=(drop,), signs=signs
            )
        for feat in ONLY_FEATURES:
            rec[only_column_name(feat)] = compute_llm_index(
                feats, lang_stats, {feat: 1.0}, interactions=(), drop=(), signs=signs
            )
        rows.append(rec)
    scored = frame.copy()
    scored = scored.reset_index(drop=True)
    idx_df = pd.DataFrame(rows)
    for col in idx_df.columns:
        scored[col] = idx_df[col].values
    return scored


def parse_args() -> argparse.Namespace:
    """Function summary: CLI."""
    p = argparse.ArgumentParser(description="Validate LLM v3 ablations and pick primary candidate.")
    p.add_argument("--config", default="config/italy_polarization_setup.yaml")
    p.add_argument("--max-shards", type=int, default=80)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    """Function summary: run candidate grid, ablation metrics, write v3 stats JSON."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    frame = _load_frame(config, args.max_shards)
    it_ge20 = frame[
        (frame["date_utc"] >= PRE_START)
        & (frame["date_utc"] <= PRE_END)
        & (frame["lang"].astype(str).str.lower() == TUNE_LANG)
        & (pd.to_numeric(frame["n_words"], errors="coerce") >= 20)
    ].copy()

    out_dir = PROJECT_ROOT / tables_subdir(config, "did")
    val_dir = out_dir / "style_index_validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    # Calibration uses union of base + interaction features.
    cal_frame = _frame_with_interactions(frame, "interact_heavy")
    stats = fit_preperiod_stats(cal_frame, version="v3")
    stats["version"] = STATS_VERSION
    stats["pre_period"] = [PRE_START, PRE_END]

    candidate_rows: List[Dict[str, Any]] = []
    all_ablation_rows: List[Dict[str, Any]] = []
    all_marginal_rows: List[Dict[str, Any]] = []
    best_id = "theory_base"
    best_score = -1e9

    for cid in LLM_CANDIDATES:
        sub = _frame_with_interactions(it_ge20, cid)
        scored = _score_candidate_on_sample(sub, stats, cid)
        prep = prepare_validation_frame(scored)
        signs = dict(cand.get("signs", SIGNS_V3))
        summary = score_candidate(prep, primary_col=PRIMARY_COL, signs=signs)
        summary["candidate_id"] = cid
        summary["description"] = LLM_CANDIDATES[cid]["description"]
        candidate_rows.append(summary)
        all_ablation_rows.extend(
            ablation_metric_rows(prep, subset=f"candidate={cid}", primary_col=PRIMARY_COL)
        )
        all_marginal_rows.extend(
            marginal_influence_rows(prep, subset=f"candidate={cid}", primary_col=PRIMARY_COL)
        )
        eligible = bool(summary.get("pass_own_ablation_nonneg", False))
        if eligible and float(summary["score"]) > best_score:
            best_score = float(summary["score"])
            best_id = cid

    if not any(r.get("pass_own_ablation_nonneg") for r in candidate_rows):
        print(
            "[validate_llm_ablations] warning: no candidate passed nonnegative own-ablation; "
            "falling back to theory_base",
            flush=True,
        )
        tb = next((r for r in candidate_rows if r["candidate_id"] == "theory_base"), None)
        if tb is not None:
            best_id = "theory_base"
            best_score = float(tb["score"])
        else:
            top = max(candidate_rows, key=lambda r: float(r["score"]))
            best_id = str(top["candidate_id"])
            best_score = float(top["score"])

    print(f"[validate_llm_ablations] winner={best_id} score={best_score:.3f}", flush=True)
    pd.DataFrame(candidate_rows).to_csv(val_dir / "llm_candidate_comparison.csv", index=False)
    pd.DataFrame(all_ablation_rows).to_csv(val_dir / "llm_ablation_report.csv", index=False)
    pd.DataFrame(all_marginal_rows).to_csv(val_dir / "llm_marginal_report.csv", index=False)

    winner = LLM_CANDIDATES[best_id]
    win_sub = _frame_with_interactions(it_ge20, best_id)
    win_scored = _score_candidate_on_sample(win_sub, stats, best_id)
    win_prep = prepare_validation_frame(win_scored)
    pd.DataFrame(
        marginal_influence_rows(win_prep, subset=f"winner={best_id}", primary_col=PRIMARY_COL)
    ).to_csv(val_dir / "llm_marginal_winner.csv", index=False)
    pd.DataFrame(
        feature_rate_vs_own_ablation_rows(win_prep, subset=f"winner={best_id}")
    ).to_csv(val_dir / "llm_feature_vs_own_ablation.csv", index=False)
    pd.DataFrame(
        feature_rate_vs_all_ablations_rows(win_prep, subset=f"winner={best_id}")
    ).to_csv(val_dir / "llm_feature_vs_ablation_matrix.csv", index=False)

    stats["primary_candidate"] = best_id
    stats["tune_meta"] = {
        "tune_lang": TUNE_LANG,
        "n_tune": int(len(it_ge20)),
        "selection": "leave_one_out_ablation_score",
        "winner_score": best_score,
    }
    lang_block = dict(stats["languages"].get(TUNE_LANG, {}))
    lang_block["bundles"] = {
        BUNDLE_LLM: {
            "candidate_id": best_id,
            "weights": dict(winner["weights"]),
            "interactions": list(winner.get("interactions", [])),
            "signs": dict(winner.get("signs", SIGNS_V3)),
            "min_features_full": 4,
            "min_features_reduced": 1,
        },
    }
    stats["languages"][TUNE_LANG] = lang_block

    if not args.dry_run:
        save_style_index_stats(stats, out_dir / style_index_stats_filename())
        print(f"[validate_style_index_weights] wrote {out_dir / style_index_stats_filename()}", flush=True)


if __name__ == "__main__":
    main()
