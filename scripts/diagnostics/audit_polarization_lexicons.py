"""
Script summary:
Audit polarization lexicons: term counts, benchmark rates, hit samples, precision/recall vs hand labels.

Functionality:
- Summarizes categorized lexicon sizes per language.
- Reports word-weighted rates on benchmark subreddits.
- Computes precision/recall when lexicon_validation_labels.csv is populated.

How to apply/run:
  .venv/bin/python scripts/diagnostics/audit_polarization_lexicons.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

BENCHMARKS = [
    ("politicaITA", "it"),
    ("litigi", "it"),
    ("Italia", "it"),
    ("Ask_Politics", "en"),
    ("de", "de"),
]



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

from src.config_utils import (  # noqa: E402
    emotion_cognition_parallel_path,
    load_config,
    polarization_lexicon_parallel_path,
)
from src.political_lexicon import (  # noqa: E402
    LEXICON_NAMES,
    get_categorized_lexicon,
    political_rate_100w,
    score_comment_polarization,
)
from src.parallel_lexicon import load_emotion_cognition_parallel  # noqa: E402
from src.v4_lexicon import get_pairs_registry, pairs_registry_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Audit polarization lexicons.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--seed", type=int, default=20260519)
    return parser.parse_args()


def term_counts_table(project_root: Path, config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: count terms per lexicon file, category, and language.

    Parameters:
    - project_root: repo root.
    - config: study YAML.

    Returns:
    - Summary dataframe.
    """
    pol_path = polarization_lexicon_parallel_path(config, project_root)
    rows: List[Dict[str, Any]] = []
    for lang in ("it", "en", "de"):
        for name in LEXICON_NAMES:
            lex = get_categorized_lexicon(project_root, lang, name, polarization_csv_path=pol_path)
            for cat, (singles, phrases) in lex.items():
                rows.append(
                    {
                        "lang": lang,
                        "lexicon": name,
                        "category": cat,
                        "n_single_tokens": len(singles),
                        "n_phrases": len(phrases),
                        "path": str(pol_path),
                    }
                )
    emo_path = emotion_cognition_parallel_path(config, project_root)
    for lang in ("it", "en", "de"):
        emo = load_emotion_cognition_parallel(emo_path, lang)
        for pole, (singles, phrases) in emo.items():
            rows.append(
                {
                    "lang": lang,
                    "lexicon": pole,
                    "category": pole,
                    "n_single_tokens": len(singles),
                    "n_phrases": len(phrases),
                    "path": str(emo_path),
                }
            )
    rows.append(
        {
            "lang": "it",
            "lexicon": "pairs",
            "category": "pairs",
            "n_single_tokens": len(get_pairs_registry(project_root)),
            "n_phrases": 0,
            "path": str(pairs_registry_path(project_root)),
        }
    )
    return pd.DataFrame(rows)


def benchmark_rates(interim_dir: Path, project_root: Path) -> pd.DataFrame:
    """Function summary: word-weighted ideology and salience rates on benchmark subs.

    Parameters:
    - interim_dir: interim data root.
    - project_root: repo root.

    Returns:
    - Benchmark summary dataframe.
    """
    rows: List[Dict[str, Any]] = []
    for subreddit, lang in BENCHMARKS:
        shard_dir = interim_dir / "cleaned_monthly_chunks" / subreddit
        if not shard_dir.is_dir():
            continue
        total_hits = {"left": 0, "right": 0, "other_side": 0, "aggression": 0}
        total_words = 0
        n_comments = 0
        for shard in sorted(shard_dir.glob("*.parquet")):
            df = pd.read_parquet(shard, columns=["body"])
            for body in df["body"].astype(str).tolist():
                scored = score_comment_polarization(body, lang, project_root)
                nw = int(scored.get("n_words", 0))
                if nw <= 0:
                    continue
                total_words += nw
                n_comments += 1
                total_hits["left"] += int(scored.get("left_hits", 0))
                total_hits["right"] += int(scored.get("right_hits", 0))
                total_hits["other_side"] += int(scored.get("other_side_salience_hits", 0))
                total_hits["aggression"] += int(scored.get("aggression_hits", 0))
        rows.append(
            {
                "subreddit": subreddit,
                "lang": lang,
                "n_comments": n_comments,
                "n_words": total_words,
                "left_rate_100w": political_rate_100w(total_hits["left"], total_words),
                "right_rate_100w": political_rate_100w(total_hits["right"], total_words),
                "other_side_salience_rate_100w": political_rate_100w(total_hits["other_side"], total_words),
                "aggression_rate_100w": political_rate_100w(total_hits["aggression"], total_words),
            }
        )
    return pd.DataFrame(rows)


def hit_samples(interim_dir: Path, project_root: Path, seed: int) -> pd.DataFrame:
    """Function summary: sample example comments with ideology hits per benchmark.

    Parameters:
    - interim_dir: interim root.
    - project_root: repo root.
    - seed: RNG seed.

    Returns:
    - Sample rows dataframe.
    """
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for subreddit, lang in BENCHMARKS[:3]:
        shard_dir = interim_dir / "cleaned_monthly_chunks" / subreddit
        if not shard_dir.is_dir():
            continue
        bodies: List[str] = []
        for shard in sorted(shard_dir.glob("*.parquet")):
            df = pd.read_parquet(shard, columns=["body"])
            bodies.extend(df["body"].astype(str).tolist())
        hits = [
            b
            for b in bodies
            if score_comment_polarization(b, lang, project_root).get("left_hits", 0)
            + score_comment_polarization(b, lang, project_root).get("right_hits", 0)
            > 0
        ]
        for body in rng.sample(hits, min(5, len(hits))):
            rows.append({"subreddit": subreddit, "lang": lang, "body_snippet": body[:240]})
    return pd.DataFrame(rows)


def validation_pr(labels_path: Path, project_root: Path) -> pd.DataFrame:
    """Function summary: precision/recall for hand-labeled comments when labels exist.

    Parameters:
    - labels_path: validation CSV path.
    - project_root: repo root.

    Returns:
    - PR metrics dataframe (empty if no labels).
    """
    if not labels_path.is_file():
        return pd.DataFrame()
    labels = pd.read_csv(labels_path)
    if labels.empty or "id" not in labels.columns:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for _, lab in labels.iterrows():
        if pd.isna(lab.get("label_left")) and pd.isna(lab.get("label_aggression")):
            continue
        lang = str(lab.get("lang", "it"))
        body = str(lab.get("body", ""))
        if not body or body == "nan":
            continue
        scored = score_comment_polarization(body, lang, project_root)
        pred_left = scored.get("left_hits", 0) > 0
        pred_agg = scored.get("aggression_hits", 0) > 0
        if not pd.isna(lab.get("label_left")):
            rows.append(
                {
                    "metric": "ideology_left",
                    "lang": lang,
                    "pred": pred_left,
                    "true": bool(int(lab["label_left"])),
                }
            )
        if not pd.isna(lab.get("label_aggression")):
            rows.append(
                {
                    "metric": "aggression",
                    "lang": lang,
                    "pred": pred_agg,
                    "true": bool(int(lab["label_aggression"])),
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out: List[Dict[str, Any]] = []
    for (metric, lang), grp in df.groupby(["metric", "lang"]):
        tp = int(((grp["pred"]) & (grp["true"])).sum())
        fp = int(((grp["pred"]) & (~grp["true"])).sum())
        fn = int(((~grp["pred"]) & (grp["true"])).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        out.append(
            {
                "metric": metric,
                "lang": lang,
                "precision": prec,
                "recall": rec,
                "n_labels": len(grp),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    """Function summary: write lexicon audit tables."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["tables_dir"]) / "descriptives"
    out_dir.mkdir(parents=True, exist_ok=True)

    term_counts_table(PROJECT_ROOT, config).to_csv(out_dir / "lexicon_audit_term_counts.csv", index=False)
    benchmark_rates(interim_dir, PROJECT_ROOT).to_csv(out_dir / "lexicon_audit_benchmark_rates.csv", index=False)
    hit_samples(interim_dir, PROJECT_ROOT, args.seed).to_csv(out_dir / "lexicon_audit_hit_samples.csv", index=False)
    labels_path = out_dir / "lexicon_validation_labels.csv"
    pr = validation_pr(labels_path, PROJECT_ROOT)
    if not pr.empty:
        pr.to_csv(out_dir / "lexicon_validation_pr.csv", index=False)
    print(f"[audit_polarization_lexicons] wrote tables to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
