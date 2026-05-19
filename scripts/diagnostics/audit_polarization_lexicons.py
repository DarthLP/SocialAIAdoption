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
    ("spain", "es"),
]


def _resolve_project_root() -> Path:
    """Function summary: load scripts/_project_root.py and return repository root Path."""
    scripts_dir = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_socialai_scripts_project_root_mod", scripts_dir / "_project_root.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/_project_root.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.project_root()


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_config  # noqa: E402
from src.political_lexicon import (  # noqa: E402
    LEXICON_NAMES,
    get_categorized_lexicon,
    lexicon_path,
    political_rate_100w,
    score_comment_polarization,
)


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Audit polarization lexicons.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--seed", type=int, default=20260519)
    return parser.parse_args()


def term_counts_table(project_root: Path) -> pd.DataFrame:
    """Function summary: count terms per lexicon file, category, and language.

    Parameters:
    - project_root: repo root.

    Returns:
    - Summary dataframe.
    """
    rows: List[Dict[str, Any]] = []
    for lang in ("it", "en", "de", "es"):
        for name in LEXICON_NAMES:
            lex = get_categorized_lexicon(project_root, lang, name)
            for cat, (singles, phrases) in lex.items():
                rows.append(
                    {
                        "lang": lang,
                        "lexicon": name,
                        "category": cat,
                        "n_single_tokens": len(singles),
                        "n_phrases": len(phrases),
                        "path": str(lexicon_path(project_root, lang, name)),
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

    term_counts_table(PROJECT_ROOT).to_csv(out_dir / "lexicon_audit_term_counts.csv", index=False)
    benchmark_rates(interim_dir, PROJECT_ROOT).to_csv(out_dir / "lexicon_audit_benchmark_rates.csv", index=False)
    hit_samples(interim_dir, PROJECT_ROOT, args.seed).to_csv(out_dir / "lexicon_audit_hit_samples.csv", index=False)
    labels_path = out_dir / "lexicon_validation_labels.csv"
    pr = validation_pr(labels_path, PROJECT_ROOT)
    if not pr.empty:
        pr.to_csv(out_dir / "lexicon_validation_pr.csv", index=False)
    print(f"[audit_polarization_lexicons] wrote tables to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
