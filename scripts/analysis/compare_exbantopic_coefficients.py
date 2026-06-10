"""
Script summary:
Compare baseline DiD coefficients to ban-topic-excluded (exbantopic) estimates.

Functionality:
- Reads estimates/summary/did_summary.csv and estimates_exbantopic/summary/did_summary.csv.
- Filters key outcomes × cross_country_all × {full_ban, early_ban_7d, post_first_2bd}.
- Writes combined CSV and a short markdown interpretation note.

How to apply/run:
  .venv/bin/python scripts/analysis/did_event_study.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/analysis/did_event_study.py --config config/italy_polarization_setup.yaml \\
    --exclude-ban-topic --families lexical,semantic_axis --no-figures
  .venv/bin/python scripts/analysis/compare_exbantopic_coefficients.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

KEY_OUTCOMES = (
    "sem_axis_emotion",
    "emotion_rate",
    "cognition_rate",
    "ai_style_rate",
    "style_index_llm",
    "pole_share",
    "sem_axis_ideology",
)

KEY_STRATEGY = "cross_country_all"

KEY_SPECS = ("full_ban", "early_ban_7d", "post_first_2bd")
LEXICAL_FORUM_OUTCOMES = ("emotion_rate", "cognition_rate")


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

from src.config_utils import load_config  # noqa: E402
from src.did.paths import did_outcome_table_path, did_root, did_summary_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compare baseline vs ban-topic-excluded DiD coefficients."
    )
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--markdown-out",
        type=str,
        default="docs/exbantopic_comparison.md",
        help="Relative path under repo root for markdown note.",
    )
    return parser.parse_args()


def _load_summary(path: Path, label: str) -> pd.DataFrame:
    """Function summary: load did_summary.csv or raise with clear message."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label} summary at {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty {label} summary at {path}")
    return df


def _filter_key_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: restrict to headline outcomes, strategy, and post specs."""
    work = df.copy()
    work = work[work["outcome_id"].astype(str).isin(KEY_OUTCOMES)]
    work = work[work["strategy_id"].astype(str) == KEY_STRATEGY]
    work = work[work["spec"].astype(str).isin(KEY_SPECS)]
    if "weights" in work.columns:
        work = work[work["weights"].astype(str).fillna("") == ""]
    # Subreddit-day forum rows only (comment-level shares outcome_id in did_summary).
    if "outcome_family" in work.columns:
        work = work[work["outcome_family"].astype(str).isin({"lexical", "semantic_axis"})]
    return work


def _patch_baseline_lexical_forum(config: Dict[str, Any], baseline: pd.DataFrame) -> pd.DataFrame:
    """Function summary: restore forum lexical rows when comment-level runs overwrote did_summary."""
    patched = baseline.copy()
    for outcome_id in LEXICAL_FORUM_OUTCOMES:
        path = did_outcome_table_path(config, "lexical", "coefficients", outcome_id)
        if not path.is_file():
            continue
        coef = pd.read_csv(path)
        coef = coef[
            (coef["outcome_id"].astype(str) == outcome_id)
            & (coef["outcome_family"].astype(str) == "lexical")
            & (coef["strategy_id"].astype(str) == KEY_STRATEGY)
            & (coef["spec"].astype(str).isin(KEY_SPECS))
        ]
        if coef.empty:
            continue
        drop = patched["outcome_id"].astype(str) == outcome_id
        patched = patched[~drop]
        patched = pd.concat([patched, coef], ignore_index=True)
    return patched


def _pick_columns(df: pd.DataFrame, sample: str) -> pd.DataFrame:
    """Function summary: select comparison columns with sample prefix."""
    cols = [
        "outcome_id",
        "outcome_family",
        "strategy_id",
        "spec",
        "beta",
        "se",
        "pvalue",
        "estimation_note",
    ]
    use = [c for c in cols if c in df.columns]
    out = df[use].copy()
    rename = {
        c: f"{sample}_{c}" if c not in ("outcome_id", "strategy_id", "spec", "outcome_family") else c
        for c in use
    }
    return out.rename(columns=rename)


def build_comparison_table(baseline: pd.DataFrame, exbantopic: pd.DataFrame) -> pd.DataFrame:
    """Function summary: merge baseline and exbantopic rows on outcome/strategy/spec."""
    b = _pick_columns(_filter_key_rows(baseline), "baseline")
    e = _pick_columns(_filter_key_rows(exbantopic), "exbantopic")
    merge_keys = ["outcome_id", "outcome_family", "strategy_id", "spec"]
    merged = b.merge(e, on=merge_keys, how="outer")
    if "baseline_beta" in merged.columns and "exbantopic_beta" in merged.columns:
        merged["delta_beta"] = merged["exbantopic_beta"] - merged["baseline_beta"]
        merged["sign_flip"] = (
            np.sign(merged["baseline_beta"].astype(float))
            != np.sign(merged["exbantopic_beta"].astype(float))
        ) & merged["baseline_beta"].notna() & merged["exbantopic_beta"].notna()
    return merged.sort_values(["outcome_id", "spec"]).reset_index(drop=True)


def _interpretation_lines(comparison: pd.DataFrame) -> List[str]:
    """Function summary: markdown bullets summarizing attention-shock vs discourse."""
    lines = [
        "# Ban-topic exclusion check (Check 1)",
        "",
        "Compare baseline subreddit-day DiD to samples excluding comments matching the "
        "ban-topic regex (`is_ban_topic`).",
        "",
        "**Interpretation:**",
        "- If `sem_axis_emotion` week-1 dip and `ai_style_rate` / `style_index_llm` bumps "
        "vanish on the exclusion sample → attention-shock (ban-discussion vocabulary) confirmed.",
        "- If the emotion dip survives exclusion → more consistent with a genuine discourse shift.",
        "",
        "## Key coefficients (`cross_country_all`)",
        "",
    ]
    for _, row in comparison.iterrows():
        oid = row.get("outcome_id", "")
        spec = row.get("spec", "")
        b_beta = row.get("baseline_beta", np.nan)
        e_beta = row.get("exbantopic_beta", np.nan)
        b_p = row.get("baseline_pvalue", np.nan)
        e_p = row.get("exbantopic_pvalue", np.nan)
        delta = row.get("delta_beta", np.nan)
        lines.append(
            f"- **{oid}** ({spec}): baseline β={b_beta:.4f} (p={b_p:.3g}) → "
            f"exbantopic β={e_beta:.4f} (p={e_p:.3g}); Δβ={delta:.4f}"
        )
    lines.append("")
    lines.append(
        "Full table: `results/tables/.../did/exbantopic_comparison.csv` "
        "(path resolved from study config)."
    )
    return lines


def main() -> None:
    """Function summary: write combined CSV and markdown comparison note."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    baseline_path, _ = did_summary_paths(config)
    exbantopic_path, _ = did_summary_paths(config, variant="exbantopic")

    baseline = _load_summary(baseline_path, "baseline")
    baseline = _patch_baseline_lexical_forum(config, baseline)
    exbantopic = _load_summary(exbantopic_path, "exbantopic")
    comparison = build_comparison_table(baseline, exbantopic)

    out_csv = did_root(config) / "exbantopic_comparison.csv"
    comparison.to_csv(out_csv, index=False)
    print(f"[compare_exbantopic_coefficients] wrote {out_csv} rows={len(comparison)}", flush=True)

    md_path = PROJECT_ROOT / args.markdown_out
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(_interpretation_lines(comparison)) + "\n", encoding="utf-8")
    print(f"[compare_exbantopic_coefficients] wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
