"""
Script summary:
Scatter forum Wordfish v1 vs v2 subreddit theta (ideology fits) with Spearman ρ by facet.

Functionality:
- Inner-join v1 and v2 theta tables on subreddit (+ language, time_bin when present).
- Facet by time_bin; color Italian vs English; annotate Spearman ρ in facet titles.

How to apply/run:
  .venv/bin/python scripts/diagnostics/plot_wordfish_v1_vs_v2.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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

from src.config_utils import figures_subdir, load_config, tables_subdir  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for v1 vs v2 theta scatter."""
    parser = argparse.ArgumentParser(description="Wordfish v1 vs v2 theta scatter.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def _load_theta(path: Path, suffix: str) -> pd.DataFrame:
    """Function summary: load theta table and rename theta column.

    Parameters:
    - path: CSV path.
    - suffix: column suffix for theta (v1 or v2).

    Returns:
    - DataFrame with theta_{suffix} column.
    """
    df = pd.read_csv(path)
    col = "theta" if "theta" in df.columns else "theta_ideology"
    out = df.copy()
    out[f"theta_{suffix}"] = pd.to_numeric(out[col], errors="coerce")
    return out


def main() -> None:
    """Function summary: write wordfish_v1_vs_v2_theta_scatter.png."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    v1_path = tables_subdir(config, "wordfish") / "wordfish_subreddit_theta_ideology.csv"
    v2_path = tables_subdir(config, "wordfish_forum_v2") / "wordfish_subreddit_theta_ideology.csv"
    if not v1_path.is_file() or not v2_path.is_file():
        raise FileNotFoundError(f"Missing theta tables: {v1_path} or {v2_path}")

    v1 = _load_theta(v1_path, "v1")
    v2 = _load_theta(v2_path, "v2")
    join_keys = [k for k in ("subreddit", "language", "time_bin", "primary_lexicon") if k in v1.columns and k in v2.columns]
    if "subreddit" not in join_keys:
        join_keys = ["subreddit"]
    merged = v1.merge(v2[join_keys + ["theta_v2"]], on=join_keys, how="inner")
    merged = merged.dropna(subset=["theta_v1", "theta_v2"])
    if merged.empty:
        print("[plot_wordfish_v1_vs_v2] empty join", flush=True)
        return

    lang_col = "language" if "language" in merged.columns else "primary_lexicon"
    if lang_col not in merged.columns:
        merged["language"] = "all"
        lang_col = "language"
    merged["lang_plot"] = merged[lang_col].astype(str).str.lower().map(
        lambda x: "it" if x in ("it", "italian") else ("en" if x.startswith("en") or x == "en" else x)
    )
    facet_col = "time_bin" if "time_bin" in merged.columns else None
    facets = sorted(merged[facet_col].dropna().unique()) if facet_col else ["all"]
    if facet_col is None:
        merged["facet"] = "all"
        facet_col = "facet"

    nfac = len(facets)
    fig, axes = plt.subplots(1, nfac, figsize=(5 * nfac, 4.5), squeeze=False)
    colors = {"it": "#c1121f", "en": "#1d3557"}
    for j, facet in enumerate(facets):
        ax = axes[0, j]
        sub = merged[merged[facet_col].astype(str) == str(facet)]
        for lang in ("it", "en"):
            g = sub[sub["lang_plot"] == lang]
            if g.empty:
                continue
            ax.scatter(g["theta_v1"], g["theta_v2"], alpha=0.5, s=18, c=colors.get(lang, "#555"), label=lang)
        rho, _ = spearmanr(sub["theta_v1"], sub["theta_v2"])
        ax.set_title(f"{facet}\nSpearman ρ = {rho:.2f}")
        ax.set_xlabel("θ v1")
        ax.set_ylabel("θ v2")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.legend(fontsize=8)
    fig.suptitle("Forum Wordfish v1 vs v2 subreddit θ", fontsize=11)
    fig.tight_layout()
    out = figures_subdir(config, "wordfish") / "wordfish_v1_vs_v2_theta_scatter.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_wordfish_v1_vs_v2] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
