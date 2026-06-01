"""
Script summary:
Lexical DiD coefficients by control-country choice (Italy vs DE/EU/UK/US and pooled).

Functionality:
- Horizontal bar chart with 95% CI for four lexical outcomes across five strategy contrasts.
- Highlights heterogeneity masked by the pooled cross_country_all estimate.

How to apply/run:
  .venv/bin/python scripts/diagnostics/plot_did_lexical_by_control.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
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

from src.config_utils import figures_subdir, load_config  # noqa: E402
from src.did.outcomes import (  # noqa: E402
    LEXICAL_BY_CONTROL_OUTCOMES,
    LEXICAL_BY_CONTROL_STRATEGIES,
    outcome_label,
)
from src.did.paths import did_legacy_coefficient_path, did_summary_paths  # noqa: E402
from src.did.specs import strategy_label  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for lexical-by-control figure."""
    parser = argparse.ArgumentParser(description="Lexical DiD by control choice.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def _load_outcome_coefs(config, outcome_id: str) -> pd.DataFrame:
    """Function summary: load coefficient rows for one lexical outcome."""
    legacy = did_legacy_coefficient_path(config, outcome_id)
    if legacy.is_file():
        return pd.read_csv(legacy)
    summary_path, _ = did_summary_paths(config)
    if summary_path.is_file():
        df = pd.read_csv(summary_path)
        return df[df["outcome_id"] == outcome_id]
    return pd.DataFrame()


def main() -> None:
    """Function summary: write did/lexical/by_control_choice.png."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    strats = list(LEXICAL_BY_CONTROL_STRATEGIES)
    outcomes = list(LEXICAL_BY_CONTROL_OUTCOMES)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes_flat = axes.flatten()
    for ax, oid in zip(axes_flat, outcomes):
        df = _load_outcome_coefs(config, oid)
        if df.empty:
            ax.set_visible(False)
            continue
        sub = df[
            (df["strategy_id"].isin(strats))
            & (df.get("spec", "full_ban").astype(str) == "full_ban")
            & (df["estimation_note"].astype(str) == "ok")
        ].copy()
        sub = sub.dropna(subset=["beta"])
        if sub.empty:
            ax.set_title(outcome_label(oid, short=True))
            continue
        order = [s for s in strats if s in set(sub["strategy_id"])]
        sub = sub.set_index("strategy_id").loc[order].reset_index()
        y = np.arange(len(sub))
        ax.errorbar(
            sub["beta"],
            y,
            xerr=1.96 * sub["se"],
            fmt="o",
            color="#2d6a4f",
            capsize=3,
        )
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([strategy_label(s, short=True) for s in sub["strategy_id"]], fontsize=8)
        ax.set_title(outcome_label(oid, short=True))
        ax.set_xlabel("β")
    fig.suptitle("Lexical DiD: Italy vs control choice", fontsize=12)
    fig.tight_layout()
    out = figures_subdir(config, "did") / "lexical" / "by_control_choice.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_did_lexical_by_control] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
