"""
Script summary:
Post-run sanity gates for comment-weighted DiD (estimates_weighted/) vs unweighted baseline.

Functionality:
- --write-baseline: snapshot unweighted did_summary.csv hash and ai_style_rate weighted full_ban β/p (run before weighted re-estimation).
- Default: verify unweighted tree untouched, n_obs parity (cross_country_all × full_ban/early_ban_7d), ai_style_rate reproduction, print 4-outcome comparison table.
- Skips wordfish families in parity checks; never gates on wordfish rows.

How to apply/run:
  .venv/bin/python scripts/analysis/verify_weighted_did_coverage.py --config config/italy_polarization_setup.yaml --write-baseline
  .venv/bin/python scripts/analysis/did_event_study.py --config config/italy_polarization_setup.yaml --weights n_comments --families lexical,semantic_axis,quantity --no-figures --no-bootstrap
  .venv/bin/python scripts/analysis/verify_weighted_did_coverage.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

COMPARISON_OUTCOMES: Tuple[Tuple[str, str], ...] = (
    ("sem_axis_emotion", "early_ban_7d"),
    ("pole_share", "full_ban"),
    ("aggression_rate", "full_ban"),
    ("sem_axis_ideology_var", "full_ban"),
)
STRATEGY = "cross_country_all"
PARITY_SPECS = ("full_ban", "early_ban_7d")
REPRO_TOL = 1e-5
SANITY_DIRNAME = ".sanity"
BASELINE_HASH_NAME = "unweighted_did_summary.sha256"
PRE_RUN_AI_STYLE_NAME = "ai_style_rate_pre_run.json"


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
from src.did.paths import did_summary_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI for weighted DiD verification."""
    parser = argparse.ArgumentParser(description="Verify weighted DiD coverage and sanity gates.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Record unweighted summary hash and pre-run ai_style_rate weighted β/p; run before weighted DiD.",
    )
    return parser.parse_args()


def _sanity_dir(config: Dict[str, Any]) -> Path:
    """Function summary: return estimates_weighted/.sanity/ path."""
    weighted_summary, _ = did_summary_paths(config, weighted=True)
    d = weighted_summary.parent.parent / SANITY_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _file_sha256(path: Path) -> str:
    """Function summary: SHA-256 hex digest of file contents."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def write_baseline(config: Dict[str, Any]) -> None:
    """Function summary: persist pre-run checksums for post-run verification.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - None; writes hash and ai_style_rate snapshot under estimates_weighted/.sanity/.
    """
    unweighted_path, _ = did_summary_paths(config, weighted=False)
    weighted_path, _ = did_summary_paths(config, weighted=True)
    if not unweighted_path.is_file():
        raise FileNotFoundError(f"Missing unweighted summary: {unweighted_path}")
    if not weighted_path.is_file():
        raise FileNotFoundError(f"Missing weighted summary: {weighted_path}")

    sanity = _sanity_dir(config)
    digest = _file_sha256(unweighted_path)
    (sanity / BASELINE_HASH_NAME).write_text(digest + "\n", encoding="utf-8")

    wdf = pd.read_csv(weighted_path)
    wsub = wdf[
        (wdf["outcome_id"].astype(str) == "ai_style_rate")
        & (wdf["strategy_id"].astype(str) == STRATEGY)
        & (wdf["spec"].astype(str) == "full_ban")
    ]
    udf = _dedupe_summary(pd.read_csv(unweighted_path))
    if "weights" in udf.columns:
        udf = udf[udf["weights"].astype(str).fillna("") == ""]
    usub = udf[
        (udf["outcome_id"].astype(str) == "ai_style_rate")
        & (udf["strategy_id"].astype(str) == STRATEGY)
        & (udf["spec"].astype(str) == "full_ban")
    ]
    if usub.empty:
        raise ValueError("No ai_style_rate unweighted full_ban row in did_summary")
    urow = usub.iloc[-1]
    payload: Dict[str, Any] = {
        "outcome_id": "ai_style_rate",
        "strategy_id": STRATEGY,
        "spec": "full_ban",
        "unweighted_n_obs": int(urow["n_obs"]),
        "unweighted_beta": float(urow["beta"]),
    }
    if not wsub.empty:
        wrow = wsub.iloc[-1]
        payload["prior_weighted_beta"] = float(wrow["beta"])
        payload["prior_weighted_n_obs"] = int(wrow["n_obs"])
        payload["prior_weighted_pvalue"] = float(wrow["pvalue"])
    (sanity / PRE_RUN_AI_STYLE_NAME).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[verify_weighted_did] wrote baseline hash → {sanity / BASELINE_HASH_NAME}", flush=True)
    if "prior_weighted_beta" in payload:
        print(
            f"[verify_weighted_did] prior weighted ai_style_rate full_ban "
            f"β={payload['prior_weighted_beta']:.6g} n_obs={payload['prior_weighted_n_obs']}",
            flush=True,
        )
    print(
        f"[verify_weighted_did] current unweighted n_obs={payload['unweighted_n_obs']} "
        f"(panel vintage anchor)",
        flush=True,
    )


def _is_wordfish_family(family: str) -> bool:
    """Function summary: True when outcome_family is a wordfish panel family."""
    return str(family).startswith("wordfish")


def _filter_headline(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: cross_country_all rows with optional weights column filter for weighted."""
    work = df[df["strategy_id"].astype(str) == STRATEGY].copy()
    if "weights" in work.columns:
        w = work["weights"].astype(str).fillna("")
        # keep unweighted rows (empty weights) or n_comments weighted rows
        pass
    return work


def _dedupe_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: keep last row per outcome/strategy/spec/weights (matches did_event_study)."""
    dedupe_cols = [
        c for c in ("outcome_id", "strategy_id", "spec", "weights") if c in df.columns
    ]
    if not dedupe_cols:
        return df
    return df.drop_duplicates(subset=dedupe_cols, keep="last")


def _load_summary(path: Path, *, weighted: bool) -> pd.DataFrame:
    """Function summary: load did_summary.csv or raise."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary: {path}")
    df = pd.read_csv(path)
    if weighted and "weights" in df.columns:
        df = df[df["weights"].astype(str).fillna("").str.lower() == "n_comments"]
    elif not weighted and "weights" in df.columns:
        df = df[df["weights"].astype(str).fillna("") == ""]
    return _dedupe_summary(df)


def _pick_beta_p(df: pd.DataFrame, outcome_id: str, spec: str) -> Tuple[float, float]:
    """Function summary: extract beta and pvalue for one outcome/spec or NaN pair."""
    sub = df[
        (df["outcome_id"].astype(str) == outcome_id) & (df["spec"].astype(str) == spec)
    ]
    if sub.empty:
        return float("nan"), float("nan")
    row = sub.iloc[0]
    return float(row.get("beta", np.nan)), float(row.get("pvalue", np.nan))


def print_comparison_table(unweighted: pd.DataFrame, weighted: pd.DataFrame) -> None:
    """Function summary: print two-column weighted vs unweighted β/p for headline outcomes."""
    lines = [
        "",
        "Weighted vs unweighted comparison (cross_country_all):",
        f"{'outcome':<24} {'spec':<14} {'unw_β':>10} {'w_β':>10} {'unw_p':>10} {'w_p':>10}",
        "-" * 82,
    ]
    for outcome_id, spec in COMPARISON_OUTCOMES:
        ub, up = _pick_beta_p(unweighted, outcome_id, spec)
        wb, wp = _pick_beta_p(weighted, outcome_id, spec)
        lines.append(
            f"{outcome_id:<24} {spec:<14} {ub:10.4g} {wb:10.4g} {up:10.4g} {wp:10.4g}"
        )
    print("\n".join(lines), flush=True)


def verify_coverage(config: Dict[str, Any]) -> int:
    """Function summary: run all sanity gates; return 0 on success else 1.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - Process exit code (0 pass, 1 fail).
    """
    sanity = _sanity_dir(config)
    hash_path = sanity / BASELINE_HASH_NAME
    pre_run_path = sanity / PRE_RUN_AI_STYLE_NAME
    if not hash_path.is_file() or not pre_run_path.is_file():
        print(
            "[verify_weighted_did] ERROR: run with --write-baseline before weighted DiD",
            file=sys.stderr,
            flush=True,
        )
        return 1

    unweighted_path, _ = did_summary_paths(config, weighted=False)
    weighted_path, _ = did_summary_paths(config, weighted=True)
    expected_hash = hash_path.read_text(encoding="utf-8").strip()
    actual_hash = _file_sha256(unweighted_path)
    if actual_hash != expected_hash:
        print(
            f"[verify_weighted_did] FAIL: unweighted summary changed (hash mismatch)",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print("[verify_weighted_did] PASS: unweighted estimates/ untouched", flush=True)

    pre_run = json.loads(pre_run_path.read_text(encoding="utf-8"))
    unw = _load_summary(unweighted_path, weighted=False)
    wgt = _load_summary(weighted_path, weighted=True)

    # Reproduction (same panel vintage as unweighted anchor)
    sub = wgt[
        (wgt["outcome_id"].astype(str) == "ai_style_rate")
        & (wgt["spec"].astype(str) == "full_ban")
    ]
    if sub.empty:
        print("[verify_weighted_did] FAIL: missing post-run ai_style_rate full_ban", file=sys.stderr)
        return 1
    post_row = sub.iloc[0]
    post_beta = float(post_row["beta"])
    post_n_obs = int(post_row["n_obs"])
    anchor_n = int(pre_run["unweighted_n_obs"])
    if post_n_obs != anchor_n:
        print(
            f"[verify_weighted_did] FAIL: weighted n_obs={post_n_obs} != unweighted anchor n_obs={anchor_n}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    prior_n = pre_run.get("prior_weighted_n_obs")
    prior_beta = pre_run.get("prior_weighted_beta")
    if prior_n is not None and int(prior_n) == post_n_obs and prior_beta is not None:
        if abs(post_beta - float(prior_beta)) > REPRO_TOL:
            print(
                f"[verify_weighted_did] FAIL: ai_style_rate reproduction "
                f"(prior={prior_beta:.6g} post={post_beta:.6g})",
                file=sys.stderr,
                flush=True,
            )
            return 1
        print(
            f"[verify_weighted_did] PASS: ai_style_rate full_ban β reproduced ({post_beta:.6g})",
            flush=True,
        )
    else:
        print(
            f"[verify_weighted_did] PASS: ai_style_rate full_ban β={post_beta:.6g} "
            f"(panel vintage n_obs={post_n_obs}; prior weighted run used n_obs={prior_n})",
            flush=True,
        )

    # n_obs parity (non-wordfish, headline strategy/specs)
    w_outcomes = wgt[~wgt["outcome_family"].astype(str).map(_is_wordfish_family)]
    failures: List[str] = []
    for outcome_id, grp in w_outcomes.groupby("outcome_id"):
        fam = str(grp["outcome_family"].iloc[0])
        if _is_wordfish_family(fam):
            continue
        for spec in PARITY_SPECS:
            wsub = grp[
                (grp["strategy_id"].astype(str) == STRATEGY)
                & (grp["spec"].astype(str) == spec)
            ]
            usub = unw[
                (unw["outcome_id"].astype(str) == outcome_id)
                & (unw["strategy_id"].astype(str) == STRATEGY)
                & (unw["spec"].astype(str) == spec)
            ]
            if wsub.empty:
                continue
            if usub.empty:
                failures.append(f"{outcome_id}/{spec}: missing unweighted row")
                continue
            w_n = int(wsub.iloc[-1]["n_obs"])
            u_n = int(usub.iloc[-1]["n_obs"])
            if w_n != u_n:
                failures.append(f"{outcome_id}/{spec}: weighted n_obs={w_n} != unweighted n_obs={u_n}")

    if failures:
        print("[verify_weighted_did] FAIL: n_obs parity", file=sys.stderr)
        for f in failures[:20]:
            print(f"  {f}", file=sys.stderr)
        return 1
    print(f"[verify_weighted_did] PASS: n_obs parity ({len(w_outcomes['outcome_id'].unique())} outcomes)", flush=True)

    # Required families present in weighted summary
    required_fams = {"lexical", "semantic_axis", "quantity"}
    present = set(wgt["outcome_family"].astype(str).unique())
    missing_fams = required_fams - present
    if missing_fams:
        print(f"[verify_weighted_did] FAIL: missing weighted families {missing_fams}", file=sys.stderr)
        return 1
    print(f"[verify_weighted_did] PASS: families present {sorted(required_fams)}", flush=True)

    print_comparison_table(unw, wgt)
    return 0


def main() -> None:
    """Function summary: CLI entry."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    if args.write_baseline:
        write_baseline(config)
        return
    raise SystemExit(verify_coverage(config))


if __name__ == "__main__":
    main()
