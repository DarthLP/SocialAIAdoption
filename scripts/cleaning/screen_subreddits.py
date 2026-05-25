"""
Script summary:
Stage-2 forum screening for the Italy polarization study: volume gates, URL-only
forum exclusion, and pooled Italian langid validation on cleaned interim Parquet.

Functionality:
- Reads cleaned monthly Parquet under paths.interim_dir/cleaned_monthly_chunks/.
- Applies PROFILE_USER, HIGH_URL_ONLY_SHARE, LOW_VOLUME_WINDOW, LOW_ITALIAN_POOLED gates.
- Writes screening CSVs and subreddit_exclusions.csv under paths.tables_dir/screening/.

How to apply/run:
  .venv/bin/python scripts/cleaning/screen_subreddits.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

try:
    import langid
except ImportError as exc:
    raise SystemExit("langid is required: pip install langid") from exc



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
    PROFILE_USER_PATTERN,
    italian_arms_for_langid,
    load_config,
    load_screening_config,
    resolve_primary_subreddits,
    subreddit_arm_map,
)


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments for forum screening."""
    parser = argparse.ArgumentParser(description="Screen subreddits after Stage-1 cleaning.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def list_parquet_shards(interim_dir: Path, subreddit: str) -> List[Path]:
    """Function summary: list monthly Parquet shards for one subreddit.

    Parameters:
    - interim_dir: interim data root.
    - subreddit: subreddit name.

    Returns:
    - Sorted list of Parquet paths.
    """
    sub_dir = interim_dir / "cleaned_monthly_chunks" / subreddit
    if not sub_dir.exists():
        return []
    return sorted(sub_dir.glob("*.parquet"))


def sample_bodies_for_langid(
    df: pd.DataFrame,
    year_month: str,
    sample_max: int,
    min_chars: int,
    rng: random.Random,
) -> List[str]:
    """Function summary: reservoir-sample comment bodies eligible for langid from one month.

    Parameters:
    - df: cleaned month dataframe.
    - year_month: YYYY-MM label for logging.
    - sample_max: maximum samples per month.
    - min_chars: minimum stripped body length.
    - rng: random generator.

    Returns:
    - List of sampled body strings.
    """
    del year_month
    bodies: List[str] = []
    if df.empty or "body" not in df.columns:
        return bodies
    eligible = [
        str(b).strip()
        for b in df["body"].tolist()
        if isinstance(b, str) and len(str(b).strip()) >= min_chars
    ]
    if len(eligible) <= sample_max:
        return eligible
    return rng.sample(eligible, sample_max)


def italian_share(bodies: List[str]) -> tuple[float, int]:
    """Function summary: compute Italian share among langid-classified bodies.

    Parameters:
    - bodies: list of text samples.

    Returns:
    - Tuple (italian_share, n_sampled).
    """
    if not bodies:
        return 0.0, 0
    italian = 0
    for body in bodies:
        lang, _ = langid.classify(body)
        if lang == "it":
            italian += 1
    return italian / len(bodies), len(bodies)


def load_stage1_audit(tables_dir: Path) -> pd.DataFrame:
    """Function summary: load Stage-1 subreddit audit aggregates if present.

    Parameters:
    - tables_dir: study tables directory.

    Returns:
    - Audit dataframe (possibly empty).
    """
    path = tables_dir / "cleaning" / "clean_daily_chunks_audit_by_subreddit.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def screen_subreddit(
    config: Dict[str, Any],
    subreddit: str,
    arm: str,
    shards: List[Path],
    screening: Dict[str, Any],
    audit_row: Dict[str, Any] | None,
    rng: random.Random,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Function summary: run screening gates for one subreddit.

    Parameters:
    - subreddit: subreddit name.
    - arm: comparison arm label.
    - shards: Parquet paths.
    - screening: screening config dict.
    - audit_row: optional Stage-1 audit aggregates.
    - rng: random generator.

    Returns:
    - Tuple (monthly_rows, pooled_summary_row).
    """
    min_chars = int(screening["langid_min_body_chars"])
    sample_max = int(screening["langid_sample_per_month"])
    url_threshold = float(screening["forum_url_only_share_exclude"])
    min_window = int(screening["min_kept_window_large_volume"])
    min_month_soft = int(screening["min_kept_per_month_soft"])
    it_threshold = float(screening["langid_italian_threshold_pooled"])

    rows_input = int(audit_row.get("rows_input", 0)) if audit_row else 0
    drop_url_only = int(audit_row.get("drop_url_only", 0)) if audit_row else 0
    url_share = (drop_url_only / rows_input) if rows_input > 0 else 0.0

    monthly_rows: List[Dict[str, Any]] = []
    pooled_bodies: List[str] = []
    n_kept_window = 0

    for shard in shards:
        year_month = shard.stem
        df = pd.read_parquet(shard, columns=["body"])
        n_kept = int(len(df))
        n_kept_window += n_kept
        bodies = sample_bodies_for_langid(df, year_month, sample_max, min_chars, rng)
        pooled_bodies.extend(bodies)
        share_month, n_sampled = italian_share(bodies)
        monthly_rows.append(
            {
                "subreddit": subreddit,
                "year_month": year_month,
                "arm": arm,
                "n_kept": n_kept,
                "n_sampled_langid": n_sampled,
                "italian_share_month": round(share_month, 4),
                "sparse_month": n_kept < min_month_soft,
            }
        )

    pooled_share, n_pooled_sampled = italian_share(pooled_bodies)
    codes: List[str] = []
    notes: List[str] = []

    if PROFILE_USER_PATTERN.match(subreddit):
        codes.append("PROFILE_USER")
    if rows_input > 0 and url_share >= url_threshold:
        codes.append("HIGH_URL_ONLY_SHARE")
    if n_kept_window < min_window:
        codes.append("LOW_VOLUME_WINDOW")

    needs_italian = arm in italian_arms_for_langid(config)
    if needs_italian and n_pooled_sampled > 0 and pooled_share < it_threshold:
        codes.append("LOW_ITALIAN_POOLED")

    if any(c in codes for c in ("PROFILE_USER", "HIGH_URL_ONLY_SHARE", "LOW_ITALIAN_POOLED")):
        action = "excluded"
        volume_band = "excluded"
    elif "LOW_VOLUME_WINDOW" in codes:
        action = "low_volume"
        volume_band = "low_volume"
    else:
        action = "large_volume"
        volume_band = "large_volume"

    pooled_row = {
        "subreddit": subreddit,
        "arm": arm,
        "rows_input_stage1": rows_input,
        "drop_url_only_stage1": drop_url_only,
        "url_only_share_stage1": round(url_share, 4),
        "n_kept_window": n_kept_window,
        "n_sampled_langid_pooled": n_pooled_sampled,
        "italian_share_pooled": round(pooled_share, 4),
        "pass_italian_gate": (not needs_italian) or (pooled_share >= it_threshold),
        "pass_volume_large_volume": n_kept_window >= min_window,
        "exclusion_codes": "|".join(codes),
        "action": action,
        "volume_band": volume_band,
        "notes": "; ".join(notes),
    }
    return monthly_rows, pooled_row


def write_notes(path: Path, screening: Dict[str, Any]) -> None:
    """Function summary: write methods-ready screening threshold notes.

    Parameters:
    - path: output text path.
    - screening: screening config dict.
    """
    lines = [
        "Subreddit Screening Notes",
        "=========================",
        "",
        "Gates:",
        "- PROFILE_USER: subreddit name matches ^u_",
        f"- HIGH_URL_ONLY_SHARE: drop_url_only/rows_input >= {screening['forum_url_only_share_exclude']}",
        f"- LOW_VOLUME_WINDOW: kept comments < {screening['min_kept_window_large_volume']} (low_volume)",
        f"- LOW_ITALIAN_POOLED: Italian arms, pooled langid share < {screening['langid_italian_threshold_pooled']}",
        "",
        "Langid:",
        f"- Up to {screening['langid_sample_per_month']} comments/month, min body {screening['langid_min_body_chars']} chars",
        f"- Pooled Mar-Apr share threshold: {screening['langid_italian_threshold_pooled']}",
        f"- RNG seed: {screening['langid_rng_seed']}",
        "",
        "Comments-only corpus: thread political labels use link_id roll-up without submission titles.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Function summary: run forum screening and write CSV outputs."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    screening = load_screening_config(config)
    interim_dir = Path(config["paths"]["interim_dir"])
    tables_dir = Path(config["paths"]["tables_dir"])
    out_dir = tables_dir / "screening"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(screening["langid_rng_seed"]))
    arms = subreddit_arm_map(config)
    audit_df = load_stage1_audit(tables_dir)
    audit_by_sub = (
        {str(r["subreddit"]): r for r in audit_df.to_dict(orient="records")} if not audit_df.empty else {}
    )

    monthly_all: List[Dict[str, Any]] = []
    pooled_all: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []

    subreddits = resolve_primary_subreddits(config)
    print(f"[screen_subreddits] subreddits={len(subreddits)}", flush=True)
    for idx, subreddit in enumerate(subreddits, start=1):
        arm = arms.get(subreddit, "discovered_italian")
        shards = list_parquet_shards(interim_dir, subreddit)
        audit_row = audit_by_sub.get(subreddit)
        monthly_rows, pooled_row = screen_subreddit(
            config=config,
            subreddit=subreddit,
            arm=arm,
            shards=shards,
            screening=screening,
            audit_row=audit_row,
            rng=rng,
        )
        monthly_all.extend(monthly_rows)
        pooled_all.append(pooled_row)
        if pooled_row["exclusion_codes"]:
            for code in pooled_row["exclusion_codes"].split("|"):
                exclusions.append(
                    {
                        "subreddit": subreddit,
                        "code": code,
                        "action": pooled_row["action"],
                        "notes": pooled_row.get("notes", ""),
                    }
                )
        print(
            f"[screen_subreddits] subreddit_done {idx}/{len(subreddits)} "
            f"subreddit={subreddit} shards={len(shards)} action={pooled_row['action']} "
            f"n_kept_window={pooled_row['n_kept_window']}",
            flush=True,
        )

    print("[screen_subreddits] writing_csv_outputs", flush=True)
    pd.DataFrame(monthly_all).to_csv(out_dir / "subreddit_screening_by_month.csv", index=False)
    pd.DataFrame(pooled_all).to_csv(out_dir / "subreddit_screening_pooled.csv", index=False)
    pd.DataFrame(exclusions).drop_duplicates().to_csv(out_dir / "subreddit_exclusions.csv", index=False)
    write_notes(out_dir / "screening_run_notes.txt", screening)

    _excl_spec = importlib.util.spec_from_file_location(
        "_write_exclusion_summary_mod",
        Path(__file__).resolve().parent / "write_exclusion_summary.py",
    )
    if _excl_spec is None or _excl_spec.loader is None:
        raise RuntimeError("Failed to load write_exclusion_summary.py")
    _excl_mod = importlib.util.module_from_spec(_excl_spec)
    _excl_spec.loader.exec_module(_excl_mod)
    _excl_mod.write_exclusion_summaries(config=config, project_root=PROJECT_ROOT, tables_dir=tables_dir)

    print(f"[screen_subreddits] wrote outputs under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
