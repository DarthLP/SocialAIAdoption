"""
Script summary:
Build likely-adopter flag CSVs for DDD analysis (schemes 1–3).

How to apply/run:
  .venv/bin/python scripts/analysis/prepare_adopter_flags.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

MENTION_RE = re.compile(r"chatgpt|openai|gpt-?[34]", re.IGNORECASE)
SCHEME3_TECH_SUB = "ItalyInformatica"


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

from src.config_utils import load_config, tables_subdir  # noqa: E402
from src.did.specs import CONTROL_FAMILIES, ITALY_FAMILIES  # noqa: E402


def _country_from_family(fam: str) -> str:
    """Function summary: map topic_family to country bucket for thresholds."""
    if fam in ITALY_FAMILIES:
        return "IT"
    if fam in CONTROL_FAMILIES:
        return str(fam).upper() if fam != "eu" else "EU"
    return str(fam)


def parse_args() -> argparse.Namespace:
    """Function summary: CLI."""
    p = argparse.ArgumentParser(description="Prepare adopter flag CSVs.")
    p.add_argument("--config", default="config/italy_polarization_setup.yaml")
    return p.parse_args()


def main() -> None:
    """Function summary: write adopter_flags.csv and overlap matrix."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    shard_root = Path(config["paths"]["interim_dir"]) / "cleaned_monthly_chunks"
    march = ("2023-03-01", "2023-03-31")
    rows: List[dict] = []
    for sub_dir in shard_root.iterdir():
        if not sub_dir.is_dir():
            continue
        for shard in sub_dir.glob("*.parquet"):
            cols = ["author", "body", "date_utc", "topic_family", "style_index_full", "subreddit"]
            try:
                df = pd.read_parquet(shard, columns=[c for c in cols if True])
            except Exception:
                continue
            df["date_utc"] = df["date_utc"].astype(str).str[:10]
            m = (df["date_utc"] >= march[0]) & (df["date_utc"] < march[1])
            df = df[m]
            if df.empty:
                continue
            for author, grp in df.groupby("author", observed=True):
                fam = str(grp["topic_family"].iloc[0]) if "topic_family" in grp.columns else "unknown"
                country = _country_from_family(fam)
                n = len(grp)
                si = grp["style_index_full"] if "style_index_full" in grp.columns else pd.Series(dtype=float)
                si_mean = float(si.mean()) if si.notna().any() else float("nan")
                mention = bool(grp["body"].astype(str).str.contains(MENTION_RE).any()) if "body" in grp.columns else False
                tech = bool((grp.get("subreddit", pd.Series(dtype=str)).astype(str) == SCHEME3_TECH_SUB).any())
                rows.append(
                    {
                        "author": str(author),
                        "country": country,
                        "topic_family": fam,
                        "n_comments_march": n,
                        "style_index_full_mean_march": si_mean,
                        "scheme3_mention": int(mention),
                        "scheme3_tech": int(tech),
                    }
                )
    meta = pd.DataFrame(rows).drop_duplicates("author")
    if meta.empty:
        raise RuntimeError("No author rows built; run style index on shards first.")

    out_rows = []
    for country, grp in meta.groupby("country", observed=True):
        p90 = grp["n_comments_march"].quantile(0.9)
        si_valid = grp.dropna(subset=["style_index_full_mean_march"])
        si_valid = si_valid[si_valid["n_comments_march"] >= 5]
        q75 = si_valid["style_index_full_mean_march"].quantile(0.75) if len(si_valid) >= 4 else float("nan")
        for _, r in grp.iterrows():
            out_rows.append(
                {
                    "author": r["author"],
                    "country": country,
                    "topic_family": r["topic_family"],
                    "scheme1_inactive": int(r["n_comments_march"] < p90),
                    "scheme2_styletop": int(
                        r["n_comments_march"] >= 5
                        and np.isfinite(r["style_index_full_mean_march"])
                        and np.isfinite(q75)
                        and r["style_index_full_mean_march"] >= q75
                    ),
                    "scheme2_firsthalf": 0,
                    "scheme3_tech": int(r["scheme3_tech"] and country == "IT"),
                    "scheme3_mention": int(r["scheme3_mention"]),
                }
            )
    out = pd.DataFrame(out_rows)
    half = ("2023-03-01", "2023-03-16")
    for shard in shard_root.rglob("*.parquet"):
        try:
            df = pd.read_parquet(shard, columns=["author", "date_utc", "style_index_full", "topic_family"])
        except Exception:
            continue
        df["date_utc"] = df["date_utc"].astype(str).str[:10]
        m = (df["date_utc"] >= half[0]) & (df["date_utc"] < half[1])
        df = df[m]
        if df.empty or "style_index_full" not in df.columns:
            continue
        for author, grp in df.groupby("author"):
            if len(grp) < 5:
                continue
            country = _country_from_family(str(grp["topic_family"].iloc[0]))
            q75h = grp["style_index_full"].quantile(0.75)
            mask = out["author"] == str(author)
            if mask.any():
                out.loc[mask, "scheme2_firsthalf"] = int(grp["style_index_full"].mean() >= q75h)
    did_dir = tables_subdir(config, "did")
    did_dir.mkdir(parents=True, exist_ok=True)
    path = did_dir / "adopter_flags.csv"
    out.to_csv(path, index=False)
    schemes = ["scheme1_inactive", "scheme2_styletop", "scheme3_mention"]
    jac = []
    for a in schemes:
        for b in schemes:
            sa = set(out.loc[out[a] == 1, "author"])
            sb = set(out.loc[out[b] == 1, "author"])
            inter = len(sa & sb)
            union = len(sa | sb) or 1
            jac.append({"scheme_a": a, "scheme_b": b, "jaccard": inter / union})
    pd.DataFrame(jac).to_csv(did_dir / "adopter_flags_overlap.csv", index=False)
    print(f"[prepare_adopter_flags] wrote {path} rows={len(out)}", flush=True)
    print(pd.DataFrame(jac).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
