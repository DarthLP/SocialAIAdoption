"""
Script summary:
Profile subreddits in Reddit comment dumps for the Italy polarization study using a short
discovery window (default: first 3 UTC days of March 2023). Counts comments per subreddit,
reservoir-samples bodies for langid on active non-control subs, flags Italian-language
candidates, and writes CSV previews with projected full-window comment volumes.

Functionality:
- Streams configured discovery_window dump file(s) from --source_dir without materializing NDJSON.
- Early-stops when created_utc reaches discovery_window.end_utc_exclusive (if dump is time-ordered).
- Skips langid when n_comments_first_3d < min_comments_for_langid; controls are counted but not langid-sampled.
- Writes subreddit_census_3d.csv, candidate_italian_subreddits.csv, extraction_size_preview.csv, discovery_run_notes.txt.

How to run:
  .venv/bin/python scripts/discovery/profile_subreddits_in_dump.py \
    --config config/italy_polarization_setup.yaml \
    --source_dir "/Volumes/Expansion/Masterthesis/RawData/reddit/comments"
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import zstandard as zstd

try:
    import orjson
except ImportError:
    orjson = None

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
    control_subreddits_for_discovery,
    load_config,
    resolve_primary_subreddits,
    subreddit_arm_map,
    subreddit_control_lists,
    utc_ts,
)


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments for discovery profiling."""
    parser = argparse.ArgumentParser(description="Profile subreddits in a short discovery dump window.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--source_dir",
        type=str,
        default="/Volumes/Expansion/Masterthesis/RawData/reddit/comments",
    )
    parser.add_argument(
        "--save-langid-samples",
        action="store_true",
        help="Write optional audit CSV of sampled bodies used for langid.",
    )
    return parser.parse_args()


def parse_line(line_bytes: bytes) -> Dict[str, Any]:
    """Function summary: parse one NDJSON/JSON line from a comment dump."""
    if orjson is not None:
        return orjson.loads(line_bytes)
    return json.loads(line_bytes.decode("utf-8"))


class ReservoirSampler:
    """Function summary: reservoir sample up to k string items from a stream."""

    def __init__(self, max_size: int, rng: random.Random) -> None:
        self.max_size = max_size
        self.rng = rng
        self.items: List[str] = []
        self.n_seen = 0

    def add(self, value: str) -> None:
        """Function summary: add one item to the reservoir sample."""
        if not value or not value.strip():
            return
        self.n_seen += 1
        if len(self.items) < self.max_size:
            self.items.append(value)
            return
        j = self.rng.randint(0, self.n_seen - 1)
        if j < self.max_size:
            self.items[j] = value


def scan_discovery_window(
    source_file: Path,
    start_ts: int,
    end_ts: int,
    sample_max: int,
    control_subs: set[str],
    rng: random.Random,
) -> Tuple[Dict[str, int], Dict[str, ReservoirSampler], Dict[str, Any]]:
    """Function summary: stream one zst dump for the discovery window and collect counts/samples.

    Returns:
    - counts per subreddit
    - reservoir samplers per subreddit (non-control only)
    - run metadata (early_stop, ordering flags, lines seen)
    """
    counts: Dict[str, int] = {}
    samplers: Dict[str, ReservoirSampler] = {}
    meta: Dict[str, Any] = {
        "source_file": source_file.name,
        "lines_seen": 0,
        "rows_in_window": 0,
        "early_stop": False,
        "ordering_violations": 0,
        "prev_created_utc": None,
        "first_in_window_utc": None,
        "last_in_window_utc": None,
    }

    with source_file.open("rb") as compressed:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)  # required for large Reddit frames
        with dctx.stream_reader(compressed) as stream:
            with io.BufferedReader(stream, buffer_size=4 * 1024 * 1024) as buffered:
                for raw_line in buffered:
                    meta["lines_seen"] += 1
                    line_bytes = raw_line.rstrip(b"\r\n")
                    if not line_bytes:
                        continue
                    try:
                        rec = parse_line(line_bytes)
                    except Exception:
                        continue
                    try:
                        created_utc = int(rec.get("created_utc", 0))
                    except (TypeError, ValueError):
                        continue

                    prev = meta["prev_created_utc"]
                    if prev is not None and created_utc < int(prev):
                        meta["ordering_violations"] = int(meta["ordering_violations"]) + 1
                    meta["prev_created_utc"] = created_utc

                    if created_utc < start_ts:
                        continue
                    if created_utc >= end_ts:
                        meta["early_stop"] = True
                        break

                    subreddit = rec.get("subreddit")
                    if not isinstance(subreddit, str) or not subreddit.strip():
                        continue
                    subreddit = subreddit.strip()
                    meta["rows_in_window"] += 1
                    if meta["first_in_window_utc"] is None:
                        meta["first_in_window_utc"] = created_utc
                    meta["last_in_window_utc"] = created_utc

                    counts[subreddit] = counts.get(subreddit, 0) + 1
                    if subreddit in control_subs:
                        continue
                    body = rec.get("body")
                    if not isinstance(body, str):
                        continue
                    sampler = samplers.get(subreddit)
                    if sampler is None:
                        sampler = ReservoirSampler(sample_max, rng)
                        samplers[subreddit] = sampler
                    sampler.add(body)

    return counts, samplers, meta


def italian_share_from_bodies(bodies: List[str]) -> Tuple[float, int]:
    """Function summary: classify bodies with langid and return Italian share and sample size."""
    if not bodies:
        return 0.0, 0
    italian = 0
    for body in bodies:
        lang, _ = langid.classify(body)
        if lang == "it":
            italian += 1
    return italian / len(bodies), len(bodies)


def projected_comments(n_first_window: int, extract_days: int, discovery_days: int) -> int:
    """Function summary: linear extrapolation from discovery-window count to full extract window."""
    if discovery_days <= 0:
        return 0
    return int(round(n_first_window * (extract_days / discovery_days)))


def build_outputs(
    config: Dict[str, Any],
    counts: Dict[str, int],
    samplers: Dict[str, ReservoirSampler],
    meta: Dict[str, Any],
    discovery_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Function summary: assemble census, Italian candidates, and extraction preview tables."""
    threshold = float(discovery_cfg.get("italian_share_threshold", 0.40))
    min_comments = int(discovery_cfg.get("min_comments_for_langid", 5))
    extract_days = int(discovery_cfg.get("extract_window_days", 61))
    discovery_days = int(discovery_cfg.get("discovery_window_days", 3))

    census_rows = [
        {"subreddit": sub, "n_comments_first_3d": n}
        for sub, n in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    census_df = pd.DataFrame(census_rows)

    candidate_rows: List[Dict[str, Any]] = []
    for subreddit, sampler in sorted(samplers.items()):
        n_comments = counts.get(subreddit, 0)
        if n_comments < min_comments:
            continue
        share, n_sampled = italian_share_from_bodies(sampler.items)
        if share < threshold:
            continue
        candidate_rows.append(
            {
                "subreddit": subreddit,
                "n_comments_first_3d": n_comments,
                "n_sampled": n_sampled,
                "italian_share_in_sample": round(share, 4),
                "projected_comments_mar_apr": projected_comments(n_comments, extract_days, discovery_days),
            }
        )
    candidates_df = pd.DataFrame(candidate_rows)

    arm_map = subreddit_arm_map(config)
    lists = subreddit_control_lists(config)
    preview_subs: Dict[str, str] = {}
    for list_key, subs in lists.items():
        if list_key == "discovered_italian":
            continue
        for sub in subs:
            preview_subs[sub] = arm_map.get(sub, list_key)
    for sub in candidates_df["subreddit"].tolist() if not candidates_df.empty else []:
        preview_subs[sub] = "discovered_italian"

    preview_rows: List[Dict[str, Any]] = []
    for subreddit, arm in sorted(preview_subs.items(), key=lambda item: item[0]):
        n_comments = int(counts.get(subreddit, 0))
        preview_rows.append(
            {
                "subreddit": subreddit,
                "arm": arm,
                "n_comments_first_3d": n_comments,
                "projected_comments_mar_apr": projected_comments(n_comments, extract_days, discovery_days),
            }
        )
    preview_df = pd.DataFrame(preview_rows)

    notes = (
        f"source_file={meta.get('source_file')}\n"
        f"lines_seen={meta.get('lines_seen')}\n"
        f"rows_in_window={meta.get('rows_in_window')}\n"
        f"early_stop={meta.get('early_stop')}\n"
        f"ordering_violations={meta.get('ordering_violations')}\n"
        f"first_in_window_utc={meta.get('first_in_window_utc')}\n"
        f"last_in_window_utc={meta.get('last_in_window_utc')}\n"
        f"italian_share_threshold={threshold}\n"
        f"min_comments_for_langid={min_comments}\n"
        f"discovery_days={discovery_days}\n"
        f"extract_days={extract_days}\n"
        f"n_subreddits_seen={len(counts)}\n"
        f"n_italian_candidates={len(candidates_df)}\n"
        f"n_preview_subreddits={len(preview_df)}\n"
        f"projected_total_comments={int(preview_df['projected_comments_mar_apr'].sum()) if not preview_df.empty else 0}\n"
    )
    if int(meta.get("ordering_violations", 0)) > 0:
        notes += (
            "WARNING: created_utc was not monotonic; early_stop may be unsafe. "
            "Re-run discovery without early_stop or verify dump ordering.\n"
        )
    if not meta.get("early_stop"):
        notes += (
            "WARNING: early_stop did not fire; entire source file was scanned. "
            "Check discovery_window vs dump date range.\n"
        )

    return census_df, candidates_df, preview_df, notes


def main() -> None:
    """Function summary: run discovery profiling and write CSV outputs under tables_dir/discovery/."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    discovery_window = config.get("discovery_window", {})
    discovery_cfg = config.get("discovery", {})
    start_ts = utc_ts(str(discovery_window["start_utc"]))
    end_ts = utc_ts(str(discovery_window["end_utc_exclusive"]))
    dump_names = list(discovery_window.get("dump_files", ["RC_2023-03.zst"]))
    source_dir = Path(args.source_dir)
    sample_max = int(discovery_cfg.get("sample_comments_for_langid", 50))
    control_subs = control_subreddits_for_discovery(config)
    rng = random.Random(42)

    tables_dir = Path(config["paths"]["tables_dir"])
    out_dir = tables_dir / "discovery"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_counts: Dict[str, int] = {}
    all_samplers: Dict[str, ReservoirSampler] = {}
    metas: List[Dict[str, Any]] = []

    for name in dump_names:
        source_file = source_dir / name
        if not source_file.exists():
            raise FileNotFoundError(f"Missing discovery dump file: {source_file}")
        counts, samplers, meta = scan_discovery_window(
            source_file, start_ts, end_ts, sample_max, control_subs, rng
        )
        for sub, n in counts.items():
            all_counts[sub] = all_counts.get(sub, 0) + n
        for sub, sampler in samplers.items():
            existing = all_samplers.get(sub)
            if existing is None:
                all_samplers[sub] = sampler
            else:
                for body in sampler.items:
                    existing.add(body)
        metas.append(meta)

    census_df, candidates_df, preview_df, notes = build_outputs(
        config, all_counts, all_samplers, metas[-1] if metas else {}, discovery_cfg
    )

    census_df.to_csv(out_dir / "subreddit_census_3d.csv", index=False)
    candidates_df.to_csv(out_dir / "candidate_italian_subreddits.csv", index=False)
    preview_df.to_csv(out_dir / "extraction_size_preview.csv", index=False)
    (out_dir / "discovery_run_notes.txt").write_text(notes, encoding="utf-8")

    if args.save_langid_samples:
        sample_rows: List[Dict[str, Any]] = []
        for subreddit, sampler in all_samplers.items():
            for body in sampler.items:
                lang, prob = langid.classify(body)
                sample_rows.append(
                    {
                        "subreddit": subreddit,
                        "detected_lang": lang,
                        "detected_lang_prob": prob,
                        "body_preview": body[:200],
                    }
                )
        pd.DataFrame(sample_rows).to_csv(out_dir / "langid_sample_audit.csv", index=False)

    print(f"Wrote discovery tables to {out_dir}")
    print(f"Italian candidates: {len(candidates_df)}")
    print(f"Preview subreddits (controls+seeds+candidates): {len(preview_df)}")
    if not preview_df.empty:
        print(f"Projected total comments (linear): {int(preview_df['projected_comments_mar_apr'].sum())}")


if __name__ == "__main__":
    main()
