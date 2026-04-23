"""
Script summary:
This script filters large Reddit comment dump files (`.zst`) into the project's
analysis-ready day-chunk layout using a two-process pipeline (one worker per month
file). It keeps only configured subreddits, date window, and required fields, then
writes NDJSON outputs per subreddit/day.

Functionality:
- Runs `RC_2022-11.zst` and `RC_2022-12.zst` concurrently in isolated workers.
- Streams compressed dumps without full decompression to disk.
- Uses byte-level subreddit prefiltering before JSON parsing for throughput.
- Uses `orjson` parsing when available, with stdlib fallback.
- Stops each worker early once its file passes the configured relevant time boundary.
- Validates source file fingerprints before trusting resume checkpoints.
- Persists low-cost resume anchors for future fast-start reruns.
- Applies subreddit/date/field filters defined in project config.
- Writes outputs to `data/raw/political_forums/daily_chunks/<subreddit>/<YYYY-MM-DD>.ndjson`.
- Maintains resumable per-worker checkpoints and logs throughput/time-at-data telemetry.
- Produces filtering audit tables from persisted counters without a recount pass.

How to run:
- Ensure dump files exist on external media.
- Run:
  `.venv/bin/python scripts/filter_dump_comments.py --config config/political_forums_setup.yaml`
"""

from __future__ import annotations

import argparse
import io
import json
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import pandas as pd
import zstandard as zstd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_config, utc_ts

try:
    import orjson
except ImportError:  # pragma: no cover - fallback for environments without orjson.
    orjson = None


def serialize_record(record: Dict[str, Any]) -> str:
    """Function summary: serialize one output record to NDJSON text."""
    if orjson is not None:
        return orjson.dumps(record).decode("utf-8")
    return json.dumps(record, ensure_ascii=True)


def parse_args() -> argparse.Namespace:
    """Function summary: parse command line args and return runtime options."""
    parser = argparse.ArgumentParser(description="Filter Reddit dump files into day-chunk outputs.")
    parser.add_argument("--config", type=str, default="config/political_forums_setup.yaml")
    parser.add_argument(
        "--source_dir",
        type=str,
        default="/Volumes/Expansion/Masterthesis/RawData/reddit/comments",
        help="Directory containing RC_2022-11.zst and RC_2022-12.zst.",
    )
    parser.add_argument("--state_file", type=str, default="results/logs/filter_dump_state.json")
    parser.add_argument("--log_file", type=str, default="results/logs/filter_dump.log")
    parser.add_argument("--checkpoint_every", type=int, default=1_000_000)
    parser.add_argument(
        "--resume_from_anchor",
        type=str,
        default="none",
        choices=["none", "first_in_window"],
        help="Optional anchor-based start for reruns. Use with care on existing outputs.",
    )
    return parser.parse_args()


def read_state(path: Path) -> Dict[str, Any]:
    """Function summary: load resumable state from disk or return an initialized state."""
    if not path.exists():
        return {"files": {}, "rows_kept_total": 0}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {"files": {}, "rows_kept_total": 0}
    return json.loads(text)


def write_state(path: Path, state: Dict[str, Any]) -> None:
    """Function summary: persist resumable state to disk in stable JSON format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def append_log(path: Path, message: str) -> None:
    """Function summary: write timestamped progress message to console and log file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def output_path(base_raw_dir: Path, subreddit: str, created_utc: int) -> Path:
    """Function summary: map a comment record to its subreddit/day output file path."""
    date_str = datetime.fromtimestamp(created_utc, tz=timezone.utc).date().isoformat()
    target_dir = base_raw_dir / "daily_chunks" / subreddit
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{date_str}.ndjson"


def trim_record(record: Dict[str, Any], fields: list[str]) -> Dict[str, Any]:
    """Function summary: keep only configured fields from a raw dump record."""
    return {field: record.get(field) for field in fields}


def worker_paths(base_state_path: Path, base_log_path: Path, source_file: Path) -> Tuple[Path, Path]:
    """Function summary: build worker-specific state/log paths to avoid write contention."""
    suffix = source_file.stem
    state_path = base_state_path.with_name(f"{base_state_path.stem}.{suffix}{base_state_path.suffix}")
    log_path = base_log_path.with_name(f"{base_log_path.stem}.{suffix}{base_log_path.suffix}")
    return state_path, log_path


def serialize_counters(counters: Dict[Tuple[str, str], int]) -> Dict[str, int]:
    """Function summary: convert tuple-key counters to JSON-serializable string-key counters."""
    return {f"{subreddit}|{date_utc}": int(rows) for (subreddit, date_utc), rows in counters.items()}


def deserialize_counters(serialized: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    """Function summary: convert JSON counter keys back to (subreddit, date) tuples."""
    out: Dict[Tuple[str, str], int] = {}
    for key, value in serialized.items():
        if "|" not in key:
            continue
        subreddit, date_utc = key.split("|", 1)
        out[(subreddit, date_utc)] = int(value)
    return out


def merge_counters(*counter_dicts: Iterable[Tuple[Tuple[str, str], int]]) -> Dict[Tuple[str, str], int]:
    """Function summary: merge many counter iterables into one summed counter dictionary."""
    merged: Dict[Tuple[str, str], int] = {}
    for counter_items in counter_dicts:
        for key, value in counter_items:
            merged[key] = merged.get(key, 0) + int(value)
    return merged


def parse_candidate(line_bytes: bytes) -> Dict[str, Any]:
    """Function summary: parse one candidate JSON line with fast parser fallback."""
    if orjson is not None:
        return orjson.loads(line_bytes)
    return json.loads(line_bytes.decode("utf-8"))


def monthly_upper_bound_ts(source_file: Path) -> int | None:
    """Function summary: infer exclusive month-end unix timestamp from RC_YYYY-MM filename."""
    stem = source_file.stem
    parts = stem.split("_")
    if len(parts) != 2:
        return None
    ym = parts[1]
    if "-" not in ym:
        return None
    year_str, month_str = ym.split("-", 1)
    try:
        year = int(year_str)
        month = int(month_str)
    except ValueError:
        return None
    if not (1 <= month <= 12):
        return None
    if month == 12:
        next_year = year + 1
        next_month = 1
    else:
        next_year = year
        next_month = month + 1
    return int(datetime(next_year, next_month, 1, tzinfo=timezone.utc).timestamp())


def source_fingerprint(path: Path) -> Dict[str, Any]:
    """Function summary: return low-cost source file fingerprint metadata for resume safety checks."""
    stat_result = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
    }


def fingerprints_match(a: Dict[str, Any] | None, b: Dict[str, Any] | None) -> bool:
    """Function summary: compare two source file fingerprints for strict equality."""
    if not a or not b:
        return False
    return (
        str(a.get("path")) == str(b.get("path"))
        and int(a.get("size_bytes", -1)) == int(b.get("size_bytes", -2))
        and int(a.get("mtime_ns", -1)) == int(b.get("mtime_ns", -2))
    )


def checkpoint_log_message(
    *,
    source_file: Path,
    rows_seen: int,
    rows_kept: int,
    run_started_monotonic: float,
    checkpoint_started_monotonic: float,
    checkpoint_started_rows: int,
    last_created_utc_seen: int | None,
) -> str:
    """Function summary: build checkpoint telemetry message including throughput and latest data timestamp."""
    now_mono = time.monotonic()
    elapsed_total_s = max(now_mono - run_started_monotonic, 1e-9)
    elapsed_checkpoint_s = max(now_mono - checkpoint_started_monotonic, 1e-9)
    avg_lines_per_s = rows_seen / elapsed_total_s
    checkpoint_lines_per_s = (rows_seen - checkpoint_started_rows) / elapsed_checkpoint_s
    if last_created_utc_seen is None:
        latest_utc = "unknown"
    else:
        latest_utc = datetime.fromtimestamp(last_created_utc_seen, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        f"checkpoint file={source_file.name} lines_processed={rows_seen} rows_kept={rows_kept} "
        f"elapsed_total_s={elapsed_total_s:.1f} avg_lines_per_s={avg_lines_per_s:.1f} "
        f"checkpoint_lines_per_s={checkpoint_lines_per_s:.1f} "
        f"last_created_utc_seen={last_created_utc_seen if last_created_utc_seen is not None else 'unknown'} "
        f"last_created_utc_iso={latest_utc}"
    )


def process_file(
    *,
    source_file: Path,
    state_path: Path,
    log_path: Path,
    fields: list[str],
    subreddits: set[str],
    start_ts: int,
    end_ts: int,
    base_raw_dir: Path,
    checkpoint_every: int,
    serialized_subreddit_tokens: list[str],
    resume_from_anchor: str,
) -> Dict[str, Any]:
    """Function summary: stream one dump file, filter records, write day-chunk outputs, and checkpoint progress."""
    state = read_state(state_path)
    file_state = state.setdefault("file", {"completed": False, "lines_processed": 0, "rows_kept": 0})
    counters = deserialize_counters(state.get("counters", {}))
    anchors = state.setdefault("anchors", {})
    current_fingerprint = source_fingerprint(source_file)
    previous_fingerprint = state.get("source_fingerprint")
    if previous_fingerprint and not fingerprints_match(previous_fingerprint, current_fingerprint):
        raise RuntimeError(
            (
                f"Resume fingerprint mismatch for {source_file.name}. "
                "Source file metadata changed; use a new state file or reset existing worker state."
            )
        )
    state["source_fingerprint"] = current_fingerprint

    anchor_line = anchors.get("first_in_window_line")
    use_anchor = resume_from_anchor == "first_in_window" and isinstance(anchor_line, int) and anchor_line >= 0
    if file_state.get("completed", False) and not use_anchor:
        append_log(log_path, f"skip completed file={source_file.name}")
        return {
            "source_file": source_file.name,
            "lines_processed": int(file_state.get("lines_processed", 0)),
            "rows_kept": int(file_state.get("rows_kept", 0)),
            "completed": True,
            "counters": serialize_counters(counters),
            "state_path": str(state_path),
            "log_path": str(log_path),
        }
    if use_anchor and file_state.get("completed", False):
        append_log(
            log_path,
            (
                f"anchor_override_completed file={source_file.name} "
                f"anchor=first_in_window line={int(anchor_line)}"
            ),
        )
        file_state["completed"] = False

    skip_lines = int(file_state.get("lines_processed", 0))
    if use_anchor:
        skip_lines = int(anchor_line)
    append_log(log_path, f"start file={source_file.name} skip_lines={skip_lines}")
    month_end_ts = monthly_upper_bound_ts(source_file)
    hard_end_ts = min(end_ts, month_end_ts) if month_end_ts is not None else end_ts

    rows_seen = 0
    rows_kept = int(file_state.get("rows_kept", 0))
    last_created_utc_seen: int | None = None
    run_started_monotonic = time.monotonic()
    checkpoint_started_monotonic = run_started_monotonic
    checkpoint_started_rows = 0
    serialized_tokens = [token.encode("utf-8") for token in serialized_subreddit_tokens]
    stop_requested = False

    def request_stop(signum: int, frame: Any) -> None:
        """Function summary: mark worker for graceful stop and checkpoint on next loop turn."""
        del frame
        nonlocal stop_requested
        stop_requested = True
        append_log(log_path, f"signal_received file={source_file.name} signal={signum} graceful_stop=true")

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def checkpoint_state() -> None:
        """Function summary: flush writers and persist resumable worker state without completing the file."""
        for handle in writers.values():
            handle.flush()
        file_state["lines_processed"] = rows_seen
        file_state["rows_kept"] = rows_kept
        file_state["completed"] = False
        state["counters"] = serialize_counters(counters)
        state["anchors"] = anchors
        state["source_fingerprint"] = current_fingerprint
        write_state(state_path, state)
        append_log(
            log_path,
            checkpoint_log_message(
                source_file=source_file,
                rows_seen=rows_seen,
                rows_kept=rows_kept,
                run_started_monotonic=run_started_monotonic,
                checkpoint_started_monotonic=checkpoint_started_monotonic,
                checkpoint_started_rows=checkpoint_started_rows,
                last_created_utc_seen=last_created_utc_seen,
            ),
        )

    writers: Dict[Path, Any] = {}
    output_cache: Dict[Tuple[str, str], Path] = {}
    completed = False
    try:
        with source_file.open("rb") as compressed:
            dctx = zstd.ZstdDecompressor(max_window_size=2**31)
            with dctx.stream_reader(compressed) as stream:
                with io.BufferedReader(stream, buffer_size=4 * 1024 * 1024) as buffered_stream:
                    try:
                        for raw_line in buffered_stream:
                            rows_seen += 1
                            if rows_seen <= skip_lines:
                                continue
                            if rows_seen % checkpoint_every == 0:
                                checkpoint_state()
                                checkpoint_started_monotonic = time.monotonic()
                                checkpoint_started_rows = rows_seen
                            if stop_requested:
                                checkpoint_state()
                                append_log(
                                    log_path,
                                    f"graceful_stop file={source_file.name} lines_processed={rows_seen} rows_kept={rows_kept}",
                                )
                                break

                            line_bytes = raw_line.rstrip(b"\r\n")
                            if not line_bytes:
                                continue

                            if not any(token in line_bytes for token in serialized_tokens):
                                continue

                            try:
                                rec = parse_candidate(line_bytes)
                            except Exception:
                                continue

                            subreddit = rec.get("subreddit")
                            if subreddit not in subreddits:
                                continue
                            try:
                                created_utc = int(rec.get("created_utc", 0))
                            except (TypeError, ValueError):
                                continue
                            last_created_utc_seen = created_utc
                            if created_utc < start_ts or created_utc >= end_ts:
                                if created_utc >= hard_end_ts:
                                    checkpoint_state()
                                    append_log(
                                        log_path,
                                        (
                                            f"early_stop file={source_file.name} lines_processed={rows_seen} "
                                            f"rows_kept={rows_kept} created_utc={created_utc} "
                                            f"hard_end_ts={hard_end_ts}"
                                        ),
                                    )
                                    completed = True
                                    break
                                continue

                            if "first_in_window_line" not in anchors:
                                anchors["first_in_window_line"] = rows_seen
                                anchors["first_in_window_created_utc"] = created_utc
                            date_key = datetime.fromtimestamp(created_utc, tz=timezone.utc).date().isoformat()
                            if datetime.fromtimestamp(created_utc, tz=timezone.utc).hour >= 12:
                                noon_map = anchors.setdefault("line_at_noon_utc", {})
                                if isinstance(noon_map, dict) and date_key not in noon_map:
                                    noon_map[date_key] = rows_seen

                            out = trim_record(rec, fields)
                            cache_key = (subreddit, date_key)
                            out_file = output_cache.get(cache_key)
                            if out_file is None:
                                out_file = output_path(base_raw_dir, subreddit, created_utc)
                                output_cache[cache_key] = out_file
                            if out_file not in writers:
                                writers[out_file] = out_file.open("a", encoding="utf-8")
                            writers[out_file].write(serialize_record(out) + "\n")
                            rows_kept += 1

                            counters[(subreddit, date_key)] = counters.get((subreddit, date_key), 0) + 1
                        else:
                            completed = True
                    finally:
                        for handle in writers.values():
                            handle.close()
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)

    file_state["lines_processed"] = rows_seen
    file_state["rows_kept"] = rows_kept
    file_state["completed"] = completed
    state["counters"] = serialize_counters(counters)
    state["rows_kept_total"] = rows_kept
    state["anchors"] = anchors
    state["source_fingerprint"] = current_fingerprint
    write_state(state_path, state)
    if completed:
        append_log(log_path, f"done file={source_file.name} lines_processed={rows_seen} rows_kept={rows_kept}")
    return {
        "source_file": source_file.name,
        "lines_processed": rows_seen,
        "rows_kept": rows_kept,
        "completed": completed,
        "counters": serialize_counters(counters),
        "anchors": anchors,
        "source_fingerprint": current_fingerprint,
        "state_path": str(state_path),
        "log_path": str(log_path),
    }


def process_file_worker(
    source_file: str,
    state_path: str,
    log_path: str,
    fields: list[str],
    subreddits: list[str],
    start_ts: int,
    end_ts: int,
    base_raw_dir: str,
    checkpoint_every: int,
    serialized_subreddit_tokens: list[str],
    resume_from_anchor: str,
) -> Dict[str, Any]:
    """Function summary: execute one file worker in a separate process and return merge metadata."""
    return process_file(
        source_file=Path(source_file),
        state_path=Path(state_path),
        log_path=Path(log_path),
        fields=fields,
        subreddits=set(subreddits),
        start_ts=start_ts,
        end_ts=end_ts,
        base_raw_dir=Path(base_raw_dir),
        checkpoint_every=checkpoint_every,
        serialized_subreddit_tokens=serialized_subreddit_tokens,
        resume_from_anchor=resume_from_anchor,
    )


def main() -> None:
    """Function summary: execute dump filtering for Nov/Dec files and write audit outputs."""
    args = parse_args()
    config = load_config(args.config)

    base_raw_dir = Path(config["paths"]["raw_dir"])
    tables_dir = Path(config["paths"]["tables_dir"])
    source_dir = Path(args.source_dir)
    state_path = Path(args.state_file)
    log_path = Path(args.log_file)
    tables_dir.mkdir(parents=True, exist_ok=True)

    required_files = [source_dir / "RC_2022-11.zst", source_dir / "RC_2022-12.zst"]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required dump files: {missing}")

    fields = list(config["dataset"]["fields"])
    subreddits = set(config["subreddits"]["primary"])
    start_ts = utc_ts(config["event_window"]["start_utc"])
    end_ts = utc_ts(config["event_window"]["end_utc_exclusive"])
    subreddits_sorted = sorted(subreddits)
    serialized_subreddit_tokens: list[str] = []
    for subreddit in subreddits_sorted:
        serialized_subreddit_tokens.append(f"\"subreddit\":\"{subreddit}\"")
        serialized_subreddit_tokens.append(f"\"subreddit\": \"{subreddit}\"")

    append_log(log_path, "start dump filtering run")
    worker_inputs = []
    for source_file in required_files:
        per_worker_state, per_worker_log = worker_paths(state_path, log_path, source_file)
        worker_inputs.append(
            {
                "source_file": str(source_file),
                "state_path": str(per_worker_state),
                "log_path": str(per_worker_log),
                "fields": fields,
                "subreddits": list(subreddits),
                "start_ts": start_ts,
                "end_ts": end_ts,
                "base_raw_dir": str(base_raw_dir),
                "checkpoint_every": int(args.checkpoint_every),
                "serialized_subreddit_tokens": serialized_subreddit_tokens,
                "resume_from_anchor": str(args.resume_from_anchor),
            }
        )

    worker_results = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_file_worker, **params) for params in worker_inputs]
        for future in futures:
            worker_results.append(future.result())

    all_counters = merge_counters(
        *[deserialize_counters(result.get("counters", {})).items() for result in worker_results]
    )
    state = {
        "files": {
            result["source_file"]: {
                "completed": bool(result.get("completed", False)),
                "lines_processed": int(result.get("lines_processed", 0)),
                "rows_kept": int(result.get("rows_kept", 0)),
            }
            for result in sorted(worker_results, key=lambda r: str(r["source_file"]))
        },
        "rows_kept_total": int(sum(int(result.get("rows_kept", 0)) for result in worker_results)),
        "counters": serialize_counters(all_counters),
        "anchors": {result["source_file"]: result.get("anchors", {}) for result in worker_results},
        "source_fingerprints": {
            result["source_file"]: result.get("source_fingerprint", {}) for result in worker_results
        },
        "worker_state_files": [result.get("state_path", "") for result in worker_results],
        "worker_log_files": [result.get("log_path", "") for result in worker_results],
    }
    write_state(state_path, state)

    audit_rows = [
        {"subreddit": subreddit, "date_utc": date_utc, "rows": rows}
        for (subreddit, date_utc), rows in all_counters.items()
    ]

    if audit_rows:
        audit_df = pd.DataFrame(audit_rows).sort_values(["subreddit", "date_utc"]).reset_index(drop=True)
    else:
        audit_df = pd.DataFrame(columns=["subreddit", "date_utc", "rows"])
    audit_path = tables_dir / "dump_filter_counts_by_day.csv"
    audit_df.to_csv(audit_path, index=False)

    summary = (
        audit_df.groupby("subreddit", as_index=False)["rows"].sum().rename(columns={"rows": "rows_total"})
        if not audit_df.empty
        else pd.DataFrame(columns=["subreddit", "rows_total"])
    )
    summary_path = tables_dir / "dump_filter_counts_by_subreddit.csv"
    summary.to_csv(summary_path, index=False)

    append_log(
        log_path,
        f"finished dump filtering run rows_total={int(audit_df['rows'].sum()) if not audit_df.empty else 0}",
    )
    append_log(log_path, f"worker_log_files={[result['log_path'] for result in worker_results]}")
    append_log(log_path, f"audit_day_file={audit_path}")
    append_log(log_path, f"audit_subreddit_file={summary_path}")


if __name__ == "__main__":
    main()
