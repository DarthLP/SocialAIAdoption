"""
Script summary:
Author-level Wordfish v2 — political-universe author×bin docs, token cap, alternating MLE.

Functionality:
- Same assignment/bins as 03b but fit_wordfish_v2, token cap, author-level validation gate.
- Optional EN split: en_us vs en_uk fits (not pooled US+UK).
- Writes to wordfish_authors_v2/ (legacy wordfish_authors/ untouched).

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_wordfish_authors_v2.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/prepare_wordfish_authors_v2.py --spec week7 --panel-mode balanced
  .venv/bin/python scripts/diagnostics/prepare_wordfish_authors_v2.py --reuse-assignment  # skip pass1 lexicon scan
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

# Keep scipy/numpy BLAS on one thread during fits (lower heat, same wall-clock order).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

READ_COLUMNS = [
    "author",
    "subreddit",
    "date_utc",
    "body",
    "primary_lexicon",
    "comment_in_political_universe",
    "is_deleted_author",
    "net_ideology",
    "sem_axis_ideology",
    "n_words",
]

FIT_LANGUAGES = ("it", "en", "de")

PASS1_SCAN_COLUMNS = (
    "author",
    "primary_lexicon",
    "comment_in_political_universe",
    "is_deleted_author",
    "date_utc",
    "net_ideology",
    "sem_axis_ideology",
)
BODY_SCAN_COLUMNS = PASS1_SCAN_COLUMNS + ("subreddit", "body", "n_words")
PROGRESS_EVERY_N_SHARDS = 25


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

from scripts.diagnostics.descriptives_util import event_dates_from_config  # noqa: E402

from src.config_utils import (  # noqa: E402
    load_config,
    load_wordfish_authors_v2_config,
    resolve_primary_subreddits,
    tables_subdir,
)
from src.wordfish import (  # noqa: E402
    DocumentRecord,
    apply_sign_anchor,
    assign_primary_language,
    bin_start_for_day,
    bin_start_for_week,
    build_vocabulary_and_matrix,
    cap_document_tokens,
    compute_center_lang_pre,
    compute_change_outcomes,
    family_dispersion,
    fit_wordfish_v2,
    load_stopwords,
    normalize_lexicon_code,
    parse_anchor_date,
    tokenize_document,
    top_axis_words,
    zscore_preban,
)

US_POLITICAL_SUBREDDITS = frozenset(
    {"Ask_Politics", "NeutralPolitics", "PoliticalDiscussion", "moderpolitics"}
)
UK_POLITICAL_SUBREDDITS = frozenset({"ukpolitics"})


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare author-level Wordfish v2 tables.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--subreddit", type=str, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--language", type=str, default="all", choices=("it", "en", "de", "all"))
    parser.add_argument("--spec", type=str, default="all")
    parser.add_argument("--panel-mode", type=str, default="all", choices=("full", "balanced", "all"))
    parser.add_argument("--drop-cross-language", action="store_true")
    parser.add_argument(
        "--reuse-assignment",
        action="store_true",
        help="Skip pass1 if wordfish_authors_assignment CSV already exists in output dir.",
    )
    return parser.parse_args()


def assign_bin_start_author(
    date_utc: str,
    time_bin: str,
    anchor: Any,
    weekly_days: int,
    window_start: str,
) -> str:
    """Function summary: bin_start for author documents (week/day/window).

    Parameters:
    - date_utc: YYYY-MM-DD.
    - time_bin: week, day, or window.
    - anchor: ban anchor date.
    - weekly_days: block width for week bins.
    - window_start: left edge for whole-window spec.

    Returns:
    - bin_start string.
    """
    if time_bin == "window":
        return window_start
    if time_bin == "day":
        return bin_start_for_day(date_utc)
    if time_bin == "week":
        return bin_start_for_week(date_utc, anchor, weekly_days)
    raise ValueError(f"unknown time_bin: {time_bin}")


def read_parquet_shard_projected(shard: Path, columns: Sequence[str]) -> Optional[pd.DataFrame]:
    """Function summary: read parquet with column projection; skip corrupt/empty files.

    Parameters:
    - shard: path to monthly parquet.
    - columns: desired columns (intersected with file schema).

    Returns:
    - DataFrame or None.
    """
    if not shard.is_file() or shard.stat().st_size < 8:
        return None
    try:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(shard).schema.names)
        use_cols = [c for c in columns if c in available]
        if not use_cols:
            return None
        return pd.read_parquet(shard, columns=use_cols)
    except Exception:
        try:
            df = pd.read_parquet(shard)
            if df is None or df.empty:
                return None
            use_cols = [c for c in columns if c in df.columns]
            return df[use_cols].copy()
        except Exception:
            return None


def enumerate_shard_paths(
    shard_root: Path,
    subreddits: Sequence[str],
    max_shards: Optional[int],
) -> List[Tuple[str, Path]]:
    """Function summary: list (subreddit, parquet path) tasks in stable scan order.

    Parameters:
    - shard_root: cleaned_monthly_chunks root.
    - subreddits: forum list.
    - max_shards: optional per-forum cap.

    Returns:
    - List of (subreddit name, shard path).
    """
    tasks: List[Tuple[str, Path]] = []
    for sub in subreddits:
        shard_dir = shard_root / sub
        if not shard_dir.is_dir():
            continue
        shards = sorted(shard_dir.glob("*.parquet"))
        if max_shards is not None:
            shards = shards[: max_shards]
        for shard in shards:
            tasks.append((sub, shard))
    return tasks


def filter_political_chunk(
    df: Optional[pd.DataFrame],
    subreddit: str,
    start: str,
    end_excl: str,
) -> Optional[pd.DataFrame]:
    """Function summary: political-universe filter and date window on a projected shard.

    Parameters:
    - df: raw projected frame or None.
    - subreddit: forum name to attach.
    - start: inclusive window start YYYY-MM-DD.
    - end_excl: exclusive window end.

    Returns:
    - Filtered frame or None if empty/unusable.
    """
    if df is None or df.empty:
        return None
    if "comment_in_political_universe" not in df.columns:
        return None
    chunk = df.copy()
    chunk["subreddit"] = subreddit
    chunk = chunk[chunk["comment_in_political_universe"].astype(bool)]
    if "is_deleted_author" in chunk.columns:
        chunk = chunk[~chunk["is_deleted_author"].fillna(False).astype(bool)]
    if "date_utc" not in chunk.columns:
        return None
    chunk["date_utc"] = chunk["date_utc"].astype(str).str[:10]
    chunk = chunk[(chunk["date_utc"] >= start) & (chunk["date_utc"] < end_excl)]
    if chunk.empty:
        return None
    return chunk


def iter_comment_chunks(
    shard_root: Path,
    subreddits: Sequence[str],
    max_shards: Optional[int],
    start: str,
    end_excl: str,
    columns: Optional[Sequence[str]] = None,
):
    """Function summary: yield filtered political comment chunks from shards.

    Parameters:
    - shard_root: cleaned_monthly_chunks root.
    - subreddits: forum list.
    - max_shards: optional per-forum cap.
    - start: window start YYYY-MM-DD.
    - end_excl: exclusive end date.
    - columns: optional column projection (defaults to READ_COLUMNS).

    Yields:
    - DataFrame chunks with required columns.
    """
    use_cols = tuple(columns or READ_COLUMNS)
    for sub, shard in enumerate_shard_paths(shard_root, subreddits, max_shards):
        raw = read_parquet_shard_projected(shard, use_cols)
        chunk = filter_political_chunk(raw, sub, start, end_excl)
        if chunk is not None:
            yield chunk


def accumulate_pass1_chunk(
    chunk: pd.DataFrame,
    author_langs: DefaultDict[str, Set[str]],
    author_counts: DefaultDict[str, int],
) -> None:
    """Function summary: update author language sets from one filtered chunk.

    Parameters:
    - chunk: filtered political comments.
    - author_langs: mutable author -> lexicon set map.
    - author_counts: mutable author comment counts.

    Returns:
    - None (updates maps in place).
    """
    authors = chunk["author"].astype(str)
    lexes = chunk["primary_lexicon"].astype(str).map(normalize_lexicon_code)
    for author, lex in zip(authors, lexes):
        if not author or author == "nan":
            continue
        author_langs[author].add(lex)
        author_counts[author] += 1


def accumulate_ideol_shard_parts(
    chunk: pd.DataFrame,
    ideol_parts: List[pd.DataFrame],
) -> None:
    """Function summary: append per-shard author ideology means (same logic as legacy second pass).

    Parameters:
    - chunk: filtered chunk with optional net_ideology / sem_axis_ideology.
    - ideol_parts: list collecting per-shard groupby frames.

    Returns:
    - None.
    """
    if "net_ideology" not in chunk.columns:
        return
    part = (
        chunk.groupby("author", as_index=False)
        .agg(
            net_ideology_mean=("net_ideology", "mean"),
            sem_axis_ideology_mean=("sem_axis_ideology", "mean"),
        )
    )
    ideol_parts.append(part)


def finalize_comments_ideol(ideol_parts: List[pd.DataFrame]) -> pd.DataFrame:
    """Function summary: merge per-shard ideology means into one author table.

    Parameters:
    - ideol_parts: shard-level author mean frames.

    Returns:
    - Author-level ideology dataframe.
    """
    if not ideol_parts:
        return pd.DataFrame()
    comments_ideol = pd.concat(ideol_parts, ignore_index=True)
    return (
        comments_ideol.groupby("author", as_index=False)
        .agg(
            net_ideology_mean=("net_ideology_mean", "mean"),
            sem_axis_ideology_mean=("sem_axis_ideology_mean", "mean"),
        )
        .reset_index(drop=True)
    )


def vectorized_bin_start_series(
    dates: pd.Series,
    time_bin: str,
    anchor: Any,
    weekly_days: int,
    window_start: str,
) -> pd.Series:
    """Function summary: vectorized bin_start for a comment date column.

    Parameters:
    - dates: YYYY-MM-DD strings.
    - time_bin: week, day, or window.
    - anchor: ban anchor date.
    - weekly_days: week block width.
    - window_start: whole-window label.

    Returns:
    - Series of bin_start strings aligned to dates.index.
    """
    dates = dates.astype(str).str[:10]
    if time_bin == "window":
        return pd.Series(window_start, index=dates.index)
    if time_bin == "day":
        return dates.map(bin_start_for_day)
    if time_bin == "week":
        return dates.map(lambda d: bin_start_for_week(d, anchor, weekly_days))
    raise ValueError(f"unknown time_bin: {time_bin}")


def accumulate_spec_bodies_chunk(
    chunk: pd.DataFrame,
    spec: Dict[str, Any],
    author_assigned: Dict[str, str],
    author_langs: Dict[str, Set[str]],
    filter_to_assigned: bool,
    drop_cross_language: bool,
    anchor: Any,
    window_start: str,
    bodies: DefaultDict[Tuple[str, str, str], List[str]],
    meta: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> None:
    """Function summary: groupby-append comment bodies into author×bin×assigned-lang buckets.

    Parameters:
    - chunk: filtered political chunk with body column.
    - spec: time_bins entry.
    - author_assigned: author -> fit language.
    - author_langs: author -> lexicons seen (cross-lang filter).
    - filter_to_assigned: keep only comments matching assigned lexicon.
    - drop_cross_language: exclude multi-lexicon authors.
    - anchor: ban anchor for week bins.
    - window_start: window spec left edge.
    - bodies: mutable (author, bin_start, lang) -> body strings.
    - meta: mutable per-key coverage metadata.

    Returns:
    - None.
    """
    work = chunk.copy()
    work["author"] = work["author"].astype(str)
    work["lex"] = work["primary_lexicon"].astype(str).map(normalize_lexicon_code)
    valid = work["author"].notna() & (work["author"] != "") & (work["author"] != "nan")
    work = work.loc[valid]
    work["assigned"] = work["author"].map(author_assigned)
    work = work.loc[work["assigned"].notna()]
    if drop_cross_language:
        cross = {a for a, langs in author_langs.items() if len(langs) >= 2}
        work = work.loc[~work["author"].isin(cross)]
    if filter_to_assigned:
        work = work.loc[work["lex"] == work["assigned"]]
    if work.empty:
        return

    time_bin = str(spec["time_bin"])
    weekly_days = int(spec.get("weekly_bin_days", 7))
    work["bin_start"] = vectorized_bin_start_series(
        work["date_utc"], time_bin, anchor, weekly_days, window_start
    )

    grp = work.groupby(["author", "bin_start", "assigned"], sort=False)
    for (author, bin_start, lang), g in grp:
        key = (str(author), str(bin_start), str(lang))
        bodies[key].extend(g["body"].fillna("").astype(str).tolist())
        if key not in meta:
            meta[key] = {
                "dates": set(),
                "subreddits": Counter(),
                "n_words_proxy": 0,
                "n_comments": 0,
            }
        m = meta[key]
        m["dates"].update(g["date_utc"].astype(str).str[:10].unique())
        m["subreddits"].update(g["subreddit"].astype(str).value_counts().to_dict())
        if "n_words" in g.columns:
            m["n_words_proxy"] += int(
                pd.to_numeric(g["n_words"], errors="coerce").fillna(0).sum()
            )
        m["n_comments"] += len(g)


@dataclass
class ShardScanResult:
    """Function summary: outputs from a two-phase single-list shard scan."""

    author_langs: Dict[str, Set[str]]
    author_counts: Dict[str, int]
    comments_ideol: pd.DataFrame
    bodies_by_spec: Dict[str, DefaultDict[Tuple[str, str, str], List[str]]]
    meta_by_spec: Dict[str, Dict[Tuple[str, str, str], Dict[str, Any]]]


def scan_shards_for_wordfish(
    shard_root: Path,
    subreddits: Sequence[str],
    max_shards: Optional[int],
    start: str,
    end_excl: str,
    specs: Sequence[Dict[str, Any]],
    wfa_cfg: Dict[str, Any],
    priority: Sequence[str],
    author_assigned: Optional[Dict[str, str]] = None,
    author_langs: Optional[Dict[str, Set[str]]] = None,
    *,
    skip_pass1_langs: bool = False,
    progress: bool = True,
) -> ShardScanResult:
    """Function summary: two-phase shard scan (pass1+ideol, then all spec bodies); one read per phase.

    Parameters:
    - shard_root: interim shard root.
    - subreddits: forums to scan.
    - max_shards: optional per-forum cap.
    - start, end_excl: event window.
    - specs: time_bin spec dicts for body aggregation.
    - wfa_cfg: wordfish authors config.
    - priority: language priority for pass1 assignment when not reusing CSV.
    - author_assigned: precomputed assignment (reuse-assignment); built after pass1 if None.
    - author_langs: precomputed langs when skip_pass1_langs.
    - skip_pass1_langs: if True, skip lexicon assignment scan (still scan ideology).
    - progress: print shard progress every N files.

    Returns:
    - ShardScanResult with pass1, ideology, and per-spec body buffers.
    """
    tasks = enumerate_shard_paths(shard_root, subreddits, max_shards)
    total = len(tasks)
    anchor = parse_anchor_date(str(wfa_cfg["ban_anchor_date"]))
    filter_assigned = bool(wfa_cfg.get("filter_comments_to_assigned_lang", True))
    drop_cross_language = bool(wfa_cfg.get("drop_cross_language", False))

    langs_acc: DefaultDict[str, Set[str]] = defaultdict(set)
    counts_acc: DefaultDict[str, int] = defaultdict(int)
    if author_langs:
        for a, ls in author_langs.items():
            langs_acc[a] = set(ls)
    ideol_parts: List[pd.DataFrame] = []

    phase1_label = "ideology" if skip_pass1_langs else "pass1+ideology"
    print(
        f"[prepare_wordfish_authors_v2] scan phase 1/2: {phase1_label} ({total} shards)",
        flush=True,
    )
    for idx, (sub, shard) in enumerate(tasks, start=1):
        raw = read_parquet_shard_projected(shard, PASS1_SCAN_COLUMNS)
        chunk = filter_political_chunk(raw, sub, start, end_excl)
        if chunk is not None:
            if not skip_pass1_langs:
                accumulate_pass1_chunk(chunk, langs_acc, counts_acc)
            accumulate_ideol_shard_parts(chunk, ideol_parts)
        if progress and (idx == 1 or idx % PROGRESS_EVERY_N_SHARDS == 0 or idx == total):
            print(
                f"[prepare_wordfish_authors_v2] phase 1: shard {idx}/{total} forum={sub}",
                flush=True,
            )

    if skip_pass1_langs:
        if author_langs is None:
            raise ValueError("skip_pass1_langs requires author_langs")
        author_langs_out = dict(author_langs)
        author_counts_out = {}
    else:
        author_langs_out = dict(langs_acc)
        author_counts_out = dict(counts_acc)

    if author_assigned is None:
        author_assigned = {
            author: assign_primary_language(langs, priority)
            for author, langs in author_langs_out.items()
        }

    bodies_by_spec: Dict[str, DefaultDict[Tuple[str, str, str], List[str]]] = {
        str(s["name"]): defaultdict(list) for s in specs
    }
    meta_by_spec: Dict[str, Dict[Tuple[str, str, str], Dict[str, Any]]] = {
        str(s["name"]): {} for s in specs
    }

    print(
        f"[prepare_wordfish_authors_v2] scan phase 2/2: bodies ({total} shards, {len(specs)} specs)",
        flush=True,
    )
    for idx, (sub, shard) in enumerate(tasks, start=1):
        raw = read_parquet_shard_projected(shard, BODY_SCAN_COLUMNS)
        chunk = filter_political_chunk(raw, sub, start, end_excl)
        if chunk is not None:
            for spec in specs:
                spec_name = str(spec["name"])
                accumulate_spec_bodies_chunk(
                    chunk,
                    spec,
                    author_assigned,
                    author_langs_out,
                    filter_assigned,
                    drop_cross_language,
                    anchor,
                    start,
                    bodies_by_spec[spec_name],
                    meta_by_spec[spec_name],
                )
        if progress and (idx == 1 or idx % PROGRESS_EVERY_N_SHARDS == 0 or idx == total):
            print(
                f"[prepare_wordfish_authors_v2] phase 2: shard {idx}/{total} forum={sub}",
                flush=True,
            )

    return ShardScanResult(
        author_langs=author_langs_out,
        author_counts=author_counts_out,
        comments_ideol=finalize_comments_ideol(ideol_parts),
        bodies_by_spec=bodies_by_spec,
        meta_by_spec=meta_by_spec,
    )


def pass1_author_languages(
    shard_root: Path,
    subreddits: Sequence[str],
    max_shards: Optional[int],
    start: str,
    end_excl: str,
) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    """Function summary: collect political lexicons seen per author.

    Parameters:
    - shard_root: interim shard root.
    - subreddits: forums to scan.
    - max_shards: optional cap.
    - start: window start.
    - end_excl: exclusive end.

    Returns:
    - Tuple (author -> set of normalized lexicons, author raw comment counts).
    """
    author_langs: DefaultDict[str, Set[str]] = defaultdict(set)
    author_counts: DefaultDict[str, int] = defaultdict(int)
    for chunk in iter_comment_chunks(
        shard_root, subreddits, max_shards, start, end_excl, columns=PASS1_SCAN_COLUMNS
    ):
        accumulate_pass1_chunk(chunk, author_langs, author_counts)
    return dict(author_langs), dict(author_counts)


def load_assignment_from_csv(path: Path) -> Tuple[Dict[str, Set[str]], Dict[str, str], pd.DataFrame]:
    """Function summary: load author assignment table written by a prior run.

    Parameters:
    - path: wordfish_authors_assignment CSV path.

    Returns:
    - Tuple (author_langs, author_assigned, assignment dataframe).
    """
    assignment = pd.read_csv(path)
    author_langs: Dict[str, Set[str]] = {}
    author_assigned: Dict[str, str] = {}
    for _, row in assignment.iterrows():
        author = str(row["author"])
        author_assigned[author] = str(row["assigned_primary_lexicon"])
        seen = str(row.get("lexicons_seen", "") or "")
        author_langs[author] = set(seen.split(";")) if seen else set()
    return author_langs, author_assigned, assignment


def documents_from_buffers(
    bodies: DefaultDict[Tuple[str, str, str], List[str]],
    meta: Dict[Tuple[str, str, str], Dict[str, Any]],
    spec: Dict[str, Any],
    wfa_cfg: Dict[str, Any],
    stopwords_by_lang: Dict[str, Set[str]],
) -> Tuple[List[DocumentRecord], pd.DataFrame]:
    """Function summary: tokenize aggregated bodies into DocumentRecords for one spec.

    Parameters:
    - bodies: (author, bin_start, lang) -> comment bodies.
    - meta: per-key coverage metadata.
    - spec: time_bins entry.
    - wfa_cfg: merged wordfish_authors config.
    - stopwords_by_lang: language -> stopword set.

    Returns:
    - Tuple (document records, coverage audit).
    """
    time_bin = str(spec["time_bin"])
    min_doc_tokens = int(spec["min_doc_tokens"])
    min_token_len = int(wfa_cfg["min_token_len"])
    spec_name = str(spec["name"])
    max_tok = int(wfa_cfg.get("max_tokens_per_doc", 0))
    seed = int(wfa_cfg.get("token_subsample_seed", 42))

    docs: List[DocumentRecord] = []
    coverage_rows: List[Dict[str, Any]] = []

    for (author, bin_start, lang), text_parts in bodies.items():
        stopwords = stopwords_by_lang.get(lang, set())
        text = " ".join(text_parts)
        tokens = tokenize_document(text, stopwords, min_token_len)
        n_tokens_raw = len(tokens)
        truncated = False
        if max_tok > 0:
            tokens, n_tokens_raw, truncated = cap_document_tokens(tokens, max_tok, seed)
        n_tokens = len(tokens)
        m = meta.get((author, bin_start, lang))
        if m is None:
            continue
        modal_sub = m["subreddits"].most_common(1)[0][0] if m["subreddits"] else ""
        doc_id = f"{author}|{bin_start}"
        rec = DocumentRecord(
            doc_id=doc_id,
            subreddit=modal_sub,
            topic_family="",
            primary_lexicon=lang,
            bin_start=bin_start,
            time_bin=time_bin,
            n_days_in_bin=len(m["dates"]),
            n_tokens=n_tokens,
            tokens=tokens,
            author=author,
            n_words_proxy=int(m["n_words_proxy"]),
        )
        kept = n_tokens >= min_doc_tokens
        if kept:
            docs.append(rec)
        coverage_rows.append(
            {
                "spec": spec_name,
                "author": author,
                "primary_lexicon": lang,
                "time_bin": time_bin,
                "bin_start": bin_start,
                "n_comments": m["n_comments"],
                "n_days_in_bin": len(m["dates"]),
                "n_words_proxy": m["n_words_proxy"],
                "n_tokens": n_tokens,
                "n_tokens_raw": n_tokens_raw,
                "tokens_truncated": int(truncated),
                "doc_kept": kept,
            }
        )

    return docs, pd.DataFrame(coverage_rows)


def build_author_assignment_table(
    author_langs: Dict[str, Set[str]],
    priority: Sequence[str],
) -> pd.DataFrame:
    """Function summary: primary language per author with cross-language flags.

    Parameters:
    - author_langs: author -> lexicon set.
    - priority: it > de > en order.

    Returns:
    - Assignment audit dataframe.
    """
    rows: List[Dict[str, Any]] = []
    for author, langs in sorted(author_langs.items()):
        assigned = assign_primary_language(langs, priority)
        rows.append(
            {
                "author": author,
                "assigned_primary_lexicon": assigned,
                "n_lexicons_seen": len(langs),
                "lexicons_seen": ";".join(sorted(langs)),
                "cross_language": int(len(langs) >= 2),
                "reassigned_by_priority": int(
                    len(langs) >= 2 and assigned != sorted(langs)[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_author_documents(
    shard_root: Path,
    subreddits: Sequence[str],
    max_shards: Optional[int],
    start: str,
    end_excl: str,
    author_assigned: Dict[str, str],
    spec: Dict[str, Any],
    wfa_cfg: Dict[str, Any],
    stopwords: Set[str],
    filter_to_assigned: bool,
    drop_cross_language: bool,
    author_langs: Dict[str, Set[str]],
    window_start: str,
    target_lang: Optional[str] = None,
) -> Tuple[List[DocumentRecord], pd.DataFrame]:
    """Function summary: stream shards into author×bin documents for one time spec.

    Parameters:
    - shard_root: shard path.
    - subreddits: forums.
    - max_shards: cap.
    - start, end_excl: event window.
    - author_assigned: author -> fit language.
    - spec: time_bins entry (name, time_bin, weekly_bin_days, min_doc_tokens).
    - wfa_cfg: merged wordfish_authors config.
    - stopwords: language stopwords for token_lang (or sole language in buffers).
    - filter_to_assigned: keep only assigned-lang comments.
    - drop_cross_language: exclude multi-lexicon authors.
    - author_langs: for cross-lang filter.
    - window_start: whole-window bin label.
    - target_lang: if set, only build documents for authors assigned to this language.

    Returns:
    - Tuple (document records, coverage audit).
    """
    anchor = parse_anchor_date(str(wfa_cfg["ban_anchor_date"]))
    bodies: DefaultDict[Tuple[str, str, str], List[str]] = defaultdict(list)
    meta: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    filter_assigned = filter_to_assigned

    for chunk in iter_comment_chunks(
        shard_root, subreddits, max_shards, start, end_excl, columns=BODY_SCAN_COLUMNS
    ):
        work = chunk.copy()
        if target_lang is not None:
            work["assigned"] = work["author"].astype(str).map(author_assigned)
            work = work.loc[work["assigned"] == target_lang]
            if work.empty:
                continue
        accumulate_spec_bodies_chunk(
            work,
            spec,
            author_assigned,
            author_langs,
            filter_assigned,
            drop_cross_language,
            anchor,
            window_start,
            bodies,
            meta,
        )

    langs = {k[2] for k in bodies}
    token_lang = target_lang or (next(iter(langs)) if len(langs) == 1 else "")
    stopwords_by_lang = {lang: stopwords if lang == token_lang else set() for lang in langs}
    docs, cov = documents_from_buffers(bodies, meta, spec, wfa_cfg, stopwords_by_lang)
    if target_lang is not None:
        docs = [d for d in docs if d.primary_lexicon == target_lang]
        if not cov.empty:
            cov = cov[cov["primary_lexicon"] == target_lang].copy()
    return docs, cov


def filter_docs_by_subreddit_set(
    docs: List[DocumentRecord],
    allowed: Set[str],
) -> List[DocumentRecord]:
    """Function summary: keep documents whose modal subreddit is in allowed set.

    Parameters:
    - docs: author documents.
    - allowed: subreddit names.

    Returns:
    - Filtered list.
    """
    return [d for d in docs if d.subreddit in allowed]


def filter_balanced_authors(
    docs: List[DocumentRecord],
    anchor_date: str,
) -> Set[str]:
    """Function summary: authors with at least one pre and one ban kept document.

    Parameters:
    - docs: surviving documents.
    - anchor_date: t* YYYY-MM-DD.

    Returns:
    - Set of balanced author ids.
    """
    pre: Set[str] = set()
    ban: Set[str] = set()
    for d in docs:
        if d.bin_start < anchor_date:
            pre.add(d.author)
        else:
            ban.add(d.author)
    return pre & ban


def filter_docs_by_authors(
    docs: List[DocumentRecord],
    authors: Set[str],
) -> List[DocumentRecord]:
    """Function summary: restrict documents to author subset.

    Parameters:
    - docs: full document list.
    - authors: allowed author ids.

    Returns:
    - Filtered documents.
    """
    return [d for d in docs if d.author in authors]


def run_single_fit(
    docs: List[DocumentRecord],
    wfa_cfg: Dict[str, Any],
    anchor_sub: str,
    time_bin: str,
) -> Tuple[Optional[Any], List[Dict[str, str]]]:
    """Function summary: fit Wordfish on author document list.

    Parameters:
    - docs: surviving documents.
    - wfa_cfg: config dict.
    - anchor_sub: anchor forum for sign flip.
    - time_bin: week/day/window.

    Returns:
    - Tuple (WordfishFitResult or None, doc_meta).
    """
    if len(docs) < 2:
        return None, []
    doc_tokens = [d.tokens for d in docs]
    mat, vocab = build_vocabulary_and_matrix(
        doc_tokens,
        min_doc_freq=int(wfa_cfg["min_doc_freq"]),
        top_freq_drop_n=int(wfa_cfg["top_freq_drop_n"]),
        max_vocab_terms=int(wfa_cfg.get("max_vocab_terms", 5000)),
    )
    if mat.shape[1] == 0:
        return None, []

    doc_ids = [d.doc_id for d in docs]
    doc_meta = [{"subreddit": d.subreddit, "doc_id": d.doc_id, "author": d.author} for d in docs]
    result = fit_wordfish_v2(
        mat,
        vocab,
        doc_ids,
        convergence_cfg=wfa_cfg.get("convergence", {}),
    )
    result = apply_sign_anchor(result, doc_meta, anchor_sub)
    return result, doc_meta


def author_positions_and_panels(
    docs: List[DocumentRecord],
    result: Any,
    wfa_cfg: Dict[str, Any],
    panel_mode: str,
    spec_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Function summary: build author positions, extremity, dispersion panels.

    Parameters:
    - docs: fitted documents.
    - result: WordfishFitResult.
    - wfa_cfg: config.
    - panel_mode: full or balanced.
    - spec_name: week7, week3, window.

    Returns:
    - Tuple (positions, extremity, dispersion).
    """
    anchor_date = str(wfa_cfg["ban_anchor_date"])
    doc_by_id = {d.doc_id: d for d in docs}
    pos_rows: List[Dict[str, Any]] = []

    for i, doc_id in enumerate(result.doc_ids):
        d = doc_by_id[doc_id]
        pos_rows.append(
            {
                "author": d.author,
                "subreddit_modal": d.subreddit,
                "primary_lexicon": d.primary_lexicon,
                "bin_start": d.bin_start,
                "time_bin": d.time_bin,
                "spec": spec_name,
                "panel_mode": panel_mode,
                "n_days_in_bin": d.n_days_in_bin,
                "theta": float(result.theta[i]),
                "n_tokens": d.n_tokens,
            }
        )

    pos_df = pd.DataFrame(pos_rows)
    center = compute_center_lang_pre(
        pos_df["bin_start"].tolist(),
        pos_df["theta"].tolist(),
        anchor_date,
    )
    ext_vals = [
        abs(float(t) - center) if not np.isnan(center) else float("nan")
        for t in pos_df["theta"]
    ]
    pre_mu, pre_sd = zscore_preban(ext_vals, pos_df["bin_start"].tolist(), anchor_date)

    ext_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(pos_df.to_dict("records")):
        extremity = ext_vals[i]
        if not np.isnan(pre_mu) and not np.isnan(extremity) and not np.isnan(pre_sd):
            extremity_z = (extremity - pre_mu) / pre_sd
        else:
            extremity_z = float("nan")
        ext_rows.append(
            {
                "author": row["author"],
                "primary_lexicon": row["primary_lexicon"],
                "bin_start": row["bin_start"],
                "time_bin": row["time_bin"],
                "spec": spec_name,
                "panel_mode": panel_mode,
                "center_lang_pre": center,
                "theta": row["theta"],
                "extremity": extremity,
                "extremity_z": extremity_z,
                "post": int(row["bin_start"] >= anchor_date),
                "IT": int(row["primary_lexicon"] == "it"),
                "n_tokens": int(row["n_tokens"]),
                "n_days_in_bin": int(row["n_days_in_bin"]),
            }
        )

    ext_df = pd.DataFrame(ext_rows)
    pre_mask = ext_df["bin_start"].astype(str) < anchor_date
    author_theta_pre = (
        ext_df.loc[pre_mask]
        .groupby("author")["theta"]
        .agg(pre_n="count", author_pre_mean_theta="mean")
    )
    ext_df = ext_df.merge(author_theta_pre, on="author", how="left")
    ext_df["extremity_within_author"] = np.where(
        ext_df["pre_n"].fillna(0) >= 2,
        (ext_df["theta"] - ext_df["author_pre_mean_theta"]).abs(),
        np.nan,
    )
    lang_pre_stats: Dict[str, tuple[float, float]] = {}
    for lang, grp in ext_df.loc[pre_mask].groupby("primary_lexicon"):
        vals = grp["extremity_within_author"].dropna()
        if len(vals) >= 2:
            lang_pre_stats[str(lang)] = (float(vals.mean()), float(vals.std()))
    ext_df["extremity_within_author_z"] = np.nan
    for author, grp in ext_df.groupby("author"):
        if float(grp["pre_n"].iloc[0] if "pre_n" in grp.columns else 0) < 2:
            continue
        pre_a = grp[grp["bin_start"].astype(str) < anchor_date]["extremity_within_author"].dropna()
        lang = str(grp["primary_lexicon"].iloc[0])
        if len(pre_a) >= 2 and pre_a.std() > 0:
            mu_a, sd_a = float(pre_a.mean()), float(pre_a.std())
        elif lang in lang_pre_stats and lang_pre_stats[lang][1] > 0:
            mu_a, sd_a = lang_pre_stats[lang]
        else:
            continue
        ext_df.loc[grp.index, "extremity_within_author_z"] = (
            grp["extremity_within_author"] - mu_a
        ) / sd_a
    ext_df = ext_df.drop(columns=["pre_n", "author_pre_mean_theta"], errors="ignore")
    rolling_w = int(wfa_cfg.get("rolling_bins_w", 2))
    ext_df = compute_change_outcomes(ext_df, anchor_date, rolling_w, group_col="author")

    disp_rows: List[Dict[str, Any]] = []
    for (lang, bin_start, tbin, pspec, pmode), grp in ext_df.groupby(
        ["primary_lexicon", "bin_start", "time_bin", "spec", "panel_mode"],
        dropna=False,
    ):
        thetas = grp["theta"].astype(float).tolist()
        disp = family_dispersion(thetas)
        disp_rows.append(
            {
                "primary_lexicon": lang,
                "bin_start": bin_start,
                "time_bin": tbin,
                "spec": pspec,
                "panel_mode": pmode,
                "dispersion_var": disp["dispersion_var"],
                "dispersion_iqr": disp["dispersion_iqr"],
                "dispersion_range": disp["dispersion_range"],
                "n_authors": len(grp),
                "post": int(bin_start >= anchor_date),
                "IT": int(lang == "it"),
            }
        )
    disp_df = pd.DataFrame(disp_rows)
    return pos_df, ext_df, disp_df


def author_validation_correlations(
    positions: pd.DataFrame,
    comments_by_author: pd.DataFrame,
) -> pd.DataFrame:
    """Function summary: Spearman author-mean theta vs lexicon means.

    Parameters:
    - positions: positions panel.
    - comments_by_author: author-level ideology aggregates.

    Returns:
    - Correlation summary per language/spec/panel_mode.
    """
    if positions.empty or comments_by_author.empty:
        return pd.DataFrame()
    auth_theta = positions.groupby(
        ["author", "primary_lexicon", "spec", "panel_mode"], as_index=False
    )["theta"].mean()
    merged = auth_theta.merge(comments_by_author, on="author", how="inner")
    rows: List[Dict[str, Any]] = []
    for (lang, spec, pmode), grp in merged.groupby(
        ["primary_lexicon", "spec", "panel_mode"]
    ):
        if len(grp) < 10:
            continue
        rows.append(
            {
                "primary_lexicon": lang,
                "spec": spec,
                "panel_mode": pmode,
                "n_authors": len(grp),
                "spearman_theta_net_ideology": float(
                    grp["theta"].corr(grp["net_ideology_mean"], method="spearman")
                ),
                "spearman_theta_sem_axis": float(
                    grp["theta"].corr(grp["sem_axis_ideology_mean"], method="spearman")
                ),
            }
        )
    return pd.DataFrame(rows)


def author_validation_gate(
    validation: pd.DataFrame,
    wfa_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Function summary: pre-registered |rho| gate for author theta vs sem_axis.

    Parameters:
    - validation: rows from author_validation_correlations.
    - wfa_cfg: config with validation.* keys.

    Returns:
    - Gate summary with gate_pass and recommendation text.
    """
    val_cfg = wfa_cfg.get("validation") or {}
    threshold = float(val_cfg.get("gate_abs_rho_sem_axis", 0.5))
    min_auth = int(val_cfg.get("min_authors", 100))
    rows: List[Dict[str, Any]] = []
    for _, row in validation.iterrows():
        n = int(row.get("n_authors", 0))
        rho_sem = row.get("spearman_theta_sem_axis")
        rho_net = row.get("spearman_theta_net_ideology")
        abs_sem = abs(float(rho_sem)) if rho_sem == rho_sem else float("nan")
        abs_net = abs(float(rho_net)) if rho_net == rho_net else float("nan")
        passed = (n >= min_auth) and (abs_sem >= threshold)
        rows.append(
            {
                "primary_lexicon": row["primary_lexicon"],
                "spec": row["spec"],
                "panel_mode": row["panel_mode"],
                "n_authors": n,
                "abs_spearman_theta_sem_axis": abs_sem,
                "abs_spearman_theta_net_ideology": abs_net,
                "spearman_theta_sem_axis": rho_sem,
                "spearman_theta_net_ideology": rho_net,
                "gate_threshold": threshold,
                "min_authors": min_auth,
                "gate_pass": int(passed),
                "recommendation": (
                    "use_author_theta_for_triangulation"
                    if passed
                    else "failed_triangulation_do_not_use_theta_as_ideology"
                ),
            }
        )
    return pd.DataFrame(rows)


def author_validation_by_author(
    positions: pd.DataFrame,
    comments_by_author: pd.DataFrame,
) -> pd.DataFrame:
    """Function summary: author-level theta vs internal ideology measures for scatter QA.

    Parameters:
    - positions: positions panel.
    - comments_by_author: author aggregates.

    Returns:
    - Merged author-level validation frame.
    """
    if positions.empty or comments_by_author.empty:
        return pd.DataFrame()
    auth_theta = positions.groupby(
        ["author", "primary_lexicon", "spec", "panel_mode"], as_index=False
    )["theta"].mean()
    merged = auth_theta.merge(comments_by_author, on="author", how="inner")
    merged["measure_note"] = "net_ideology_and_sem_axis_are_internal_constructed_scores"
    return merged


def stability_across_specs(
    positions_week7: pd.DataFrame,
    positions_week3: pd.DataFrame,
) -> pd.DataFrame:
    """Function summary: rank correlation of author-mean theta week7 vs week3 (balanced).

    Parameters:
    - positions_week7: balanced week7 positions.
    - positions_week3: balanced week3 positions.

    Returns:
    - Stability summary row(s).
    """
    if positions_week7.empty or positions_week3.empty:
        return pd.DataFrame()
    m7 = positions_week7.groupby(["author", "primary_lexicon"])["theta"].mean()
    m3 = positions_week3.groupby(["author", "primary_lexicon"])["theta"].mean()
    joined = pd.DataFrame({"week7": m7, "week3": m3}).dropna()
    if len(joined) < 5:
        return pd.DataFrame()
    rho = float(joined["week7"].corr(joined["week3"], method="spearman"))
    return pd.DataFrame(
        [
            {
                "comparison": "week7_vs_week3_balanced",
                "spearman_rank_rho_author_mean_theta": rho,
                "n_authors": len(joined),
            }
        ]
    )


def build_run_notes(
    wfa_cfg: Dict[str, Any],
    assignment: pd.DataFrame,
    fit_lines: List[str],
    headline_path: str,
) -> List[str]:
    """Function summary: assemble wordfish_authors_run_notes.txt lines.

    Parameters:
    - wfa_cfg: config.
    - assignment: author language assignment table.
    - fit_lines: per-fit log lines.
    - headline_path: default 04 input CSV path.

    Returns:
    - List of note lines.
    """
    n_cross = int(assignment["cross_language"].sum()) if not assignment.empty else 0
    n_auth = len(assignment)
    by_lang = (
        assignment["assigned_primary_lexicon"].value_counts().to_dict()
        if not assignment.empty
        else {}
    )
    lines = [
        "Wordfish author-level v2 run notes",
        "==================================",
        wfa_cfg.get("note", ""),
        "",
        "Estimator: alternating conditional MLE (fit_wordfish_v2); token cap per author-bin doc.",
        "Pre-registered gate: |Spearman(theta, sem_axis)| >= threshold on IT (see validation_gate CSV).",
        "If gate_pass=0: do NOT use author theta as polarization outcome — failed triangulation only.",
        "",
        "Scope: secondary/robustness (like subreddit Wordfish 03).",
        "Most assumption-laden Wordfish variant: no within-language Italian control.",
        "Identification is IT vs DE/EN across separate per-language fits.",
        "Raw theta and dispersion levels are NOT comparable across it/en/de.",
        "Cross-language outcomes: extremity_z and change_z only.",
        "Report sign/direction only for cross-language contrasts — never magnitude.",
        "change_z = ban-window move in units of each language's pre-ban extremity churn.",
        "SUTVA: language-of-participation proxies treatment; English-only residents misclassified.",
        "German axis-words check is decisive for this variant.",
        "",
        "Design choices:",
        "- Dual fits: full RCS and balanced (pre+ban authors), separate output suffixes.",
        "- Bins: ban-anchored via bin_start_for_week (not Mar-1 dayidx).",
        "- Comments filtered to assigned primary_lexicon per author.",
        "",
        f"Authors assigned: n={n_auth} cross_language={n_cross} by_lang={by_lang}",
        f"placebo_launch_date={wfa_cfg.get('placebo_launch_date')} (for prompt 04 placebo DiD).",
        f"rolling_bins_w={wfa_cfg.get('rolling_bins_w')}",
        "",
        "Prompt 04 contract:",
        f"- Input headline panel: {headline_path}",
        "- Outcomes: extremity_z, change_z (primary); extremity, change (secondary).",
        "- Cluster by author; IT = (primary_lexicon == 'it').",
        "- 04 runs TWFE, event study, pre-trend joint-F, placebo — NOT in 03b.",
        "",
        "Fits:",
    ]
    lines.extend(fit_lines)
    return lines


def main() -> None:
    """Function summary: run author Wordfish pipeline and write tables."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    wfa_cfg = load_wordfish_authors_v2_config(config)
    if not wfa_cfg.get("enabled", True):
        print("[prepare_wordfish_authors_v2] disabled in config", flush=True)
        return

    subdir = str(wfa_cfg.get("output_tables_subdir", "wordfish_authors_v2"))
    out_dir = tables_subdir(config, subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start, end_excl, _launch, _lift = event_dates_from_config(config)
    anchor_date = str(wfa_cfg["ban_anchor_date"])
    shard_root = Path(config["paths"]["interim_dir"]) / "cleaned_monthly_chunks"
    stop_dir = PROJECT_ROOT / str(wfa_cfg.get("stopwords_dir", "config/lexicons"))
    priority = list(wfa_cfg.get("primary_lang_priority", ["it", "de", "en"]))
    drop_xlang = args.drop_cross_language or bool(wfa_cfg.get("drop_cross_language", False))
    suffix_xlang = "_noxlang" if drop_xlang else ""

    subs = [args.subreddit] if args.subreddit else resolve_primary_subreddits(config)

    specs = list(wfa_cfg.get("time_bins", []))
    if args.spec != "all":
        specs = [s for s in specs if str(s.get("name")) == args.spec]
    panel_modes = list(wfa_cfg.get("panel_modes", ["full", "balanced"]))
    if args.panel_mode != "all":
        panel_modes = [args.panel_mode]

    langs = list(wfa_cfg.get("languages", list(FIT_LANGUAGES)))
    if args.language != "all":
        langs = [args.language]

    assignment_path = out_dir / f"wordfish_authors_assignment{suffix_xlang}.csv"
    reuse = bool(args.reuse_assignment and assignment_path.is_file())

    pre_langs: Optional[Dict[str, Set[str]]] = None
    pre_assigned: Optional[Dict[str, str]] = None
    assignment: pd.DataFrame

    if reuse:
        print(
            f"[prepare_wordfish_authors_v2] reusing assignment from {assignment_path}",
            flush=True,
        )
        pre_langs, pre_assigned, assignment = load_assignment_from_csv(assignment_path)
    else:
        print("[prepare_wordfish_authors_v2] pass1: author language assignment", flush=True)

    wfa_scan_cfg = dict(wfa_cfg)
    if drop_xlang:
        wfa_scan_cfg["drop_cross_language"] = True

    scan = scan_shards_for_wordfish(
        shard_root,
        subs,
        args.max_shards,
        start,
        end_excl,
        specs,
        wfa_scan_cfg,
        priority,
        author_assigned=pre_assigned,
        author_langs=pre_langs,
        skip_pass1_langs=reuse,
    )
    author_langs = scan.author_langs
    if not reuse:
        assignment = build_author_assignment_table(author_langs, priority)
        assignment.to_csv(assignment_path, index=False)
    author_assigned = {
        row["author"]: row["assigned_primary_lexicon"] for _, row in assignment.iterrows()
    }
    comments_ideol = scan.comments_ideol

    stopwords_by_lang = {
        lang: load_stopwords(stop_dir / f"stopwords_{lang}.txt") for lang in langs
    }

    fit_lines: List[str] = []
    all_validation: List[pd.DataFrame] = []
    all_gate: List[pd.DataFrame] = []
    all_by_author: List[pd.DataFrame] = []
    en_mode = str(wfa_cfg.get("en_fit_mode", "split_us_uk"))
    pos_by_spec_mode: Dict[Tuple[str, str], pd.DataFrame] = {}

    headline_spec = str(wfa_cfg.get("headline_spec", "week7"))
    headline_mode = str(wfa_cfg.get("headline_mode", "balanced"))

    for spec in specs:
        spec_name = str(spec["name"])
        time_bin = str(spec["time_bin"])
        parts_by_tag: DefaultDict[str, Dict[str, List[pd.DataFrame]]] = defaultdict(
            lambda: {"pos": [], "ext": [], "disp": [], "cov": [], "val": []}
        )

        lang_targets: List[Tuple[str, Optional[str]]] = []
        for lang in langs:
            if lang == "en" and en_mode == "split_us_uk":
                lang_targets.append(("en_us", "en"))
                lang_targets.append(("en_uk", "en"))
            else:
                lang_targets.append((lang, lang))

        docs_all, cov = documents_from_buffers(
            scan.bodies_by_spec[spec_name],
            scan.meta_by_spec[spec_name],
            spec,
            wfa_cfg,
            stopwords_by_lang,
        )

        for fit_lang, token_lang in lang_targets:
            anchor_key = "en" if fit_lang.startswith("en_") else fit_lang
            anchor_sub = str((wfa_cfg.get("anchor_subreddit") or {}).get(anchor_key, ""))

            lang_docs = [d for d in docs_all if d.primary_lexicon == token_lang]
            if fit_lang == "en_us":
                lang_docs = filter_docs_by_subreddit_set(lang_docs, US_POLITICAL_SUBREDDITS)
            elif fit_lang == "en_uk":
                lang_docs = filter_docs_by_subreddit_set(lang_docs, UK_POLITICAL_SUBREDDITS)
            balanced_authors = filter_balanced_authors(lang_docs, anchor_date)

            cov_lang = cov[cov["primary_lexicon"] == token_lang].copy() if not cov.empty else cov
            if not cov_lang.empty:
                cov_lang["n_balanced_authors"] = len(balanced_authors)

            for panel_mode in panel_modes:
                if panel_mode == "balanced":
                    docs = filter_docs_by_authors(lang_docs, balanced_authors)
                else:
                    docs = lang_docs

                tag = f"{panel_mode}_{spec_name}{suffix_xlang}"
                n_authors = len({d.author for d in docs})
                min_auth = int(wfa_cfg.get("min_authors_per_language", 50))

                if n_authors < min_auth:
                    fit_lines.append(
                        f"SKIP {fit_lang}/{tag}: n_authors={n_authors} < min={min_auth}"
                    )
                    continue

                if anchor_sub and anchor_sub not in {d.subreddit for d in docs}:
                    fit_lines.append(
                        f"WARN {fit_lang}/{tag}: anchor {anchor_sub!r} absent — sign anchor may be no-op"
                    )

                result, _meta = run_single_fit(docs, wfa_cfg, anchor_sub, time_bin)
                if result is None:
                    fit_lines.append(f"SKIP {fit_lang}/{tag}: fit failed")
                    continue

                fit_lines.append(
                    f"FIT {fit_lang}/{tag}: n_docs={len(docs)} n_authors={n_authors} "
                    f"obj={result.objective_final:.4f} conv={result.converged} "
                    f"flip={result.sign_flipped} one_sided_beta={result.one_sided_beta}"
                )

                pos, ext, disp = author_positions_and_panels(
                    docs, result, wfa_cfg, panel_mode, spec_name
                )
                if not pos.empty:
                    pos["primary_lexicon"] = token_lang
                if not ext.empty:
                    ext["primary_lexicon"] = token_lang
                if not disp.empty:
                    disp["primary_lexicon"] = token_lang

                n_top = int(wfa_cfg.get("top_axis_words", 25))
                axis_df = pd.DataFrame(
                    [
                        {
                            "word": w,
                            "beta": b,
                            "sign": s,
                            "rank": i,
                            "primary_lexicon": fit_lang,
                            "spec": spec_name,
                            "panel_mode": panel_mode,
                        }
                        for i, (w, b, s) in enumerate(
                            top_axis_words(result.beta, n_top), start=1
                        )
                    ]
                )
                axis_df.to_csv(
                    out_dir / f"wordfish_authors_axis_words_{token_lang}_{tag}.csv",
                    index=False,
                )

                bucket = parts_by_tag[tag]
                if not pos.empty:
                    bucket["pos"].append(pos)
                if not ext.empty:
                    bucket["ext"].append(ext)
                if not disp.empty:
                    bucket["disp"].append(disp)
                if not cov_lang.empty:
                    bucket["cov"].append(
                        cov_lang.assign(panel_mode=panel_mode, spec=spec_name)
                    )

                val = author_validation_correlations(pos, comments_ideol)
                if not val.empty:
                    val["primary_lexicon"] = fit_lang
                    bucket["val"].append(val)
                    all_validation.append(val)
                    gate = author_validation_gate(val, wfa_cfg)
                    if not gate.empty:
                        all_gate.append(gate)
                    by_auth = author_validation_by_author(pos, comments_ideol)
                    if not by_auth.empty:
                        by_auth["primary_lexicon"] = fit_lang
                        all_by_author.append(by_auth)

        for panel_mode in panel_modes:
            tag = f"{panel_mode}_{spec_name}{suffix_xlang}"
            bucket = parts_by_tag[tag]
            if bucket["pos"]:
                pos_all = pd.concat(bucket["pos"], ignore_index=True)
                pos_by_spec_mode[(spec_name, panel_mode)] = pos_all
                pos_all.to_csv(
                    out_dir / f"wordfish_authors_positions_{tag}.csv", index=False
                )
            if bucket["ext"]:
                ext_all = pd.concat(bucket["ext"], ignore_index=True)
                ext_all.to_csv(
                    out_dir / f"wordfish_authors_extremity_panel_{tag}.csv", index=False
                )
                if spec_name == headline_spec and panel_mode == headline_mode:
                    ext_all.to_csv(
                        out_dir / "wordfish_authors_extremity_panel.csv", index=False
                    )
                    for lang_h in langs:
                        src = out_dir / f"wordfish_authors_axis_words_{lang_h}_{tag}.csv"
                        if src.is_file():
                            shutil.copy(
                                src,
                                out_dir / f"wordfish_authors_axis_words_{lang_h}.csv",
                            )
            if bucket["disp"]:
                pd.concat(bucket["disp"], ignore_index=True).to_csv(
                    out_dir / f"wordfish_authors_dispersion_panel_{tag}.csv", index=False
                )
            if bucket["cov"]:
                pd.concat(bucket["cov"], ignore_index=True).to_csv(
                    out_dir / f"wordfish_authors_coverage_{tag}.csv", index=False
                )
            if bucket["val"]:
                pd.concat(bucket["val"], ignore_index=True).to_csv(
                    out_dir / f"wordfish_authors_validation_{tag}.csv", index=False
                )

    if all_validation:
        val_all = pd.concat(all_validation, ignore_index=True)
        val_all.to_csv(out_dir / "wordfish_authors_validation_summary.csv", index=False)
    if all_gate:
        gate_all = pd.concat(all_gate, ignore_index=True)
        gate_all.to_csv(out_dir / "wordfish_authors_validation_gate.csv", index=False)
        fit_lines.append("")
        fit_lines.append("Validation gate (author-level |rho| vs sem_axis):")
        for _, g in gate_all.iterrows():
            fit_lines.append(
                f"  {g['primary_lexicon']}/{g['spec']}/{g['panel_mode']}: "
                f"gate_pass={g['gate_pass']} abs_rho_sem={g['abs_spearman_theta_sem_axis']:.4f} "
                f"-> {g['recommendation']}"
            )
    if all_by_author:
        pd.concat(all_by_author, ignore_index=True).to_csv(
            out_dir / "wordfish_authors_validation_by_author.csv", index=False
        )

    w7 = pos_by_spec_mode.get(("week7", "balanced"), pd.DataFrame())
    w3 = pos_by_spec_mode.get(("week3", "balanced"), pd.DataFrame())
    if not w7.empty and not comments_ideol.empty:
        auth_theta = w7.groupby("author", as_index=False)["theta"].mean()
        scatter_df = auth_theta.merge(comments_ideol, on="author", how="inner")
        scatter_df.to_csv(out_dir / "wordfish_authors_theta_ideology.csv", index=False)
    stab = stability_across_specs(w7, w3)
    if not stab.empty:
        stab.to_csv(out_dir / "wordfish_authors_stability.csv", index=False)

    headline_path = str(out_dir / "wordfish_authors_extremity_panel.csv")
    notes = build_run_notes(wfa_cfg, assignment, fit_lines, headline_path)
    (out_dir / "wordfish_authors_run_notes.txt").write_text(
        "\n".join(notes) + "\n", encoding="utf-8"
    )
    print(f"[prepare_wordfish_authors_v2] wrote tables to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
