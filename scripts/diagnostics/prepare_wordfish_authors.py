"""
Script summary:
Author-level Wordfish pipeline — political author×bin documents, per-language fits.

Functionality:
- Assigns each author to one language (it > de > en); builds ban-anchored week/window bins.
- Runs separate full and balanced-panel Wordfish fits per language×time spec.
- Writes positions, extremity (with change/change_z), dispersion, axis words, coverage, validation.
- Emits wordfish_authors_run_notes.txt with cross-language ID caveats and prompt-04 contract.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_wordfish_authors.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/prepare_wordfish_authors.py --spec week7 --panel-mode balanced
  .venv/bin/python scripts/diagnostics/prepare_wordfish_authors.py --drop-cross-language
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

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
    load_wordfish_authors_config,
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
    compute_center_lang_pre,
    compute_change_outcomes,
    family_dispersion,
    fit_wordfish,
    load_stopwords,
    normalize_lexicon_code,
    parse_anchor_date,
    tokenize_document,
    top_axis_words,
    zscore_preban,
)


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare author-level Wordfish tables.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--subreddit", type=str, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--language", type=str, default="all", choices=("it", "en", "de", "all"))
    parser.add_argument("--spec", type=str, default="all")
    parser.add_argument("--panel-mode", type=str, default="all", choices=("full", "balanced", "all"))
    parser.add_argument("--drop-cross-language", action="store_true")
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


def iter_comment_chunks(
    shard_root: Path,
    subreddits: Sequence[str],
    max_shards: Optional[int],
    start: str,
    end_excl: str,
):
    """Function summary: yield filtered political comment chunks from shards.

    Parameters:
    - shard_root: cleaned_monthly_chunks root.
    - subreddits: forum list.
    - max_shards: optional per-forum cap.
    - start: window start YYYY-MM-DD.
    - end_excl: exclusive end date.

    Yields:
    - DataFrame chunks with required columns.
    """
    for sub in subreddits:
        shard_dir = shard_root / sub
        if not shard_dir.is_dir():
            continue
        shards = sorted(shard_dir.glob("*.parquet"))
        if max_shards is not None:
            shards = shards[: max_shards]
        for shard in shards:
            try:
                df = pd.read_parquet(shard)
            except Exception:
                continue
            if "comment_in_political_universe" not in df.columns:
                continue
            cols = [c for c in READ_COLUMNS if c in df.columns]
            if not cols:
                continue
            chunk = df[cols].copy()
            chunk["subreddit"] = sub
            chunk = chunk[chunk["comment_in_political_universe"].astype(bool)]
            if "is_deleted_author" in chunk.columns:
                chunk = chunk[~chunk["is_deleted_author"].fillna(False).astype(bool)]
            chunk["date_utc"] = chunk["date_utc"].astype(str).str[:10]
            chunk = chunk[(chunk["date_utc"] >= start) & (chunk["date_utc"] < end_excl)]
            if chunk.empty:
                continue
            yield chunk


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
    for chunk in iter_comment_chunks(shard_root, subreddits, max_shards, start, end_excl):
        for author, lex in zip(
            chunk["author"].astype(str),
            chunk["primary_lexicon"].astype(str).map(normalize_lexicon_code),
        ):
            if not author or author == "nan":
                continue
            author_langs[author].add(lex)
            author_counts[author] += 1
    return dict(author_langs), dict(author_counts)


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
    - stopwords: language stopwords.
    - filter_to_assigned: keep only assigned-lang comments.
    - drop_cross_language: exclude multi-lexicon authors.
    - author_langs: for cross-lang filter.
    - window_start: whole-window bin label.
    - target_lang: if set, only build documents for authors assigned to this language.

    Returns:
    - Tuple (document records, coverage audit).
    """
    anchor = parse_anchor_date(str(wfa_cfg["ban_anchor_date"]))
    time_bin = str(spec["time_bin"])
    weekly_days = int(spec.get("weekly_bin_days", 7))
    min_doc_tokens = int(spec["min_doc_tokens"])
    min_token_len = int(wfa_cfg["min_token_len"])
    spec_name = str(spec["name"])

    bodies: DefaultDict[Tuple[str, str, str], List[str]] = defaultdict(list)
    meta: DefaultDict[Tuple[str, str, str], Dict[str, Any]] = {}

    for chunk in iter_comment_chunks(shard_root, subreddits, max_shards, start, end_excl):
        chunk["lex"] = chunk["primary_lexicon"].astype(str).map(normalize_lexicon_code)
        for row in chunk.itertuples(index=False):
            author = str(getattr(row, "author", ""))
            if not author or author == "nan":
                continue
            if drop_cross_language and len(author_langs.get(author, set())) >= 2:
                continue
            assigned = author_assigned.get(author)
            if not assigned:
                continue
            if target_lang is not None and assigned != target_lang:
                continue
            lex = str(getattr(row, "lex", ""))
            if filter_to_assigned and lex != assigned:
                continue
            date_utc = str(getattr(row, "date_utc", ""))[:10]
            bin_start = assign_bin_start_author(
                date_utc, time_bin, anchor, weekly_days, window_start
            )
            key = (author, bin_start, assigned)
            bodies[key].append(str(getattr(row, "body", "") or ""))
            sub = str(getattr(row, "subreddit", ""))
            if key not in meta:
                meta[key] = {
                    "dates": set(),
                    "subreddits": Counter(),
                    "n_words_proxy": 0,
                    "n_comments": 0,
                }
            meta[key]["dates"].add(date_utc)
            meta[key]["subreddits"][sub] += 1
            meta[key]["n_words_proxy"] += int(getattr(row, "n_words", 0) or 0)
            meta[key]["n_comments"] += 1

    docs: List[DocumentRecord] = []
    coverage_rows: List[Dict[str, Any]] = []

    for (author, bin_start, lang), text_parts in bodies.items():
        text = " ".join(text_parts)
        tokens = tokenize_document(text, stopwords, min_token_len)
        n_tokens = len(tokens)
        m = meta[(author, bin_start, lang)]
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
                "doc_kept": kept,
            }
        )

    return docs, pd.DataFrame(coverage_rows)


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
    lr = float(wfa_cfg["learning_rate"])
    if time_bin == "week":
        lr *= float(wfa_cfg.get("week_learning_rate_scale", 0.1))
    result = fit_wordfish(
        mat,
        vocab,
        doc_ids,
        train_iters=int(wfa_cfg["train_iters"]),
        learning_rate=lr,
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
        "Wordfish author-level run notes",
        "==============================",
        wfa_cfg.get("note", ""),
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
    wfa_cfg = load_wordfish_authors_config(config)
    if not wfa_cfg.get("enabled", True):
        print("[prepare_wordfish_authors] disabled in config", flush=True)
        return

    out_dir = tables_subdir(config, "wordfish_authors")
    out_dir.mkdir(parents=True, exist_ok=True)

    start, end_excl, _launch, _lift = event_dates_from_config(config)
    anchor_date = str(wfa_cfg["ban_anchor_date"])
    shard_root = Path(config["paths"]["interim_dir"]) / "cleaned_monthly_chunks"
    stop_dir = PROJECT_ROOT / str(wfa_cfg.get("stopwords_dir", "config/lexicons"))
    priority = list(wfa_cfg.get("primary_lang_priority", ["it", "de", "en"]))
    filter_assigned = bool(wfa_cfg.get("filter_comments_to_assigned_lang", True))
    drop_xlang = args.drop_cross_language or bool(wfa_cfg.get("drop_cross_language", False))
    suffix_xlang = "_noxlang" if drop_xlang else ""

    subs = [args.subreddit] if args.subreddit else resolve_primary_subreddits(config)

    print("[prepare_wordfish_authors] pass1: author language assignment", flush=True)
    author_langs, _ = pass1_author_languages(shard_root, subs, args.max_shards, start, end_excl)
    assignment = build_author_assignment_table(author_langs, priority)
    author_assigned = {
        row["author"]: row["assigned_primary_lexicon"]
        for _, row in assignment.iterrows()
    }
    assignment.to_csv(out_dir / f"wordfish_authors_assignment{suffix_xlang}.csv", index=False)

    specs = list(wfa_cfg.get("time_bins", []))
    if args.spec != "all":
        specs = [s for s in specs if str(s.get("name")) == args.spec]
    panel_modes = list(wfa_cfg.get("panel_modes", ["full", "balanced"]))
    if args.panel_mode != "all":
        panel_modes = [args.panel_mode]

    all_langs = list(wfa_cfg.get("languages", list(FIT_LANGUAGES)))
    langs = list(all_langs)
    if args.language != "all":
        langs = [args.language]
    is_partial = (args.language != "all") and (len(langs) < len(all_langs))
    tags_written: List[str] = []

    fit_lines: List[str] = []
    all_validation: List[pd.DataFrame] = []
    comments_ideol_parts: List[pd.DataFrame] = []
    pos_by_spec_mode: Dict[Tuple[str, str], pd.DataFrame] = {}

    for chunk in iter_comment_chunks(shard_root, subs, args.max_shards, start, end_excl):
        if "net_ideology" not in chunk.columns:
            continue
        part = (
            chunk.groupby("author")
            .agg(
                net_ideology_mean=("net_ideology", "mean"),
                sem_axis_ideology_mean=("sem_axis_ideology", "mean"),
            )
            .reset_index()
        )
        comments_ideol_parts.append(part)
    comments_ideol = (
        pd.concat(comments_ideol_parts, ignore_index=True)
        if comments_ideol_parts
        else pd.DataFrame()
    )
    if not comments_ideol.empty:
        comments_ideol = (
            comments_ideol.groupby("author")
            .agg(
                net_ideology_mean=("net_ideology_mean", "mean"),
                sem_axis_ideology_mean=("sem_axis_ideology_mean", "mean"),
            )
            .reset_index()
        )

    headline_spec = str(wfa_cfg.get("headline_spec", "week7"))
    headline_mode = str(wfa_cfg.get("headline_mode", "balanced"))

    for spec in specs:
        spec_name = str(spec["name"])
        time_bin = str(spec["time_bin"])

        for panel_mode in panel_modes:
            tag = f"{panel_mode}_{spec_name}{suffix_xlang}"
            pos_parts: List[pd.DataFrame] = []
            ext_parts: List[pd.DataFrame] = []
            disp_parts: List[pd.DataFrame] = []
            cov_parts: List[pd.DataFrame] = []
            val_parts: List[pd.DataFrame] = []

            for lang in langs:
                stopwords = load_stopwords(stop_dir / f"stopwords_{lang}.txt")
                anchor_sub = str((wfa_cfg.get("anchor_subreddit") or {}).get(lang, ""))

                docs_all, cov = aggregate_author_documents(
                    shard_root,
                    subs,
                    args.max_shards,
                    start,
                    end_excl,
                    author_assigned,
                    spec,
                    wfa_cfg,
                    stopwords,
                    filter_assigned,
                    drop_xlang,
                    author_langs,
                    window_start=start,
                    target_lang=lang,
                )
                lang_docs = docs_all
                balanced_authors = filter_balanced_authors(lang_docs, anchor_date)

                if panel_mode == "balanced":
                    docs = filter_docs_by_authors(lang_docs, balanced_authors)
                else:
                    docs = lang_docs

                n_authors = len({d.author for d in docs})
                min_auth = int(wfa_cfg.get("min_authors_per_language", 50))

                if n_authors < min_auth:
                    fit_lines.append(
                        f"SKIP {lang}/{tag}: n_authors={n_authors} < min={min_auth}"
                    )
                    continue

                if anchor_sub and anchor_sub not in {d.subreddit for d in docs}:
                    fit_lines.append(
                        f"WARN {lang}/{tag}: anchor {anchor_sub!r} absent — sign anchor may be no-op"
                    )

                result, _meta = run_single_fit(docs, wfa_cfg, anchor_sub, time_bin)
                if result is None:
                    fit_lines.append(f"SKIP {lang}/{tag}: fit failed")
                    continue

                fit_lines.append(
                    f"FIT {lang}/{tag}: n_docs={len(docs)} n_authors={n_authors} "
                    f"obj={result.objective_final:.4f} conv={result.converged} "
                    f"flip={result.sign_flipped}"
                )

                pos, ext, disp = author_positions_and_panels(
                    docs, result, wfa_cfg, panel_mode, spec_name
                )
                pos_parts.append(pos)
                ext_parts.append(ext)
                disp_parts.append(disp)

                n_top = int(wfa_cfg.get("top_axis_words", 25))
                axis_df = pd.DataFrame(
                    [
                        {
                            "word": w,
                            "beta": b,
                            "sign": s,
                            "rank": i,
                            "primary_lexicon": lang,
                            "spec": spec_name,
                            "panel_mode": panel_mode,
                        }
                        for i, (w, b, s) in enumerate(
                            top_axis_words(result.beta, n_top), start=1
                        )
                    ]
                )
                axis_df.to_csv(
                    out_dir / f"wordfish_authors_axis_words_{lang}_{tag}.csv", index=False
                )

                cov_lang = cov[cov["primary_lexicon"] == lang].copy() if not cov.empty else cov
                if not cov_lang.empty:
                    cov_lang = cov_lang.assign(
                        panel_mode=panel_mode,
                        spec=spec_name,
                        n_balanced_authors=len(balanced_authors),
                    )
                    cov_parts.append(cov_lang)

                val = author_validation_correlations(pos, comments_ideol)
                if not val.empty:
                    val_parts.append(val)
                    all_validation.append(val)

            if is_partial:
                if ext_parts:
                    ext_parts[0].to_csv(
                        out_dir
                        / f"wordfish_authors_extremity_panel_{tag}_{args.language}.csv",
                        index=False,
                    )
            else:
                if pos_parts:
                    pos_all = pd.concat(pos_parts, ignore_index=True)
                    pos_by_spec_mode[(spec_name, panel_mode)] = pos_all
                    pos_all.to_csv(
                        out_dir / f"wordfish_authors_positions_{tag}.csv", index=False
                    )
                if ext_parts:
                    ext_all = pd.concat(ext_parts, ignore_index=True)
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
                    tags_written.append(tag)
                if disp_parts:
                    pd.concat(disp_parts, ignore_index=True).to_csv(
                        out_dir / f"wordfish_authors_dispersion_panel_{tag}.csv", index=False
                    )
                if cov_parts:
                    pd.concat(cov_parts, ignore_index=True).to_csv(
                        out_dir / f"wordfish_authors_coverage_{tag}.csv", index=False
                    )
                if val_parts:
                    pd.concat(val_parts, ignore_index=True).to_csv(
                        out_dir / f"wordfish_authors_validation_{tag}.csv", index=False
                    )

    if all_validation:
        pd.concat(all_validation, ignore_index=True).to_csv(
            out_dir / "wordfish_authors_validation_summary.csv", index=False
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
    if is_partial:
        print(
            "[prepare_wordfish_authors] Partial-language run; pooled headline files unchanged.",
            flush=True,
        )
    expected_lex = set(all_langs)
    for tag in tags_written:
        panel_path = out_dir / f"wordfish_authors_extremity_panel_{tag}.csv"
        if not panel_path.is_file():
            continue
        present = set(pd.read_csv(panel_path)["primary_lexicon"].dropna().astype(str).unique())
        if present != expected_lex:
            print(
                f"[prepare_wordfish_authors] WARNING: {tag}: extremity panel has "
                f"{sorted(present)}, expected {sorted(expected_lex)}",
                flush=True,
            )
    print(f"[prepare_wordfish_authors] wrote tables to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
