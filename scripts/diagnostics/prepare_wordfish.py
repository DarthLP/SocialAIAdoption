"""
Script summary:
Wordfish robustness pipeline — event-binned subreddit documents, dual day/week fits.

Functionality:
- Loads political-universe comments; builds it/en × day/week Wordfish fits (de excluded).
- Writes positions, extremity, dispersion panels, axis words, coverage, validation, stability.
- Emits wordfish_run_notes.txt with tiering and cross-fit caveats.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_wordfish.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/prepare_wordfish.py --language it --time-bin day
"""

from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

READ_COLUMNS = [
    "subreddit",
    "date_utc",
    "body",
    "topic_family",
    "primary_lexicon",
    "comment_in_political_universe",
    "net_ideology",
    "sem_axis_ideology",
    "n_words",
]

ITALY_TOPIC_FAMILIES = frozenset({"it_political", "it_others"})


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
    load_wordfish_config,
    resolve_primary_subreddits,
    subreddit_family_map,
    tables_subdir,
)
from src.wordfish import (  # noqa: E402
    DocumentRecord,
    add_date_utc_column,
    add_placebo_flags,
    apply_sign_anchor,
    bin_start_for_day,
    bin_start_for_week,
    build_placebo_window_summary,
    build_vocabulary_and_matrix,
    compute_center_lang_pre,
    compute_change_outcomes,
    family_dispersion,
    fit_wordfish,
    load_stopwords,
    parse_anchor_date,
    tokenize_document,
    top_axis_words,
    zscore_preban,
)


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare Wordfish robustness tables.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--subreddit", type=str, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--language", type=str, default="all", choices=("it", "en", "all"))
    parser.add_argument("--time-bin", type=str, default="all", choices=("day", "week", "all"))
    return parser.parse_args()


def load_comment_frame(
    shard_root: Path,
    subreddits: Sequence[str],
    max_shards: Optional[int],
) -> pd.DataFrame:
    """Function summary: load political-universe comment columns from enriched shards.

    Parameters:
    - shard_root: cleaned_monthly_chunks root.
    - subreddits: forum names.
    - max_shards: optional cap per subreddit.

    Returns:
    - Combined dataframe.
    """
    parts: List[pd.DataFrame] = []
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
            cols = [c for c in READ_COLUMNS if c in df.columns]
            if not cols:
                continue
            chunk = df[cols].copy()
            chunk["subreddit"] = sub
            parts.append(chunk)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def assign_bin_start(
    date_utc: str,
    time_bin: str,
    anchor_date: Any,
    weekly_days: int,
) -> str:
    """Function summary: map calendar date to bin_start for day or event-week bin.

    Parameters:
    - date_utc: YYYY-MM-DD.
    - time_bin: day or week.
    - anchor_date: t* date object.
    - weekly_days: block width.

    Returns:
    - bin_start string.
    """
    if time_bin == "day":
        return bin_start_for_day(date_utc)
    if time_bin == "week":
        return bin_start_for_week(date_utc, anchor_date, weekly_days)
    raise ValueError(f"unknown time_bin: {time_bin}")


def build_documents(
    df: pd.DataFrame,
    language: str,
    time_bin: str,
    wf_cfg: Dict[str, Any],
    stopwords: Set[str],
    family_map: Dict[str, str],
) -> Tuple[List[DocumentRecord], pd.DataFrame]:
    """Function summary: aggregate comments into subreddit×bin documents.

    Parameters:
    - df: comment-level frame (political universe).
    - language: primary_lexicon filter.
    - time_bin: day or week.
    - wf_cfg: wordfish config.
    - stopwords: stopword set.
    - family_map: subreddit -> topic_family.

    Returns:
    - Tuple (document records, coverage audit frame).
    """
    anchor = parse_anchor_date(str(wf_cfg["ban_anchor_date"]))
    weekly_days = int(wf_cfg.get("weekly_bin_days", 7))
    min_doc_tokens = int(wf_cfg["min_doc_tokens"])
    min_token_len = int(wf_cfg["min_token_len"])

    work = df[df["primary_lexicon"] == language].copy()
    work = work[work["comment_in_political_universe"].astype(bool)]
    work["date_utc"] = work["date_utc"].astype(str).str[:10]
    work["topic_family"] = work["subreddit"].map(family_map).fillna("")
    work["bin_start"] = work["date_utc"].apply(
        lambda d: assign_bin_start(d, time_bin, anchor, weekly_days)
    )

    grouped = work.groupby(["subreddit", "bin_start", "topic_family"], dropna=False)
    docs: List[DocumentRecord] = []
    coverage_rows: List[Dict[str, Any]] = []

    for (sub, bin_start, family), grp in grouped:
        bodies = grp["body"].fillna("").astype(str).tolist()
        text = " ".join(bodies)
        tokens = tokenize_document(text, stopwords, min_token_len)
        n_days = int(grp["date_utc"].nunique())
        n_tokens = len(tokens)
        doc_id = f"{sub}|{bin_start}"
        rec = DocumentRecord(
            doc_id=doc_id,
            subreddit=str(sub),
            topic_family=str(family),
            primary_lexicon=language,
            bin_start=str(bin_start),
            time_bin=time_bin,
            n_days_in_bin=n_days,
            n_tokens=n_tokens,
            tokens=tokens,
        )
        if n_tokens >= min_doc_tokens:
            docs.append(rec)
        coverage_rows.append(
            {
                "topic_family": family,
                "primary_lexicon": language,
                "time_bin": time_bin,
                "bin_start": bin_start,
                "subreddit": sub,
                "n_comments": len(grp),
                "n_days_in_bin": n_days,
                "n_tokens": n_tokens,
                "doc_kept": n_tokens >= min_doc_tokens,
            }
        )

    coverage = pd.DataFrame(coverage_rows)
    return docs, coverage


def run_single_fit(
    docs: List[DocumentRecord],
    wf_cfg: Dict[str, Any],
    anchor_sub: str,
    time_bin: str,
    prune_override: Optional[Dict[str, int]] = None,
) -> Tuple[Optional[Any], List[Dict[str, str]]]:
    """Function summary: fit Wordfish on document list.

    Parameters:
    - docs: surviving documents.
    - wf_cfg: config dict.
    - anchor_sub: anchor subreddit for sign.
    - time_bin: day or week (week uses scaled learning rate).
    - prune_override: optional {min_doc_freq, top_freq_drop_n}.

    Returns:
    - Tuple (WordfishFitResult or None, doc_meta list).
    """
    if len(docs) < 2:
        return None, []
    min_df = int(wf_cfg["min_doc_freq"])
    top_drop = int(wf_cfg["top_freq_drop_n"])
    if prune_override:
        min_df = int(prune_override.get("min_doc_freq", min_df))
        top_drop = int(prune_override.get("top_freq_drop_n", top_drop))

    doc_tokens = [d.tokens for d in docs]
    mat, vocab = build_vocabulary_and_matrix(
        doc_tokens,
        min_doc_freq=min_df,
        top_freq_drop_n=top_drop,
        max_vocab_terms=int(wf_cfg.get("max_vocab_terms", 5000)),
    )
    if mat.shape[1] == 0:
        return None, []

    doc_ids = [d.doc_id for d in docs]
    doc_meta = [{"subreddit": d.subreddit, "doc_id": d.doc_id} for d in docs]
    lr = float(wf_cfg["learning_rate"])
    if time_bin == "week":
        lr *= float(wf_cfg.get("week_learning_rate_scale", 0.1))
    result = fit_wordfish(
        mat,
        vocab,
        doc_ids,
        train_iters=int(wf_cfg["train_iters"]),
        learning_rate=lr,
        convergence_cfg=wf_cfg.get("convergence", {}),
    )
    result = apply_sign_anchor(result, doc_meta, anchor_sub)
    return result, doc_meta


def positions_and_panels(
    docs: List[DocumentRecord],
    result: Any,
    wf_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Function summary: build positions, extremity, and dispersion panels.

    Parameters:
    - docs: document metadata aligned with fit rows.
    - result: WordfishFitResult.
    - wf_cfg: config.

    Returns:
    - Tuple (positions, extremity_panel, dispersion_panel).
    """
    anchor_date = str(wf_cfg["ban_anchor_date"])
    doc_by_id = {d.doc_id: d for d in docs}
    pos_rows: List[Dict[str, Any]] = []
    ext_rows: List[Dict[str, Any]] = []

    for i, doc_id in enumerate(result.doc_ids):
        d = doc_by_id[doc_id]
        theta = float(result.theta[i])
        pos_rows.append(
            {
                "subreddit": d.subreddit,
                "topic_family": d.topic_family,
                "primary_lexicon": d.primary_lexicon,
                "bin_start": d.bin_start,
                "time_bin": d.time_bin,
                "n_days_in_bin": d.n_days_in_bin,
                "theta": theta,
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

    for i, row in enumerate(pos_df.to_dict("records")):
        theta = float(row["theta"])
        extremity = ext_vals[i]
        if not np.isnan(pre_mu) and not np.isnan(extremity) and not np.isnan(pre_sd):
            extremity_z = (extremity - pre_mu) / pre_sd
        else:
            extremity_z = float("nan")

        ext_rows.append(
            {
                "subreddit": row["subreddit"],
                "topic_family": row["topic_family"],
                "primary_lexicon": row["primary_lexicon"],
                "bin_start": row["bin_start"],
                "time_bin": row["time_bin"],
                "center_lang_pre": center,
                "theta": theta,
                "extremity": extremity,
                "extremity_z": extremity_z,
                "post": int(row["bin_start"] >= anchor_date),
                "IT": int(row["topic_family"] in ITALY_TOPIC_FAMILIES),
                "n_days_in_bin": int(row["n_days_in_bin"]),
            }
        )

    ext_df = pd.DataFrame(ext_rows)
    fam_counts = (
        ext_df.groupby(["topic_family", "bin_start", "time_bin"])["subreddit"]
        .nunique()
        .reset_index(name="n_subreddits_in_family_day")
    )
    ext_df = ext_df.merge(
        fam_counts,
        on=["topic_family", "bin_start", "time_bin"],
        how="left",
    )

    window_days = int((wf_cfg.get("change_window_days") or [7])[0])
    ext_df = compute_change_outcomes(ext_df, anchor_date, window_days)
    placebo_date = str(wf_cfg.get("placebo_launch_date", "2023-03-16"))
    ext_df = add_placebo_flags(ext_df, placebo_date, anchor_date)
    ext_df = add_date_utc_column(ext_df)

    pos_df = add_date_utc_column(pos_df)

    disp_rows: List[Dict[str, Any]] = []
    for (family, bin_start, tbin), grp in ext_df.groupby(
        ["topic_family", "bin_start", "time_bin"], dropna=False
    ):
        thetas = grp["theta"].astype(float).tolist()
        disp = family_dispersion(thetas)
        disp_rows.append(
            {
                "topic_family": family,
                "primary_lexicon": grp["primary_lexicon"].iloc[0],
                "bin_start": bin_start,
                "time_bin": tbin,
                "dispersion_var": disp["dispersion_var"],
                "dispersion_iqr": disp["dispersion_iqr"],
                "dispersion_range": disp["dispersion_range"],
                "n_subreddits": len(grp),
                "n_days_in_bin": int(grp["n_days_in_bin"].max()),
                "post": int(bin_start >= anchor_date),
                "IT": int(family in ITALY_TOPIC_FAMILIES),
            }
        )
    disp_df = pd.DataFrame(disp_rows)
    disp_df = add_date_utc_column(disp_df)
    return pos_df, ext_df, disp_df


def export_panel_columns(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Function summary: select prompt-facing column order for CSV export.

    Parameters:
    - df: positions, extremity, or dispersion panel.
    - kind: positions | extremity | dispersion.

    Returns:
    - Reordered dataframe.
    """
    if kind == "positions":
        cols = [
            "subreddit",
            "topic_family",
            "primary_lexicon",
            "date_utc",
            "bin_start",
            "time_bin",
            "n_days_in_bin",
            "theta",
            "n_tokens",
        ]
    elif kind == "extremity":
        cols = [
            "subreddit",
            "topic_family",
            "primary_lexicon",
            "date_utc",
            "bin_start",
            "time_bin",
            "center_lang_pre",
            "theta",
            "extremity",
            "extremity_z",
            "change",
            "change_z",
            "post",
            "IT",
            "pre_placebo",
            "post_placebo",
            "n_subreddits_in_family_day",
            "n_days_in_bin",
        ]
    else:
        cols = [
            "topic_family",
            "primary_lexicon",
            "date_utc",
            "bin_start",
            "time_bin",
            "dispersion_var",
            "dispersion_iqr",
            "dispersion_range",
            "n_subreddits",
            "n_days_in_bin",
            "post",
            "IT",
        ]
    present = [c for c in cols if c in df.columns]
    extra = [c for c in df.columns if c not in present]
    return df[present + extra]


def axis_words_table(result: Any, language: str, time_bin: str, n_top: int) -> pd.DataFrame:
    """Function summary: top +/- beta words for one fit.

    Parameters:
    - result: WordfishFitResult.
    - language: lexicon code.
    - time_bin: day or week.
    - n_top: words per tail.

    Returns:
    - DataFrame for CSV export.
    """
    rows = []
    for rank, (word, beta, sign) in enumerate(top_axis_words(result.beta, n_top), start=1):
        rows.append(
            {
                "word": word,
                "beta": beta,
                "sign": sign,
                "rank": rank,
                "primary_lexicon": language,
                "time_bin": time_bin,
            }
        )
    return pd.DataFrame(rows)


def validation_correlations(
    positions: pd.DataFrame,
    comments: pd.DataFrame,
) -> pd.DataFrame:
    """Function summary: Spearman rho subreddit-mean theta vs lexicon/semantic axes.

    Parameters:
    - positions: positions panel.
    - comments: raw comments with ideology columns.

    Returns:
    - One-row correlation summary per (language, time_bin).
    """
    rows = []
    if positions.empty or comments.empty:
        return pd.DataFrame()

    sub_theta = positions.groupby("subreddit")["theta"].mean().reset_index()
    sub_lex = (
        comments.groupby("subreddit")
        .agg(
            net_ideology_mean=("net_ideology", "mean"),
            sem_axis_ideology_mean=("sem_axis_ideology", "mean"),
        )
        .reset_index()
    )
    merged = sub_theta.merge(sub_lex, on="subreddit", how="inner")
    for (lang, tbin), grp in positions.groupby(["primary_lexicon", "time_bin"]):
        subs = grp["subreddit"].unique()
        m = merged[merged["subreddit"].isin(subs)]
        if len(m) < 5:
            continue
        r_net = float(m["theta"].corr(m["net_ideology_mean"], method="spearman"))
        r_sem = float(m["theta"].corr(m["sem_axis_ideology_mean"], method="spearman"))
        rows.append(
            {
                "primary_lexicon": lang,
                "time_bin": tbin,
                "n_subreddits": len(m),
                "spearman_theta_net_ideology": r_net,
                "spearman_theta_sem_axis": r_sem,
            }
        )
    return pd.DataFrame(rows)


def stability_rank_rho(
    docs: List[DocumentRecord],
    wf_cfg: Dict[str, Any],
    anchor_sub: str,
    time_bin: str = "day",
) -> pd.DataFrame:
    """Function summary: rank correlation of subreddit-mean theta across prune profiles.

    Parameters:
    - docs: documents for baseline fit language/time_bin.
    - wf_cfg: config with sensitivity_profiles.
    - anchor_sub: anchor forum.

    Returns:
    - DataFrame with rank rho per profile pair.
    """
    profiles = wf_cfg.get("sensitivity_profiles") or []
    if len(profiles) < 2:
        return pd.DataFrame()

    means: Dict[int, pd.Series] = {}
    for idx, prof in enumerate(profiles[:2]):
        res, _ = run_single_fit(docs, wf_cfg, anchor_sub, time_bin, prune_override=prof)
        if res is None:
            continue
        doc_by_id = {d.doc_id: d for d in docs}
        vals = []
        subs = []
        for i, doc_id in enumerate(res.doc_ids):
            subs.append(doc_by_id[doc_id].subreddit)
            vals.append(float(res.theta[i]))
        frame = pd.DataFrame({"subreddit": subs, "theta": vals})
        means[idx] = frame.groupby("subreddit")["theta"].mean()

    if 0 not in means or 1 not in means:
        return pd.DataFrame()
    joined = pd.DataFrame({"a": means[0], "b": means[1]}).dropna()
    if len(joined) < 3:
        return pd.DataFrame()
    rho = float(joined["a"].corr(joined["b"], method="spearman"))
    return pd.DataFrame(
        [
            {
                "profile_a": str(profiles[0]),
                "profile_b": str(profiles[1]),
                "spearman_rank_rho_subreddit_mean_theta": rho,
                "n_subreddits": len(joined),
            }
        ]
    )


def write_run_notes(
    path: Path,
    notes: List[str],
) -> None:
    """Function summary: write wordfish_run_notes.txt.

    Parameters:
    - path: output path.
    - notes: lines to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(notes) + "\n", encoding="utf-8")


def main() -> None:
    """Function summary: run Wordfish pipeline and write all tables."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    wf_cfg = load_wordfish_config(config)
    out_dir = tables_subdir(config, "wordfish")
    out_dir.mkdir(parents=True, exist_ok=True)

    start, end_excl, _launch, _lift = event_dates_from_config(config)
    anchor_date = str(wf_cfg["ban_anchor_date"])
    shard_root = Path(config["paths"]["interim_dir"]) / "cleaned_monthly_chunks"
    stop_dir = PROJECT_ROOT / str(wf_cfg.get("stopwords_dir", "config/lexicons"))
    family_map = subreddit_family_map(config)

    subs = [args.subreddit] if args.subreddit else resolve_primary_subreddits(config)
    raw = load_comment_frame(shard_root, subs, args.max_shards)
    if raw.empty:
        print("[prepare_wordfish] no data", flush=True)
        return
    if "comment_in_political_universe" not in raw.columns:
        raise SystemExit("comment_in_political_universe missing — run apply_political_universe.py")

    raw = raw[raw["date_utc"].astype(str).str[:10] >= start]
    raw = raw[raw["date_utc"].astype(str).str[:10] < end_excl]
    if "topic_family" not in raw.columns:
        raw["topic_family"] = raw["subreddit"].map(family_map)

    langs = wf_cfg.get("languages", ["it", "en"])
    if args.language != "all":
        langs = [args.language]
    time_bins = wf_cfg.get("time_bins", ["day", "week"])
    if args.time_bin != "all":
        time_bins = [args.time_bin]

    all_pos: List[pd.DataFrame] = []
    all_ext: List[pd.DataFrame] = []
    all_disp: List[pd.DataFrame] = []
    all_cov: List[pd.DataFrame] = []
    all_val: List[pd.DataFrame] = []
    all_stab: List[pd.DataFrame] = []
    run_notes: List[str] = [
        "Wordfish robustness run notes",
        "===========================",
        'German (de) excluded from Wordfish fit (single forum); de remains lexical/semantic-axis control; stopwords_de.txt generated for 03b-authors.',
        "Tier A (it_political vs it_others) is the political-specificity test (DDD), not the ban effect — both arms are Italian/banned. Do not write up Tier A as estimating ban impact.",
        "Tier B (IT vs EN controls) is the ban-effect robustness test; use extremity_z with cross-language caveat.",
        "Raw theta and dispersion levels are not comparable across it vs en fits — only within-fit trajectories and Tier B z-scores.",
        "Family dispersion NaN when n_subreddits < 2; all family-bins still exported.",
        "Tier B: compare daily-it vs daily-en and weekly-it vs weekly-en only (matched time_bin).",
        "Tier B: report sign/direction only for cross-country contrasts — never cross-fit magnitude.",
        "change/change_z use rolling prior extremity (W=change_window_days[0]); week bins use prior-row window, not calendar days.",
        f"placebo_launch_date={wf_cfg.get('placebo_launch_date', '2023-03-16')}: use post_placebo for in-month placebo split.",
        "",
    ]

    for lang in langs:
        if lang == "de":
            continue
        stopwords = load_stopwords(stop_dir / f"stopwords_{lang}.txt")
        anchor_sub = str((wf_cfg.get("anchor_subreddit") or {}).get(lang, ""))

        for time_bin in time_bins:
            docs, cov = build_documents(raw, lang, time_bin, wf_cfg, stopwords, family_map)
            all_cov.append(cov)

            n_subs = len({d.subreddit for d in docs})
            if n_subs < int(wf_cfg.get("min_subreddits_per_language", 2)):
                run_notes.append(
                    f"SKIP fit {lang}/{time_bin}: only {n_subs} subreddits with surviving docs"
                )
                continue

            if anchor_sub and anchor_sub not in {d.subreddit for d in docs}:
                run_notes.append(
                    f"WARN {lang}/{time_bin}: anchor subreddit {anchor_sub!r} not in corpus — sign flip may be no-op"
                )

            result, _meta = run_single_fit(docs, wf_cfg, anchor_sub, time_bin)
            if result is None:
                run_notes.append(f"SKIP fit {lang}/{time_bin}: Wordfish fit failed (empty vocab or too few docs)")
                continue

            lr_note = ""
            if time_bin == "week":
                lr_note = f" lr_scaled={float(wf_cfg.get('week_learning_rate_scale', 0.1))}"
            run_notes.append(
                f"FIT {lang}/{time_bin}: n_docs={len(docs)} n_subreddits={n_subs} "
                f"objective={result.objective_final:.4f} converged={result.converged} "
                f"sign_flipped={result.sign_flipped}{lr_note}"
            )

            pos, ext, disp = positions_and_panels(docs, result, wf_cfg)
            center = float(ext["center_lang_pre"].iloc[0]) if not ext.empty else float("nan")
            run_notes.append(f"  center_lang_pre={center:.6f}")

            all_pos.append(pos)
            all_ext.append(ext)
            all_disp.append(disp)

            n_top = int(wf_cfg.get("top_axis_words", 25))
            axis_df = axis_words_table(result, lang, time_bin, n_top)
            axis_path = out_dir / f"wordfish_axis_words_{lang}_{time_bin}.csv"
            axis_df.to_csv(axis_path, index=False)
            if time_bin == str(wf_cfg.get("primary_time_bin", "day")):
                axis_df.to_csv(out_dir / f"wordfish_axis_words_{lang}.csv", index=False)

            if lang == "it" and time_bin == "day":
                stab = stability_rank_rho(docs, wf_cfg, anchor_sub)
                if not stab.empty:
                    stab["primary_lexicon"] = lang
                    stab["time_bin"] = time_bin
                    all_stab.append(stab)

    if all_pos:
        export_panel_columns(pd.concat(all_pos, ignore_index=True), "positions").to_csv(
            out_dir / "wordfish_positions.csv", index=False
        )
    if all_ext:
        ext_all = export_panel_columns(pd.concat(all_ext, ignore_index=True), "extremity")
        ext_all.to_csv(out_dir / "wordfish_extremity_panel.csv", index=False)
        placebo_sum = build_placebo_window_summary(ext_all)
        if not placebo_sum.empty:
            placebo_sum.to_csv(out_dir / "wordfish_placebo_window_summary.csv", index=False)
    if all_disp:
        export_panel_columns(pd.concat(all_disp, ignore_index=True), "dispersion").to_csv(
            out_dir / "wordfish_dispersion_panel.csv", index=False
        )
    if all_cov:
        pd.concat(all_cov, ignore_index=True).to_csv(out_dir / "wordfish_doc_coverage.csv", index=False)
    if all_pos:
        pos_all = pd.concat(all_pos, ignore_index=True)
        val = validation_correlations(pos_all, raw)
        if not val.empty:
            val.to_csv(out_dir / "wordfish_validation_correlations.csv", index=False)
        sub_theta = pos_all.groupby(["subreddit", "primary_lexicon", "time_bin"])["theta"].mean().reset_index()
        sub_ideol = (
            raw.groupby("subreddit")
            .agg(
                net_ideology_mean=("net_ideology", "mean"),
                sem_axis_ideology_mean=("sem_axis_ideology", "mean"),
            )
            .reset_index()
        )
        sub_theta.merge(sub_ideol, on="subreddit", how="left").to_csv(
            out_dir / "wordfish_subreddit_theta_ideology.csv",
            index=False,
        )
    if all_stab:
        pd.concat(all_stab, ignore_index=True).to_csv(out_dir / "wordfish_stability.csv", index=False)

    write_run_notes(out_dir / "wordfish_run_notes.txt", run_notes)
    print(f"[prepare_wordfish] wrote tables to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
