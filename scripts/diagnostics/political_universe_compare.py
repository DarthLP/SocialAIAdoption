"""
Script summary:
Compare political-universe definitions: agreement, coverage by topic_family, P/R vs hand labels.

Functionality:
- Loads enriched shards (Mar–Apr) and recomputes mode columns if missing.
- Writes tables and figures under results/.../political_coverage/.

How to apply/run:
  .venv/bin/python scripts/diagnostics/political_universe_compare.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVENT_MONTHS = ("2023-03", "2023-04")
DEFINITION_COLUMNS = [
    "in_political_universe_comment",
    "in_political_universe_tree",
    "in_political_universe_thread_sum",
    "in_political_universe_thread_rate",
    "in_political_universe_embedding",
]


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
    figures_subdir,
    load_config,
    utc_ts,
    load_political_universe_config,
    load_screening_config,
    load_screening_pooled,
    resolve_primary_subreddits,
    screening_by_subreddit,
    should_skip_screened_subreddit,
    subreddit_screening_action,
    tables_subdir,
)
from src.political_filter import apply_all_modes  # noqa: E402


def event_dates(config: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Function summary: parse event window dates from study config.

    Parameters:
    - config: loaded YAML.

    Returns:
    - Tuple (start, end_exclusive, launch, lift) as YYYY-MM-DD strings.
    """
    ew = config["event_window"]
    start = datetime.fromtimestamp(utc_ts(ew["start_utc"]), tz=timezone.utc).strftime("%Y-%m-%d")
    end_excl = datetime.fromtimestamp(utc_ts(ew["end_utc_exclusive"]), tz=timezone.utc).strftime(
        "%Y-%m-%d"
    )
    launch = datetime.fromtimestamp(utc_ts(ew["launch_day_utc"]), tz=timezone.utc).strftime(
        "%Y-%m-%d"
    )
    refs = config.get("plot_reference_dates_utc") or []
    lift = "2023-04-29"
    if isinstance(refs, list) and len(refs) >= 2:
        lift = datetime.fromisoformat(str(refs[1]).replace("Z", "+00:00")).strftime("%Y-%m-%d")
    return start, end_excl, launch, lift


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compare political universe definitions.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--include-excluded", action="store_true")
    return parser.parse_args()


def agreement_table(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: pairwise agreement rates among universe definitions."""
    present = [c for c in DEFINITION_COLUMNS if c in df.columns]
    rows: List[Dict[str, Any]] = []
    for i, a in enumerate(present):
        for b in present[i:]:
            aa = df[a].astype(bool)
            bb = df[b].astype(bool)
            agree = (aa == bb).mean()
            both = (aa & bb).mean()
            a_only = (aa & ~bb).mean()
            b_only = (~aa & bb).mean()
            rows.append(
                {
                    "definition_a": a.replace("in_political_universe_", ""),
                    "definition_b": b.replace("in_political_universe_", ""),
                    "agreement_share": round(float(agree), 4),
                    "both_share": round(float(both), 4),
                    "a_only_share": round(float(a_only), 4),
                    "b_only_share": round(float(b_only), 4),
                }
            )
    return pd.DataFrame(rows)


def coverage_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: political comment and word share by topic_family per definition."""
    if df.empty or "topic_family" not in df.columns:
        return pd.DataFrame()
    present = [c for c in DEFINITION_COLUMNS if c in df.columns]
    rows: List[Dict[str, Any]] = []
    nw = df["n_words"].fillna(0).astype(float)
    for family, grp in df.groupby("topic_family", sort=True):
        gnw = grp["n_words"].fillna(0).astype(float)
        for col in present:
            mask = grp[col].astype(bool)
            rows.append(
                {
                    "topic_family": family,
                    "definition": col.replace("in_political_universe_", ""),
                    "political_comment_share": round(float(mask.mean()), 4),
                    "political_word_share": round(
                        float(gnw[mask].sum() / gnw.sum()) if gnw.sum() > 0 else 0.0, 4
                    ),
                    "n_comments": len(grp),
                }
            )
        rows.append(
            {
                "topic_family": family,
                "definition": "_all",
                "political_comment_share": round(float(len(grp) / len(df)), 4),
                "political_word_share": round(float(gnw.sum() / nw.sum()) if nw.sum() > 0 else 0.0, 4),
                "n_comments": len(grp),
            }
        )
    return pd.DataFrame(rows)


def validation_pr(
    labels_path: Path,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Function summary: precision/recall per definition vs hand labels when available."""
    if not labels_path.is_file():
        return pd.DataFrame()
    labels = pd.read_csv(labels_path, comment="#")
    if labels.empty or "id" not in labels.columns or "label_political" not in labels.columns:
        return pd.DataFrame()
    labels = labels.dropna(subset=["id", "label_political"])
    if labels.empty:
        return pd.DataFrame()
    if "id" not in df.columns:
        return pd.DataFrame()
    merged = labels.merge(
        df[["id"] + [c for c in DEFINITION_COLUMNS if c in df.columns]],
        on="id",
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for col in DEFINITION_COLUMNS:
        if col not in merged.columns:
            continue
        pred = merged[col].astype(bool)
        true = merged["label_political"].astype(bool)
        tp = int((pred & true).sum())
        fp = int((pred & ~true).sum())
        fn = int((~pred & true).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (2 * prec * rec / (prec + rec)) if prec == prec and rec == rec and (prec + rec) > 0 else float("nan")
        rows.append(
            {
                "definition": col.replace("in_political_universe_", ""),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "n_labels": len(merged),
            }
        )
    return pd.DataFrame(rows)


def write_notes(
    path: Path,
    df: pd.DataFrame,
    sub_stats: List[Dict[str, float]],
) -> None:
    """Function summary: write political_coverage_notes.txt summary."""
    lines = [
        "Political universe coverage notes",
        "",
    ]
    if sub_stats:
        st = pd.DataFrame(sub_stats)
        lines.append(
            f"Mean orphan_share across subreddits: {st['orphan_share'].mean():.4f}"
            if "orphan_share" in st.columns
            else "Orphan stats: n/a (run apply_political_universe.py)"
        )
        if "orphan_fallback_share" in st.columns:
            lines.append(f"Mean orphan_fallback_share: {st['orphan_fallback_share'].mean():.4f}")
        if "missing_parent_share" in st.columns:
            lines.append(f"Mean missing_parent_share: {st['missing_parent_share'].mean():.4f}")
    lines.extend(
        [
            "",
            "Title-only recall ceiling: threads whose politics appears only in the",
            "submission title (not in RC comments) cannot be recovered without RS dumps.",
            "Qualitative estimate: sample hand labels on short-reply threads after populating",
            "data/raw/political_universe_labels.csv.",
            "",
        ]
    )
    if not df.empty and "comment_in_political_universe" in df.columns:
        lines.append(
            f"Active universe (comment_in_political_universe) share: "
            f"{df['comment_in_political_universe'].astype(bool).mean():.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_agreement_heatmap(agree_df: pd.DataFrame, out_path: Path) -> None:
    """Function summary: heatmap of pairwise agreement shares."""
    if agree_df.empty:
        return
    defs = sorted(set(agree_df["definition_a"]).union(set(agree_df["definition_b"])))
    n = len(defs)
    mat = np.eye(n)
    idx = {d: i for i, d in enumerate(defs)}
    for row in agree_df.itertuples(index=False):
        i, j = idx[row.definition_a], idx[row.definition_b]
        mat[i, j] = row.agreement_share
        mat[j, i] = row.agreement_share
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(n), defs, rotation=45, ha="right")
    ax.set_yticks(range(n), defs)
    fig.colorbar(im, ax=ax, label="Agreement share")
    ax.set_title("Political definition agreement")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pr_bars(pr_df: pd.DataFrame, out_path: Path) -> None:
    """Function summary: grouped P/R/F1 bar chart per definition."""
    if pr_df.empty:
        return
    defs = pr_df["definition"].tolist()
    x = np.arange(len(defs))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, pr_df["precision"], width=w, label="precision")
    ax.bar(x, pr_df["recall"], width=w, label="recall")
    ax.bar(x + w, pr_df["f1"], width=w, label="f1")
    ax.set_xticks(x, defs, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("P/R/F1 by political universe definition")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_bars(cov_df: pd.DataFrame, out_path: Path) -> None:
    """Function summary: bar chart of political comment share by topic_family."""
    if cov_df.empty:
        return
    sub = cov_df[cov_df["definition"] == "tree"].copy()
    if sub.empty:
        sub = cov_df[cov_df["definition"] != "_all"].drop_duplicates("topic_family")
    if sub.empty:
        return
    sub = sub.sort_values("political_comment_share", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(sub))))
    ax.barh(sub["topic_family"], sub["political_comment_share"])
    ax.set_xlabel("Political comment share (tree definition)")
    ax.set_title("Coverage by topic_family")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Function summary: run comparison diagnostics and write outputs."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    pu_cfg = load_political_universe_config(config)
    screening = load_screening_config(config)
    interim_dir = Path(config["paths"]["interim_dir"])
    tables_dir = tables_subdir(config, "political_coverage")
    figures_dir = figures_subdir(config, "political_coverage")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    start, end_excl, _, _ = event_dates(config)
    subs = resolve_primary_subreddits(config)
    screening_by_sub = screening_by_subreddit(load_screening_pooled(Path(config["paths"]["tables_dir"])))

    frames: List[pd.DataFrame] = []
    sub_stats: List[Dict[str, float]] = []
    for subreddit in subs:
        action = subreddit_screening_action(screening_by_sub, subreddit)
        if should_skip_screened_subreddit(action, include_excluded=args.include_excluded):
            continue
        shard_dir = interim_dir / "cleaned_monthly_chunks" / subreddit
        if not shard_dir.is_dir():
            continue
        parts: List[pd.DataFrame] = []
        for month in EVENT_MONTHS:
            path = shard_dir / f"{month}.parquet"
            if not path.is_file():
                continue
            try:
                raw = pd.read_parquet(path)
            except Exception:
                continue
            if raw.empty:
                continue
            if "date_utc" in raw.columns:
                raw = raw[(raw["date_utc"] >= start) & (raw["date_utc"] < end_excl)]
            parts.append(raw)
        if not parts:
            continue
        combined_sub = pd.concat(parts, ignore_index=True)
        if "in_political_universe_tree" not in combined_sub.columns:
            combined_sub, st = apply_all_modes(combined_sub, pu_cfg, screening)
            st["subreddit"] = subreddit
            sub_stats.append(st)
        frames.append(combined_sub)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    agree = agreement_table(df)
    agree.to_csv(tables_dir / "political_definition_agreement.csv", index=False)
    coverage_by_family(df).to_csv(tables_dir / "political_coverage_by_family.csv", index=False)

    labels_path = PROJECT_ROOT / "data/raw/political_universe_labels.csv"
    pr = validation_pr(labels_path, df)
    if not pr.empty:
        pr.to_csv(tables_dir / "political_pr_by_definition.csv", index=False)

    if sub_stats:
        pd.DataFrame(sub_stats).to_csv(tables_dir / "political_universe_subreddit_stats.csv", index=False)
    write_notes(tables_dir / "political_coverage_notes.txt", df, sub_stats)

    plot_agreement_heatmap(agree, figures_dir / "definition_agreement_heatmap.png")
    if not pr.empty:
        plot_pr_bars(pr, figures_dir / "pr_by_definition_bars.png")
    plot_coverage_bars(pd.read_csv(tables_dir / "political_coverage_by_family.csv"), figures_dir / "coverage_by_family_bars.png")

    print(f"[political_universe_compare] wrote tables to {tables_dir}", flush=True)
    print(f"[political_universe_compare] wrote figures to {figures_dir}", flush=True)


if __name__ == "__main__":
    main()
