"""
Script summary:
Builds short pooled time summaries of `detector_primary_ai_prob` from a Colab-export
zip tree of Parquet ML shards (e.g. `production_run/<subreddit>/<YYYY-MM>.parquet`).
Reads only `date_utc` and the primary detector column to stay light on memory.

Functionality:
- Walks every `*.parquet` under a configurable prefix inside the zip archive.
- Parses `date_utc` to UTC calendar dates, drops rows with invalid dates or non-finite scores.
- Writes pooled daily and monthly CSV tables: count, mean, median, plus counts/shares of
  comments with `detector_primary_ai_prob` above configurable thresholds (defaults: 0.5,
  0.75, 0.9) to track high-score / extreme tail mass over time.
- Optionally draws vertical markers at `launch_day_utc` (from `--config` or `--launch-day-utc`) and at a
  second release marker (default GPT-4 public date) for the same visual language as other project plots.
- Adds **volume-weighted** trailing rolling means for pooled mean and for each threshold share (noise
  reduction). Writes `launch_window_summary.csv` plus `ml_zip_time_trends_notes.txt` documenting that
  **this detector alone often will not show a sharp launch-day jump**; pooled narrative launch effects
  are usually clearer on `ai_likeness_index` / lexical rates from `prepare_event_time_metrics.py`.

How to apply/run (from repository root):
- Default paths match the Drive-style export name used in this project:
  `.venv/bin/python scripts/diagnostics/describe_ml_zip_time_trends.py`
- Custom zip / outputs:
  `.venv/bin/python scripts/diagnostics/describe_ml_zip_time_trends.py \\
      --zip-path data/interim/my_export.zip --internal-prefix production_run/ \\
      --tables-dir results/tables/ml_zip_time_trends \\
      --figures-dir results/figures/ml_zip_time_trends`
- Launch markers default from `config/political_forums_setup.yaml` (`launch_day_utc`) plus GPT-4 date; disable second line with `--no-gpt4-marker`.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq


def _resolve_project_root() -> Path:
    """Load scripts/_project_root.py and return the repository root Path."""
    _scripts_dir = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_socialai_scripts_project_root_mod",
        _scripts_dir / "_project_root.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/_project_root.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.project_root()


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_config, utc_ts


def _launch_ts_pd_from_utc_string(iso_z: str) -> pd.Timestamp:
    """Parse an ISO Zulu timestamp into a UTC-normalized pandas Timestamp (midnight anchor day)."""
    ts = utc_ts(str(iso_z))
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return pd.Timestamp(dt.date(), tz="UTC")


def _resolve_launch_timestamp(config_path: Path | None, launch_cli: str | None) -> pd.Timestamp:
    """
    Return launch day as UTC midnight Timestamp from YAML config or from a CLI ISO string.

    Parameters:
        config_path: Optional project YAML containing `event_window.launch_day_utc`.
        launch_cli: ISO Zulu string used when `config_path` is None.

    Returns:
        pandas Timestamp at UTC midnight for the launch calendar day.
    """
    if config_path is not None:
        cfg = load_config(config_path)
        return _launch_ts_pd_from_utc_string(str(cfg["event_window"]["launch_day_utc"]))
    if not launch_cli or not str(launch_cli).strip():
        raise ValueError("Provide --config or --launch-day-utc")
    return _launch_ts_pd_from_utc_string(str(launch_cli).strip())


def _iter_zip_parquet_names(zf: zipfile.ZipFile, internal_prefix: str) -> Iterable[str]:
    """Yield member paths ending in .parquet under the given zip-internal directory prefix."""
    prefix = internal_prefix.strip()
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"
    for name in sorted(zf.namelist()):
        if not name.endswith(".parquet"):
            continue
        if prefix and not name.startswith(prefix):
            continue
        yield name


def _load_score_frame_from_zip(zf: zipfile.ZipFile, member_names: list[str]) -> pd.DataFrame:
    """
    Read `date_utc` and `detector_primary_ai_prob` from every listed Parquet member and concatenate.

    Parameters:
        zf: Open ZipFile handle positioned at the ML export archive.
        member_names: Sorted list of zip member paths pointing at Parquet files.

    Returns:
        DataFrame with columns `date_utc` (datetime64 UTC, date-normalized) and `detector_primary_ai_prob`.
    """
    pieces: list[pd.DataFrame] = []
    for m in member_names:
        raw = zf.read(m)
        table = pq.read_table(io.BytesIO(raw), columns=["date_utc", "detector_primary_ai_prob"])
        pieces.append(table.to_pandas())
    if not pieces:
        return pd.DataFrame(columns=["date_utc", "detector_primary_ai_prob"])
    out = pd.concat(pieces, ignore_index=True)
    out["date_utc"] = pd.to_datetime(out["date_utc"], utc=True, errors="coerce").dt.normalize()
    out = out.dropna(subset=["date_utc"])
    out = out[pd.to_numeric(out["detector_primary_ai_prob"], errors="coerce").notna()]
    out["detector_primary_ai_prob"] = out["detector_primary_ai_prob"].astype("float64")
    return out


def _threshold_suffix(threshold: float) -> str:
    """Return a stable column-name fragment for a threshold (e.g. 0.5 -> '0_5')."""
    s = f"{float(threshold):g}".replace(".", "_")
    return s


def _build_agg_dict(thresholds: Sequence[float]) -> dict[str, tuple[str, object]]:
    """
    Build a named aggregation dict for `detector_primary_ai_prob`: size, mean, median, and counts above each threshold.

    Parameters:
        thresholds: Strictly numeric cutoffs in (0, 1); each adds a count column `n_primary_ai_prob_gt_<suffix>`.

    Returns:
        Mapping suitable for `DataFrame.groupby(...).agg(**kwargs)`.
    """
    aggs: dict[str, tuple[str, object]] = {
        "n_comments": ("detector_primary_ai_prob", "size"),
        "mean_primary_ai_prob": ("detector_primary_ai_prob", "mean"),
        "median_primary_ai_prob": ("detector_primary_ai_prob", "median"),
    }
    for thr in thresholds:
        suf = _threshold_suffix(thr)

        def _gt_count(s: pd.Series, t: float = thr) -> int:
            return int((s > t).sum())

        aggs[f"n_primary_ai_prob_gt_{suf}"] = ("detector_primary_ai_prob", _gt_count)
    return aggs


def _append_share_columns(table: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame:
    """
    For each threshold, add `share_primary_ai_prob_gt_<suffix>` = count / n_comments.

    Parameters:
        table: Grouped table including `n_comments` and each `n_primary_ai_prob_gt_*` count column.
        thresholds: Same thresholds used to build count column names.

    Returns:
        Same table with additional share_* float columns (NaN when n_comments == 0).
    """
    out = table.copy()
    n = out["n_comments"].astype("float64")
    for thr in thresholds:
        suf = _threshold_suffix(thr)
        ccol = f"n_primary_ai_prob_gt_{suf}"
        scol = f"share_primary_ai_prob_gt_{suf}"
        out[scol] = out[ccol] / n.replace(0.0, float("nan"))
    return out


def _daily_table(frame: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame:
    """
    Aggregate mean/median/count and high-score counts/shares of primary AI probability by calendar date.

    Parameters:
        frame: Input with `date_utc` and `detector_primary_ai_prob`.
        thresholds: Cutoffs for counting comments with score strictly greater than each value.

    Returns:
        DataFrame sorted by date with base stats plus n_/share_primary_ai_prob_gt_* columns.
    """
    aggs = _build_agg_dict(thresholds)
    g = frame.groupby("date_utc", sort=True).agg(**aggs)
    out = g.reset_index()
    return _append_share_columns(out, thresholds)


def _monthly_table(frame: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame:
    """
    Aggregate mean/median/count and high-score counts/shares by calendar month (month_start = first of month).

    Parameters:
        frame: Same schema as `_daily_table` input.
        thresholds: Same as `_daily_table`.

    Returns:
        DataFrame with month_key, month_start, n_comments, mean/median, and n_/share_primary_ai_prob_gt_* columns.
    """
    month_key = frame["date_utc"].dt.strftime("%Y-%m")
    tmp = frame.assign(month_key=month_key)
    aggs = _build_agg_dict(thresholds)
    g = tmp.groupby("month_key", sort=True).agg(**aggs)
    out = g.reset_index()
    out["month_start"] = pd.to_datetime(out["month_key"] + "-01", utc=True)
    return _append_share_columns(out, thresholds)


def _enrich_daily_rollings_event_time(
    daily: pd.DataFrame,
    thresholds: Sequence[float],
    rolling_days: int,
    launch_ts: pd.Timestamp,
) -> pd.DataFrame:
    """
    Append volume-weighted trailing rolling means and integer event-time days vs launch.

    Parameters:
        daily: Sorted or unsorted daily aggregate table.
        thresholds: Threshold list (drives rolling share column names).
        rolling_days: Trailing window length in days for weighted rolling columns.
        launch_ts: UTC-normalized launch instant (used as day anchor for event_time_t_days).

    Returns:
        Same rows sorted by `date_utc` with added rolling and `event_time_t_days` columns.
    """
    out = daily.sort_values("date_utc").reset_index(drop=True)
    k = max(1, int(rolling_days))
    n = out["n_comments"].astype("float64")
    num = out["mean_primary_ai_prob"].astype("float64") * n
    out[f"rolling{k}d_wmean_mean_primary_ai_prob"] = num.rolling(k, min_periods=1).sum() / n.rolling(k, min_periods=1).sum()
    for thr in thresholds:
        suf = _threshold_suffix(thr)
        ccol = f"n_primary_ai_prob_gt_{suf}"
        out[f"rolling{k}d_wmean_share_primary_ai_prob_gt_{suf}"] = (
            out[ccol].astype("float64").rolling(k, min_periods=1).sum() / n.rolling(k, min_periods=1).sum()
        )
    out["event_time_t_days"] = ((out["date_utc"] - launch_ts) / pd.Timedelta(days=1)).astype(int)
    return out


def _weighted_slice_stats(daily: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> dict[str, float | int]:
    """
    Pooled volume-weighted mean score and share>0.5 between two UTC dates inclusive.

    Parameters:
        daily: Daily table with n_comments, mean_primary_ai_prob, and n_primary_ai_prob_gt_0_5 (if available).
        t0: Window start (inclusive), UTC.
        t1: Window end (inclusive), UTC.

    Returns:
        Dict with n_comments, mean_primary_ai_prob, share_primary_ai_prob_gt_0_5 (NaN if undefined).
    """
    m = (daily["date_utc"] >= t0) & (daily["date_utc"] <= t1)
    sub = daily.loc[m]
    w = int(sub["n_comments"].sum())
    if w == 0:
        return {"n_comments": 0, "mean_primary_ai_prob": float("nan"), "share_primary_ai_prob_gt_0_5": float("nan")}
    wm = float((sub["mean_primary_ai_prob"] * sub["n_comments"]).sum() / w)
    suf = _threshold_suffix(0.5)
    ccol = f"n_primary_ai_prob_gt_{suf}"
    if ccol not in sub.columns:
        return {"n_comments": w, "mean_primary_ai_prob": wm, "share_primary_ai_prob_gt_0_5": float("nan")}
    sh = float(sub[ccol].sum() / w)
    return {"n_comments": w, "mean_primary_ai_prob": wm, "share_primary_ai_prob_gt_0_5": sh}


def _launch_window_summary_table(daily: pd.DataFrame, launch_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Build a small CSV-ready table comparing pre-launch, launch-day, and post-launch calendar windows.

    Parameters:
        daily: Enriched daily aggregates (must include threshold 0.5 counts for the share column).
        launch_ts: ChatGPT (or configured) launch anchor in UTC.

    Returns:
        One row per named window with weighted summary statistics.
    """
    L = launch_ts.normalize()
    windows = [
        ("early_nov_pre", pd.Timestamp("2022-11-01", tz="UTC"), pd.Timestamp("2022-11-15", tz="UTC")),
        ("late_nov_pre", pd.Timestamp("2022-11-16", tz="UTC"), L - pd.Timedelta(days=1)),
        ("launch_day", L, L),
        ("first_week_post", L + pd.Timedelta(days=1), L + pd.Timedelta(days=7)),
        ("days_8_14_post", L + pd.Timedelta(days=8), L + pd.Timedelta(days=14)),
        ("days_15_28_post", L + pd.Timedelta(days=15), L + pd.Timedelta(days=28)),
    ]
    rows = []
    for label, a, b in windows:
        stats = _weighted_slice_stats(daily, a, b)
        rows.append({"window": label, "start_utc": a.date().isoformat(), "end_utc": b.date().isoformat(), **stats})
    return pd.DataFrame(rows)


def _write_interpretation_note(path: Path, launch_iso: str, summary: pd.DataFrame) -> None:
    """
    Write a short plain-text note warning that pooled primary-detector scores can miss launch-day drama.

    Parameters:
        path: Output text path under results tables.
        launch_iso: Launch day ISO label for human readers.
        summary: DataFrame from `_launch_window_summary_table` for embedding example contrasts.

    Returns:
        None; writes UTF-8 text to disk.
    """
    lines = [
        "ML zip diagnostic — interpretation",
        "",
        f"Launch anchor referenced here: {launch_iso} (UTC calendar day).",
        "",
        "The Hugging Face `detector_primary_ai_prob` score measures one stylometric 'AI-like' signal.",
        "Pooled across many subreddits and days, it often will NOT show a sharp step on the public",
        "ChatGPT release day: daily volume composition, forum mix, and detector calibration dominate.",
        "",
        "For launch-aligned figures in this repository, prefer the main event-time pipeline:",
        "`prepare_event_time_metrics.py` → `plot_event_time_metrics.py`, especially `ai_likeness_index`",
        "(composite) and lexical / AI-word rates, which are designed for event-study plots.",
        "",
        "Adjacent-window summary (volume-weighted within each window):",
        summary.to_string(index=False),
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_pooled_daily(
    daily: pd.DataFrame,
    output_png: Path,
    title: str,
    thresholds: Sequence[float],
    launch_ts: pd.Timestamp | None,
    second_marker_ts: pd.Timestamp | None,
    rolling_days: int,
) -> None:
    """
    Save a two-panel figure: (top) mean/median with launch markers; (bottom) daily + volume-weighted rolling shares.

    Parameters:
        daily: Daily aggregates including rolling columns from `_enrich_daily_rollings_event_time`.
        output_png: Destination PNG path (parent directories created if missing).
        title: Overall figure suptitle string.
        thresholds: Threshold list for translucent daily share traces.
        launch_ts: Optional UTC vertical marker (ChatGPT launch).
        second_marker_ts: Optional second vertical marker (e.g. GPT-4 public date).
        rolling_days: Window length k used in rolling column names for the dashed overlay.

    Returns:
        None; writes the figure to disk.
    """
    if daily.empty:
        return
    output_png.parent.mkdir(parents=True, exist_ok=True)
    k = max(1, int(rolling_days))
    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(9, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.0], "hspace": 0.12},
        layout="constrained",
    )
    ax0.plot(daily["date_utc"], daily["mean_primary_ai_prob"], label="Mean", color="#1f77b4", linewidth=1.4)
    ax0.plot(
        daily["date_utc"],
        daily["median_primary_ai_prob"],
        label="Median",
        color="#ff7f0e",
        linewidth=1.4,
        alpha=0.9,
    )
    if launch_ts is not None:
        ax0.axvline(launch_ts, color="#c0392b", linestyle=":", linewidth=1.6, label="Launch")
    if second_marker_ts is not None:
        ax0.axvline(second_marker_ts, color="#c0392b", linestyle=(0, (3, 6)), linewidth=1.2, label="GPT-4")
    ax0.set_ylabel("detector_primary_ai_prob")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper left", fontsize=8)
    ax0.set_title("Central tendency (pooled)")

    colors = ["#2ca02c", "#d62728", "#9467bd"]
    for i, thr in enumerate(thresholds):
        suf = _threshold_suffix(thr)
        scol = f"share_primary_ai_prob_gt_{suf}"
        if scol not in daily.columns:
            continue
        ax1.plot(
            daily["date_utc"],
            daily[scol],
            label=f"Share > {thr:g} (daily)",
            color=colors[i % len(colors)],
            linewidth=1.1,
            alpha=0.55,
        )
    thr_roll = next((t for t in thresholds if abs(t - 0.5) < 1e-12), thresholds[-1])
    suf_r = _threshold_suffix(thr_roll)
    rcol = f"rolling{k}d_wmean_share_primary_ai_prob_gt_{suf_r}"
    if rcol in daily.columns:
        ax1.plot(
            daily["date_utc"],
            daily[rcol],
            color="black",
            linewidth=2.0,
            linestyle="--",
            label=f">{thr_roll:g} share, {k}d weighted",
        )
    if launch_ts is not None:
        ax1.axvline(launch_ts, color="#c0392b", linestyle=":", linewidth=1.6)
    if second_marker_ts is not None:
        ax1.axvline(second_marker_ts, color="#c0392b", linestyle=(0, (3, 6)), linewidth=1.2)
    ax1.set_ylabel("Share of comments")
    ax1.set_xlabel("UTC date (from shard date_utc)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_ylim(bottom=0.0)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.suptitle(title)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    for lab in ax1.get_xticklabels():
        lab.set_rotation(35)
        lab.set_ha("right")
    fig.savefig(output_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Parse CLI args, aggregate ML scores over time from the zip, write tables and one figure."""
    parser = argparse.ArgumentParser(
        description="Pooled daily/monthly descriptives of detector_primary_ai_prob from a Colab ML zip export."
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=PROJECT_ROOT / "data/interim/production_run-20260511T145305Z-3-001.zip",
        help="Path to the zip archive containing Parquet ML shards.",
    )
    parser.add_argument(
        "--internal-prefix",
        type=str,
        default="production_run/",
        help="Zip-internal directory prefix before <subreddit>/<YYYY-MM>.parquet paths.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=PROJECT_ROOT / "results/tables/ml_zip_time_trends",
        help="Directory for output CSV summaries.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "results/figures/ml_zip_time_trends",
        help="Directory for the pooled daily PNG figure.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.5,0.75,0.9",
        help="Comma-separated cutoffs for counting shares with detector_primary_ai_prob strictly greater than each.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/political_forums_setup.yaml",
        help="YAML with event_window.launch_day_utc; used when the file exists (otherwise --launch-day-utc).",
    )
    parser.add_argument(
        "--launch-day-utc",
        type=str,
        default="2022-11-30T00:00:00Z",
        help="Fallback launch anchor if --config is missing or unreadable.",
    )
    parser.add_argument(
        "--rolling-days",
        type=int,
        default=7,
        help="Trailing calendar-day window for volume-weighted rolling columns and dashed overlay.",
    )
    parser.add_argument(
        "--gpt4-day-utc",
        type=str,
        default="2023-03-14T00:00:00Z",
        help="Second vertical marker on figures (public GPT-4 date used elsewhere in this repo).",
    )
    parser.add_argument(
        "--no-gpt4-marker",
        action="store_true",
        help="Omit the second vertical marker from figures.",
    )
    args = parser.parse_args()

    thresholds = tuple(
        sorted(
            {float(x.strip()) for x in args.thresholds.split(",") if x.strip()} | {0.5},
        )
    )
    if not thresholds:
        raise ValueError("Provide at least one numeric --thresholds value")

    zip_path = args.zip_path.expanduser().resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    tables_dir = args.tables_dir.expanduser().resolve()
    figures_dir = args.figures_dir.expanduser().resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = list(_iter_zip_parquet_names(zf, args.internal_prefix))
        if not names:
            raise RuntimeError(
                f"No Parquet files under prefix {args.internal_prefix!r} inside {zip_path}"
            )
        frame = _load_score_frame_from_zip(zf, names)

    daily = _daily_table(frame, thresholds)
    monthly = _monthly_table(frame, thresholds)

    cfg_path = args.config.expanduser().resolve()
    if cfg_path.is_file():
        launch_ts = _resolve_launch_timestamp(cfg_path, None)
    else:
        launch_ts = _resolve_launch_timestamp(None, args.launch_day_utc)

    second_marker_ts: pd.Timestamp | None = None
    if not args.no_gpt4_marker:
        second_marker_ts = _launch_ts_pd_from_utc_string(args.gpt4_day_utc)

    daily = _enrich_daily_rollings_event_time(daily, thresholds, args.rolling_days, launch_ts)
    summary = _launch_window_summary_table(daily, launch_ts)

    daily_path = tables_dir / "pooled_daily_primary_ai_prob.csv"
    monthly_path = tables_dir / "pooled_monthly_primary_ai_prob.csv"
    summary_path = tables_dir / "launch_window_summary.csv"
    notes_path = tables_dir / "ml_zip_time_trends_notes.txt"
    daily.to_csv(daily_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    summary.to_csv(summary_path, index=False)
    _write_interpretation_note(notes_path, str(launch_ts.date()), summary)

    title = f"Pooled primary AI detector (n={len(frame):,} comments, {len(names)} shards)"
    fig_path = figures_dir / "pooled_daily_primary_ai_prob_mean_median.png"
    _plot_pooled_daily(daily, fig_path, title, thresholds, launch_ts, second_marker_ts, args.rolling_days)

    print(f"[describe_ml_zip_time_trends] zip={zip_path}")
    print(f"[describe_ml_zip_time_trends] shards={len(names)} rows_used={len(frame):,}")
    print(f"[describe_ml_zip_time_trends] launch_marker={launch_ts.date()} (from config file)" if cfg_path.is_file() else f"[describe_ml_zip_time_trends] launch_marker={launch_ts.date()} (CLI fallback)")
    print(f"[describe_ml_zip_time_trends] wrote {daily_path}")
    print(f"[describe_ml_zip_time_trends] wrote {monthly_path}")
    print(f"[describe_ml_zip_time_trends] wrote {summary_path}")
    print(f"[describe_ml_zip_time_trends] wrote {notes_path}")
    print(f"[describe_ml_zip_time_trends] wrote {fig_path}")


if __name__ == "__main__":
    main()
