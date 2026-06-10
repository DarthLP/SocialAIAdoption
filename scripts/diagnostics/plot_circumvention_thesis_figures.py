"""
Script summary:
Thesis-ready Google Trends circumvention figures (VPN + ChatGPT) for Italy's ChatGPT ban.

Functionality:
- Reads circumvention_daily_by_geo.csv; plots Italy vs DE/FR/US control band (min–max + mean).
- Level figures (raw Trends scale) plus pre-ban-indexed variants (Mar 1–30 mean = 100).
- 7-day trailing rolling means; ban window shaded with ban/lift reference lines.
- Raw-data sanity checks on known Italy peaks before smoothing; exits on column mismatch.
- Writes PNG figures and matching caption .txt footnotes under circumvention/thesis/.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_circumvention_descriptives.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/plot_circumvention_thesis_figures.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _setup_project_root() -> Path:
    """Function summary: resolve repo root via scripts/_bootstrap.py.

    Returns:
    - Absolute path to repository root.
    """
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

from scripts.diagnostics.descriptives_util import grouped_trailing_daily_rolling  # noqa: E402
from src.config_utils import figures_subdir, load_config, tables_subdir  # noqa: E402

STUDY_START = "2023-03-01"
STUDY_END = "2023-04-30"
PRE_BAN_END = "2023-03-30"
BAN_START = "2023-03-31"
BAN_END = "2023-04-28"
ITALY_GEO = "IT"
CONTROL_GEOS = ("DE", "FR", "US")
ROLLING_WINDOW = 7
FOOTNOTE = (
    "Google Trends indices are scaled 0–100 within each country; "
    "compare timing and shape, not levels across countries. "
    "Controls: DE, FR, US (GB/ES series unavailable in this extract)."
)
FOOTNOTE_INDEXED = (
    "7-day trailing means re-indexed to pre-ban mean (Mar 1–30, 2023) = 100 "
    "within each country; facilitates comparison of relative changes. "
    "Controls: DE, FR, US (GB/ES series unavailable in this extract)."
)
INDEXED_YLABEL = "Index (pre-ban mean = 100)"
INDEXED_REFERENCE_Y = 100.0

ITALY_COLOR = "#c1121f"
CONTROL_COLOR = "#457b9d"

FIGURE_SPECS = (
    (
        "vpn_interest",
        "Google Trends VPN interest (topic)",
        "trends_vpn_thesis",
    ),
    (
        "chatgpt_interest",
        "Google Trends ChatGPT interest (topic; attention/salience)",
        "trends_chatgpt_thesis",
    ),
)


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments.

    Returns:
    - Parsed namespace with config path.
    """
    parser = argparse.ArgumentParser(
        description="Plot thesis-ready Google Trends circumvention figures."
    )
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument("--rolling-window", type=int, default=ROLLING_WINDOW)
    return parser.parse_args()


def _load_study_panel(path: Path) -> pd.DataFrame:
    """Function summary: load daily circumvention table clipped to study window.

    Parameters:
    - path: path to circumvention_daily_by_geo.csv.

    Returns:
    - Frame with IT + DE/FR/US geos, dates in [STUDY_START, STUDY_END].
    """
    if not path.is_file():
        raise FileNotFoundError(f"Circumvention daily table not found: {path}")
    df = pd.read_csv(path)
    df["geo"] = df["geo"].astype(str).str.upper()
    keep_geos = {ITALY_GEO, *CONTROL_GEOS}
    df = df[df["geo"].isin(keep_geos)].copy()
    dates = df["date_utc"].astype(str)
    return df[(dates >= STUDY_START) & (dates <= STUDY_END)].copy()


def _it_value(raw: pd.DataFrame, date: str, metric_col: str) -> float:
    """Function summary: extract Italy raw metric on a single date.

    Parameters:
    - raw: study-window panel.
    - date: YYYY-MM-DD string.
    - metric_col: column name (vpn_interest or chatgpt_interest).

    Returns:
    - Scalar value or NaN if missing.
    """
    row = raw[(raw["geo"] == ITALY_GEO) & (raw["date_utc"].astype(str) == date)]
    if row.empty:
        return float("nan")
    return float(row[metric_col].iloc[0])


def _control_mean(raw: pd.DataFrame, date: str, metric_col: str) -> float:
    """Function summary: mean of control geos on a single date.

    Parameters:
    - raw: study-window panel.
    - date: YYYY-MM-DD string.
    - metric_col: column to average.

    Returns:
    - Mean across DE/FR/US or NaN if no data.
    """
    sub = raw[(raw["date_utc"].astype(str) == date) & (raw["geo"].isin(CONTROL_GEOS))]
    if sub.empty:
        return float("nan")
    return float(sub[metric_col].astype(float).mean())


def _assert_raw_sanity_checks(raw: pd.DataFrame) -> None:
    """Function summary: verify known Italy peaks on unsmoothed data.

    Parameters:
    - raw: study-window panel before rolling.

    Returns:
    - None; exits with code 1 and diagnostic output if checks fail.
    """
    errors: list[str] = []

    vpn_apr1 = _it_value(raw, "2023-04-01", "vpn_interest")
    if vpn_apr1 != 100:
        errors.append(f"IT vpn_interest on 2023-04-01 expected 100, got {vpn_apr1}")

    for d in ("2023-04-07", "2023-04-08", "2023-04-12"):
        v = _it_value(raw, d, "vpn_interest")
        if not (35 <= v <= 45):
            errors.append(f"IT vpn_interest on {d} expected high-30s (35–45), got {v}")

    cg_ban = _it_value(raw, "2023-03-31", "chatgpt_interest")
    if cg_ban != 100:
        errors.append(f"IT chatgpt_interest on 2023-03-31 expected 100, got {cg_ban}")

    for d in ("2023-04-06", "2023-04-07", "2023-04-08"):
        it_v = _it_value(raw, d, "chatgpt_interest")
        ctrl_v = _control_mean(raw, d, "chatgpt_interest")
        if not (it_v < ctrl_v):
            errors.append(
                f"IT chatgpt_interest on {d} expected below control mean "
                f"({ctrl_v:.1f}), got IT={it_v}"
            )

    cg_lift = _it_value(raw, "2023-04-28", "chatgpt_interest")
    cg_lift2 = _it_value(raw, "2023-04-29", "chatgpt_interest")
    if cg_lift != 62:
        errors.append(f"IT chatgpt_interest on 2023-04-28 expected 62, got {cg_lift}")
    if cg_lift2 != 71:
        errors.append(f"IT chatgpt_interest on 2023-04-29 expected 71, got {cg_lift2}")

    if errors:
        it_diag = raw[raw["geo"] == ITALY_GEO].sort_values("date_utc")
        cols = ["date_utc", "vpn_interest", "chatgpt_interest"]
        print(
            "[plot_circumvention_thesis_figures] SANITY CHECK FAILED — "
            "column mapping may be wrong:",
            file=sys.stderr,
            flush=True,
        )
        for err in errors:
            print(f"  - {err}", file=sys.stderr, flush=True)
        print("\nItaly raw series (study window):", file=sys.stderr, flush=True)
        print(it_diag[cols].to_string(index=False), file=sys.stderr, flush=True)
        sys.exit(1)


def _smooth_by_geo(df: pd.DataFrame, metric_col: str, rolling_window: int) -> pd.DataFrame:
    """Function summary: apply trailing rolling mean by geo.

    Parameters:
    - df: study-window panel with geo column.
    - metric_col: interest column to smooth.
    - rolling_window: trailing window length in days.

    Returns:
    - Smoothed long frame (date_utc, geo, metric_col).
    """
    slim = df[["date_utc", "geo", metric_col]].copy()
    rolled = grouped_trailing_daily_rolling(
        slim,
        group_col="geo",
        rolling_window_days=rolling_window,
        date_col="date_utc",
    )
    rolled["date_utc"] = pd.to_datetime(rolled["date_utc"])
    return rolled.sort_values(["geo", "date_utc"])


def _index_to_preban_by_geo(smoothed: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Function summary: re-index smoothed series to pre-ban mean = 100 per geo.

    Parameters:
    - smoothed: rolled long panel (date_utc, geo, metric_col).
    - metric_col: smoothed interest column.

    Returns:
    - Long frame (date_utc, geo, value) with indexed values.
    """
    pre_end = pd.Timestamp(PRE_BAN_END)
    parts: list[pd.DataFrame] = []
    for geo, grp in smoothed.groupby("geo", sort=True):
        g = grp.sort_values("date_utc").copy()
        pre = g[g["date_utc"] <= pre_end][metric_col].astype(float)
        base = float(pre.mean()) if pre.notna().any() else float("nan")
        if not base or base <= 0 or pd.isna(base):
            print(
                f"[plot_circumvention_thesis_figures] skip indexing {geo}: "
                f"invalid pre-ban mean ({base})",
                flush=True,
            )
            continue
        out = g[["date_utc", "geo"]].copy()
        out["value"] = 100.0 * g[metric_col].astype(float) / base
        parts.append(out)
    if not parts:
        return pd.DataFrame(columns=["date_utc", "geo", "value"])
    return pd.concat(parts, ignore_index=True)


def _warn_indexed_sanity(
    italy: pd.DataFrame,
    band: pd.DataFrame,
    metric_col: str,
) -> None:
    """Function summary: non-fatal checks that indexed series anchor near 100 pre-ban.

    Parameters:
    - italy: indexed Italy series (date_utc, value).
    - band: indexed control band.
    - metric_col: metric name for log messages.

    Returns:
    - None.
    """
    check_date = pd.Timestamp(PRE_BAN_END)
    tol = 5.0
    it_row = italy[italy["date_utc"] == check_date]
    if not it_row.empty:
        it_v = float(it_row["value"].iloc[0])
        if abs(it_v - INDEXED_REFERENCE_Y) > tol:
            print(
                f"[plot_circumvention_thesis_figures] warn {metric_col}: "
                f"IT indexed on {PRE_BAN_END} = {it_v:.1f} (expected ~100)",
                flush=True,
            )
    band_row = band[band["date_utc"] == check_date]
    if not band_row.empty:
        ctrl_v = float(band_row["ctrl_mean"].iloc[0])
        if abs(ctrl_v - INDEXED_REFERENCE_Y) > tol:
            print(
                f"[plot_circumvention_thesis_figures] warn {metric_col}: "
                f"control mean indexed on {PRE_BAN_END} = {ctrl_v:.1f} (expected ~100)",
                flush=True,
            )
    if metric_col == "vpn_interest":
        ban_mask = (italy["date_utc"] >= pd.Timestamp(BAN_START)) & (
            italy["date_utc"] <= pd.Timestamp(BAN_END)
        )
        peak = float(italy.loc[ban_mask, "value"].max()) if ban_mask.any() else float("nan")
        if pd.notna(peak) and peak < 150:
            print(
                f"[plot_circumvention_thesis_figures] warn {metric_col}: "
                f"IT VPN indexed ban-window peak = {peak:.1f} (expected >150)",
                flush=True,
            )


def _control_band(
    panel: pd.DataFrame,
    value_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Function summary: build Italy series and control min/max/mean band.

    Parameters:
    - panel: long panel with geo and value column.
    - value_col: column holding series values (metric or indexed value).

    Returns:
    - Tuple of (italy_daily, band_daily) with columns date_utc, value / ctrl_min, ctrl_max, ctrl_mean.
    """
    italy = panel[panel["geo"] == ITALY_GEO][["date_utc", value_col]].rename(
        columns={value_col: "value"}
    )
    ctrl = panel[panel["geo"].isin(CONTROL_GEOS)].copy()
    wide = ctrl.pivot(index="date_utc", columns="geo", values=value_col)
    band = wide.assign(
        ctrl_min=wide.min(axis=1),
        ctrl_max=wide.max(axis=1),
        ctrl_mean=wide.mean(axis=1),
    )[["ctrl_min", "ctrl_max", "ctrl_mean"]].reset_index()
    return italy.sort_values("date_utc"), band.sort_values("date_utc")


def _plot_thesis_figure(
    italy: pd.DataFrame,
    band: pd.DataFrame,
    ylabel: str,
    out_path: Path,
    *,
    footnote: str = FOOTNOTE,
    reference_y: float | None = None,
) -> None:
    """Function summary: render single-panel thesis figure with ban annotations.

    Parameters:
    - italy: smoothed Italy series (date_utc, value).
    - band: control band (date_utc, ctrl_min, ctrl_max, ctrl_mean).
    - ylabel: y-axis label.
    - out_path: PNG output path (.txt caption written alongside).
    - footnote: caption text for figure and .txt sidecar.
    - reference_y: optional horizontal reference line (e.g. 100 for indexed plots).

    Returns:
    - None.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x_start = pd.Timestamp(STUDY_START)
    x_end = pd.Timestamp(STUDY_END)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axvspan(
        pd.Timestamp(BAN_START),
        pd.Timestamp(BAN_END),
        color="0.85",
        alpha=0.45,
        zorder=0,
    )
    if reference_y is not None:
        ax.axhline(reference_y, color="0.6", linestyle=":", linewidth=0.8, zorder=1)
    ax.fill_between(
        band["date_utc"],
        band["ctrl_min"],
        band["ctrl_max"],
        color=CONTROL_COLOR,
        alpha=0.25,
        label="Controls (range)",
        zorder=1,
    )
    ax.plot(
        band["date_utc"],
        band["ctrl_mean"],
        color=CONTROL_COLOR,
        linewidth=1.0,
        label="Controls (mean)",
        zorder=2,
    )
    ax.plot(
        italy["date_utc"],
        italy["value"],
        color=ITALY_COLOR,
        linewidth=2.5,
        label="Italy",
        zorder=3,
    )

    ban_ts = pd.Timestamp(BAN_START)
    lift_ts = pd.Timestamp(BAN_END)
    ax.axvline(ban_ts, color="0.35", linestyle="-", linewidth=1.0, zorder=4)
    ax.axvline(lift_ts, color="0.35", linestyle="--", linewidth=1.0, zorder=4)
    y_top = ax.get_ylim()[1]
    ax.text(ban_ts, y_top * 0.98, " ban", fontsize=8, va="top", ha="left", color="0.35")
    ax.text(lift_ts, y_top * 0.98, " lift", fontsize=8, va="top", ha="left", color="0.35")

    ax.set_xlim(x_start, x_end)
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.text(0.01, -0.02, footnote, fontsize=8, wrap=True)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    caption_path = out_path.with_suffix(".txt")
    caption_path.write_text(footnote + "\n", encoding="utf-8")


def main() -> None:
    """Function summary: CLI entry — sanity-check raw data and write thesis figures."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    rolling_window = int(max(1, args.rolling_window))

    table_path = tables_subdir(config, "circumvention") / "circumvention_daily_by_geo.csv"
    raw = _load_study_panel(table_path)
    if raw.empty:
        print("[plot_circumvention_thesis_figures] empty study panel; nothing to plot", flush=True)
        return

    _assert_raw_sanity_checks(raw)

    out_dir = figures_subdir(config, "circumvention") / "thesis"
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric_col, ylabel, stem in FIGURE_SPECS:
        smoothed = _smooth_by_geo(raw, metric_col, rolling_window)

        italy, band = _control_band(smoothed, metric_col)
        level_path = out_dir / f"{stem}.png"
        _plot_thesis_figure(italy, band, ylabel, level_path)
        print(f"[plot_circumvention_thesis_figures] wrote {level_path}", flush=True)
        print(
            f"[plot_circumvention_thesis_figures] wrote {level_path.with_suffix('.txt')}",
            flush=True,
        )

        indexed = _index_to_preban_by_geo(smoothed, metric_col)
        italy_idx, band_idx = _control_band(indexed, "value")
        _warn_indexed_sanity(italy_idx, band_idx, metric_col)
        indexed_path = out_dir / f"{stem}_indexed.png"
        _plot_thesis_figure(
            italy_idx,
            band_idx,
            INDEXED_YLABEL,
            indexed_path,
            footnote=FOOTNOTE_INDEXED,
            reference_y=INDEXED_REFERENCE_Y,
        )
        print(f"[plot_circumvention_thesis_figures] wrote {indexed_path}", flush=True)
        print(
            f"[plot_circumvention_thesis_figures] wrote {indexed_path.with_suffix('.txt')}",
            flush=True,
        )

    print(f"[plot_circumvention_thesis_figures] wrote figures to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
