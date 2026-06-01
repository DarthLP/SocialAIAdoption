"""
Script summary:
Merge circumvention VPN/Tor proxies onto Reddit DiD-ready panels (lexical country + semantic).

Functionality:
- Lexical: bin daily_country_panel (and universe slice) to 1/3/7d, merge geo-matched circumvention.
- Semantic: pass through semantic_axis_panel_* with Italy broadcast columns only (no geo-matched VPN).

Outputs (under results/tables/.../did/panels/):
- panels/country/: did_country_panel_{1,3,7}d.csv, did_country_panel_by_universe_slice_{1,3,7}d.csv
- panels/semantic/: did_semantic_topic_family_{1,3,7}d.csv, language_*, language_universe_*

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_circumvention_descriptives.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/prepare_polarization_descriptives.py --config config/italy_polarization_setup.yaml
  .venv/bin/python scripts/diagnostics/prepare_semantic_axis_descriptives.py --config config/italy_polarization_setup.yaml --panels-only
  .venv/bin/python scripts/diagnostics/prepare_did_merged_panels.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


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

from scripts.diagnostics.descriptives_util import (  # noqa: E402
    bin_lexical_daily_panel,
    event_dates_from_config,
)
from src.circumvention import GEO_MATCHED_INTENSITY_COLS, merge_circumvention_by_geo  # noqa: E402
from src.config_utils import load_circumvention_config, load_config, tables_subdir  # noqa: E402
from src.did.paths import did_panels_dir  # noqa: E402

# topic_family on semantic panels -> country_panel labels in daily_country_panel.csv
# EU_hub_en has no entry in circumvention.country_panel_geo_map; VPN columns stay NaN (by design).
TOPIC_FAMILY_TO_COUNTRY_PANEL: Dict[str, str] = {
    "it_political": "Italy_political",
    "it_pure_political": "Italy_political",
    "it_others": "Italy_others",
    "de": "Germany",
    "us": "US_political",
    "uk": "UK",
    "eu": "EU_hub_en",
}

SEMANTIC_PANEL_SLUGS: tuple[str, ...] = (
    "by_topic_family",
    "by_language",
    "by_language_universe",
)

SEMANTIC_DROP_COLS: tuple[str, ...] = (
    "country_panel",
    "geo",
    "post_circ",
    "treated_circ",
    *GEO_MATCHED_INTENSITY_COLS,
)

PANEL_BIN_DAYS: tuple[int, ...] = (1, 3, 7)


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Merge circumvention onto DiD-ready Reddit panels.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def _prepare_semantic_did_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Function summary: keep Italy broadcast circumvention columns; drop geo-matched VPN."""
    if panel.empty:
        return panel.copy()
    out = panel.copy()
    drop = [c for c in SEMANTIC_DROP_COLS if c in out.columns]
    if drop:
        out = out.drop(columns=drop)
    out["circumvention_intensity_spec"] = "it_broadcast"
    return out


def _merge_country_daily(
    descriptives_dir: Path,
    circum_dir: Path,
    did_dir: Path,
    geo_map: dict,
    bin_days: int,
    launch: str,
) -> None:
    """Function summary: bin lexical country panels and merge geo-matched circumvention."""
    circ_path = circum_dir / f"circumvention_panel_by_geo_{bin_days}d.csv"
    if not circ_path.is_file():
        print(f"[prepare_did_merged_panels] skip: missing {circ_path.name}", flush=True)
        return
    circum = pd.read_csv(circ_path)
    circum = circum.drop(columns=["n_days_in_bin"], errors="ignore")

    specs: List[tuple[str, Sequence[str], str]] = [
        (
            "daily_country_panel.csv",
            ("country_panel",),
            f"did_country_panel_{bin_days}d.csv",
        ),
        (
            "daily_country_panel_by_universe_slice.csv",
            ("country_panel", "universe_slice"),
            f"did_country_panel_by_universe_slice_{bin_days}d.csv",
        ),
    ]
    for src_name, entity_cols, out_name in specs:
        src = descriptives_dir / src_name
        if not src.is_file():
            continue
        panel = pd.read_csv(src)
        if "date_utc" not in panel.columns:
            continue
        binned = bin_lexical_daily_panel(panel, entity_cols, int(bin_days), launch)
        merged = merge_circumvention_by_geo(
            binned,
            circum,
            geo_map,
            panel_geo_col="country_panel",
            date_col="period_start",
        )
        out_path = did_dir / out_name
        merged.to_csv(out_path, index=False)
        print(
            f"[prepare_did_merged_panels] {src_name} -> {out_path.name} rows={len(merged)}",
            flush=True,
        )


def _merge_semantic_panels(semantic_dir: Path, did_dir: Path) -> None:
    """Function summary: write semantic DiD tables from panels (Italy *_it columns only)."""
    for slug in SEMANTIC_PANEL_SLUGS:
        for bin_days in PANEL_BIN_DAYS:
            src = semantic_dir / f"semantic_axis_panel_{slug}_{bin_days}d.csv"
            if not src.is_file():
                continue
            panel = pd.read_csv(src)
            if slug == "by_topic_family" and "topic_family" in panel.columns:
                panel["country_panel"] = panel["topic_family"].astype(str).map(
                    TOPIC_FAMILY_TO_COUNTRY_PANEL
                )
                unmapped = panel["country_panel"].isna()
                if unmapped.any():
                    missing = sorted(panel.loc[unmapped, "topic_family"].astype(str).unique())
                    raise ValueError(
                        "topic_family values missing from TOPIC_FAMILY_TO_COUNTRY_PANEL: "
                        + ", ".join(missing)
                    )
            out = _prepare_semantic_did_frame(panel)
            slug_out = slug.replace("by_", "")
            out_path = did_dir / f"did_semantic_{slug_out}_{bin_days}d.csv"
            out.to_csv(out_path, index=False)
            print(
                f"[prepare_did_merged_panels] semantic {slug} bin={bin_days}d "
                f"rows={len(out)} -> {out_path.name}",
                flush=True,
            )


def main() -> None:
    """Function summary: CLI entry for DiD panel merges."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    circ_cfg = load_circumvention_config(config)
    geo_map = dict(circ_cfg.get("country_panel_geo_map") or {})
    _, _, launch, _ = event_dates_from_config(config)

    descriptives_dir = tables_subdir(config, "descriptives")
    circum_dir = tables_subdir(config, "circumvention")
    semantic_dir = tables_subdir(config, "semantic_axis")
    country_dir = did_panels_dir(config, "country")
    semantic_out_dir = did_panels_dir(config, "semantic")
    country_dir.mkdir(parents=True, exist_ok=True)
    semantic_out_dir.mkdir(parents=True, exist_ok=True)

    for bin_days in PANEL_BIN_DAYS:
        _merge_country_daily(descriptives_dir, circum_dir, country_dir, geo_map, bin_days, launch)
    _merge_semantic_panels(semantic_dir, semantic_out_dir)
    print(
        f"[prepare_did_merged_panels] wrote country -> {country_dir}, semantic -> {semantic_out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
