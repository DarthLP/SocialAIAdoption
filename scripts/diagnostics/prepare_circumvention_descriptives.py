"""
Script summary:
Build DiD-ready circumvention panels from Tor Metrics and Google Trends VPN + ChatGPT data.

Functionality:
- Loads combined circumvention CSVs under data/raw/circumvention/.
- Writes geo-day and geo-period (1d/3d/7d launch-aligned) tables with post/treated flags.
- Emits methods note on Trends normalization and Tor sparsity.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_circumvention_descriptives.py \
    --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

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

from scripts.diagnostics.descriptives_util import assign_period_start, event_dates_from_config  # noqa: E402
from src.circumvention import (  # noqa: E402
    build_circumvention_geo_panel,
    enrich_daily_with_transforms,
    load_circumvention_daily,
)
from src.config_utils import load_circumvention_config, load_config, tables_subdir  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare circumvention descriptives tables.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    return parser.parse_args()


def _write_methods_note(out_dir: Path, circ_cfg: dict, launch: str, lift: str) -> None:
    """Function summary: write circumvention_methods_note.txt for DiD interpretation."""
    treated = circ_cfg.get("treated_geo", "IT")
    controls = ", ".join(circ_cfg.get("control_geos") or [])
    text = f"""# Circumvention methods note (auto-generated)

## Sources
- Google Trends topic "Virtual private network" (not bare keyword VPN).
- Google Trends topic "ChatGPT" (attention/salience proxy, not usage or adoption).
- Tor Metrics daily relay and bridge user estimates (Kreitmeir & Raschky 2023 replication).

## Event anchors
- Ban onset (launch): {launch}
- Ban lift (reference): {lift}
- Treated geo: {treated}; controls: {controls}

## DiD usage
- On circumvention_panel_by_geo_*: outcome = vpn_interest / chatgpt_interest / tor_*; treatment = treated x post (IT vs other geos).
- Google Trends levels are NOT comparable across countries; use within-geo over-time variation or treated x post on IT only.
- chatgpt_interest measures search attention to ChatGPT, not confirmed tool usage.
- Tor series have sparse calendar days; missing days remain NaN. Multi-day bins average available days only.

## Reddit merges
- did_country_panel_* joins geo-matched VPN/Tor to polarization country_panel outcomes.
- semantic_axis_panel_* adds vpn_interest_it / tor_*_it by period_start (Italy national proxy).
"""
    (out_dir / "circumvention_methods_note.txt").write_text(text, encoding="utf-8")


def main() -> None:
    """Function summary: CLI entry for circumvention panel preparation."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    circ_cfg = load_circumvention_config(config)
    start, end_excl, launch, lift = event_dates_from_config(config)
    panel_bin_days = list(circ_cfg.get("panel_bin_days") or [1, 3, 7])

    out_dir = tables_subdir(config, "circumvention")
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = load_circumvention_daily(PROJECT_ROOT, config, start=start, end_exclusive=end_excl)
    daily = enrich_daily_with_transforms(daily)
    daily.to_csv(out_dir / "circumvention_daily_by_geo.csv", index=False)
    print(
        f"[prepare_circumvention_descriptives] daily rows={len(daily)} geos={daily['geo'].nunique()}",
        flush=True,
    )

    for bin_days in panel_bin_days:
        panel = build_circumvention_geo_panel(
            daily, launch, int(bin_days), assign_period_start=assign_period_start
        )
        path = out_dir / f"circumvention_panel_by_geo_{int(bin_days)}d.csv"
        panel.to_csv(path, index=False)
        print(
            f"[prepare_circumvention_descriptives] bin_days={bin_days} rows={len(panel)} -> {path.name}",
            flush=True,
        )

    _write_methods_note(out_dir, circ_cfg, launch, lift)
    print(f"[prepare_circumvention_descriptives] wrote tables to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
