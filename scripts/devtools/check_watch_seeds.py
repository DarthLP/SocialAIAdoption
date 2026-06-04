"""
Script summary:
FastText Watch-seed gate report from semantic_seed_sense_audit.xlsx (no CSV rewrite).

Functionality:
- Parses Watch verdict rows for issue axes and runs one fastText load per language.
- Prints a compact summary and writes watch_seed_gate.csv (same schema as export).

How to apply/run:
  .venv/bin/python scripts/devtools/check_watch_seeds.py
  .venv/bin/python scripts/devtools/check_watch_seeds.py --audit data/raw/semantic_seed_sense_audit.xlsx
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


def _setup_project_root(caller_file: Path) -> Path:
    """Function summary: resolve repo root via scripts/_bootstrap.py."""
    for parent in caller_file.resolve().parents:
        if parent.name == "scripts" and (parent / "_bootstrap.py").is_file():
            spec = importlib.util.spec_from_file_location(
                "_socialai_bootstrap_mod", parent / "_bootstrap.py"
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("Failed to load scripts/_bootstrap.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.setup_project_path(caller_file)
    raise RuntimeError("Could not locate scripts/_bootstrap.py")


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI for Watch-only gate check."""
    parser = argparse.ArgumentParser(description="Check Watch seeds against fastText (3 language waves).")
    parser.add_argument(
        "--audit",
        type=str,
        default="data/raw/semantic_seed_sense_audit.xlsx",
    )
    return parser.parse_args()


def main() -> None:
    """Function summary: run Watch gate and print summary table."""
    args = parse_args()
    project_root = _setup_project_root(Path(__file__))
    audit_path = Path(args.audit)
    if not audit_path.is_absolute():
        audit_path = project_root / audit_path

    from src.config_utils import load_config, load_semantic_axis_config

    export_path = project_root / "scripts" / "devtools" / "export_semantic_seed_audit.py"
    spec = importlib.util.spec_from_file_location("export_semantic_seed_audit", export_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {export_path}")
    export_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_mod)

    sem_cfg = load_semantic_axis_config(load_config(project_root / "config/italy_polarization_setup.yaml"))
    by_axis = export_mod.collect_rows_by_axis(export_mod.read_audit_sheet(audit_path))
    passed_by_axis, gate_log = export_mod.gate_all_watch_rows_language_wave(
        project_root, sem_cfg, by_axis, log_prefix="check_watch_seeds"
    )

    out_dir = project_root / "results" / "tables" / "italy_polarization" / "semantic_axis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "watch_seed_gate.csv"
    pd.DataFrame(gate_log).to_csv(out_path, index=False)

    print("\ncheck_watch_seeds summary (included = pass all 3 languages):")
    print("-" * 72)
    concepts = sorted({(g["axis"], g["concept"]) for g in gate_log})
    for axis, concept in concepts:
        included = any(
            g["included"] for g in gate_log if g["axis"] == axis and g["concept"] == concept
        )
        fails = [
            f"{g['lang']}:{g.get('reason') or ('ok' if g['in_vocab'] else 'oov')}"
            for g in gate_log
            if g["axis"] == axis and g["concept"] == concept and not g["included"]
        ]
        status = "INCLUDE" if included else "SKIP"
        note = "" if included else f" ({', '.join(fails)})"
        print(f"  {status:7} {axis:18} {concept:22}{note}")

    n_inc = sum(len(v) for v in passed_by_axis.values())
    print(f"\n[check_watch_seeds] wrote {out_path} ({n_inc} concepts would be added to seed CSVs)", flush=True)


if __name__ == "__main__":
    main()
