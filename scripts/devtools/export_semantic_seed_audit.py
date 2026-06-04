"""
Script summary:
Export issue-specific semantic seed CSVs from semantic_seed_sense_audit.xlsx.

Functionality:
- Parses the Seed audit sheet into economic, cultural, nationalism, anti_establishment parallel CSVs.
- Includes Keep verdict rows; gates Watch rows via fastText in-vocab and draft-axis cosine direction.
- Watch gating uses one fastText load per language (it, en, de), then unload (fixed RAM).
- In-vocab checks use fastText embeddings, not polarization_lexicon_parallel.csv.
- Writes watch_seed_gate.csv under results/tables/italy_polarization/semantic_axis/.

How to apply/run:
  .venv/bin/python scripts/devtools/export_semantic_seed_audit.py
  .venv/bin/python scripts/devtools/export_semantic_seed_audit.py --skip-watch-gate
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

# Audit axis -> (output filename, pole label in audit -> CSV pole value)
AXIS_EXPORT: Dict[str, Tuple[str, Dict[str, str]]] = {
    "economic": (
        "economic_parallel.csv",
        {"+ market": "market", "- equality": "equality"},
    ),
    "cultural": (
        "cultural_parallel.csv",
        {"+ traditional": "traditional", "- progressive": "progressive"},
    ),
    "nationalism": (
        "nationalism_parallel.csv",
        {"+ nationalist": "nationalist", "- cosmopolitan": "cosmopolitan"},
    ),
    "anti_establishment": (
        "anti_establishment_parallel.csv",
        {"+ anti-est": "anti_est", "- pro-inst": "pro_inst"},
    ),
}

LANGS = ("it", "en", "de")
LANG_COL = {"it": "IT", "en": "EN", "de": "DE"}


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


def read_audit_sheet(xlsx_path: Path) -> List[Dict[str, str]]:
    """Function summary: parse Seed audit sheet rows from xlsx XML."""
    rows: List[Dict[str, str]] = []
    with zipfile.ZipFile(xlsx_path) as zf:
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    for row in root.findall(".//m:sheetData/m:row", _NS):
        cells: Dict[str, str] = {}
        for cell in row.findall("m:c", _NS):
            ref = cell.get("r", "")
            col = "".join(ch for ch in ref if ch.isalpha())
            is_elem = cell.find("m:is", _NS)
            val = ""
            if cell.get("t") == "inlineStr" and is_elem is not None:
                t_elem = is_elem.find(".//m:t", _NS)
                val = (t_elem.text or "") if t_elem is not None else ""
            cells[col] = val.strip()
        if not cells.get("A"):
            continue
        rows.append(
            {
                "axis": cells.get("A", ""),
                "pole": cells.get("B", ""),
                "concept": cells.get("C", ""),
                "IT": cells.get("D", ""),
                "EN": cells.get("E", ""),
                "DE": cells.get("F", ""),
                "status": cells.get("G", ""),
                "verdict": cells.get("H", ""),
            }
        )
    return rows


def collect_rows_by_axis(
    audit_rows: List[Dict[str, str]],
) -> Dict[str, List[Dict[str, str]]]:
    """Function summary: group audit rows by axis, excluding Drop verdict."""
    out: Dict[str, List[Dict[str, str]]] = {k: [] for k in AXIS_EXPORT}
    for row in audit_rows:
        axis = row["axis"]
        if axis not in AXIS_EXPORT:
            continue
        if (row.get("verdict") or "").strip().lower() == "drop":
            continue
        out[axis].append(row)
    return out


def _write_parallel_csv(path: Path, rows: List[Dict[str, str]], pole_map: Dict[str, str]) -> int:
    """Function summary: write pole,concept,IT,EN,DE CSV from normalized audit rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pole", "concept", "IT", "EN", "DE"])
        writer.writeheader()
        for row in rows:
            pole_key = row["pole"]
            pole = pole_map.get(pole_key)
            if not pole:
                continue
            writer.writerow(
                {
                    "pole": pole,
                    "concept": row["concept"],
                    "IT": row["IT"],
                    "EN": row["EN"],
                    "DE": row["DE"],
                }
            )
            written += 1
    return written


def _terms_for_pole(
    keep_rows: List[Dict[str, str]],
    pole_map: Dict[str, str],
    pole_key: str,
    lang_col: str,
) -> List[str]:
    """Function summary: extract Keep-row seed strings for one pole and language column."""
    target = pole_map.get(pole_key)
    terms: List[str] = []
    for row in keep_rows:
        if pole_map.get(row["pole"]) != target:
            continue
        cell = (row.get(lang_col) or "").strip().lower()
        if cell:
            terms.append(cell)
    return terms


def gate_all_watch_rows_language_wave(
    project_root: Path,
    sem_cfg: Dict[str, Any],
    by_axis: Dict[str, List[Dict[str, str]]],
    log_prefix: str = "export_semantic_seed_audit",
) -> Tuple[Dict[str, List[Dict[str, str]]], List[Dict[str, Any]]]:
    """Function summary: gate Watch rows with one fastText load per language (it, en, de).

    Parameters:
    - project_root: repository root.
    - sem_cfg: semantic_axis config block.
    - by_axis: axis name -> all non-Drop audit rows for that axis.
    - log_prefix: log tag prefix.

    Returns:
    - (passed_watch_by_axis, gate_log rows for watch_seed_gate.csv).
    """
    from src.embeddings import (
        _cosine,
        _seed_term_in_vocab,
        build_axis,
        comment_vector,
        run_language_vector_wave,
    )
    from src.political_lexicon import tokenize

    axis_ctx: Dict[str, Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, str]]] = {}
    n_watch = 0
    for axis, rows in by_axis.items():
        if axis not in AXIS_EXPORT:
            continue
        _, pole_map = AXIS_EXPORT[axis]
        keep = [r for r in rows if (r.get("verdict") or "").strip().lower() == "keep"]
        watch = [r for r in rows if (r.get("verdict") or "").strip().lower() == "watch"]
        if watch:
            axis_ctx[axis] = (keep, watch, pole_map)
            n_watch += len(watch)

    gate_log: List[Dict[str, Any]] = []

    def _wave_callback(lang: str, kv: Any) -> None:
        col = LANG_COL[lang]
        print(f"[{log_prefix}] lang={lang} gating watch terms (n={n_watch}) ...", flush=True)
        if kv is None:
            for axis, (_, watch, _) in axis_ctx.items():
                for row in watch:
                    gate_log.append(
                        {
                            "axis": axis,
                            "concept": row["concept"],
                            "lang": lang,
                            "term": (row.get(col) or "").strip().lower(),
                            "in_vocab": 0,
                            "cosine": float("nan"),
                            "expected_sign": 1 if row["pole"].strip().startswith("+") else -1,
                            "included": 0,
                            "reason": "fasttext_missing",
                        }
                    )
            return

        draft_axes: Dict[str, Any] = {}
        for axis, (keep, watch, pole_map) in axis_ctx.items():
            pos_key = next(k for k in pole_map if k.startswith("+"))
            neg_key = next(k for k in pole_map if k.startswith("-"))
            draft_axes[axis] = build_axis(
                _terms_for_pole(keep, pole_map, pos_key, col),
                _terms_for_pole(keep, pole_map, neg_key, col),
                kv,
            )
            for row in watch:
                term = (row.get(col) or "").strip().lower()
                expected_sign = 1 if row["pole"].strip().startswith("+") else -1
                in_vocab = int(_seed_term_in_vocab(term, kv)) if term else 0
                cosine = float("nan")
                if in_vocab and term:
                    vec, _ = comment_vector(tokenize(term), kv)
                    if vec is not None:
                        cosine = _cosine(vec, draft_axes[axis])
                gate_log.append(
                    {
                        "axis": axis,
                        "concept": row["concept"],
                        "lang": lang,
                        "term": term,
                        "in_vocab": in_vocab,
                        "cosine": cosine,
                        "expected_sign": expected_sign,
                        "included": 0,
                        "reason": "",
                    }
                )

    print(f"[{log_prefix}] watch gate: 3 language waves (one fastText model at a time)", flush=True)
    run_language_vector_wave(project_root, sem_cfg, _wave_callback)

    passed_by_axis: Dict[str, List[Dict[str, str]]] = {axis: [] for axis in axis_ctx}
    for axis, (_, watch, _) in axis_ctx.items():
        for row in watch:
            checks = [g for g in gate_log if g["concept"] == row["concept"] and g["axis"] == axis]
            ok = (
                len(checks) == len(LANGS)
                and all(int(g["in_vocab"]) == 1 for g in checks)
                and all(
                    not (g["term"] and g["in_vocab"])
                    or (g["cosine"] == g["cosine"] and (g["cosine"] * g["expected_sign"]) > 0)
                    for g in checks
                )
            )
            for g in checks:
                g["included"] = int(ok)
                if not ok and not g.get("reason"):
                    g["reason"] = "oov" if not g["in_vocab"] else "wrong_direction"
            if ok:
                passed_by_axis[axis].append(row)

    n_pass = sum(len(v) for v in passed_by_axis.values())
    print(f"[{log_prefix}] watch gate done: included {n_pass}/{n_watch} concepts", flush=True)
    return passed_by_axis, gate_log


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI for audit export."""
    parser = argparse.ArgumentParser(description="Export semantic seed CSVs from audit xlsx.")
    parser.add_argument(
        "--audit",
        type=str,
        default="data/raw/semantic_seed_sense_audit.xlsx",
        help="Path to audit workbook.",
    )
    parser.add_argument(
        "--skip-watch-gate",
        action="store_true",
        help="Exclude Watch verdict rows entirely (no fastText check).",
    )
    return parser.parse_args()


def main() -> None:
    """Function summary: export parallel seed CSVs and watch gate table."""
    args = parse_args()
    project_root = _setup_project_root(Path(__file__))
    audit_path = Path(args.audit)
    if not audit_path.is_absolute():
        audit_path = project_root / audit_path
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing audit file: {audit_path}")

    from src.config_utils import load_config, load_semantic_axis_config

    sem_cfg = load_semantic_axis_config(load_config(project_root / "config/italy_polarization_setup.yaml"))
    seeds_dir = project_root / "data" / "raw" / "seeds"
    audit_rows = read_audit_sheet(audit_path)
    by_axis = collect_rows_by_axis(audit_rows)

    passed_watch_by_axis: Dict[str, List[Dict[str, str]]] = {}
    all_gate_logs: List[Dict[str, Any]] = []
    if not args.skip_watch_gate:
        passed_watch_by_axis, all_gate_logs = gate_all_watch_rows_language_wave(
            project_root, sem_cfg, by_axis
        )

    for axis, (filename, pole_map) in AXIS_EXPORT.items():
        rows = by_axis[axis]
        keep = [r for r in rows if (r.get("verdict") or "").strip().lower() == "keep"]
        if args.skip_watch_gate:
            final = keep
        else:
            final = keep + passed_watch_by_axis.get(axis, [])
        n = _write_parallel_csv(seeds_dir / filename, final, pole_map)
        print(
            f"[export_semantic_seed_audit] {filename}: {n} concepts "
            f"(keep={len(keep)} watch_included={n - len(keep)})",
            flush=True,
        )

    if all_gate_logs and not args.skip_watch_gate:
        out_dir = project_root / "results" / "tables" / "italy_polarization" / "semantic_axis"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_gate_logs).to_csv(out_dir / "watch_seed_gate.csv", index=False)
        print(f"[export_semantic_seed_audit] wrote {out_dir / 'watch_seed_gate.csv'}", flush=True)


if __name__ == "__main__":
    main()
