"""
Script summary:
Prepare raw parallel lexicon CSVs: merge ideology_parallel into polarization_lexicon_parallel
and export style_phrase_parallel.csv from archived style phrase txt (optional).

Functionality:
- Appends ideology_parallel rows not already covered by polarization ideology IT lemmas.
- Writes style_phrase_parallel.csv (lexicon, IT, EN, DE) with one phrase per row.
- Optionally writes parallel_vs_config_gap.csv for manual IT curation.

How to apply/run:
  .venv/bin/python scripts/devtools/prepare_parallel_lexicon_raw.py
  .venv/bin/python scripts/devtools/prepare_parallel_lexicon_raw.py --gap-report
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
from typing import Dict, List, Set


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

POLARIZATION_HEADER = [
    "lexicon",
    "bucket",
    "slot_concept",
    "type",
    "IT",
    "IT_grade",
    "EN (US/UK)",
    "EN_grade",
    "DE",
    "DE_grade",
    "notes",
]

STYLE_KINDS = ("hedging", "signposting", "polite_closer")
STYLE_LANG_SUFFIX = {"it": "IT", "en": "EN", "de": "DE"}


def _norm_term(term: str) -> str:
    """Function summary: normalize a lemma key for deduplication."""
    return " ".join((term or "").strip().lower().split())


def _split_cell(cell: str) -> List[str]:
    """Function summary: split a language cell on semicolons only."""
    return [p.strip() for p in (cell or "").split(";") if p.strip()]


def _existing_ideology_it_keys(rows: List[Dict[str, str]]) -> Set[str]:
    """Function summary: normalized IT ideology lemmas already in polarization CSV."""
    keys: Set[str] = set()
    for row in rows:
        if (row.get("lexicon") or "").strip().lower() != "ideology":
            continue
        for piece in _split_cell(row.get("IT", "")):
            keys.add(_norm_term(piece))
    return keys


def merge_ideology_parallel(
    polarization_path: Path,
    ideology_path: Path,
) -> int:
    """Function summary: append ideology_parallel rows missing from polarization ideology.

    Parameters:
    - polarization_path: polarization_lexicon_parallel.csv path.
    - ideology_path: ideology_parallel.csv path.

    Returns:
    - Number of rows appended.
    """
    with polarization_path.open(encoding="utf-8-sig", newline="") as f:
        pol_rows = list(csv.DictReader(f))
    existing = _existing_ideology_it_keys(pol_rows)
    appended = 0
    with ideology_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            it_cell = (row.get("IT") or "").strip()
            if not it_cell:
                continue
            new_pieces = [_norm_term(p) for p in _split_cell(it_cell)]
            if all(p in existing for p in new_pieces):
                continue
            pole = (row.get("pole") or "").strip().lower()
            if pole not in ("left", "right"):
                continue
            concept = (row.get("concept") or "").strip()
            en_cell = (row.get("EN") or "").strip()
            de_cell = (row.get("DE") or "").strip()
            pol_rows.append(
                {
                    "lexicon": "ideology",
                    "bucket": pole,
                    "slot_concept": concept,
                    "type": "generic",
                    "IT": it_cell,
                    "IT_grade": "2",
                    "EN (US/UK)": en_cell,
                    "EN_grade": "2",
                    "DE": de_cell,
                    "DE_grade": "2",
                    "notes": f"Merged from ideology_parallel.csv ({concept})",
                }
            )
            for p in new_pieces:
                existing.add(p)
            appended += 1
    with polarization_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=POLARIZATION_HEADER, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pol_rows)
    return appended


def _read_phrase_lines(path: Path) -> List[str]:
    """Function summary: load lowercased phrases from a flat lexicon txt file."""
    if not path.is_file():
        return []
    out: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip().lower()
        if line:
            out.append(line)
    return out


def export_style_phrase_parallel(
    lex_dir: Path,
    out_path: Path,
) -> int:
    """Function summary: write style_phrase_parallel.csv from config phrase txt files.

    Parameters:
    - lex_dir: config/archive/lexicons (categorized/ for gap report).
    - out_path: output CSV path.

    Returns:
    - Row count written.
    """
    by_lex: Dict[str, Dict[str, List[str]]] = {
        kind: {"IT": [], "EN": [], "DE": []} for kind in STYLE_KINDS
    }
    for kind in STYLE_KINDS:
        for lang, col in STYLE_LANG_SUFFIX.items():
            path = lex_dir / f"{kind}_{lang}.txt"
            by_lex[kind][col].extend(_read_phrase_lines(path))
    rows: List[Dict[str, str]] = []
    for kind in STYLE_KINDS:
        max_len = max(len(by_lex[kind][c]) for c in ("IT", "EN", "DE"))
        for i in range(max_len):
            rows.append(
                {
                    "lexicon": kind,
                    "IT": by_lex[kind]["IT"][i] if i < len(by_lex[kind]["IT"]) else "",
                    "EN": by_lex[kind]["EN"][i] if i < len(by_lex[kind]["EN"]) else "",
                    "DE": by_lex[kind]["DE"][i] if i < len(by_lex[kind]["DE"]) else "",
                }
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lexicon", "IT", "EN", "DE"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_gap_report(
    lex_dir: Path,
    polarization_path: Path,
    tables_dir: Path,
) -> Path:
    """Function summary: list ideology_it.txt terms missing from polarization IT ideology.

    Parameters:
    - lex_dir: config/archive/lexicons root.
    - polarization_path: merged polarization CSV.
    - tables_dir: output directory.

    Returns:
    - Path to gap CSV.
    """
    with polarization_path.open(encoding="utf-8-sig", newline="") as f:
        pol_rows = list(csv.DictReader(f))
    parallel_keys = _existing_ideology_it_keys(pol_rows)
    missing: List[Dict[str, str]] = []
    path = lex_dir / "categorized" / "ideology_it.txt"
    if not path.is_file():
        path = lex_dir / "ideology_it.txt"
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            cat, term = line.split(":", 1)
            key = _norm_term(term)
            if key and key not in parallel_keys:
                missing.append({"category": cat.strip(), "term": term.strip(), "source": "ideology_it.txt"})
    out_dir = tables_dir / "lexicon_export"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "parallel_vs_config_gap.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "term", "source"])
        writer.writeheader()
        writer.writerows(missing)
    return out_path


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI for raw lexicon preparation."""
    parser = argparse.ArgumentParser(description="Prepare parallel lexicon raw CSVs.")
    parser.add_argument("--gap-report", action="store_true", help="Write parallel_vs_config_gap.csv")
    return parser.parse_args()


def main() -> None:
    """Function summary: run merge, style export, and optional gap report."""
    args = parse_args()
    root = PROJECT_ROOT
    raw = root / "data" / "raw"
    pol_path = raw / "polarization_lexicon_parallel.csv"
    ide_path = raw / "ideology_parallel.csv"
    style_path = raw / "style_phrase_parallel.csv"
    n_merge = merge_ideology_parallel(pol_path, ide_path)
    archive_lex = root / "config" / "archive" / "lexicons"
    style_src = archive_lex / "style_phrases"
    n_style = export_style_phrase_parallel(style_src, style_path)
    print(f"[prepare] merged {n_merge} ideology_parallel rows -> {pol_path.name}")
    print(f"[prepare] wrote {n_style} style rows -> {style_path.name}")
    if args.gap_report:
        gap = write_gap_report(
            archive_lex,
            pol_path,
            root / "results" / "tables" / "italy_polarization",
        )
        print(f"[prepare] gap report -> {gap}")


if __name__ == "__main__":
    main()
