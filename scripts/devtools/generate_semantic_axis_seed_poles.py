"""
Script summary:
Materialize per-language semantic-axis pole lists from parallel seed CSVs.

Functionality:
- Reads parallel seed CSVs under data/raw/seeds/ (ideology, emotion, aggression, economic, cultural,
  nationalism, anti_establishment).
- Writes per-language pole txt files under data/raw/seeds/poles/.

How to apply/run:
  .venv/bin/python scripts/devtools/generate_semantic_axis_seed_poles.py
"""

from __future__ import annotations

import csv
import importlib.util
import re
from pathlib import Path


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


PROJECT_ROOT = _setup_project_root(Path(__file__))
SEEDS_DIR = PROJECT_ROOT / "data" / "raw" / "seeds"
POLES_DIR = SEEDS_DIR / "poles"
LANG_COLS = {"it": "IT", "en": "EN", "de": "DE"}


def _terms_from_csv(csv_path: Path, lang: str, pole: str) -> list[str]:
    """Function summary: extract lowercase terms for one pole and language column."""
    col = LANG_COLS[lang]
    terms: list[str] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row.get("pole") or "").strip().lower() != pole:
                continue
            cell = (row.get(col) or "").strip()
            for part in re.split(r"[;,]", cell):
                part = part.strip().lower()
                if part:
                    terms.append(part)
    return terms


def _write_pole(path: Path, terms: list[str]) -> None:
    """Function summary: write one term per line to a pole txt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(terms) + ("\n" if terms else ""), encoding="utf-8")


def main() -> None:
    """Function summary: generate all pole txt files under data/raw/seeds/poles/."""
    ideology_csv = SEEDS_DIR / "ideology_parallel.csv"
    emotion_csv = SEEDS_DIR / "emotion_cognition_parallel.csv"
    aggression_csv = SEEDS_DIR / "aggression_parallel.csv"
    if not ideology_csv.is_file():
        raise FileNotFoundError(f"Missing {ideology_csv}")
    if not emotion_csv.is_file():
        raise FileNotFoundError(f"Missing {emotion_csv}")
    if not aggression_csv.is_file():
        raise FileNotFoundError(f"Missing {aggression_csv}")

    from src.embeddings import NEUTRAL_POLE_TERMS

    extended_specs = (
        ("economic", "economic_parallel.csv", "market", "equality"),
        ("cultural", "cultural_parallel.csv", "traditional", "progressive"),
        ("nationalism", "nationalism_parallel.csv", "nationalist", "cosmopolitan"),
        ("anti_establishment", "anti_establishment_parallel.csv", "anti_est", "pro_inst"),
    )

    for lang in LANG_COLS:
        _write_pole(POLES_DIR / f"ideology_pos_{lang}.txt", _terms_from_csv(ideology_csv, lang, "right"))
        _write_pole(POLES_DIR / f"ideology_neg_{lang}.txt", _terms_from_csv(ideology_csv, lang, "left"))
        _write_pole(POLES_DIR / f"emotion_pos_{lang}.txt", _terms_from_csv(emotion_csv, lang, "emotion"))
        _write_pole(POLES_DIR / f"emotion_neg_{lang}.txt", _terms_from_csv(emotion_csv, lang, "cognition"))
        ag_pos = _terms_from_csv(aggression_csv, lang, "aggression")
        if len(ag_pos) != 25:
            raise ValueError(f"aggression_parallel.csv must yield 25 terms for {lang}, got {len(ag_pos)}")
        _write_pole(POLES_DIR / f"aggression_pos_{lang}.txt", ag_pos)
        _write_pole(
            POLES_DIR / f"aggression_neg_{lang}.txt",
            list(NEUTRAL_POLE_TERMS.get(lang, NEUTRAL_POLE_TERMS["en"])),
        )
        for axis, csv_name, pos_pole, neg_pole in extended_specs:
            csv_path = SEEDS_DIR / csv_name
            if csv_path.is_file():
                _write_pole(
                    POLES_DIR / f"{axis}_pos_{lang}.txt",
                    _terms_from_csv(csv_path, lang, pos_pole),
                )
                _write_pole(
                    POLES_DIR / f"{axis}_neg_{lang}.txt",
                    _terms_from_csv(csv_path, lang, neg_pole),
                )
        print(f"wrote poles for {lang} -> {POLES_DIR}")


if __name__ == "__main__":
    main()
