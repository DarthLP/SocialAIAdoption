"""
Script summary:
Generate stopword lists for Wordfish DTM pruning (it, en, de) from a pinned upstream source.

Functionality:
- Writes config/lexicons/stopwords_{it,en,de}.txt with provenance headers.
- Uses stopwordsiso when installed; falls back to NLTK corpus stopwords.

How to apply/run:
  .venv/bin/python scripts/devtools/generate_wordfish_stopwords.py
  .venv/bin/python scripts/devtools/generate_wordfish_stopwords.py --output-dir config/lexicons
"""

from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set


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

LANGS = ("it", "en", "de")


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate Wordfish stopword files.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="config/lexicons",
        help="Directory for stopwords_<lang>.txt",
    )
    return parser.parse_args()


def _load_via_stopwordsiso() -> tuple[Dict[str, Set[str]], str, str]:
    """Function summary: load stopwords from stopwordsiso package.

    Returns:
    - Tuple (lang->set, source_name, version_string).
    """
    import stopwordsiso as swiso  # type: ignore

    version = getattr(swiso, "__version__", "unknown")
    out: Dict[str, Set[str]] = {}
    for lang in LANGS:
        words = swiso.stopwords(lang)
        out[lang] = {w.strip().lower() for w in words if w.strip()}
    return out, "stopwordsiso", str(version)


def _load_via_nltk() -> tuple[Dict[str, Set[str]], str, str]:
    """Function summary: load stopwords from NLTK corpora.

    Returns:
    - Tuple (lang->set, source_name, version_string).
    """
    import nltk
    from nltk.corpus import stopwords as nltk_sw

    try:
        nltk_sw.words("italian")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    mapping = {
        "it": "italian",
        "en": "english",
        "de": "german",
    }
    out: Dict[str, Set[str]] = {}
    for lang, corpus in mapping.items():
        out[lang] = {w.strip().lower() for w in nltk_sw.words(corpus) if w.strip()}
    return out, "nltk.corpus.stopwords", getattr(nltk, "__version__", "unknown")


def load_stopword_sets() -> tuple[Dict[str, Set[str]], str, str]:
    """Function summary: load stopwords preferring stopwordsiso then NLTK.

    Returns:
    - Tuple (lang->set, source, version).
    """
    try:
        return _load_via_stopwordsiso()
    except ImportError:
        return _load_via_nltk()


def write_stopword_file(
    path: Path,
    lang: str,
    words: Set[str],
    source: str,
    version: str,
) -> None:
    """Function summary: write one stopwords file with provenance header.

    Parameters:
    - path: output .txt path.
    - lang: language code.
    - words: stopword set.
    - source: upstream package name.
    - version: pinned version string.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = [
        f"# source: {source}",
        f"# version: {version}",
        f"# language: {lang}",
        f"# generated_utc: {stamp}",
        f"# generator: scripts/devtools/generate_wordfish_stopwords.py",
        "",
    ]
    body = sorted(words)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")


def main() -> None:
    """Function summary: write stopwords_it/en/de.txt in one pass."""
    args = parse_args()
    out_dir = PROJECT_ROOT / args.output_dir
    sets, source, version = load_stopword_sets()
    for lang in LANGS:
        path = out_dir / f"stopwords_{lang}.txt"
        write_stopword_file(path, lang, sets[lang], source, version)
        print(f"[generate_wordfish_stopwords] wrote {path} ({len(sets[lang])} words)", flush=True)


if __name__ == "__main__":
    main()
