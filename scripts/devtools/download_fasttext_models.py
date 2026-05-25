"""
Script summary:
One-time download and decompress fastText Common Crawl .bin models for semantic-axis scoring.

Functionality:
- Fetches cc.it.300.bin.gz, cc.en.300.bin.gz, cc.de.300.bin.gz when missing.
- Gunzips into data/external/embeddings/ for reuse by load_vectors.

How to apply/run:
  .venv/bin/python scripts/devtools/download_fasttext_models.py
  .venv/bin/python scripts/devtools/download_fasttext_models.py --lang it
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import shutil
import urllib.request
from pathlib import Path

BASE_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl"
LANGS = ("it", "en", "de")


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
OUT_DIR = PROJECT_ROOT / "data" / "external" / "embeddings"


def _download_one(lang: str, out_dir: Path) -> None:
    """Function summary: download and gunzip one language model if .bin is absent."""
    bin_path = out_dir / f"cc.{lang}.300.bin"
    if bin_path.is_file():
        print(f"skip {lang}: exists {bin_path}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    gz_name = f"cc.{lang}.300.bin.gz"
    url = f"{BASE_URL}/{gz_name}"
    gz_path = out_dir / gz_name
    print(f"downloading {url} -> {gz_path}")
    urllib.request.urlretrieve(url, gz_path)
    print(f"decompressing {gz_path} -> {bin_path}")
    with gzip.open(gz_path, "rb") as src, bin_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    gz_path.unlink(missing_ok=True)
    print(f"done {bin_path}")


def main() -> None:
    """Function summary: CLI entry for fastText model download."""
    parser = argparse.ArgumentParser(description="Download fastText cc.*.300.bin models.")
    parser.add_argument("--lang", choices=LANGS, default=None, help="Single language (default: all).")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory.")
    args = parser.parse_args()
    langs = (args.lang,) if args.lang else LANGS
    for lang in langs:
        _download_one(lang, args.out_dir)


if __name__ == "__main__":
    main()
