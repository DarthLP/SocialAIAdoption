"""
Script summary:
One-off generator for `notebooks/colab_compute_comment_features_gpu.ipynb` embedding
`config/political_forums_setup.yaml` and `src/comment_feature_models.py` so Colab can
run without cloning the repository. Re-run after changing those sources:
`.venv/bin/python scripts/_gen_colab_standalone_nb.py`
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YAML_TEXT = (ROOT / "config/political_forums_setup.yaml").read_text(encoding="utf-8")
MODELS_TEXT = (ROOT / "src" / "comment_feature_models.py").read_text(encoding="utf-8")
needle = "from __future__ import annotations\n"
HELPERS = MODELS_TEXT[MODELS_TEXT.index(needle) :]

if '"""' in YAML_TEXT:
    raise SystemExit("political_forums_setup.yaml must not contain triple-double-quotes for embedding")

CONTROL_CELL = (
    "# -----------------------------------------------------------------------------\n"
    "# MASTER CONFIG — edit before running downstream cells (no Git / no repo clone)\n"
    "# -----------------------------------------------------------------------------\n"
    "from __future__ import annotations\n"
    "from pathlib import Path\n"
    "\n"
    "WORKDIR = Path(\"/content/comment_ml_standalone\")\n"
    "\n"
    "# Google Drive: mirror cleaned_monthly_chunks layout under this folder.\n"
    "DRIVE_CLEANED_MONTHLY_ROOT = \"/content/drive/MyDrive/SocialAIAdoption_interim/cleaned_monthly_chunks\"\n"
    "DRIVE_COMMENT_FEATURES_ML_ROOT = (\n"
    "    \"/content/drive/MyDrive/SocialAIAdoption_interim/comment_features_ml_colab\"\n"
    ")\n"
    "\n"
    "RUN_MODE = \"bounded\"  # \"bounded\" | \"full\"\n"
    "OVERWRITE = False\n"
    "SUBREDDITS = \"\"\n"
    "MONTHS = \"\"\n"
    "DEVICE = \"cuda\"\n"
    "BATCH_SIZE = 128\n"
    "BOUNDED_MAX_TOTAL_MONTH_FILES = 2\n"
    "BOUNDED_MAX_DAYS_PER_MONTH = 10\n"
    "CHECKPOINT_AFTER_RUN = True\n"
    "RUN_TAG = \"\"\n"
    "\n"
    '# Embedded project YAML (defaults match repo — edit model IDs inline if needed)\n'
    "CONFIG_YAML = r'''\n"
    + YAML_TEXT
    + "\n'''\n"
)

RUNTIME_CELL = (
    HELPERS
    + "\n\n"
    + """

import shutil
import time
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
import yaml


def load_config_yaml() -> dict[str, object]:
    # Parse CONFIG_YAML from the control-cell variable.

    out = yaml.safe_load(CONFIG_YAML)
    assert isinstance(out, dict)
    return out


def interim_chunks() -> tuple[Path, Path, Path]:
    interim = WORKDIR / "data" / "interim" / "political_forums"
    return interim, interim / "cleaned_monthly_chunks", interim / "comment_features_ml"


def iter_monthly_files(
    cleaned_monthly_chunks_dir: Path,
    subreddits: list[str],
    allowed_months: set[str],
    max_month_files_per_subreddit: int,
    max_total_month_files: int,
) -> Iterable[tuple[str, Path]]:
    total = 0
    for subreddit in sorted(subreddits):
        sub_dir = cleaned_monthly_chunks_dir / subreddit
        if not sub_dir.exists():
            continue
        per_sub_count = 0
        for file_path in sorted(sub_dir.glob("*.parquet")):
            month = file_path.stem
            if allowed_months and month not in allowed_months:
                continue
            if max_month_files_per_subreddit > 0 and per_sub_count >= max_month_files_per_subreddit:
                break
            if max_total_month_files > 0 and total >= max_total_month_files:
                return
            yield subreddit, file_path
            per_sub_count += 1
            total += 1


def process_month_file_ml_nb(
    subreddit: str,
    file_path: Path,
    output_path: Path,
    overwrite: bool,
    max_days_per_month: int,
    device_arg: str,
    batch_size: int,
    model_config: dict[str, str],
) -> None:
    if output_path.exists() and not overwrite:
        print(f"[colab_ml_nb] skip_existing subreddit={subreddit} month={file_path.stem}", flush=True)
        return

    print(f"[colab_ml_nb] start subreddit={subreddit} month={file_path.stem}", flush=True)
    frame = pd.read_parquet(file_path, columns=["id", "subreddit", "date_utc", "body"])
    frame = frame[frame["subreddit"].astype("string") == subreddit].copy()
    frame["body"] = frame["body"].astype("string").fillna("")
    frame["date_utc"] = frame["date_utc"].astype("string")
    if max_days_per_month > 0 and not frame.empty:
        keep_days = sorted(frame["date_utc"].dropna().unique())[: int(max_days_per_month)]
        frame = frame[frame["date_utc"].isin(keep_days)].copy()
    if frame.empty:
        print(f"[colab_ml_nb] empty_after_filter subreddit={subreddit} month={file_path.stem}", flush=True)
        return

    records: list[dict[str, object]] = []
    texts: list[str] = []
    for row in frame.itertuples(index=False):
        text = str(getattr(row, "body", "") or "")
        records.append(
            {
                "id": str(getattr(row, "id", "") or ""),
                "subreddit": str(getattr(row, "subreddit", "") or ""),
                "date_utc": str(getattr(row, "date_utc", "") or ""),
            }
        )
        texts.append(text[:2000])

    device = resolve_inference_device(device_arg)
    print(f"[colab_ml_nb] device requested={device_arg!r} resolved={device!r}", flush=True)
    append_ml_columns_to_records(records, texts, model_config, device, batch_size)
    ml_ts = datetime.now(timezone.utc).isoformat()
    for rec in records:
        rec["ml_features_computed_at_utc"] = ml_ts

    column_order = [
        "id",
        "subreddit",
        "date_utc",
        "detector_primary_ai_prob",
        "detector_primary_human_score",
        "detector_secondary_ai_prob",
        "detector_secondary_human_score",
        "hostility_score",
        "emotion_anger",
        "emotion_fear",
        "emotion_sadness",
        "emotion_surprise",
        "perplexity",
        "log_perplexity",
        "detector_primary_model_id",
        "detector_secondary_model_id",
        "hostility_model_id",
        "emotion_model_id",
        "perplexity_model_id",
        "device_used",
        "ml_features_computed_at_utc",
    ]
    out_df = pd.DataFrame.from_records(records)
    out_df = out_df[[c for c in column_order if c in out_df.columns]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False, compression="zstd")
    print(f"[colab_ml_nb] done subreddit={subreddit} month={file_path.stem} rows={len(out_df)}", flush=True)


def sync_cleaned_from_drive(
    src_root: Path,
    dst_root: Path,
    subreddit_allowlist: set[str],
    month_allowlist: set[str],
) -> int:
    copied = 0
    dst_root.mkdir(parents=True, exist_ok=True)
    if not src_root.is_dir():
        raise NotADirectoryError(src_root)
    for sub_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        name = sub_dir.name
        if subreddit_allowlist and name not in subreddit_allowlist:
            continue
        out_sub = dst_root / name
        out_sub.mkdir(parents=True, exist_ok=True)
        for fp in sorted(sub_dir.glob("*.parquet")):
            stem = fp.stem
            if month_allowlist and stem not in month_allowlist:
                continue
            shutil.copy2(fp, out_sub / fp.name)
            copied += 1
    print(f"[colab_ml_nb] sync_cleaned copied {copied} parquet → {dst_root}")
    return copied


def sync_tree_to_drive(src: Path, dst: Path) -> int:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(src.rglob("*.parquet")):
        rel = path.relative_to(src)
        out_p = dst / rel
        out_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out_p)
        n += 1
    print(f"[colab_ml_nb] checkpoint copied {n} parquet → {dst}")
    return n


def run_ml_jobs(max_total_files: int, max_days: int) -> None:
    cfg = load_config_yaml()
    _, cleaned_dir, out_ml = interim_chunks()

    configured_subreddits = list(cfg["subreddits"]["primary"])  # type: ignore[index]
    if SUBREDDITS.strip():
        allow_subs = {s.strip() for s in SUBREDDITS.split(",") if s.strip()}
        configured_subreddits = [s for s in configured_subreddits if s in allow_subs]
    allow_months = {m.strip() for m in MONTHS.split(",") if m.strip()}

    feature_cfg = cfg.get("comment_features", {})
    assert isinstance(feature_cfg, dict)
    model_config = {
        "detector_primary": str(feature_cfg.get("detector_primary_model", "desklib/ai-text-detector-v1.01")),
        "detector_secondary": str(
            feature_cfg.get("detector_secondary_model", "fakespot-ai/roberta-base-ai-text-detection-v1")
        ),
        "hostility": str(feature_cfg.get("hostility_model", "unitary/unbiased-toxic-roberta")),
        "emotion": str(feature_cfg.get("emotion_model", "j-hartmann/emotion-english-distilroberta-base")),
        "perplexity": str(feature_cfg.get("perplexity_model", "gpt2")),
    }

    jobs = list(
        iter_monthly_files(
            cleaned_dir,
            configured_subreddits,
            allow_months,
            0,
            max_total_files,
        )
    )
    if not jobs:
        raise FileNotFoundError(f"No cleaned parquet under {cleaned_dir}")

    started = time.perf_counter()
    for idx, (subreddit, fp) in enumerate(jobs, start=1):
        process_month_file_ml_nb(
            subreddit,
            fp,
            out_ml / subreddit / f"{fp.stem}.parquet",
            OVERWRITE,
            max_days,
            DEVICE,
            BATCH_SIZE,
            model_config,
        )
        print(f"[colab_ml_nb] progress {idx}/{len(jobs)} elapsed_s={time.perf_counter() - started:.1f}", flush=True)
"""
)

PATHS_AND_SYNC_IN = """
from pathlib import Path

_drive_cleaned = Path(DRIVE_CLEANED_MONTHLY_ROOT).expanduser().resolve()
_drive_ml_out = Path(DRIVE_COMMENT_FEATURES_ML_ROOT).expanduser().resolve()
drive_output_root = _drive_ml_out / RUN_TAG.strip() if RUN_TAG.strip() else _drive_ml_out
drive_output_root.mkdir(parents=True, exist_ok=True)

WORKDIR.mkdir(parents=True, exist_ok=True)
_, cleaned_local, features_ml_local = interim_chunks()
cleaned_local.mkdir(parents=True, exist_ok=True)
features_ml_local.mkdir(parents=True, exist_ok=True)

if not _drive_cleaned.exists():
    raise FileNotFoundError(f"Drive input not found: {_drive_cleaned}")

allow_subs = {s.strip() for s in SUBREDDITS.split(",") if s.strip()}
allow_months = {m.strip() for m in MONTHS.split(",") if m.strip()}

n_copied = sync_cleaned_from_drive(_drive_cleaned, cleaned_local, allow_subs, allow_months)
if n_copied == 0:
    raise RuntimeError("No parquet copied; check Drive path and SUBREDDITS/MONTHS filters.")
"""

RUN_SELECTION = '''
if RUN_MODE == "bounded":
    run_ml_jobs(int(BOUNDED_MAX_TOTAL_MONTH_FILES), int(BOUNDED_MAX_DAYS_PER_MONTH))
elif RUN_MODE == "full":
    run_ml_jobs(0, 0)
else:
    raise ValueError("RUN_MODE must be bounded or full")
'''

CHECKPOINT = """
_, _c, ml_out = interim_chunks()

if CHECKPOINT_AFTER_RUN:
    sync_tree_to_drive(ml_out, drive_output_root)
else:
    print("CHECKPOINT_AFTER_RUN is False; skipping Drive upload.")
"""

VERIFY = '''
import pandas as pd

_, _, ml_root = interim_chunks()

parquets = sorted(ml_root.rglob("*.parquet"))
if not parquets:
    raise FileNotFoundError(f"No output under {ml_root}")

df = pd.read_parquet(parquets[0])
need = {"id", "detector_primary_ai_prob", "device_used", "ml_features_computed_at_utc"}
missing = need - set(df.columns)
if missing:
    raise AssertionError(f"missing columns: {missing}")
print("sample:", parquets[0])
print("rows:", len(df))
print("device_used:", df["device_used"].iloc[0] if len(df) else None)
print("columns:", list(df.columns))
'''


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": cell_source_lines(text)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cell_source_lines(text),
    }


def cell_source_lines(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    return [text]


INTRO_MD = """# ML-only comment features — **standalone notebook**

**No Git clone.** No subprocess to repo scripts. This file embeds the project YAML and CUDA-capable inference helpers copied from `src/comment_feature_models.py`.

**You provide:** cleaned monthly Parquet on Drive (`cleaned_monthly_chunks/<subreddit>/<YYYY-MM>.parquet`).

**Notebook writes:** VM staging under `WORKDIR` then checkpoints `comment_features_ml/` back to Drive.

**On your laptop:** copy `comment_features_ml` into `data/interim/political_forums/` and run repo script:

`scripts/merge_ml_shards_into_comment_features.py --config config/political_forums_setup.yaml`

to merge Colab ML shards with locally computed lexical fields into final `comment_features/` for downstream metrics.

Runtime: enable **GPU** when `DEVICE = \"cuda\"`."""

PIP = "!pip install -q pandas pyarrow transformers torch sentencepiece pyyaml\n"

PREFLIGHT = '''import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
'''

MERGE_NOTE = """## After Colab finishes (laptop)

1. Sync `comment_features_ml` from Drive → `data/interim/political_forums/comment_features_ml/` in your repo checkout.
2. Run: `.venv/bin/python scripts/merge_ml_shards_into_comment_features.py --config config/political_forums_setup.yaml`
3. Proceed with `prepare_event_time_metrics.py --prefer_comment_features`."""


def main() -> None:
    cells: list[dict] = [
        markdown_cell(INTRO_MD),
        markdown_cell("## 1) Control panel — edit embedded YAML and Drive paths"),
        code_cell(CONTROL_CELL),
        markdown_cell("## 2) Dependencies"),
        code_cell(PIP),
        markdown_cell("## 3) Preflight accelerator"),
        code_cell(PREFLIGHT),
        markdown_cell("## 4) Inference helpers + job runner (inlined project code — run once per session after Control)"),
        code_cell(RUNTIME_CELL.strip() + "\n"),
        markdown_cell("## 5) Mount Google Drive"),
        code_cell('from google.colab import drive\n\ndrive.mount("/content/drive")\n'),
        markdown_cell("## 6) Create VM paths — sync `cleaned_monthly_chunks` from Drive"),
        code_cell(PATHS_AND_SYNC_IN.strip() + "\n"),
        markdown_cell("## 7) Run ML extraction"),
        code_cell(RUN_SELECTION.strip() + "\n"),
        markdown_cell("## 8) Upload `comment_features_ml` parquet tree to Drive"),
        code_cell(CHECKPOINT.strip() + "\n"),
        markdown_cell(MERGE_NOTE),
        markdown_cell("## 9) Verification"),
        code_cell(VERIFY.strip() + "\n"),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }

    out = ROOT / "notebooks" / "colab_compute_comment_features_gpu.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
