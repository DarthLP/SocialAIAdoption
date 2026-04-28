# Scripts Pipeline Guide

## Purpose
This guide explains the end-to-end script pipeline in execution order, including:
- what each script does and why it exists,
- which data layer it reads from (`raw`, `interim`, `processed`, `results`),
- what it writes,
- and exactly how to run it from the project root.

All commands below use the project-local virtual environment:
- `.venv/bin/python ...`

Configuration default:
- `--config config/political_forums_setup.yaml`

---

## Recommended Execution Order

### 0) Preconditions (once per machine/project setup)
- Ensure `.venv` exists and dependencies are installed.
- Ensure required monthly Reddit dump files (`RC_YYYY-MM.zst`) are available in your source directory.

### 1) Filter raw monthly dumps into per-day subreddit chunks (required)
- Script: `filter_dump_comments.py`
- Why: Converts very large monthly dumps into analysis-ready day chunks for selected forums and event window.
- Input layer: External raw dump files (`RC_YYYY-MM.zst`) + config.
- Output layer:
  - `data/raw/political_forums/daily_chunks/<subreddit>/<YYYY-MM-DD>.ndjson`
  - `results/tables/filtering/dump_filter_counts_by_day.csv`
  - `results/tables/filtering/dump_filter_counts_by_subreddit.csv`
  - `results/logs/filter_dump/*`
- Run:
  - `.venv/bin/python scripts/filter_dump_comments.py --config config/political_forums_setup.yaml`

### 2) Remove duplicate IDs after interrupted/restarted filtering (optional but recommended after restarts)
- Script: `dedupe_daily_chunks.py`
- Why: Cleans duplicate rows introduced by resumed/interrupted dump filtering.
- Input layer: `data/raw/political_forums/daily_chunks/`
- Output layer:
  - Updated deduped daily chunk files (with `--apply`)
  - `results/tables/filtering/dedupe_daily_chunks_report.csv`
- Run:
  - Dry run: `.venv/bin/python scripts/dedupe_daily_chunks.py --config config/political_forums_setup.yaml`
  - Apply: `.venv/bin/python scripts/dedupe_daily_chunks.py --config config/political_forums_setup.yaml --apply`

### 3) Inspect pre-cleaning quality trends (recommended diagnostics)
- Script: `plot_data_quality_trends.py`
- Why: Shows day-level quality indicators (e.g., removed/deleted/AutoModerator/stickied rates) before cleaning decisions.
- Input layer:
  - `data/raw/political_forums/daily_chunks/`
  - `results/tables/filtering/dump_filter_counts_by_day.csv` (validation baseline)
- Output layer:
  - `results/tables/data_quality_trends/*`
  - `results/figures/data_quality_trends/*`
- Run:
  - `.venv/bin/python scripts/plot_data_quality_trends.py --config config/political_forums_setup.yaml`

### 4) Apply deterministic cleaning policy (required for downstream metrics)
- Script: `clean_daily_chunks.py`
- Why: Produces interim cleaned corpus by dropping moderation/deletion placeholders and adding analysis flags.
- Input layer: `data/raw/political_forums/daily_chunks/`
- Output layer:
  - `data/interim/political_forums/cleaned_monthly_chunks/<subreddit>/<YYYY-MM>.parquet`
  - `results/tables/cleaning/clean_daily_chunks_audit_by_day.csv`
  - `results/tables/cleaning/clean_daily_chunks_audit_by_subreddit.csv`
  - `results/tables/cleaning/clean_daily_chunks_schema_coercion_by_month.csv`
  - `results/tables/cleaning/clean_daily_chunks_schema_coercion_field_issues.csv`
  - `results/tables/cleaning/clean_daily_chunks_schema_invalid_row_samples.csv`
  - `results/tables/cleaning/clean_daily_chunks_notes.txt`
- Run:
  - `.venv/bin/python scripts/clean_daily_chunks.py --config config/political_forums_setup.yaml`

### 5) Build event-time metric tables (required for event-time plotting)
- Script: `prepare_event_time_metrics.py`
- Why: Aggregates cleaned daily comments into subreddit-level and pooled event-time metrics.
- Input layer: `data/interim/political_forums/cleaned_monthly_chunks/`
- Output layer:
  - `results/tables/event_time/event_time_daily_metrics_by_subreddit.csv`
  - `results/tables/event_time/event_time_daily_metrics_pooled.csv`
  - `results/tables/event_time/ai_word_rates_daily_long.csv`
  - `results/tables/event_time/event_time_metrics_notes.txt`
  - `results/tables/event_time_daily_metrics.csv` (compatibility export)
- Run:
  - `.venv/bin/python scripts/prepare_event_time_metrics.py --config config/political_forums_setup.yaml`

### 6) Create event-time figures (required for visual analysis)
- Script: `plot_event_time_metrics.py`
- Why: Generates pooled and per-subreddit trend figures across linguistic, AI-style, and toxicity proxies.
- Input layer: `results/tables/event_time/*` (or compatibility file when needed)
- Output layer:
  - `results/figures/event_time/*`
  - `results/figures/event_time/by_subreddit/*`
- Run:
  - `.venv/bin/python scripts/plot_event_time_metrics.py --config config/political_forums_setup.yaml`

### 7) Optional sampled detector robustness check
- Script: `run_llm_detector_sample.py`
- Why: Adds an optional robustness layer using sampled detector-like scoring.
- Input layer: `data/interim/political_forums/cleaned_monthly_chunks/`
- Output layer:
  - `results/tables/event_time/llm_detector_sample_scores_daily.csv`
- Run:
  - Heuristic only: `.venv/bin/python scripts/run_llm_detector_sample.py --config config/political_forums_setup.yaml`
  - Optional HF model: `.venv/bin/python scripts/run_llm_detector_sample.py --config config/political_forums_setup.yaml --use_hf_model`

### 8) Optional user-overlap diagnostics (after cleaning)
- Script: `user_overlap_across_forums.py`
- Why: Measures cross-forum author overlap across the configured forum set.
- Input layer: `data/interim/political_forums/cleaned_monthly_chunks/`
- Output layer: `results/tables/user_overlap/user_overlap_*.csv`
- Run:
  - `.venv/bin/python scripts/user_overlap_across_forums.py --config config/political_forums_setup.yaml`

### 9) Optional same-day cross-forum activity diagnostics (after cleaning)
- Script: `user_same_day_cross_forum.py`
- Why: Focuses on users active in multiple forums on the same UTC day.
- Input layer: `data/interim/political_forums/cleaned_monthly_chunks/`
- Output layer: `results/tables/user_overlap/user_same_day_cross_forum_*.csv`
- Run:
  - `.venv/bin/python scripts/user_same_day_cross_forum.py --config config/political_forums_setup.yaml`

---

## Short Pipeline Map by Data Layer
- External dump files (`RC_*.zst`) -> `filter_dump_comments.py` -> `data/raw/.../daily_chunks/`
- `data/raw/.../daily_chunks/` -> `dedupe_daily_chunks.py` (optional) -> deduped raw chunks
- `data/raw/.../daily_chunks/` -> `plot_data_quality_trends.py` -> quality tables/figures in `results/`
- `data/raw/.../daily_chunks/` -> `clean_daily_chunks.py` -> `data/interim/.../cleaned_monthly_chunks/`
- `data/interim/.../cleaned_monthly_chunks/` -> `prepare_event_time_metrics.py` -> `results/tables/event_time/`
- `results/tables/event_time/` -> `plot_event_time_metrics.py` -> `results/figures/event_time/`
- `data/interim/.../cleaned_monthly_chunks/` -> overlap and sampled-detector scripts (optional) -> `results/tables/*`

---

## Typical Minimal Run (Core Analysis)
1. `filter_dump_comments.py`
2. `dedupe_daily_chunks.py --apply` (when restart overlap risk exists)
3. `plot_data_quality_trends.py`
4. `clean_daily_chunks.py`
5. `prepare_event_time_metrics.py`
6. `plot_event_time_metrics.py`

Optional additions after step 4:
- `run_llm_detector_sample.py`
- `user_overlap_across_forums.py`
- `user_same_day_cross_forum.py`
