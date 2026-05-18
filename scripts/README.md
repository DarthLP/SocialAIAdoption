# Scripts Pipeline Guide

## Purpose
This guide explains the end-to-end script pipeline in execution order, including:
- what each script does and why it exists,
- which data layer it reads from (`raw`, `interim`, `processed`, `results`),
- what it writes,
- and exactly how to run it from the project root.

All commands below use the project-local virtual environment:
- `.venv/bin/python ...`

Configuration default (active study):
- `--config config/italy_polarization_setup.yaml`

Archived AI-adoption config: `config/archive/ai_adoption_political_forums_setup.yaml`

---

## Italy polarization — extraction order (active)

0. **Required backup** — `rsync -a data/raw/ $ARCHIVE/data/raw/` and `rsync -a results/ $ARCHIVE/results/` (see README).
1. **Discovery** — `scripts/discovery/profile_subreddits_in_dump.py` (first 3 UTC days of `RC_2023-03` only).
2. **Apply** — `scripts/discovery/apply_discovery_to_config.py` after reviewing `extraction_size_preview.csv`.
3. **Filter** — `scripts/filtering/filter_dump_comments.py` with `italy_polarization_state.json` / log paths.

---

## Directory layout

Runnable entrypoints live under **`scripts/<domain>/<script>.py`** (exactly one level below `scripts/`). Add new scripts to the folder that matches their primary role:

| Domain | Role |
|--------|------|
| [`scripts/discovery/`](discovery/) | 3-day dump profiling; apply Italian subs to config |
| [`scripts/filtering/`](filtering/) | Monthly dump → per-day NDJSON chunks |
| [`scripts/cleaning/`](cleaning/) | Dedupe raw chunks; build cleaned monthly Parquet |
| [`scripts/diagnostics/`](diagnostics/) | Pre-clean QC plots; cross-forum overlap; optional sampled detector |
| [`scripts/features/`](features/) | Per-comment feature shards; Colab merge; daily repetition CSV |
| [`scripts/event_time/`](event_time/) | Event-time metric tables and figures |
| [`scripts/user_week/`](user_week/) | User-week panel, pre/post shift analysis, figures |
| [`scripts/devtools/`](devtools/) | Maintainer tools (e.g. regenerate standalone Colab notebook) |

Shared helper (imported by domain scripts, not run as a step): [`scripts/_project_root.py`](_project_root.py) resolves the repository root after this layout change.

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
  - `.venv/bin/python scripts/filtering/filter_dump_comments.py --config config/political_forums_setup.yaml`

### 2) Remove duplicate IDs after interrupted/restarted filtering (optional but recommended after restarts)
- Script: `dedupe_daily_chunks.py`
- Why: Cleans duplicate rows introduced by resumed/interrupted dump filtering.
- Input layer: `data/raw/political_forums/daily_chunks/`
- Output layer:
  - Updated deduped daily chunk files (with `--apply`)
  - `results/tables/filtering/dedupe_daily_chunks_report.csv`
- Run:
  - Dry run: `.venv/bin/python scripts/cleaning/dedupe_daily_chunks.py --config config/political_forums_setup.yaml`
  - Apply: `.venv/bin/python scripts/cleaning/dedupe_daily_chunks.py --config config/political_forums_setup.yaml --apply`

### 3) Inspect pre-cleaning quality trends (recommended diagnostics)
- Script: `plot_data_quality_trends.py`
- Why: Shows day-level quality indicators (e.g., removed/deleted/AutoModerator/stickied rates) before cleaning decisions, plus **author-share** rates (distinct authors with a signal ÷ distinct non-empty authors that day; pooled ALL/family/topic rows union authors across subreddits).
- Input layer:
  - `data/raw/political_forums/daily_chunks/`
  - `results/tables/filtering/dump_filter_counts_by_day.csv` (validation baseline)
- Output layer:
  - `results/tables/data_quality_trends/*` (including `daily_quality_metrics_by_topic_and_family.csv`)
  - `results/figures/data_quality_trends/*`
- Notes:
  - Enforces `event_window.start_utc` to `event_window.end_utc_exclusive` during plotting-table generation.
  - Draws vertical red dotted markers from optional YAML `plot_reference_dates_utc` (list of ISO UTC strings); when omitted, defaults to `2022-11-30` (ChatGPT) and `2023-03-14` (GPT-4).
  - Uses month-start date ticks (`YYYY-MM-01`) for consistent calendar alignment across plots.
  - Family outputs now include:
    - `by_family_<metric>.png` (aggregate family lines),
    - `by_subreddit_by_family/<family>/<metric>.png` (one page per family and metric with one subplot per topic and subreddit lines inside each topic panel),
    - `by_topic_by_family/by_topic_by_family_<metric>.png` (one page per metric with one subplot per family and topic lines inside each family panel).
  - Uses explicit high-contrast palettes for multi-line subreddit overlays to keep lines visually distinguishable.
  - Prints `plot_progress` lines per metric so long runs show forward progress.
  - Uses non-interactive Matplotlib backend by default for terminal-safe figure rendering.
- Run:
  - `.venv/bin/python scripts/diagnostics/plot_data_quality_trends.py --config config/political_forums_setup.yaml`
  - Italy ban exploratory corpus: `--config config/italy_chatgpt_ban_setup.yaml` (outputs under `results/tables/italy_chatgpt_ban/` and `results/figures/italy_chatgpt_ban/` per that YAML’s `paths`).

### 3b) Colab ML-export zip — pooled primary-detector trend (optional)
- Script: `describe_ml_zip_time_trends.py`
- Why: Quick sanity check that `detector_primary_ai_prob` from a Drive-style `production_run/...` Parquet zip looks sensible over calendar time before merging into `comment_features/`.
- Input layer: A zip archive (default `data/interim/production_run-20260511T145305Z-3-001.zip`) with `*.parquet` members under `--internal-prefix` (default `production_run/`).
- Output layer:
  - `results/tables/ml_zip_time_trends/pooled_daily_primary_ai_prob.csv` (mean, median, tail counts/shares, volume-weighted rolling columns, `event_time_t_days`)
  - `results/tables/ml_zip_time_trends/pooled_monthly_primary_ai_prob.csv`
  - `results/tables/ml_zip_time_trends/launch_window_summary.csv` (pre/post windows vs `launch_day_utc`)
  - `results/tables/ml_zip_time_trends/ml_zip_time_trends_notes.txt` (interpretation caveats)
  - `results/figures/ml_zip_time_trends/pooled_daily_primary_ai_prob_mean_median.png` (launch + GPT-4 markers; rolling tail overlay)
- Run:
  - `.venv/bin/python scripts/diagnostics/describe_ml_zip_time_trends.py`
  - Custom paths: add `--zip-path`, `--internal-prefix`, `--tables-dir`, and `--figures-dir` as needed.
  - Custom cutoffs: `--thresholds 0.5,0.95`. Omit GPT-4 vertical: `--no-gpt4-marker`.

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
  - `.venv/bin/python scripts/cleaning/clean_daily_chunks.py --config config/political_forums_setup.yaml`

### 5) Compute reusable per-comment feature shards (recommended before event-time aggregation)
- **Monolithic (local / one pass):** `compute_comment_features.py`
  - Why: Computes lexical/style/toxicity proxies plus HF-based detector, hostility, emotion, perplexity in one parquet tree; passes through `author` and `created_utc` from cleaned shards when present. Extra lexical columns (also in merge path): `em_dash_count`, `en_dash_count`, `ascii_double_hyphen_count`, `colon_count`, `open_paren_count`, `curly_quote_count`, `markdown_bold_pair_count`, `markdown_heading_line_count`, `hedging_phrase_hits`, `polite_closer_hits`, `signposting_phrase_hits`, `avg_words_per_sentence_comment` (NaN when no words).
  - Input: `data/interim/political_forums/cleaned_monthly_chunks/`
  - Output: `data/interim/political_forums/comment_features/<subreddit>/<YYYY-MM>.parquet`
  - Run: `.venv/bin/python scripts/features/compute_comment_features.py --config config/political_forums_setup.yaml`
  - Italy ban corpus (same script; outputs under `data/interim/italy_chatgpt_ban/comment_features/`): `--config config/italy_chatgpt_ban_setup.yaml`
  - Device: `--device auto|mps|cpu`, plus `--batch_size`, `--workers`, bounded flags, `--profile`, `--overwrite`, filters.
- **Split Colab GPU + laptop finalize (merge + lexical):**
  - `merge_ml_shards_into_comment_features.py`: reads `cleaned_monthly_chunks/`, merges optional **`comment_features_ml/`** shards by `id` (e.g. from Colab), computes lexical/rule fields using the same logic as `compute_comment_features.py` (shared via import), writes **`comment_features/`** for downstream steps.
  - Colab: [`notebooks/colab_compute_comment_features_gpu.ipynb`](../notebooks/colab_compute_comment_features_gpu.ipynb) runs **standalone** (embedded YAML + inlined inference helpers; **no** clone, **no** call to repo scripts). Drive sync pulls `cleaned_monthly_chunks` in and checkpoints `comment_features_ml` back; GPU runtime + `DEVICE="cuda"` for CUDA.
  - **Maintainers:** regenerate that notebook after edits to `config/political_forums_setup.yaml` or `src/comment_feature_models.py`: `.venv/bin/python scripts/devtools/_gen_colab_standalone_nb.py`.
  - Laptop: copy Drive `comment_features_ml/` into interim, then `.venv/bin/python scripts/features/merge_ml_shards_into_comment_features.py --config config/political_forums_setup.yaml` (same bounded/filter flags).

### 5b) Daily repetition / template similarity (optional; merged into event-time tables)
- Script: `compute_daily_repetition_similarity.py`
- Why: Computes `repetition_template_similarity` from cleaned monthly Parquet (within-day max-Jaccard-to-recent stream, ordered by `created_utc`).
- Input: `data/interim/political_forums/cleaned_monthly_chunks/`
- Output: `results/tables/event_time/repetition_daily_by_subreddit.csv` (`subreddit`, `date_utc`, `repetition_template_similarity`, `n_comments`)
- Run: `.venv/bin/python scripts/features/compute_daily_repetition_similarity.py --config config/political_forums_setup.yaml`
- Flags: `--similarity_window` (default 20), `--min_words_for_similarity` (default 0), same bounded filters as other monthly scripts (`--max_month_files_per_subreddit`, `--max_total_month_files`, `--max_days_per_month`, `--subreddits`, `--months`).

### 6) Build event-time metric tables (required for event-time plotting)
- Script: `prepare_event_time_metrics.py`
- Why: Aggregates subreddit-level and pooled event-time metrics **only** from `comment_features/` shards; left-merges `repetition_daily_by_subreddit.csv` when present (otherwise repetition is NaN). Emits matching `*_rate_100w` columns for the new lexical counts and `avg_words_per_sentence_mean` (see `event_time_metrics_notes.txt`).
- Input layer: `data/interim/political_forums/comment_features/` (required); optional `results/tables/event_time/repetition_daily_by_subreddit.csv`
- Output layer:
  - `results/tables/event_time/event_time_daily_metrics_by_subreddit.csv`
  - `results/tables/event_time/event_time_daily_metrics_pooled.csv`
  - `results/tables/event_time/ai_word_rates_daily_long.csv`
  - `results/tables/event_time/comment_feature_validation_associations.csv` (when comment features are used)
  - `results/tables/event_time/event_time_metrics_notes.txt`
  - `results/tables/event_time_daily_metrics.csv` (compatibility export)
- Run:
  - `.venv/bin/python scripts/event_time/prepare_event_time_metrics.py --config config/political_forums_setup.yaml`
  - Italy: same command with `--config config/italy_chatgpt_ban_setup.yaml` (writes under `results/tables/italy_chatgpt_ban/event_time/`).
- Performance/bounded-benchmark options:
  - Sample one month file per subreddit with phase timing:
    - `.venv/bin/python scripts/event_time/prepare_event_time_metrics.py --config config/political_forums_setup.yaml --max_month_files_per_subreddit 1 --profile`
  - Hard cap total processed month files and days per month:
    - `.venv/bin/python scripts/event_time/prepare_event_time_metrics.py --config config/political_forums_setup.yaml --max_total_month_files 2 --max_days_per_month 10 --profile_output results/tables/event_time/prepare_event_time_metrics_profile.json`

### 7) Create event-time figures (required for visual analysis)
- Script: `plot_event_time_metrics.py`
- Why: Generates pooled and per-family (default) trend figures across linguistic, AI-style, and toxicity proxies. Per-subreddit-by-family grids are opt-in.
- Input layer: `results/tables/event_time/*` (or compatibility file when needed)
- Output layer:
  - `results/figures/event_time/pooled/{daily,rolling_daily}/*` by default (`weekly/*` only with `--include_weekly`)
  - `results/figures/event_time/by_family/{daily,rolling_daily}/*` by default (disable with `--no_topic_views`; add weekly with `--include_weekly`)
  - `results/figures/event_time/by_subreddit_by_family/{daily,rolling_daily}/<family>/*` by default (disable with `--no_by_subreddit`; add weekly with `--include_weekly`)
- Notes:
  - Plots use calendar-date x-axes with month-start ticks (`YYYY-MM-01`), not relative day offsets.
  - Red dotted vertical markers come from optional YAML `plot_reference_dates_utc` (parsed by `src.config_utils.plot_reference_dates_calendar_utc`); when omitted, defaults to `2022-11-30` and `2023-03-14`. `plot_event_time_metrics.main` registers these before plotting; `plot_event_time_stratified_metrics` does the same so imported helpers stay aligned.
  - `rolling_daily` uses a 7-day trailing (past-only) window by default; pandas `.rolling(window="ND")` with default `center=False` includes only the current day plus the prior `N-1` days.
  - Pooled quote signal is rendered as one combined dual-axis figure (`event_time_quote_rates_and_curly_share.png`) with curly + all-quote rates on the left axis and curly share on the right axis.
  - Coverage metrics (`coverage_perplexity`, `coverage_detector_primary`, …) are skipped when the series is all-NaN or all-zero.
  - Strict colon counts mirror extended colon counts: both are computed on text after URL spans and clock-time tokens are stripped, so `colon_extended_rate_100w >= colon_rate_100w` always holds.
- Run:
  - Default (pooled + by-family + by-subreddit-by-family, daily + rolling_daily with rolling 7 days): `.venv/bin/python scripts/event_time/plot_event_time_metrics.py --config config/political_forums_setup.yaml`
  - Italy: same with `--config config/italy_chatgpt_ban_setup.yaml` (figures under `results/figures/italy_chatgpt_ban/event_time/`).
  - Add weekly extras: `.venv/bin/python scripts/event_time/plot_event_time_metrics.py --config config/political_forums_setup.yaml --include_weekly`
  - Disable per-family outputs: `.venv/bin/python scripts/event_time/plot_event_time_metrics.py --config config/political_forums_setup.yaml --no_topic_views`
  - Disable per-subreddit-by-family grids: `.venv/bin/python scripts/event_time/plot_event_time_metrics.py --config config/political_forums_setup.yaml --no_by_subreddit`
  - Family map used in family-level figures is loaded from `config/political_forums_setup.yaml` (`topics` + `topic_families` sections).
  - Pooled, per-family, and optional per-subreddit-by-family figures are split into `daily/` and `rolling_daily/` by default; add `--include_weekly` to also render `weekly/`.

### 7b) Stratified pooled event-time tables and figures (optional)
- Scripts: `prepare_event_time_stratified_metrics.py`, `plot_event_time_stratified_metrics.py`
- Why: Pooled daily metrics and plots split by **user series** (`old`, `new`, `debut_observed` where debut is first observed comment in subreddit regardless of cohort) and by **`length_bucket`** (`short` / `medium` / `long` from comment features). Omits repetition/Jaccard (`repetition_template_similarity`). Stratified CSVs carry the same extended lexical `*_rate_100w` columns as `prepare_event_time_metrics.py`. Requires `author` and `created_utc` on comment shards (use `merge_ml_shards_into_comment_features.py` when ML path was used).
- Input: `data/interim/political_forums/comment_features/`
- Output tables (`results/tables/event_time/`):
  - `event_time_daily_metrics_pooled_by_user_cohort.csv`
  - `event_time_daily_metrics_pooled_by_length_bucket.csv`
  - `event_time_length_bucket_daily_shares_pooled.csv`
  - `event_time_stratified_metrics_notes.txt`
- Output figures: `results/figures/event_time/stratified_pooled/user_series/{daily,rolling_daily}/` and `stratified_pooled/length_bucket/{daily,rolling_daily}/` by default (add `--include_weekly` to plotting for `weekly/`; length-bucket figures exclude ML/coverage series that do not make sense when stratifying by length).
- Run:
  - `.venv/bin/python scripts/event_time/prepare_event_time_stratified_metrics.py --config config/political_forums_setup.yaml`
  - `.venv/bin/python scripts/event_time/plot_event_time_stratified_metrics.py --config config/political_forums_setup.yaml`
  - Add weekly extras for stratified plots: `.venv/bin/python scripts/event_time/plot_event_time_stratified_metrics.py --config config/political_forums_setup.yaml --include_weekly`
- Bounded sampling (same pattern as `prepare_event_time_metrics.py`): `--max_month_files_per_subreddit`, `--max_total_month_files`, `--max_days_per_month`, `--profile`, `--profile_output`

### 7c) Within-user pre/post style shift (author × ISO-week analysis layer)
- Scripts: `prepare_user_week_style_panel.py` -> `analyze_user_pre_post_shift.py` -> `plot_user_pre_post_shift.py`
- Why: Build a per-author per-ISO-week style panel from `comment_features/`, then compare each user's post-launch writing to their own pre-launch baseline. The panel aggregates the same extended lexical hit counts into weekly `*_rate_100w` columns plus `avg_words_per_sentence_comment_mean` (from per-comment NaN-skipped means). Two parallel comparisons are produced for every user and feature: a **weekly view** (word-weighted weekly mean / SD; std_delta with winsorized SD floor) and a **pooled-comments view** (precision-aware standard errors built from raw hit counts and sumsq fields stored in the panel). A composite `ai_likeness_user_week` mirrors the existing event-time AI-likeness index (z-scales frozen on the pre-launch user-week pool).
- Inputs:
  - `data/interim/political_forums/comment_features/<subreddit>/<YYYY-MM>.parquet`
  - `config/political_forums_setup.yaml` (`event_window.launch_day_utc`, `subreddits.primary`)
- Outputs:
  - `data/interim/political_forums/user_week_style_panel/<YYYY-MM>.parquet` (per-month panel shards)
  - `results/tables/user_week/user_week_panel.parquet` (merged panel)
  - `results/tables/user_week/user_week_panel_notes.txt`
  - `results/tables/user_week/shift_per_user_<cohort>.csv` (one row per user; weekly + pooled columns)
  - `results/tables/user_week/shift_summary_<cohort>.csv` (one row per cohort + topic-stable + per-topic + audit categories + placebo)
  - `results/tables/user_week/shift_audit_per_user_<cohort>.csv` (panel / pre_only / post_only / below_thresholds)
  - `results/tables/user_week/composite_zscale_pre_<cohort>.json` (frozen z-scales for reproducibility)
  - `results/tables/user_week/shift_methods_note.txt`
  - `results/figures/user_week/<cohort>/{dist_std_delta_composite,dist_t_user_pooled_composite,weekly_vs_pooled_scatter,components_grid,spaghetti_sample,mirror_top_movers}.png`
  - `results/logs/user_week/analyze_user_pre_post_shift.log`
- Cohort defaults:
  - **strict**: `--min_words_per_week 100`, `--min_pre_weeks 4`, `--min_post_weeks 4`, `--min_total_words_pre 400`, `--min_total_words_post 400`.
  - **loose**: `--min_words_per_week 30`, `--min_pre_weeks 2`, `--min_post_weeks 2`, `--min_total_words_pre 100`, `--min_total_words_post 100`.
  - Hard pre-launch + post-launch requirement: a user enters the comparison only with both pre and post coverage above thresholds; pre-only / post-only / below-thresholds users are surfaced in the side audit, not silently dropped.
- Author hygiene matches `scripts/diagnostics/plot_data_quality_trends.py`: empty author, `[deleted]`, `AutoModerator`, and the `bot`-substring heuristic are dropped during panel construction.
- Run:
  - `.venv/bin/python scripts/user_week/prepare_user_week_style_panel.py --config config/political_forums_setup.yaml`
  - `.venv/bin/python scripts/user_week/analyze_user_pre_post_shift.py --config config/political_forums_setup.yaml`
  - `.venv/bin/python scripts/user_week/plot_user_pre_post_shift.py --config config/political_forums_setup.yaml`
- Bounded benchmarks (panel script): `--max_total_month_files`, `--max_month_files_per_subreddit`, `--max_days_per_month`, `--profile`, `--profile_output`.
- Sensitivity layers in `analyze_user_pre_post_shift.py`: topic-stable sub-cohort row (`panel_topic_stable`), per-topic stratified rows (`panel_topic=<topic from config>`), and a placebo run with the launch shifted back by `--placebo_offset_weeks` (default 8).

### 8) Optional sampled detector robustness check
- Script: `run_llm_detector_sample.py`
- Why: Adds an optional robustness layer using sampled detector-like scoring.
- Input layer: `data/interim/political_forums/cleaned_monthly_chunks/`
- Output layer:
  - `results/tables/event_time/llm_detector_sample_scores_daily.csv`
- Run:
  - Heuristic only: `.venv/bin/python scripts/diagnostics/run_llm_detector_sample.py --config config/political_forums_setup.yaml`
  - Optional HF model: `.venv/bin/python scripts/diagnostics/run_llm_detector_sample.py --config config/political_forums_setup.yaml --use_hf_model`

### 9) Optional user-overlap diagnostics (after cleaning)
- Script: `user_overlap_across_forums.py`
- Why: Measures cross-forum author overlap across the configured forum set.
- Input layer: `data/interim/political_forums/cleaned_monthly_chunks/`
- Output layer: `results/tables/user_overlap/user_overlap_*.csv`
- Run:
  - `.venv/bin/python scripts/diagnostics/user_overlap_across_forums.py --config config/political_forums_setup.yaml`

### 10) Optional same-day cross-forum activity diagnostics (after cleaning)
- Script: `user_same_day_cross_forum.py`
- Why: Focuses on users active in multiple forums on the same UTC day.
- Input layer: `data/interim/political_forums/cleaned_monthly_chunks/`
- Output layer: `results/tables/user_overlap/user_same_day_cross_forum_*.csv`
- Run:
  - `.venv/bin/python scripts/diagnostics/user_same_day_cross_forum.py --config config/political_forums_setup.yaml`

---

## Short Pipeline Map by Data Layer
- External dump files (`RC_*.zst`) -> `filter_dump_comments.py` -> `data/raw/.../daily_chunks/`
- `data/raw/.../daily_chunks/` -> `dedupe_daily_chunks.py` (optional) -> deduped raw chunks
- `data/raw/.../daily_chunks/` -> `plot_data_quality_trends.py` -> quality tables/figures in `results/`
- `data/raw/.../daily_chunks/` -> `clean_daily_chunks.py` -> `data/interim/.../cleaned_monthly_chunks/`
- `data/interim/.../cleaned_monthly_chunks/` -> `compute_comment_features.py` **or** (`colab_compute_comment_features_gpu.ipynb` → `comment_features_ml/` then `merge_ml_shards_into_comment_features.py`) -> `data/interim/.../comment_features/` (optional intermediate `comment_features_ml/` when using the split path)
- `data/interim/.../cleaned_monthly_chunks/` -> `compute_daily_repetition_similarity.py` (optional) -> `results/tables/event_time/repetition_daily_by_subreddit.csv`
- `data/interim/.../comment_features/` (+ optional repetition CSV) -> `prepare_event_time_metrics.py` -> `results/tables/event_time/`
- `results/tables/event_time/` -> `plot_event_time_metrics.py` -> `results/figures/event_time/`
- `data/interim/.../comment_features/` -> `prepare_event_time_stratified_metrics.py` -> `results/tables/event_time/` (stratified CSVs) -> `plot_event_time_stratified_metrics.py` -> `results/figures/event_time/stratified_pooled/user_series/` and `.../stratified_pooled/length_bucket/`
- `data/interim/.../comment_features/` -> `prepare_user_week_style_panel.py` -> `data/interim/.../user_week_style_panel/` and `results/tables/user_week/user_week_panel.parquet` -> `analyze_user_pre_post_shift.py` -> `results/tables/user_week/` (shift CSVs + JSON scales + methods note) -> `plot_user_pre_post_shift.py` -> `results/figures/user_week/<cohort>/`
- `data/interim/.../cleaned_monthly_chunks/` -> overlap and sampled-detector scripts (optional) -> `results/tables/*`
- Colab ML-export zip (optional) -> `describe_ml_zip_time_trends.py` -> `results/tables/ml_zip_time_trends/` and `results/figures/ml_zip_time_trends/`

---

## Typical Minimal Run (Core Analysis)
1. `filter_dump_comments.py`
2. `dedupe_daily_chunks.py --apply` (when restart overlap risk exists)
3. `plot_data_quality_trends.py`
4. `clean_daily_chunks.py`
5. `compute_comment_features.py`
6. `compute_daily_repetition_similarity.py` (optional, for repetition column in tables)
7. `prepare_event_time_metrics.py`
8. `plot_event_time_metrics.py`
9. (Optional) `prepare_event_time_stratified_metrics.py` then `plot_event_time_stratified_metrics.py`
10. (Optional, within-user analysis) `prepare_user_week_style_panel.py` -> `analyze_user_pre_post_shift.py` -> `plot_user_pre_post_shift.py`

Optional additions after step 4:
- `run_llm_detector_sample.py`
- `describe_ml_zip_time_trends.py` (Colab ML Parquet zip export; no YAML)
- `user_overlap_across_forums.py`
- `user_same_day_cross_forum.py`
