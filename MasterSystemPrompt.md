# Master System Prompt

## Script Summary
This document defines stable technical context for the thesis workspace, including architecture, reproducibility standards, and project execution policy. Keep it concise and update it when workflow conventions or core technical decisions change.

## Project Objective
Build a reproducible pipeline to study AI-writing adoption in Reddit communities around ChatGPT launch using monthly dump ingestion plus local filtering, with forum scope and topic grouping centrally configured in `config/political_forums_setup.yaml` (`subreddits.primary` + `topics`) so corpus updates are config-only.

## Scope Boundaries
- This file stores stable context and execution conventions.
- This file does not store transient daily task chatter.
- Daily operational state belongs in `TODO.md`.

## Technical Architecture Overview
- Code layer: `src/`, `scripts/` (domain subfolders under `scripts/<domain>/`; see `scripts/README.md`), `config/`
- Data layer:
  - External raw dumps on mounted storage (`/Volumes/Expansion/Masterthesis/RawData/...`); additional months are acquired with `aria2c` selective `--select-file=` on the Academic Torrents bundle (see `README.md` for examples, preflight `pgrep -x aria2c`, and optional `caffeinate` wrapper).
  - Project filtered raw outputs (`data/raw/political_forums/daily_chunks/`)
  - `data/interim/`, `data/processed/` for downstream transformations
  - Interim canonical format: monthly Parquet (`data/interim/political_forums/cleaned_monthly_chunks/<subreddit>/<YYYY-MM>.parquet`)
- Output layer: `results/figures/`, `results/tables/`, `results/logs/`
  - Filtering audits and dedupe reports are grouped in `results/tables/filtering/`
  - Cleaning audits are grouped in `results/tables/cleaning/`
  - Dump filtering logs/state files are grouped in `results/logs/filter_dump/`
  - Overlap analysis outputs are grouped in `results/tables/user_overlap/`
  - Data-quality trend figures/tables are grouped in `results/figures/data_quality_trends/` and `results/tables/data_quality_trends/`
  - Event-time metric tables/figures are grouped in `results/tables/event_time/` and `results/figures/event_time/`
  - Policy: artifacts should always be written to grouped subfolders under `results/*/`
- Dump filtering architecture:
  - Required monthly comment dumps are derived from `event_window.start_utc` / `end_utc_exclusive` via `src.config_utils.comment_dump_filenames` (chronological `RC_YYYY-MM.zst` list).
  - If `start_utc` moves **earlier** within a calendar month already partially processed, delete that month’s filter state JSON under `results/logs/filter_dump/` before re-run so early lines are not skipped by resume.
  - If `subreddits.primary` **gains or loses** subreddits while `event_window` is unchanged, delete per-month `filter_dump_state.RC_*.json` (and merged `filter_dump_state.json` if present) before re-run: a `completed` month is skipped when the stored filter window matches the current run, and the worker does **not** detect subreddit-list drift, so new names would never be scanned from already-completed dumps.
  - Per-file state records `filter_window_start_ts` / `filter_window_end_ts_exclusive`; a completed file is skipped only when those match the current run, so extending `end_utc_exclusive` forward within the same dump month resumes instead of being skipped incorrectly.
  - Configurable worker mode for monthly filtering (`--worker_mode one|two|auto`)
  - Byte-level subreddit prefilter before JSON parsing
  - Optional regex prefilter mode for controlled A/B benchmarking
  - Binary NDJSON append writes to reduce text encoding overhead
  - UTC-day cache for repeated date computations in hot path
  - Per-worker resumable state/log files with merged final audit counters
  - Resume fingerprint guard (path/size/mtime) to prevent unsafe checkpoint reuse if source file changes
  - Optional anchor-based rerun start from persisted `first_in_window_line`
  - Graceful-stop signal handling that checkpoints immediately for duplicate-safe resume
  - Early-stop boundary logic to end monthly scans once configured window data is exhausted
  - Post-run dedupe utility (`scripts/cleaning/dedupe_daily_chunks.py`) for id-based overlap cleanup if needed
  - Cross-forum user overlap utility (`scripts/diagnostics/user_overlap_across_forums.py`) for author-level forum-membership diagnostics
  - Same-day cross-forum activity utility (`scripts/diagnostics/user_same_day_cross_forum.py`) for temporally-aligned author overlap diagnostics
  - Pre-cleaning trend utility (`scripts/diagnostics/plot_data_quality_trends.py`) for daily quality-indicator counts/rates with month-start date ticks, ChatGPT (`2022-11-30`) and GPT-4 (`2023-03-14`) markers, plus family-panel outputs (`by_family_*`, `by_subreddit_by_family/<family>/*`, `by_topic_by_family/by_topic_by_family_*`)
  - Interim cleaning utility (`scripts/cleaning/clean_daily_chunks.py`) for deterministic drop-rules plus retained-row metadata flags, canonical schema enforcement, and coercion diagnostics
- Reusable comment-feature utilities: monolithic `scripts/features/compute_comment_features.py` (lexical + HF on one pass, `--device auto|mps|cpu`; passes through `author` / `created_utc` from cleaned Parquet when present). Lexical/style extensions now include `semicolon_count` (single ASCII variant; the redundant `semicolon_extended_count` was removed), `em_dash` strict/extended pair, and a `colon` strict/extended pair where **both** counts strip URL spans and clock-time tokens before counting (extended adds the fullwidth colon `：` only, guaranteeing extended is a true superset of strict). Plus quote-style counts (`curly`, `straight`, `quote_all`, `quote_curly_share_num/den`), URL/time-expression counts, em/en dash counts, spaced ASCII ` -- ` count, markdown bold-pair and heading-line counts, disjoint hedging / polite-closer / signposting phrase hit totals, and `avg_words_per_sentence_comment`. Optional split: self-contained Colab `notebooks/colab_compute_comment_features_gpu.ipynb` (GPU ML → `comment_features_ml/`) then laptop `scripts/features/merge_ml_shards_into_comment_features.py` (merge ML shards + lexical via shared monolithic helpers → `comment_features/`)
  - Shared HF inference module: `src/comment_feature_models.py` (CUDA/MPS/CPU); used locally by monolithic script and embedded in the Colab notebook for standalone runs. Refresh notebook via `scripts/devtools/_gen_colab_standalone_nb.py` when config or inference code changes.
  - Standalone `scripts/features/compute_daily_repetition_similarity.py` reads `cleaned_monthly_chunks/`, computes daily `repetition_template_similarity` (time-ordered by `created_utc`), writes `results/tables/event_time/repetition_daily_by_subreddit.csv` for merge into event-time tables.
- Event-time metrics utility (`scripts/event_time/prepare_event_time_metrics.py`) aggregates **only** from `comment_features/`, left-merges optional `repetition_daily_by_subreddit.csv`, and writes subreddit/pooled daily CSVs (pooled `ALL` mixes all `subreddits.primary` forums unless stratified pools are added later). Daily tables include `semicolon_rate_100w`, dash and colon strict/extended pairs (colon strict/extended both URL/time-stripped), quote-style density/share (`quote_all_rate_100w`, `curly_quote_rate_100w`, `quote_curly_share` from aggregated numerator/denominator), URL/time rates, and existing lexical/ML metrics. `semicolon_extended_*` is no longer emitted. Bounded controls: `--max_month_files_per_subreddit`, `--max_total_month_files`, `--max_days_per_month`; profiling: `--profile`, `--profile_output`.
- Event-time plotting utility (`scripts/event_time/plot_event_time_metrics.py`) for calendar-date pooled and (default) per-family trend figures with month-start ticks and dual release markers (`2022-11-30`, `2023-03-14`). Pooled outputs include style-proxy panels, the strict-vs-extended AI lexicon overlay, strict-vs-extended overlays for `em_dash` and `colon` (URL/time-stripped on both sides), a single dual-axis quote figure (`event_time_quote_rates_and_curly_share.png` — left axis: curly + all-quote rates; right axis: curly share), the z-score component plot, the strict top-10 stem-aware per-word plus combined trajectory figure, and single-series line plots for each metric (same temporal views under pooled / by_family / by_subreddit_by_family). The pooled URL-vs-time-expression paired overlay and the standalone pooled `quote_curly_share` / `quote_strict_vs_extended` figures were removed in favor of the dual-axis quote panel. CLI defaults: by-family outputs are produced unless `--no_topic_views` is passed; by-subreddit-by-family grids are also on by default and can be disabled with `--no_by_subreddit`; weekly views are optional extras enabled with `--include_weekly` (default outputs are `daily` + `rolling_daily`). Legend convention: family aggregates use in-plot legends; subreddit overlays use below-plot legends for readability. All `rolling_daily` views (pooled, by-family, by-subreddit-by-family, stratified) use pandas time-based `.rolling(window="ND")` with default `center=False`, i.e. trailing past-only windows (no future leakage). Coverage shares (`coverage_perplexity`, `coverage_detector_primary`, `coverage_detector_secondary`, `coverage_hostility`, `coverage_emotion`) are skipped when the series is all-NaN or all-zero.
  - Stratified pooled event-time (`scripts/event_time/prepare_event_time_stratified_metrics.py` → `scripts/event_time/plot_event_time_stratified_metrics.py`): user series `old` / `new` / `debut_observed` (first observed row in subreddit, min created_utc with id tie-break, regardless of cohort), `length_bucket` short/medium/long from shards; AI-likeness z-scores recomputed within each stratum over time; **no** `repetition_template_similarity`; tables under `results/tables/event_time/`; figures split into `results/figures/event_time/stratified_pooled/user_series/{daily,rolling_daily}/` and `.../length_bucket/...` by default (`weekly/` added with `--include_weekly`; length-bucket plots omit detector/perplexity/hostility/emotion/coverage metrics as nonsensical for that stratifier). Stratified `rolling_daily` is also trailing past-only.
  - Within-user pre/post style shift (author × ISO-week layer): `scripts/user_week/prepare_user_week_style_panel.py` → `scripts/user_week/analyze_user_pre_post_shift.py` → `scripts/user_week/plot_user_pre_post_shift.py`. Panel artifacts at `data/interim/political_forums/user_week_style_panel/<YYYY-MM>.parquet` and `results/tables/user_week/user_week_panel.parquet` (`.gitignore` also excludes `user_week_panel.parquet` and `shift_per_user_loose.csv` when they exceed GitHub’s per-file size cap—regenerate locally). Per-user shift tables, frozen-pre composite z-scales, methods note, audit CSVs, and figures under `results/tables/user_week/` and `results/figures/user_week/<cohort>/`. Two parallel comparisons per user: weekly view (word-weighted weekly mean / Kish SD; std_delta with winsor floor; robust MAD; Welch t) and pooled-comments view (Poisson / binomial / sumsq-derived SE on raw fields stored in the panel; composite SE via independence-approx delta method on z-scaled components). Strict and loose cohorts both required pre **and** post coverage above thresholds; pre-only / post-only / below-thresholds users surface in `shift_audit_per_user_<cohort>.csv`. Sensitivity layers in the summary: topic-stable sub-cohort, per-topic strata, placebo (`--placebo_offset_weeks`, default 8). Logs at `results/logs/user_week/`.
  - Optional sampled detector utility (`scripts/diagnostics/run_llm_detector_sample.py`) for CPU-first robustness scoring with deterministic sampling
  - Script execution order and concise script I/O descriptions are centralized in `scripts/README.md`
- Operational rules: `.cursor/rules/project.mdc`
- Durable memory: `Projects/`, `Decisions/`

## Architecture and Debugging Policy
- Keep only a concise architecture summary in this file for now.
- Document major architecture changes here when they affect workflows or reproducibility.
- Record repeated or high-impact debugging lessons in decision notes until a dedicated debugging area is needed again.

## Core Workflow
1. Read rules + targeted memory notes.
2. Ensure required monthly dumps are available on external storage.
3. Filter dumps to project scope (subreddits, date window, required fields).
4. Build analysis outputs reproducibly from filtered data.
5. Update `TODO.md` and only minimal durable notes.

## Code and Note Conventions
- Use repository-local Python environment.
- Keep files and folders consistent with naming conventions.
- Use Obsidian-compatible markdown and wikilinks for knowledge notes.
- For AI lexicon features: entries ending in `*` are matched by Porter stems; non-`*` entries are exact-token matches.

## Quality Gates
A work item is complete only if:
- The implementation is reproducible from scripts/config.
- Impacted documentation is updated where necessary (`README.md`, `TODO.md`, this file).
- Durable notes are updated only when new stable knowledge emerged.
- The vault remains low-noise and non-duplicative.

## Current Status
Recent feature-pipeline contract changes require regeneration of comment_features and event-time tables: `semicolon_extended_count` was removed and `colon_count` is now URL/time-stripped (matching `colon_extended_count`'s cleanup). After updating, rerun `scripts/features/compute_comment_features.py` (or the merge variant for the Colab split) followed by `scripts/event_time/prepare_event_time_metrics.py` to refresh tables before plotting.

Default `event_window` spans **2022-11-01** through **2023-04-30** UTC (exclusive end `2023-05-01`), with launch anchor **2022-11-30**. Grouping is config-driven from `topics` and `topic_families` in the same YAML and consumed by event-time family plots and family-faceted subreddit plots. Filtering ingests required monthly comment dumps using configurable worker concurrency (default sequential one-worker for external-disk stability), checkpoints, and time-aware progress logging into per-subreddit/day NDJSON. Pre-cleaning quality trends live under `results/tables/data_quality_trends/` and `results/figures/data_quality_trends/` (AutoModerator plot notes use the window-summed count from each run, event-window bounds are enforced at table/plot time, plotting prints per-metric progress with terminal-safe non-interactive rendering, and date axes use month-start ticks plus release markers at **2022-11-30** and **2023-03-14**). Cleaned interim data is stored as monthly Parquet per subreddit in `data/interim/political_forums/cleaned_monthly_chunks/`, with explicit schema coercion and mismatch reporting under `results/tables/cleaning/clean_daily_chunks_schema_*.csv`. Reusable per-comment feature shards are written to `data/interim/political_forums/comment_features/` (includes `author` / `created_utc` when present in cleaned shards; MPS-first device auto-routing, batching controls, skip-existing behavior, no short-comment exclusion, and coverage/confidence metadata). Optional `scripts/features/compute_daily_repetition_similarity.py` supplies `repetition_daily_by_subreddit.csv` for `scripts/event_time/prepare_event_time_metrics.py` to merge. Event-time tables and figures remain under `results/tables/event_time/` and `results/figures/event_time/`, now rendered on calendar-date axes with month-start ticks and high-contrast multi-line color assignment, with pooled, by-family, and optional by-subreddit-by-family defaults focused on daily plus rolling-daily (weekly is an opt-in extra via `--include_weekly`). Optional **stratified pooled** event-time adds `scripts/event_time/prepare_event_time_stratified_metrics.py` and `scripts/event_time/plot_event_time_stratified_metrics.py` (user cohort + length buckets; stratified outputs omit repetition/Jaccard; default views are daily + rolling-daily, with optional weekly; see `event_time_stratified_metrics_notes.txt`). Optional **within-user pre/post style shift** layer adds `scripts/user_week/prepare_user_week_style_panel.py` → `scripts/user_week/analyze_user_pre_post_shift.py` → `scripts/user_week/plot_user_pre_post_shift.py` (author × ISO-week panel with raw counts and sumsq for precision-aware pooled SEs; weekly view and pooled-comments view per user × feature; topic-stable sub-cohort and per-topic strata in the summary; placebo offset; outputs under `results/tables/user_week/` and `results/figures/user_week/<cohort>/`).
