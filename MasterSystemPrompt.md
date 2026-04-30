# Master System Prompt

## Script Summary
This document defines stable technical context for the thesis workspace, including architecture, reproducibility standards, and project execution policy. Keep it concise and update it when workflow conventions or core technical decisions change.

## Project Objective
Build a reproducible pipeline to study AI-writing adoption in political Reddit communities around ChatGPT launch using monthly dump ingestion plus local filtering, with expanded comparison corpora in the same `subreddits.primary` list: **tech** (coding + career) and **general Q&A** subs for answer-format and topic diversity (see `config/political_forums_setup.yaml`).

## Scope Boundaries
- This file stores stable context and execution conventions.
- This file does not store transient daily task chatter.
- Daily operational state belongs in `TODO.md`.

## Technical Architecture Overview
- Code layer: `src/`, `scripts/`, `config/`
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
  - Post-run dedupe utility (`scripts/dedupe_daily_chunks.py`) for id-based overlap cleanup if needed
  - Cross-forum user overlap utility (`scripts/user_overlap_across_forums.py`) for author-level forum-membership diagnostics
  - Same-day cross-forum activity utility (`scripts/user_same_day_cross_forum.py`) for temporally-aligned author overlap diagnostics
  - Pre-cleaning trend utility (`scripts/plot_data_quality_trends.py`) for daily quality-indicator counts/rates with month-start date ticks, ChatGPT (`2022-11-30`) and GPT-4 (`2023-03-14`) markers, and high-contrast multi-line subreddit palettes
  - Interim cleaning utility (`scripts/clean_daily_chunks.py`) for deterministic drop-rules plus retained-row metadata flags, canonical schema enforcement, and coercion diagnostics
  - Reusable comment-feature utility (`scripts/compute_comment_features.py`) for per-comment lexical/style/toxicity channels plus detector, passive-proxy, perplexity, hostility, and emotion features with no short-comment exclusion and coverage/confidence metadata
  - Event-time metrics utility (`scripts/prepare_event_time_metrics.py`) for daily subreddit/pooled language and AI-style aggregates (pooled `ALL` mixes all `subreddits.primary` forums unless stratified pools are added later), now with bounded benchmark controls (`--max_month_files_per_subreddit`, `--max_total_month_files`, `--max_days_per_month`), phase-level profiling (`--profile`, `--profile_output`), and optional month-level parallel workers (`--workers`)
  - Event-time plotting utility (`scripts/plot_event_time_metrics.py`) for calendar-date pooled and per-subreddit trend figures with month-start ticks and dual release markers (`2022-11-30`, `2023-03-14`), plus style-proxy panels, strict-vs-extended lexicon overlay, z-score component plot, strict-10 per-word plus combined trajectory figure, with outputs split by temporal view (`daily`, `weekly`, `rolling_daily`) under pooled/per-subreddit folders and optional topic-level overlays under `results/figures/event_time/by_topic/`
  - Optional sampled detector utility (`scripts/run_llm_detector_sample.py`) for CPU-first robustness scoring with deterministic sampling
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

## Quality Gates
A work item is complete only if:
- The implementation is reproducible from scripts/config.
- Impacted documentation is updated where necessary (`README.md`, `TODO.md`, this file).
- Durable notes are updated only when new stable knowledge emerged.
- The vault remains low-noise and non-duplicative.

## Current Status
Default `event_window` spans **2022-11-01** through **2023-04-30** UTC (exclusive end `2023-05-01`), with launch anchor **2022-11-30**. `subreddits.primary` includes five political subs, six tech comparison subs (`learnprogramming`, `AskProgramming`, `CodingHelp`, `cscareerquestions`, `ITCareerQuestions`, `csMajors`), and three general-question subs (`answers`, `TooAfraidToAsk`, `OutOfTheLoop`). Filtering ingests six monthly comment dumps (`RC_2022-11` … `RC_2023-04`) using configurable worker concurrency (default sequential one-worker for external-disk stability), checkpoints, and time-aware progress logging into per-subreddit/day NDJSON. Pre-cleaning quality trends live under `results/tables/data_quality_trends/` and `results/figures/data_quality_trends/` (AutoModerator plot notes use the window-summed count from each run, event-window bounds are enforced at table/plot time, plotting prints per-metric progress with terminal-safe non-interactive rendering, and date axes use month-start ticks plus release markers at **2022-11-30** and **2023-03-14**). Cleaned interim data is stored as monthly Parquet per subreddit in `data/interim/political_forums/cleaned_monthly_chunks/`, with explicit schema coercion and mismatch reporting under `results/tables/cleaning/clean_daily_chunks_schema_*.csv`. Reusable per-comment feature shards are written to `data/interim/political_forums/comment_features/` (MPS-first device auto-routing, batching controls, skip-existing behavior, no short-comment exclusion, and coverage/confidence metadata). Event-time tables and figures remain under `results/tables/event_time/` and `results/figures/event_time/`, now rendered on calendar-date axes with month-start ticks and high-contrast multi-line color assignment, with optional topic-level views (daily, weekly, rolling-daily) based on the fixed map: coding (`AskProgramming`, `CodingHelp`, `learnprogramming`), politics (`Ask_Politics`, `NeutralPolitics`, `PoliticalDiscussion`, `politics`, `moderatepolitics`), career (`cscareerquestions`, `ITCareerQuestions`, `csMajors`), and general_questions (`answers`, `OutOfTheLoop`, `TooAfraidToAsk`).
