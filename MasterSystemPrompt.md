# Master System Prompt

## Script Summary
This document defines stable technical context for the thesis workspace, including architecture, reproducibility standards, and project execution policy. Keep it concise and update it when workflow conventions or core technical decisions change.

## Project Objective
Build a reproducible pipeline for **The Effects of AI Access on Online Political Polarization**: Reddit comments for **Mar–Apr 2023** (Italy ChatGPT ban window) with **data-driven Italian subreddit discovery** (first 3 UTC days of `RC_2023-03` only) plus fixed English/EU comparison forums. Active config: `config/italy_polarization_setup.yaml` (`resolve_primary_subreddits` unions controls, seeds, and `discovered_italian`). Raw extract: `data/raw/italy_polarization/daily_chunks/`. Archived AI-adoption study: `config/archive/ai_adoption_political_forums_setup.yaml`.

## Scope Boundaries
- This file stores stable context and execution conventions.
- This file does not store transient daily task chatter.
- Daily operational state belongs in `TODO.md`.

## Technical Architecture Overview
- Code layer: `src/`, `scripts/` (domain subfolders under `scripts/<domain>/`; see `scripts/README.md`), `config/`
- Data layer:
  - External raw dumps on mounted storage (`/Volumes/Expansion/Masterthesis/RawData/...`); additional months are acquired with `aria2c` selective `--select-file=` on the Academic Torrents bundle (see `README.md` for examples, preflight `pgrep -x aria2c`, and optional `caffeinate` wrapper).
  - Project filtered raw outputs (`data/raw/italy_polarization/daily_chunks/`) for the active study; legacy `data/raw/political_forums/` archived to external disk.
  - Discovery tables: `results/tables/italy_polarization/discovery/` (`subreddit_census_3d.csv`, `candidate_italian_subreddits.csv`, `extraction_size_preview.csv`).
  - `data/interim/italy_polarization/`, `data/processed/italy_polarization/` for the active study; legacy `political_forums` paths archived on external disk.
  - Interim canonical format: monthly Parquet (`data/interim/italy_polarization/cleaned_monthly_chunks/<subreddit>/<YYYY-MM>.parquet`).
  - Italy cleaning/taxonomy: `clean_daily_chunks.py` → `screen_subreddits.py` → `enrich_cleaned_chunks.py` → `export_italian_lexicon_v4.py --policy dominant` → `compute_enriched_shard_features.py --pass all` → `prepare_lexicon_descriptives.py` / `plot_lexicon_descriptives.py` (+ legacy `prepare_polarization_descriptives.py` / `plot_polarization_descriptives.py`). **Order:** dominant export + stage-4 features **before** P/R hand-labeling. Config: `polarization.ideology_scoring: dominant_v1` (mandatory assert in descriptives). Italian `net_ideology` uses dominant `ideology_it.txt`; pair columns `pair_framing_net_strict` / `_all` from `pairs_it.json` (zeros on non-`it` shards). Primary outcomes: `net_ideology`, `pair_framing_net_strict`, W0 launch, 7d rolling. Optional within-user layer: `scripts/user_week/prepare_user_week_style_panel.py` → `analyze_user_pre_post_shift.py` → `plot_user_pre_post_shift.py` (enriched shards only; legacy `comment_features` path under `scripts/archive/user_week/`). Topics: `it_political` / `it_pure_political` / `it_others` from graded WW (`forum_political_soft_threshold`, `forum_political_pure_threshold` — recalibrate after enrichment; placeholders 0.35/0.7). Controls `de`, `eu`, `us`, `uk`, `uk_political`. Family `it_political` pools both Italian political topics. Profile CSV includes `primary_lexicon` for EN control QA (`europe`, `ukpolitics`). Thread flag: `thread_political_weighted_points >= thread_political_min_points` (**3**); one grade-3 unique term suffices. Salience: `paths.political_lexicon_parallel` → `data/raw/political_lexicon_parallel.csv` (grades 1–3 → points 1/2/3, unique hits). Categorized lists: v4 dominant export (`ideology_it.txt`, `pairs_it.json`, `stance_it.txt`, `valence_it.txt`, `polarized_it.txt`; archive `ideology_it_broad.txt`). Volume bands: `large_volume`, `low_volume`, `excluded`. Canonical interim: enriched `cleaned_monthly_chunks/`. `r/europe` = English lexicon.
- Output layer: `results/figures/`, `results/tables/`, `results/logs/` — index in `results/README.md`; helpers in `src/config_utils.py` (`tables_subdir`, `figures_subdir`, `logs_subdir`, `filter_dump_logs_dir`).
  - Active study: `results/tables/italy_polarization/` and `results/figures/italy_polarization/` with stage subfolders (e.g. `descriptives/daily/by_family/`, `cleaning_pipeline/political_qa/`, `data_quality_trends/overall/`).
  - Per-study logs: `results/logs/italy_polarization/filter_dump/` (state JSON, filter log) and `runs/` (ad-hoc reruns).
  - Study-scoped overlap: `results/tables/italy_polarization/user_overlap/`; dedupe report under `.../filtering/`.
  - Legacy studies: `results/tables/archive/`, `results/figures/archive/`.
  - Policy: artifacts go in grouped subfolders under the study root, not flat dumps at stage root.
- Dump filtering architecture:
  - Required monthly comment dumps are derived from `event_window.start_utc` / `end_utc_exclusive` via `src.config_utils.comment_dump_filenames` (chronological `RC_YYYY-MM.zst` list).
  - If `start_utc` moves **earlier** within a calendar month already partially processed, delete that month’s filter state JSON under `results/logs/italy_polarization/filter_dump/` before re-run so early lines are not skipped by resume.
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
  - Pre-cleaning trend utility (`scripts/diagnostics/plot_data_quality_trends.py`) for daily quality-indicator counts; vertical markers from YAML `plot_reference_dates_utc` via `plot_reference_dates_calendar_utc` (Italy default: ban `2023-03-31`, lift `2023-04-28`).
  - Interim cleaning utility (`scripts/cleaning/clean_daily_chunks.py`) for deterministic drop-rules (incl. URL-only via `src/text_hygiene.py`), retained-row flags, canonical schema enforcement, and coercion diagnostics; Italy audits under `results/tables/italy_polarization/cleaning/`
- Italy style (active): `src/comment_style.py` + `scripts/features/_enriched_shard_runner.py` (style pass) write unprefixed punctuation/markdown/phrase counts in-place on enriched shards (`semicolon_count`, dash/colon/quote counts, `hedging_{lang}.txt`, etc.). Shared bootstrap: `scripts/_bootstrap.py`.
- Within-user pre/post (active): `scripts/user_week/*` on enriched shards; ban anchor `event_window.launch_day_utc` (`2023-03-31`); composites `polarization_composite_user_week` and `ai_style_composite_user_week` (see config `user_week`); outputs under `results/tables/italy_polarization/user_week/` and `results/figures/italy_polarization/user_week/<cohort>/<composite>/`.
- **Archived** (`scripts/archive/`, `config/archive/`): AI-adoption `comment_features/` + HF detectors + calendar event-time plots (`archive/event_time/`). Italy does **not** use that stack. See `scripts/archive/README.md`.
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
**Active milestone:** Mar–Apr 2023 extract → cleaning (stages 1–3) → polarization + AI + style features + descriptives on `data/interim/italy_polarization/cleaned_monthly_chunks/`. Optional author-week pre/post on enriched shards. Next: lexicon hand-label P/R, then event-study/DiD at ban dates. Filter: `italy_polarization_state.json`. **Archived:** `scripts/archive/` ML + `comment_features/` stack for legacy `ai_adoption_political_forums` config.
