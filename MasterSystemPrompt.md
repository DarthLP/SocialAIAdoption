# Master System Prompt

## Script Summary
This document defines stable technical context for the thesis workspace, including architecture, reproducibility standards, and project execution policy. Keep it concise and update it when workflow conventions or core technical decisions change.

## Project Objective
Build a reproducible pipeline to study AI-writing adoption in political Reddit communities around ChatGPT launch using monthly dump ingestion plus local filtering.

## Scope Boundaries
- This file stores stable context and execution conventions.
- This file does not store transient daily task chatter.
- Daily operational state belongs in `TODO.md`.

## Technical Architecture Overview
- Code layer: `src/`, `scripts/`, `config/`
- Data layer:
  - External raw dumps on mounted storage (`/Volumes/Expansion/Masterthesis/RawData/...`)
  - Project filtered raw outputs (`data/raw/political_forums/daily_chunks/`)
  - `data/interim/`, `data/processed/` for downstream transformations
- Output layer: `results/figures/`, `results/tables/`, `results/logs/`
  - Filtering audits and dedupe reports are grouped in `results/tables/filtering/`
  - Cleaning audits are grouped in `results/tables/cleaning/`
  - Dump filtering logs/state files are grouped in `results/logs/filter_dump/`
  - Overlap analysis outputs are grouped in `results/tables/user_overlap/`
  - Data-quality trend figures/tables are grouped in `results/figures/data_quality_trends/` and `results/tables/data_quality_trends/`
  - Event-time metric tables/figures are grouped in `results/tables/event_time/` and `results/figures/event_time/`
  - Policy: artifacts should always be written to grouped subfolders under `results/*/`
- Dump filtering architecture:
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
  - Pre-cleaning trend utility (`scripts/plot_data_quality_trends.py`) for daily quality-indicator counts/rates and launch-anchored visualization
  - Interim cleaning utility (`scripts/clean_daily_chunks.py`) for deterministic drop-rules plus retained-row metadata flags
  - Event-time metrics utility (`scripts/prepare_event_time_metrics.py`) for daily subreddit/pooled language and AI-style aggregates
  - Event-time plotting utility (`scripts/plot_event_time_metrics.py`) for launch-anchored pooled and per-subreddit trend figures, style-proxy panels, strict-vs-extended lexicon overlay, z-score component plot, and strict-10 per-word plus combined trajectory figure
  - Optional sampled detector utility (`scripts/run_llm_detector_sample.py`) for CPU-first robustness scoring with deterministic sampling
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
Dump-first ingestion path established; filtering uses configurable worker concurrency (default sequential one-worker for external-disk stability), large checkpoints, and time-aware progress logging to generate per-subreddit/day datasets for Nov/Dec 2022. Pre-cleaning quality trend outputs are standardized under `results/tables/data_quality_trends/` and `results/figures/data_quality_trends/`. Interim cleaned outputs exist under `data/interim/political_forums/cleaned_daily_chunks/` with explicit drop rules (`[removed]`, `[deleted]`, `AutoModerator`, `stickied`, `distinguished=moderator`) and retained-row flags (`is_deleted_author`, `is_bot_name_heuristic`, `is_url_only`, `is_short_text`). Event-time metrics are now generated under `results/tables/event_time/` with pooled + subreddit daily aggregates for linguistic style, AI-likeness components, strict/extended AI lexicon rates, and toxicity proxies; event-time figures are generated under `results/figures/event_time/` including strict-10 individual-word plus combined trajectory visualization.
