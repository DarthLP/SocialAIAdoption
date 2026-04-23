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
Dump-first ingestion path established; filtering now uses a throughput-optimized two-worker pipeline with large checkpoints and time-aware progress logging to generate per-subreddit/day datasets for Nov/Dec 2022.
