# SocialAIAdoption

## Objective
This repository supports thesis analysis of AI-writing adoption in political Reddit communities around the ChatGPT launch date. The current pipeline is dump-first: download monthly Reddit dumps to external storage, then filter locally into a compact project dataset for reproducible analysis.

## Quick Start
1. Create and activate the local environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Verify dump files are present on external storage:
   - `/Volumes/Expansion/Masterthesis/RawData/reddit/comments/RC_2022-11.zst`
   - `/Volumes/Expansion/Masterthesis/RawData/reddit/comments/RC_2022-12.zst`
4. Run filtering:
   - `.venv/bin/python scripts/filter_dump_comments.py --config config/political_forums_setup.yaml`
   - Optional anchor rerun mode: `.venv/bin/python scripts/filter_dump_comments.py --config config/political_forums_setup.yaml --resume_from_anchor first_in_window`
5. Check outputs:
   - `data/raw/political_forums/daily_chunks/`
   - `results/tables/dump_filter_counts_by_day.csv`
   - `results/tables/dump_filter_counts_by_subreddit.csv`
   - `results/logs/filter_dump.RC_2022-11.log`
   - `results/logs/filter_dump.RC_2022-12.log`
6. Optional overlap cleanup after stop/restart:
   - Dry run: `.venv/bin/python scripts/dedupe_daily_chunks.py --config config/political_forums_setup.yaml`
   - Apply: `.venv/bin/python scripts/dedupe_daily_chunks.py --config config/political_forums_setup.yaml --apply`

## External Resource
- Academic Torrents Reddit dataset page:
  - [reddit-ba051999301b109eab37d16f027b3f49ade2de13](https://academictorrents.com/details/ba051999301b109eab37d16f027b3f49ade2de13/tech&filelist=1)
- Example torrent command for Nov/Dec 2022 comments only:
  - `aria2c --dir "/Volumes/Expansion/Masterthesis/RawData" --seed-ratio=0 --file-allocation=none --select-file=204,205 "data/reddit-ba051999301b109eab37d16f027b3f49ade2de13.torrent"`

## Directory Structure
- `.cursor/rules/`: Cursor operational rules.
- `src/`: Reusable Python modules.
- `scripts/`: Reproducible run entrypoints (filtering + plotting).
- `config/`: Run configuration files.
- `data/raw/political_forums/daily_chunks/`: Filtered per-subreddit per-day comments.
- `data/interim/`, `data/processed/`: Intermediate and model-ready data layers.
- `results/figures/`, `results/tables/`, `results/logs/`: Generated artifacts.
- `Projects/`, `Decisions/`: Obsidian durable memory notes.
- `Templates/`: Standardized lightweight note templates.
- `MasterSystemPrompt.md`: Stable project-level context and execution policy.
- `TODO.md`: Active implementation board.

## Implementation Timeline
- Stage 1: Acquire monthly dump files on external storage (Nov/Dec 2022).
- Stage 2: Filter dumps to target subreddits/date window/fields.
- Stage 3: Build normalized analysis tables and daily event-time aggregates.
- Stage 4: Produce descriptives, plots, and regression-ready datasets.

## Obsidian Compatibility Notes
- Internal note links should use wikilinks (`[[NoteName]]`).
- Keep note formatting Obsidian-compatible markdown.
- Exclude caches, virtual environments, and heavy generated artifacts from Obsidian indexing workflows.

## Dependencies
- Core Python packages are listed in `requirements.txt`.
- Key packages: `pandas`, `pyarrow`, `zstandard`, `orjson`, `PyYAML`, `matplotlib`, `seaborn`, `textstat`, `vaderSentiment`.

## Usage
- Use external raw dumps as source of truth for ingestion.
- Use `scripts/filter_dump_comments.py` to generate filtered day-chunk comments in the project data directory.
- Use `scripts/dedupe_daily_chunks.py` when needed to remove duplicate comment ids introduced by interrupted/restarted filtering.
- The filter runs two worker processes by default (one per monthly file) and checkpoints every `1_000_000` scanned lines.
- Progress logs include throughput (`lines/s`) and latest seen `created_utc` timestamp to monitor where the run is in event time.
- On graceful stop (`Ctrl+C`/`SIGTERM`), workers checkpoint immediately so restart resumes from the exact saved line and avoids tail-interval duplicate appends.
- Workers stop early once data has passed the relevant time window boundary (for example, `RC_2022-11` stops after reaching Dec 1 UTC).
- Worker state includes source file fingerprint checks (`path`, `size`, `mtime`); resume fails fast if file metadata changed.
- Worker state also stores low-cost anchors (for example `first_in_window_line`) for optional fast-start reruns.
- Use `--resume_from_anchor first_in_window` only when you intentionally want to rerun from that saved anchor (typically with fresh outputs or followed by dedupe).
- Use downstream scripts on filtered outputs only; avoid direct analysis on full raw dumps.
- Track major methodological decisions in `Decisions/` and task flow in `TODO.md`.
