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
   - `.venv/bin/python scripts/filter_dump_comments.py --config config/political_forums_setup.yaml` (default: one worker, Nov then Dec in order)
   - Optional anchor rerun mode: `.venv/bin/python scripts/filter_dump_comments.py --config config/political_forums_setup.yaml --resume_from_anchor first_in_window`
   - Optional two-worker parallel mode: `.venv/bin/python scripts/filter_dump_comments.py --config config/political_forums_setup.yaml --worker_mode two`
   - Optional prefilter A/B mode: `.venv/bin/python scripts/filter_dump_comments.py --config config/political_forums_setup.yaml --prefilter_mode regex`
5. Check outputs:
   - `data/raw/political_forums/daily_chunks/`
   - `results/tables/filtering/dump_filter_counts_by_day.csv`
   - `results/tables/filtering/dump_filter_counts_by_subreddit.csv`
   - `results/logs/filter_dump/filter_dump.RC_2022-11.log`
   - `results/logs/filter_dump/filter_dump.RC_2022-12.log`
6. Optional overlap cleanup after stop/restart:
   - Dry run: `.venv/bin/python scripts/dedupe_daily_chunks.py --config config/political_forums_setup.yaml`
   - Apply: `.venv/bin/python scripts/dedupe_daily_chunks.py --config config/political_forums_setup.yaml --apply`
7. Optional cross-forum user overlap analysis (unique-author matching across subreddits):
   - `.venv/bin/python scripts/user_overlap_across_forums.py --config config/political_forums_setup.yaml`
   - Uses cleaned input: `data/interim/political_forums/cleaned_daily_chunks/`
   - Writes `results/tables/user_overlap/user_overlap_by_forum.csv`, `user_overlap_forum_count_distribution.csv`, and `user_overlap_pairwise.csv`.
8. Optional same-day cross-forum activity analysis (users posting in >=2 forums on the same UTC day):
   - `.venv/bin/python scripts/user_same_day_cross_forum.py --config config/political_forums_setup.yaml`
   - Uses cleaned input: `data/interim/political_forums/cleaned_daily_chunks/`
   - Writes `results/tables/user_overlap/user_same_day_cross_forum_summary.csv`, `user_same_day_cross_forum_distribution.csv`, and `user_same_day_cross_forum_pairwise.csv`.
9. Pre-cleaning data-quality trend analysis (percentages, ChatGPT event marker):
   - `.venv/bin/python scripts/plot_data_quality_trends.py --config config/political_forums_setup.yaml`
   - Writes tables to `results/tables/data_quality_trends/` and figures to `results/figures/data_quality_trends/`.
10. Deterministic cleaning pass for interim analysis dataset:
   - `.venv/bin/python scripts/clean_daily_chunks.py --config config/political_forums_setup.yaml`
   - Writes cleaned daily chunks to `data/interim/political_forums/cleaned_daily_chunks/`.
   - Writes cleaning audits to `results/tables/cleaning/clean_daily_chunks_audit_by_day.csv`, `results/tables/cleaning/clean_daily_chunks_audit_by_subreddit.csv`, and `results/tables/cleaning/clean_daily_chunks_notes.txt`.
11. Event-time metric preparation (subreddit + pooled, lexical/structure/toxicity proxies):
   - `.venv/bin/python scripts/prepare_event_time_metrics.py --config config/political_forums_setup.yaml`
   - Writes tables to `results/tables/event_time/` and compatibility export to `results/tables/event_time_daily_metrics.csv`.
12. Event-time plotting:
   - `.venv/bin/python scripts/plot_event_time_metrics.py --config config/political_forums_setup.yaml`
   - Writes pooled figures to `results/figures/event_time/` (lexicon, style proxies, toxicity, strict-vs-extended overlay, style panel, z-score components).
   - Writes per-subreddit overlays to `results/figures/event_time/by_subreddit/`.
   - Includes one figure with strict 10-word individual rates plus strict-10 combined rate in one graph (pooled).
13. Optional sampled LLM-detector robustness table (CPU-only default heuristic, optional HF model):
   - `.venv/bin/python scripts/run_llm_detector_sample.py --config config/political_forums_setup.yaml`
   - Optional HF detector branch: add `--use_hf_model` (requires `transformers` installed in `.venv`).
   - Writes `results/tables/event_time/llm_detector_sample_scores_daily.csv`.

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
  - Policy: generated outputs are grouped in subfolders under each artifact root.
  - `results/tables/filtering/`: Filtering audit outputs and dedupe reports.
  - `results/tables/cleaning/`: Cleaning audit tables and cleaning run notes.
  - `results/tables/data_quality_trends/`: Daily pre-cleaning quality metrics and validation tables.
  - `results/tables/user_overlap/`: Cross-forum overlap and same-day overlap analysis tables.
- `results/tables/event_time/`: Event-time metric aggregates, lexicon trajectories, and optional sampled detector outputs.
  - `results/logs/filter_dump/`: Dump filtering run logs and resumable state files.
  - `results/figures/data_quality_trends/`: Daily percentage trend plots with launch-day marker.
- `results/figures/event_time/`: Event-time figures for linguistic, AI-style, and toxicity proxies.
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
- Optional package for sampled detector robustness branch: `transformers`.

## Usage
- Use external raw dumps as source of truth for ingestion.
- Use `scripts/filter_dump_comments.py` to generate filtered day-chunk comments in the project data directory.
- Use `scripts/dedupe_daily_chunks.py` when needed to remove duplicate comment ids introduced by interrupted/restarted filtering.
- Use `scripts/user_overlap_across_forums.py` to check how many users post in more than one target subreddit (exact match on Reddit's globally-unique `author` field; `[deleted]` and known bots excluded by default).
- Use `scripts/user_same_day_cross_forum.py` for a stricter, temporally-aligned overlap check: same user posting in >=2 different subreddits on the same UTC day.
- Use `scripts/plot_data_quality_trends.py` before cleaning decisions to inspect indicator behavior over time around launch day (`2022-11-30`).
- Trend metrics include: `rows_total`, `body_removed_count`, `body_deleted_count`, `author_deleted_count`, `automod_author_count`, `stickied_count`, and exploratory `bot_name_heuristic_count` plus daily percent rates.
- Trend figures are percentage-based for comparability across variable daily volume; absolute counts remain available in the output tables.
- For moderation automation, use `author == "AutoModerator"` as the canonical plotted series. A documented near-equivalence check found only one mismatch row versus `distinguished == "moderator"` (AutoModerator with null distinguished).
- Figures include the policy note: `author == "AutoModerator"` total = `8602` for this analysis window.
- Use `scripts/clean_daily_chunks.py` to build the interim cleaned corpus with the current policy:
  - Drop rows where `body == "[removed]"` or `body == "[deleted]"`.
  - Drop rows where `author == "AutoModerator"`.
  - Drop rows where `stickied == true`.
  - Drop rows where `distinguished == "moderator"`.
  - Keep URL-only text rows and keep `author == "[deleted]"` rows when body text remains.
- The cleaned dataset adds metadata flags without dropping on them:
  - `is_deleted_author` (`author == "[deleted]"`)
  - `is_bot_name_heuristic` (`"bot"` substring in lowercase author)
  - `is_url_only` (body is URL-only)
  - `is_short_text` (body character length `< 20`)
- The filter supports `--worker_mode one|two|auto`; default is `one` (sequential: both month files always run, best for typical external USB throughput).
- Use `scripts/prepare_event_time_metrics.py` to build metric-ready daily aggregates from cleaned chunks for pooled and subreddit event-time analysis.
- Event-time outputs include:
  - semicolon rate, comment length, complexity index
  - AI-likeness composite index and component columns
  - strict 10-word and extended AI-typical word rates
  - formality markers, list-structure intensity, repetition/template similarity, assistant-tone phrase rate
  - toxicity proxies: VADER negativity mean and lexical toxic incidence rate
- Use `scripts/plot_event_time_metrics.py` to render event-time plots, including a combined figure with strict 10 individual word trajectories plus strict-10 combined trajectory.
- Use `scripts/run_llm_detector_sample.py` for optional sampled robustness scoring:
  - deterministic stratified sampling by subreddit x day
  - default free heuristic LLM-style score
  - optional pinned Hugging Face classifier branch (CPU-compatible; slower)
- Use `--worker_mode two` if your storage can sustain parallel reads (e.g. fast internal disk).
- The filter supports `--prefilter_mode tokens|regex`; default is `tokens` and `regex` is intended for A/B benchmarking.
- Checkpointing remains at `1_000_000` scanned lines by default.
- Progress logs include throughput (`lines/s`) and latest seen `created_utc` timestamp to monitor where the run is in event time.
- On graceful stop (`Ctrl+C`/`SIGTERM`), workers checkpoint immediately so restart resumes from the exact saved line and avoids tail-interval duplicate appends.
- Workers stop early once data has passed the relevant time window boundary (for example, `RC_2022-11` stops after reaching Dec 1 UTC).
- Worker state includes source file fingerprint checks (`path`, `size`, `mtime`); resume fails fast if file metadata changed.
- Worker state also stores low-cost anchors (for example `first_in_window_line`) for optional fast-start reruns.
- Use `--resume_from_anchor first_in_window` only when you intentionally want to rerun from that saved anchor (typically with fresh outputs or followed by dedupe).
- Use downstream scripts on filtered outputs only; avoid direct analysis on full raw dumps.
- Track major methodological decisions in `Decisions/` and task flow in `TODO.md`.
