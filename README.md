# SocialAIAdoption

## Objective
This repository supports thesis analysis of AI-writing adoption in **political** Reddit communities around the ChatGPT launch date, plus a **tech comparison** arm (non-political) for contrast. The default [`config/political_forums_setup.yaml`](config/political_forums_setup.yaml) `subreddits.primary` list is:

- **Political:** `politics`, `PoliticalDiscussion`, `NeutralPolitics`, `moderatepolitics`, `Ask_Politics`
- **Coding comparison:** `learnprogramming`, `AskProgramming`, `CodingHelp`
- **Career comparison:** `cscareerquestions`, `ITCareerQuestions`, `csMajors`
- **General Q&A (mid-size):** `answers`, `TooAfraidToAsk`, `OutOfTheLoop`

The pipeline is dump-first: download monthly Reddit dumps to external storage, then filter locally into a compact project dataset for reproducible analysis.

## Quick Start
1. Create and activate the local environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Verify dump files are present on external storage (must cover every calendar month in `event_window`; default config uses **2022-11-01** through **2023-04-30** UTC, so six comment files):
   - `.../reddit/comments/RC_2022-11.zst` … `RC_2023-04.zst` under `/Volumes/Expansion/Masterthesis/RawData/` (or your `--source_dir`).
4. Run filtering:
   - `.venv/bin/python scripts/filter_dump_comments.py --config config/political_forums_setup.yaml` (default: one worker, all required `RC_*.zst` files in chronological order)
   - **If you add or remove entries under `subreddits.primary` after a month is already marked complete** in worker state, delete the corresponding `results/logs/filter_dump/filter_dump_state.RC_YYYY-MM.json` files (and merged `filter_dump_state.json` if present) before re-running; otherwise completed months are skipped and new subreddits will not be extracted.
   - Optional anchor rerun mode: `.venv/bin/python scripts/filter_dump_comments.py --config config/political_forums_setup.yaml --resume_from_anchor first_in_window`
   - Optional two-worker parallel mode: `.venv/bin/python scripts/filter_dump_comments.py --config config/political_forums_setup.yaml --worker_mode two`
   - Optional prefilter A/B mode: `.venv/bin/python scripts/filter_dump_comments.py --config config/political_forums_setup.yaml --prefilter_mode regex`
5. Check outputs:
   - `data/raw/political_forums/daily_chunks/`
   - `results/tables/filtering/dump_filter_counts_by_day.csv`
   - `results/tables/filtering/dump_filter_counts_by_subreddit.csv`
   - `results/logs/filter_dump/filter_dump.RC_<YYYY-MM>.log` (one log per monthly source file)
6. Optional overlap cleanup after stop/restart:
   - Dry run: `.venv/bin/python scripts/dedupe_daily_chunks.py --config config/political_forums_setup.yaml`
   - Apply: `.venv/bin/python scripts/dedupe_daily_chunks.py --config config/political_forums_setup.yaml --apply`
7. Optional cross-forum user overlap analysis (unique-author matching across subreddits):
   - `.venv/bin/python scripts/user_overlap_across_forums.py --config config/political_forums_setup.yaml`
   - Uses cleaned input: `data/interim/political_forums/cleaned_monthly_chunks/`
   - Writes `results/tables/user_overlap/user_overlap_by_forum.csv`, `user_overlap_forum_count_distribution.csv`, and `user_overlap_pairwise.csv`.
8. Optional same-day cross-forum activity analysis (users posting in >=2 forums on the same UTC day):
   - `.venv/bin/python scripts/user_same_day_cross_forum.py --config config/political_forums_setup.yaml`
   - Uses cleaned input: `data/interim/political_forums/cleaned_monthly_chunks/`
   - Writes `results/tables/user_overlap/user_same_day_cross_forum_summary.csv`, `user_same_day_cross_forum_distribution.csv`, and `user_same_day_cross_forum_pairwise.csv`.
9. Pre-cleaning data-quality trend analysis (percentages, ChatGPT/GPT-4 event markers):
   - `.venv/bin/python scripts/plot_data_quality_trends.py --config config/political_forums_setup.yaml`
   - Writes tables to `results/tables/data_quality_trends/` and figures to `results/figures/data_quality_trends/`.
   - Uses calendar-date month-start ticks and red dotted vertical markers at `2022-11-30` and `2023-03-14`.
10. Deterministic cleaning pass for interim analysis dataset:
   - `.venv/bin/python scripts/clean_daily_chunks.py --config config/political_forums_setup.yaml`
   - Writes cleaned monthly Parquet files to `data/interim/political_forums/cleaned_monthly_chunks/<subreddit>/<YYYY-MM>.parquet`.
   - Writes cleaning audits to `results/tables/cleaning/clean_daily_chunks_audit_by_day.csv`, `results/tables/cleaning/clean_daily_chunks_audit_by_subreddit.csv`, `results/tables/cleaning/clean_daily_chunks_notes.txt`, and schema-coercion diagnostics (`clean_daily_chunks_schema_*.csv`).
11. Reusable per-comment feature extraction (recommended before event-time aggregation):
   - **Single-machine (default):** `.venv/bin/python scripts/compute_comment_features.py --config config/political_forums_setup.yaml` writes `data/interim/political_forums/comment_features/<subreddit>/<YYYY-MM>.parquet` (lexical + HF models together; `--device auto|mps|cpu`; includes `author` and `created_utc` when present in cleaned Parquet).
   - **Split Colab GPU + laptop CPU:** (1) On Colab upload and run only [`colab_compute_comment_features_gpu.ipynb`](notebooks/colab_compute_comment_features_gpu.ipynb): the notebook is self-contained (embedded config + inlined inference from `src/comment_feature_models.py`); sync `cleaned_monthly_chunks` from Drive → run cells with GPU runtime when `DEVICE = "cuda"` → sync `comment_features_ml/` back to Drive. No Git clone and no subprocess to repo scripts on Colab. (2) On this machine copy Drive’s `comment_features_ml/` into `data/interim/political_forums/comment_features_ml/` and run `.venv/bin/python scripts/merge_ml_shards_into_comment_features.py --config config/political_forums_setup.yaml` to merge ML shards with locally computed lexical fields into `comment_features/` (same schema as the monolithic script).
   - Fast-run controls (both ML and lexical scripts mirror the bounded flags where applicable): `--batch_size`, `--workers`, `--max_month_files_per_subreddit`, `--max_total_month_files`, `--max_days_per_month`, `--profile`, `--overwrite`, `--subreddits`, `--months`.
11b. (Optional) Daily repetition / template similarity for event-time merge:
   - `.venv/bin/python scripts/compute_daily_repetition_similarity.py --config config/political_forums_setup.yaml` writes `results/tables/event_time/repetition_daily_by_subreddit.csv` from cleaned monthly chunks (ordered by `created_utc` within each day).
12. Event-time metric preparation (subreddit + pooled, lexical/structure/toxicity proxies):
   - `.venv/bin/python scripts/prepare_event_time_metrics.py --config config/political_forums_setup.yaml` (requires `comment_features/`; merges repetition CSV when present).
   - Writes tables to `results/tables/event_time/` and compatibility export to `results/tables/event_time_daily_metrics.csv`.
   - Writes validation associations to `results/tables/event_time/comment_feature_validation_associations.csv` when comment-features are used.
   - For fast benchmarking without full-run wait time, use bounded sampling controls:
     - `--max_month_files_per_subreddit`, `--max_total_month_files`, `--max_days_per_month`
     - optional phase profiling via `--profile` / `--profile_output ...json`
13. Event-time plotting:
   - `.venv/bin/python scripts/plot_event_time_metrics.py --config config/political_forums_setup.yaml`
   - Writes pooled figures to `results/figures/event_time/pooled/{daily,weekly,rolling_daily}/` (lexicon, style proxies, toxicity, strict-vs-extended overlay, style panel, z-score components).
   - Writes per-subreddit overlays to `results/figures/event_time/by_subreddit/{daily,weekly,rolling_daily}/`.
   - Includes one figure with strict 10-word individual rates plus strict-10 combined rate in one graph (pooled).
   - Uses calendar-date x-axes with month-start ticks, plus red dotted release markers for ChatGPT (`2022-11-30`) and GPT-4 (`2023-03-14`).
   - Multi-line subreddit overlays use explicit high-contrast palettes for clearer line separation.
  - Optional topic-level views (daily, weekly, rolling-daily):
     - `.venv/bin/python scripts/plot_event_time_metrics.py --config config/political_forums_setup.yaml --topic_views --topic_rolling_window 7`
     - Writes topic overlays to `results/figures/event_time/by_topic/{daily,weekly,rolling_daily}/`.
     - Topic map:
       - `coding`: `AskProgramming`, `CodingHelp`, `learnprogramming`
       - `politics`: `Ask_Politics`, `NeutralPolitics`, `PoliticalDiscussion`, `politics`, `moderatepolitics`
       - `career`: `cscareerquestions`, `ITCareerQuestions`, `csMajors`
       - `general_questions`: `answers`, `OutOfTheLoop`, `TooAfraidToAsk`
13b. Optional stratified pooled event-time (old / new / new users’ debut comments; length buckets; no Jaccard repetition):
   - `.venv/bin/python scripts/prepare_event_time_stratified_metrics.py --config config/political_forums_setup.yaml`
   - `.venv/bin/python scripts/plot_event_time_stratified_metrics.py --config config/political_forums_setup.yaml`
   - Tables: `results/tables/event_time/event_time_daily_metrics_pooled_by_user_cohort.csv`, `..._by_length_bucket.csv`, `event_time_length_bucket_daily_shares_pooled.csv`, notes in `event_time_stratified_metrics_notes.txt`.
   - Figures: `results/figures/event_time/stratified_pooled/user_series/{daily,weekly,rolling_daily}/` and `stratified_pooled/length_bucket/{daily,weekly,rolling_daily}/` (length-bucket plots omit detector/perplexity/hostility/emotion/coverage metrics as nonsensical for that stratifier).
   - Cohort definitions use earliest observed post per `(author, subreddit)` vs `launch_day_utc` (left-censoring if history starts after true first post); see notes file.
13c. Optional within-user pre/post style shift analysis (author × ISO-week layer, parallel to event-time):
   - `.venv/bin/python scripts/prepare_user_week_style_panel.py --config config/political_forums_setup.yaml` builds a per-author per-ISO-week style panel from `comment_features/`, persisting both display rates / weighted means and the raw hit counts / sums-of-squares needed for precision-aware pooled estimates (`data/interim/political_forums/user_week_style_panel/<YYYY-MM>.parquet`, merged at `results/tables/user_week/user_week_panel.parquet`). Author hygiene drops empty / `[deleted]` / `AutoModerator` / `bot`-substring accounts.
   - `.venv/bin/python scripts/analyze_user_pre_post_shift.py --config config/political_forums_setup.yaml` produces per-user shift tables for two parallel comparisons: a **weekly view** (word-weighted mean and Kish-corrected SD across pre vs post weeks, std_delta with a winsorized SD floor, robust MAD variant, Welch-style across-weeks t) and a **pooled-comments view** (rate features use Poisson SE on raw hit counts, binary-mean uses binomial SE, mean features use sumsq-derived SE, composite SE via independence-approx delta method on z-scaled components). Hard pre+post requirement: pre-only / post-only / below-thresholds users surface in `shift_audit_per_user_<cohort>.csv` and the audit_* rows of `shift_summary_<cohort>.csv` instead of being silently dropped. Composite z-scales are frozen on the pre-launch user-week pool (`composite_zscale_pre_<cohort>.json`). Topic-stable sub-cohort, topic-stratified rows, inverse-variance weighted pooled effect, agreement diagnostic between weekly and pooled views, and a placebo run with the launch shifted back `--placebo_offset_weeks` weeks (default 8) all live in the summary CSV. Default cohorts: **strict** (≥4 good pre weeks AND ≥4 good post weeks, ≥100 words/week, ≥400 words/period) and **loose** (≥2/≥2, ≥30 words/week, ≥100 words/period); both are produced by default.
   - `.venv/bin/python scripts/plot_user_pre_post_shift.py --config config/political_forums_setup.yaml` renders the figure set under `results/figures/user_week/<cohort>/`: `dist_std_delta_composite.png`, `dist_t_user_pooled_composite.png`, `weekly_vs_pooled_scatter.png`, `components_grid.png`, `spaghetti_sample.png`, `mirror_top_movers.png`. Spaghetti and mirror plots use the same red dotted release markers (`2022-11-30`, `2023-03-14`) as the event-time figures.
14. Optional sampled LLM-detector robustness table (CPU-only default heuristic, optional HF model):
   - `.venv/bin/python scripts/run_llm_detector_sample.py --config config/political_forums_setup.yaml`
   - Optional HF detector branch: add `--use_hf_model` (requires `transformers` installed in `.venv`).
   - Writes `results/tables/event_time/llm_detector_sample_scores_daily.csv`.

## External Resource
- Academic Torrents Reddit dataset page:
  - [reddit-ba051999301b109eab37d16f027b3f49ade2de13](https://academictorrents.com/details/ba051999301b109eab37d16f027b3f49ade2de13/tech&filelist=1)
- Example torrent command for Nov/Dec 2022 comments only:
  - `aria2c --dir "/Volumes/Expansion/Masterthesis/RawData" --seed-ratio=0 --file-allocation=none --select-file=204,205 "data/reddit-ba051999301b109eab37d16f027b3f49ade2de13.torrent"`
- **January–April 2023 comment months** (same torrent; file indices `206`–`209` = `RC_2023-01.zst` … `RC_2023-04.zst`):
  1. **Preflight** (do not start a second client on the same files):
     - `pgrep -x aria2c || echo "ok: no aria2c"`
     - If that prints a PID, wait or quit the other run before starting.
  2. **Download** (writes under `RawData/reddit/comments/` next to any existing dumps; `--continue=true` resumes partials without touching unrelated files):
     - From repo root, with the `.torrent` saved as `data/reddit-ba051999301b109eab37d16f027b3f49ade2de13.torrent`:
       - `caffeinate -dims aria2c --dir "/Volumes/Expansion/Masterthesis/RawData" --seed-ratio=0 --file-allocation=none --continue=true --select-file=206,207,208,209 "data/reddit-ba051999301b109eab37d16f027b3f49ade2de13.torrent"`
     - `caffeinate -dims` keeps the display, system, and disk awake while `aria2c` runs (use `-i` only if you prefer a lighter hold).
  3. **Other calendar months**: run `aria2c --show-files=true path/to.torrent` and adjust `--select-file=` to the indices you need (indices are **1-based** and specific to this torrent revision). For **2024** January–April comments on the same listing, use `--select-file=218,219,220,221`.

## Directory Structure
- `.cursor/rules/`: Cursor operational rules.
- `src/`: Reusable Python modules.
- `scripts/`: Reproducible run entrypoints (filtering + plotting).
- `notebooks/`: Self-contained Colab notebook(s) (e.g. ML comment features + Drive sync; source kept in repo via `scripts/_gen_colab_standalone_nb.py` when YAML or inference code changes).
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
- `results/tables/user_week/`: Author × ISO-week panel and within-user pre/post shift outputs (`user_week_panel.parquet`, `shift_per_user_<cohort>.csv`, `shift_summary_<cohort>.csv`, `shift_audit_per_user_<cohort>.csv`, `composite_zscale_pre_<cohort>.json`, `shift_methods_note.txt`).
- `results/figures/user_week/<cohort>/`: Within-user shift figures (composite distribution, weekly-vs-pooled scatter, components grid, spaghetti sample, top-mover mirror plot).
- `results/figures/data_quality_trends/`: Daily percentage trend plots with ChatGPT and GPT-4 release markers.
- `results/figures/event_time/`: Event-time figures for linguistic, AI-style, and toxicity proxies.
- `Projects/`, `Decisions/`: Obsidian durable memory notes.
- `Templates/`: Standardized lightweight note templates.
- `MasterSystemPrompt.md`: Stable project-level context and execution policy.
- `TODO.md`: Active implementation board.

## Implementation Timeline
- Stage 1: Acquire monthly dump files on external storage (default window: **Nov 2022–Apr 2023** comment months).
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
- Optional model-feature packages: `transformers`, `torch`, `sentencepiece`.

## Usage
- For script-by-script execution order, short purpose, I/O data layers, and exact run commands, see `scripts/README.md`.
- Use external raw dumps as source of truth for ingestion.
- Use `scripts/filter_dump_comments.py` to generate filtered day-chunk comments in the project data directory. Required `RC_YYYY-MM.zst` basenames are derived from `event_window` using `src.config_utils.comment_dump_filenames`.
- Use `scripts/dedupe_daily_chunks.py` when needed to remove duplicate comment ids introduced by interrupted/restarted filtering.
- Use `scripts/user_overlap_across_forums.py` to check how many users post in more than one target subreddit (exact match on Reddit's globally-unique `author` field; `[deleted]` and known bots excluded by default).
- Use `scripts/user_same_day_cross_forum.py` for a stricter, temporally-aligned overlap check: same user posting in >=2 different subreddits on the same UTC day.
- Use `scripts/plot_data_quality_trends.py` before cleaning decisions to inspect indicator behavior over time around key release dates (`2022-11-30` and `2023-03-14`) with month-start tick alignment.
- Trend metrics include: `rows_total`, `body_removed_count`, `body_deleted_count`, `author_deleted_count`, `automod_author_count`, `stickied_count`, and exploratory `bot_name_heuristic_count` plus daily percent rates.
- Trend figures are percentage-based for comparability across variable daily volume; absolute counts remain available in the output tables.
- For moderation automation, use `author == "AutoModerator"` as the canonical plotted series. A documented near-equivalence check on an earlier narrow window found only one mismatch row versus `distinguished == "moderator"` (AutoModerator with null distinguished).
- Quality-trend figures annotate AutoModerator plots with the **sum of `automod_author_count` over the current event window** (see `results/tables/data_quality_trends/quality_trends_notes.txt`).
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
- Interim cleaned storage is Parquet-only and month-per-forum at `data/interim/political_forums/cleaned_monthly_chunks/<subreddit>/<YYYY-MM>.parquet`.
- `scripts/clean_daily_chunks.py` enforces a fixed schema for interim data and writes mismatch/coercion diagnostics under `results/tables/cleaning/clean_daily_chunks_schema_*.csv`.
- The filter supports `--worker_mode one|two|auto`; default is `one` (sequential: every required month file runs in order, best for typical external USB throughput).
- Use `scripts/prepare_event_time_metrics.py` to build metric-ready daily aggregates from `comment_features/` for pooled and subreddit event-time analysis (optional merge of `repetition_daily_by_subreddit.csv`). Pooled `ALL` rows aggregate across **every** forum in `subreddits.primary` (political and tech); use per-subreddit tables or stratify later if you need domain-pure pools.
- Event-time outputs include:
  - semicolon rate, comment length, complexity index
  - AI-likeness composite index and component columns
  - strict 10-word and extended AI-typical word rates
  - formality markers, list-structure intensity, repetition/template similarity, assistant-tone phrase rate
  - toxicity proxies: VADER negativity mean and lexical toxic incidence rate
- Additional reusable comment-feature outputs include detector-based human scores, passive proxy rate, perplexity, hostility score, emotion scores (anger/fear/sadness/surprise), and per-day coverage columns.
- Use `scripts/plot_event_time_metrics.py` to render date-based trend plots, including a combined figure with strict 10 individual word trajectories plus strict-10 combined trajectory.
- Use `scripts/run_llm_detector_sample.py` for optional sampled robustness scoring:
  - deterministic stratified sampling by subreddit x day
  - default free heuristic LLM-style score
  - optional pinned Hugging Face classifier branch (CPU-compatible; slower)
- Use `--worker_mode two` if your storage can sustain parallel reads (e.g. fast internal disk).
- The filter supports `--prefilter_mode tokens|regex`; default is `tokens` and `regex` is intended for A/B benchmarking.
- Checkpointing remains at `1_000_000` scanned lines by default.
- Progress logs include throughput (`lines/s`) and latest seen `created_utc` timestamp to monitor where the run is in event time.
- On graceful stop (`Ctrl+C`/`SIGTERM`), workers checkpoint immediately so restart resumes from the exact saved line and avoids tail-interval duplicate appends.
- Workers stop early once data has passed the relevant time window boundary (for example, `RC_2022-11` stops after the scan passes **Dec 1 UTC** file-internal ordering, and the last file stops after passing `end_utc_exclusive`).
- **Widening `start_utc` earlier inside a month you already filtered** (e.g. moving from Nov 16 to Nov 1 while keeping `RC_2022-11.zst`): delete that month’s per-file state under `results/logs/filter_dump/` (`filter_dump_state.RC_2022-11.json` and the merged `filter_dump_state.json` if present) before re-running, or the worker will resume mid-file and **miss** the new early-window rows.
- **Widening `end_utc_exclusive` within the same `RC_*.zst` month** (e.g. you previously stopped in mid-December and now include all of December): worker state stores `filter_window_start_ts` / `filter_window_end_ts_exclusive`; if the window changes, a previously `completed` month file is **not** short-circuited and resumes from the saved line offset (no manual delete). Legacy state files without those keys still resume once instead of skipping, which may re-stream a fully finished large month; delete that worker state if you want to avoid the extra pass.
- Worker state includes source file fingerprint checks (`path`, `size`, `mtime`); resume fails fast if file metadata changed.
- Worker state also stores low-cost anchors (for example `first_in_window_line`) for optional fast-start reruns.
- Use `--resume_from_anchor first_in_window` only when you intentionally want to rerun from that saved anchor (typically with fresh outputs or followed by dedupe).
- Use downstream scripts on filtered outputs only; avoid direct analysis on full raw dumps.
- Track major methodological decisions in `Decisions/` and task flow in `TODO.md`.
