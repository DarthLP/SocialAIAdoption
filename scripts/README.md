# Scripts Pipeline Guide

Results paths are indexed in [`results/README.md`](../results/README.md).

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
3. **Filter** — `scripts/filtering/filter_dump_comments.py` (state/log default to `results/logs/italy_polarization/filter_dump/`).
4. **Clean** — `scripts/cleaning/clean_daily_chunks.py` (row drops, URL-only removal).
5. **Screen** — `scripts/cleaning/screen_subreddits.py` (forum exclusions, pooled Italian langid).
6. **Enrich** — `scripts/cleaning/enrich_cleaned_chunks.py` (taxonomy + political lexicon scores; canonical interim = subreddit shards only).
7. **Lexicon audit** — `scripts/diagnostics/audit_political_lexicon.py`.
8. **Pipeline plots** — `scripts/diagnostics/plot_cleaning_pipeline_trends.py` (family/topic tables and figures).
9. **Raw QA (optional)** — `scripts/diagnostics/plot_data_quality_trends.py` (Stage 0; **no row drops**).

---

## Italy polarization — cleaning pipeline (active)

### 4a) Stage-1 row cleaning
- Script: `clean_daily_chunks.py`
- Why: Drops moderation placeholders, stickied/distinguished moderator rows, URL-only spam; keeps `[deleted]` authors.
- Input: `data/raw/italy_polarization/daily_chunks/`
- Output: `data/interim/italy_polarization/cleaned_monthly_chunks/<subreddit>/<YYYY-MM>.parquet`
- Audits: `results/tables/italy_polarization/cleaning/clean_daily_chunks_audit_*.csv`
- Run: `.venv/bin/python scripts/cleaning/clean_daily_chunks.py --config config/italy_polarization_setup.yaml`

### 4b) Stage-2 forum screening
- Script: `screen_subreddits.py`
- Why: Applies `PROFILE_USER`, `HIGH_URL_ONLY_SHARE`, `LOW_VOLUME_WINDOW`, `LOW_ITALIAN_POOLED` gates (thresholds in config `screening`).
- Input: cleaned monthly Parquet + Stage-1 audits
- Output: `results/tables/italy_polarization/screening/subreddit_screening_*.csv`, `subreddit_exclusions.csv`, `subreddit_exclusion_summary.csv`, `exclusion_summary_by_code.csv`, `screening_run_notes.txt`
- Volume bands: `large_volume` (≥100 kept), `low_volume` (<100), `excluded` (hard gates)
- Run: `.venv/bin/python scripts/cleaning/screen_subreddits.py --config config/italy_polarization_setup.yaml`

### 4c) Stage-3 enrichment
- Script: `enrich_cleaned_chunks.py`
- Why: Adds `topic`, `topic_family`, `volume_band`, `arm`, graded political salience (IT/EN/DE), `thread_id` roll-ups. Salience: `paths.political_lexicon_parallel` → `data/raw/political_lexicon_parallel.csv` (grades 1–3 → points 1/2/3 per unique term). Forum WW = `100 × Σ(points)/Σ(words)`; Italian topics: WW ≥ `forum_political_pure_threshold` → `it_pure_political`; WW ≥ `forum_political_soft_threshold` → `it_political` (recalibrate after enrichment via `political_threshold_sensitivity.csv`). Thread flag: `thread_political_weighted_points >= thread_political_min_points` (**3**). Comment columns: `political_g1_hits`, `political_g2_hits`, `political_g3_hits`, `political_weighted_points`, `political_rate_100w`.
- Polarization lexicons: `data/raw/polarization_lexicon_parallel.csv` (+ salience from `political_lexicon_parallel.csv`)
- Output: enriched Parquet in `cleaned_monthly_chunks/` (canonical); `subreddit_topic_assignment.csv`, `subreddit_forum_political_profile.csv`, `subreddit_topic_political_audit.csv`
- Run: `.venv/bin/python scripts/cleaning/enrich_cleaned_chunks.py --config config/italy_polarization_setup.yaml`
- **Auto:** on completion, calls `plot_cleaning_pipeline_trends.run_cleaning_pipeline_plots` (skip with `--skip-pipeline-plots`)
- Deprecated: `--write-by-family` (prefer `groupby('topic_family')` on shards)

### 4c-bis) Raw parallel lexicon preparation
- Script: `prepare_parallel_lexicon_raw.py`
- Why: Merge `ideology_parallel.csv` into `polarization_lexicon_parallel.csv`; build `style_phrase_parallel.csv` from legacy phrase txt.
- Output: updated `data/raw/*.csv`; optional `lexicon_export/parallel_vs_config_gap.csv` with `--gap-report`
- Run: `.venv/bin/python scripts/devtools/prepare_parallel_lexicon_raw.py --gap-report`
- Optional legacy snapshot: `export_italian_lexicon_v4.py --policy dominant` (not required for stage-4 scoring).

### 4i-bis) Lexicon descriptives (pair framing + metadata)
- Scripts: `prepare_lexicon_descriptives.py`, `plot_lexicon_descriptives.py`
- Shared helpers: `descriptives_util.py`
- Tables: `descriptives/primary_outcomes_launch_w0.csv`, `ban_windows_launch_primary.csv`, `rolling_*_by_topic_family.csv`
- Figures: `results/figures/italy_polarization/descriptives/{primary,ideology_dominant,pairs,stance,valence,polarized,trajectory_scatter}/`
- Primary (ex ante): `net_ideology` + `pair_framing_net_strict`, W0 launch, 7d rolling, `it_political`

### 4d) Lexicon audit
- Script: `audit_political_lexicon.py`
- Output: `results/tables/italy_polarization/cleaning_pipeline/lexicon_audit_*.csv`
- Run: `.venv/bin/python scripts/diagnostics/audit_political_lexicon.py --config config/italy_polarization_setup.yaml`

### 4e) Cleaning pipeline diagnostics
- Script: `plot_cleaning_pipeline_trends.py`
- Why: Stage-1 drop time-series, volume-band window summaries, langid by topic, word-weighted political metrics.
- Output: `results/tables/italy_polarization/cleaning_pipeline/`, `results/figures/italy_polarization/cleaning_pipeline/{volume,stage1_drop_rates,political_qa}/` (political audit: boxplot, subreddit bars, `by_subreddit_political_rate_vs_topic.png` name-axis scatter, `by_subreddit_political_rate_vs_thread_share.png` bubble; screened-in only)

### 4f–4g) Enriched-shard features (polarization, AI-use, style)
- Canonical: `compute_enriched_shard_features.py --pass all` (single parquet read/write per shard when all passes run).
- Wrappers (same behavior): `compute_polarization_features.py`, `compute_ai_use_features.py`, `compute_comment_style_features.py`.
- Input/output: enriched `cleaned_monthly_chunks/*.parquet` (in place).
- Parallelism: `--workers N` (default `min(8, cpu_count-1)`; `1` = sequential). Logs include per-shard `elapsed=` seconds.
- Scoring: module-level caches for `pairs_it.json` / `term_meta_it.json`; one tokenize per comment for IT polarization.
- Run (all passes):
  `.venv/bin/python scripts/features/compute_enriched_shard_features.py --config config/italy_polarization_setup.yaml --pass all --workers 8`
- Run (polarization only):
  `.venv/bin/python scripts/features/compute_polarization_features.py --config config/italy_polarization_setup.yaml --workers 8`
- Polarization lexicons: `ideology_it.txt` (dominant v4), `pairs_it.json`, `stance_it.txt`, `valence_it.txt`, `polarized_it.txt`; EN/DE ideology lists hand-curated
- `_polarization_score_row` copies all `POLARIZATION_COMMENT_COLUMNS` from scorer (KeyError if missing)

### 4h) Polarization lexicon audit
- Script: `audit_polarization_lexicons.py`
- Output: `results/tables/italy_polarization/descriptives/lexicon_audit_*.csv`, optional `lexicon_validation_pr.csv`
- Run: `.venv/bin/python scripts/diagnostics/audit_polarization_lexicons.py --config config/italy_polarization_setup.yaml`

### 4i) Polarization descriptives tables
- Script: `prepare_polarization_descriptives.py`
- Output: `results/tables/italy_polarization/descriptives/` (daily by subreddit/topic/topic_family, country panel, author retention, balanced panel, attrition)
- Run: `.venv/bin/python scripts/diagnostics/prepare_polarization_descriptives.py --config config/italy_polarization_setup.yaml`

### 4j) Polarization descriptives plots
- Script: `plot_polarization_descriptives.py`
- Output: `results/figures/italy_polarization/descriptives/daily/{by_family,by_topic,by_topic_italian,country_panel,ideology}/` and the same view tree under `descriptives/rolling_daily/` (7-day trailing past-only default)
- Run: `.venv/bin/python scripts/diagnostics/plot_polarization_descriptives.py --config config/italy_polarization_setup.yaml`
- Optional: `--rolling_window N` (days) for `rolling_daily/` figures
- Run: `.venv/bin/python scripts/diagnostics/plot_cleaning_pipeline_trends.py --config config/italy_polarization_setup.yaml`

### Stage 0 — raw data quality (no cleaning)
- Script: `plot_data_quality_trends.py`
- Why: Pre-cleaning QA on **raw** NDJSON; counts every line — does **not** remove `[removed]`/`[deleted]` bodies or deleted authors.
- Run: `.venv/bin/python scripts/diagnostics/plot_data_quality_trends.py --config config/italy_polarization_setup.yaml`

---

## Within-user pre/post (ban anchor: 2023-03-31 UTC)

Prerequisites: enriched shards with polarization + AI + style columns (stage 4 above).

- Scripts: `prepare_user_week_style_panel.py` → `analyze_user_pre_post_shift.py` → `plot_user_pre_post_shift.py`
- Pre/post split: `event_window.launch_day_utc` in Italy YAML (Italy ChatGPT ban onset).
- Composites (config `user_week`): `polarization_composite_user_week`, `ai_style_composite_user_week`
- Outputs:
  - `results/tables/italy_polarization/user_week/user_week_panel.parquet`
  - `results/tables/italy_polarization/user_week/shift_per_user_<cohort>_<style|polarization>.csv`
  - `results/figures/italy_polarization/user_week/<cohort>/<style|polarization>/`
- Run:
  - `.venv/bin/python scripts/user_week/prepare_user_week_style_panel.py --config config/italy_polarization_setup.yaml`
  - `.venv/bin/python scripts/user_week/analyze_user_pre_post_shift.py --config config/italy_polarization_setup.yaml`
  - `.venv/bin/python scripts/user_week/plot_user_pre_post_shift.py --config config/italy_polarization_setup.yaml`

---

## Optional diagnostics (Italy)

- **Raw QA (no cleaning):** `plot_data_quality_trends.py` on `data/raw/italy_polarization/daily_chunks/`
- **Dedupe after filter restart:** `dedupe_daily_chunks.py --apply`
- **Cross-forum overlap:** `user_overlap_across_forums.py`, `user_same_day_cross_forum.py`

---

## Directory layout

| Domain | Role |
|--------|------|
| [`scripts/discovery/`](discovery/) | 3-day dump profiling; apply Italian subs to config |
| [`scripts/filtering/`](filtering/) | Monthly dump → per-day NDJSON chunks |
| [`scripts/cleaning/`](cleaning/) | Dedupe, clean, screen, enrich |
| [`scripts/diagnostics/`](diagnostics/) | QA plots, polarization descriptives, overlap |
| [`scripts/features/`](features/) | Polarization, AI-use, comment-style (in-place on enriched shards) |
| [`scripts/user_week/`](user_week/) | Ban-window within-user pre/post panel and figures |
| [`scripts/devtools/`](devtools/) | Raw lexicon prep (`prepare_parallel_lexicon_raw.py`); optional v4 txt export |
| [`scripts/archive/`](archive/) | Archived AI-adoption ML, event-time, legacy user-week — see [`archive/README.md`](archive/README.md) |

Shared helpers: [`scripts/_bootstrap.py`](_bootstrap.py), [`scripts/_project_root.py`](_project_root.py)

---

## Archived AI-adoption pipeline

The Nov 2022–Apr 2023 cross-domain study (comment features, HF detectors, event-time plots) lives under [`scripts/archive/`](archive/) with config [`config/archive/ai_adoption_political_forums_setup.yaml`](../config/archive/ai_adoption_political_forums_setup.yaml). Not used by the active Italy polarization study.

---

## Short pipeline map (active study)

- External `RC_*.zst` → `filter_dump_comments.py` → `data/raw/italy_polarization/daily_chunks/`
- Raw chunks → `clean_daily_chunks.py` → `screen_subreddits.py` → `enrich_cleaned_chunks.py` → `data/interim/italy_polarization/cleaned_monthly_chunks/`
- Enriched shards → `compute_polarization_features.py` → `compute_ai_use_features.py` → `compute_comment_style_features.py` (in place)
- Enriched shards → `prepare_polarization_descriptives.py` → `plot_polarization_descriptives.py` → `results/figures/italy_polarization/descriptives/{daily,rolling_daily}/`
- Enriched shards → `prepare_user_week_style_panel.py` → `analyze_user_pre_post_shift.py` → `plot_user_pre_post_shift.py` → `results/*/italy_polarization/user_week/`

---
