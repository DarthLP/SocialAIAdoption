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
- Output: `results/tables/italy_polarization/cleaning_pipeline/`, `results/figures/italy_polarization/cleaning_pipeline/{volume,stage1_drop_rates,political_qa}/` (political audit: boxplot, subreddit bars, `by_subreddit_political_rate_vs_topic.png` scatter, `by_subreddit_political_rate_vs_thread_share.png` and `by_subreddit_political_rate_vs_comment_share.png` bubbles; screened-in only)

### 4f-bis) Political universe (comment-level scope)
- Script: `apply_political_universe.py`
- Why: Default `tree` universe (`comment_political_min_points: 2` lexical seeds + reply subtree + one-up parent; `thread_political_min_points: 3` unchanged for `thread_is_political`); frozen Mar–Apr per `link_id`; comparison modes on shards.
- Input: enriched shards with `political_weighted_points` from `data/raw/political_lexicon_parallel.csv`
- Output: `comment_in_political_universe`, `in_political_universe_*` on Parquet; stats `results/tables/italy_polarization/political_coverage/`
- Run: `.venv/bin/python scripts/features/apply_political_universe.py --config config/italy_polarization_setup.yaml`
- Compare: `political_universe_compare.py` → agreement, coverage by family, P/R vs `data/raw/political_universe_labels.csv`
- Re-run `enrich_cleaned_chunks.py --assign-only` to refresh `political_universe_share` in forum profile

### 4f–4g) Enriched-shard features (polarization, semantic axis, AI-use, style)
- Canonical: `compute_enriched_shard_features.py --pass all` (single parquet read/write per shard when all passes run).
- Wrappers (same behavior): `compute_polarization_features.py`, `compute_semantic_axis_features.py`, `compute_semantic_axis_extend.py` (append extended `sem_axis_*` only; preserves legacy scores), `compute_ai_use_features.py`, `compute_comment_style_features.py`.
- Input/output: enriched `cleaned_monthly_chunks/*.parquet` (in place).
- Parallelism: `--workers N` (default `min(8, cpu_count-1)`; `1` = sequential). Logs include per-shard `elapsed=` seconds.
- Semantic axis / `--pass all`: **language waves** (`language_waves: true`) run IT → EN → DE with a fresh ProcessPool per wave. **Exclusive cache** (`vector_cache_exclusive: true`): each worker holds at most one fastText model; switching `lex_lang` unloads others. `--lex-lang {it,en,de}` limits to one language. On ~8GB RAM use `--workers 1` (~7GB per model per worker).
- Scoring: module-level caches for `pairs_it.json` / `term_meta_it.json`; one tokenize per comment for IT polarization.
- Run (all passes):
  `.venv/bin/python scripts/features/compute_enriched_shard_features.py --config config/italy_polarization_setup.yaml --pass all --workers 8`
- Run (polarization only):
  `.venv/bin/python scripts/features/compute_polarization_features.py --config config/italy_polarization_setup.yaml --workers 8`
- Semantic axis (fastText; one-time model download):
  `.venv/bin/python scripts/devtools/download_fasttext_models.py` (or `--lang it` first)
  `.venv/bin/python scripts/devtools/export_semantic_seed_audit.py`  # Watch gate: 3 sequential fastText loads (it, en, de)
  `.venv/bin/python scripts/devtools/check_watch_seeds.py`  # Watch report only; does not rewrite seed CSVs
  `.venv/bin/python scripts/devtools/generate_semantic_axis_seed_poles.py`
  `.venv/bin/python scripts/features/compute_semantic_axis_features.py --config config/italy_polarization_setup.yaml --workers 1`
  `.venv/bin/python scripts/features/compute_semantic_axis_extend.py --config config/italy_polarization_setup.yaml --workers 1`
  `.venv/bin/python scripts/features/compute_semantic_axis_features.py --config config/italy_polarization_setup.yaml --lex-lang it --workers 1`
- Columns: `sem_axis_ideology`, `sem_axis_emotion`, `sem_axis_aggression`, `sem_axis_coverage`, `has_sem_axis`; vector cache `{interim_dir}/embeddings/<sub>/<month>.npz`
- Polarization lexicons: `ideology_it.txt` (dominant v4), `pairs_it.json`, `stance_it.txt`, `valence_it.txt`, `polarized_it.txt`; EN/DE ideology lists hand-curated
- `_polarization_score_row` copies all `POLARIZATION_COMMENT_COLUMNS` from scorer (KeyError if missing)

### 4h) Polarization lexicon audit
- Script: `audit_polarization_lexicons.py`
- Output: `results/tables/italy_polarization/descriptives/lexicon_audit_*.csv`, optional `lexicon_validation_pr.csv`
- Run: `.venv/bin/python scripts/diagnostics/audit_polarization_lexicons.py --config config/italy_polarization_setup.yaml`

### 4i-bis) Semantic-axis descriptives
- Prepare: `prepare_semantic_axis_descriptives.py` → `results/tables/italy_polarization/semantic_axis/`
- **Fast / low-RAM:** `--panels-only` (no fastText, no `body`, shard streaming). Example:
  `.venv/bin/python scripts/diagnostics/prepare_semantic_axis_descriptives.py --config config/italy_polarization_setup.yaml --panels-only`
- **Subset bins:** `--bin-days 1` or `--bin-days 1,3` (default from config: 1,3,7)
  - Optional skips: `--skip-seed-validation`, `--skip-validation`, `--skip-examples` (seed check: `validate_semantic_axis_seeds.py`; one fastText load per language)
  - **Panels** (5 levels × 3 bin sizes): `semantic_axis_panel_by_{forum,topic_family,topic,language,language_universe}_{1,3,7}d.csv`
  - **Time bins:** 1d = calendar `period_start`; 3d/7d = launch-aligned (`2023-03-31`); `n_days_in_bin`, `is_partial_bin` on all panels
  - **Axes:** seven (`ideology`, `emotion`, `aggression`, `economic`, `cultural`, `nationalism`, `anti_establishment`); pole buckets per-lexicon absolute (`*_abs`) + percentile (`above_p90` / `below_p10`); `share_unscored` (not saturated `sem_axis_coverage_mean`)
  - Calibration: `semantic_axis_lexicon_percentile_thresholds.csv`
  - Validation: `semantic_axis_validation.csv`, `semantic_axis_examples.csv`, `ideology_axis_orientation_report.csv`, seed OOV/sanity CSVs
- Seeds: `data/raw/seeds/aggression_parallel.csv` (25 aligned insult concepts × IT/EN/DE)
- Config: `pole_thresholds_by_lexicon` (all seven axes per `it`/`en`/`de`), `pole_percentiles`, `pole_cutoffs` (default `[0.25]` only), `panel_bin_days`
- **DiD:** `src/did/outcomes.py` registers all seven `sem_axis_*` for `semantic_axis`, `semantic_axis_comment`, `semantic_axis_author_day`, and `semantic_axis_author_week` (use `sem_axis_*_mean` within language for forum panels); intensity `vpn_interest_it` / `tor_*_it` only on `did_semantic_*` (not geo-matched VPN). `prepare_did_merged_panels.py` maps `us`→`US_political`, `eu`→`EU_hub_en` (all six families retained).
- Plot: `plot_semantic_axis_descriptives.py` → `results/figures/italy_polarization/semantic_axis/`:
  - `_global/` — seed OOV, score histograms, forum scatter
  - `bins_{1,3,7}d/{topic_family,topic,language,language_universe}/{timeseries,pole_shares_abs,pole_percentiles}/` — all seven axes + `share_unscored` (+ pole charts per axis)
  - `bins_{bd}d/audit/` — bin completeness, Italy `vpn_interest_it`; `bins_{bd}d/lexical_country/` — from `did_country_panel_{bd}d`
- Panels include Italy VPN/Tor (`vpn_interest_it`, `tor_bridge_users_it`, …) by `period_start` when circumvention tables exist.
- **Extended axes full rerun** (after `compute_semantic_axis_extend.py` on shards): `prepare_semantic_axis_descriptives.py` → `plot_semantic_axis_descriptives.py` → `prepare_did_subreddit_panel.py` + `prepare_did_merged_panels.py` + `prepare_did_aggregated_panels.py` → `did_event_study.py` (default families include all seven forum semantic outcomes) → `did_aggregated_event_study.py`; comment/bucket: `prepare_did_comment_panel.py --bin-days 3` → `assign_author_ideology_buckets.py --cohort strict` → `bucket_event_study.py`; author-week: `prepare_user_week_style_panel.py` → `prepare_did_author_semantic_week_panel.py` → `did_event_study.py --families semantic_axis_author_week`. Smoke: `--outcome sem_axis_economic --no-bootstrap`.

### 4i-ter) Circumvention descriptives + DiD merges
- Prepare: `prepare_circumvention_descriptives.py` → `results/tables/italy_polarization/circumvention/`
- Merge: `prepare_did_merged_panels.py` → `did/panels/country/` (lexical + geo VPN: `did_country_panel_{1,3,7}d.csv`, universe-slice variants) and `did/panels/semantic/` (`did_semantic_{topic_family,language,language_universe}_{1,3,7}d.csv`)
- Plot: `plot_circumvention_descriptives.py` → `circumvention/daily/` (VPN/Tor daily), `semantic_ideology_vs_vpn_it.png`, `circumvention/bins_{1,3,7}d/vpn_geo_levels_vs_it_broadcast.png`
- Config: `circumvention.country_panel_geo_map`, `circumvention.panel_bin_days`
- **Estimate:** `prepare_did_subreddit_panel.py` → `did/panels/subreddit/`; optional `prepare_did_comment_panel.py` → `did/panels/comment/` (partitioned comment Parquet + `did_author_day_panel_1d.csv`) → `scripts/analysis/did_event_study.py` (TWFE DiD via `linearmodels`; comment-level via `pyfixest` author+calendar FE; strategies include `italy_only_post` entity-FE-only on IT forums/comments; `did.author_wordfish_spec` / `--author-spec week3` for author-bin robustness; `did_summary.spec` includes `full_ban`, early-ban, and post-phase windows per `did.post_phases`) → `did/estimates/summary/` and `did/estimates/{family}/`; figures under `results/figures/italy_polarization/did/{family}/` with per-folder `README.md`. **Inference:** `inference_role` on summary rows; cross-country forum-clustered p is descriptive; **`p_placebo_space` / `perm_p` only for pooled multi-control strategies** (`cross_country_all`, IT-family splits, etc.) via `is_placebo_in_space_eligible_strategy`; **`cross_country_vs_{de,eu,uk,us}` get `placebo_note=not_applicable_single_country_contrast`** (patch: `scripts/diagnostics/patch_did_inference.py --placebo-only`). Headline `p_placebo_space` + `scripts/analysis/run_did_gsynth.py`; within-Italy/author restricted WCB (`wildboottest`, default 9999 draws).
- **Migrate** flat `did/*.csv`: `scripts/devtools/migrate_did_table_layout.py` (`--dry-run` optional)
- Run: `.venv/bin/python scripts/diagnostics/prepare_did_subreddit_panel.py --config config/italy_polarization_setup.yaml` then `.venv/bin/python scripts/analysis/did_event_study.py --config config/italy_polarization_setup.yaml` (add `--no-bootstrap` for a fast pass; `--outcomes wf_change,wf_extremity_z` to filter outcome ids; `--full-coefplots` for all strategies per outcome; `--figures-only` to rebuild plots from `did_summary.csv` without re-estimating)
- **Comment DiD:** after enriched shards + `--pass all`: `.venv/bin/python scripts/diagnostics/prepare_did_comment_panel.py --config config/italy_polarization_setup.yaml` then `.venv/bin/python scripts/analysis/did_event_study.py --families lexical_comment,semantic_axis_comment,lexical_author_day,semantic_axis_author_day --no-bootstrap` (smoke: `--max-shards 2` on prepare; `--comment-sample-frac 0.05` on estimate). **Italy-only forum:** `italy_only_post` included in default subreddit families. **Author bins:** `--author-spec week3` robustness pass.
- **Bucket-then-comment event study** (polarization decomposition): **two stratifications** — **lexical** (same asymmetric L/R + `net_ideology` rule as `user_week`, applied to each scheme’s March labeling-window comments) and **semantic** (pre-ban user-week `semantic_bucket` from `assign_author_ideology_buckets.py`, replicated across schemes). Run `prepare_did_comment_panel.py --bin-days 3` once (includes lexical rate columns: `aggression_rate_100w`, `negative_rate_100w`, `extremity`), then `assign_author_ideology_buckets.py --cohort strict`, then `scripts/analysis/bucket_event_study.py`. Outputs: `did/lean_buckets/` (lexical) + `did/lean_buckets_semantic/` + `did/bucket_event_study/{1,3}d/` (legacy root = lexical + `net_ideology`) + nested `strat_lexical/<outcome>/` and `strat_semantic/<outcome>/` + matching figure trees. Config: `did.bucket_event_study` (`label_outcome`, `bucket_stratifications`, `additional_outcomes` for semantic axes and lexical rates). CLI: `--stratification lexical|semantic`, `--outcome <col>`. **`split_sample` cross-fit:** `split_method: random` with `n_splits: 5`. **Table 1 static (headline):** `y ~ post + post:IT | author`. **Event study:** `i(rel_period, IT)` with `author + time_id` FE. **Inference:** placebo-in-space on headline static rows for `all_controls_pooled` only. Smoke: `--max-shards 2 --scheme holdout_2wk --no-bootstrap --no-figures --stratification lexical --outcome net_ideology`. **8 GB RAM:** one stratification × outcome at a time.
- Headline event-study overlays: `scripts/analysis/did_event_study_plots.py` → `did/event_study_{outcome}.csv` and `results/figures/.../did/event_study/*.png`
- Overview-only replot: `scripts/diagnostics/plot_did_overview.py`
- Lexical control heterogeneity: `scripts/diagnostics/plot_did_lexical_by_control.py` → `did/lexical/by_control_choice.png`
- Ban-window descriptives: `scripts/diagnostics/plot_descriptives_ban_shaded.py` → `descriptives/ban_window/`
- Wordfish v1 vs v2 θ scatter: `scripts/diagnostics/plot_wordfish_v1_vs_v2.py`
- DiD guards: `degenerate_collinear` and `insufficient_panel` in `src/did/estimate.py`; `pretrend_F_p` Wald test on leads \(k\in\{-3,-2,-1\}\)
- Thesis handoff prose: `results/notes/context_updates_for_thesis_repo.md`
- **Aggregated event studies** (1d ±14d / 3d ±30d; lexical + semantic + forum Wordfish v1/v2): `prepare_semantic_axis_descriptives.py --panels-only` (pole_share, Esteban–Ray, p10/p90 tails on forum panel) → `prepare_did_aggregated_panels.py` + `prepare_did_subreddit_panel.py` → `scripts/analysis/did_aggregated_event_study.py` (`--bundle`, `--panel-level`, `--figures-only`). Dual-tail ideology figures: `sem_axis_ideology_tail_shift.png` (and `..._in_tree` / `..._out_tree` for `language_universe/in_out_slice`) under each bundle × `{1,3}d`. Stale pre-bundle dirs (`language/1d`, `language_universe/1d`, `topic_family/1d`, etc.) are removed automatically after a full run. Figure tree (migration: old `event_study/language/1d/` → `event_study/language/subreddit/1d/`):
  - `topic_family/overlay_pooled/` — five-series overlay on family panel (entity = topic_family; ~6 clusters).
  - `topic_family/it_political/`, `topic_family/it_others/` — single IT-family vs controls (`plot_event_study`).
  - `language/subreddit/` — five-series overlay on subreddit TWFE (3d bins outcomes via `bin_lexical_daily_panel`; entity = subreddit).
  - `language/hub_pooled/` — five-series overlay on `language_hub` aggregated panel (~5 clusters).
  - `language_universe/in_out_slice/` — two-series overlay (`cross_country_political_universe_in` / `_out`) on subreddit×`universe_slice` panel.
  - Error bars: **±1.96 × SE** from entity-clustered TWFE (`cluster_entity=True`), not comment-level SD. **Degenerate series** (non-finite SE, e.g. single-entity `cross_country_vs_us` on hub panel) are skipped at export/plot (`_event_study_series_usable`; overlay drops series with &lt;2 finite SE).
  - CSVs: `did/estimates/{family}/event_study/{panel_level}/{bundle}/{bin}d/{strategy_id}/{outcome}.csv`
- Wordfish v2 DiD runs when `wordfish_forum_v2/wordfish_extremity_panel.csv` and `wordfish_authors_v2/wordfish_authors_extremity_panel.csv` exist; otherwise v2 families are skipped with a log message

### 4i) Polarization descriptives tables
- Script: `prepare_polarization_descriptives.py`
- Output: `results/tables/italy_polarization/descriptives/` (daily by subreddit/topic/topic_family, country panel, author retention, balanced panel, attrition)
- Universe slices (requires `comment_in_political_universe` on shards): `daily_country_panel_by_universe_slice.csv`, `daily_italy_all_by_universe_slice.csv`, `daily_it_political_by_universe_slice.csv`, `daily_it_others_by_universe_slice.csv` (`universe_slice`: `in_political_tree` / `out_political_tree`; includes `share_of_panel_comments`)
- Run: `.venv/bin/python scripts/diagnostics/prepare_polarization_descriptives.py --config config/italy_polarization_setup.yaml`

### 4j) Polarization descriptives plots
- Script: `plot_polarization_descriptives.py`
- Output: `results/figures/italy_polarization/descriptives/daily/{by_family,by_topic,by_topic_italian,country_panel,ideology}/` and the same view tree under `descriptives/rolling_daily/` (7-day trailing past-only default)
- Dual-universe overlays (thick in-tree, translucent non-tree): `descriptives/{daily,rolling_daily}/{country_panel_dual_universe,italy_all_dual_universe,italy_it_political_dual_universe,italy_it_others_dual_universe}/`
- Run: `.venv/bin/python scripts/diagnostics/plot_polarization_descriptives.py --config config/italy_polarization_setup.yaml`
- Optional: `--rolling_window N` (days) for `rolling_daily/` figures
- Run: `.venv/bin/python scripts/diagnostics/plot_cleaning_pipeline_trends.py --config config/italy_polarization_setup.yaml`

### 4k) Wordfish robustness (prompt 03)
- **Stopwords (one-time):** `scripts/devtools/generate_wordfish_stopwords.py` → `config/lexicons/stopwords_{it,en,de}.txt` (de for 03b-authors; **de not fitted** here)
- **Prepare:** `prepare_wordfish.py` — four fits (`it`/`en` × `day`/`week`), event bins anchored at `2023-03-31`; German excluded from fits; adds `change`/`change_z` (rolling prior extremity, W from `change_window_days[0]`), placebo flags (`placebo_launch_date`), `date_utc` on day rows, `wordfish_placebo_window_summary.csv`
- **Plot:** `plot_wordfish.py` → `results/figures/italy_polarization/wordfish/` including `extremity_timeseries_by_family.png` (day-primary; IT vs EN panels) and `axis_words_{it,en}.png` aliases
- Tables: `results/tables/italy_polarization/wordfish/` (`wordfish_extremity_panel.csv` for prompt 04 DiD/ES; `wordfish_axis_words_{lang}.csv` day-primary copies; dispersion descriptive only)
- Prerequisites: `apply_political_universe.py`, stage-4 features on shards
- Run:
  - `.venv/bin/python scripts/devtools/generate_wordfish_stopwords.py`
  - `.venv/bin/python scripts/diagnostics/prepare_wordfish.py --config config/italy_polarization_setup.yaml`
  - `.venv/bin/python scripts/diagnostics/plot_wordfish.py --config config/italy_polarization_setup.yaml`

### 4l) Wordfish authors (prompt 03b)
- **Prepare:** `prepare_wordfish_authors.py` — author×bin documents; `it`/`en`/`de` fits; dual `full` / `balanced` panels per `week7`/`week3`/`window` spec; ban-anchored bins; `it > de > en` assignment; `change`/`change_z` (`rolling_bins_w`); headline `balanced_week7` copied to `wordfish_authors_extremity_panel.csv` for prompt **04** (TWFE/ES/placebo regressions run in 04, not here). Full runs pool languages into `wordfish_authors_extremity_panel_{tag}.csv`; `--language it` (etc.) writes only `_{tag}_{lang}.csv` and leaves pooled/headline files unchanged.
- **Plot:** `plot_wordfish_authors.py` → `results/figures/italy_polarization/wordfish_authors/`
- Tables: `results/tables/italy_polarization/wordfish_authors/` (`wordfish_authors_extremity_panel_{mode}_{spec}.csv`, assignment audit, validation, stability, `wordfish_authors_run_notes.txt`)
- Config: `wordfish_authors` in `config/italy_polarization_setup.yaml`
- Run:
  - `.venv/bin/python scripts/diagnostics/prepare_wordfish_authors.py --config config/italy_polarization_setup.yaml`
  - `.venv/bin/python scripts/diagnostics/plot_wordfish_authors.py --config config/italy_polarization_setup.yaml`
  - Optional: `--spec week7 --panel-mode balanced --language it`; `--drop-cross-language` for robustness

### 4k-bis) Wordfish v2 (validity pass; legacy paths unchanged)
- **Authors v2 (primary ideology attempt):** `prepare_wordfish_authors_v2.py` → `wordfish_authors_v2/` — `fit_wordfish_v2`, 8k token cap, author-level `wordfish_authors_validation_gate.csv`; EN `split_us_uk` (`en_us` / `en_uk` fits); per-tag positions/extremity/dispersion/coverage/validation CSVs are concatenated across languages (`primary_lexicon` = `it`/`en`/`de` on panels). **Performance:** two shard passes (pass1+ideology, then all specs/languages in one body pass) with column-pruned parquet reads; `--reuse-assignment` skips pass1 lexicon scan when `wordfish_authors_assignment.csv` exists; BLAS pinned to 1 thread during fits (lower heat).
- **Forum v2 (Tier B / labels):** `prepare_wordfish_forum_v2.py` → `wordfish_forum_v2/` — shard `topic_family` preserved, token cap; θ not validated as ideology
- **Plot:** `plot_wordfish_authors_v2.py`, `plot_wordfish_forum_v2.py`
- Config: `wordfish_authors_v2`, `wordfish_forum_v2` in `config/italy_polarization_setup.yaml`
- Run:
  - `.venv/bin/python scripts/diagnostics/prepare_wordfish_authors_v2.py --config config/italy_polarization_setup.yaml`
  - Resume after assignment: add `--reuse-assignment` (re-scans ideology + bodies only)
  - `.venv/bin/python scripts/diagnostics/prepare_wordfish_forum_v2.py --config config/italy_polarization_setup.yaml`

### Stage 0 — raw data quality (no cleaning)
- Script: `plot_data_quality_trends.py`
- Why: Pre-cleaning QA on **raw** NDJSON; counts every line — does **not** remove `[removed]`/`[deleted]` bodies or deleted authors.
- Run: `.venv/bin/python scripts/diagnostics/plot_data_quality_trends.py --config config/italy_polarization_setup.yaml`

---

## Within-user pre/post (ban anchor: 2023-03-31 UTC)

Prerequisites: enriched shards with polarization + AI + style columns (stage 4 above).

- Scripts: `prepare_user_week_style_panel.py` → `analyze_user_pre_post_shift.py` → `plot_user_pre_post_shift.py` → `estimate_user_week_panel.py` → `plot_user_week_event_study.py` → `plot_user_pole_decomposition.py` → `plot_user_lexical_by_lexicon.py` → `plot_user_semantic_by_lexicon.py` → `plot_user_week_overview.py` → `assign_author_ideology_buckets.py` → `compare_lexical_semantic_author_buckets.py` → `plot_user_shift_by_ideology_bucket.py`
- Pre/post split: `event_window.launch_day_utc` in Italy YAML (Italy ChatGPT ban onset).
- Composites (config `user_week`): `polarization_composite_user_week`, `ai_style_composite_user_week`, `semantic_composite_user_week`
- Default features include pole margins, `sem_axis_*` weekly means, and `share_scored` (semantic coverage QA). Requires enriched shards with semaxis pass (`compute_enriched_shard_features.py --pass all`).
- **Author semantic DiD (cross-country):** after user-week panel → `prepare_did_author_semantic_week_panel.py` → `did_event_study.py --families semantic_axis_author_week` (joins `wordfish_authors_assignment.csv`).
- Outputs:
  - `results/tables/italy_polarization/user_week/user_week_panel.parquet` (`sem_axis_*_mean`, `share_scored`, `pole_share`)
  - `results/tables/italy_polarization/user_week/shift_per_user_<cohort>_<style|polarization|semantic>.csv`
  - `results/tables/italy_polarization/user_week/regression_summary_<cohort>.csv` (author FE: `y ~ post`, cluster author)
  - `results/tables/italy_polarization/user_week/event_study_<cohort>_{net_ideology|sem_axis_ideology}.csv`
  - `results/figures/italy_polarization/user_week/<cohort>/<style|polarization|semantic>/` (auto `README.md` via `src/user_week/figure_readmes.py`)
  - `results/figures/italy_polarization/user_week/<cohort>/polarization|semantic/by_primary_lexicon/` (descriptive; not causal IT vs control)
  - `results/figures/italy_polarization/user_week/<cohort>/event_study/`, `.../overview/`, `.../pole_decomposition/`
  - `results/tables/italy_polarization/user_week/author_ideology_buckets_<cohort>.csv` (pre-ban **asymmetric** buckets: lexical neutral if no L/R hits; semantic neutral if no p25/p75 tail weeks (config `tail_percentiles`); `semantic_bucket_mag_band` for diagnostics)
  - Prerequisite for assignment: `results/tables/italy_polarization/semantic_axis/semantic_axis_lexicon_percentile_thresholds.csv` from `prepare_semantic_axis_descriptives.py`
  - `results/tables/italy_polarization/user_week/ideology_bucket_agreement_<cohort>/` (marginals, tail vs mag_band κ/confusion, pole-only agreement)
  - `results/figures/italy_polarization/user_week/<cohort>/ideology_bucket_agreement/` and `.../by_ideology_bucket/` (shift violins: semantic ideology/emotion/aggression + lexical net_ideology/pole_share/aggression_rate/negative_rate, grouped by lexical_bucket and semantic_bucket)
  - `results/tables/italy_polarization/did/panels/author/did_author_semantic_week_panel.csv`
- Run (full panel; do not use `--max_total_month_files` for production):
  - `.venv/bin/python scripts/user_week/prepare_user_week_style_panel.py --config config/italy_polarization_setup.yaml`
  - `.venv/bin/python scripts/user_week/analyze_user_pre_post_shift.py --config config/italy_polarization_setup.yaml`
  - `.venv/bin/python scripts/user_week/plot_user_pre_post_shift.py --config config/italy_polarization_setup.yaml`
  - `.venv/bin/python scripts/user_week/estimate_user_week_panel.py --config config/italy_polarization_setup.yaml --cohort both`
  - `.venv/bin/python scripts/user_week/plot_user_week_event_study.py --config config/italy_polarization_setup.yaml`
  - `.venv/bin/python scripts/user_week/plot_user_pole_decomposition.py --config config/italy_polarization_setup.yaml`
  - `.venv/bin/python scripts/user_week/plot_user_lexical_by_lexicon.py --config config/italy_polarization_setup.yaml`
  - `.venv/bin/python scripts/user_week/plot_user_semantic_by_lexicon.py --config config/italy_polarization_setup.yaml`
  - `.venv/bin/python scripts/user_week/plot_user_week_overview.py --config config/italy_polarization_setup.yaml --cohort strict`
  - `.venv/bin/python scripts/diagnostics/prepare_semantic_axis_descriptives.py --config config/italy_polarization_setup.yaml --panels-only` (if percentile thresholds missing)
  - `.venv/bin/python scripts/user_week/assign_author_ideology_buckets.py --config config/italy_polarization_setup.yaml --cohort both`
  - `.venv/bin/python scripts/user_week/compare_lexical_semantic_author_buckets.py --config config/italy_polarization_setup.yaml --cohort both`
  - `.venv/bin/python scripts/user_week/plot_user_shift_by_ideology_bucket.py --config config/italy_polarization_setup.yaml`
  - `.venv/bin/python scripts/diagnostics/prepare_did_author_semantic_week_panel.py --config config/italy_polarization_setup.yaml`
  - `.venv/bin/python scripts/analysis/did_event_study.py --config config/italy_polarization_setup.yaml --families semantic_axis_author_week`

---

## First-stage inference upgrades (Italy DiD)

Run after `prepare_did_subreddit_panel.py` (and enriched shards for style index).

| Script | Purpose |
|--------|---------|
| `scripts/analysis/did_event_study.py --weights n_comments` | Comment-count weighted TWFE → `did/estimates_weighted/` |
| `scripts/analysis/placebo_in_time.py` | Fixed 7d placebo-in-time + RI-t; `placebo_in_time.csv` |
| `scripts/analysis/first_stage_mde.py` | MDE = 2.8×SE from saved `by_outcome/*.csv` |
| `scripts/diagnostics/fit_style_index_stats.py` | Pre-period clip bounds → `did/style_index_stats.json` |
| `scripts/features/compute_style_index_on_shards.py` | Persist `style_index_full` / `style_index_reduced` on shards |
| `scripts/diagnostics/validate_style_index_gates.py` | Histogram, Spearman vs `ai_style_rate_100w`, pretrend F, 20+20 review CSV |
| `scripts/analysis/prepare_adopter_flags.py` | Schemes 1–3 flags (thresholds **within country**) |
| `scripts/analysis/adopter_ddd.py` | Triple-diff `post×flag`, `post×IT×flag` with `author` + `topic_family×date` FE |

Descriptive date-placebo figures from `did_event_study.py` robustness grid are **not** permutation tests; do not merge into `placebo_in_time.csv`.

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

## External circumvention proxies (VPN + Tor)

- Script: `download_circumvention_data.py` (repo root `scripts/`, not domain subfolder)
- Why: Tor Metrics daily relay/bridge users + Google Trends **topic** “Virtual private network” for IT + DE/FR/ES/GB/US around the ChatGPT ban (Kreitmeir & Raschky 2023).
- Output: `data/raw/circumvention/` (`tor/`, `google_trends/`, combined `tor_*_users_by_country.csv`, `google_trends_vpn_by_country.csv`, `_manifest.json`, `README.md`)
- Run: `.venv/bin/python scripts/download_circumvention_data.py`
- Note: `data/` is gitignored; re-run is idempotent. If Google rate-limits (429), export CSV manually into `google_trends/` and re-run.

---

## Short pipeline map (active study)

- External `RC_*.zst` → `filter_dump_comments.py` → `data/raw/italy_polarization/daily_chunks/`
- Raw chunks → `clean_daily_chunks.py` → `screen_subreddits.py` → `enrich_cleaned_chunks.py` → `data/interim/italy_polarization/cleaned_monthly_chunks/`
- Enriched shards → `compute_polarization_features.py` → `compute_semantic_axis_features.py` → `compute_ai_use_features.py` → `compute_comment_style_features.py` (in place; or `--pass all`)
- One-time: `scripts/devtools/download_fasttext_models.py`; `scripts/devtools/generate_semantic_axis_seed_poles.py` (after editing `data/raw/seeds/aggression_parallel.csv`, re-run to refresh `poles/aggression_pos_*.txt`, 25 terms each)
- Seed validation (no shards): `scripts/diagnostics/validate_semantic_axis_seeds.py` → `semantic_axis_seed_coverage.csv`, `semantic_axis_axis_sanity.csv`
- Enriched shards → `prepare_polarization_descriptives.py` → `plot_polarization_descriptives.py` → `results/figures/italy_polarization/descriptives/{daily,rolling_daily}/`
- `download_circumvention_data.py` → `prepare_circumvention_descriptives.py` → `prepare_did_merged_panels.py` (with polarization + semantic panels) → `prepare_did_subreddit_panel.py` → `did_event_study.py` → `plot_circumvention_descriptives.py`
- Enriched shards → `prepare_user_week_style_panel.py` → `analyze_user_pre_post_shift.py` → `plot_user_pre_post_shift.py` → `results/*/italy_polarization/user_week/`

---
