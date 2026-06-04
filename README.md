# SocialAIAdoption

## Objective

Thesis study: **The Effects of AI Access on Online Political Polarization** — Reddit comment corpora around Italy’s ChatGPT access restriction (**March–April 2023** UTC). The active pipeline discovers Italian-language subreddits from a **3-day March 2023** screen, then extracts a fixed comparison set plus approved Italian communities into local NDJSON day chunks.

**Active config:** [`config/italy_polarization_setup.yaml`](config/italy_polarization_setup.yaml)

**Comparison forums (fixed):**

- **English political (discussion-style, not r/politics):** `Ask_Politics`, `NeutralPolitics`, `PoliticalDiscussion`, `moderatepolitics`
- **EU hubs:** `de`, `unitedkingdom`, `europe`
- **EU political:** `ukpolitics`
- **Italian:** data-driven discovery + seeds (`Italia`, `politicaITA`)

**Archived AI-adoption corpus:** [`config/archive/ai_adoption_political_forums_setup.yaml`](config/archive/ai_adoption_political_forums_setup.yaml) (Nov 2022–Apr 2023 cross-domain study). Legacy ML and event-time scripts live under [`scripts/archive/`](scripts/archive/README.md) only.

---

## Quick Start (extraction milestone)

Prerequisites: `.venv` with `pip install -r requirements.txt`; external dumps `RC_2023-03.zst` and `RC_2023-04.zst` under `/Volumes/Expansion/Masterthesis/RawData/reddit/comments/` (or your `--source_dir`).

### 1. Required backup to external disk (before deleting local legacy data)

```bash
ARCHIVE="/Volumes/Expansion/Masterthesis/SocialAIAdoption_archive_$(date +%Y%m%d)"
mkdir -p "$ARCHIVE/data" "$ARCHIVE/results"
rsync -a data/raw/ "$ARCHIVE/data/raw/"
rsync -a data/interim/ "$ARCHIVE/data/interim/"
rsync -a results/ "$ARCHIVE/results/"
# Verify: compare ndjson counts (expect ~12k+ under data/raw/.../daily_chunks)
find "$ARCHIVE/data/raw" -name '*.ndjson' | wc -l
```

Use `data/` and `results/` subfolders on the archive so files are not mixed at the top level. **Do not delete** anything on the expansion drive’s `RawData/` tree.

**exFAT note:** The expansion drive is exFAT (4 GB max per file). Do not use a single `data_raw.tar` for the full legacy corpus (~4 GB+). Prefer:

- `raw_italy_chatgpt_ban.tar` (small subtree)
- `results.tar`
- `rsync -a --ignore-errors data/raw/political_forums/ "$ARCHIVE/data/raw/political_forums/"` for NDJSON day chunks

### 2. Italian subreddit discovery (first 3 UTC days of March 2023 only)

```bash
.venv/bin/python scripts/discovery/profile_subreddits_in_dump.py \
  --config config/italy_polarization_setup.yaml \
  --source_dir "/Volumes/Expansion/Masterthesis/RawData/reddit/comments"
```

Outputs under `results/tables/italy_polarization/discovery/`:

- `subreddit_census_3d.csv` — `n_comments_first_3d` per subreddit
- `candidate_italian_subreddits.csv` — Italian langid candidates with `projected_comments_mar_apr`
- `extraction_size_preview.csv` — controls + seeds + candidates with size projections
- `discovery_run_notes.txt`

Review `extraction_size_preview.csv` (drop subs that project too large if needed).

### 3. Lock subreddit list in config

```bash
.venv/bin/python scripts/discovery/apply_discovery_to_config.py \
  --config config/italy_polarization_setup.yaml
```

### 4. Full Mar–Apr extract to local NDJSON

```bash
.venv/bin/python scripts/filtering/filter_dump_comments.py \
  --config config/italy_polarization_setup.yaml \
  --source_dir "/Volumes/Expansion/Masterthesis/RawData/reddit/comments" \
  # state/log default to results/logs/italy_polarization/filter_dump/italy_polarization_state.json (and .log)
```

Writes: `data/raw/italy_polarization/daily_chunks/<subreddit>/<YYYY-MM-DD>.ndjson`

Optional after interrupt: `.venv/bin/python scripts/cleaning/dedupe_daily_chunks.py --config config/italy_polarization_setup.yaml --apply`

---

## Cleaning and taxonomy pipeline (after extract)

Run in order (`.venv` active):

```bash
# Stage 1 — row drops (moderation placeholders, URL-only spam); keeps [deleted] authors
.venv/bin/python scripts/cleaning/clean_daily_chunks.py --config config/italy_polarization_setup.yaml

# Stage 2 — forum gates (profile subs, URL-only forums, pooled Italian langid ≥70%, volume bands)
.venv/bin/python scripts/cleaning/screen_subreddits.py --config config/italy_polarization_setup.yaml

# Refresh raw parallel lexicons (merge ideology_parallel, export style phrases)
.venv/bin/python scripts/devtools/prepare_parallel_lexicon_raw.py --gap-report

# Stage 3 — taxonomy columns, language-matched political lexicon, thread roll-ups
# (on success, auto-runs plot_cleaning_pipeline_trends.py unless --skip-pipeline-plots)
.venv/bin/python scripts/cleaning/enrich_cleaned_chunks.py --config config/italy_polarization_setup.yaml

# Lexicon QA + pipeline diagnostics
.venv/bin/python scripts/diagnostics/audit_political_lexicon.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/diagnostics/plot_cleaning_pipeline_trends.py --config config/italy_polarization_setup.yaml

# Political universe (tree default; Mar–Apr cross-month per subreddit; after enrich, before or after stage 4)
.venv/bin/python scripts/features/apply_political_universe.py --config config/italy_polarization_setup.yaml
# Refresh forum profile column political_universe_share (or full enrich --assign-only after universe pass)
.venv/bin/python scripts/cleaning/enrich_cleaned_chunks.py --config config/italy_polarization_setup.yaml --assign-only
.venv/bin/python scripts/diagnostics/political_universe_compare.py --config config/italy_polarization_setup.yaml

# Stage 4 — polarization + AI + style features (run only AFTER enrich finishes on all shards)
# --workers: parallel shards (default min(8, cpu_count-1)); --pass all uses one parquet read/write per shard
.venv/bin/python scripts/features/compute_enriched_shard_features.py \
  --config config/italy_polarization_setup.yaml --pass all --workers 8
# Or single pass:
.venv/bin/python scripts/features/compute_polarization_features.py \
  --config config/italy_polarization_setup.yaml --workers 8

# Semantic axis (requires fastText .bin models; download once)
.venv/bin/python scripts/devtools/download_fasttext_models.py
.venv/bin/python scripts/devtools/export_semantic_seed_audit.py   # issue seeds from audit xlsx (~3 fastText loads for Watch gate)
.venv/bin/python scripts/devtools/check_watch_seeds.py   # Watch-only fastText report (no CSV rewrite; same 3 loads)
.venv/bin/python scripts/devtools/generate_semantic_axis_seed_poles.py   # optional pole txt under data/raw/seeds/poles/
.venv/bin/python scripts/diagnostics/validate_semantic_axis_seeds.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/features/compute_semantic_axis_features.py \
  --config config/italy_polarization_setup.yaml --workers 1
# Extended issue axes (economic, cultural, nationalism, anti_establishment) on shards that already have sem_axis_ideology/emotion/aggression:
.venv/bin/python scripts/features/compute_semantic_axis_extend.py \
  --config config/italy_polarization_setup.yaml --workers 1
# Watch / seed validation: one ~7GB fastText model at a time (it → en → de); in-vocab is embedding coverage, not the polarization lexicon CSV.
# Language waves (default): all IT shards, then EN, then DE; ProcessPool restarts between waves.
# Exclusive cache (vector_cache_exclusive): one ~7GB model per worker, not IT+EN+DE stacked.
# Low RAM (~8GB): --workers 1; or one language: --lex-lang it

.venv/bin/python scripts/features/compute_ai_use_features.py --config config/italy_polarization_setup.yaml --workers 8
.venv/bin/python scripts/features/compute_comment_style_features.py --config config/italy_polarization_setup.yaml --workers 8

# Preflight: one shard must contain net_ideology and semicolon_count
.venv/bin/python -c "import pandas as pd; from pathlib import Path; p=next(Path('data/interim/italy_polarization/cleaned_monthly_chunks').rglob('*.parquet')); d=pd.read_parquet(p); assert 'net_ideology' in d.columns and 'semicolon_count' in d.columns"

# Optional — within-user pre/post (author × ISO week) on enriched shards (needs semaxis on shards)
.venv/bin/python scripts/user_week/prepare_user_week_style_panel.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/user_week/analyze_user_pre_post_shift.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/user_week/plot_user_pre_post_shift.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/user_week/estimate_user_week_panel.py --config config/italy_polarization_setup.yaml --cohort both
.venv/bin/python scripts/user_week/plot_user_week_event_study.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/user_week/plot_user_pole_decomposition.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/user_week/plot_user_lexical_by_lexicon.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/user_week/plot_user_semantic_by_lexicon.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/user_week/plot_user_week_overview.py --config config/italy_polarization_setup.yaml --cohort strict
.venv/bin/python scripts/diagnostics/prepare_semantic_axis_descriptives.py --config config/italy_polarization_setup.yaml --panels-only  # percentile thresholds for ideology buckets
.venv/bin/python scripts/user_week/assign_author_ideology_buckets.py --config config/italy_polarization_setup.yaml --cohort both
.venv/bin/python scripts/user_week/compare_lexical_semantic_author_buckets.py --config config/italy_polarization_setup.yaml --cohort both
.venv/bin/python scripts/user_week/plot_user_shift_by_ideology_bucket.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/diagnostics/prepare_did_author_semantic_week_panel.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/analysis/did_event_study.py --config config/italy_polarization_setup.yaml --families semantic_axis_author_week

.venv/bin/python scripts/diagnostics/audit_polarization_lexicons.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/diagnostics/prepare_polarization_descriptives.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/diagnostics/plot_polarization_descriptives.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/diagnostics/prepare_semantic_axis_descriptives.py --config config/italy_polarization_setup.yaml
# Fast path (recommended on laptops): add --panels-only [--bin-days 1]
# Panels: by_forum|topic_family|topic|language|language_universe × 1d|3d|7d (see results/README.md)
# DiD on semantic means (within-language); pole shares are per-lexicon calibrated — see scripts/README §4i-bis
.venv/bin/python scripts/diagnostics/plot_semantic_axis_descriptives.py --config config/italy_polarization_setup.yaml
# Figures: results/figures/italy_polarization/semantic_axis/bins_{1,3,7}d/ (organized by level and chart type)
.venv/bin/python scripts/diagnostics/plot_circumvention_descriptives.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/diagnostics/prepare_circumvention_descriptives.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/diagnostics/prepare_did_merged_panels.py --config config/italy_polarization_setup.yaml

# Wordfish robustness (after political universe + stage 4)
.venv/bin/python scripts/devtools/generate_wordfish_stopwords.py
.venv/bin/python scripts/diagnostics/prepare_wordfish.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/diagnostics/plot_wordfish.py --config config/italy_polarization_setup.yaml
# Headline panel for prompt 04: wordfish_extremity_panel.csv (extremity_z, change_z, placebo flags; day-primary time_bin)
# Wordfish v2 (additive; legacy wordfish/ + wordfish_authors/ unchanged):
.venv/bin/python scripts/diagnostics/prepare_wordfish_authors_v2.py --config config/italy_polarization_setup.yaml
# Optional: --reuse-assignment after wordfish_authors_assignment.csv exists (skip pass1 lexicon scan)
.venv/bin/python scripts/diagnostics/plot_wordfish_authors_v2.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/diagnostics/prepare_wordfish_forum_v2.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/diagnostics/plot_wordfish_forum_v2.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/diagnostics/prepare_wordfish_authors.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/diagnostics/plot_wordfish_authors.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/diagnostics/plot_circumvention_descriptives.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/diagnostics/prepare_lexicon_descriptives.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/diagnostics/plot_lexicon_descriptives.py --config config/italy_polarization_setup.yaml
```

Stage 4 adds columns **in-place** on enriched Parquet: polarization (`net_ideology` from `polarization_lexicon_parallel.csv`, pair framing `pair_framing_*` from v4 pairs CSV, emotion/cognition rates, …), **semantic axes** (`sem_axis_ideology`, `sem_axis_emotion`, `sem_axis_aggression`, plus extended `sem_axis_economic`, `sem_axis_cultural`, `sem_axis_nationalism`, `sem_axis_anti_establishment` from fastText + `data/raw/seeds/`), AI lexicon, and style counts from `style_phrase_parallel.csv`. All shards share the same column schema (Italian pair framing only on `it` shards). Config requires `polarization.ideology_scoring: dominant_v1`.

Lexicon descriptives (pair framing, stance, valence) live under `results/figures/italy_polarization/descriptives/{primary,ideology_dominant,pairs,stance,valence,polarized,trajectory_scatter}/`. Legacy polarization descriptives remain under `descriptives/daily/` and `descriptives/rolling_daily/` (all comments). **Universe-sliced** tables (`daily_*_by_universe_slice.csv`) and dual-line figures (`*_dual_universe/`) overlay **in political tree** (thick) vs **non-tree** (translucent) per country panel, pooled Italy, and `it_political` / `it_others` separately. Re-running **enrich** after stage 4 removes feature columns — run stage 4 again if that happens.

**Results layout:** see [`results/README.md`](results/README.md).

**Pre-registered thresholds** (`config/italy_polarization_setup.yaml` → `screening`):

- URL-only rows dropped at Stage 1; forums with ≥80% URL-only input rows excluded.
- Italian arms: pooled Mar–Apr langid ≥ **70%** (up to 500 comments/month sampled).
- **large_volume**: ≥ **100** kept comments over the window; **low_volume** if only `LOW_VOLUME_WINDOW`; soft monthly floor **50** for sparse-month flags.
- `r/europe` treated as **general English** (`primary_lexicon: en`).

**Topic / family taxonomy:** Italian **topics** `it_political` / `it_pure_political` / `it_others` from graded word-weighted rate (WW); **family** `it_political` pools both political topics. Controls: `de`, `eu`, `us`, `uk` (`uk`, `uk_political`). **Salience:** [`data/raw/political_lexicon_parallel.csv`](data/raw/political_lexicon_parallel.csv) (grades 1–3 → points 1/2/3 per unique term; IT/EN/DE columns). **Assignment** (Italian arms): metadata overrides → controls → WW ≥ `forum_political_pure_threshold` → `it_pure_political` → WW ≥ `forum_political_soft_threshold` → `it_political` → else `it_others` (recalibrate thresholds after enrichment; current **0.6** / **1.2**). **Headline political scope:** share of comments in the **political universe** (`political_universe.mode: tree` by default — lexical seed with `comment_political_min_points: 2` on the comment body, reply subtree, optional one-up parent; frozen over Mar–Apr per `link_id`). Column: `comment_in_political_universe`; forum audit: `political_universe_share`. **Thread political flag** (comparison, separate threshold): `thread_political_weighted_points >= 3` (`screening.thread_political_min_points`) per monthly shard (`thread_is_political`); `thread_political_share` retained. P/R labels: `data/raw/political_universe_labels.csv`. Coverage tables/figures: `results/.../political_coverage/`.

**Outputs:**

| Stage | Interim | Tables / figures |
|-------|---------|------------------|
| 1 | `data/interim/italy_polarization/cleaned_monthly_chunks/` | `results/tables/italy_polarization/cleaning/` |
| 2 | — | `screening/subreddit_screening_*.csv`, `subreddit_exclusion_summary.csv` |
| 3 | enriched parquet in place (canonical) | `subreddit_topic_assignment.csv`, `subreddit_forum_political_profile.csv`, `subreddit_topic_political_audit.csv` |
| plots | — | `results/tables/italy_polarization/cleaning_pipeline/`, `results/figures/italy_polarization/cleaning_pipeline/{volume,stage1_drop_rates,political_qa}/` |
| 4 | enriched parquet + feature columns | `results/tables/italy_polarization/descriptives/`, `results/figures/italy_polarization/descriptives/{daily,rolling_daily}/` |

**Stage 0 (raw only):** `plot_data_quality_trends.py` counts all NDJSON rows — it does **not** drop comments.

### Cross-forum and cross-country author overlap

Reddit `author` is globally unique, so overlap is an exact username match. Diagnostics live in [`scripts/diagnostics/user_overlap_across_forums.py`](scripts/diagnostics/user_overlap_across_forums.py) (any subreddit in the event window) and [`scripts/diagnostics/user_same_day_cross_forum.py`](scripts/diagnostics/user_same_day_cross_forum.py) (authors posting in ≥2 forums on the same UTC day). CSVs: `results/tables/italy_polarization/user_overlap/` (also indexed in [`MasterSystemPrompt.md`](MasterSystemPrompt.md)).

Re-run (defaults: exclude `[deleted]` and known bots):

```bash
.venv/bin/python scripts/diagnostics/user_overlap_across_forums.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/diagnostics/user_same_day_cross_forum.py --config config/italy_polarization_setup.yaml
```

**Headline counts** on stage-1 `cleaned_monthly_chunks/` (Mar–Apr 2023, 117 primary subreddits, run 2026-06-03):

| Metric | Count | Share of unique authors |
|--------|------:|------------------------:|
| Unique authors (all forums) | 182,191 | 100% |
| Authors in **>1 subreddit** | 26,376 | **14.5%** |
| Authors in **>1 country family** (`de`, `eu`, `uk`, `us`, `it_others` via config `subreddit_family_map`) | 10,491 | **5.8%** |
| Authors in **Italian + any control** family | 1,884 | **1.0%** |

Most multi-subreddit activity is within Italy (many discovered Italian communities); cross-country-family overlap is much smaller. The country-family row uses the static YAML topic→family map (discovered Italian subs → `it_others`); screening assigns `it_political` vs `it_others` for DiD but is not applied in this overlap scan. Wordfish author runs separately flag **cross-language** authors (`cross_language` in `wordfish_authors_assignment.csv`) for sign-only contrasts — a different notion from forum/country overlap.

Distribution detail: `user_overlap/user_overlap_forum_count_distribution.csv` (e.g. 18,796 authors in exactly 2 subreddits; max 18 subreddits for one author).

**Polarization lexicons (runtime):** `data/raw/polarization_lexicon_parallel.csv` (incl. **affect** buckets), `emotion_cognition_parallel.csv`, `style_phrase_parallel.csv` (incl. **hedging**), `italian_political_lexicon_v4.csv` (pairs). Archived txt: [`config/archive/lexicons/`](config/archive/lexicons/ARCHIVE.md). Methods: `results/tables/italy_polarization/descriptives/polarization_metrics_notes.txt`.

## Next steps (after measurement layer)

1. Re-run stage 4 on all shards after editing raw lexicon CSVs (`prepare_parallel_lexicon_raw.py` if needed)
2. Run `prepare_lexicon_descriptives.py` + `plot_lexicon_descriptives.py` (primary outcomes: `net_ideology`, `pair_framing_net_strict`, W0 launch, 7d rolling)
3. **Then** hand-label P/R in `lexicon_validation_labels.csv` on the **new** lexicon
4. Event-study / DiD (launch-primary windows; lift appendix only)

---

## Directory structure (active study)

| Path | Role |
|------|------|
| `config/italy_polarization_setup.yaml` | Event window, discovery window, control lists, paths |
| `config/archive/` | Archived AI-adoption YAML |
| `scripts/discovery/` | 3-day dump profiling and config apply |
| `scripts/filtering/` | Monthly dump → daily NDJSON |
| `data/raw/italy_polarization/daily_chunks/` | Filtered comments |
| `data/raw/political_lexicon_parallel.csv` | Graded trilingual political salience (runtime source for enrichment) |
| `data/raw/polarization_lexicon_parallel.csv` | Trilingual categorized polarization lexicons (runtime) |
| `data/raw/style_phrase_parallel.csv` | Hedging / signposting / polite-closer phrases (runtime) |
| `data/raw/emotion_cognition_parallel.csv` | Emotion and cognition lemma lists (runtime) |
| `data/raw/italian_political_lexicon_v4.csv` | Italian framing pairs (`section=pairs`; runtime for pair columns) |
| `data/interim/italy_polarization/cleaned_monthly_chunks/` | Stage-1 cleaned Parquet |
| `data/interim/italy_polarization/cleaned_monthly_by_family/` | Deprecated optional copies (`--write-by-family`) |
| `results/tables/italy_polarization/cleaning/` | Stage-1 audits |
| `results/tables/italy_polarization/screening/` | Stage-2/3 screening and topic assignment |
| `results/tables/italy_polarization/cleaning_pipeline/` | Pipeline diagnostic tables |
| `results/figures/italy_polarization/cleaning_pipeline/` | Nested QA figures (`volume/`, `stage1_drop_rates/`, `political_qa/`) |
| `results/README.md` | Index of all tables, figures, logs |
| `config/archive/lexicons/` | Archived txt/json snapshots; runtime uses `data/raw/*.csv` |
| `results/tables/italy_polarization/user_overlap/` | Cross-forum / cross-country author overlap CSVs |
| `results/tables/italy_polarization/discovery/` | Discovery CSVs |
| `results/tables/italy_polarization/lexicon_export/` | v4 lexicon export audits |
| `results/logs/italy_polarization/filter_dump/` | Filter resume state and logs |

| `data/raw/circumvention/` | VPN/Tor circumvention proxies (gitignored; see script below) |

See [`scripts/README.md`](scripts/README.md) for script-level detail.

---

## Circumvention / adaptation data (VPN + Tor)

Replicates Kreitmeir & Raschky (2023) proxies around Italy’s ChatGPT ban: **Tor Metrics** daily users (relay + bridge), **Google Trends** VPN topic interest (not the bare keyword `VPN`), and a parallel **ChatGPT** topic/keyword series for attention/salience (not usage). Window defaults: `2023-01-01`–`2023-06-30`; treated `IT`, controls `DE`, `FR`, `ES`, `GB`, `US`.

```bash
.venv/bin/python scripts/download_circumvention_data.py
```

Writes under `data/raw/circumvention/` (raw per-country files + combined CSVs + `_manifest.json`). **`data/` is gitignored** — the script is the reproducible artifact; archive outputs to external disk with other raw data.

DiD tables: `prepare_polarization_descriptives.py`, `prepare_circumvention_descriptives.py`, `prepare_semantic_axis_descriptives.py` (seven axes; `--panels-only` for fast path), then `prepare_did_merged_panels.py` → `did/panels/{country,semantic}/`, `prepare_did_subreddit_panel.py` → `did/panels/subreddit/`, `prepare_did_comment_panel.py` → `did/panels/comment/` (after enriched shards + `compute_semantic_axis_extend.py`; includes all `sem_axis_*` for bucket ES), and `prepare_did_aggregated_panels.py` → `did/panels/aggregated/`. Default `did_event_study.py` estimates all seven semantic-axis forum outcomes (`OUTCOME_REGISTRY`). **Bucket comment DiD:** after `prepare_did_comment_panel.py --bin-days 3` and `assign_author_ideology_buckets.py --cohort strict`, run `scripts/analysis/bucket_event_study.py` (dual **lexical** / **semantic** stratification × multi-outcome: semantic axes + lexical rates; nested `strat_lexical/` / `strat_semantic/` under `did/bucket_event_study/{1,3}d/`; legacy root = lexical + `net_ideology`). **Within-user shift violins:** `plot_user_shift_by_ideology_bucket.py` (emotion/aggression semantic + lexical aggression/negative by bucket). Estimation: `scripts/analysis/did_event_study.py` (forum TWFE + `italy_only_post` Italy-only entity FE; comment-level `pyfixest` via `--families lexical_comment,semantic_axis_comment`; author×day robustness `*_author_day`; `--author-spec week3`; `did_summary` post-phase specs under `did.post_phases`). **Inference routing** (`src/did/inference.py`): forum-clustered `pvalue` is **descriptive** for `cross_country_*` (one treated country); headline cross-country p uses **placebo-in-space** (p floor `1/5` with four controls) plus **gsynth** on `did/panels/aggregated/did_language_*d.csv` (`scripts/analysis/run_did_gsynth.py`); `within_italy_ddd` / `author_*` use restricted **wild cluster bootstrap** (`wildboottest`, 9999 draws). Compare before/after: `did/inference_before_after.md` from `scripts/diagnostics/build_inference_before_after.py`. Also `scripts/analysis/did_aggregated_event_study.py` (bundled PNGs under `did/event_study/{panel}/{bundle}/{1,3}d/` plus dual-tail `sem_axis_ideology_tail_shift*.png` on all bundles × `{1,3}d`; e.g. `language/subreddit/`, `language/hub_pooled/`, `language_universe/in_out_slice/` with `_in_tree` / `_out_tree` variants; stale flat `language/1d/` trees are deleted on full run) (`src/did/`, `linearmodels`) → `did/estimates/summary/` (`did_summary.csv`, `by_family/`, `by_outcome/`, `by_theme/` CSV+txt for all/aggression/ideology/emotion/ai_style/wordfish/lexical) and `did/estimates/{family}/{coefficients,robustness,event_study}/`. Migrate legacy flat files: `scripts/devtools/migrate_did_table_layout.py`. Figures: `results/figures/italy_polarization/did/{lexical,semantic_axis,wordfish_*,overview}/` (headline coefplots show short/medium/long post phases; forest/heatmap use `full_ban`; duplicate rows disambiguated with `(full ban)` / `(early ban)` / phase parentheticals; per-folder `README.md`; `--figures-only` rebuilds PNGs from `did_summary.csv`). Wordfish v2 DiD runs when forum/author v2 extremity panels exist. Semantic intensity: `vpn_interest_it` / `tor_*_it` only; lexical country rows use geo-matched circumvention per `country_panel`.

**Google Trends caveat:** each country geo is scaled 0–100 **within that country and window**; compare **within-country over time**, not levels across countries.

### First-stage inference upgrades (2026-06)

After panels exist under `results/tables/italy_polarization/did/`:

| Step | Script | Output |
|------|--------|--------|
| Weighted TWFE | `did_event_study.py --weights n_comments` | `did/estimates_weighted/` (parallel figures under `did_weighted/`) |
| Placebo-in-time (7d fixed window) | `placebo_in_time.py` | `did/estimates/summary/placebo_in_time.csv` |
| MDE table | `first_stage_mde.py` (`--weighted` optional) | `did/estimates/summary/first_stage_mde.csv` |
| Style index B1 | `fit_style_index_stats.py` → `compute_style_index_on_shards.py` → re-run `prepare_polarization_descriptives.py` | `did/style_index_stats.json`; panel cols `style_index_*_mean` |
| Validation gates (Task 5) | `validate_style_index_gates.py` | `did/style_index_validation/` |
| K&R short window | `post_first_2bd` spec in `did_event_study.py` (Apr 3–4 only; 0–2d rel_day = existing `post_short_3d`) | `by_outcome/*.csv` rows with `spec=post_first_2bd` |
| Adopter DDD | `prepare_adopter_flags.py` → `adopter_ddd.py` | `did/adopter_flags.csv`, `did/adopter_ddd/` |

**Inference distinction:** `placebo_in_time.csv` is design-based (fixed 7-day post, truncated pre-ban sample). `run_robustness_grid` date placebos remain **descriptive** (unequal windows; labeled in figures/CSV as `inference_role=descriptive_unequal_window`).

---

## External dumps

Academic Torrents Reddit comments bundle; for **March–April 2023** only:

```bash
# Indices 206–207 = RC_2023-03.zst, RC_2023-04.zst (verify with aria2c --show-files on your .torrent)
aria2c --dir "/Volumes/Expansion/Masterthesis/RawData" --seed-ratio=0 \
  --file-allocation=none --continue=true --select-file=206,207 \
  "data/reddit-ba051999301b109eab37d16f027b3f49ade2de13.torrent"
```

---

## Archived pipeline (AI-writing adoption)

The Nov 2022–Apr 2023 cross-domain study (comment features, HF detectors, calendar event-time plots) is archived under [`config/archive/`](config/archive/) and [`scripts/archive/`](scripts/archive/README.md). The active study uses only `italy_polarization_setup.yaml` and enriched shards (no `comment_features/` tree).
