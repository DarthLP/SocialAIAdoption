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

# Stage 4 — polarization + AI + style features (run only AFTER enrich finishes on all shards)
# --workers: parallel shards (default min(8, cpu_count-1)); --pass all uses one parquet read/write per shard
.venv/bin/python scripts/features/compute_enriched_shard_features.py \
  --config config/italy_polarization_setup.yaml --pass all --workers 8
# Or single pass:
.venv/bin/python scripts/features/compute_polarization_features.py \
  --config config/italy_polarization_setup.yaml --workers 8
.venv/bin/python scripts/features/compute_ai_use_features.py --config config/italy_polarization_setup.yaml --workers 8
.venv/bin/python scripts/features/compute_comment_style_features.py --config config/italy_polarization_setup.yaml --workers 8

# Preflight: one shard must contain net_ideology and semicolon_count
.venv/bin/python -c "import pandas as pd; from pathlib import Path; p=next(Path('data/interim/italy_polarization/cleaned_monthly_chunks').rglob('*.parquet')); d=pd.read_parquet(p); assert 'net_ideology' in d.columns and 'semicolon_count' in d.columns"

# Optional — within-user pre/post (author × ISO week) on enriched shards
.venv/bin/python scripts/user_week/prepare_user_week_style_panel.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/user_week/analyze_user_pre_post_shift.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/user_week/plot_user_pre_post_shift.py --config config/italy_polarization_setup.yaml

.venv/bin/python scripts/diagnostics/audit_polarization_lexicons.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/diagnostics/prepare_polarization_descriptives.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/diagnostics/plot_polarization_descriptives.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/diagnostics/prepare_lexicon_descriptives.py --config config/italy_polarization_setup.yaml
.venv/bin/python scripts/diagnostics/plot_lexicon_descriptives.py --config config/italy_polarization_setup.yaml
```

Stage 4 adds columns **in-place** on enriched Parquet: polarization (`net_ideology` from `polarization_lexicon_parallel.csv`, pair framing `pair_framing_*` from v4 pairs CSV, emotion/cognition rates, …), AI lexicon, and style counts from `style_phrase_parallel.csv`. All shards share the same column schema (Italian pair framing only on `it` shards). Config requires `polarization.ideology_scoring: dominant_v1`.

Lexicon descriptives (pair framing, stance, valence) live under `results/figures/italy_polarization/descriptives/{primary,ideology_dominant,pairs,stance,valence,polarized,trajectory_scatter}/`. Legacy polarization descriptives remain under `descriptives/daily/` and `descriptives/rolling_daily/`. Re-running **enrich** after stage 4 removes feature columns — run stage 4 again if that happens.

**Results layout:** see [`results/README.md`](results/README.md).

**Pre-registered thresholds** (`config/italy_polarization_setup.yaml` → `screening`):

- URL-only rows dropped at Stage 1; forums with ≥80% URL-only input rows excluded.
- Italian arms: pooled Mar–Apr langid ≥ **70%** (up to 500 comments/month sampled).
- **large_volume**: ≥ **100** kept comments over the window; **low_volume** if only `LOW_VOLUME_WINDOW`; soft monthly floor **50** for sparse-month flags.
- `r/europe` treated as **general English** (`primary_lexicon: en`).

**Topic / family taxonomy:** Italian **topics** `it_political` / `it_pure_political` / `it_others` from graded word-weighted rate (WW); **family** `it_political` pools both political topics. Controls: `de`, `eu`, `us`, `uk` (`uk`, `uk_political`). **Salience:** [`data/raw/political_lexicon_parallel.csv`](data/raw/political_lexicon_parallel.csv) (grades 1–3 → points 1/2/3 per unique term; IT/EN/DE columns). **Assignment** (Italian arms): metadata overrides → controls → WW ≥ `forum_political_pure_threshold` → `it_pure_political` → WW ≥ `forum_political_soft_threshold` → `it_political` → else `it_others` (recalibrate thresholds after enrichment; current **0.6** / **1.2**). **Thread political flag:** `thread_political_weighted_points >= 3` (`thread_political_min_points`). Audit: `subreddit_forum_political_profile.csv`; `political_threshold_sensitivity.csv`; QA under `cleaning_pipeline/political_qa/`.

**Outputs:**

| Stage | Interim | Tables / figures |
|-------|---------|------------------|
| 1 | `data/interim/italy_polarization/cleaned_monthly_chunks/` | `results/tables/italy_polarization/cleaning/` |
| 2 | — | `screening/subreddit_screening_*.csv`, `subreddit_exclusion_summary.csv` |
| 3 | enriched parquet in place (canonical) | `subreddit_topic_assignment.csv`, `subreddit_forum_political_profile.csv`, `subreddit_topic_political_audit.csv` |
| plots | — | `results/tables/italy_polarization/cleaning_pipeline/`, `results/figures/italy_polarization/cleaning_pipeline/{volume,stage1_drop_rates,political_qa}/` |
| 4 | enriched parquet + feature columns | `results/tables/italy_polarization/descriptives/`, `results/figures/italy_polarization/descriptives/{daily,rolling_daily}/` |

**Stage 0 (raw only):** `plot_data_quality_trends.py` counts all NDJSON rows — it does **not** drop comments.

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
| `results/tables/italy_polarization/discovery/` | Discovery CSVs |
| `results/tables/italy_polarization/lexicon_export/` | v4 lexicon export audits |
| `results/logs/italy_polarization/filter_dump/` | Filter resume state and logs |

See [`scripts/README.md`](scripts/README.md) for script-level detail.

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
