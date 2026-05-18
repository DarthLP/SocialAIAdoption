# SocialAIAdoption

## Objective

Thesis study: **The Effects of AI Access on Online Political Polarization** — Reddit comment corpora around Italy’s ChatGPT access restriction (**March–April 2023** UTC). The active pipeline discovers Italian-language subreddits from a **3-day March 2023** screen, then extracts a fixed comparison set plus approved Italian communities into local NDJSON day chunks.

**Active config:** [`config/italy_polarization_setup.yaml`](config/italy_polarization_setup.yaml)

**Comparison forums (fixed):**

- **English political (discussion-style, not r/politics):** `Ask_Politics`, `NeutralPolitics`, `PoliticalDiscussion`, `moderatepolitics`
- **EU hubs:** `de`, `spain`, `unitedkingdom`, `europe`
- **EU political:** `ukpolitics`
- **Italian:** data-driven discovery + seeds (`Italia`, `politicaITA`)

**Archived AI-adoption corpus:** [`config/archive/ai_adoption_political_forums_setup.yaml`](config/archive/ai_adoption_political_forums_setup.yaml) (Nov 2022–Apr 2023 cross-domain study). Old scripts under `scripts/features/`, `scripts/event_time/`, `scripts/user_week/` remain for reference.

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
  --state_file results/logs/filter_dump/italy_polarization_state.json \
  --log_file results/logs/filter_dump/italy_polarization.log
```

Writes: `data/raw/italy_polarization/daily_chunks/<subreddit>/<YYYY-MM-DD>.ndjson`

Optional after interrupt: `.venv/bin/python scripts/cleaning/dedupe_daily_chunks.py --config config/italy_polarization_setup.yaml --apply`

---

## Next steps (not part of extraction milestone)

1. `clean_daily_chunks.py` → `data/interim/italy_polarization/cleaned_monthly_chunks/`
2. Per-comment and author-level language tagging on cleaned Parquet
3. Political lexicons (IT/EN/DE/ES) and subreddit political/social classification
4. Comparable Mar–Apr descriptives and polarization metrics
5. Event-study / DiD around ban dates (`2023-03-31`, lift `2023-04-28`)

---

## Directory structure (active study)

| Path | Role |
|------|------|
| `config/italy_polarization_setup.yaml` | Event window, discovery window, control lists, paths |
| `config/archive/` | Archived AI-adoption YAML |
| `scripts/discovery/` | 3-day dump profiling and config apply |
| `scripts/filtering/` | Monthly dump → daily NDJSON |
| `data/raw/italy_polarization/daily_chunks/` | Filtered comments |
| `results/tables/italy_polarization/discovery/` | Discovery CSVs |
| `results/logs/filter_dump/italy_polarization_*` | Filter resume state |

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

The previous README steps for `config/political_forums_setup.yaml` (comment features, event-time, user-week, Colab ML) applied to the cross-domain Nov 2022–Apr 2023 corpus. Config is archived under `config/archive/`. Re-enable only if you restore that study arm.
