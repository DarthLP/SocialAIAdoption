# Political and polarization lexicons

**Archived copy** — runtime lexicons are CSV files under `data/raw/`. See [`ARCHIVE.md`](ARCHIVE.md) for the path table (including **affect** → `polarization_lexicon_parallel.csv`, **hedging** → `style_phrase_parallel.csv`).

Historical notes below describe the pre-CSV export workflow.

## Salience (graded parallel CSV)

**Runtime source:** [`data/raw/political_lexicon_parallel.csv`](../../data/raw/political_lexicon_parallel.csv) (`paths.political_lexicon_parallel`).

| Column pair | Language |
|-------------|----------|
| `IT` / `IT_grade` | Italian |
| `EN (US/UK)` / `EN_grade` | English |
| `DE` / `DE_grade` | German |

One lemma per row. Grades 1–3 → points 1/2/3 per unique matched term (Unicode tokenization + spelling variants). Thread flag: `thread_political_weighted_points >= 3`.

## Polarization / AI (parallel categorized CSV)

**Runtime source:** [`data/raw/polarization_lexicon_parallel.csv`](../../data/raw/polarization_lexicon_parallel.csv) (`paths.polarization_lexicon_parallel`).

| `lexicon` column | `bucket` values |
|------------------|-----------------|
| `ideology` | `left`, `center`, `right` |
| `other_side` | `other_side` |
| `aggression` | `aggression` |
| `affect` | `negative`, `anger` |
| `issue` | `eu`, `migration`, `economy`, `culture` |
| `ai_style` | `ai_discourse` |

**Multi-lemma cells:** separate lemmas with **`;`** only (never comma inside `IT` / `EN` / `DE`). Commas appear only in CSV structure or in `notes` / `slot_concept`.

Supplemental ideology rows from [`data/raw/ideology_parallel.csv`](../../data/raw/ideology_parallel.csv) are merged into the polarization CSV via `scripts/devtools/prepare_parallel_lexicon_raw.py`.

## Emotion / cognition

**Runtime source:** [`data/raw/emotion_cognition_parallel.csv`](../../data/raw/emotion_cognition_parallel.csv) (`paths.emotion_cognition_parallel`).

Columns: `pole` (`emotion` | `cognition`), `concept`, `IT`, `EN`, `DE`. Scored as `emotion_*` and `cognition_*` rates (distinct from `affect` negative/anger in the polarization CSV).

## Style phrases (substring match)

**Runtime source:** [`data/raw/style_phrase_parallel.csv`](../../data/raw/style_phrase_parallel.csv) (`paths.style_phrase_parallel`).

`lexicon` ∈ `hedging`, `signposting`, `polite_closer`. One phrase per row per language column.

## Italian pair framing (v4 CSV, pairs section only)

**Runtime source:** [`data/raw/italian_political_lexicon_v4.csv`](../../data/raw/italian_political_lexicon_v4.csv) (`paths.italian_lexicon_v4_pairs`), semicolon-delimited file; only `section=pairs` rows are loaded.

Comment columns: `pair_framing_net_strict`, `pair_framing_net_all`, and related `pair_*` fields when `lang_code=it`.

## Matching

- Unicode word tokens: `[\w']+` ([`src/political_lexicon.py`](../../src/political_lexicon.py)).
- German/Italian spelling alternates via [`src/parallel_lexicon.py`](../../src/parallel_lexicon.py) `expand_lexicon_variants`.
- Ideology negation window: `polarization.negation_window_tokens` in study YAML.

## Prepare / audit

```bash
.venv/bin/python scripts/devtools/prepare_parallel_lexicon_raw.py --gap-report
.venv/bin/python scripts/diagnostics/audit_polarization_lexicons.py --config config/italy_polarization_setup.yaml
```

Gap table (optional): `results/tables/italy_polarization/lexicon_export/parallel_vs_config_gap.csv`.

## Legacy export (optional)

`scripts/devtools/export_italian_lexicon_v4.py` writes snapshots under `categorized/` and `v4_export/`; **stage-4 features do not require this step.**

See `CHANGELOG.md` for history.
