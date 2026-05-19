# Political and polarization lexicons

Curated term lists for the Italy polarization study. Political **salience** uses flat `political_{lang}.txt`; polarization dimensions use **categorized** files.

## Salience (flat)

| File | Language |
|------|----------|
| `political_it.txt` | Italian |
| `political_en.txt` | English |
| `political_de.txt` | German |
| `political_es.txt` | Spanish |

## Polarization / AI (categorized)

Format: `category:term` per line; `#` starts a comment. Multi-word: `category:word word`.

| Pattern | Categories |
|---------|------------|
| `ideology_{lang}.txt` | `left`, `center`, `right` |
| `other_side_{lang}.txt` | `other_side` (other-side salience; not speaker-conditioned hostility) |
| `aggression_{lang}.txt` | `aggression` |
| `affect_{lang}.txt` | `negative`, `anger` |
| `issue_{lang}.txt` | `eu`, `migration`, `economy`, `culture` |
| `ai_style_{lang}.txt` | `ai_discourse` |

Languages: `it`, `en`, `de`, `es`. Matched per forum via `primary_lexicon` in enrichment.

## Matching

- Token-level matching on lowercased word tokens ([`src/political_lexicon.py`](../../src/political_lexicon.py)).
- Multi-word phrases: consecutive token sequences.
- Optional negation window for ideology hits (see `polarization.negation_window_tokens` in study YAML).

## Validation

- Run `scripts/diagnostics/audit_polarization_lexicons.py`.
- Hand labels: `results/tables/italy_polarization/descriptives/lexicon_validation_labels.csv`.

## Provenance

Mar–Apr 2023 context; versioned in git. See `CHANGELOG.md`.
