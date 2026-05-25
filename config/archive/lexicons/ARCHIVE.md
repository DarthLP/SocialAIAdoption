# Archived config lexicons (not used at runtime)

These files are **historical snapshots** from the v4 export and pre-CSV workflow. Stage-4 features and enrichment read **`data/raw/*.csv` only**.

## Runtime sources (edit these)

| Feature | File |
|---------|------|
| Political salience (grades 1–3) | `data/raw/political_lexicon_parallel.csv` |
| Ideology, other_side, aggression, **affect**, issue, ai_style | `data/raw/polarization_lexicon_parallel.csv` (`lexicon` + `bucket` columns) |
| **Hedging**, signposting, polite closer | `data/raw/style_phrase_parallel.csv` (`lexicon` = hedging / signposting / polite_closer) |
| Emotion / cognition (broader than affect buckets) | `data/raw/emotion_cognition_parallel.csv` |
| Italian pair framing | `data/raw/italian_political_lexicon_v4.csv` (`section=pairs`) |

**Affect** (`negative_rate_100w`, `anger_rate_100w`) comes from rows with `lexicon=affect` in the polarization CSV — not from `categorized/affect_*.txt`.

**Hedging** (`hedging_phrase_*`) comes from `style_phrase_parallel.csv` — not from `style_phrases/hedging_*.txt`.

## Archive layout

| Folder | Contents |
|--------|----------|
| `categorized/` | `ideology_*`, `other_side_*`, `aggression_*`, `affect_*`, `issue_*`, `ai_style_*` |
| `style_phrases/` | `hedging_*`, `signposting_*`, `polite_closer_*` (source for one-time `prepare_parallel_lexicon_raw.py` export) |
| `v4_export/` | `stance_it.txt`, `valence_it.txt`, `polarized_it.txt`, `pairs_it.json`, `term_meta_it.json`, `ideology_it_broad.txt`, `dominant_export_stamp.txt` |

Optional regeneration: `scripts/devtools/export_italian_lexicon_v4.py` writes to `v4_export/` by default.

Do not edit archived lists expecting pipeline updates — update the raw CSVs instead.
