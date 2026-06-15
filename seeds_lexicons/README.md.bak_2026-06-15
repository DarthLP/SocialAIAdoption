# seeds_lexicons — word lists and seeds behind the thesis results

Snapshot (2026-06-11) of every lexicon/seed file behind a result that appears
in the thesis. These are COPIES for documentation/inspection — **the pipeline
reads the originals**, so edits here have no effect on results. Files that are
pipeline-active but feed only non-thesis outcomes are deliberately excluded
(list at the bottom).

## lexicons/ (comment-level hit counting → lexical outcomes)

| File | Original | Thesis role |
|---|---|---|
| `polarization_lexicon_parallel.csv` | `data/raw/` | The core categorized lexicon (per language IT/EN/DE): **ideology bucket = 37 left / 5 center / 34 right concept slots** → left/center/right_hits → `pole_share`, `pole_rate`, `left_rate`, `right_rate`, `net_ideology`, `extremity`; plus aggression/affect families → `aggression_rate`, `negative_rate`, `anger_rate`. Read via `src/parallel_lexicon.py` / `src/political_lexicon.py`. |
| `political_lexicon_parallel.csv` | `data/raw/` | Political-salience lexicon (467 terms, 23 topic categories, graded G1–G3 per language) → `political_rate_100w`, political-thread classification used in sample construction. |
| `emotion_cognition_parallel.csv` | `data/raw/` | Emotion vs cognition pole terms → lexical `emotion_rate` / `cognition_rate` (the lexical companions to the semantic emotion result). |
| `style_phrase_parallel.csv` | `data/raw/` | AI-style phrase markers → `ai_style_rate` and the `style_index_llm` components — the thesis **first stage**. |

## seeds/ (semantic-axis construction → semantic outcomes)

Originals in `data/raw/seeds/` (config `semantic_axis.seeds_dir`). Read by
`src/embeddings.py`: axis = mean(pos-pole fastText vectors) − mean(neg-pole
vectors), built per language; comment score = projection onto the unit axis.

- `ideology_parallel.csv` — left/right pole concepts → `sem_axis_ideology`,
  extreme tails, pole-share variants (the headline semantic results).
- `emotion_cognition_parallel.csv` — emotion axis → `sem_axis_emotion`
  (early-ban result + fd_mean figure).
- `aggression_parallel.csv` — `sem_axis_aggression` (appendix companion).

### seeds/poles/ — per-language txt overrides (what the pipeline actually uses)

`{axis}_{pos|neg}_{it|en|de}.txt`, one term per line; these take precedence
over the parallel CSVs. Ideology: 15 terms per pole per language.

## stopwords/ (Wordfish preprocessing)

Originals in `config/lexicons/`; used by `src/wordfish.py`. Wordfish appears
in the thesis as the gated triangulation companion (A1/A2 appendices).

## Excluded — pipeline-active or present in data/, but NOT behind any thesis result

- `data/raw/italian_political_lexicon_v4.csv` — feeds only the `pair_framing`
  outcome (not in the thesis).
- `data/raw/political_universe_labels.csv` — used only by the
  `political_universe_compare.py` diagnostic.
- `data/raw/seeds/{economic,cultural,nationalism,anti_establishment}_parallel.csv`
  — extended issue axes; estimated but not cited in the thesis.
- `data/raw/seeds/poles/emotion_neg_*_pruned.txt` — leakage-pruned emotion
  robustness variant; not discussed in the thesis.
- `data/raw/political_it_v5.txt`, root-level `data/raw/ideology_parallel.csv`
  duplicate, `config/archive/lexicons/**` — not read by the live pipeline at all.
