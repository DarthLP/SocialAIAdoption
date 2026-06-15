# seeds_lexicons — word lists and seeds behind the thesis results

Snapshot (2026-06-11; corrected 2026-06-15) of every lexicon/seed file behind a
result that appears in the thesis. These are COPIES for documentation/inspection
— **the pipeline reads the originals**, so edits here have no effect on results.
Files that are pipeline-active but feed only non-thesis outcomes are deliberately
excluded (list at the bottom).

## lexicons/ (comment-level hit counting → lexical outcomes)

| File | Original | Thesis role |
|---|---|---|
| `polarization_lexicon_parallel.csv` | `data/raw/` | The core categorized lexicon (per language IT/EN/DE), 172 concept slots in six families. **ideology** family = 37 left / 5 center / 34 right → left/center/right_hits → `pole_share`, `pole_rate`, `left_rate`, `right_rate`, `net_ideology`, `extremity`; **aggression / affect** families → `aggression_rate`, `negative_rate`, `anger_rate`; **`ai_style` family = the 17 graded discourse markers → `ai_style_rate_100w`, the thesis FIRST STAGE** (built by `score_comment_ai_style` → `count_categorized_hits(..., "ai_style", ...)`). Also holds `issue` (39) and `other_side` (12). Read via `src/parallel_lexicon.py` / `src/political_lexicon.py`. |
| `political_lexicon_parallel.csv` | `data/raw/` | Political-salience lexicon (467 terms, 23 topic categories, graded G1–G3 per language) → `political_rate_100w`, political-thread classification used in sample construction. |
| `emotion_cognition_parallel.csv` | `data/raw/` | Emotion vs cognition pole terms → lexical `emotion_rate` / `cognition_rate` (the lexical companions to the semantic emotion result). |
| `style_phrase_parallel.csv` | `data/raw/` | Style phrase markers (hedging / signposting / polite_closer). Only the **`hedging`** block feeds a thesis outcome, as `hedging_phrase_rate_100w` — one of the eight components of the composite `style_index_llm` (`src/comment_style.py`, `src/style_index_llm.py`; IT weight ≈ 0.11). It does **NOT** feed `ai_style_rate` (that comes from the `ai_style` family of `polarization_lexicon_parallel.csv`, above). The `signposting` and `polite_closer` blocks are not consumed by any thesis measure. |

## seeds/ (semantic-axis construction → semantic outcomes)

Originals in `data/raw/seeds/` (config `semantic_axis.seeds_dir`). Read by
`src/embeddings.py`: axis = mean(pos-pole fastText vectors) − mean(neg-pole
vectors), built per language; comment score = projection onto the unit axis.

Headline (analysis-carrying) axes:

- `ideology_parallel.csv` — left/right pole concepts → `sem_axis_ideology`,
  extreme tails, pole-share variants (the headline semantic results).
- `emotion_cognition_parallel.csv` — emotion axis → `sem_axis_emotion`
  (early-ban result + fd_mean figure).
- `aggression_parallel.csv` — `sem_axis_aggression` (appendix companion). **NB:
  this CSV lists only the 25-term `aggression` (hostile) pole; the 15-term
  `civil` pole lives only in `seeds/poles/aggression_neg_{it,en,de}.txt`.**

Issue axes (semantic robustness scan only, Appendix A2 `app:results-semantic` —
e.g. Nationalism −0.0038, p=0.005; Anti-establishment −0.0059, p=0.008 within
author; also listed in Table A1 `tab:seed-inventory`):

- `economic_parallel.csv` — market / equality (6/5) → `sem_axis_economic`.
- `cultural_parallel.csv` — traditional / progressive (6/8) → `sem_axis_cultural`.
- `nationalism_parallel.csv` — nationalist / cosmopolitan (6/4) → `sem_axis_nationalism`.
- `anti_establishment_parallel.csv` — anti-elite / pro-institution (15/10) → `sem_axis_anti_establishment`.

These four axes have no `poles/*.txt` overrides; the pipeline reads them straight
from these parallel CSVs.

### seeds/poles/ — per-language txt overrides (what the pipeline actually uses)

`{axis}_{pos|neg}_{it|en|de}.txt`, one term per line; these take precedence over
the parallel CSVs (the CSV is read only when the matching txt is empty/absent —
`embeddings.py::_load_csv_pole_pair`). Pole sizes per language: ideology 15/15,
emotion 25/25, aggression 25 (hostile) / 15 (civil). The aggression civil pole
exists only here in the txt files, not in `aggression_parallel.csv`.

## stopwords/ (Wordfish preprocessing)

Originals in `config/lexicons/` (`stopwordsiso` v0.7.0); used by `src/wordfish.py`.
Wordfish appears in the thesis as the gated triangulation companion (A1/A2
appendices).

## Excluded — pipeline-active or present in data/, but NOT behind any thesis result

- `data/raw/italian_political_lexicon_v4.csv` — feeds only the `pair_framing`
  outcome (not in the thesis).
- `data/raw/political_universe_labels.csv` — used only by the
  `political_universe_compare.py` diagnostic.
- `data/raw/seeds/poles/emotion_neg_*_pruned.txt` — leakage-pruned emotion
  robustness variant (drops the 8 ban-exposed cognition seeds: analysis, data,
  estimate, evidence, method, proof, result, statistics → 17 remain). Referenced
  in §robustness but not reproduced here.
- `data/raw/political_it_v5.txt`, root-level `data/raw/ideology_parallel.csv`
  duplicate, `config/archive/lexicons/**` — not read by the live pipeline at all.
