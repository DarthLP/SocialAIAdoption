# Political lexicon changelog

## 2026-05-22 — Graded parallel salience CSV

- Runtime political salience from `data/raw/political_lexicon_parallel.csv` (grades 1–3 → weighted points 1/2/3; unique term hits; max-grade dedupe on duplicate rows).
- Thread political: `thread_political_weighted_points >= 3` (`screening.thread_political_min_points`).
- Forum topics: `word_weighted_political_rate_100w` on weighted points; recalibrate `forum_political_soft_threshold` / `forum_political_pure_threshold` after enrichment.
- Flat `political_{it,en,de}.txt` no longer read at runtime.

## 2026-05-21 — Dominant ideology + pair framing + metadata lexicons

- Default export policy: **dominant** (`export_italian_lexicon_v4.py --policy dominant`).
- `ideology_it.txt` rebuilt with one L/C/R side per v4 term; archive at `ideology_it_broad.txt` on first run.
- Added `pairs_it.json`, `term_meta_it.json`, `stance_it.txt`, `valence_it.txt`, `polarized_it.txt`.
- Comment features: `pair_framing_*`, stance/valence rates, `net_ideology_weighted` (exploratory).
- Requires `polarization.ideology_scoring: dominant_v1` in study YAML.

## 2026-05-20 — Salience curation (EN/DE/IT) + dual topic thresholds

- **Italian** [`political_it.txt`](political_it.txt): added `politica`, `politico`, `politiche`, `politici`, news/TV phrases (`dibattito politico`, `tg politico`, …), `giornalista`/`giornalisti`.
- **English** [`political_en.txt`](political_en.txt): removed ambiguous singles (`green`, `house`, `party`, `eu`, `climate`, …); deduped `sanctions`; kept `climate change`, `energy`, `rights`, `independence`.
- **German** [`political_de.txt`](political_de.txt): removed `eu`, `stimme`, `links`/`rechts`, broad domain singles, COVID-era singles, `integration`; added `klimawandel`, `flüchtlinge`.

## 2026-05-20 — Italian salience v5 (curated)

- Replaced runtime [`political_it.txt`](political_it.txt) with curated list from [`data/raw/political_it_v5.txt`](../../data/raw/political_it_v5.txt).
- Removed high-FP singles (`italia`, `camera`, bare `bonus`, `fratelli`, `crisi`, `lavoro`, …); prefer multi-word phrases (`camera dei deputati`, `bonus 80 euro`).
- Diff table: `results/tables/italy_polarization/lexicon_v5_diff.csv`.
- Categorized lexicons (`ideology_it`, `issue_it`, …) unchanged in this pass.

## 2026-05-20 — Style phrase lexicons

- Added `hedging_{it,en,de}.txt`, `signposting_{it,en,de}.txt`, `polite_closer_{it,en,de}.txt` for `compute_comment_style_features.py`.

## 2026-05-19 — Italian v4 export (broad policy)

- Exported `data/raw/italian_political_lexicon_v4.csv` (185 rows) via `scripts/devtools/export_italian_lexicon_v4.py --policy broad`.
- Merged into `political_it.txt`, `ideology_it.txt`, `issue_it.txt`, `other_side_it.txt` (union with prior lists; audit in `results/tables/italy_polarization/lexicon_export/lexicon_v4_export_*.csv`).
- Pairs contribute both lemmas to salience/ideology/issue lists; pair-opposition scoring not implemented.

## 2026-05-19 — Polarization categorized lexicons

- Added `ideology_{it,en,de}.txt`, `other_side_{lang}.txt`, `aggression_{lang}.txt`, `affect_{lang}.txt`, `issue_{lang}.txt`, `ai_style_{lang}.txt`.
- Categorized format `category:term`; scored by `compute_polarization_features.py` and `compute_ai_use_features.py`.

## 2026-05-18 — Italy polarization expansion pass

- Expanded `political_it.txt`, `political_en.txt`, `political_de.txt` for 2023-relevant parties, institutions, and policy terms.
- Removed ambiguous Italian singles (`dl`, `dlb`, bare `ue`) that caused false positives.
- Deduplicated repeated Italian tokens (`maggioranza`, `opposizione`).
- Lexicons are matched per forum via `primary_lexicon` in enrichment (Italian subs → `it`, `europe` → `en`, etc.).
