# Political lexicon changelog

## 2026-05-19 — Polarization categorized lexicons

- Added `ideology_{it,en,de,es}.txt`, `other_side_{lang}.txt`, `aggression_{lang}.txt`, `affect_{lang}.txt`, `issue_{lang}.txt`, `ai_style_{lang}.txt`.
- Categorized format `category:term`; scored by `compute_polarization_features.py` and `compute_ai_use_features.py`.

## 2026-05-18 — Italy polarization expansion pass

- Expanded `political_it.txt`, `political_en.txt`, `political_de.txt`, `political_es.txt` for 2023-relevant parties, institutions, and policy terms.
- Removed ambiguous Italian singles (`dl`, `dlb`, bare `ue`) that caused false positives.
- Deduplicated repeated Italian tokens (`maggioranza`, `opposizione`).
- Lexicons are matched per forum via `primary_lexicon` in enrichment (Italian subs → `it`, `europe` → `en`, etc.).
