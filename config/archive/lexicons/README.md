# Political and polarization lexicons

Curated term lists for the Italy polarization study. Political **salience** uses graded parallel CSV; polarization dimensions use **categorized** text files.

## Salience (graded parallel CSV)

**Runtime source:** `data/raw/political_lexicon_parallel.csv` (see `paths.political_lexicon_parallel` in study YAML).

| Column pair | Language |
|-------------|----------|
| `IT` / `IT_grade` | Italian |
| `EN (US/UK)` / `EN_grade` | English |
| `DE` / `DE_grade` | German |

Grades 1–3 map to points 1/2/3 per unique matched term. Forum assignment uses word-weighted rate `100 × Σ(points) / Σ(words)`. Thread flag: `thread_political_weighted_points >= 3` (one grade-3 term suffices).

Legacy flat `political_{it,en,de}.txt` files are deprecated for runtime salience (kept in git for reference only).

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

## Style phrase lists (flat, substring match)

One phrase per line (`#` comments allowed). Used by `compute_comment_style_features.py` via `primary_lexicon`:

| Pattern | Purpose |
|---------|---------|
| `hedging_{lang}.txt` | Hedging / epistemic softeners |
| `signposting_{lang}.txt` | Discourse signposting |
| `polite_closer_{lang}.txt` | Polite or assistant-style closers |

Languages: `it`, `en`, `de`. Matched per forum via `primary_lexicon` in enrichment.

## Matching

- Token-level matching on lowercased word tokens ([`src/political_lexicon.py`](../../src/political_lexicon.py)).
- Multi-word phrases: consecutive token sequences.
- Optional negation window for ideology hits (see `polarization.negation_window_tokens` in study YAML).

## Validation

- Run `scripts/diagnostics/audit_polarization_lexicons.py`.
- Hand labels: `results/tables/italy_polarization/descriptives/lexicon_validation_labels.csv`.

## Provenance

Mar–Apr 2023 context; versioned in git. See `CHANGELOG.md`.

### Italian curated list (v4)

- **Source of truth (local):** `data/raw/italian_political_lexicon_v4.csv` (semicolon-separated; hand-checked terms, sources, L/C/R use columns).
- **Runtime lists:** exported into `political_it.txt`, `ideology_it.txt`, `issue_it.txt`, `other_side_it.txt` via:

```bash
.venv/bin/python scripts/devtools/export_italian_lexicon_v4.py --policy dominant
```

- **Dominant export (default):** one L/C/R bucket per v4 term (`yes=3 > some=2 > rarely=1`; left+center→left, right+center→right). First dominant run auto-archives prior `ideology_it.txt` to `ideology_it_broad.txt`.
- **Also writes:** `pairs_it.json`, `term_meta_it.json`, `stance_it.txt`, `valence_it.txt`, `polarized_it.txt`, `dominant_export_stamp.txt`.
- **Pair framing:** `pair_framing_net_strict` (polarized=yes pairs) and `pair_framing_net_all` scored at comment level when `lang_code=it`.
- **Legacy policies:** `--policy broad` or `conservative` still available for reproducing the broad archive.
- **Audit tables:** `results/tables/italy_polarization/lexicon_export/lexicon_v4_export_audit.csv`, `lexicon_v4_export_diff.csv`.
- After export, re-run `enrich_cleaned_chunks.py` (salience) and/or `compute_polarization_features.py` (polarization columns) — see root `README.md`.
