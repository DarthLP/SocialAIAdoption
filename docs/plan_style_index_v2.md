# Style index v2 — implementation plan (updated 2026-06)

Consolidates validation findings, construct discussion (lexicon vs punctuation vs caps), and pipeline wiring. **v1 columns stay frozen** for reproducibility.

> **Next:** [plan_style_index_v3.md](plan_style_index_v3.md) — combined `style_index_llm_v3`, ρ-tuned robustness bundles (no ai_rate in formula), facet outcomes (`log_len_mean`, `ttr_50w_mean`), and descent weight tuning in `src/style_index_tuning.py`.

## Goals

1. **Primary first-stage outcome:** `style_index_lexical_v2` — tracks `ai_style_rate_100w` in Italian (bulk of sample), without length dilution.
2. **Robustness outcomes:** formality bundles **without** lexicon in the mix; optional punctuation layer tested, not assumed.
3. **No single proxy is ground truth** — lexicon, em dash, semicolon, and “all-caps share” measure different facets; validate jointly, not via ρ(index_v1, feature).
4. **STOP after validation** on a shard sample before full recompute + DiD.

---

## Construct map (what each feature means)

| Feature | Measures | v2 role |
|---------|----------|---------|
| `ai_style_rate_100w` | Parallel lexicon hits / 100 words | **Core of lexical v2** (audit IT lexicon separately) |
| `hedging_phrase_rate_100w` | Hedging phrases | Formality v2; optional small lexical weight only if IT validation helps |
| `avg_words_per_sentence`, `sentence_length_variance` | Sentence rhythm / structure | **Formality v2 only** (long formal ≠ lexicon hit) |
| `em_dash_rate_100w`, `semicolon_colon_rate_100w` | Punctuation register (sparse rates) | **Formality_punct v2** — test branch, not dropped by v1 ρ ≈ −0.03 |
| `exclamation_rate_100w` | Informal emphasis | Anti-casual (negative sign) in lexical or formality |
| `caps_word_share` | Share of words (len>1) that are **ALL UPPERCASE** (shouting) | Anti-casual **−1**; **not** title-case / “good caps” |
| `log_len` | Comment length | **Never in v2 composites**; separate outcomes `log_len_mean`, `share_ge20w` |

**Future (out of v2 scope unless quick win):** `title_case_share` or sentence-initial capitalization — needs new feature code + validation.

---

## v2 index variants (persist all on shards)

| Column | Features (full, n_words ≥ 20) | Reduced (< 20 w) |
|--------|------------------------------|------------------|
| `style_index_lexical_v2` | Weighted z: `ai_style_rate_100w` (dominant), `exclamation_rate_100w` (−), `caps_word_share` (−); optional small `hedging` if IT ρ improves | `ai_style_rate_100w` z only if computable, else NaN full |
| `style_index_formality_v2` | Equal or weighted z: hedging, avg WPS, sentence variance (−), exclamation (−), caps (−); **no** lexicon, **no** log_len | Subset with ≥2 features or NaN |
| `style_index_formality_punct_v2` | formality_v2 **+** em_dash (+), semicolon_colon (+) | Same rule as formality reduced |

**Default weights (lexical v2, tune after 30-shard validation):**

- IT: `ai_style_rate_100w` 0.65, `exclamation_rate_100w` 0.15, `caps_word_share` 0.20 (renormalize if missing).
- EN: `ai_style_rate_100w` 0.50, anti-casual 0.25 each (EN subsample showed lexicon alignment).

**Scoring mechanics:** same pre-period winsorize + per-language μ/σ as v1; **weighted** signed z-mean; `STATS_VERSION = v2_<date>` in `did/style_index_stats_v2.json`.

**Optional phase 2:** residualize rate features on `log_len` before z-scoring (store β per lang in stats JSON) if lexical v2 still fails IT gates without length confound.

---

## What we do *not* do

- Use ρ(`style_index_full`, em_dash) ≈ −0.03 as proof em dashes are useless — only that **v1** did not encode them.
- Treat `ai_style_rate_100w` as LLM ground truth.
- Drop subreddits — lexicon audit edits **entries**, not forums.
- Overwrite v1 columns on shards.

---

## Validation extensions (before freeze)

Run on same sample as v1 (`--max-shards 30` then full).

### A. Convergence (per `primary_lexicon` / IT / EN)

- Add `semicolon_colon_rate_100w` to `convergence_correlation_rows`.
- Report ρ for each **v2 column** vs: `ai_style_rate_100w`, `log_len`, hedging, em_dash, semicolon.
- **Gates (IT subset):** lexical_v2 vs ai_rate Spearman > **0.15** (stretch 0.25); |ρ vs log_len| < **0.5**; length-stratified Δ in bin 20–49 > 0.

### B. Joint buckets (new)

`joint_signal_buckets.csv` — within length bins (20–49, 50–99, …):

- mean lexical_v2 / formality_v2 by cells: ai_hit × em_dash_hit × (optional semicolon_hit)
- quartile crosses: high lexicon & high em dash vs low/low

Answers: “Do we want comments with **both**?” without forcing one equal-weight index.

### C. Index redundancy

`compare_indices.csv`: Spearman across v1_full, lexical_v2, formality_v2, formality_punct_v2. Target: lexical vs formality not > 0.85 (else one outcome enough).

### D. Manual review

Keep `review_20plus20_by_ai_rate.csv`; add `review_20plus20_joint_high_lex_em.csv` (high lexicon + high em dash in 20–49 bin).

### E. Pretrend (panel)

After panel rebuild: pretrend F on `style_index_lexical_v2_mean` (p > 0.05) — same as v1 gate.

---

## Lexicon audit (parallel, high leverage for IT)

1. Sample high `ai_style_rate_100w` IT rows from review CSV.
2. Tag false positives vs plausible AI-discourse markers.
3. Trim/tier `ai_style_it` (or categorized list) — **no forum exclusion**.
4. Re-score `ai_style_rate_100w` on shards (or at feature dict if lexicon is runtime).
5. Re-run validation **before** freezing weights.

---

## Code touch list

| Layer | Files |
|-------|--------|
| Core | `src/style_index.py` — `INDEX_V2_*` configs, `compute_index(..., variant=)`, weighted mean |
| Stats | `scripts/diagnostics/fit_style_index_stats.py` — `--version v2` |
| Shards | `scripts/features/compute_style_index_on_shards.py` — new columns, `--index-version` |
| Validation | `src/style_index_validation.py`, `scripts/diagnostics/validate_style_index_gates.py` |
| Panels | `prepare_polarization_descriptives.py`, `prepare_did_comment_panel.py` |
| DiD | `src/did/outcomes.py`, `FIRST_STAGE_OUTCOMES`, descriptives plots |
| Tests | `tests/test_style_index.py`, `tests/test_style_index_validation.py` |
| Docs | `MasterSystemPrompt.md`, `README.md`, `scripts/README.md` |

---

## Execution order

```text
1. Implement v2 scoring + tests (v1 unchanged)
2. fit_style_index_stats.py --version v2
3. compute_style_index_on_shards.py --index-version v2 [--max-shards 30]
4. validate_style_index_gates.py  → STOP: read gates + joint buckets + compare_indices
5. [fail] tune weights / lexicon audit / optional residualize → repeat 2–4
6. [pass] full shard recompute
7. prepare_polarization_descriptives → subreddit panel → comment panel
8. did_event_study / first_stage_mde (lexical_v2 primary; formality_v2 + formality_punct_v2 + v1 robustness)
```

**Parallel:** lexicon audit during step 1–4.

---

## DiD reporting strategy

| Outcome | Role |
|---------|------|
| `ai_style_rate` (raw aggregate) | Sanity / transparency |
| `style_index_lexical_v2` | **Primary** formula first stage |
| `style_index_formality_v2` | Robustness: effect without lexicon in bundle |
| `style_index_formality_punct_v2` | Robustness: + em dash / semicolon |
| `style_index_full` (v1) | Legacy comparison only |
| `log_len_mean`, `share_ge20w` | Length controls / separate first stage |

Interpretation language: **all-caps share** = shouting proxy; **punctuation indices** = register; **lexicon** = discourse-marker proxy (noisy in IT).

---

## Freeze criteria (v2)

- [ ] IT ρ(lexical_v2, ai_style_rate) > 0.15
- [ ] IT |ρ(lexical_v2, log_len)| < 0.5
- [ ] IT length-stratified Δ(20–49) > 0
- [ ] Pretrend p > 0.05 on panel lexical_v2 mean
- [ ] Joint-bucket table shows incremental signal (document in README_validation)
- [ ] Manual review: joint high-lex + high-em dash more plausible than high v1 alone
- [ ] Choose formality_punct vs formality_core for paper based on redundancy + story (not ρ vs v1)

---

## Agent handoff (one line)

> Implement v2: keep v1 columns; add lexical_v2, formality_v2, formality_punct_v2; weighted IT lexical; validation joint buckets + semicolon convergence; STOP after 30 shards.

**v3 planning (2026-06):** v2 gates passed on 30-shard sample (`v2_it_spearman_lexical_vs_ai_style_rate` ≈ 0.26). Follow **plan_style_index_v3.md** for primary `style_index_llm_v3` and tuned robustness indices; do not remove v2 columns when v3 ships.
