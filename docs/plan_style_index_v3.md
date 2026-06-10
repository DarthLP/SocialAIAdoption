# Style index v3 — implementation plan (2026-06)

Successor to [plan_style_index_v2.md](plan_style_index_v2.md). **v1 and v2 columns stay frozen** on shards; v3 adds new columns and a **tuned-weights JSON** (`style_index_stats_v3.json`).

## Motivation (v2 → v3)

| Issue in v2 | v3 response |
|-------------|-------------|
| `style_index_lexical_v2` is lexicon-heavy (ai_rate dominant); not a single “LLM-style” composite | **`style_index_llm_v3`**: theory-weighted bundle of lexicon + formality + punctuation, **no `log_len`** |
| Formality v2 uses equal weights; ρ vs ai_rate uncontrolled on IT | **Robustness bundles** tuned on **March 2023 IT only** so ρ(index, ai_style_rate) ∈ **[0.15, 0.30]** without putting ai_rate in the formula |
| Length / vocabulary diversity confounded v1 | **`log_len_mean`**, **`ttr_50w_mean`** as **separate panel outcomes**, validated vs ai_rate but **not** inside composites |
| Hand-picked v2 IT lexical weights (0.65/0.15/0.20) | **Descent / coordinate search** on pre-period matrix; weights frozen in JSON; **April 2023 or shard holdout** for overfit check |

---

## Index definitions

### 1. Primary: `style_index_llm_v3`

**Role:** First-stage primary outcome (replaces `style_index_lexical_v2` in reporting once gates pass).

**Full index (n_words ≥ 20)** — signed z-mean with **theory weights** (IT defaults below; EN renormalized copy or separate row in JSON after validation):

| Feature | Sign | Theory weight (IT, pre-tune) |
|---------|------|------------------------------|
| `ai_style_rate_100w` | + | 0.40 |
| `hedging_phrase_rate_100w` | + | 0.12 |
| `avg_words_per_sentence` | + | 0.10 |
| `sentence_length_variance` | − | 0.08 |
| `exclamation_rate_100w` | − | 0.10 |
| `caps_word_share` | − | 0.10 |
| `em_dash_rate_100w` | + | 0.05 |
| `semicolon_colon_rate_100w` | + | 0.05 |

**Explicit exclusions:** `log_len`, `ttr_50w`, `readability` — never in the composite.

**Reduced (< 20 w):** `ai_style_rate_100w` z only if computable (same pattern as v2 lexical reduced).

**Leave-one-out robustness (shard columns `style_index_llm_v3_no_*`):** drop one feature and **renormalize** remaining bundle weights (`ablation_renorm=True` default in `compute_llm_index`). Fixed-denominator LOO is opt-in only. Candidate grid excludes stress-test punct-heavy specs; winner chosen via `validate_style_index_llm_ablations.py` on IT March sample.

**Optional IT calibration (phase 1b):** After theory weights, allow **single scalar** `α ∈ [0.8, 1.2]` on ai_rate weight only if IT ρ(llm_v3, ai_rate) < 0.15 on tune sample — do **not** full re-optimize all weights for primary (avoids collapsing to lexicon-only).

### 2. Robustness: `style_index_formality_reweighted_v3`

**Features (no `ai_style_rate_100w`):** hedging, avg WPS, sentence variance (−), exclamation (−), caps (−).

**Tuning target (IT, March 2023 tune matrix):** Spearman ρ(index, `ai_style_rate_100w`) ∈ **[0.15, 0.30]**.

**Intent:** Correlated with lexicon proxy enough for first-stage interpretability, but **not tautological** (ai_rate not in sum).

### 3. Robustness: `style_index_punct_reweighted_v3`

**Features (no ai_rate):** `em_dash_rate_100w`, `semicolon_colon_rate_100w` only.

**Same ρ band** on IT pre-period tune sample.

### 4. Facets (validation / DiD only — not in any composite)

| Panel / shard field | Aggregation | Validation |
|---------------------|-------------|------------|
| `log_len` (comment) | `log_len_mean` | ρ vs ai_rate on IT; |ρ vs llm_v3| < 0.5 gate |
| `ttr_50w` (comment) | `ttr_50w_mean` | ρ vs ai_rate on IT (informative, no hard band) |

Shard writer already has `log_len`; ensure `ttr_50w` persisted when missing.

---

## Scoring mechanics (unchanged pipeline, new variant)

1. **Calibration:** `fit_preperiod_stats(..., version="v3")` on March 1–30, 2023 — clip 1%/99%, μ, σ per feature × language (same as v1/v2).
2. **Per comment:** clip → signed z → weighted mean with bundle weights from JSON.
3. **`STATS_VERSION_V3`:** `v3_<freeze_date>` in `results/tables/.../did/style_index_stats_v3.json`.
4. **JSON shape (extend v2 languages block):**

```json
{
  "version": "v3_2026-06-04",
  "pre_period": ["2023-03-01", "2023-03-30"],
  "tune_meta": {
    "tune_period": ["2023-03-01", "2023-03-30"],
    "holdout_period": ["2023-04-01", "2023-04-30"],
    "tune_lang": "it",
    "rho_band": [0.15, 0.3],
    "n_tune": 12000
  },
  "languages": {
    "it": {
      "ai_style_rate_100w": { "clip_lo": 0, "clip_hi": 5, "mu": 0.1, "sigma": 0.4 },
      "bundles": {
        "llm_v3": {
          "features": ["ai_style_rate_100w", "..."],
          "signs": { "ai_style_rate_100w": 1, "..." },
          "weights": { "ai_style_rate_100w": 0.4, "..." },
          "min_features_full": 4,
          "min_features_reduced": 1
        },
        "formality_reweighted_v3": { "weights": { "...": 0.2 }, "min_features_full": 3 },
        "punct_reweighted_v3": { "weights": { "...": 0.5 }, "min_features_full": 1 }
      }
    }
  }
}
```

---

## Weight tuning via descent

### Input matrix

Build once per tuning run (reuse `fit_style_index_stats.py` shard loop or dedicated script):

| Column | Source |
|--------|--------|
| `date_utc`, `primary_lexicon` | shard |
| Raw features | `comment_feature_dict` |
| `z_<feat>` | clip + signed z using **v3 pre-period stats** (fit before tune, or iterative: fit → tune → refit if distribution shift large) |
| `ai_style_rate_100w` | target for ρ only (never a feature in robustness bundles) |

**Filter:** `date_utc` in March 2023, `primary_lexicon == "it"`, `n_words >= 20`, non-null ai_rate.

**Holdout:** April 2023 IT (same subsample size cap) **or** 20% of March rows by `hash(comment_id) % 5 == 0` if ids available.

### Index from weights

For feature set `F`, non-negative weights `w ≥ 0`:

```
index_i = Σ_{f∈F} w_f · z_{i,f}  /  Σ_{f∈F} w_f     (renormalize over available z)
```

Missing z → drop feature from sum; require `|F_used| ≥ min_features`.

### Objective (robustness bundles)

```
ρ = Spearman(index(w), ai_rate)
penalty = max(0, ρ_lo - ρ)² + max(0, ρ - ρ_hi)²     # ρ_lo=0.15, ρ_hi=0.30
loss = penalty + λ · Σ_f (w_f - w̃_f)²               # w̃ = equal or theory prior
```

**Primary `llm_v3`:** Start from theory `w̃`; optional **only** monitor ρ (pass if > 0.15 IT); tune only if review fails — prefer theory + light ai_rate scale, not full descent.

### Descent pseudocode

```text
procedure TUNE_BUNDLE(Z, ai_rate, F, w_prior, ρ_band, holdout):
    Z ← Z[:, F]   # signed z columns, rows = IT March, n_words≥20
    w ← normalize(max(w_prior, 0))

    # Phase A: coarse — random restarts + best feasible
    best ← (w, loss=∞)
    repeat R=200 times:
        w' ← random_simplex(|F|)   # Dirichlet samples
        if feasible_band(w', Z, ai_rate, ρ_band):
            best ← argmin loss(w', ...)
    w ← best.w or w_prior

    # Phase B: coordinate descent on log-weights (non-negative)
    repeat until loss stable or 500 steps:
        for f in F:
            for grid t in logspace(-2, 2, 21):
                w_try ← w with w_f ← t · w_prior_f
                L ← loss(w_try)
                if L < loss(w): w ← w_try
        if no improvement: break

    # Phase C (optional): Nelder-Mead on softmax(log w) with penalty
    w ← project_simplex(softmax_opt(loss))

    assert Spearman(w, Z, ai_rate) ∈ ρ_band on TUNE
    ρ_hold ← Spearman(w, Z_holdout, ai_rate)
    if ρ_hold outside [0.10, 0.35]: flag OVERFIT in tune_report.csv
    return w, ρ_tune, ρ_hold
```

**Implementation home:** `src/style_index_tuning.py` (pure functions) + `scripts/diagnostics/tune_style_index_v3_weights.py` (CLI writes weights into stats JSON).

**Circularity guardrails:**

- Robustness bundles: **assert** `ai_style_rate_100w ∉ F`.
- Tune **only** on March IT; report April ρ separately.
- Do not tune on post-treatment months.
- Cap weight ratio `max(w)/min(w) ≤ 20` to avoid one-feature indices.

---

## Validation exports (extend v2)

Under `did/style_index_validation/`:

| Artifact | Contents |
|----------|----------|
| `correlation_matrix_v3.csv` | Spearman pairs among: llm_v3, formality_rw_v3, punct_rw_v3, ai_style_rate_100w, log_len, ttr_50w (IT + all) |
| `convergence_correlation_rows` | Per-index vs benchmarks (extend `index_col` list) |
| `joint_signal_buckets_v3.csv` | ai_hit × em_dash × semicolon; means of v3 indices |
| `compare_indices_v3.csv` | v1, v2 lexical, v3 llm, robustness v3 — redundancy |
| `gates_summary.csv` | New rows: v3 IT ρ bands, facet correlations, pretrend on `style_index_llm_v3_mean` |
| `tune_report_v3.csv` | ρ_tune, ρ_holdout, final weights per bundle |

### Gates (freeze v3)

**IT subset, March tune / April holdout reported separately:**

| Gate | Criterion |
|------|-----------|
| `v3_it_spearman_llm_vs_ai_rate` | ρ > **0.15** (stretch 0.25) |
| `v3_it_spearman_llm_vs_log_len` | \|ρ\| < **0.5** |
| `v3_it_length_stratified_20_49` | Δ(hit) > 0 |
| `v3_it_formality_rw_rho_band` | ρ ∈ [0.15, 0.30] on tune; holdout in [0.10, 0.35] |
| `v3_it_punct_rw_rho_band` | same |
| `v3_pretrend_llm_v3` | F-test p > 0.05 on subreddit panel |
| Facets | `log_len_mean`, `ttr_50w_mean` exported; document ρ vs ai_rate |

---

## Implementation phases

### Phase 0 — Planning (this doc) ✓

### Variant sweep (30-shard sample, 2026-06-04)

Run in one pass:

```bash
.venv/bin/python scripts/diagnostics/fit_style_index_stats.py --config config/italy_polarization_setup.yaml --version v3 --max-shards 30
.venv/bin/python scripts/diagnostics/compare_style_index_v3_variants.py --config config/italy_polarization_setup.yaml --max-shards 30
```

Artifacts:

| File | Role |
|------|------|
| `results/tables/italy_polarization/did/style_index_v3_variant_comparison.csv` | Grid metrics per variant |
| `results/tables/italy_polarization/did/style_index_stats_v3.json` | Frozen calibration + `best_variant_id` |
| `results/tables/italy_polarization/did/style_index_validation/tune_report_v3.csv` | Robustness tune ρ |

**IT March tune subset:** n_words ≥ 20, n ≈ 2421 comments (30 shards).

| variant_id | llm ρ(ai) | formality_rw ρ(ai) | punct_rw ρ(ai) | pass_all_gates |
|------------|-----------|-------------------|----------------|----------------|
| llm_theory | 0.331 | 0.148 | −0.074 | no |
| llm_ai_50 | 0.331 | 0.148 | −0.074 | no |
| llm_ai_35 | 0.331 | 0.148 | −0.074 | no |
| llm_anti_casual | 0.331 | 0.148 | −0.074 | no |
| **robustness_mid_band** | 0.331 | **0.177** | −0.074 | no (formality only) |
| punct_emdash_heavy | 0.331 | 0.148 | −0.074 | no |

**Winners (partial freeze on sample):**

| Index | Pick | IT ρ vs ai_rate | Notes |
|-------|------|-----------------|-------|
| `style_index_llm_v3` | `llm_theory` (theory weights) | 0.331 | ρ > 0.15; \|ρ vs log_len\| ≈ 0.15; Δ(20_49) > 0 |
| `style_index_formality_reweighted_v3` | `robustness_mid_band` weights | 0.177 | In [0.15, 0.30] with tune band [0.18, 0.28] |
| `style_index_punct_reweighted_v3` | — | −0.074 | **Blocker:** no variant hit ρ band (sparse punct; negative marginal ρ) |

**Facets (IT, n_words ≥ 20):** ρ(log_len, ai_rate) ≈ **0.19**; ρ(ttr_50w, ai_rate) ≈ **0.10** (informative, no hard band). llm_v3 ρ(ttr) ≈ 0.04 on tune rows.

**STOP:** Do not run full-corpus shard recompute until `punct_reweighted_v3` ρ band is resolved (wider tune grid, more shards, or feature set review).

### LLM v3 leave-one-out validation (80 shards, 2026-06-04)

- **LOO denominator:** fixed total weight (`ablation_renorm=False`); legacy renormalize inflated capture bonuses and produced negative ρ(feature, index_no_feature) for heavy punct weights.
- **Winner:** `theory_base` (fallback when no candidate passes all own-ablation gates; `rare_signal_heavy` removed from grid).
- **Own-ablation ρ (feature vs index without that feature), n=2421 IT March ge20:**

| Feature | Before (`rare_signal_heavy`) | After (`theory_base`) |
|---------|------------------------------|------------------------|
| ai_style | −0.03 | **+0.06** |
| semicolon | −0.27 | −0.24 |
| hedging | +0.04 | +0.13 |
| exclamation | +0.32 | −0.39 (sign −1; gate applies only to sign +1 features) |

- **Signs:** keep `exclamation_rate_100w` and `caps_word_share` at **−1** in `SIGNS_V3`. Keep **hedging at +1** — marginal ρ(hedging, primary) ≈ 0.36 with lift ≈ 18× on hit; `anti_casual` (hedging −1) worsens LOO score and does not improve semicolon/ai diagnostics.
- **Re-run:** `.venv/bin/python scripts/diagnostics/validate_style_index_llm_ablations.py --config config/italy_polarization_setup.yaml --max-shards 80`

### Phase 1 — Core + tuning (no full shard recompute)

1. `src/style_index.py`: `SIGNS_V3`, bundle constants, `compute_index_v3()`, `compute_all_indices` extended, `style_index_stats_filename("v3")`.
2. `src/style_index_tuning.py`: matrix builder, `tune_bundle_weights`, band penalty (prototype present).
3. `scripts/diagnostics/tune_style_index_v3_weights.py`: load March sample → tune robustness → merge weights into stats dict → save JSON.
4. `scripts/diagnostics/fit_style_index_stats.py`: `--version v3`.
5. Tests: `tests/test_style_index_v3.py`, `tests/test_style_index_tuning.py`.

### Phase 2 — 30-shard prototype

```bash
.venv/bin/python scripts/diagnostics/fit_style_index_stats.py --config config/italy_polarization_setup.yaml --version v3 --max-shards 30
.venv/bin/python scripts/diagnostics/tune_style_index_v3_weights.py --config config/italy_polarization_setup.yaml --max-shards 30
.venv/bin/python scripts/features/compute_style_index_on_shards.py --config config/italy_polarization_setup.yaml --index-version v3 --max-shards 30
.venv/bin/python scripts/diagnostics/validate_style_index_gates.py --config config/italy_polarization_setup.yaml --index-version v3 --max-shards 30
```

**STOP:** Read `gates_summary.csv`, `tune_report_v3.csv`, `correlation_matrix_v3.csv`.

### Phase 3 — Freeze + full recompute

6. Full shard recompute v3 columns only (v1/v2 untouched).
7. `prepare_polarization_descriptives.py` / `prepare_did_comment_panel.py`: aggregate `style_index_llm_v3_mean`, `ttr_50w_mean`, robustness means.
8. `src/did/outcomes.py`: `FIRST_STAGE_OUTCOMES` primary → `style_index_llm_v3`; keep v2 + v1 as legacy robustness.

### Phase 4 — DiD

9. Event study / first stage with llm_v3 primary; formality_rw + punct_rw + facets in appendix tables.

---

## File touch list

| Layer | Files |
|-------|--------|
| Core | `src/style_index.py` |
| Tuning | `src/style_index_tuning.py` (new) |
| Stats fit | `scripts/diagnostics/fit_style_index_stats.py` |
| Tune CLI | `scripts/diagnostics/tune_style_index_v3_weights.py` (new) |
| Shards | `scripts/features/compute_style_index_on_shards.py` |
| Validation | `src/style_index_validation.py`, `scripts/diagnostics/validate_style_index_gates.py` |
| Panels | `prepare_polarization_descriptives.py`, `prepare_did_comment_panel.py` |
| DiD | `src/did/outcomes.py` |
| Tests | `tests/test_style_index_v3.py`, `tests/test_style_index_tuning.py`, extend `test_style_index_validation.py` |
| Docs | `docs/plan_style_index_v2.md` (pointer), `README.md`, `MasterSystemPrompt.md`, `scripts/README.md` |

---

## DiD reporting (target)

| Outcome | Role |
|---------|------|
| `style_index_llm_v3` | **Primary** first stage |
| `style_index_formality_reweighted_v3` | Robustness (no lexicon in mix) |
| `style_index_punct_reweighted_v3` | Robustness (punctuation register) |
| `style_index_lexical_v2` / v1 | Legacy comparison |
| `ai_style_rate` | Transparency |
| `log_len_mean`, `ttr_50w_mean` | Length / diversity facets |

---

## Pitfalls

| Risk | Mitigation |
|------|------------|
| **Circularity** | ai_rate in llm_v3 only; robustness bundles exclude it; ρ target is diagnostic not definition |
| **Overfitting weights** | March tune / April holdout; weight cap; prefer theory weights on primary |
| **Sparse punctuation** | punct bundle may need wider grid; report n_nonzero em_dash on tune sample |
| **Lexicon audit** | Parallel track: bad IT hits deflate ai_rate → re-tune after lexicon fix |
| **Replacing v2 too early** | Freeze only after 30-shard STOP passes; keep v2 columns on shards |

---

## Commands (quick reference)

```bash
# Unit tests (no shards)
.venv/bin/python -m pytest tests/test_style_index_tuning.py tests/test_style_index_v3.py -q

# Pipeline slice (30 shards)
.venv/bin/python scripts/diagnostics/fit_style_index_stats.py --config config/italy_polarization_setup.yaml --version v3 --max-shards 30
.venv/bin/python scripts/diagnostics/tune_style_index_v3_weights.py --config config/italy_polarization_setup.yaml --max-shards 30
.venv/bin/python scripts/features/compute_style_index_on_shards.py --config config/italy_polarization_setup.yaml --index-version v3 --max-shards 30
.venv/bin/python scripts/diagnostics/validate_style_index_gates.py --config config/italy_polarization_setup.yaml --max-shards 30
```

---

## Agent handoff (one line)

> Implement v3: `style_index_llm_v3` (theory weights, no log_len); tune `formality_reweighted_v3` and `punct_reweighted_v3` on March IT for ρ∈[0.15,0.30] without ai_rate in features; facets `log_len_mean`/`ttr_50w_mean`; validation matrix + gates; STOP after 30 shards; do not drop v1/v2 columns.
