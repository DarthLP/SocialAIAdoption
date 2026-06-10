# Style index (`style_index_llm`)

Single LLM-likelihood composite for first-stage DiD. Legacy v1 (`style_index_full`) and v2 (`style_index_lexical_v2`) are removed.

## Pipeline order (Italy overnight)

1. `validate_style_index_weights.py` — pick `theory_base` (or other candidate) → `style_index_stats.json`
2. `compute_style_index_on_shards.py` — writes `style_index_llm` + `style_index_llm_no_*` on shards
3. `prepare_polarization_descriptives.py` — `style_index_llm_mean` on subreddit-day panel
4. `validate_style_index_gates.py` — construct validity on `style_index_llm`
5. Panels + DiD (`FIRST_STAGE_OUTCOMES` in `src/did/outcomes.py`)

## Columns

| Column | Role |
|--------|------|
| `style_index_llm` | Primary continuous outcome |
| `style_index_llm_no_*` | Leave-one-out robustness (renormalized weights) |

## Calibration

- File: `results/tables/italy_polarization/did/style_index_stats.json`
- Bundle key: `style_index_llm` under `languages.<lang>.bundles`
