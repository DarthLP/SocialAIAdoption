# Plan: style-index descriptives & first-stage gaps

See `scripts/README.md` § "First-stage inference upgrades" for run commands.

## Critical path

1. `fit_style_index_stats.py` → `compute_style_index_on_shards.py` (interim shards; not part of `--pass all`)
2. `prepare_polarization_descriptives.py` → `prepare_did_subreddit_panel.py`
3. `validate_style_index_gates.py` — **STOP** before freezing SIGNS v1
4. `did_event_study.py` (lexical / first-stage outcomes) after panel rebuild
5. `placebo_in_time.py` ∥ `first_stage_mde.py`; optional `--weighted`
6. `prepare_did_comment_panel.py` → `prepare_adopter_flags.py` → `adopter_ddd.py` — **STOP** at scheme-2 placebo

## Code follow-ups (2026-06)

- `FIRST_STAGE_OUTCOMES` includes `log_len_mean`, `share_ge20w`
- `plot_polarization_descriptives.py` plots style-index daily metrics
- Author×day panel: `log_len` in `WEIGHTED_OUTCOME_COLS`

## Still optional

- Integrate style index into `_enriched_shard_runner --pass all`
- `did_event_study.py --figures-only` for placebo PNG labels
- `patch_did_inference.py` for `perm_p_t` on existing `did_summary.csv`
- `lexical_author_day` OutcomeSpecs for style index
