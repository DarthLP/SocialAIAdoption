# TODO Board

## Active milestones (Italy polarization)

- Dominant export + re-run stage 4 + `prepare_lexicon_descriptives.py` / `plot_lexicon_descriptives.py` (before P/R labeling).
- Lexicon hand-label P/R validation (`audit_polarization_lexicons.py`, `lexicon_validation_pr.csv`) — **after** dominant features exist on shards.
- Event-study / DiD at ban dates (`plot_reference_dates_utc` in config).
- Within-user pre/post: panel → analyze → plot (`scripts/user_week/`; composites in `config/italy_polarization_setup.yaml` `user_week` block).

## Maintenance

- Re-run feature passes after lexicon export: `export_italian_lexicon_v4.py --policy dominant` then `compute_enriched_shard_features.py --pass all`.
- Filter state: `italy_polarization_state.json` under `data/logs/`.

## Archived (reproducibility only)

- AI-adoption ML + event-time + legacy user-week: `scripts/archive/`, `config/archive/ai_adoption_political_forums_setup.yaml`.
