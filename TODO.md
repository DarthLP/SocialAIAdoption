# TODO Board

## Active milestones (Italy polarization)

- Dominant export + re-run stage 4 + `prepare_lexicon_descriptives.py` / `plot_lexicon_descriptives.py` (before P/R labeling).
- Lexicon hand-label P/R validation (`audit_polarization_lexicons.py`, `lexicon_validation_pr.csv`) — **after** dominant features exist on shards.
- ~~Event-study / DiD at ban dates~~ — `prepare_did_subreddit_panel.py` + `scripts/analysis/did_event_study.py` → `results/tables/italy_polarization/did/`.
- ~~DiD design expansion~~ — `italy_only_post`, `prepare_did_comment_panel.py`, comment/author-day families, `pyfixest`, `did.author_wordfish_spec` / `--author-spec week3`. Full run: prepare comment panel (no caps) → `did_event_study.py` with new families; optional second pass `--author-spec week3`.
- ~~Within-user pre/post~~ — full user-week stack: `prepare_user_week_style_panel.py` → `analyze_user_pre_post_shift.py` (polarization/style/semantic) → `plot_user_pre_post_shift.py` → `estimate_user_week_panel.py` → `plot_user_week_event_study.py` → pole/lexical/semantic-by-lexicon/overview plots → ideology buckets. Verify `user_week_panel.parquet` row count > 0 before production run.

## Maintenance

- Re-run feature passes after lexicon export: `export_italian_lexicon_v4.py --policy dominant` then `compute_enriched_shard_features.py --pass all`.
- Filter state: `italy_polarization_state.json` under `data/logs/`.

## Archived (reproducibility only)

- AI-adoption ML + event-time + legacy user-week: `scripts/archive/`, `config/archive/ai_adoption_political_forums_setup.yaml`.
