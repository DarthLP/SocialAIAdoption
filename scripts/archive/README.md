# Archived scripts (AI-adoption / political_forums ML stack)

These paths are **not** used by the active Italy polarization study. Active pipeline: [`scripts/README.md`](../README.md) with [`config/italy_polarization_setup.yaml`](../../config/italy_polarization_setup.yaml).

## Archived here

| Path | Purpose |
|------|---------|
| `features/compute_comment_features.py` | Monolithic lexical + HF detectors |
| `features/merge_ml_shards_into_comment_features.py` | Colab ML merge into `comment_features/` |
| `features/compute_daily_repetition_similarity.py` | Daily repetition CSV for event-time merge |
| `event_time/prepare_event_time_metrics.py` | Aggregate `comment_features/` to daily metrics |
| `event_time/plot_event_time_metrics.py` | Calendar event-time figures |
| `event_time/prepare_event_time_stratified_metrics.py` | Stratified pooled event-time tables |
| `event_time/plot_event_time_stratified_metrics.py` | Stratified event-time figures |
| `user_week/prepare_user_week_style_panel.py` | Per-author ISO-week panel from `comment_features/` |
| `user_week/analyze_user_pre_post_shift.py` | Within-user pre/post (`ai_likeness_user_week`) |
| `user_week/plot_user_pre_post_shift.py` | Pre/post shift figures |
| `diagnostics/run_llm_detector_sample.py` | Detector sample QA |
| `diagnostics/describe_ml_zip_time_trends.py` | ML zip export descriptives |
| `devtools/_gen_colab_standalone_nb.py` | Regenerate Colab notebook |
| `notebooks/colab_compute_comment_features_gpu.ipynb` | GPU ML shards (standalone) |

**Legacy configs:**

- [`config/archive/ai_adoption_political_forums_setup.yaml`](../../config/archive/ai_adoption_political_forums_setup.yaml) — Nov 2022–Apr 2023 cross-domain corpus
- [`config/archive/italy_chatgpt_ban_setup.yaml`](../../config/archive/italy_chatgpt_ban_setup.yaml) — superseded by `italy_polarization_setup.yaml`

**HF module:** [`src/archive/comment_feature_models.py`](../../src/archive/comment_feature_models.py)

## Example legacy run (after local `comment_features/` exist)

```bash
.venv/bin/python scripts/archive/features/compute_comment_features.py \
  --config config/archive/ai_adoption_political_forums_setup.yaml

.venv/bin/python scripts/archive/features/merge_ml_shards_into_comment_features.py \
  --config config/archive/ai_adoption_political_forums_setup.yaml

.venv/bin/python scripts/archive/user_week/prepare_user_week_style_panel.py \
  --config config/archive/ai_adoption_political_forums_setup.yaml

.venv/bin/python scripts/archive/user_week/analyze_user_pre_post_shift.py \
  --config config/archive/ai_adoption_political_forums_setup.yaml

.venv/bin/python scripts/archive/user_week/plot_user_pre_post_shift.py \
  --config config/archive/ai_adoption_political_forums_setup.yaml

.venv/bin/python scripts/archive/event_time/prepare_event_time_metrics.py \
  --config config/archive/ai_adoption_political_forums_setup.yaml

.venv/bin/python scripts/archive/event_time/plot_event_time_metrics.py \
  --config config/archive/ai_adoption_political_forums_setup.yaml
```

Regenerate Colab notebook after editing archive config or `src/archive/comment_feature_models.py`:

```bash
.venv/bin/python scripts/archive/devtools/_gen_colab_standalone_nb.py
```
