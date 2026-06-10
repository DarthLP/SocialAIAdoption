#!/usr/bin/env bash
# Run lexical emotion/cognition DiD full stack (forum, weighted, comment, exbantopic).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
CFG=config/italy_polarization_setup.yaml
OUTCOMES=emotion_rate,cognition_rate

$PY -m pytest tests/test_did_lexical_emotion_cognition_outcomes.py -q

$PY scripts/diagnostics/prepare_polarization_descriptives.py --config "$CFG"
$PY scripts/diagnostics/prepare_did_subreddit_panel.py --config "$CFG"
$PY scripts/diagnostics/prepare_did_comment_panel.py --config "$CFG"

$PY scripts/features/compute_ban_topic_flag.py --config "$CFG" || true
$PY scripts/diagnostics/prepare_polarization_descriptives.py --config "$CFG" --exclude-ban-topic
$PY scripts/diagnostics/prepare_did_subreddit_panel.py --config "$CFG" --exclude-ban-topic

$PY scripts/analysis/did_event_study.py --config "$CFG" --families lexical --outcomes "$OUTCOMES"
$PY scripts/analysis/did_event_study.py --config "$CFG" --families lexical --outcomes "$OUTCOMES" --weights n_comments --no-figures
$PY scripts/analysis/did_event_study.py --config "$CFG" --families lexical_comment --outcomes "$OUTCOMES" --no-bootstrap --no-figures
$PY scripts/analysis/did_event_study.py --config "$CFG" --families lexical --outcomes "$OUTCOMES" --exclude-ban-topic --no-figures --no-bootstrap
$PY scripts/analysis/compare_exbantopic_coefficients.py --config "$CFG"

echo "[done] lexical emotion/cognition DiD full stack"
