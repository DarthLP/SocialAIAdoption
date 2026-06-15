#!/usr/bin/env bash
# Script summary:
# Resume Italy overnight pipeline after compute_style_index_v3 (append to existing log).
#
# How to apply/run:
#   screen -dmS italy_overnight bash -lc './scripts/diagnostics/run_italy_overnight_resume.sh'

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PY="${ROOT}/.venv/bin/python"
CONFIG="config/italy_polarization_setup.yaml"
LEX_OUTCOMES="ai_style_rate,em_dash_rate,exclamation_rate,sentence_len_var,avg_wps,style_index_llm,style_index_llm_no_ai_style,style_index_llm_no_em_dash,style_index_llm_no_semicolon_colon,ttr_50w,readability,log_len_mean,share_ge20w"
LOG="${ROOT}/results/logs/italy_overnight_20260604T214732Z.log"
STATUS="${ROOT}/results/logs/italy_overnight_status.txt"

exec >>"$LOG" 2>&1
echo "=== RESUME $(date -u +%Y-%m-%dT%H:%M:%SZ) from validate_style_index_gates ==="

CURRENT_STEP="init"
on_pipeline_error() {
  echo "FAILED step=${CURRENT_STEP} at=$(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$$" >"$STATUS"
}
trap on_pipeline_error ERR

write_status() {
  echo "running step=${CURRENT_STEP} at=$(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$$ log=${LOG}" >"$STATUS"
}

run_step() {
  local name="$1"
  shift
  CURRENT_STEP="${name}"
  write_status
  echo ""
  echo ">>> STEP ${name} START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$@"
  echo "<<< STEP ${name} OK exit=$? $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep 3
}

run_step_parallel() {
  local n1="$1" c1="$2" n2="$3" c2="$4"
  echo ">>> PARALLEL ${n1} + ${n2} START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  eval "$c1" &
  local p1=$!
  eval "$c2" &
  local p2=$!
  wait "$p1" || { echo "FAILED: ${n1}"; exit 1; }
  wait "$p2" || { echo "FAILED: ${n2}"; exit 1; }
  echo "<<< PARALLEL ${n1} + ${n2} OK $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

run_step "validate_style_index_gates" \
  "$PY" scripts/diagnostics/validate_style_index_gates.py --config "$CONFIG"

run_step "prepare_did_subreddit_panel" \
  "$PY" scripts/diagnostics/prepare_did_subreddit_panel.py --config "$CONFIG"

run_step "prepare_did_comment_panel_1d" \
  "$PY" scripts/diagnostics/prepare_did_comment_panel.py --config "$CONFIG"

run_step "did_event_study_lexical" \
  "$PY" scripts/analysis/did_event_study.py \
  --config "$CONFIG" --families lexical --outcomes "$LEX_OUTCOMES"

run_step_parallel \
  "placebo_in_time" "$PY scripts/analysis/placebo_in_time.py --config $CONFIG" \
  "first_stage_mde" "$PY scripts/analysis/first_stage_mde.py --config $CONFIG"

run_step "did_event_study_full" \
  "$PY" scripts/analysis/did_event_study.py --config "$CONFIG"

run_step "did_event_study_weighted_lexical" \
  "$PY" scripts/analysis/did_event_study.py \
  --config "$CONFIG" --weights n_comments --families lexical --outcomes "$LEX_OUTCOMES"

run_step "first_stage_mde_weighted" \
  "$PY" scripts/analysis/first_stage_mde.py --config "$CONFIG" --weighted

run_step_parallel \
  "patch_did_inference" "$PY scripts/diagnostics/patch_did_inference.py --config $CONFIG" \
  "did_figures_only" "$PY scripts/analysis/did_event_study.py --config $CONFIG --figures-only"

run_step "prepare_adopter_flags" \
  "$PY" scripts/analysis/prepare_adopter_flags.py --config "$CONFIG"

run_step "adopter_ddd" \
  "$PY" scripts/analysis/adopter_ddd.py --config "$CONFIG"

run_step "prepare_user_week_style_panel" \
  "$PY" scripts/user_week/prepare_user_week_style_panel.py --config "$CONFIG"

run_step "analyze_user_pre_post_shift" \
  "$PY" scripts/user_week/analyze_user_pre_post_shift.py --config "$CONFIG"

run_step "assign_author_ideology_buckets_strict" \
  "$PY" scripts/user_week/assign_author_ideology_buckets.py --config "$CONFIG" --cohort strict

run_step "prepare_did_comment_panel_3d" \
  "$PY" scripts/diagnostics/prepare_did_comment_panel.py --config "$CONFIG" --bin-days 3

run_step "bucket_event_study_3d" \
  "$PY" scripts/analysis/bucket_event_study.py --config "$CONFIG" --bin-days 3

run_step "bucket_event_study_1d" \
  "$PY" scripts/analysis/bucket_event_study.py --config "$CONFIG" --bin-days 1

echo "COMPLETE at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$STATUS"
echo "=== RESUME COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
