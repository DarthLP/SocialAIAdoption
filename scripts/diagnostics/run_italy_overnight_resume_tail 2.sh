#!/usr/bin/env bash
# Script summary:
# Resume Italy overnight pipeline from analyze_user_pre_post_shift (append to existing log).
#
# How to apply/run:
#   screen -dmS italy_overnight bash -lc './scripts/diagnostics/run_italy_overnight_resume_tail.sh'

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PY="${ROOT}/.venv/bin/python"
CONFIG="config/italy_polarization_setup.yaml"
LOG="${ROOT}/results/logs/italy_overnight_20260604T230132Z.log"
STATUS="${ROOT}/results/logs/italy_overnight_status.txt"

exec >>"$LOG" 2>&1
echo "=== RESUME_TAIL $(date -u +%Y-%m-%dT%H:%M:%SZ) from analyze_user_pre_post_shift ==="

CURRENT_STEP="init"
on_pipeline_error() {
  echo "FAILED step=${CURRENT_STEP} at=$(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$$" >"$STATUS"
}
trap on_pipeline_error ERR

run_step() {
  local name="$1"
  shift
  CURRENT_STEP="${name}"
  echo "running step=${CURRENT_STEP} at=$(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$$ log=${LOG}" >"$STATUS"
  echo ""
  echo ">>> STEP ${name} START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$@"
  echo "<<< STEP ${name} OK exit=$? $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep 3
}

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
echo "=== RESUME_TAIL COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
