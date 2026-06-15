#!/usr/bin/env bash
# Script summary:
# Run the Italy polarization analysis pipeline overnight in dependency order.
# One RAM-heavy Python job at a time; optional light parallel steps (placebo+MDE, patch+figures).
# Logs to results/logs/italy_overnight_<timestamp>.log; does not delete any data on disk.
#
# How to apply/run:
#   chmod +x scripts/diagnostics/run_italy_overnight_pipeline.sh
#   nohup ./scripts/diagnostics/run_italy_overnight_pipeline.sh >> results/logs/italy_overnight_nohup.log 2>&1 &
#   tail -f results/logs/italy_overnight_*.log

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PY="${ROOT}/.venv/bin/python"
CONFIG="config/italy_polarization_setup.yaml"
LEX_OUTCOMES="ai_style_rate,em_dash_rate,exclamation_rate,sentence_len_var,avg_wps,style_index_llm,style_index_llm_no_ai_style,style_index_llm_no_em_dash,style_index_llm_no_semicolon_colon,style_index_llm_no_hedging_phrase,style_index_llm_no_exclamation,ttr_50w,readability,log_len_mean,share_ge20w"

# Match only project Python workers (not the bash driver).
PIPELINE_RE="${PY}.*/scripts/(features/compute_style_index|diagnostics/prepare_did|diagnostics/patch_did|analysis/did_event_study|analysis/bucket_event_study|analysis/placebo_in_time|analysis/first_stage_mde|analysis/adopter_ddd|analysis/prepare_adopter|user_week/)"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${ROOT}/results/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/italy_overnight_${TS}.log"
STATUS="${LOG_DIR}/italy_overnight_status.txt"

# Append only to log file (avoid tee+pipefail SIGPIPE when the launching terminal closes).
exec >>"$LOG" 2>&1
CURRENT_STEP="init"
on_pipeline_error() {
  echo "FAILED step=${CURRENT_STEP} at=$(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$$" >"$STATUS"
}
trap on_pipeline_error ERR

write_status() {
  echo "running step=${CURRENT_STEP} at=$(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$$ log=${LOG}" >"$STATUS"
}

write_status

echo "=== Italy overnight pipeline started ${TS} ==="
echo "log=${LOG}"
echo "root=${ROOT}"

# Function summary: block until no other pipeline Python processes are running.
wait_no_pipeline_jobs() {
  local n=0
  while pgrep -f "$PIPELINE_RE" >/dev/null 2>&1; do
    n=$((n + 1))
    if (( n % 12 == 1 )); then
      echo "[wait] pipeline jobs still running: $(pgrep -fl "$PIPELINE_RE" 2>/dev/null | head -5 || true)"
    fi
    sleep 5
  done
}

# Function summary: pause after a heavy step so the OS can reclaim RAM (no disk deletes).
release_memory() {
  wait_no_pipeline_jobs
  sleep 3
  sync 2>/dev/null || true
  echo "[memory] idle pause complete ($(date -u +%H:%M:%S) UTC)"
}

# Function summary: run one command; exit pipeline on failure with step name in log.
run_step() {
  local name="$1"
  shift
  wait_no_pipeline_jobs
  release_memory
  CURRENT_STEP="${name}"
  write_status
  echo ""
  echo ">>> STEP ${name} START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$@"
  local ec=$?
  echo "<<< STEP ${name} OK exit=${ec} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  release_memory
}

# Function summary: run two light steps in parallel; fail if either fails.
run_step_parallel() {
  local n1="$1" c1="$2" n2="$3" c2="$4"
  wait_no_pipeline_jobs
  release_memory
  echo ""
  echo ">>> PARALLEL ${n1} + ${n2} START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  eval "$c1" &
  local p1=$!
  eval "$c2" &
  local p2=$!
  wait "$p1" || { echo "FAILED: ${n1}"; exit 1; }
  wait "$p2" || { echo "FAILED: ${n2}"; exit 1; }
  echo "<<< PARALLEL ${n1} + ${n2} OK $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  release_memory
}

# --- Phase 1: style index ---
run_step "compute_style_index" \
  "$PY" scripts/features/compute_style_index_on_shards.py --config "$CONFIG"

run_step "prepare_polarization_descriptives" \
  "$PY" scripts/diagnostics/prepare_polarization_descriptives.py --config "$CONFIG"

run_step "plot_descriptives_ban_shaded" \
  "$PY" scripts/diagnostics/plot_descriptives_ban_shaded.py --config "$CONFIG"

# --- Phase 2: panels (gates need subreddit panel with style_index_llm_mean) ---
run_step "prepare_did_subreddit_panel" \
  "$PY" scripts/diagnostics/prepare_did_subreddit_panel.py --config "$CONFIG"

run_step "validate_style_index_gates" \
  "$PY" scripts/diagnostics/validate_style_index_gates.py --config "$CONFIG"

run_step "prepare_did_comment_panel_1d" \
  "$PY" scripts/diagnostics/prepare_did_comment_panel.py --config "$CONFIG"

# --- Phase 3: DiD ---
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

# --- Phase 4: adopter DDD ---
run_step "prepare_adopter_flags" \
  "$PY" scripts/analysis/prepare_adopter_flags.py --config "$CONFIG"

run_step "adopter_ddd" \
  "$PY" scripts/analysis/adopter_ddd.py --config "$CONFIG"

# --- Phase 5: user-week + buckets (sequential heavy) ---
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

echo ""
CURRENT_STEP="complete"
echo "COMPLETE at=$(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$$ log=${LOG}" >"$STATUS"
trap - ERR
echo "=== Italy overnight pipeline COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Review log: ${LOG}"
echo "Manual STOP 7c: check adopter_ddd console output for scheme2_reversion_placebo + style_index_llm"
