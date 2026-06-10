#!/usr/bin/env bash
# Script summary:
# Poll Italy overnight pipeline status/log and emit AGENT_LOOP_WAKE_italy_pipeline when
# a step finishes, fails, or the screen session ends — so Cursor can wake the agent.
#
# How to apply/run (from repo root, in a monitored Cursor shell):
#   ./scripts/diagnostics/italy_overnight_agent_watcher.sh
# Pair with a 25–30m heartbeat loop as fallback (see arm_italy_overnight_agent_watch.sh).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATUS="${ROOT}/results/logs/italy_overnight_status.txt"
PROMPT_FILE="${ROOT}/results/logs/italy_overnight_AGENT_PROMPT.txt"
POLL_SEC="${POLL_SEC:-90}"

read_prompt() {
  if [[ -f "$PROMPT_FILE" ]]; then
    PROMPT_FILE="$PROMPT_FILE" python3 -c 'import json, os, pathlib; print(json.dumps(pathlib.Path(os.environ["PROMPT_FILE"]).read_text()))'
  else
    echo '"Monitor Italy overnight pipeline per results/logs/ITALY_OVERNIGHT_README.txt"'
  fi
}

latest_log() {
  ls -t "${ROOT}"/results/logs/italy_overnight_2026*.log 2>/dev/null | head -1
}

emit_wake() {
  local reason="$1"
  local extra="${2:-}"
  local prompt_json
  prompt_json=$(read_prompt)
  echo "AGENT_LOOP_WAKE_italy_pipeline {\"reason\":\"${reason}\",\"status\":\"${extra}\",\"prompt\":${prompt_json}}"
}

last_status=""
last_step_marker=""
screen_was=0

echo "[italy_overnight_agent_watcher] started poll=${POLL_SEC}s" >&2

while true; do
  LOG="$(latest_log)"
  cur_status=""
  [[ -f "$STATUS" ]] && cur_status="$(tr -d '\n' <"$STATUS")"

  if screen -ls 2>/dev/null | grep -q 'italy_overnight'; then
    screen_was=1
  elif (( screen_was )); then
    emit_wake "screen_ended" "${cur_status}"
    screen_was=0
    last_status="${cur_status}__screen_end"
    sleep "$POLL_SEC"
    continue
  fi

  if [[ "$cur_status" != "$last_status" ]]; then
    if [[ "$cur_status" == FAILED* ]]; then
      emit_wake "failed" "${cur_status}"
    elif [[ "$cur_status" == COMPLETE* ]]; then
      emit_wake "complete" "${cur_status}"
    elif [[ -n "$last_status" ]]; then
      emit_wake "status_changed" "${cur_status}"
    fi
    last_status="$cur_status"
  fi

  if [[ -n "$LOG" && -f "$LOG" ]]; then
    if grep -q Traceback "$LOG" 2>/dev/null; then
      marker="traceback"
      if [[ "$marker" != "$last_step_marker" ]]; then
        emit_wake "traceback_in_log" "${cur_status}"
        last_step_marker="$marker"
      fi
    else
      step_line="$(grep -E '>>> STEP|<<< STEP|FAILED:' "$LOG" 2>/dev/null | tail -1 || true)"
      if [[ -n "$step_line" && "$step_line" != "$last_step_marker" ]]; then
        emit_wake "log_step" "${step_line}"
        last_step_marker="$step_line"
      fi
    fi
  fi

  sleep "$POLL_SEC"
done
