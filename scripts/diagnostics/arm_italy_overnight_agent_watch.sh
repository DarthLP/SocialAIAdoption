#!/usr/bin/env bash
# Script summary:
# Start Italy pipeline agent watcher (event-driven) + 30m heartbeat fallback.
# Requires Cursor IDE open; kills prior watcher/heartbeat PIDs from pid files.
#
# How to apply/run:
#   ./scripts/diagnostics/arm_italy_overnight_agent_watch.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PID_DIR="${ROOT}/results/logs"
WATCH_PID_FILE="${PID_DIR}/italy_overnight_watcher.pid"
HEART_PID_FILE="${PID_DIR}/italy_overnight_heartbeat.pid"
PROMPT_FILE="${PID_DIR}/italy_overnight_AGENT_PROMPT.txt"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-1800}"

stop_pid() {
  local f="$1"
  if [[ -f "$f" ]]; then
    local p
    p="$(cat "$f")"
    kill "$p" 2>/dev/null || true
    rm -f "$f"
  fi
}

stop_pid "$WATCH_PID_FILE"
stop_pid "$HEART_PID_FILE"

chmod +x "${ROOT}/scripts/diagnostics/italy_overnight_agent_watcher.sh"

nohup "${ROOT}/scripts/diagnostics/italy_overnight_agent_watcher.sh" </dev/null >>"${PID_DIR}/italy_overnight_watcher.log" 2>&1 &
echo $! >"$WATCH_PID_FILE"

# Fallback heartbeat (30m) if watcher misses an edge
nohup bash -c '
  pf="'"${PROMPT_FILE}"'"
  while true; do
    sleep '"${HEARTBEAT_SEC}"'
    P=$(PROMPT_FILE="$pf" python3 -c "import json,os,pathlib; print(json.dumps(pathlib.Path(os.environ[\"PROMPT_FILE\"]).read_text()))")
    echo "AGENT_LOOP_WAKE_italy_pipeline {\"reason\":\"heartbeat\",\"prompt\":${P}}"
  done
' </dev/null >>"${PID_DIR}/italy_overnight_heartbeat.log" 2>&1 &
echo $! >"$HEART_PID_FILE"

echo "Watcher PID=$(cat "$WATCH_PID_FILE") log=${PID_DIR}/italy_overnight_watcher.log"
echo "Heartbeat PID=$(cat "$HEART_PID_FILE") every ${HEARTBEAT_SEC}s log=${PID_DIR}/italy_overnight_heartbeat.log"
echo "In Cursor: use monitored shell on watcher stdout OR rely on heartbeat log + 30m wake."
echo "Stop: kill \$(cat $WATCH_PID_FILE) \$(cat $HEART_PID_FILE)"
