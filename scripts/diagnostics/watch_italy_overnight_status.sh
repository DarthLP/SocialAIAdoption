#!/usr/bin/env bash
# Script summary:
# Print a one-screen snapshot of the Italy overnight pipeline (screen session, status file, log tail).
#
# How to apply/run:
#   ./scripts/diagnostics/watch_italy_overnight_status.sh
#   watch -n 60 ./scripts/diagnostics/watch_italy_overnight_status.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATUS="${ROOT}/results/logs/italy_overnight_status.txt"
LOG="$(ls -t "${ROOT}"/results/logs/italy_overnight_2026*.log 2>/dev/null | head -1)"

echo "=== Italy overnight status $(date) ==="
echo ""
echo "--- screen ---"
screen -ls 2>/dev/null | grep -E 'italy_overnight|No Sockets' || echo "(no screen session)"
echo ""
echo "--- status file ---"
if [[ -f "$STATUS" ]]; then cat "$STATUS"; else echo "(missing $STATUS)"; fi
echo ""
echo "--- python workers ---"
pgrep -fl "${ROOT}/.venv/bin/python.*scripts/" 2>/dev/null | grep -E 'compute_style|did_event|bucket_event|prepare_did|user_week|adopter|placebo|first_stage|patch_did' | head -5 || echo "(none)"
echo ""
echo "--- log milestones ---"
if [[ -n "$LOG" && -f "$LOG" ]]; then
  echo "log=$LOG ($(wc -l <"$LOG" | tr -d ' ') lines, mtime $(stat -f '%Sm' "$LOG" 2>/dev/null || stat -c '%y' "$LOG"))"
  grep -E '>>> STEP|<<< STEP|FAILED|Traceback|COMPLETE' "$LOG" 2>/dev/null | tail -6 || echo "(no step markers yet)"
  echo ""
  echo "--- last 5 log lines ---"
  tail -5 "$LOG"
else
  echo "(no italy_overnight_2026*.log found)"
fi
