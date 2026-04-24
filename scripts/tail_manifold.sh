#!/usr/bin/env bash
# Tail progress of scripts/22_manifold_analysis.py running on SageMaker.
# Run from the repo root on the SageMaker instance.

set -u

LOG=${LOG:-run_manifold.log}
OUT_CSV=final_results/phase7_manifold_analysis.csv
OUT_JSON=final_results/phase7_manifold_pairs.json

echo "=== watching $LOG ==="
echo "(Ctrl-C to stop tailing; the background job keeps running)"
echo

while true; do
  clear
  echo "===== $(date) ====="

  # Is the job still alive?
  if pgrep -f "22_manifold_analysis.py" > /dev/null; then
    PID=$(pgrep -f "22_manifold_analysis.py" | head -1)
    echo "[status] RUNNING (pid $PID)"
    ps -o pid,etime,%cpu,%mem,cmd -p "$PID" 2>/dev/null | sed 's/^/  /'
  else
    echo "[status] NOT RUNNING"
  fi

  echo
  echo "[gpu]"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  /' || echo "  (nvidia-smi unavailable)"

  echo
  echo "[log tail]"
  if [ -f "$LOG" ]; then
    tail -n 25 "$LOG" | sed 's/^/  /'
  else
    echo "  (no $LOG yet)"
  fi

  echo
  echo "[artifacts]"
  if [ -f "$OUT_CSV" ]; then
    echo "  $OUT_CSV  ($(wc -l < "$OUT_CSV") lines, $(stat -c %y "$OUT_CSV" 2>/dev/null || stat -f %Sm "$OUT_CSV"))"
  else
    echo "  $OUT_CSV  (not yet written)"
  fi
  if [ -f "$OUT_JSON" ]; then
    echo "  $OUT_JSON  ($(du -h "$OUT_JSON" | cut -f1))"
  else
    echo "  $OUT_JSON  (not yet written)"
  fi

  # Exit once the process is gone AND the csv exists — the run is complete.
  if ! pgrep -f "22_manifold_analysis.py" > /dev/null && [ -f "$OUT_CSV" ]; then
    echo
    echo "===== DONE ====="
    tail -n 5 "$OUT_CSV"
    break
  fi

  sleep 10
done
