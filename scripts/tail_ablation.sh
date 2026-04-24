#!/usr/bin/env bash
# Monitor the two Phase 8 ablation runs. Exits when both are done OR either errors.
# Run from repo root on SageMaker.

set -u

RUN1_LOG=${RUN1_LOG:-run_mixedfmt.log}
RUN1_NAME=mixedfmt
RUN1_OUT=outputs/ablation_kl_reg_sft_mixedfmt_r15_qd_scale200/model

RUN2_LOG=${RUN2_LOG:-run_fi_sft.log}
RUN2_NAME=fi_sft
RUN2_OUT=outputs/ablation_fi_sft_r15_qd_scale200/model

_status() {
  # prints: RUNNING | DONE | ERROR
  local log=$1
  local out=$2
  local pattern=$3
  if pgrep -f "$pattern" > /dev/null; then
    echo RUNNING
    return
  fi
  if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
    echo DONE
    return
  fi
  # process is gone AND no output dir -- something went wrong
  echo ERROR
}

while true; do
  clear
  echo "===== $(date) ====="

  s1=$(_status "$RUN1_LOG" "$RUN1_OUT" "kl_reg_sft_mixedfmt")
  s2=$(_status "$RUN2_LOG" "$RUN2_OUT" "fi_sft")

  echo "[$RUN1_NAME] status=$s1"
  echo "[$RUN2_NAME] status=$s2"
  echo

  echo "[gpu]"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  /' || echo "  (nvidia-smi unavailable)"

  echo
  echo "[$RUN1_NAME tail ($RUN1_LOG)]"
  [ -f "$RUN1_LOG" ] && tail -n 8 "$RUN1_LOG" | sed 's/^/  /' || echo "  (no log yet)"
  echo
  echo "[$RUN2_NAME tail ($RUN2_LOG)]"
  [ -f "$RUN2_LOG" ] && tail -n 8 "$RUN2_LOG" | sed 's/^/  /' || echo "  (no log yet)"

  # Exit conditions
  if [ "$s1" = "ERROR" ] || [ "$s2" = "ERROR" ]; then
    echo
    echo "===== ERROR DETECTED ====="
    if [ "$s1" = "ERROR" ]; then
      echo "--- $RUN1_NAME last 40 lines ---"
      tail -n 40 "$RUN1_LOG" 2>/dev/null || echo "(no log)"
    fi
    if [ "$s2" = "ERROR" ]; then
      echo "--- $RUN2_NAME last 40 lines ---"
      tail -n 40 "$RUN2_LOG" 2>/dev/null || echo "(no log)"
    fi
    exit 1
  fi

  if [ "$s1" = "DONE" ] && [ "$s2" = "DONE" ]; then
    echo
    echo "===== ALL DONE ====="
    echo "mixedfmt -> $RUN1_OUT"
    echo "fi_sft   -> $RUN2_OUT"
    exit 0
  fi

  sleep 15
done
