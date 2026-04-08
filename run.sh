#!/bin/bash
set -e

# Load secrets from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded .env"
fi

# === Setup ===
pip install -e . -q

# === Run ===
PHASE="${1:-all}"
LOGFILE="logs/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "Running phase: $PHASE"
echo "Logging to: $LOGFILE"
echo "Follow with: tail -f $LOGFILE"

nohup python sagemaker/train.py --phase "$PHASE" > "$LOGFILE" 2>&1 &
PID=$!
echo "PID: $PID"
echo "$PID" > logs/.pid

echo ""
echo "Running in background. Commands:"
echo "  tail -f $LOGFILE      # watch progress"
echo "  kill $PID             # stop the run"
