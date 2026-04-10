#!/bin/bash
set -e

# Load secrets from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded .env"
fi

# === Setup ===
echo "Installing package..."
pip install -e . 2>&1 | tail -1
echo "Setup complete."

# Detect GPU count for distributed training
NPROC_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
export NPROC_PER_NODE
echo "GPUs detected: $NPROC_PER_NODE"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === Run ===
PHASE="${1:-all}"
shift 2>/dev/null || true  # remaining args passed through

LOGFILE="logs/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo ""
echo "=== Starting phase: $PHASE ==="
echo "Logging to: $LOGFILE"
echo "Output shown below (also saved to log)."
echo ""

# Run in foreground with tee so output goes to both terminal and log
python -u sagemaker/train.py --phase "$PHASE" "$@" 2>&1 | tee "$LOGFILE"
