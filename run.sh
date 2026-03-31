#!/bin/bash
set -e

# === Configuration ===
# Set these before running, or export them in your environment
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_TOKEN="${HF_TOKEN:-}"

# === Setup ===
pip install -e . -q
git submodule update --init --recursive

# === Run ===
PHASE="${1:-all}"
echo "Running phase: $PHASE"

python sagemaker/train.py --phase "$PHASE"

echo ""
echo "Done. To see results:"
echo "  python scripts/14_generate_tables.py"
