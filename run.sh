#!/bin/bash
set -e

# Load secrets from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded .env"
fi

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
