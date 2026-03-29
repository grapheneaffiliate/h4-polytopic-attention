#!/bin/bash
# Setup script for running ARC-AGI-3 on GitHub Codespaces / Termux / any Linux
# Run: bash setup_and_run.sh

set -e

echo "=== ARC-AGI-3 Setup ==="

# Install dependencies
pip install arc-agi arcengine scipy numpy 2>/dev/null || pip3 install arc-agi arcengine scipy numpy

# Set API key
export ARC_API_KEY="58b421be-5980-4ee8-8e57-0f18dc9369f3"

echo "=== Dependencies installed ==="
echo "=== Running v4 explorer (200K actions, all 25 games) ==="
echo ""

# Run the best explorer
PYTHONIOENCODING=utf-8 python3 olympus/arc3/explorer_v4.py
