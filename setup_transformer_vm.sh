#!/usr/bin/env bash
# Setup script for transformer-vm integration with H4 Polytopic Attention
#
# Prerequisites:
#   - Python 3.11+
#   - clang with wasm32 target (apt install clang lld)
#   - uv package manager
#
# Usage:
#   bash setup_transformer_vm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TVM_DIR="$SCRIPT_DIR/transformer-vm"

echo "=== Transformer-VM Setup ==="

# 1. Clone if not present
if [ -d "$TVM_DIR/transformer_vm" ]; then
    echo "[ok] transformer-vm already cloned at $TVM_DIR"
else
    echo "[..] Cloning transformer-vm..."
    git clone https://github.com/Percepta-Core/transformer-vm.git "$TVM_DIR"
    echo "[ok] Cloned."
fi

# 2. Check clang wasm32 support
CLANG="${CLANG_PATH:-$(command -v clang 2>/dev/null || true)}"
if [ -z "$CLANG" ]; then
    echo "[!!] clang not found. Install with: sudo apt install clang lld"
    exit 1
fi

if $CLANG --print-targets 2>/dev/null | grep -q wasm32; then
    echo "[ok] clang has wasm32 support: $CLANG"
    export CLANG_PATH="$CLANG"
else
    echo "[!!] clang at $CLANG lacks wasm32 target. Install clang with wasm support."
    exit 1
fi

# 3. Install dependencies
echo "[..] Installing transformer-vm dependencies..."
cd "$TVM_DIR"
uv sync
echo "[ok] Dependencies installed."

# 4. Verify integration
echo "[..] Running integration check..."
cd "$SCRIPT_DIR"
CLANG_PATH="$CLANG" python3 -c "
import sys
sys.path.insert(0, 'transformer-vm')
from olympus.tvm_engine import TVMEngine
engine = TVMEngine()
assert engine.available, 'TVMEngine not available'
r = engine.compute('3 + 5')
assert r and r['result'] == '8', f'Unexpected result: {r}'
print('[ok] TVMEngine verified: 3 + 5 = 8 (exact, via WASM)')
r2 = engine.compute('fib 10')
assert r2 and r2['result'] == '55', f'Unexpected result: {r2}'
print('[ok] Fibonacci verified: fib(10) = 55')
print('[ok] All checks passed.')
"

echo ""
echo "=== Setup Complete ==="
echo "transformer-vm is ready. Set CLANG_PATH=$CLANG in your environment."
echo "Usage from Python:"
echo "  from olympus.tvm_engine import TVMEngine"
echo "  engine = TVMEngine()"
echo "  engine.compute('15 * 23')  # -> {'result': '345', 'exact': True}"
