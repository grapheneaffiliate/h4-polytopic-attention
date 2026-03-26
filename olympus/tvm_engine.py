"""
Transformer-VM Compute Engine for Olympus

Primary compute engine: compiles C programs to WASM, executes through
analytically-constructed transformer at ~30K tokens/sec. Exact computation,
no training, no approximation.

Falls back to compiled_arithmetic.py when transformer-vm is not installed.

Usage:
    engine = TVMEngine()
    result = engine.compute("15 * 23")          # -> "345"
    result = engine.compute("fib 10")           # -> "55"
    result = engine.compute("gcd 12 8")         # -> "gcd=4 lcm=24"
    result = engine.compute("prime 97")         # -> "prime"
    result = engine.compute("collatz 7")        # -> "7 22 11 34 17 52 26 13 40 20 10 5 16 8 4 2 1"
"""

import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
TVM_DIR = PROJECT_ROOT / "transformer-vm"
WASM_TOOLS_DIR = Path(__file__).parent / "wasm_tools"
COMPILED_DIR = WASM_TOOLS_DIR / "compiled"

# Arithmetic expression pattern: supports nested parens and basic ops
ARITH_RE = re.compile(
    r'^[\s]*'
    r'(-?\d+(?:\.\d+)?)'
    r'\s*([+\-*/%^])\s*'
    r'(-?\d+(?:\.\d+)?)'
    r'[\s]*$'
)

# More complex expressions with parens
COMPLEX_ARITH_RE = re.compile(
    r'^[\s]*\(?[\d+\-*/%^()\s.]+\)?[\s]*$'
)

# Named operations
NAMED_OPS = {
    'fib': ('fibonacci', 'math/fibonacci.c'),
    'fibonacci': ('fibonacci', 'math/fibonacci.c'),
    'gcd': ('gcd', 'math/gcd.c'),
    'lcm': ('gcd', 'math/gcd.c'),
    'prime': ('prime_check', 'math/prime_check.c'),
    'isprime': ('prime_check', 'math/prime_check.c'),
    'collatz': ('collatz', 'math/collatz.c'),
    'lis': ('lis', 'code/lis.c'),
}

# Patterns that trigger compiled tools instead of specialist generation
COMPILED_PATTERNS = {
    'lis': re.compile(r'(?i)longest\s+increasing\s+subsequence.*\[([\d,\s\-]+)\]'),
}


class TVMEngine:
    """Compute engine powered by Percepta's transformer-vm."""

    def __init__(self):
        self.available = TVM_DIR.exists() and (TVM_DIR / "transformer_vm").exists()
        self._evaluator_runtime = None
        self._graph_built = False
        self._compile_cache = {}  # c_file -> wasm compiled flag

        if self.available:
            # Add transformer-vm to path for imports
            tvm_parent = str(TVM_DIR)
            if tvm_parent not in sys.path:
                sys.path.insert(0, tvm_parent)
            logger.info("transformer-vm found at %s", TVM_DIR)
        else:
            logger.info("transformer-vm not found, will use fallback")

    def _find_clang(self) -> Optional[str]:
        """Find clang with wasm32 support."""
        # Check env var first
        clang = os.environ.get("CLANG_PATH")
        if clang and os.path.exists(clang):
            return clang
        # Check wasi-sdk bundled with transformer-vm
        for name in ("clang.exe", "clang"):
            wasi_clang = TVM_DIR / "wasi-sdk" / "bin" / name
            if wasi_clang.exists():
                return str(wasi_clang)
        return None

    def _ensure_clang_env(self):
        """Set CLANG_PATH if we can find wasi-sdk."""
        if "CLANG_PATH" not in os.environ:
            clang = self._find_clang()
            if clang:
                os.environ["CLANG_PATH"] = clang

    def _compile_c_to_tokens(self, c_source: str, args_str: str, out_name: str) -> Optional[str]:
        """Compile a C source file to token prefix with given args.

        Returns path to the .txt token file, or None on failure.
        """
        if not self.available:
            return None

        self._ensure_clang_env()

        try:
            from transformer_vm.compilation.compile_wasm import compile_program

            COMPILED_DIR.mkdir(parents=True, exist_ok=True)
            out_base = str(COMPILED_DIR / out_name)
            compile_program(c_source, args_str, out_base=out_base)
            txt_path = out_base + ".txt"
            if os.path.exists(txt_path):
                return txt_path
        except Exception as e:
            logger.warning("Failed to compile %s: %s", c_source, e)
        return None

    def _run_wasm_direct(self, token_file: str) -> Optional[str]:
        """Run a token file through the direct WASM interpreter.

        This executes the actual WASM bytecode — fast and exact.
        No transformer weights needed. Returns the output string.
        """
        try:
            from transformer_vm.wasm.reference import load_program, run

            program, input_str = load_program(token_file)
            result = run(program, input_str, max_tokens=100000, trace=False)
            # run() returns (instr_count, token_count, output_str) without trace
            output = result[2]
            return output.strip() if output else None

        except Exception as e:
            logger.warning("WASM direct execution failed: %s", e)
            return None

    def _run_arithmetic(self, a: str, op: str, b: str) -> Optional[str]:
        """Run arithmetic through transformer-vm."""
        c_source = str(WASM_TOOLS_DIR / "math" / "arithmetic.c")
        if not os.path.exists(c_source):
            return None

        args_str = f"{a} {op} {b}"
        # Sanitize for filesystem: replace operators with names
        op_names = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '%': 'mod', '^': 'pow'}
        op_safe = op_names.get(op, 'op')
        cache_key = f"arith_{a}_{op_safe}_{b}"
        token_file = self._compile_c_to_tokens(c_source, args_str, cache_key)
        if token_file is None:
            return None

        return self._run_wasm_direct(token_file)

    def _run_named_op(self, op_name: str, args_str: str) -> Optional[str]:
        """Run a named operation (fibonacci, gcd, prime, collatz)."""
        if op_name not in NAMED_OPS:
            return None

        name, c_rel_path = NAMED_OPS[op_name]
        c_source = str(WASM_TOOLS_DIR / c_rel_path)
        if not os.path.exists(c_source):
            return None

        cache_key = f"{name}_{args_str.replace(' ', '_')}"
        token_file = self._compile_c_to_tokens(c_source, args_str, cache_key)
        if token_file is None:
            return None

        return self._run_wasm_direct(token_file)

    def can_handle(self, query: str) -> bool:
        """Check if this query can be handled by transformer-vm."""
        if not self.available:
            return False

        query = query.strip()

        # Simple arithmetic: "15 * 23", "100 + 200"
        if ARITH_RE.match(query):
            return True

        # Named operations: "fib 10", "prime 97", "gcd 12 8"
        parts = query.lower().split()
        if parts and parts[0] in NAMED_OPS:
            return True

        # Natural language math: "what is 15 * 23"
        cleaned = re.sub(r'(?i)^(what\s+is|calculate|compute|evaluate)\s+', '', query)
        if ARITH_RE.match(cleaned.strip()):
            return True

        # Compiled algorithm patterns
        for name, pattern in COMPILED_PATTERNS.items():
            if pattern.search(query):
                return True

        return False

    def compute(self, query: str) -> Optional[dict]:
        """Execute a computation. Returns dict with result, method, timing, or None."""
        if not self.available:
            return None

        query_clean = query.strip()
        t_start = time.time()

        # Try simple arithmetic first
        cleaned = re.sub(r'(?i)^(what\s+is|calculate|compute|evaluate)\s+', '', query_clean)
        cleaned = re.sub(r'[?!.,;]+$', '', cleaned)  # strip trailing punctuation
        m = ARITH_RE.match(cleaned.strip())
        if m:
            a, op, b = m.group(1), m.group(2), m.group(3)
            # Check for floats — transformer-vm is i32 only
            if '.' in a or '.' in b:
                return None
            result = self._run_arithmetic(a, op, b)
            if result is not None:
                elapsed_ms = (time.time() - t_start) * 1000
                return {
                    'result': result,
                    'expression': f"{a} {op} {b}",
                    'method': 'transformer-vm',
                    'engine': 'wasm-direct',
                    'time_ms': elapsed_ms,
                    'exact': True,
                }

        # Try compiled algorithm patterns (e.g. "longest increasing subsequence on [10,9,2,5,3,7,101,18]")
        for pattern_name, pattern in COMPILED_PATTERNS.items():
            m_pat = pattern.search(query_clean)
            if m_pat:
                # Extract the list from the match
                list_str = m_pat.group(1).strip()
                # Convert "10, 9, 2, 5" to "10 9 2 5" for the C program
                nums = re.sub(r'[,]+', ' ', list_str)
                nums = re.sub(r'\s+', ' ', nums).strip()
                result = self._run_named_op(pattern_name, nums)
                if result is not None:
                    elapsed_ms = (time.time() - t_start) * 1000
                    return {
                        'result': result,
                        'expression': f'{pattern_name}([{list_str}])',
                        'method': 'transformer-vm',
                        'engine': 'wasm-direct',
                        'tool': pattern_name,
                        'time_ms': elapsed_ms,
                        'exact': True,
                    }

        # Try named operations
        parts = query_clean.lower().split()
        if parts and parts[0] in NAMED_OPS:
            op_name = parts[0]
            args_str = " ".join(parts[1:])
            result = self._run_named_op(op_name, args_str)
            if result is not None:
                elapsed_ms = (time.time() - t_start) * 1000
                return {
                    'result': result,
                    'expression': query_clean,
                    'method': 'transformer-vm',
                    'engine': 'wasm-direct',
                    'tool': NAMED_OPS[op_name][0],
                    'time_ms': elapsed_ms,
                    'exact': True,
                }

        return None

    def status(self) -> dict:
        """Return engine status."""
        tools = []
        for c_file in sorted(WASM_TOOLS_DIR.glob("**/*.c")):
            tools.append(str(c_file.relative_to(WASM_TOOLS_DIR)))

        return {
            'available': self.available,
            'tvm_path': str(TVM_DIR) if self.available else None,
            'clang': self._find_clang(),
            'tools': tools,
        }
