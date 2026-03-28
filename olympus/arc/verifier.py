"""
ARC Verifier — compiles hypothesis C programs and verifies against training pairs.

The verification loop:
1. Write C source to temp file (with runtime.h + arc_grid.h includes)
2. Compile via transformer-vm (C → WASM → tokens)
3. Execute on each training input
4. Parse output grid, compare cell-by-cell with expected
5. Return detailed pass/fail with error analysis
"""

import logging
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from .grid_io import Grid, grid_to_string, string_to_grid, grids_equal, grid_diff, format_grid

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
WASM_TOOLS_DIR = Path(__file__).parent.parent / "wasm_tools"
TVM_DIR = PROJECT_ROOT / "transformer-vm"
RUNTIME_H = TVM_DIR / "transformer_vm" / "compilation" / "runtime.h"
ARC_GRID_H = WASM_TOOLS_DIR / "arc" / "arc_grid.h"
COMPILED_DIR = WASM_TOOLS_DIR / "compiled"


class ARCVerifier:
    """Compiles and verifies ARC rule hypotheses against training data."""

    def __init__(self):
        self.tvm_engine = None
        self._init_tvm()

    def _init_tvm(self):
        """Initialize transformer-vm for compilation."""
        try:
            import sys
            tvm_parent = str(TVM_DIR)
            if tvm_parent not in sys.path:
                sys.path.insert(0, tvm_parent)
            from transformer_vm.compilation.compile_wasm import compile_program
            from transformer_vm.wasm.reference import load_program, run
            self._compile_program = compile_program
            self._load_program = load_program
            self._run_program = run
            self.available = True
            logger.info("ARC Verifier: transformer-vm ready")
        except Exception as e:
            logger.warning(f"ARC Verifier: transformer-vm not available: {e}")
            self.available = False

    def _ensure_clang(self):
        """Ensure CLANG_PATH is set."""
        if "CLANG_PATH" not in os.environ:
            for name in ("clang.exe", "clang"):
                p = TVM_DIR / "wasi-sdk" / "bin" / name
                if p.exists():
                    os.environ["CLANG_PATH"] = str(p)
                    return
            raise RuntimeError("Cannot find wasi-sdk clang")

    def compile_and_run(self, c_code: str, input_str: str, tag: str = "arc") -> Optional[str]:
        """Compile a C program and run it with given input.

        Args:
            c_code: Complete C source code (with includes resolved)
            input_str: Input string for compute()
            tag: Cache tag for compiled output

        Returns:
            Output string from the program, or None on failure.
        """
        if not self.available:
            return None

        self._ensure_clang()

        # Write C source to a temp file, copying runtime.h and arc_grid.h alongside
        tmp_dir = tempfile.mkdtemp(prefix="arc_verify_")
        try:
            # Copy runtime.h
            if RUNTIME_H.exists():
                shutil.copy2(RUNTIME_H, os.path.join(tmp_dir, "runtime.h"))

            # Copy arc_grid.h (and create arc/ subdir)
            arc_dir = os.path.join(tmp_dir, "arc")
            os.makedirs(arc_dir, exist_ok=True)
            if ARC_GRID_H.exists():
                shutil.copy2(ARC_GRID_H, os.path.join(arc_dir, "arc_grid.h"))

            # Write the C source
            c_path = os.path.join(tmp_dir, "rule.c")

            # Prepend the runtime.h include (arc_grid.h already includes it conceptually,
            # but the TVM compilation pipeline auto-includes runtime.h)
            # We need to make the include path work. The compile_program function
            # handles runtime.h automatically. We need arc_grid.h to be findable.
            # Strategy: inline arc_grid.h content into the C source before compiling.
            resolved_code = self._resolve_includes(c_code)
            with open(c_path, "w", encoding="utf-8") as f:
                f.write(resolved_code)

            # Compile
            COMPILED_DIR.mkdir(parents=True, exist_ok=True)
            out_base = str(COMPILED_DIR / tag)
            self._compile_program(c_path, input_str, out_base=out_base)

            token_file = out_base + ".txt"
            if not os.path.exists(token_file):
                logger.warning(f"Compilation produced no token file: {token_file}")
                return None

            # Execute
            program, input_data = self._load_program(token_file)
            result = self._run_program(program, input_data, max_tokens=10000000, trace=False)
            output = result[2]
            return output.strip() if output else None

        except Exception as e:
            logger.warning(f"compile_and_run failed: {e}")
            return None
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _resolve_includes(self, c_code: str) -> str:
        """Replace #include "arc/arc_grid.h" with inline content.

        The TVM compile_program auto-prepends runtime.h, so we only need
        to inline our custom headers. We strip block comments from the
        header to avoid nested comment issues in the generated C.
        """
        # Read arc_grid.h content
        if ARC_GRID_H.exists():
            arc_content = ARC_GRID_H.read_text(encoding="utf-8")
        else:
            logger.warning("arc_grid.h not found")
            arc_content = ""

        # Strip block comments (/* ... */) to avoid nested comment errors
        import re
        arc_content = re.sub(r'/\*.*?\*/', '', arc_content, flags=re.DOTALL)
        # Strip the #ifndef/#define/#endif guards (runtime.h already included)
        arc_content = re.sub(r'#ifndef\s+\w+\s*\n', '', arc_content)
        arc_content = re.sub(r'#define\s+\w+\s*\n', '', arc_content)
        arc_content = re.sub(r'#endif\s*\n?', '', arc_content)

        # Replace the include directive with inline content
        resolved = c_code.replace('#include "arc/arc_grid.h"', arc_content)
        resolved = resolved.replace('#include "arc_grid.h"', arc_content)
        return resolved

    def verify_hypothesis(self, c_code: str, task: dict) -> dict:
        """Verify a hypothesis C program against all training pairs.

        Args:
            c_code: The C source code implementing the rule
            task: ARC task dict with "train" key

        Returns:
            dict with:
                passed: bool (all training pairs match)
                pairs_tested: int
                pairs_passed: int
                results: list of per-pair results
                error_summary: str (human-readable summary of failures)
        """
        pairs = task["train"]
        results = []
        pairs_passed = 0

        for i, pair in enumerate(pairs):
            inp_grid = pair["input"]
            expected_grid = pair["output"]

            # Serialize input for C program
            input_str = grid_to_string(inp_grid)

            # Compile and run
            tag = f"arc_verify_{i}"
            output = self.compile_and_run(c_code, input_str, tag=tag)

            if output is None:
                results.append({
                    "pair_index": i,
                    "passed": False,
                    "error": "compilation or execution failed",
                    "output_raw": None,
                })
                continue

            # Parse output grid
            try:
                actual_grid = string_to_grid(output)
            except ValueError as e:
                results.append({
                    "pair_index": i,
                    "passed": False,
                    "error": f"output parse error: {e}",
                    "output_raw": output,
                })
                continue

            # Compare
            diff = grid_diff(expected_grid, actual_grid)
            passed = diff["match"]
            if passed:
                pairs_passed += 1

            results.append({
                "pair_index": i,
                "passed": passed,
                "diff": diff,
                "output_raw": output,
            })

        all_passed = pairs_passed == len(pairs)

        # Build error summary
        error_lines = []
        for r in results:
            if not r["passed"]:
                if "error" in r:
                    error_lines.append(f"  Pair {r['pair_index']}: {r['error']}")
                elif r.get("diff", {}).get("size_mismatch"):
                    d = r["diff"]
                    error_lines.append(
                        f"  Pair {r['pair_index']}: size mismatch "
                        f"expected {d['expected_size']} got {d['actual_size']}"
                    )
                else:
                    d = r.get("diff", {})
                    error_lines.append(
                        f"  Pair {r['pair_index']}: {d.get('error_count', '?')} cells wrong "
                        f"({d.get('accuracy', 0):.0%} accuracy)"
                    )

        return {
            "passed": all_passed,
            "pairs_tested": len(pairs),
            "pairs_passed": pairs_passed,
            "results": results,
            "error_summary": "\n".join(error_lines) if error_lines else "all pairs passed",
        }

    def apply_to_test(self, c_code: str, task: dict) -> list:
        """Apply a verified hypothesis to test inputs.

        Returns list of predicted output grids.
        """
        test_pairs = task.get("test", [])
        predictions = []

        for i, pair in enumerate(test_pairs):
            inp_grid = pair["input"]
            input_str = grid_to_string(inp_grid)
            tag = f"arc_test_{i}"
            output = self.compile_and_run(c_code, input_str, tag=tag)

            if output is None:
                predictions.append(None)
                continue

            try:
                pred_grid = string_to_grid(output)
                predictions.append(pred_grid)
            except ValueError:
                predictions.append(None)

        return predictions
