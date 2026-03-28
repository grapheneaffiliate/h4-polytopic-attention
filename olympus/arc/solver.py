"""
ARC-AGI Solver — Self-Compiling Intelligence Loop

The full pipeline:
1. Load ARC task (training pairs + test input)
2. Hypothesize transformation rules (pattern matching → C code)
3. Compile via transformer-vm
4. Verify against ALL training pairs (exact match required)
5. On failure: analyze errors, refine hypothesis, retry
6. On success: apply to test input → answer
7. Register solved rule as permanent compiled tool

Each solved puzzle becomes a permanently compiled tool in the
tool registry, growing the system's capability over time.

Usage:
    solver = ARCSolver()
    result = solver.solve_task("data/arc1/0d3d703e.json")
    # or
    results = solver.solve_dataset("data/arc1/")
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from .grid_io import Grid, format_grid, grids_equal
from .hypothesizer import generate_hypotheses, describe_task, Hypothesis
from .composer import generate_compositions
from .object_hypotheses import generate_object_hypotheses
from .verifier import ARCVerifier

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
SOLVED_DIR = Path(__file__).parent.parent / "wasm_tools" / "arc" / "solved"
TOOL_REGISTRY = Path(__file__).parent.parent / "tool_registry.json"


class ARCSolver:
    """Self-compiling ARC-AGI solver."""

    def __init__(self, max_hypotheses: int = 10, max_retries: int = 3):
        self.verifier = ARCVerifier()
        self.max_hypotheses = max_hypotheses
        self.max_retries = max_retries
        self.solved_tasks = {}  # task_id → Hypothesis
        self.stats = {
            "tasks_attempted": 0,
            "tasks_solved": 0,
            "hypotheses_generated": 0,
            "hypotheses_verified": 0,
            "compilations": 0,
        }

    def load_task(self, path: str) -> dict:
        """Load an ARC task from JSON file."""
        with open(path) as f:
            return json.load(f)

    def solve_task(self, task_path: str, verbose: bool = True) -> dict:
        """Attempt to solve a single ARC task.

        Returns:
            dict with:
                task_id: str
                solved: bool
                hypothesis: Hypothesis name or None
                predictions: list of output grids for test inputs
                attempts: int
                time_ms: float
        """
        task_id = Path(task_path).stem
        task = self.load_task(task_path)
        self.stats["tasks_attempted"] += 1
        t_start = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"Task: {task_id}")
            print(f"Training pairs: {len(task['train'])}")
            print(describe_task(task))
            print(f"{'='*60}")

        # Step 1: Generate hypotheses (singles, then objects, then compositions)
        hypotheses = generate_hypotheses(task)
        object_hyps = generate_object_hypotheses(task)
        compositions = generate_compositions(task)
        # Singles first (highest confidence), then objects, then compositions
        all_hypotheses = hypotheses + object_hyps + compositions
        self.stats["hypotheses_generated"] += len(all_hypotheses)

        if verbose:
            print(f"\nHypotheses: {len(hypotheses)} single + {len(object_hyps)} object + {len(compositions)} compositions")
            for h in all_hypotheses:
                print(f"  - {h.name} ({h.confidence:.0%}): {h.description}")

        hypotheses = all_hypotheses
        if not hypotheses:
            if verbose:
                print("  No hypotheses found — task requires deeper analysis")
            elapsed = (time.time() - t_start) * 1000
            return {
                "task_id": task_id,
                "solved": False,
                "hypothesis": None,
                "predictions": [],
                "attempts": 0,
                "time_ms": elapsed,
                "reason": "no_hypotheses",
            }

        # Step 2: Try each hypothesis
        for i, hyp in enumerate(hypotheses[:self.max_hypotheses]):
            self.stats["hypotheses_verified"] += 1
            self.stats["compilations"] += 1

            if verbose:
                print(f"\n  Testing hypothesis {i+1}/{len(hypotheses)}: {hyp.name}")

            # Compile and verify against training pairs
            result = self.verifier.verify_hypothesis(hyp.c_code, task)

            if verbose:
                print(f"    Pairs: {result['pairs_passed']}/{result['pairs_tested']}")
                if not result["passed"]:
                    print(f"    Errors:\n{result['error_summary']}")

            if result["passed"]:
                # SUCCESS — apply to test input
                if verbose:
                    print(f"    >>> VERIFIED: {hyp.name} <<<")

                predictions = self.verifier.apply_to_test(hyp.c_code, task)

                # Register as permanent tool
                self._register_solved(task_id, hyp)

                elapsed = (time.time() - t_start) * 1000
                self.stats["tasks_solved"] += 1

                if verbose:
                    print(f"\n  Solved in {elapsed:.0f}ms")
                    for j, pred in enumerate(predictions):
                        if pred:
                            print(f"\n  Test output {j}:")
                            print("  " + format_grid(pred).replace("\n", "\n  "))

                return {
                    "task_id": task_id,
                    "solved": True,
                    "hypothesis": hyp.name,
                    "description": hyp.description,
                    "predictions": predictions,
                    "attempts": i + 1,
                    "time_ms": elapsed,
                    "c_code": hyp.c_code,
                }

        # All hypotheses failed
        elapsed = (time.time() - t_start) * 1000
        if verbose:
            print(f"\n  UNSOLVED after {len(hypotheses)} hypotheses ({elapsed:.0f}ms)")

        return {
            "task_id": task_id,
            "solved": False,
            "hypothesis": None,
            "predictions": [],
            "attempts": len(hypotheses),
            "time_ms": elapsed,
            "reason": "all_hypotheses_failed",
        }

    def _register_solved(self, task_id: str, hyp: Hypothesis):
        """Register a solved task as a permanent compiled tool."""
        self.solved_tasks[task_id] = hyp

        # Save the C source
        SOLVED_DIR.mkdir(parents=True, exist_ok=True)
        c_path = SOLVED_DIR / f"{task_id}.c"
        c_path.write_text(hyp.c_code)
        logger.info(f"Saved solved rule: {c_path}")

        # Update tool registry
        try:
            if TOOL_REGISTRY.exists():
                with open(TOOL_REGISTRY) as f:
                    registry = json.load(f)
            else:
                registry = {"tools": [], "failures_logged": []}

            # Check for duplicate
            existing = {t["name"] for t in registry["tools"]}
            tool_name = f"arc_{task_id}_{hyp.name}"
            if tool_name not in existing:
                registry["tools"].append({
                    "name": tool_name,
                    "c_source": str(c_path),
                    "query_patterns": [f"arc.*{task_id}"],
                    "description": f"ARC {task_id}: {hyp.description}",
                    "compiled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "uses": 0,
                })
                with open(TOOL_REGISTRY, "w") as f:
                    json.dump(registry, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to update tool registry: {e}")

    def solve_dataset(self, data_dir: str, limit: int = 0, verbose: bool = True) -> dict:
        """Run the solver on all tasks in a directory.

        Returns summary statistics.
        """
        data_path = Path(data_dir)
        task_files = sorted(data_path.glob("*.json"))

        if limit > 0:
            task_files = task_files[:limit]

        results = []
        solved = 0
        total = len(task_files)

        print(f"\nARC-AGI Self-Compiling Solver")
        print(f"Tasks: {total}")
        print(f"TVM available: {self.verifier.available}")
        print(f"{'='*60}")

        for i, tf in enumerate(task_files):
            result = self.solve_task(str(tf), verbose=verbose)
            results.append(result)
            if result["solved"]:
                solved += 1
            if not verbose:
                status = "SOLVED" if result["solved"] else "FAILED"
                hyp = result.get("hypothesis", "-")
                ms = result["time_ms"]
                print(f"  [{i+1}/{total}] {tf.stem}: {status} ({hyp}, {ms:.0f}ms)")

        print(f"\n{'='*60}")
        print(f"Results: {solved}/{total} solved ({100*solved/total:.1f}%)")
        print(f"Stats: {json.dumps(self.stats, indent=2)}")

        return {
            "solved": solved,
            "total": total,
            "accuracy": solved / total if total > 0 else 0,
            "results": results,
            "stats": self.stats,
        }


# ── CLI ───────────────────────────────────────────────────────────

def main():
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="ARC-AGI Self-Compiling Solver")
    parser.add_argument("path", help="Task JSON file or directory of tasks")
    parser.add_argument("--limit", type=int, default=0, help="Max tasks to attempt (0=all)")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    solver = ARCSolver()

    if os.path.isdir(args.path):
        solver.solve_dataset(args.path, limit=args.limit, verbose=not args.quiet)
    else:
        solver.solve_task(args.path, verbose=True)


if __name__ == "__main__":
    main()
