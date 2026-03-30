"""
Test TransformEngine on real ARC-AGI-1 tasks.

Loads all 400 tasks from data/arc1/, runs the transform engine,
reports solve rates at depth 1, 2, and 3.

This is the most important test — it shows the real ceiling of
pure algorithmic transformation search on ARC.
"""

import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))


def load_all_tasks(data_dir="data/arc1"):
    """Load all ARC-AGI-1 tasks."""
    tasks = {}
    if not os.path.exists(data_dir):
        print(f"Data dir not found: {data_dir}")
        return tasks
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json"):
            continue
        task_id = fname.replace(".json", "")
        with open(os.path.join(data_dir, fname)) as f:
            tasks[task_id] = json.load(f)
    return tasks


def task_to_examples(task):
    """Convert task dict to list of (input_array, output_array) pairs."""
    examples = []
    for pair in task.get("train", []):
        inp = np.array(pair["input"], dtype=int)
        out = np.array(pair["output"], dtype=int)
        examples.append((inp, out))
    return examples


def task_test_pairs(task):
    """Get test input/output pairs."""
    pairs = []
    for pair in task.get("test", []):
        inp = np.array(pair["input"], dtype=int)
        out = np.array(pair["output"], dtype=int) if "output" in pair else None
        pairs.append((inp, out))
    return pairs


def run_benchmark(data_dir="data/arc1", max_tasks=None):
    from agent_zero.transforms.engine import TransformEngine

    tasks = load_all_tasks(data_dir)
    if not tasks:
        print("No tasks found. Make sure data/arc1/ exists with task JSON files.")
        return

    if max_tasks:
        task_ids = sorted(tasks.keys())[:max_tasks]
    else:
        task_ids = sorted(tasks.keys())

    print(f"Running TransformEngine on {len(task_ids)} ARC-AGI-1 tasks...")
    print()

    results = {"d1": 0, "d2": 0, "d3": 0, "total": 0, "test_correct": 0}
    times = []
    solved_tasks = []

    for i, task_id in enumerate(task_ids):
        task = tasks[task_id]
        examples = task_to_examples(task)
        test_pairs = task_test_pairs(task)

        if not examples:
            continue

        results["total"] += 1
        t0 = time.time()

        # Try at each depth
        for depth in [1, 2, 3]:
            engine = TransformEngine(max_depth=depth, timeout=10.0)
            chain = engine.try_solve(examples, game_id=task_id)
            elapsed = time.time() - t0

            if chain:
                # Verify on test data
                test_correct = True
                for test_inp, test_out in test_pairs:
                    if test_out is not None:
                        predicted = chain.apply(test_inp)
                        if not np.array_equal(predicted, test_out):
                            test_correct = False
                            break

                results[f"d{depth}"] += 1
                times.append(elapsed)
                if test_correct:
                    results["test_correct"] += 1
                    solved_tasks.append((task_id, depth, chain.describe(), elapsed))

                if (i + 1) % 50 == 0 or depth == 1:
                    pass  # suppress per-task output for speed
                break  # found at this depth, don't try deeper

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(task_ids)}] "
                  f"d1={results['d1']} d2={results['d2']} d3={results['d3']} "
                  f"test_correct={results['test_correct']}")

    # Final report
    total = results["total"]
    print()
    print("=" * 60)
    print(f"RESULTS: {total} ARC-AGI-1 tasks tested")
    print("=" * 60)
    print(f"  Depth 1 solves: {results['d1']}/{total} ({results['d1']/max(total,1)*100:.1f}%)")
    print(f"  Depth 2 solves: {results['d2']}/{total} ({results['d2']/max(total,1)*100:.1f}%)")
    print(f"  Depth 3 solves: {results['d3']}/{total} ({results['d3']/max(total,1)*100:.1f}%)")
    print(f"  Total training match: {results['d1']+results['d2']+results['d3']}/{total}")
    print(f"  Test-verified correct: {results['test_correct']}/{total}")
    if times:
        print(f"  Average solve time: {sum(times)/len(times):.2f}s")
        print(f"  Max solve time: {max(times):.2f}s")
    print()

    if solved_tasks:
        print("Solved tasks:")
        for tid, depth, desc, elapsed in solved_tasks[:30]:
            print(f"  {tid}: depth={depth}, {desc} ({elapsed:.1f}s)")
        if len(solved_tasks) > 30:
            print(f"  ... and {len(solved_tasks) - 30} more")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/arc1")
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()
    run_benchmark(args.data_dir, args.max_tasks)
