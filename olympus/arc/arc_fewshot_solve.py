#!/usr/bin/env python3
"""
Few-shot ARC solver — uses solved tasks as in-context examples.

For each unsolved task:
1. Find 5 most similar solved tasks (by grid features)
2. Include their task descriptions + solutions in the prompt
3. Ask the model to solve the new task following similar patterns
4. Cross-validate against ALL training pairs

Usage:
    python arc_fewshot_solve.py --batch-start 0 --batch-size 10
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from collections import Counter


# ── Grid helpers ─────────────────────────────────────────────────

def grids_equal(a, b):
    if len(a) != len(b): return False
    return all(a[r] == b[r] for r in range(len(a)))

def format_grid(grid):
    return "\n".join(" ".join(str(c) for c in row) for row in grid)

def grid_features(task):
    """Extract features for similarity matching."""
    p = task["train"][0]
    inp, out = p["input"], p["output"]
    h_i, w_i = len(inp), len(inp[0])
    h_o, w_o = len(out), len(out[0])
    colors_in = set(c for row in inp for c in row)
    colors_out = set(c for row in out for c in row)

    # Count connected components (simple)
    def count_objects(grid, bg=0):
        h, w = len(grid), len(grid[0])
        visited = [[False]*w for _ in range(h)]
        count = 0
        for r in range(h):
            for c in range(w):
                if not visited[r][c] and grid[r][c] != bg:
                    count += 1
                    stack = [(r, c)]
                    color = grid[r][c]
                    while stack:
                        cr, cc = stack.pop()
                        if 0 <= cr < h and 0 <= cc < w and not visited[cr][cc] and grid[cr][cc] == color:
                            visited[cr][cc] = True
                            stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
        return count

    return {
        "h_in": h_i, "w_in": w_i,
        "h_out": h_o, "w_out": w_o,
        "same_size": h_i == h_o and w_i == w_o,
        "n_colors_in": len(colors_in),
        "n_colors_out": len(colors_out),
        "new_colors": len(colors_out - colors_in),
        "n_objects": count_objects(inp),
        "n_pairs": len(task["train"]),
        "size_ratio": (h_o * w_o) / max(h_i * w_i, 1),
    }


def similarity(feat_a, feat_b):
    """Score similarity between two task feature dicts. Lower = more similar."""
    score = 0
    score += abs(feat_a["h_in"] - feat_b["h_in"]) * 2
    score += abs(feat_a["w_in"] - feat_b["w_in"]) * 2
    score += abs(feat_a["h_out"] - feat_b["h_out"]) * 2
    score += abs(feat_a["w_out"] - feat_b["w_out"]) * 2
    score += (0 if feat_a["same_size"] == feat_b["same_size"] else 10)
    score += abs(feat_a["n_colors_in"] - feat_b["n_colors_in"]) * 3
    score += abs(feat_a["n_objects"] - feat_b["n_objects"]) * 2
    score += abs(feat_a["new_colors"] - feat_b["new_colors"]) * 5
    score += abs(feat_a["size_ratio"] - feat_b["size_ratio"]) * 5
    return score


# ── Load solved examples ─────────────────────────────────────────

def load_solved_examples(tasks_dir, solved_dir):
    """Load all solved tasks with their solutions."""
    examples = []
    for sol_file in sorted(Path(solved_dir).glob("*.py")):
        task_id = sol_file.stem.replace("_llm", "")
        task_file = Path(tasks_dir) / f"{task_id}.json"
        if not task_file.exists():
            continue
        with open(task_file) as f:
            task = json.load(f)
        code = sol_file.read_text(encoding="utf-8")
        features = grid_features(task)
        examples.append({
            "task_id": task_id,
            "task": task,
            "code": code,
            "features": features,
        })
    # Also load C solutions (convert description to code)
    for sol_file in sorted(Path(solved_dir).glob("*.c")):
        task_id = sol_file.stem
        if any(e["task_id"] == task_id for e in examples):
            continue
        task_file = Path(tasks_dir) / f"{task_id}.json"
        if not task_file.exists():
            continue
        with open(task_file) as f:
            task = json.load(f)
        c_code = sol_file.read_text(encoding="utf-8")
        features = grid_features(task)
        examples.append({
            "task_id": task_id,
            "task": task,
            "code": c_code,  # C code as example
            "features": features,
        })
    return examples


def find_similar(target_features, examples, k=5):
    """Find k most similar solved examples."""
    scored = [(similarity(target_features, e["features"]), e) for e in examples]
    scored.sort(key=lambda x: x[0])
    return [e for _, e in scored[:k]]


# ── Prompt building ──────────────────────────────────────────────

def build_fewshot_prompt(target_task, similar_examples):
    """Build a prompt with similar solved tasks as examples."""
    lines = ["You are an expert ARC-AGI puzzle solver. Here are similar puzzles that have been solved:\n"]

    for i, ex in enumerate(similar_examples):
        lines.append(f"=== Solved Example {i+1} ===")
        for j, pair in enumerate(ex["task"]["train"][:2]):  # Show max 2 pairs per example
            inp, out = pair["input"], pair["output"]
            lines.append(f"Input ({len(inp)}x{len(inp[0])}):")
            lines.append(format_grid(inp))
            lines.append(f"Output ({len(out)}x{len(out[0])}):")
            lines.append(format_grid(out))
        lines.append(f"Solution:")
        # Show just the core logic (truncate long solutions)
        code = ex["code"]
        if len(code) > 800:
            code = code[:800] + "\n# ... (truncated)"
        lines.append(f"```python\n{code}\n```\n")

    lines.append("=== NEW PUZZLE TO SOLVE ===")
    for j, pair in enumerate(target_task["train"]):
        inp, out = pair["input"], pair["output"]
        lines.append(f"Training pair {j+1} ({len(inp)}x{len(inp[0])} -> {len(out)}x{len(out[0])}):")
        lines.append("Input:")
        lines.append(format_grid(inp))
        lines.append("Output:")
        lines.append(format_grid(out))
        lines.append("")

    lines.append("""Find the GENERAL transformation rule. Write a Python function:
```python
def solve(grid):
    # grid is list of lists of ints (0-9)
    # return the output grid
```
The function must work on ANY input following this pattern, not just these examples.
Write ONLY the function.""")

    return "\n".join(lines)


# ── Code extraction + verification ───────────────────────────────

def extract_python(text):
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    for block in blocks:
        if 'def solve' in block:
            return block.strip()
    if blocks:
        return blocks[-1].strip()
    lines = text.split('\n')
    func_lines = []
    in_func = False
    for line in lines:
        if re.match(r'^def solve\s*\(', line):
            in_func = True
        if in_func:
            func_lines.append(line)
    return "\n".join(func_lines).strip() if func_lines else None


def execute_solve(code, input_grid, timeout=15):
    prog = f"""import json, sys
{code}
grid = json.loads(sys.argv[1])
try:
    result = solve(grid)
    print(json.dumps(result))
except Exception as e:
    sys.exit(1)
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(prog)
        tmp = f.name
    try:
        proc = subprocess.run([sys.executable, tmp, json.dumps(input_grid)],
                              capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            return None
        out = proc.stdout.strip()
        return json.loads(out) if out else None
    except:
        return None
    finally:
        os.unlink(tmp)


def verify_all_pairs(code, task):
    for pair in task["train"]:
        result = execute_solve(code, pair["input"])
        if result is None or not grids_equal(result, pair["output"]):
            return False
    return True


# ── Main ─────────────────────────────────────────────────────────

def solve_batch(model_dir, tasks_dir, solved_dir, output_dir,
                batch_start, batch_size, attempts=10):
    from vllm import LLM, SamplingParams

    print(f"Loading model from {model_dir}...")
    llm = LLM(model=model_dir, tensor_parallel_size=1, dtype="float16",
              max_model_len=8192, gpu_memory_utilization=0.9)
    print("Model loaded.")

    # Load solved examples
    examples = load_solved_examples(tasks_dir, solved_dir)
    print(f"Loaded {len(examples)} solved examples for few-shot")

    # Load task list
    task_files = sorted(Path(tasks_dir).glob("*.json"))
    solved_ids = {e["task_id"] for e in examples}

    # Filter to unsolved only
    unsolved = [f for f in task_files if f.stem not in solved_ids]
    batch = unsolved[batch_start:batch_start + batch_size]
    print(f"Batch: {len(batch)} unsolved tasks")

    os.makedirs(output_dir, exist_ok=True)
    results = {}
    solved = 0

    TEMPS = [0.3, 0.6, 0.9]
    SAMPLES = max(1, attempts // len(TEMPS))

    for ti, task_file in enumerate(batch):
        task_id = task_file.stem
        with open(task_file) as f:
            task = json.load(f)

        features = grid_features(task)
        similar = find_similar(features, examples, k=5)

        prompt = build_fewshot_prompt(task, similar)
        task_solved = False
        total_attempts = 0

        print(f"\n[{ti+1}/{len(batch)}] {task_id} (similar: {[e['task_id'][:8] for e in similar]})",
              end="", flush=True)

        for temp in TEMPS:
            if task_solved:
                break

            params = SamplingParams(
                temperature=max(temp, 0.01),
                max_tokens=1024,
                n=SAMPLES,
                top_p=0.95,
            )

            try:
                outputs = llm.generate([prompt], params)
            except Exception as e:
                print("G", end="", flush=True)
                continue

            for output in outputs[0].outputs:
                total_attempts += 1
                code = extract_python(output.text)
                if code is None:
                    print(".", end="", flush=True)
                    continue

                if verify_all_pairs(code, task):
                    solved += 1
                    task_solved = True
                    results[task_id] = {
                        "solved": True,
                        "code": code,
                        "temperature": temp,
                        "attempts": total_attempts,
                        "similar_tasks": [e["task_id"] for e in similar],
                    }
                    print(f" SOLVED (t={temp} #{total_attempts})")
                    break
                else:
                    print("x", end="", flush=True)

        if not task_solved:
            results[task_id] = {"solved": False, "attempts": total_attempts}
            print(f" FAILED ({total_attempts})")

        # Save incrementally
        with open(os.path.join(output_dir, "fewshot_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results: {solved}/{len(batch)} solved")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/runpod-volume/models/qwen32b")
    parser.add_argument("--tasks-dir", default="/runpod-volume/arc_tasks")
    parser.add_argument("--solved-dir", default="/runpod-volume/arc_solved")
    parser.add_argument("--output-dir", default="/runpod-volume/arc_results")
    parser.add_argument("--batch-start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--attempts", type=int, default=10)
    args = parser.parse_args()

    solve_batch(args.model_dir, args.tasks_dir, args.solved_dir,
                args.output_dir, args.batch_start, args.batch_size,
                args.attempts)
