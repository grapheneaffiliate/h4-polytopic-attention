#!/usr/bin/env python3
"""
Pod-side ARC solver for RunPod GPU instances.

Standalone script — no transformer-vm dependency. Uses Qwen2.5-Coder-32B
to generate Python transform functions, executes them directly, verifies
against training pairs. Saves all passing solutions.

Usage on pod:
    python pod_solver.py --tasks-dir /runpod-volume/arc_tasks/ \
                         --output-dir /runpod-volume/arc_results/ \
                         --model-path /runpod-volume/models/qwen32b.gguf \
                         --batch-start 0 --batch-size 50 \
                         --attempts 50

Results saved as JSON: {task_id, solved, code, attempts_needed, description}
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


# ── Grid helpers (self-contained, no imports) ────────────────────

def grids_equal(a, b):
    if len(a) != len(b):
        return False
    for r in range(len(a)):
        if len(a[r]) != len(b[r]):
            return False
        for c in range(len(a[r])):
            if a[r][c] != b[r][c]:
                return False
    return True


def format_grid(grid):
    return "\n".join(" ".join(str(v) for v in row) for row in grid)


# ── LLM prompt ───────────────────────────────────────────────────

SYSTEM_MSG = """You solve ARC-AGI puzzles by writing Python functions.

Given input/output grid pairs, find the transformation rule and implement it.

Write a Python function with this exact signature:
```python
def transform(grid):
    # grid is a list of lists of ints (0-9)
    # Return the transformed grid (list of lists of ints)
```

Rules:
- grid[r][c] gives the value at row r, column c
- 0 is usually background
- Return a new grid (don't modify the input)
- The function must work for ANY valid input, not just the examples
- Keep it simple and general"""


def make_prompt(task):
    lines = ["Here are the training examples:\n"]
    for i, pair in enumerate(task["train"]):
        inp, out = pair["input"], pair["output"]
        lines.append(f"Example {i+1}:")
        lines.append(f"Input ({len(inp)}x{len(inp[0])}):")
        lines.append(format_grid(inp))
        lines.append(f"Output ({len(out)}x{len(out[0])}):")
        lines.append(format_grid(out))
        lines.append("")

    lines.append("Write a Python function `def transform(grid):` that implements this transformation.")
    lines.append("Think about the pattern first, then write the code.")
    lines.append("Return ONLY the function definition in a ```python code block.")
    return "\n".join(lines)


# ── Code extraction and verification ─────────────────────────────

def extract_python(response):
    """Extract Python function from LLM response."""
    # Try fenced code block
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
    if blocks:
        for block in blocks:
            if 'def transform' in block:
                return block.strip()
        return blocks[-1].strip()

    # Try to find the function definition
    lines = response.split('\n')
    func_lines = []
    in_func = False
    for line in lines:
        if line.strip().startswith('def transform'):
            in_func = True
        if in_func:
            func_lines.append(line)
            # End of function: non-empty line at indent 0 that isn't def
            if func_lines and line.strip() and not line.startswith(' ') and not line.startswith('def'):
                func_lines.pop()
                break

    return "\n".join(func_lines) if func_lines else None


def verify_solution(code, task, timeout=10):
    """Execute Python code and verify against ALL training pairs.

    Returns (passed, details) where passed is True if all pairs match.
    """
    pairs = task["train"]
    results = []

    for i, pair in enumerate(pairs):
        inp = pair["input"]
        expected = pair["output"]

        # Execute in subprocess for safety
        test_code = f"""
import json, sys
{code}

inp = json.loads(sys.argv[1])
try:
    result = transform(inp)
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            tmp_path = f.name

        try:
            proc = subprocess.run(
                [sys.executable, tmp_path, json.dumps(inp)],
                capture_output=True, text=True, timeout=timeout
            )
            if proc.returncode != 0:
                results.append({"pair": i, "passed": False, "error": proc.stderr[:200]})
                continue

            output = proc.stdout.strip()
            if not output:
                results.append({"pair": i, "passed": False, "error": "no output"})
                continue

            actual = json.loads(output)
            if isinstance(actual, dict) and "error" in actual:
                results.append({"pair": i, "passed": False, "error": actual["error"]})
                continue

            if grids_equal(actual, expected):
                results.append({"pair": i, "passed": True})
            else:
                # Count differences
                h_e, h_a = len(expected), len(actual)
                w_e = len(expected[0]) if h_e > 0 else 0
                w_a = len(actual[0]) if h_a > 0 else 0
                if h_e != h_a or w_e != w_a:
                    results.append({"pair": i, "passed": False,
                                    "error": f"size {h_a}x{w_a} vs expected {h_e}x{w_e}"})
                else:
                    diffs = sum(1 for r in range(h_e) for c in range(w_e)
                                if actual[r][c] != expected[r][c])
                    total = h_e * w_e
                    results.append({"pair": i, "passed": False,
                                    "error": f"{diffs}/{total} cells wrong"})
        except subprocess.TimeoutExpired:
            results.append({"pair": i, "passed": False, "error": "timeout"})
        except Exception as e:
            results.append({"pair": i, "passed": False, "error": str(e)[:200]})
        finally:
            os.unlink(tmp_path)

    all_passed = all(r["passed"] for r in results)
    return all_passed, results


# ── Also generate C code for passing solutions ───────────────────

C_CONVERT_PROMPT = """Convert this Python ARC transform function to C code body.

Python function:
```python
{python_code}
```

Write ONLY the C code body for this template:
```c
void compute(const char *input) {{
    int *grid = _grid;  // input grid, row-major
    int *out = _out;    // output grid
    // h = input height, w = input width (already parsed)
    int h = _h_arr[0], w = _w_arr[0];
    int oh, ow;  // output dimensions — YOU MUST SET THESE

    // YOUR CODE HERE
    // Use arc_get(grid, w, row, col) to read input
    // Use arc_set(out, ow, row, col, val) to write output
    // No multiplication — use addition loops

    arc_emit_grid(out, oh, ow);
}}
```

Write ONLY the C body (no function definition, no includes):"""


def generate_c_version(llm, python_code, max_tokens=1000, temperature=0.3):
    """Ask the LLM to convert a passing Python solution to C."""
    prompt = C_CONVERT_PROMPT.format(python_code=python_code)
    response = llm(
        f"<|im_start|>system\nYou convert Python to C code.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n",
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<|im_end|>", "<|im_start|>"],
    )
    return response["choices"][0]["text"].strip()


# ── Main solver loop ─────────────────────────────────────────────

def solve_batch(model_path, tasks_dir, output_dir, batch_start, batch_size,
                attempts_per_task=50, n_gpu_layers=-1):
    """Solve a batch of ARC tasks using the LLM."""
    from llama_cpp import Llama

    # Load model
    print(f"Loading model: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    print("Model loaded.")

    # Load tasks
    task_files = sorted(Path(tasks_dir).glob("*.json"))
    batch = task_files[batch_start:batch_start + batch_size]
    print(f"Batch: {len(batch)} tasks (index {batch_start}-{batch_start + len(batch) - 1})")

    os.makedirs(output_dir, exist_ok=True)
    results = []
    solved = 0

    temperatures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for ti, task_file in enumerate(batch):
        task_id = task_file.stem
        with open(task_file) as f:
            task = json.load(f)

        print(f"\n[{ti+1}/{len(batch)}] {task_id}", end="", flush=True)
        prompt = make_prompt(task)
        full_prompt_template = (
            f"<|im_start|>system\n{SYSTEM_MSG}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        task_result = {
            "task_id": task_id,
            "solved": False,
            "python_code": None,
            "c_code": None,
            "attempts": 0,
            "temperature": None,
            "description": None,
        }

        attempt = 0
        for temp in temperatures:
            n_at_temp = attempts_per_task // len(temperatures)
            for _ in range(max(1, n_at_temp)):
                attempt += 1
                try:
                    response = llm(
                        full_prompt_template,
                        max_tokens=1500,
                        temperature=temp,
                        top_p=0.95,
                        stop=["<|im_end|>", "<|im_start|>"],
                    )
                    text = response["choices"][0]["text"]
                    code = extract_python(text)
                    if code is None:
                        print(".", end="", flush=True)
                        continue

                    passed, details = verify_solution(code, task)
                    if passed:
                        print(f" SOLVED (attempt {attempt}, temp={temp})")
                        solved += 1
                        task_result["solved"] = True
                        task_result["python_code"] = code
                        task_result["attempts"] = attempt
                        task_result["temperature"] = temp

                        # Try to get C version
                        try:
                            c_code = generate_c_version(llm, code)
                            task_result["c_code"] = c_code
                        except Exception:
                            pass

                        # Extract description from LLM response
                        desc_match = re.search(r'(?:pattern|rule|transformation)[:\s]+(.+?)(?:\n|$)',
                                               text, re.IGNORECASE)
                        if desc_match:
                            task_result["description"] = desc_match.group(1).strip()[:100]

                        break
                    else:
                        # Show progress
                        pair_results = [r["passed"] for r in details]
                        passed_count = sum(pair_results)
                        total_pairs = len(pair_results)
                        if passed_count > 0:
                            print(f"~{passed_count}/{total_pairs}", end="", flush=True)
                        else:
                            print(".", end="", flush=True)

                except Exception as e:
                    print(f"E", end="", flush=True)

                if task_result["solved"]:
                    break
            if task_result["solved"]:
                break

        if not task_result["solved"]:
            print(f" FAILED ({attempt} attempts)")

        task_result["attempts"] = attempt
        results.append(task_result)

        # Save incrementally
        result_path = Path(output_dir) / f"batch_{batch_start}.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print(f"Batch complete: {solved}/{len(batch)} solved")
    print(f"Results saved to: {output_dir}/batch_{batch_start}.json")
    return results


# ── Setup script (run on pod before solving) ─────────────────────

SETUP_SCRIPT = """#!/bin/bash
set -e

echo "=== ARC-AGI Pod Setup ==="

# Install llama-cpp-python with CUDA
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

# Download model if not cached
MODEL_DIR="/runpod-volume/models"
MODEL_PATH="$MODEL_DIR/qwen2.5-coder-32b-instruct-q4_k_m.gguf"
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading Qwen2.5-Coder-32B-Instruct Q4_K_M..."
    pip install huggingface-hub
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Qwen/Qwen2.5-Coder-32B-Instruct-GGUF',
    filename='qwen2.5-coder-32b-instruct-q4_k_m.gguf',
    local_dir='/runpod-volume/models/',
)
print('Download complete.')
"
else
    echo "Model already cached at $MODEL_PATH"
fi

echo "=== Setup complete ==="
"""


def write_setup_script(path="setup_pod.sh"):
    with open(path, "w", newline="\n") as f:
        f.write(SETUP_SCRIPT)
    print(f"Setup script written to {path}")


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARC-AGI Pod Solver")
    parser.add_argument("--tasks-dir", required=True, help="Directory with task JSONs")
    parser.add_argument("--output-dir", required=True, help="Directory for results")
    parser.add_argument("--model-path", required=True, help="Path to GGUF model")
    parser.add_argument("--batch-start", type=int, default=0, help="Start index in sorted task list")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of tasks per batch")
    parser.add_argument("--attempts", type=int, default=50, help="Max attempts per task")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="GPU layers (-1=all)")
    parser.add_argument("--setup", action="store_true", help="Write setup script and exit")
    args = parser.parse_args()

    if args.setup:
        write_setup_script()
        sys.exit(0)

    solve_batch(
        model_path=args.model_path,
        tasks_dir=args.tasks_dir,
        output_dir=args.output_dir,
        batch_start=args.batch_start,
        batch_size=args.batch_size,
        attempts_per_task=args.attempts,
        n_gpu_layers=args.n_gpu_layers,
    )
