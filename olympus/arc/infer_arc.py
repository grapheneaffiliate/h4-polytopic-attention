#!/usr/bin/env python3
"""
Cross-validated ARC inference with a fine-tuned model.

For each task:
1. Hold out 1 training pair
2. Generate C code hypothesis from remaining pairs
3. Verify against the non-held-out pairs (like normal)
4. ALSO verify against the held-out pair (generalization check)
5. If BOTH pass → the solution generalizes

Usage:
    python infer_arc.py \
        --model-dir /runpod-volume/arc_lora_r1 \
        --tasks-dir /runpod-volume/arc_tasks \
        --output-dir /runpod-volume/arc_results_r1 \
        --attempts 10 \
        --skip-solved solved_ids.json
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


# ── Grid helpers ─────────────────────────────────────────────────

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


# ── Prompt ───────────────────────────────────────────────────────

SYSTEM_MSG = (
    "You solve ARC-AGI puzzles by writing C code bodies. "
    "Given input/output grid examples, write C code that transforms any input grid to its output. "
    "Use arc_get(grid, w, row, col) to read, arc_set(out, ow, row, col, val) to write. "
    "Set oh and ow for output dimensions. No multiplication - use addition loops."
)


def make_prompt(pairs):
    """Format training pairs as prompt text."""
    lines = []
    for i, pair in enumerate(pairs):
        inp, out = pair["input"], pair["output"]
        h_i, w_i = len(inp), len(inp[0])
        h_o, w_o = len(out), len(out[0])
        lines.append(f"Example {i+1} ({h_i}x{w_i} -> {h_o}x{w_o}):")
        lines.append("Input:")
        lines.append(format_grid(inp))
        lines.append("Output:")
        lines.append(format_grid(out))
        lines.append("")
    return "\n".join(lines)


# ── Code extraction ──────────────────────────────────────────────

def extract_c_body(response):
    """Extract C code body from model response."""
    # Try code blocks
    blocks = re.findall(r'```(?:c|C)?\s*\n(.*?)```', response, re.DOTALL)
    if blocks:
        return blocks[-1].strip()

    # Take everything that looks like C
    lines = []
    for line in response.split('\n'):
        s = line.strip()
        if any(s.startswith(kw) for kw in ['oh ', 'oh=', 'ow ', 'ow=', 'int ', 'for ',
                                             'for(', 'if ', 'if(', 'while', '{', '}',
                                             'arc_', '/*', '//']):
            lines.append(line)
        elif lines and (s.startswith('}') or s.startswith('else') or line.startswith('    ')):
            lines.append(line)

    return "\n".join(lines).strip() if lines else None


# ── Python verification ──────────────────────────────────────────

def make_python_transform(c_body):
    """Convert C-style grid code to executable Python (best effort).

    This is a rough translator that handles common patterns from our
    training data. Falls back to direct execution via a C-to-Python
    transpiler approach.
    """
    # For cross-validation, we actually run the C code logic as Python
    # Build a Python version of the C code
    py = c_body

    # Replace C constructs with Python
    py = re.sub(r'arc_get\((\w+),\s*(\w+),\s*([^,]+),\s*([^)]+)\)',
                r'\1[\3][\4]', py)  # approximate - won't work for all cases

    # This is too fragile. Instead, let's verify by actually compiling
    # and running the C code through a Python subprocess.
    return None  # Fall back to subprocess verification


def verify_c_body_python(c_body, pairs, timeout=10):
    """Verify a C code body against grid pairs using Python execution.

    We wrap the C body logic in a Python function that simulates
    arc_get/arc_set operations on nested lists.
    """
    # Build a Python script that simulates the C grid operations
    py_script = '''
import json, sys

def arc_get(grid, w, row, col):
    return grid[row][col]

def arc_set(grid, w, row, col, val):
    grid[row][col] = val

def arc_fill(grid, h, w, val):
    for r in range(h):
        for c in range(w):
            grid[r][c] = val

def arc_copy(dst, src, h, w):
    for r in range(h):
        for c in range(w):
            dst[r][c] = src[r][c]

def solve(input_grid):
    h = len(input_grid)
    w = len(input_grid[0]) if h > 0 else 0

    # Create flat-ish grid representation matching C code expectations
    grid = [row[:] for row in input_grid]
    out = [[0]*30 for _ in range(30)]
    _tmp = [0]*32
    _qr = [0]*900
    _qc = [0]*900
    oh = h
    ow = w

    C_BODY_PLACEHOLDER

    return [out[r][:ow] for r in range(oh)]

pairs = json.loads(sys.argv[1])
results = []
for pair in pairs:
    try:
        actual = solve(pair["input"])
        expected = pair["output"]
        match = (len(actual) == len(expected) and
                 all(len(actual[r]) == len(expected[r]) and
                     all(actual[r][c] == expected[r][c] for c in range(len(expected[r])))
                     for r in range(len(expected))))
        results.append({"passed": match})
    except Exception as e:
        results.append({"passed": False, "error": str(e)[:100]})

print(json.dumps(results))
'''

    # Convert C body to Python-ish code
    py_body = c_body

    # Basic C-to-Python conversions
    py_body = re.sub(r'\bint\s+(\w+)\s*;', r'\1 = 0', py_body)  # int x; -> x = 0
    py_body = re.sub(r'\bint\s+(\w+)\s*=\s*', r'\1 = ', py_body)  # int x = -> x =
    py_body = re.sub(r'\bint\s+(\w+)\s*,\s*(\w+)\s*;', r'\1 = 0\n    \2 = 0', py_body)
    py_body = re.sub(r'\bfor\s*\(([^;]+);\s*([^;]+);\s*([^)]+)\)', _for_to_python, py_body)
    py_body = py_body.replace('&&', 'and').replace('||', 'or').replace('!', 'not ')
    py_body = py_body.replace('{', ':').replace('}', '')
    py_body = re.sub(r'//.*$', '', py_body, flags=re.MULTILINE)

    # This C-to-Python conversion is very fragile. For a more robust approach,
    # we should just compile and run the actual C. But for cross-validation
    # during training, this gives us a quick check.

    # Actually, let's not do C-to-Python translation. It's too error-prone.
    # Instead, use a simpler approach: just check if the code compiles and runs.
    return None


def _for_to_python(match):
    """Convert C for loop to Python while loop."""
    init, cond, incr = match.group(1), match.group(2), match.group(3)
    init = init.strip().rstrip(';')
    cond = cond.strip()
    incr = incr.strip()
    return f"{init}\n    while {cond}:"


# ── Direct subprocess verification ───────────────────────────────

def verify_against_pairs(c_body, pairs, timeout=10):
    """Verify C code body by wrapping in Python and executing.

    Uses a Python emulation of the arc_get/arc_set API.
    Returns (all_passed, results_list).
    """
    # Build full Python script
    script = f'''
import json, sys

def solve(input_grid):
    h = len(input_grid)
    w = len(input_grid[0]) if h > 0 else 0
    grid = input_grid  # read-only reference
    out = [[0]*30 for _ in range(30)]
    _tmp = [0]*32
    _qr = [0]*900
    _qc = [0]*900
    oh = h
    ow = w

    def arc_get(g, gw, r, c):
        return g[r][c]

    def arc_set(g, gw, r, c, v):
        g[r][c] = v

    def arc_fill(g, gh, gw, v):
        for rr in range(gh):
            for cc in range(gw):
                g[rr][cc] = v

    def arc_copy(dst, src, gh, gw):
        for rr in range(gh):
            for cc in range(gw):
                dst[rr][cc] = src[rr][cc]

    # ---- BEGIN GENERATED CODE ----
{_indent(c_body, 4)}
    # ---- END GENERATED CODE ----

    return [out[r][:ow] for r in range(oh)]

pairs = json.loads(sys.argv[1])
results = []
for pair in pairs:
    try:
        actual = solve(pair["input"])
        expected = pair["output"]
        match = actual == expected
        results.append({{"passed": match}})
    except Exception as e:
        results.append({{"passed": False, "error": str(e)[:100]}})

print(json.dumps(results))
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(script)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path, json.dumps(pairs)],
            capture_output=True, text=True, timeout=timeout
        )
        if proc.returncode != 0:
            return False, [{"passed": False, "error": proc.stderr[:200]}]

        results = json.loads(proc.stdout.strip())
        all_passed = all(r["passed"] for r in results)
        return all_passed, results

    except subprocess.TimeoutExpired:
        return False, [{"passed": False, "error": "timeout"}]
    except Exception as e:
        return False, [{"passed": False, "error": str(e)[:200]}]
    finally:
        os.unlink(tmp_path)


def _indent(code, spaces):
    """Indent each line of code by spaces."""
    prefix = " " * spaces
    lines = code.split("\n")
    # Convert C syntax to Python-compatible
    result = []
    for line in lines:
        # Basic C-to-Python: this is rough but handles our training data patterns
        l = line
        # Remove type declarations but keep assignments
        l = re.sub(r'^\s*int\s+(\w+)\s*=\s*', r'    \1 = ', l)
        l = re.sub(r'^\s*int\s+(\w+)\s*;', r'    \1 = 0', l)
        l = re.sub(r'^\s*int\s+(\w+),\s*(\w+)\s*;', r'    \1 = 0; \2 = 0', l)
        l = re.sub(r'^\s*int\s+(\w+),\s*(\w+),\s*(\w+)\s*;', r'    \1 = 0; \2 = 0; \3 = 0', l)
        result.append(prefix + l)
    return "\n".join(result)


# ── Main inference loop ──────────────────────────────────────────

def run_inference(model_dir, tasks_dir, output_dir, base_model, attempts=10,
                  skip_solved=None, temperatures=None):
    """Run cross-validated inference on all tasks."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    if temperatures is None:
        temperatures = [0.3, 0.5, 0.7, 0.9, 1.0]

    # Load model
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {model_dir}")
    model = PeftModel.from_pretrained(model, model_dir)
    model.eval()

    # Load skip list
    skip_ids = set()
    if skip_solved and os.path.exists(skip_solved):
        with open(skip_solved) as f:
            skip_ids = set(json.load(f))
        print(f"Skipping {len(skip_ids)} already-solved tasks")

    # Load tasks
    task_files = sorted(Path(tasks_dir).glob("*.json"))
    os.makedirs(output_dir, exist_ok=True)

    results = []
    solved = 0
    total = 0

    for ti, task_file in enumerate(task_files):
        task_id = task_file.stem
        if task_id in skip_ids:
            continue
        total += 1

        with open(task_file) as f:
            task = json.load(f)

        pairs = task["train"]
        if len(pairs) < 2:
            continue  # Need at least 2 pairs for cross-validation

        print(f"\n[{total}] {task_id} ({len(pairs)} pairs)", end="", flush=True)

        task_result = {
            "task_id": task_id,
            "solved": False,
            "c_body": None,
            "attempts": 0,
            "cv_passed": False,
        }

        # Cross-validation: hold out each pair in turn
        for holdout_idx in range(len(pairs)):
            if task_result["solved"]:
                break

            train_pairs = [p for i, p in enumerate(pairs) if i != holdout_idx]
            test_pair = pairs[holdout_idx]

            prompt_text = make_prompt(train_pairs)
            full_prompt = f"<|im_start|>system\n{SYSTEM_MSG}<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

            input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(model.device)

            for temp in temperatures:
                if task_result["solved"]:
                    break

                n_at_temp = max(1, attempts // len(temperatures))
                for _ in range(n_at_temp):
                    task_result["attempts"] += 1

                    try:
                        with torch.no_grad():
                            outputs = model.generate(
                                input_ids,
                                max_new_tokens=1024,
                                temperature=temp,
                                top_p=0.95,
                                do_sample=True,
                                pad_token_id=tokenizer.eos_token_id,
                            )

                        response = tokenizer.decode(outputs[0][input_ids.shape[1]:],
                                                     skip_special_tokens=True)
                        c_body = extract_c_body(response)
                        if c_body is None:
                            print(".", end="", flush=True)
                            continue

                        # Verify against train pairs (not held-out)
                        train_passed, _ = verify_against_pairs(c_body, train_pairs)
                        if not train_passed:
                            print(".", end="", flush=True)
                            continue

                        # Cross-validation: verify against held-out pair
                        cv_passed, _ = verify_against_pairs(c_body, [test_pair])
                        if cv_passed:
                            print(f" SOLVED (attempt {task_result['attempts']}, cv={holdout_idx})")
                            solved += 1
                            task_result["solved"] = True
                            task_result["c_body"] = c_body
                            task_result["cv_passed"] = True
                            break
                        else:
                            print("x", end="", flush=True)  # Train passed but CV failed

                    except Exception as e:
                        print("E", end="", flush=True)

        if not task_result["solved"]:
            print(f" FAILED ({task_result['attempts']} attempts)")

        results.append(task_result)

        # Save incrementally
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Round complete: {solved}/{total} new solves (cross-validated)")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="LoRA adapter directory")
    parser.add_argument("--tasks-dir", required=True, help="ARC task JSONs")
    parser.add_argument("--output-dir", required=True, help="Results output directory")
    parser.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    parser.add_argument("--attempts", type=int, default=10, help="Attempts per task")
    parser.add_argument("--skip-solved", default=None, help="JSON file with already-solved task IDs")
    args = parser.parse_args()

    run_inference(
        model_dir=args.model_dir,
        tasks_dir=args.tasks_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        attempts=args.attempts,
        skip_solved=args.skip_solved,
    )
