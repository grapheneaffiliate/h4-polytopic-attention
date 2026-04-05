"""Build fine-tuning JSONL from ARC solutions and task data."""
import json
import glob
import os
import re
import textwrap

BASE = "C:/Users/atchi/h4-polytopic-attention"

def is_solve_code(val):
    """Check if a value is actual Python solve code (not a grid output)."""
    if not isinstance(val, str):
        return False
    return 'def solve' in val


def extract_code(val):
    """Extract code string from a solution value."""
    if isinstance(val, dict):
        return val.get('code', '')
    if isinstance(val, str):
        return val
    return ''


def load_agi1_solutions():
    """Load all AGI-1 solutions from JSON files and standalone .py files."""
    solutions = {}

    # Load from JSON files
    for path in glob.glob(os.path.join(BASE, "data/arc_python_solutions*.json")):
        with open(path) as f:
            data = json.load(f)
        for task_id, val in data.items():
            code = extract_code(val)
            if code and is_solve_code(code) and task_id not in solutions:
                solutions[task_id] = code

    # Standalone: solve_234bbc79.py
    path_234 = os.path.join(BASE, "solve_234bbc79.py")
    if os.path.exists(path_234):
        with open(path_234) as f:
            content = f.read()
        # Extract from the imports through solve function (everything needed)
        # Find where solve function and its helpers start
        lines = content.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            # Skip docstring, imports of json, main block
            if line.startswith('def ') or in_code:
                in_code = True
                # Stop at def main
                if line.startswith('def main'):
                    break
                code_lines.append(line)
            elif line.startswith('import ') and 'json' not in line:
                code_lines.append(line)
        code = '\n'.join(code_lines).rstrip()
        if code:
            solutions['234bbc79'] = code

    # Standalone: solve_3631a71a.py
    path_3631 = os.path.join(BASE, "solve_3631a71a.py")
    if os.path.exists(path_3631):
        with open(path_3631) as f:
            content = f.read()
        lines = content.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if line.startswith('def solve') or in_code:
                in_code = True
                if line.startswith('if __name__'):
                    break
                code_lines.append(line)
        code = '\n'.join(code_lines).rstrip()
        if code:
            solutions['3631a71a'] = code

    return solutions

def load_agi2_solutions():
    """Load all AGI-2 solutions from JSON files."""
    solutions = {}
    for path in glob.glob(os.path.join(BASE, "data/arc2_solutions*.json")):
        with open(path) as f:
            data = json.load(f)
        for task_id, val in data.items():
            code = extract_code(val)
            if code and is_solve_code(code) and task_id not in solutions:
                solutions[task_id] = code
    return solutions

def load_task(task_id, task_dirs):
    """Load task JSON from one of the task directories."""
    for d in task_dirs:
        path = os.path.join(d, f"{task_id}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None

def format_input(task_data):
    """Format task data as the model input string."""
    parts = []

    # Training examples
    for i, pair in enumerate(task_data['train'], 1):
        parts.append(f"Input {i}: {pair['input']}")
        parts.append(f"Output {i}: {pair['output']}")

    # Test input(s) - typically one, but handle multiple
    for i, test in enumerate(task_data['test']):
        if len(task_data['test']) == 1:
            parts.append(f"Test input: {test['input']}")
        else:
            parts.append(f"Test input {i+1}: {test['input']}")

    return "Training examples:\n" + "\n".join(parts)

def main():
    agi1_dirs = [os.path.join(BASE, "data/arc1")]
    agi2_dirs = [os.path.join(BASE, "data/arc2")]

    agi1_solutions = load_agi1_solutions()
    agi2_solutions = load_agi2_solutions()

    print(f"Loaded {len(agi1_solutions)} AGI-1 solutions")
    print(f"Loaded {len(agi2_solutions)} AGI-2 solutions")

    output_path = os.path.join(BASE, "data/arc_finetune_all.jsonl")

    agi1_count = 0
    agi2_count = 0
    missing_tasks = []

    with open(output_path, 'w') as out:
        # AGI-1
        for task_id, code in sorted(agi1_solutions.items()):
            task_data = load_task(task_id, agi1_dirs + agi2_dirs)
            if task_data is None:
                missing_tasks.append(('AGI-1', task_id))
                continue
            entry = {
                "task_id": task_id,
                "input": format_input(task_data),
                "output": code
            }
            out.write(json.dumps(entry) + '\n')
            agi1_count += 1

        # AGI-2
        for task_id, code in sorted(agi2_solutions.items()):
            task_data = load_task(task_id, agi2_dirs + agi1_dirs)
            if task_data is None:
                missing_tasks.append(('AGI-2', task_id))
                continue
            entry = {
                "task_id": task_id,
                "input": format_input(task_data),
                "output": code
            }
            out.write(json.dumps(entry) + '\n')
            agi2_count += 1

    total = agi1_count + agi2_count
    file_size = os.path.getsize(output_path)

    print(f"\n=== Summary ===")
    print(f"Total entries: {total}")
    print(f"AGI-1 count:   {agi1_count}")
    print(f"AGI-2 count:   {agi2_count}")
    print(f"File size:     {file_size / 1024 / 1024:.2f} MB")
    print(f"Output:        {output_path}")

    if missing_tasks:
        print(f"\nMissing task files ({len(missing_tasks)}):")
        for source, tid in missing_tasks[:10]:
            print(f"  {source}: {tid}")
        if len(missing_tasks) > 10:
            print(f"  ... and {len(missing_tasks) - 10} more")

if __name__ == '__main__':
    main()
