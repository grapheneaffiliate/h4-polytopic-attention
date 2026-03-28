"""
ARC-AGI-2 retry solver for 26 tasks.
Only saves solutions that pass ALL training pairs.
"""
import json
from collections import Counter, defaultdict

DATA_DIR = "data/arc2"
OUT_FILE = "data/arc2_solutions_retry1.json"

TASK_IDS = [
    "409aa875","446ef5d2","4a21e3da","4c3d4a41","4c416de3","4c7dc4dd",
    "4e34c42c","53fb4810","5545f144","581f7754","58f5dbd5","5961cc34",
    "5dbc8537","62593bfd","64efde09","65b59efc","67e490f4","6e4f6532",
    "6ffbe589","71e489b6","7491f3cf","7666fa5d","7b0280bc","7b3084d4",
    "7b80bb43","7c66cb00"
]

def load_task(task_id):
    with open(f"{DATA_DIR}/{task_id}.json") as f:
        return json.load(f)


def solve_7491f3cf(grid):
    """4 sections: A defines split axis, B and C fill opposite sides, D gets result."""
    R, C = len(grid), len(grid[0])
    bg = grid[0][0]
    sec_starts = []
    in_sec = False
    for c in range(C):
        if grid[1][c] != bg:
            if not in_sec:
                sec_starts.append(c)
                in_sec = True
        else:
            in_sec = False
    sections = []
    for s in sec_starts:
        w = 0
        while s + w < C and grid[1][s+w] != bg:
            w += 1
        sec = [[grid[r][cc] for cc in range(s, s+w)] for r in range(1, 6)]
        sections.append((s, sec))
    if len(sections) != 4:
        return None
    inner_bg = Counter([v for row in sections[-1][1] for v in row]).most_common(1)[0][0]
    a = sections[0][1]
    b = sections[1][1]
    cs = sections[2][1]
    w = len(a[0])
    ls = sections[-1][0]

    a_nonbg = [(r, c) for r in range(5) for c in range(w) if a[r][c] != inner_bg]
    if not a_nonbg:
        return None

    def compute_vals(name, pts):
        if name == 'r': return [r for r, c in pts]
        if name == 'c': return [c for r, c in pts]
        if name == 'r+c': return [r + c for r, c in pts]
        if name == 'r-c': return [r - c for r, c in pts]

    candidates = []
    for name in ['r', 'c', 'r+c', 'r-c']:
        vals = compute_vals(name, a_nonbg)
        mode_val, mode_count = Counter(vals).most_common(1)[0]
        var = max(vals) - min(vals)
        candidates.append((var, -mode_count, name, mode_val))
    candidates.sort()
    axis_type = candidates[0][2]
    axis_val = candidates[0][3]

    vals = compute_vals(axis_type, a_nonbg)
    outliers = [(r, c, v) for (r, c), v in zip(a_nonbg, vals) if v != axis_val]
    b_fills_less = outliers[0][2] < axis_val if outliers else True

    pred = [[inner_bg] * w for _ in range(5)]
    for r in range(5):
        for c in range(w):
            if axis_type == 'r+c': cell_val = r + c
            elif axis_type == 'r-c': cell_val = r - c
            elif axis_type == 'r': cell_val = r
            else: cell_val = c

            if b_fills_less:
                b_side = cell_val < axis_val
                c_side = cell_val > axis_val
            else:
                b_side = cell_val > axis_val
                c_side = cell_val < axis_val
            boundary = cell_val == axis_val

            if boundary:
                if b[r][c] != inner_bg:
                    pred[r][c] = b[r][c]
                elif cs[r][c] != inner_bg:
                    pred[r][c] = cs[r][c]
            elif b_side:
                if b[r][c] != inner_bg:
                    pred[r][c] = b[r][c]
            elif c_side:
                if cs[r][c] != inner_bg:
                    pred[r][c] = cs[r][c]

    out = [row[:] for row in grid]
    for r in range(5):
        for c in range(w):
            out[r + 1][ls + c] = pred[r][c]
    return out


def solve_default(grid):
    return None


SOLVERS = {
    "7491f3cf": solve_7491f3cf,
}

for tid in TASK_IDS:
    if tid not in SOLVERS:
        SOLVERS[tid] = solve_default


def test_solver(task_id, solver):
    task = load_task(task_id)
    for pair in task['train']:
        result = solver(pair['input'])
        if result is None:
            return False
        if [list(r) for r in result] != [list(r) for r in pair['output']]:
            return False
    return True


def main():
    solutions = {}
    passed = []
    failed = []
    skipped = []

    for task_id in TASK_IDS:
        solver = SOLVERS.get(task_id, solve_default)
        if solver == solve_default:
            skipped.append(task_id)
            continue

        try:
            if test_solver(task_id, solver):
                passed.append(task_id)
                task = load_task(task_id)
                task_solutions = []
                for test_pair in task['test']:
                    result = solver(test_pair['input'])
                    if result is not None:
                        task_solutions.append([list(r) for r in result])
                if task_solutions:
                    solutions[task_id] = task_solutions
            else:
                failed.append(task_id)
                print(f"FAILED: {task_id}")
        except Exception as e:
            failed.append(task_id)
            print(f"Error on {task_id}: {e}")

    print(f"\nResults:")
    print(f"  Passed: {len(passed)} - {passed}")
    print(f"  Failed: {len(failed)} - {failed}")
    print(f"  Skipped: {len(skipped)} - {skipped}")

    with open(OUT_FILE, 'w') as f:
        json.dump(solutions, f)
    print(f"\nSaved {len(solutions)} solutions to {OUT_FILE}")


if __name__ == "__main__":
    main()
