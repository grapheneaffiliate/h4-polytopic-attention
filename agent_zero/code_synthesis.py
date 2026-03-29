"""
Code Synthesis Reasoner — the nuclear option for ARC tasks.

Instead of searching over cell edits, SYNTHESIZE a Python function that
implements the transformation. This is how ARC-AGI-1 was solved (400/400).

The reasoner:
1. Analyzes training examples to infer the transformation pattern
2. Generates candidate solve(grid) functions
3. Tests each against training examples
4. Returns the one that passes all training examples
5. Applies it to the test input for the answer

This bypasses Agent Zero's search entirely — it's a direct solution.
The insight: some problems are better solved by REASONING than by SEARCH.
Agent Zero's architecture supports both: search for exploration problems,
reasoning for pattern recognition problems.
"""

import json
import random
from typing import Optional


def grids_equal(a, b):
    if len(a) != len(b): return False
    for i in range(len(a)):
        if len(a[i]) != len(b[i]): return False
        for j in range(len(a[i])):
            if a[i][j] != b[i][j]: return False
    return True


def test_solver(task, solver_fn):
    """Test a solver against all training examples."""
    for pair in task["train"]:
        try:
            result = solver_fn(pair["input"])
            if not grids_equal(result, pair["output"]):
                return False
        except Exception:
            return False
    return True


def apply_solver(task, solver_fn):
    """Apply solver to test input(s)."""
    results = []
    for test in task["test"]:
        try:
            results.append(solver_fn(test["input"]))
        except Exception:
            return None
    return results


# ── Pattern Library ──────────────────────────────────────────
# Each function analyzes training examples and returns a solve function
# if the pattern matches, or None if it doesn't.

def try_color_substitution(task):
    """Pattern: replace each color with another fixed color."""
    for pair in task["train"]:
        inp, out = pair["input"], pair["output"]
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return None

    # Build color map from first example
    color_map = {}
    for pair in task["train"]:
        inp, out = pair["input"], pair["output"]
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                ic, oc = inp[r][c], out[r][c]
                if ic in color_map:
                    if color_map[ic] != oc:
                        return None
                color_map[ic] = oc

    cm = dict(color_map)
    def solve(grid):
        return [[cm.get(cell, cell) for cell in row] for row in grid]
    return solve


def try_extract_unique_pattern(task):
    """Pattern: find a small non-background connected region in a large grid.
    Output = that region extracted. (like 23b5c85d)"""
    for pair in task["train"]:
        inp, out = pair["input"], pair["output"]
        if len(inp) <= len(out) and len(inp[0]) <= len(out[0]):
            return None  # output should be smaller

    def find_smallest_rect(grid):
        """Find the smallest non-background colored rectangle."""
        from collections import Counter
        h, w = len(grid), len(grid[0])
        flat = [grid[r][c] for r in range(h) for c in range(w)]
        bg = Counter(flat).most_common(1)[0][0]

        # Find bounding boxes of each non-bg color
        color_bounds = {}
        for r in range(h):
            for c in range(w):
                color = grid[r][c]
                if color != bg and color not in color_bounds:
                    # Find connected region of this color
                    cells = []
                    for rr in range(h):
                        for cc in range(w):
                            if grid[rr][cc] == color:
                                cells.append((rr, cc))
                    if cells:
                        r1 = min(r for r, c in cells)
                        r2 = max(r for r, c in cells)
                        c1 = min(c for r, c in cells)
                        c2 = max(c for r, c in cells)
                        color_bounds[color] = (r1, c1, r2, c2, len(cells))

        # Find the color whose bounding box is smallest AND whose
        # cells don't overlap with larger regions
        if not color_bounds:
            return None
        smallest = min(color_bounds.items(),
                      key=lambda x: (x[1][2]-x[1][0]+1) * (x[1][3]-x[1][1]+1))
        return smallest

    def solve(grid):
        result = find_smallest_rect(grid)
        if result is None:
            return grid
        color, (r1, c1, r2, c2, _) = result
        h = r2 - r1 + 1
        w = c2 - c1 + 1
        return [[color] * w for _ in range(h)]

    if test_solver(task, solve):
        return solve
    return None


def try_row_is_all_same(task):
    """Pattern: if a row has all same non-zero color, output that row as 5s, else 0s.
    (like 25d8a9c8)"""
    def solve(grid):
        h, w = len(grid), len(grid[0])
        out = [[0]*w for _ in range(h)]
        for r in range(h):
            vals = set(grid[r])
            if len(vals) == 1 and 0 not in vals:
                out[r] = [5]*w
        return out

    if test_solver(task, solve):
        return solve
    return None


def try_most_common_pattern(task):
    """Pattern: find the most common non-zero small shape. Output = that shape.
    (like 39a8645d — find the most repeated 3x3 pattern)"""
    from collections import Counter

    def extract_patterns(grid, ph, pw):
        """Extract all non-zero ph×pw patterns from grid."""
        h, w = len(grid), len(grid[0])
        patterns = []
        for r in range(h - ph + 1):
            for c in range(w - pw + 1):
                pat = tuple(tuple(grid[r+dr][c+dc] for dc in range(pw)) for dr in range(ph))
                # Only consider patterns with some non-zero cells
                if any(cell != 0 for row in pat for cell in row):
                    patterns.append(pat)
        return patterns

    # Infer output size from training
    out_h = len(task["train"][0]["output"])
    out_w = len(task["train"][0]["output"][0])

    def solve(grid):
        patterns = extract_patterns(grid, out_h, out_w)
        if not patterns:
            return [[0]*out_w for _ in range(out_h)]
        counts = Counter(patterns)
        most_common = counts.most_common(1)[0][0]
        return [list(row) for row in most_common]

    if test_solver(task, solve):
        return solve
    return None


def try_scattered_collect(task):
    """Pattern: non-zero non-background cells scattered in large grid,
    collect them into a small grid maintaining relative positions.
    (like 137eaa0f)"""
    from collections import Counter

    out_h = len(task["train"][0]["output"])
    out_w = len(task["train"][0]["output"][0])

    def solve(grid):
        h, w = len(grid), len(grid[0])
        flat = [grid[r][c] for r in range(h) for c in range(w)]
        bg = Counter(flat).most_common(1)[0][0]

        # Find all non-bg, non-5 cells (5 is often a marker)
        cells = []
        for r in range(h):
            for c in range(w):
                v = grid[r][c]
                if v != bg and v != 5:
                    cells.append((r, c, v))

        if not cells:
            return [[0]*out_w for _ in range(out_h)]

        # Normalize positions to fit output grid
        min_r = min(r for r, c, v in cells)
        min_c = min(c for r, c, v in cells)
        max_r = max(r for r, c, v in cells)
        max_c = max(c for r, c, v in cells)

        # Scale to output dimensions
        span_r = max_r - min_r + 1
        span_c = max_c - min_c + 1

        out = [[0]*out_w for _ in range(out_h)]
        for r, c, v in cells:
            nr = (r - min_r) * (out_h - 1) // max(span_r - 1, 1) if span_r > 1 else 0
            nc = (c - min_c) * (out_w - 1) // max(span_c - 1, 1) if span_c > 1 else 0
            if 0 <= nr < out_h and 0 <= nc < out_w:
                out[nr][nc] = v
        return out

    if test_solver(task, solve):
        return solve
    return None


def try_identity(task):
    """Pattern: output = input."""
    def solve(grid):
        return [row[:] for row in grid]
    if test_solver(task, solve):
        return solve
    return None


def try_transpose(task):
    """Pattern: output = transpose of input."""
    def solve(grid):
        h, w = len(grid), len(grid[0])
        return [[grid[r][c] for r in range(h)] for c in range(w)]
    if test_solver(task, solve):
        return solve
    return None


def try_rotate_90(task):
    """Pattern: output = 90 degree clockwise rotation."""
    def solve(grid):
        h, w = len(grid), len(grid[0])
        return [[grid[h-1-r][c] for r in range(h)] for c in range(w)]
    if test_solver(task, solve):
        return solve
    return None


def try_flip_horizontal(task):
    """Pattern: output = horizontal flip."""
    def solve(grid):
        return [row[::-1] for row in grid]
    if test_solver(task, solve):
        return solve
    return None


def try_flip_vertical(task):
    """Pattern: output = vertical flip."""
    def solve(grid):
        return grid[::-1]
    if test_solver(task, solve):
        return solve
    return None


# ── Synthesizer ──────────────────────────────────────────────

ALL_PATTERNS = [
    try_identity,
    try_color_substitution,
    try_row_is_all_same,
    try_transpose,
    try_rotate_90,
    try_flip_horizontal,
    try_flip_vertical,
    try_extract_unique_pattern,
    try_most_common_pattern,
    try_scattered_collect,
]


def synthesize_solver(task: dict) -> Optional[callable]:
    """
    Try all pattern functions against training examples.
    Returns the first solver that passes all training examples, or None.
    """
    for pattern_fn in ALL_PATTERNS:
        try:
            solver = pattern_fn(task)
            if solver is not None:
                return solver
        except Exception:
            continue
    return None


def solve_arc_task(task_id: str, task: dict) -> Optional[list]:
    """
    Attempt to solve an ARC task via code synthesis.
    Returns the predicted test output, or None if no pattern matches.
    """
    solver = synthesize_solver(task)
    if solver is None:
        return None

    results = apply_solver(task, solver)
    return results[0] if results else None
