"""
Grid I/O: serialize/deserialize ARC grids for transformer-vm C programs.

Format: "H W v00 v01 v02 ... v(H-1)(W-1)"
  - H = number of rows, W = number of columns
  - Values are 0-9 (ARC colors), space-separated, row-major order

Example: a 3x3 grid [[1,2,3],[4,5,6],[7,8,9]] becomes "3 3 1 2 3 4 5 6 7 8 9"
"""

from typing import List

Grid = List[List[int]]


def grid_to_string(grid: Grid) -> str:
    """Serialize a 2D grid to the C program input format."""
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    vals = []
    for row in grid:
        for v in row:
            vals.append(str(v))
    return f"{h} {w} " + " ".join(vals)


def string_to_grid(s: str) -> Grid:
    """Deserialize C program output back to a 2D grid."""
    parts = s.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Grid string too short: {s!r}")
    h = int(parts[0])
    w = int(parts[1])
    vals = [int(x) for x in parts[2:]]
    if len(vals) != h * w:
        raise ValueError(f"Expected {h}*{w}={h*w} values, got {len(vals)}")
    grid = []
    for r in range(h):
        grid.append(vals[r * w : r * w + w])
    return grid


def grids_equal(a: Grid, b: Grid) -> bool:
    """Check if two grids are identical cell-by-cell."""
    if len(a) != len(b):
        return False
    for r in range(len(a)):
        if len(a[r]) != len(b[r]):
            return False
        for c in range(len(a[r])):
            if a[r][c] != b[r][c]:
                return False
    return True


def grid_diff(expected: Grid, actual: Grid) -> dict:
    """Compare two grids and return detailed diff info."""
    errors = []
    h_exp, h_act = len(expected), len(actual)
    w_exp = len(expected[0]) if h_exp > 0 else 0
    w_act = len(actual[0]) if h_act > 0 else 0

    if h_exp != h_act or w_exp != w_act:
        return {
            "match": False,
            "size_mismatch": True,
            "expected_size": (h_exp, w_exp),
            "actual_size": (h_act, w_act),
            "errors": [],
        }

    for r in range(h_exp):
        for c in range(w_exp):
            if expected[r][c] != actual[r][c]:
                errors.append({
                    "row": r, "col": c,
                    "expected": expected[r][c],
                    "actual": actual[r][c],
                })

    return {
        "match": len(errors) == 0,
        "size_mismatch": False,
        "expected_size": (h_exp, w_exp),
        "actual_size": (h_act, w_act),
        "errors": errors,
        "error_count": len(errors),
        "total_cells": h_exp * w_exp,
        "accuracy": 1.0 - len(errors) / (h_exp * w_exp) if h_exp * w_exp > 0 else 0,
    }


def format_grid(grid: Grid) -> str:
    """Pretty-print a grid for display."""
    lines = []
    for row in grid:
        lines.append(" ".join(str(v) for v in row))
    return "\n".join(lines)
