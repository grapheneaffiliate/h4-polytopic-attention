#!/usr/bin/env python3
"""Re-solve 38 previously solved ARC-AGI-1 tasks."""

import json
import copy
from collections import deque

# ─── Task solvers ───

def solve_007bbfb7(grid):
    """Each non-zero cell in 3x3 input becomes a copy of the grid; zero cells become 3x3 zeros."""
    n = len(grid)
    out = [[0]*(n*n) for _ in range(n*n)]
    for r in range(n):
        for c in range(n):
            if grid[r][c] != 0:
                for dr in range(n):
                    for dc in range(n):
                        out[r*n+dr][c*n+dc] = grid[dr][dc]
    return out

def solve_00d62c1b(grid):
    """Fill enclosed regions within green(3) borders with yellow(4)."""
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    border_reachable = [[False]*cols for _ in range(rows)]
    q = deque()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and (r == 0 or r == rows-1 or c == 0 or c == cols-1):
                border_reachable[r][c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and not border_reachable[nr][nc] and grid[nr][nc] == 0:
                border_reachable[nr][nc] = True
                q.append((nr, nc))
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and not border_reachable[r][c]:
                out[r][c] = 4
    return out

def solve_0520fde7(grid):
    """Two halves split by col of 5s. Where both left=1 AND right=1, output 2."""
    rows = len(grid)
    sep = next(c for c in range(len(grid[0])) if all(grid[r][c] == 5 for r in range(rows)))
    left = [row[:sep] for row in grid]
    right = [row[sep+1:] for row in grid]
    n = len(left[0])
    out = [[0]*n for _ in range(rows)]
    for r in range(rows):
        for c in range(n):
            if left[r][c] == 1 and right[r][c] == 1:
                out[r][c] = 2
    return out

def solve_08ed6ac7(grid):
    """Columns of 5s get numbered 1,2,3,4 by length (longest=1)."""
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    col_lengths = {}
    for c in range(cols):
        length = sum(1 for r in range(rows) if grid[r][c] == 5)
        if length > 0:
            col_lengths[c] = length
    sorted_cols = sorted(col_lengths.keys(), key=lambda c: -col_lengths[c])
    color = 1
    for col in sorted_cols:
        for r in range(rows):
            if grid[r][col] == 5:
                out[r][col] = color
        color += 1
    return out

def solve_0ca9ddb6(grid):
    """Red(2) gets diagonal yellow(4) cross; blue(1) gets orthogonal orange(7) cross."""
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                for d in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r+d[0], c+d[1]
                    if 0 <= nr < rows and 0 <= nc < cols and out[nr][nc] == 0:
                        out[nr][nc] = 4
            elif grid[r][c] == 1:
                for d in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+d[0], c+d[1]
                    if 0 <= nr < rows and 0 <= nc < cols and out[nr][nc] == 0:
                        out[nr][nc] = 7
    return out

def solve_0d3d703e(grid):
    """Color mapping: 1->5, 2->6, 3->4, 4->3, 5->1, 6->2, 8->9, 9->8."""
    mapping = {1:5, 2:6, 3:4, 4:3, 5:1, 6:2, 8:9, 9:8, 0:0, 7:7}
    return [[mapping.get(c, c) for c in row] for row in grid]

def solve_178fcbfb(grid):
    """1->fill row, 2->fill col, 3->fill row. Draw order: 2 first, then 3, then 1."""
    rows, cols = len(grid), len(grid[0])
    out = [[0]*cols for _ in range(rows)]
    positions = {1: [], 2: [], 3: []}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v in positions:
                positions[v].append((r, c))
    # Draw in order: 2 (vertical), then 3 (horizontal), then 1 (horizontal)
    for r, c in positions[2]:
        for rr in range(rows):
            out[rr][c] = 2
    for r, c in positions[3]:
        for cc in range(cols):
            out[r][cc] = 3
    for r, c in positions[1]:
        for cc in range(cols):
            out[r][cc] = 1
    return out

def solve_1cf80156(grid):
    """Crop the non-zero region (bounding box)."""
    rows, cols = len(grid), len(grid[0])
    min_r, max_r, min_c, max_c = rows, -1, cols, -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    return [row[min_c:max_c+1] for row in grid[min_r:max_r+1]]

def solve_1e0a9b12(grid):
    """Gravity: non-zero values fall to the bottom of their column."""
    rows, cols = len(grid), len(grid[0])
    out = [[0]*cols for _ in range(rows)]
    for c in range(cols):
        vals = [grid[r][c] for r in range(rows) if grid[r][c] != 0]
        for i, v in enumerate(reversed(vals)):
            out[rows-1-i][c] = v
    return out

def solve_22168020(grid):
    """For each row, find pairs of same non-zero value and fill between them."""
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(rows):
        # Find non-zero positions
        nz = [(c, grid[r][c]) for c in range(cols) if grid[r][c] != 0]
        # For pairs of same value, fill between
        for i in range(len(nz)):
            for j in range(i+1, len(nz)):
                c1, v1 = nz[i]
                c2, v2 = nz[j]
                if v1 == v2:
                    for c in range(c1, c2+1):
                        out[r][c] = v1
    return out

def solve_22eb0ac0(grid):
    """Triangle fill: fill between diagonal arms of same-color groups."""
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    colors_positions = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != 0:
                if v not in colors_positions:
                    colors_positions[v] = []
                colors_positions[v].append((r, c))
    for color, positions in colors_positions.items():
        for r in range(min(p[0] for p in positions), max(p[0] for p in positions)+1):
            cols_in_row = sorted([c for rr, c in positions if rr == r])
            if len(cols_in_row) >= 2:
                for c in range(cols_in_row[0], cols_in_row[-1]+1):
                    out[r][c] = color
    return out

def solve_253bf280(grid):
    """Pairs of 8s: same row -> fill between with 3. Same col -> fill between with 3."""
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    eights = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 8]
    # Same row pairs
    from collections import defaultdict
    by_row = defaultdict(list)
    by_col = defaultdict(list)
    for r, c in eights:
        by_row[r].append(c)
        by_col[c].append(r)
    for r, cs in by_row.items():
        if len(cs) == 2:
            cs.sort()
            for c in range(cs[0]+1, cs[1]):
                out[r][c] = 3
    for c, rs in by_col.items():
        if len(rs) == 2:
            rs.sort()
            for r in range(rs[0]+1, rs[1]):
                out[r][c] = 3
    return out

def solve_3428a4f5(grid):
    """Two halves separated by row of 4s. XOR: where exactly one has 2, output 3."""
    rows, cols = len(grid), len(grid[0])
    sep = next(r for r in range(rows) if all(grid[r][c] == 4 for c in range(cols)))
    top = grid[:sep]
    bottom = grid[sep+1:]
    out = [[0]*cols for _ in range(len(top))]
    for r in range(len(top)):
        for c in range(cols):
            t = (top[r][c] != 0)
            b = (bottom[r][c] != 0)
            if t != b:
                out[r][c] = 3
    return out

def solve_3906de3d(grid):
    """2-columns slide upward into 0-holes in the 1-block."""
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # Find boundary: rows with 1s vs rows with 2s
    one_rows = set()
    for r in range(rows):
        if any(grid[r][c] == 1 for c in range(cols)):
            one_rows.add(r)
    if not one_rows:
        return out

    # The 1-block rows
    min_one = min(one_rows)
    max_one = max(one_rows)

    # For each column, find 2-values below the 1-block
    for c in range(cols):
        # Count 2s in this column below the block
        twos_count = 0
        two_positions = []
        for r in range(rows):
            if grid[r][c] == 2:
                twos_count += 1
                two_positions.append(r)

        if twos_count == 0:
            continue

        # Find holes (0s) in the 1-block for this column, from top to bottom
        holes = []
        for r in range(min_one, max_one + 1):
            if grid[r][c] == 0:
                # Check if this is truly inside the 1-block (row has 1s)
                if any(grid[r][cc] == 1 for cc in range(cols)):
                    holes.append(r)

        # Clear original 2s
        for r in two_positions:
            out[r][c] = 0

        # Fill holes from top with 2s
        filled = 0
        for r in holes:
            if filled < twos_count:
                out[r][c] = 2
                filled += 1

        # Remaining 2s go just below the block
        remaining = twos_count - filled
        if remaining > 0:
            # Place remaining 2s below the last hole or below the block
            start_r = max_one + 1
            placed = 0
            for r in range(start_r, rows):
                if placed >= remaining:
                    break
                out[r][c] = 2
                placed += 1

    return out

def solve_3aa6fb7a(grid):
    """L-shaped 8-groups: find the missing corner of each 2x2 and fill with 1."""
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(rows - 1):
        for c in range(cols - 1):
            # Check each 2x2 block
            block = [grid[r][c], grid[r][c+1], grid[r+1][c], grid[r+1][c+1]]
            count_8 = block.count(8)
            count_0 = block.count(0)
            if count_8 == 3 and count_0 == 1:
                # Find the 0 and fill with 1
                if grid[r][c] == 0: out[r][c] = 1
                elif grid[r][c+1] == 0: out[r][c+1] = 1
                elif grid[r+1][c] == 0: out[r+1][c] = 1
                elif grid[r+1][c+1] == 0: out[r+1][c+1] = 1
    return out

def solve_3c9b0459(grid):
    """Rotate 180 degrees."""
    return [row[::-1] for row in grid[::-1]]

def solve_4258a5f9(grid):
    """Draw 3x3 border of 1s around each 5."""
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < rows and 0 <= nc < cols and out[nr][nc] == 0:
                            out[nr][nc] = 1
    return out

def solve_5bd6f4ac(grid):
    """Extract the top-right 3x3 block from a 9x9 grid."""
    return [row[6:9] for row in grid[0:3]]

def solve_6150a2bd(grid):
    """Rotate 180 degrees."""
    return [row[::-1] for row in grid[::-1]]

def solve_67a3c6ac(grid):
    """Reverse each row (horizontal flip)."""
    return [row[::-1] for row in grid]

def solve_68b16354(grid):
    """Flip vertically (reverse row order)."""
    return grid[::-1]

def solve_74dd1130(grid):
    """Transpose the matrix."""
    rows, cols = len(grid), len(grid[0])
    return [[grid[r][c] for r in range(rows)] for c in range(cols)]

def solve_99b1bc43(grid):
    """Two halves separated by row of 4s. XOR: where exactly one has non-zero -> 3."""
    rows, cols = len(grid), len(grid[0])
    sep = next(r for r in range(rows) if all(grid[r][c] == 4 for c in range(cols)))
    top = grid[:sep]
    bottom = grid[sep+1:]
    out = [[0]*cols for _ in range(len(top))]
    for r in range(len(top)):
        for c in range(cols):
            t = (top[r][c] != 0)
            b = (bottom[r][c] != 0)
            if t != b:
                out[r][c] = 3
    return out

def solve_9dfd6313(grid):
    """Transpose across main diagonal."""
    n = len(grid)
    return [[grid[c][r] for c in range(n)] for r in range(n)]

def solve_a416b8f3(grid):
    """Tile the grid horizontally (duplicate each row)."""
    return [row + row for row in grid]

def solve_a5313dff(grid):
    """Fill enclosed 0-regions bounded by 2s with 1."""
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    visited = [[False]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and not visited[r][c]:
                q = deque([(r, c)])
                visited[r][c] = True
                cells = [(r, c)]
                touches_border = False
                while q:
                    cr, cc = q.popleft()
                    if cr == 0 or cr == rows-1 or cc == 0 or cc == cols-1:
                        touches_border = True
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if grid[nr][nc] == 0 and not visited[nr][nc]:
                                visited[nr][nc] = True
                                q.append((nr, nc))
                                cells.append((nr, nc))
                        else:
                            touches_border = True
                if not touches_border:
                    for cr, cc in cells:
                        out[cr][cc] = 1
    return out

def solve_a699fb00(grid):
    """Fill between 1s on same row with 2s."""
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(rows):
        ones = [c for c in range(cols) if grid[r][c] == 1]
        if len(ones) >= 2:
            start, end = ones[0], ones[-1]
            for c in range(start+1, end):
                if grid[r][c] == 0:
                    out[r][c] = 2
    return out

def solve_aabf363d(grid):
    """Replace shape color with indicator color from bottom-left corner. Clear indicator."""
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    indicator = grid[rows-1][0]
    shape_color = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and grid[r][c] != indicator:
                shape_color = grid[r][c]
                break
        if shape_color:
            break
    if shape_color is None:
        return out
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == shape_color:
                out[r][c] = indicator
            elif grid[r][c] == indicator:
                out[r][c] = 0
    return out

def solve_b1948b0a(grid):
    """Replace 6 with 2, keep 7."""
    return [[2 if c == 6 else c for c in row] for row in grid]

def solve_c59eb873(grid):
    """Scale up 2x: each cell becomes a 2x2 block."""
    rows, cols = len(grid), len(grid[0])
    out = []
    for r in range(rows):
        row1 = []
        for c in range(cols):
            row1.extend([grid[r][c], grid[r][c]])
        out.append(row1)
        out.append(row1[:])
    return out

def solve_c8f0f002(grid):
    """Replace 7 with 5."""
    return [[5 if c == 7 else c for c in row] for row in grid]

def solve_d10ecb37(grid):
    """Find the minimal tile that, with all 8 symmetries, generates the grid."""
    rows, cols = len(grid), len(grid[0])

    def get_symmetries(tile):
        results = set()
        def to_tuple(t): return tuple(tuple(r) for r in t)
        def rot90(t):
            r, c = len(t), len(t[0])
            return [[t[r-1-j][i] for j in range(r)] for i in range(c)]
        def flip_h(t): return [row[::-1] for row in t]
        current = [row[:] for row in tile]
        for _ in range(4):
            results.add(to_tuple(current))
            results.add(to_tuple(flip_h(current)))
            current = rot90(current)
        return results

    for th in range(1, rows+1):
        if rows % th != 0:
            continue
        for tw in range(1, cols+1):
            if cols % tw != 0:
                continue
            if th == rows and tw == cols:
                continue
            tile = [grid[r][:tw] for r in range(th)]
            syms = get_symmetries(tile)
            ok = True
            for br in range(0, rows, th):
                for bc in range(0, cols, tw):
                    block = tuple(tuple(grid[r][bc:bc+tw]) for r in range(br, br+th))
                    if block not in syms:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return tile
    return grid

def solve_d511f180(grid):
    """Swap 5s and 8s."""
    return [[8 if c == 5 else (5 if c == 8 else c) for c in row] for row in grid]

def solve_dbc1a6ce(grid):
    """Connect pairs of 1s on same row/col with 8s between them."""
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    ones = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    for i in range(len(ones)):
        for j in range(i+1, len(ones)):
            r1, c1 = ones[i]
            r2, c2 = ones[j]
            if r1 == r2:
                for c in range(min(c1, c2)+1, max(c1, c2)):
                    if out[r1][c] == 0:
                        out[r1][c] = 8
            elif c1 == c2:
                for r in range(min(r1, r2)+1, max(r1, r2)):
                    if out[r][c1] == 0:
                        out[r][c1] = 8
    return out

def solve_dc1df850(grid):
    """2 gets surrounded by 1s (3x3 ring)."""
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                            out[nr][nc] = 1
    return out

def solve_ded97339(grid):
    """Connect pairs of 8s: pairs on same row → horizontal line, pairs on same col → vertical line.
    For unpaired 8s with shared column to another pair's endpoint, connect vertically."""
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    eights = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 8]

    from collections import defaultdict
    by_row = defaultdict(list)
    by_col = defaultdict(list)
    for r, c in eights:
        by_row[r].append(c)
        by_col[c].append(r)

    # Connect pairs on same row
    for r, cs in by_row.items():
        if len(cs) == 2:
            c1, c2 = sorted(cs)
            for c in range(c1, c2+1):
                out[r][c] = 8

    # Connect pairs on same col
    for c, rs in by_col.items():
        if len(rs) == 2:
            r1, r2 = sorted(rs)
            for r in range(r1, r2+1):
                out[r][c] = 8

    return out

def solve_ed36ccf7(grid):
    """Rotate 90 degrees counter-clockwise."""
    rows, cols = len(grid), len(grid[0])
    return [[grid[r][c] for r in range(rows)] for c in range(cols-1, -1, -1)]

def solve_f25ffba3(grid):
    """Mirror staircase: output is symmetric around the middle, row i = merge of row i and row (n-1-i)."""
    rows, cols = len(grid), len(grid[0])
    result = [None] * rows
    for i in range(rows):
        j = rows - 1 - i
        row_i = grid[i]
        row_j = grid[j]
        merged = []
        for c in range(cols):
            if row_i[c] != 0:
                merged.append(row_i[c])
            elif row_j[c] != 0:
                merged.append(row_j[c])
            else:
                merged.append(0)
        result[i] = merged
    return result

# ─── Task registry ───
SOLVERS = {
    "007bbfb7": solve_007bbfb7,
    "00d62c1b": solve_00d62c1b,
    "0520fde7": solve_0520fde7,
    "08ed6ac7": solve_08ed6ac7,
    "0ca9ddb6": solve_0ca9ddb6,
    "0d3d703e": solve_0d3d703e,
    "178fcbfb": solve_178fcbfb,
    "1cf80156": solve_1cf80156,
    "1e0a9b12": solve_1e0a9b12,
    "22168020": solve_22168020,
    "22eb0ac0": solve_22eb0ac0,
    "253bf280": solve_253bf280,
    "3428a4f5": solve_3428a4f5,
    "3906de3d": solve_3906de3d,
    "3aa6fb7a": solve_3aa6fb7a,
    "3c9b0459": solve_3c9b0459,
    "4258a5f9": solve_4258a5f9,
    "5bd6f4ac": solve_5bd6f4ac,
    "6150a2bd": solve_6150a2bd,
    "67a3c6ac": solve_67a3c6ac,
    "68b16354": solve_68b16354,
    "74dd1130": solve_74dd1130,
    "99b1bc43": solve_99b1bc43,
    "9dfd6313": solve_9dfd6313,
    "a416b8f3": solve_a416b8f3,
    "a5313dff": solve_a5313dff,
    "a699fb00": solve_a699fb00,
    "aabf363d": solve_aabf363d,
    "b1948b0a": solve_b1948b0a,
    "c59eb873": solve_c59eb873,
    "c8f0f002": solve_c8f0f002,
    "d10ecb37": solve_d10ecb37,
    "d511f180": solve_d511f180,
    "dbc1a6ce": solve_dbc1a6ce,
    "dc1df850": solve_dc1df850,
    "ded97339": solve_ded97339,
    "ed36ccf7": solve_ed36ccf7,
    "f25ffba3": solve_f25ffba3,
}

def test_solver(task_id, solver):
    """Test solver against all training pairs. Return True if all pass."""
    with open(f"data/arc1/{task_id}.json") as f:
        task = json.load(f)

    for i, pair in enumerate(task["train"]):
        try:
            result = solver(pair["input"])
            if result != pair["output"]:
                print(f"  FAIL {task_id} train[{i}]")
                expected = pair["output"]
                for r in range(min(len(result), len(expected))):
                    if r < len(result) and r < len(expected) and result[r] != expected[r]:
                        print(f"    Row {r}: got {result[r]}")
                        print(f"    Row {r}: exp {expected[r]}")
                        break
                if len(result) != len(expected):
                    print(f"    Size: got {len(result)}x{len(result[0]) if result else 0}, exp {len(expected)}x{len(expected[0]) if expected else 0}")
                return False
        except Exception as e:
            print(f"  ERROR {task_id} train[{i}]: {e}")
            import traceback
            traceback.print_exc()
            return False
    return True

def main():
    passed = {}
    failed = []

    for task_id, solver in SOLVERS.items():
        if test_solver(task_id, solver):
            print(f"  PASS {task_id}")
            passed[task_id] = solver
        else:
            failed.append(task_id)

    print(f"\nPassed: {len(passed)}/{len(SOLVERS)}")
    if failed:
        print(f"Failed: {failed}")

    # Generate solutions for test pairs
    solutions = {}
    for task_id, solver in passed.items():
        with open(f"data/arc1/{task_id}.json") as f:
            task = json.load(f)
        test_outputs = []
        for pair in task["test"]:
            test_outputs.append(solver(pair["input"]))
        solutions[task_id] = test_outputs

    with open("data/arc_python_solutions_recovery.json", "w") as f:
        json.dump(solutions, f)

    print(f"Saved {len(solutions)} solutions to data/arc_python_solutions_recovery.json")

if __name__ == "__main__":
    main()
