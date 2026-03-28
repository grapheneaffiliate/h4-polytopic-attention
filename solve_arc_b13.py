import json
import copy
import inspect

# Task 88a62173: 5x5 grid divided by zero-row and zero-col into 4 quadrants.
# Find the unique quadrant (the one that differs from the other 3).
def solve_88a62173(grid):
    rows, cols = len(grid), len(grid[0])
    div_r = div_c = None
    for r in range(rows):
        if all(v == 0 for v in grid[r]):
            div_r = r
            break
    for c in range(cols):
        if all(grid[r][c] == 0 for r in range(rows)):
            div_c = c
            break
    tl = [row[:div_c] for row in grid[:div_r]]
    tr = [row[div_c+1:] for row in grid[:div_r]]
    bl = [row[:div_c] for row in grid[div_r+1:]]
    br = [row[div_c+1:] for row in grid[div_r+1:]]
    quads = [tl, tr, bl, br]
    for i in range(4):
        others = [quads[j] for j in range(4) if j != i]
        if all(o == others[0] for o in others):
            return quads[i]
    return tl

# Task 8be77c9e: Stack original grid on top of its vertical flip (reverse rows).
def solve_8be77c9e(grid):
    return grid + grid[::-1]

# Task 8d5021e8: Input 3x2 -> Output 9x4.
# Mirror each row (reverse+original), tile vertically as [reversed_rows, original, reversed_rows].
def solve_8d5021e8(grid):
    h = len(grid)
    mirrored_orig = [row[::-1] + row for row in grid]
    mirrored_rev = [row[::-1] + row for row in grid[::-1]]
    return mirrored_rev + mirrored_orig + mirrored_rev

# Task 8d510a79: Grid with horizontal line of 5s.
# Color 1 extends away from 5-line. Color 2 extends toward 5-line.
def solve_8d510a79(grid):
    rows, cols = len(grid), len(grid[0])
    out = copy.deepcopy(grid)
    five_row = None
    for r in range(rows):
        if all(v == 5 for v in grid[r]):
            five_row = r
            break
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v == 0 or v == 5:
                continue
            if v == 1:
                if r < five_row:
                    for rr in range(r - 1, -1, -1):
                        out[rr][c] = 1
                else:
                    for rr in range(r + 1, rows):
                        out[rr][c] = 1
            elif v == 2:
                if r < five_row:
                    for rr in range(r + 1, five_row):
                        out[rr][c] = 2
                else:
                    for rr in range(r - 1, five_row, -1):
                        out[rr][c] = 2
    return out

# Task 8e5a5113: 3-row grid with 3 sections separated by columns of 5.
# Section 2 = section 1 rotated 90 CW. Section 3 = section 1 rotated 180.
def solve_8e5a5113(grid):
    rows = len(grid)
    cols = len(grid[0])
    sep_cols = [c for c in range(cols) if grid[0][c] == 5]
    s1 = [row[:sep_cols[0]] for row in grid]
    w = sep_cols[0]
    n = rows  # should equal w for square section
    # Rotate 90 CW: new[i][j] = old[n-1-j][i]
    s2 = [[s1[n-1-j][i] for j in range(n)] for i in range(w)]
    # Rotate 180: reverse rows, reverse each row
    s3 = [r[::-1] for r in s1[::-1]]
    result = []
    for r in range(rows):
        result.append(s1[r] + [5] + s2[r] + [5] + s3[r])
    return result

# Task 8f2ea7aa: 9x9 grid divided into 3x3 blocks.
# Shape in one block. Each non-zero cell position in the shape defines a block
# in the 3x3 grid where the shape gets placed.
def solve_8f2ea7aa(grid):
    rows, cols = len(grid), len(grid[0])
    bh, bw = rows // 3, cols // 3
    shape = None
    for br in range(3):
        for bc in range(3):
            block = []
            for r in range(br * bh, (br + 1) * bh):
                block.append(grid[r][bc * bw:(bc + 1) * bw])
            if any(v != 0 for row in block for v in row):
                shape = block
                break
        if shape:
            break
    out = [[0] * cols for _ in range(rows)]
    for sr in range(bh):
        for sc in range(bw):
            if shape[sr][sc] != 0:
                for r in range(bh):
                    for c in range(bw):
                        out[sr * bh + r][sc * bw + c] = shape[r][c]
    return out

# Task 90f3ed37: Grid with rows of 8s forming staircases.
# Template group covers all columns. Partial groups get extended with 1s.
def solve_90f3ed37(grid):
    rows, cols = len(grid), len(grid[0])
    out = copy.deepcopy(grid)

    groups = []
    i = 0
    while i < rows:
        if any(v == 8 for v in grid[i]):
            start = i
            while i < rows and any(v == 8 for v in grid[i]):
                i += 1
            groups.append((start, i))
        else:
            i += 1

    template = None
    template_data = None
    for s, e in groups:
        covered = set()
        for r in range(s, e):
            for c in range(cols):
                if grid[r][c] == 8:
                    covered.add(c)
        if len(covered) == cols:
            template = (s, e)
            template_data = [grid[r][:] for r in range(s, e)]
            break

    if template is None:
        return out

    tlen = len(template_data)

    for s, e in groups:
        if (s, e) == template:
            continue

        glen = e - s
        covered = set()
        for r in range(s, e):
            for c in range(cols):
                if grid[r][c] == 8:
                    covered.add(c)

        if glen == tlen:
            for i in range(glen):
                gr = s + i
                for c in range(cols):
                    if template_data[i][c] == 8 and out[gr][c] == 0 and c not in covered:
                        out[gr][c] = 1
        elif glen < tlen:
            for i in range(glen):
                gr = s + i
                for c in range(cols):
                    if template_data[i][c] == 8 and out[gr][c] == 0 and c not in covered:
                        out[gr][c] = 1
            insert_r = e
            for i in range(glen, tlen):
                if insert_r < rows:
                    for c in range(cols):
                        if template_data[i][c] == 8 and c not in covered:
                            out[insert_r][c] = 1
                    insert_r += 1

    return out

# Task 913fb3ed: Each non-zero pixel gets a 3x3 border.
# Color mapping: 3->6, 2->1, 8->4.
def solve_913fb3ed(grid):
    rows, cols = len(grid), len(grid[0])
    out = [[0]*cols for _ in range(rows)]
    color_map = {3: 6, 2: 1, 8: 4}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                val = grid[r][c]
                border = color_map.get(val, val)
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if dr == 0 and dc == 0:
                                out[nr][nc] = val
                            else:
                                out[nr][nc] = border
    return out

# Task 91413438: Input NxN, count non-zero (N_nz) and zero (N_z) cells.
# Output size = N*N_z x N*N_z. Tile input pattern N_nz times, left-to-right top-to-bottom.
def solve_91413438(grid):
    h = len(grid)
    w = len(grid[0])
    n_nonzero = sum(1 for r in range(h) for c in range(w) if grid[r][c] != 0)
    n_zero = h * w - n_nonzero
    out_h = h * n_zero
    out_w = w * n_zero
    out = [[0] * out_w for _ in range(out_h)]
    count = 0
    for br in range(n_zero):
        for bc in range(n_zero):
            if count >= n_nonzero:
                break
            for r in range(h):
                for c in range(w):
                    out[br * h + r][bc * w + c] = grid[r][c]
            count += 1
        if count >= n_nonzero:
            break
    return out

# Task 9172f3a0: Each cell in NxN input becomes an NxN block of that color.
def solve_9172f3a0(grid):
    n = len(grid)
    scale = n
    out = []
    for r in range(n):
        for _ in range(scale):
            row = []
            for c in range(n):
                row.extend([grid[r][c]] * scale)
            out.append(row)
    return out

# ---- Run tests ----
task_ids = ['88a62173', '8be77c9e', '8d5021e8', '8d510a79', '8e5a5113',
            '8f2ea7aa', '90f3ed37', '913fb3ed', '91413438', '9172f3a0']

solvers = {
    '88a62173': solve_88a62173,
    '8be77c9e': solve_8be77c9e,
    '8d5021e8': solve_8d5021e8,
    '8d510a79': solve_8d510a79,
    '8e5a5113': solve_8e5a5113,
    '8f2ea7aa': solve_8f2ea7aa,
    '90f3ed37': solve_90f3ed37,
    '913fb3ed': solve_913fb3ed,
    '91413438': solve_91413438,
    '9172f3a0': solve_9172f3a0,
}

all_pass = True
for tid in task_ids:
    with open(f'data/arc1/{tid}.json') as f:
        task = json.load(f)
    solver = solvers[tid]
    for split in ['train', 'test']:
        for i, pair in enumerate(task[split]):
            result = solver(pair['input'])
            expected = pair['output']
            if result != expected:
                print(f"FAIL {tid} {split}[{i}]")
                for r in range(min(len(result), len(expected))):
                    if r < len(result) and r < len(expected) and result[r] != expected[r]:
                        print(f"  Row {r}: got={result[r]}")
                        print(f"         exp={expected[r]}")
                if len(result) != len(expected):
                    print(f"  Size mismatch: got {len(result)} rows, expected {len(expected)}")
                all_pass = False
            else:
                print(f"PASS {tid} {split}[{i}]")

print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

# Save solutions
solution_data = {}
for tid in task_ids:
    solver = solvers[tid]
    solution_data[tid] = {
        'solver_name': solver.__name__,
        'source': inspect.getsource(solver)
    }

with open('data/arc_python_solutions_b13.json', 'w') as f:
    json.dump(solution_data, f, indent=2)

print("Saved to data/arc_python_solutions_b13.json")
