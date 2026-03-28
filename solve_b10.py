import json
import inspect

task_ids = ['6d0160f0', '6d0aefbc', '6d75e8bb', '6e02f1e3', '6e19193c', '6e82a1ae', '6f8cd79b', '6fa7a44f', '72ca375d', '7447852a']
tasks = {}
for tid in task_ids:
    with open(f'data/arc1/{tid}.json') as f:
        tasks[tid] = json.load(f)

# 6d0160f0: Grid 11x11 divided by 5-lines into 3x3 blocks.
# Find block containing value 4 (unique). Local pos of 4 determines output block position.
def solve_6d0160f0(grid):
    out = [[0]*11 for _ in range(11)]
    for r in range(11):
        for c in range(11):
            if r == 3 or r == 7 or c == 3 or c == 7:
                out[r][c] = 5
    # Find block with value 4
    for bi in range(3):
        for bj in range(3):
            r0, c0 = bi*4, bj*4
            for r in range(3):
                for c in range(3):
                    if grid[r0+r][c0+c] == 4:
                        # local pos (r,c) = dest block pos
                        dr0, dc0 = r*4, c*4
                        for rr in range(3):
                            for cc in range(3):
                                out[dr0+rr][dc0+cc] = grid[r0+rr][c0+cc]
                        return out
    return out

# 6d0aefbc: Mirror horizontally and concatenate
def solve_6d0aefbc(grid):
    return [list(row) + list(reversed(row)) for row in grid]

# 6d75e8bb: Fill bounding box of 8s - 0s inside become 2s
def solve_6d75e8bb(grid):
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    min_r = min_c = float('inf')
    max_r = max_c = -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8:
                min_r, max_r = min(min_r, r), max(max_r, r)
                min_c, max_c = min(min_c, c), max(max_c, c)
    for r in range(min_r, max_r+1):
        for c in range(min_c, max_c+1):
            if out[r][c] == 0:
                out[r][c] = 2
    return out

# 6e02f1e3: Count distinct values. 1->top row 5s, 2->main diagonal, 3->anti-diagonal
def solve_6e02f1e3(grid):
    vals = set(v for row in grid for v in row)
    n = len(vals)
    out = [[0]*3 for _ in range(3)]
    if n == 1:
        out[0] = [5, 5, 5]
    elif n == 2:
        for i in range(3):
            out[i][i] = 5
    else:
        for i in range(3):
            out[i][2-i] = 5
    return out

# 6e19193c: L-shaped pieces extend diagonal trails from their open corner
def solve_6e19193c(grid):
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] != 0]
    color = grid[cells[0][0]][cells[0][1]]
    cell_set = set(cells)
    visited = set()
    for start in cells:
        if start in visited:
            continue
        comp = []
        stack = [start]
        while stack:
            cr, cc = stack.pop()
            if (cr,cc) in visited:
                continue
            visited.add((cr,cc))
            comp.append((cr,cc))
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc = cr+dr,cc+dc
                if (nr,nc) in cell_set and (nr,nc) not in visited:
                    stack.append((nr,nc))
        # Find bounding box and missing cell
        min_r = min(r for r,c in comp)
        max_r = max(r for r,c in comp)
        min_c = min(c for r,c in comp)
        max_c = max(c for r,c in comp)
        comp_set = set(comp)
        missing = None
        for r in range(min_r, max_r+1):
            for c in range(min_c, max_c+1):
                if (r,c) not in comp_set:
                    missing = (r,c)
        if missing is None:
            continue
        opposite = (min_r + max_r - missing[0], min_c + max_c - missing[1])
        dr = missing[0] - opposite[0]
        dc = missing[1] - opposite[1]
        r, c = missing[0] + dr, missing[1] + dc
        while 0 <= r < rows and 0 <= c < cols:
            out[r][c] = color
            r += dr; c += dc
    return out

# 6e82a1ae: Color blobs of 5s by size: largest->1, medium->2, smallest->3
def solve_6e82a1ae(grid):
    rows, cols = len(grid), len(grid[0])
    out = [[0]*cols for _ in range(rows)]
    visited = set()
    blobs = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5 and (r,c) not in visited:
                blob = []
                stack = [(r,c)]
                while stack:
                    cr,cc = stack.pop()
                    if (cr,cc) in visited or grid[cr][cc] != 5:
                        continue
                    visited.add((cr,cc))
                    blob.append((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc = cr+dr,cc+dc
                        if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in visited:
                            stack.append((nr,nc))
                blobs.append(blob)
    sizes = sorted(set(len(b) for b in blobs), reverse=True)
    size_to_color = {s: i+1 for i, s in enumerate(sizes)}
    for blob in blobs:
        color = size_to_color[len(blob)]
        for r,c in blob:
            out[r][c] = color
    return out

# 6f8cd79b: Draw border frame with 8s
def solve_6f8cd79b(grid):
    rows, cols = len(grid), len(grid[0])
    out = [[0]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if r == 0 or r == rows-1 or c == 0 or c == cols-1:
                out[r][c] = 8
    return out

# 6fa7a44f: Stack original + vertically flipped copy
def solve_6fa7a44f(grid):
    out = [row[:] for row in grid]
    for row in reversed(grid):
        out.append(row[:])
    return out

# 72ca375d: Output the shape with left-right symmetry in its bounding box
def solve_72ca375d(grid):
    rows, cols = len(grid), len(grid[0])
    # Group all cells by color
    from collections import defaultdict
    color_cells = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color_cells[grid[r][c]].append((r,c))
    shapes = list(color_cells.items())
    for color, cells in shapes:
        min_r = min(r for r,c in cells)
        max_r = max(r for r,c in cells)
        min_c = min(c for r,c in cells)
        max_c = max(c for r,c in cells)
        h, w = max_r-min_r+1, max_c-min_c+1
        cropped = [[0]*w for _ in range(h)]
        for r,c in cells:
            cropped[r-min_r][c-min_c] = color
        symmetric = all(row == list(reversed(row)) for row in cropped)
        if symmetric:
            return cropped
    return grid

# 7447852a: Zigzag pattern - fill triangles at every 3rd vertex (cols 0,6,12,18,...)
def solve_7447852a(grid):
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    c = 0
    while c < cols + 2:  # allow slightly beyond grid for partial fills
        if c % 4 == 0:  # valley (row 0)
            for r, cc in [(1, c), (2, c), (2, c-1), (2, c+1)]:
                if 0 <= r < rows and 0 <= cc < cols and out[r][cc] == 0:
                    out[r][cc] = 4
        else:  # peak (row 2), c % 4 == 2
            for r, cc in [(0, c-1), (0, c), (0, c+1), (1, c)]:
                if 0 <= r < rows and 0 <= cc < cols and out[r][cc] == 0:
                    out[r][cc] = 4
        c += 6
    return out

# Test all
solve_funcs = {
    '6d0160f0': solve_6d0160f0,
    '6d0aefbc': solve_6d0aefbc,
    '6d75e8bb': solve_6d75e8bb,
    '6e02f1e3': solve_6e02f1e3,
    '6e19193c': solve_6e19193c,
    '6e82a1ae': solve_6e82a1ae,
    '6f8cd79b': solve_6f8cd79b,
    '6fa7a44f': solve_6fa7a44f,
    '72ca375d': solve_72ca375d,
    '7447852a': solve_7447852a,
}

results = {}
for tid, func in solve_funcs.items():
    task = tasks[tid]
    all_pass = True
    for i, pair in enumerate(task['train']):
        pred = func(pair['input'])
        expected = pair['output']
        if pred != expected:
            all_pass = False
            print(f"FAIL {tid} train {i}")
            for r in range(min(len(pred), len(expected))):
                if r < len(pred) and r < len(expected) and pred[r] != expected[r]:
                    print(f"  Row {r}: got {pred[r]}")
                    print(f"  Row {r}: exp {expected[r]}")
                    break
            if len(pred) != len(expected):
                print(f"  Row count: got {len(pred)}, expected {len(expected)}")
    if all_pass:
        print(f"PASS {tid}")
    results[tid] = all_pass

solution_code = {}
for tid, func in solve_funcs.items():
    if results[tid]:
        solution_code[tid] = inspect.getsource(func)

with open('data/arc_python_solutions_b10.json', 'w') as f:
    json.dump(solution_code, f, indent=2)

print(f"\nSaved {len(solution_code)} solutions. Passing: {sum(results.values())}/{len(results)}")
print(f"Failing: {[t for t,v in results.items() if not v]}")
