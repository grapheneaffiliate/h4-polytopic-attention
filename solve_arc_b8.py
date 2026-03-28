import json
import copy
import inspect
from collections import defaultdict, Counter

# Load all tasks
tasks = {}
task_ids = ['5521c0d9', '5582e5ca', '5614dbcf', '56ff96f3', '5c0a986e', '60b61512', '623ea044', '62c24649', '63613498', '6430c8c4']
for tid in task_ids:
    with open(f'data/arc1/{tid}.json') as f:
        tasks[tid] = json.load(f)

# 5521c0d9: Each colored rectangle shifts up by its own height
def solve_5521c0d9(grid):
    rows = len(grid)
    cols = len(grid[0])
    color_cells = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color_cells[grid[r][c]].append((r, c))
    result = [[0]*cols for _ in range(rows)]
    for color, cells in color_cells.items():
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        height = max_r - min_r + 1
        for r, c in cells:
            new_r = r - height
            if 0 <= new_r < rows:
                result[new_r][c] = color
    return result

# 5582e5ca: Fill with most common color
def solve_5582e5ca(grid):
    flat = [v for row in grid for v in row]
    count = Counter(flat)
    most_common = count.most_common(1)[0][0]
    return [[most_common]*len(grid[0]) for _ in range(len(grid))]

# 5614dbcf: 9x9 grid -> 3x3, each cell = dominant non-zero non-5 color of 3x3 block
def solve_5614dbcf(grid):
    result = [[0]*3 for _ in range(3)]
    for br in range(3):
        for bc in range(3):
            colors = set()
            for r in range(br*3, br*3+3):
                for c in range(bc*3, bc*3+3):
                    if grid[r][c] != 0 and grid[r][c] != 5:
                        colors.add(grid[r][c])
            if colors:
                result[br][bc] = colors.pop()
    return result

# 56ff96f3: Two dots of same color -> fill rectangle between them
def solve_56ff96f3(grid):
    rows = len(grid)
    cols = len(grid[0])
    color_pos = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color_pos[grid[r][c]].append((r, c))
    result = [[0]*cols for _ in range(rows)]
    for color, positions in color_pos.items():
        if len(positions) == 2:
            (r1, c1), (r2, c2) = positions
            min_r, max_r = min(r1, r2), max(r1, r2)
            min_c, max_c = min(c1, c2), max(c1, c2)
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    result[r][c] = color
    return result

# 5c0a986e: Color 1 trails up-left from top-left corner, color 2 trails down-right from bottom-right
def solve_5c0a986e(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]
    color_cells = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color_cells[grid[r][c]].append((r, c))
    for color, cells in color_cells.items():
        min_r = min(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_r = max(r for r, c in cells)
        max_c = max(c for r, c in cells)
        if color == 1:
            r, c = min_r - 1, min_c - 1
            while r >= 0 and c >= 0:
                result[r][c] = color
                r -= 1; c -= 1
        elif color == 2:
            r, c = max_r + 1, max_c + 1
            while r < rows and c < cols:
                result[r][c] = color
                r += 1; c += 1
    return result

# 60b61512: Fill 0s in bounding box of each connected component of 4s with 7
def solve_60b61512(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]
    visited = [[False]*cols for _ in range(rows)]
    def find_component(sr, sc):
        stack = [(sr, sc)]
        cells = []
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if visited[r][c] or grid[r][c] != 4:
                continue
            visited[r][c] = True
            cells.append((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((r+dr, c+dc))
        return cells
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 4 and not visited[r][c]:
                cells = find_component(r, c)
                if cells:
                    min_r = min(x[0] for x in cells)
                    max_r = max(x[0] for x in cells)
                    min_c = min(x[1] for x in cells)
                    max_c = max(x[1] for x in cells)
                    for rr in range(min_r, max_r + 1):
                        for cc in range(min_c, max_c + 1):
                            if result[rr][cc] == 0:
                                result[rr][cc] = 7
    return result

# 623ea044: X pattern from single dot - diagonals in all 4 directions
def solve_623ea044(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [[0]*cols for _ in range(rows)]
    dot_r, dot_c, color = -1, -1, 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                dot_r, dot_c, color = r, c, grid[r][c]
    result[dot_r][dot_c] = color
    for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        r, c = dot_r + dr, dot_c + dc
        while 0 <= r < rows and 0 <= c < cols:
            result[r][c] = color
            r += dr; c += dc
    return result

# 62c24649: Mirror 3x3 -> 6x6 (horizontal flip + vertical flip)
def solve_62c24649(grid):
    n = len(grid)
    result = []
    for i in range(n):
        result.append(grid[i] + grid[i][::-1])
    for i in range(n-1, -1, -1):
        result.append(result[i][:])
    return result

# 63613498: Shapes matching template shape (in top-left quadrant of 5-cross) become 5
def solve_63613498(grid):
    rows = len(grid)
    cols = len(grid[0])
    # Find the 5-cross: horizontal row and vertical column of 5s
    h_row = -1
    v_col = -1
    for r in range(rows):
        if sum(1 for c in range(cols) if grid[r][c] == 5) > 1:
            h_row = r
    for c in range(cols):
        if sum(1 for r in range(rows) if grid[r][c] == 5) > 1:
            v_col = c
    visited = [[False]*cols for _ in range(rows)]
    components = []
    def flood_fill(sr, sc, color):
        stack = [(sr, sc)]
        cells = []
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if visited[r][c] or grid[r][c] != color:
                continue
            visited[r][c] = True
            cells.append((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((r+dr, c+dc))
        return cells
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and grid[r][c] != 5 and not visited[r][c]:
                cells = flood_fill(r, c, grid[r][c])
                components.append((grid[r][c], cells))
    def normalize(cells):
        min_r = min(r for r, c in cells)
        min_c = min(c for r, c in cells)
        return frozenset((r - min_r, c - min_c) for r, c in cells)
    # Template = shape entirely in top-left quadrant (r < h_row and c < v_col)
    template_shapes = set()
    for color, cells in components:
        in_tl = all(r < h_row and c < v_col for r, c in cells)
        if in_tl:
            template_shapes.add(normalize(cells))
    result = [row[:] for row in grid]
    for color, cells in components:
        in_tl = all(r < h_row and c < v_col for r, c in cells)
        if not in_tl:
            shape = normalize(cells)
            if shape in template_shapes:
                for r, c in cells:
                    result[r][c] = 5
    return result

# 6430c8c4: Where both top and bottom sections are 0, output 3
def solve_6430c8c4(grid):
    rows = len(grid)
    cols = len(grid[0])
    sep_row = -1
    for r in range(rows):
        if all(v == 4 for v in grid[r]):
            sep_row = r
            break
    top_rows = sep_row
    result = [[0]*cols for _ in range(top_rows)]
    for r in range(top_rows):
        for c in range(cols):
            top_val = grid[r][c]
            bot_val = grid[sep_row + 1 + r][c]
            if top_val == 0 and bot_val == 0:
                result[r][c] = 3
    return result

# Test all
def test_solution(task_id, solve_fn, task_data):
    for i, pair in enumerate(task_data['train']):
        inp = pair['input']
        expected = pair['output']
        got = solve_fn(inp)
        if got != expected:
            print(f"FAIL {task_id} train[{i}]")
            for r in range(max(len(expected), len(got))):
                if r >= len(got) or r >= len(expected) or got[r] != expected[r]:
                    print(f"  Row {r}: expected {expected[r] if r < len(expected) else 'N/A'}")
                    print(f"  Row {r}: got      {got[r] if r < len(got) else 'N/A'}")
                    break
            return False
    print(f"PASS {task_id}")
    return True

results = {}
for task_id, solve_fn in [
    ('5521c0d9', solve_5521c0d9),
    ('5582e5ca', solve_5582e5ca),
    ('5614dbcf', solve_5614dbcf),
    ('56ff96f3', solve_56ff96f3),
    ('5c0a986e', solve_5c0a986e),
    ('60b61512', solve_60b61512),
    ('623ea044', solve_623ea044),
    ('62c24649', solve_62c24649),
    ('63613498', solve_63613498),
    ('6430c8c4', solve_6430c8c4),
]:
    passed = test_solution(task_id, solve_fn, tasks[task_id])
    if passed:
        results[task_id] = inspect.getsource(solve_fn)

print(f"\nPassed: {len(results)}/{len(task_ids)}")
print("Passed tasks:", list(results.keys()))

# Save results
with open('data/arc_python_solutions_b8.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved to data/arc_python_solutions_b8.json")
