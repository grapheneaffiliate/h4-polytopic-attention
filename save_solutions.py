import json

solutions = {}

solutions["25ff71a9"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    out = [[0]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                nr = r + 1
                if nr < rows:
                    out[nr][c] = grid[r][c]
    return out"""

solutions["27a28665"] = """def solve(grid):
    def canonical(pattern):
        def rotate90(m):
            n = len(m)
            return [[m[n-1-c][r] for c in range(n)] for r in range(n)]
        def flip(m):
            return [row[::-1] for row in m]
        forms = []
        m = [row[:] for row in pattern]
        for _ in range(4):
            forms.append(tuple(tuple(r) for r in m))
            forms.append(tuple(tuple(r) for r in flip(m)))
            m = rotate90(m)
        return min(forms)
    lookup = {
        ((0, 1, 0), (1, 0, 1), (0, 1, 1)): 1,
        ((1, 0, 1), (0, 1, 0), (1, 0, 1)): 2,
        ((0, 0, 1), (1, 1, 0), (1, 1, 0)): 3,
        ((0, 1, 0), (1, 1, 1), (0, 1, 0)): 6,
    }
    binary = [[1 if grid[r][c]!=0 else 0 for c in range(3)] for r in range(3)]
    key = canonical(binary)
    return [[lookup[key]]]"""

solutions["28bf18c6"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    min_r, max_r, min_c, max_c = rows, 0, cols, 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    shape = []
    for r in range(min_r, max_r+1):
        row = []
        for c in range(min_c, max_c+1):
            row.append(grid[r][c])
        shape.append(row)
    result = []
    for row in shape:
        result.append(row + row)
    return result"""

solutions["28e73c20"] = """def solve(grid):
    n = len(grid)
    out = [[0]*n for _ in range(n)]
    for layer in range(0, (n+1)//2):
        if layer % 2 == 0:
            top = layer; bot = n-1-layer; left = layer; right = n-1-layer
            for c in range(left, right+1): out[top][c] = 3
            for r in range(top, bot+1): out[r][right] = 3
            for c in range(left, right+1): out[bot][c] = 3
            for r in range(top, bot+1): out[r][left] = 3
    num_flips = n//2 if n % 4 != 2 else n//2 - 1
    for d in range(num_flips):
        r, c = d+1, d
        out[r][c] = 0 if out[r][c] == 3 else 3
    return out"""

solutions["29623171"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    div_rows = set(r for r in range(rows) if all(v == 5 for v in grid[r]))
    div_cols = set(c for c in range(cols) if all(grid[r][c] == 5 for r in range(rows)))
    row_bounds = [0] + sorted([r+1 for r in div_rows]) + [rows]
    col_bounds = [0] + sorted([c+1 for c in div_cols]) + [cols]
    sections = {}
    color = 0
    for si in range(len(row_bounds)-1):
        for sj in range(len(col_bounds)-1):
            r1, r2 = row_bounds[si], row_bounds[si+1]
            c1, c2 = col_bounds[sj], col_bounds[sj+1]
            count = 0
            for r in range(r1, r2):
                for c in range(c1, c2):
                    if grid[r][c] != 0 and grid[r][c] != 5:
                        count += 1
                        color = grid[r][c]
            sections[(si,sj)] = (count, r1, r2, c1, c2)
    max_count = max(v[0] for v in sections.values())
    out = [[0]*cols for _ in range(rows)]
    for r in div_rows:
        for c in range(cols):
            out[r][c] = 5
    for c in div_cols:
        for r in range(rows):
            out[r][c] = 5
    for k, (count, r1, r2, c1, c2) in sections.items():
        if count == max_count:
            for r in range(r1, r2):
                for c in range(c1, c2):
                    if r not in div_rows and c not in div_cols:
                        out[r][c] = color
    return out"""

solutions["29c11459"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]
    for r in range(rows):
        nonzero = [(c, grid[r][c]) for c in range(cols) if grid[r][c] != 0]
        if len(nonzero) == 2:
            (c1, v1), (c2, v2) = nonzero
            mid = (c1 + c2) // 2
            for c in range(c1, mid):
                out[r][c] = v1
            out[r][mid] = 5
            for c in range(mid+1, c2+1):
                out[r][c] = v2
    return out"""

solutions["2bcee788"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    twos = []
    shape_cells = []
    shape_color = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                twos.append((r,c))
            elif grid[r][c] != 0:
                shape_cells.append((r,c))
                shape_color = grid[r][c]
    out = [[3]*cols for _ in range(rows)]
    for r,c in shape_cells:
        out[r][c] = shape_color
    if len(twos) == 1 or all(r == twos[0][0] for r,c in twos):
        two_r = twos[0][0]
        shape_rows = [r for r,c in shape_cells]
        if two_r > max(shape_rows):
            mirror = (max(shape_rows) + two_r) / 2.0
        else:
            mirror = (min(shape_rows) + two_r) / 2.0
        for r,c in shape_cells:
            nr = int(2 * mirror - r)
            if 0 <= nr < rows:
                out[nr][c] = shape_color
    else:
        two_c = twos[0][1]
        shape_cols = [c for r,c in shape_cells]
        if two_c > max(shape_cols):
            mirror = (max(shape_cols) + two_c) / 2.0
        else:
            mirror = (min(shape_cols) + two_c) / 2.0
        for r,c in shape_cells:
            nc = int(2 * mirror - c)
            if 0 <= nc < cols:
                out[r][nc] = shape_color
    return out"""

solutions["2bee17df"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    row_zeros = [sum(1 for v in grid[r] if v == 0) for r in range(rows)]
    col_zeros = [sum(1 for r in range(rows) if grid[r][c] == 0) for c in range(cols)]
    max_rz = max(row_zeros)
    max_cz = max(col_zeros)
    max_rows = set(r for r in range(rows) if row_zeros[r] == max_rz)
    max_cols = set(c for c in range(cols) if col_zeros[c] == max_cz)
    out = [row[:] for row in grid]
    for r in max_rows:
        for c in range(cols):
            if grid[r][c] == 0:
                out[r][c] = 3
    for c in max_cols:
        for r in range(rows):
            if grid[r][c] == 0:
                out[r][c] = 3
    return out"""

solutions["2c608aff"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    non_bg = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                non_bg.setdefault(grid[r][c], []).append((r,c))
    rect_color = max(non_bg, key=lambda k: len(non_bg[k]))
    rect_cells = non_bg[rect_color]
    rect_r1 = min(r for r,c in rect_cells)
    rect_r2 = max(r for r,c in rect_cells)
    rect_c1 = min(c for r,c in rect_cells)
    rect_c2 = max(c for r,c in rect_cells)
    out = [row[:] for row in grid]
    for color, cells in non_bg.items():
        if color == rect_color:
            continue
        for r, c in cells:
            in_row = rect_r1 <= r <= rect_r2
            in_col = rect_c1 <= c <= rect_c2
            if in_row and not in_col:
                if c > rect_c2:
                    for cc in range(rect_c2 + 1, c + 1):
                        out[r][cc] = color
                else:
                    for cc in range(c, rect_c1):
                        out[r][cc] = color
            elif in_col and not in_row:
                if r > rect_r2:
                    for rr in range(rect_r2 + 1, r + 1):
                        out[rr][c] = color
                elif r < rect_r1:
                    for rr in range(r, rect_r1):
                        out[rr][c] = color
    return out"""

solutions["2dc579da"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    bg = grid[0][0]
    div_row = None
    div_col = None
    div_color = None
    for r in range(rows):
        vals = set(grid[r][c] for c in range(cols))
        if len(vals) == 1 and grid[r][0] != bg:
            div_row = r
            div_color = grid[r][0]
            break
    for c in range(cols):
        vals = set(grid[r][c] for r in range(rows))
        if len(vals) == 1 and grid[0][c] != bg:
            div_col = c
            break
    quads = [
        (0, div_row, 0, div_col),
        (0, div_row, div_col+1, cols),
        (div_row+1, rows, 0, div_col),
        (div_row+1, rows, div_col+1, cols),
    ]
    for r1, r2, c1, c2 in quads:
        has_anomaly = False
        for r in range(r1, r2):
            for c in range(c1, c2):
                if grid[r][c] != bg and grid[r][c] != div_color:
                    has_anomaly = True
        if has_anomaly:
            return [[grid[r][c] for c in range(c1, c2)] for r in range(r1, r2)]
    return grid"""

# Verify all solutions
for task_id, code in solutions.items():
    with open(f"data/arc1/{task_id}.json") as f:
        data = json.load(f)

    exec(code)
    all_pass = True
    for i, pair in enumerate(data["train"]):
        result = solve(pair["input"])
        if result != pair["output"]:
            print(f"FAIL: {task_id} train {i}")
            all_pass = False

    if all_pass:
        print(f"PASS: {task_id} ({len(data['train'])} train pairs)")

# Save
try:
    with open("data/arc_python_solutions_b3.json", "r") as f:
        existing = json.load(f)
except FileNotFoundError:
    existing = {}

existing.update(solutions)

with open("data/arc_python_solutions_b3.json", "w") as f:
    json.dump(existing, f, indent=2)

print(f"\nSaved {len(solutions)} solutions. Total in file: {len(existing)}")
