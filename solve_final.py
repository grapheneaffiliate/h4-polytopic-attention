import json, inspect
from collections import deque

# ============================================================
# Task 3bdb4ada
# ============================================================
def solve_3bdb4ada(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    visited = [[False]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                r_min, r_max = r, r
                c_min, c_max = c, c
                while r_max + 1 < h and grid[r_max+1][c] == color:
                    r_max += 1
                while c_max + 1 < w and grid[r][c_max+1] == color:
                    c_max += 1
                for rr in range(r_min, r_max+1):
                    for cc in range(c_min, c_max+1):
                        visited[rr][cc] = True
                if r_max - r_min == 2:
                    mid = r_min + 1
                    for cc in range(c_min, c_max+1):
                        if (cc - c_min) % 2 == 1:
                            out[mid][cc] = 0
    return out

# ============================================================
# Task 3befdf3e
# ============================================================
def solve_3befdf3e(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    cells = [(r,c) for r in range(h) for c in range(w) if grid[r][c] != 0]
    r_min = min(r for r,c in cells)
    r_max = max(r for r,c in cells)
    c_min = min(c for r,c in cells)
    c_max = max(c for r,c in cells)
    border_color = grid[r_min][c_min]
    inner_color = None
    for r in range(r_min+1, r_max):
        for c in range(c_min+1, c_max):
            if grid[r][c] != border_color:
                inner_color = grid[r][c]
                break
        if inner_color: break
    inner_h = r_max - r_min - 1
    inner_w = c_max - c_min - 1
    for r in range(r_min, r_max+1):
        for c in range(c_min, c_max+1):
            out[r][c] = inner_color if grid[r][c] == border_color else border_color
    for dr in range(1, inner_h+1):
        for c in range(c_min, c_max+1):
            if r_min-dr >= 0: out[r_min-dr][c] = border_color
            if r_max+dr < h: out[r_max+dr][c] = border_color
    for dc in range(1, inner_w+1):
        for r in range(r_min, r_max+1):
            if c_min-dc >= 0: out[r][c_min-dc] = border_color
            if c_max+dc < w: out[r][c_max+dc] = border_color
    return out

# ============================================================
# Task 3de23699
# ============================================================
def solve_3de23699(grid):
    h, w = len(grid), len(grid[0])
    color_positions = {}
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                color_positions.setdefault(grid[r][c], []).append((r, c))
    marker_color = None
    for color, positions in color_positions.items():
        if len(positions) == 4:
            rows = set(r for r, c in positions)
            cols = set(c for r, c in positions)
            if len(rows) == 2 and len(cols) == 2:
                marker_color = color
    fill_color = next(c for c in color_positions if c != marker_color)
    mp = color_positions[marker_color]
    r_min, r_max = min(r for r,c in mp), max(r for r,c in mp)
    c_min, c_max = min(c for r,c in mp), max(c for r,c in mp)
    inner = []
    for r in range(r_min+1, r_max):
        row = []
        for c in range(c_min+1, c_max):
            row.append(marker_color if grid[r][c] == fill_color else 0)
        inner.append(row)
    return inner

# ============================================================
# Task 3e980e27
# ============================================================
def solve_3e980e27(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    from collections import deque
    visited = [[False]*w for _ in range(h)]
    components = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 and not visited[r][c]:
                q = deque([(r, c)])
                visited[r][c] = True
                cells = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and grid[nr][nc]!=0:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append(cells)
    lone_points = []
    templates_list = []
    for comp in components:
        colors = set(grid[r][c] for r, c in comp)
        if len(colors) > 1:
            templates_list.append(comp)
        elif len(comp) == 1:
            lone_points.append(comp[0])
    lone_colors = set(grid[r][c] for r, c in lone_points)
    template_info = {}
    for tmpl in templates_list:
        colors = set(grid[r][c] for r, c in tmpl)
        acs = colors & lone_colors
        if not acs: continue
        ac = acs.pop()
        anchor_pos = next((r,c) for r,c in tmpl if grid[r][c] == ac)
        shape = [(r-anchor_pos[0], c-anchor_pos[1], grid[r][c]) for r,c in tmpl]
        template_info[ac] = shape
    for lr, lc in lone_points:
        ac = grid[lr][lc]
        if ac in template_info:
            shape = template_info[ac]
            for dr, dc, color in shape:
                if ac == 2:
                    nr, nc = lr + dr, lc - dc
                else:
                    nr, nc = lr + dr, lc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr][nc] = color
    return out

# ============================================================
# Task 3eda0437
# ============================================================
def solve_3eda0437(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    best = None
    best_area = 0
    for r1 in range(h):
        for r2 in range(r1+1, h):
            rh = r2 - r1 + 1
            if h > 2 and rh >= h:
                continue
            c = 0
            while c < w:
                if all(grid[r][c] == 0 for r in range(r1, r2+1)):
                    c_start = c
                    while c < w and all(grid[r][c] == 0 for r in range(r1, r2+1)):
                        c += 1
                    c_end = c - 1
                    area = rh * (c_end - c_start + 1)
                    if area > best_area:
                        best_area = area
                        best = (r1, r2, c_start, c_end)
                else:
                    c += 1
    if best:
        for r in range(best[0], best[1]+1):
            for c in range(best[2], best[3]+1):
                out[r][c] = 6
    return out

# ============================================================
# Task 3f7978a0
# ============================================================
def solve_3f7978a0(grid):
    h, w = len(grid), len(grid[0])
    fives = [(r,c) for r in range(h) for c in range(w) if grid[r][c] == 5]
    if not fives:
        return grid
    r_min = min(r for r,c in fives)
    r_max = max(r for r,c in fives)
    c_min = min(c for r,c in fives)
    c_max = max(c for r,c in fives)
    rect_r_min = r_min - 1
    rect_r_max = r_max + 1
    out = []
    for r in range(rect_r_min, rect_r_max + 1):
        row = []
        for c in range(c_min, c_max + 1):
            row.append(grid[r][c])
        out.append(row)
    return out

# ============================================================
# Task 40853293
# ============================================================
def solve_40853293(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    color_positions = {}
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                color_positions.setdefault(grid[r][c], []).append((r, c))
    h_lines = []
    v_lines = []
    for color, positions in color_positions.items():
        if len(positions) == 2:
            (r1, c1), (r2, c2) = positions
            if r1 == r2:
                h_lines.append((color, r1, min(c1,c2), max(c1,c2)))
            elif c1 == c2:
                v_lines.append((color, c1, min(r1,r2), max(r1,r2)))
    for color, row, c1, c2 in h_lines:
        for c in range(c1, c2+1):
            out[row][c] = color
    for color, col, r1, r2 in v_lines:
        for r in range(r1, r2+1):
            out[r][col] = color
    return out

# ============================================================
# Task 4093f84a
# ============================================================
def solve_4093f84a(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    band_rows = [r for r in range(h) if all(grid[r][c] == 5 for c in range(w))]
    band_cols = [c for c in range(w) if all(grid[r][c] == 5 for r in range(h))]
    if band_rows:
        band_r_min = min(band_rows)
        band_r_max = max(band_rows)
        for c in range(w):
            above = sorted([r for r in range(band_r_min) if grid[r][c] != 0 and grid[r][c] != 5], reverse=True)
            target_row = band_r_min - 1
            for r in above:
                out[r][c] = 0
                out[target_row][c] = 5
                target_row -= 1
            below = sorted([r for r in range(band_r_max+1, h) if grid[r][c] != 0 and grid[r][c] != 5])
            target_row = band_r_max + 1
            for r in below:
                out[r][c] = 0
                out[target_row][c] = 5
                target_row += 1
    elif band_cols:
        band_c_min = min(band_cols)
        band_c_max = max(band_cols)
        for r in range(h):
            left = sorted([c for c in range(band_c_min) if grid[r][c] != 0 and grid[r][c] != 5], reverse=True)
            target_col = band_c_min - 1
            for c in left:
                out[r][c] = 0
                out[r][target_col] = 5
                target_col -= 1
            right = sorted([c for c in range(band_c_max+1, w) if grid[r][c] != 0 and grid[r][c] != 5])
            target_col = band_c_max + 1
            for c in right:
                out[r][c] = 0
                out[r][target_col] = 5
                target_col += 1
    return out

# ============================================================
# Task 41e4d17e
# ============================================================
def solve_41e4d17e(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    from collections import deque
    visited = [[False]*w for _ in range(h)]
    rectangles = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 1 and not visited[r][c]:
                q = deque([(r, c)])
                visited[r][c] = True
                cells = [(r, c)]
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and grid[nr][nc]==1:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                            cells.append((nr, nc))
                r_min = min(r for r,c in cells)
                r_max = max(r for r,c in cells)
                c_min = min(c for r,c in cells)
                c_max = max(c for r,c in cells)
                rectangles.append((r_min, r_max, c_min, c_max))
    for r_min, r_max, c_min, c_max in rectangles:
        center_r = (r_min + r_max) // 2
        center_c = (c_min + c_max) // 2
        for r in range(h):
            if grid[r][center_c] != 1:
                out[r][center_c] = 6
        for c in range(w):
            if grid[center_r][c] != 1:
                out[center_r][c] = 6
    return out

# ============================================================
# Task 4290ef0e
# ============================================================
def solve_4290ef0e(grid):
    h, w = len(grid), len(grid[0])
    bg = grid[0][0]
    by_color = {}
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg:
                by_color.setdefault(grid[r][c], []).append((r, c))
    frames = []
    for color, cells in by_color.items():
        r_min = min(r for r, c in cells)
        r_max = max(r for r, c in cells)
        c_min = min(c for r, c in cells)
        c_max = max(c for r, c in cells)
        fh = r_max - r_min + 1
        fw = c_max - c_min + 1
        max_dim = max(fh, fw)
        if max_dim % 2 == 0:
            max_dim += 1
        if len(cells) == 1:
            corner = 0
        else:
            pattern = [[bg] * fw for _ in range(fh)]
            for r, c in cells:
                pattern[r - r_min][c - c_min] = color
            corner_h = 0
            for c in range(fw):
                if pattern[0][c] == color:
                    corner_h += 1
                else:
                    break
            corner_v = 0
            for r in range(fh):
                if pattern[r][0] == color:
                    corner_v += 1
                else:
                    break
            if fw <= fh:
                corner = corner_h
            else:
                corner = corner_v
        frames.append((max_dim, color, corner))
    frames.sort()
    if not frames:
        return grid
    max_size = frames[-1][0]
    out_size = max_size
    center = out_size // 2
    out = [[bg] * out_size for _ in range(out_size)]
    for ring_size, color, corner in reversed(frames):
        if ring_size == 1:
            out[center][center] = color
            continue
        half = ring_size // 2
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                if max(abs(dr), abs(dc)) != half:
                    continue
                r = center + dr
                c = center + dc
                colored = False
                if abs(dr) == half and abs(dc) == half:
                    colored = True
                elif abs(dr) == half:
                    colored = (abs(dc) >= half - corner + 1)
                elif abs(dc) == half:
                    colored = (abs(dr) >= half - corner + 1)
                if colored:
                    out[r][c] = color
    return out

# ============================================================
# Verify all tasks and save
# ============================================================
task_solvers = {
    '3bdb4ada': solve_3bdb4ada,
    '3befdf3e': solve_3befdf3e,
    '3de23699': solve_3de23699,
    '3e980e27': solve_3e980e27,
    '3eda0437': solve_3eda0437,
    '3f7978a0': solve_3f7978a0,
    '40853293': solve_40853293,
    '4093f84a': solve_4093f84a,
    '41e4d17e': solve_41e4d17e,
    '4290ef0e': solve_4290ef0e,
}

passing = {}
for task_id, solver in task_solvers.items():
    with open(f'data/arc1/{task_id}.json') as f:
        data = json.load(f)
    all_pass = True
    for i, ex in enumerate(data['train']):
        result = solver(ex['input'])
        match = result == ex['output']
        if not match:
            all_pass = False
            print(f"{task_id} train[{i}]: FAIL")
    if all_pass:
        print(f"{task_id}: ALL PASS")
        passing[task_id] = solver.__name__

print(f"\nPassing: {len(passing)}/{len(task_solvers)}")

solutions_dict = {}
for task_id, func_name in passing.items():
    solver = task_solvers[task_id]
    source = inspect.getsource(solver)
    source = source.replace(f"def {func_name}(", "def solve(", 1)
    solutions_dict[task_id] = source

with open('data/arc_python_solutions_b5.json', 'w') as f:
    json.dump(solutions_dict, f, indent=2)

print(f"Saved {len(passing)} solutions to data/arc_python_solutions_b5.json")
