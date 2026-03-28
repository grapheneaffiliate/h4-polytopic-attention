import json

solutions = {}

# ============ d0f5fe59 ============
solutions["d0f5fe59"] = """def solve(grid):
    rows, cols = len(grid), len(grid[0])
    visited = set()
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8 and (r,c) not in visited:
                count += 1
                stack = [(r,c)]
                while stack:
                    cr, cc = stack.pop()
                    if (cr,cc) in visited:
                        continue
                    visited.add((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]==8 and (nr,nc) not in visited:
                            stack.append((nr,nc))
    return [[8 if i==j else 0 for j in range(count)] for i in range(count)]"""

# ============ dc0a314f ============
solutions["dc0a314f"] = """def solve(grid):
    rows, cols = len(grid), len(grid[0])
    cr, cc = (rows-1)/2.0, (cols-1)/2.0
    threes = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 3]
    min_r = min(r for r,c in threes)
    max_r = max(r for r,c in threes)
    min_c = min(c for r,c in threes)
    max_c = max(c for r,c in threes)
    output = []
    for r in range(min_r, max_r+1):
        row = []
        for c in range(min_c, max_c+1):
            sr, sc = round(2*cr - r), round(2*cc - c)
            row.append(grid[sr][sc])
        output.append(row)
    return output"""

# ============ de1cd16c ============
solutions["de1cd16c"] = """def solve(grid):
    from collections import defaultdict
    rows, cols = len(grid), len(grid[0])
    colors = set()
    for r in range(rows):
        for c in range(cols):
            colors.add(grid[r][c])
    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if (r,c) not in visited:
                color = grid[r][c]
                cells = []
                stack = [(r,c)]
                while stack:
                    cr, cc = stack.pop()
                    if (cr,cc) in visited:
                        continue
                    visited.add((cr,cc))
                    cells.append((cr, cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]==color and (nr,nc) not in visited:
                            stack.append((nr,nc))
                components.append((color, len(cells), cells))
    color_total = defaultdict(int)
    for color, size, cells in components:
        color_total[color] += size
    noise_color = min(colors, key=lambda c: color_total[c])
    bg_regions = []
    seen_bg = set()
    for color in colors:
        if color == noise_color or color in seen_bg:
            continue
        seen_bg.add(color)
        all_cells = []
        for c2, s2, cells2 in components:
            if c2 == color:
                all_cells.extend(cells2)
        min_r = min(r for r,c in all_cells)
        max_r = max(r for r,c in all_cells)
        min_c = min(c for r,c in all_cells)
        max_c = max(c for r,c in all_cells)
        noise_count = 0
        for r in range(min_r, max_r+1):
            for c in range(min_c, max_c+1):
                if grid[r][c] == noise_color:
                    noise_count += 1
        bg_regions.append((color, noise_count))
    bg_regions.sort(key=lambda x: -x[1])
    return [[bg_regions[0][0]]]"""

# ============ e26a3af2 ============
solutions["e26a3af2"] = """def solve(grid):
    from collections import Counter
    rows, cols = len(grid), len(grid[0])
    col_colors = []
    col_agree = 0
    for c in range(cols):
        vals = [grid[r][c] for r in range(rows)]
        maj = Counter(vals).most_common(1)[0][0]
        col_colors.append(maj)
        col_agree += sum(1 for r in range(rows) if grid[r][c] == maj)
    row_colors = []
    row_agree = 0
    for r in range(rows):
        maj = Counter(grid[r]).most_common(1)[0][0]
        row_colors.append(maj)
        row_agree += sum(1 for c in range(cols) if grid[r][c] == maj)
    if col_agree >= row_agree:
        return [[col_colors[c] for c in range(cols)] for r in range(rows)]
    else:
        return [[row_colors[r] for c in range(cols)] for r in range(rows)]"""

# ============ e509e548 ============
solutions["e509e548"] = """def solve(grid):
    rows, cols = len(grid), len(grid[0])
    output = [row[:] for row in grid]
    visited = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3 and (r,c) not in visited:
                cells = []
                stack = [(r,c)]
                while stack:
                    cr, cc = stack.pop()
                    if (cr,cc) in visited:
                        continue
                    visited.add((cr,cc))
                    cells.append((cr, cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]==3 and (nr,nc) not in visited:
                            stack.append((nr,nc))
                cell_set = set(cells)
                corners = 0
                t_junctions = 0
                for cr2, cc2 in cells:
                    neighbors = [(cr2+dr,cc2+dc) for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)] if (cr2+dr,cc2+dc) in cell_set]
                    if len(neighbors) == 2:
                        (r1,c1), (r2,c2) = neighbors
                        if r1 != r2 and c1 != c2:
                            corners += 1
                    elif len(neighbors) == 3:
                        t_junctions += 1
                if t_junctions >= 1:
                    color = 2
                elif corners >= 2:
                    color = 6
                else:
                    color = 1
                for cr2, cc2 in cells:
                    output[cr2][cc2] = color
    return output"""

# ============ e6721834 ============
solutions["e6721834"] = """def solve(grid):
    from collections import Counter, defaultdict
    rows, cols = len(grid), len(grid[0])
    v_split = None
    for c in range(1, cols):
        all_diff = True
        for r in range(rows):
            if grid[r][c-1] == grid[r][c]:
                all_diff = False
                break
        if all_diff:
            v_split = c
            break
    h_split = None
    for r in range(1, rows):
        all_diff = True
        for c in range(cols):
            if grid[r-1][c] == grid[r][c]:
                all_diff = False
                break
        if all_diff:
            h_split = r
            break
    if h_split is not None:
        half1_bg = Counter(grid[0]).most_common(1)[0][0]
        half2_bg = Counter(grid[h_split]).most_common(1)[0][0]
        half1_nonbg = sum(1 for r in range(h_split) for c in range(cols) if grid[r][c] != half1_bg)
        half2_nonbg = sum(1 for r in range(h_split, rows) for c in range(cols) if grid[r][c] != half2_bg)
        if half1_nonbg > half2_nonbg:
            pat_r_range = (0, h_split)
            pat_bg = half1_bg
            mark_r_range = (h_split, rows)
            mark_bg = half2_bg
        else:
            pat_r_range = (h_split, rows)
            pat_bg = half2_bg
            mark_r_range = (0, h_split)
            mark_bg = half1_bg
        out_rows = mark_r_range[1] - mark_r_range[0]
        out_cols = cols
        output = [[mark_bg]*out_cols for _ in range(out_rows)]
        visited = set()
        patterns = []
        for r in range(pat_r_range[0], pat_r_range[1]):
            for c in range(cols):
                if grid[r][c] != pat_bg and (r,c) not in visited:
                    cells = []
                    stack = [(r,c)]
                    while stack:
                        cr, cc = stack.pop()
                        if (cr,cc) in visited: continue
                        visited.add((cr,cc))
                        cells.append((cr,cc))
                        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = cr+dr, cc+dc
                            if pat_r_range[0]<=nr<pat_r_range[1] and 0<=nc<cols and grid[nr][nc]!=pat_bg and (nr,nc) not in visited:
                                stack.append((nr,nc))
                    patterns.append(cells)
        markers = []
        for r in range(mark_r_range[0], mark_r_range[1]):
            for c in range(cols):
                if grid[r][c] != mark_bg:
                    markers.append((r - mark_r_range[0], c, grid[r][c]))
        markers_by_color = defaultdict(list)
        for mr, mc, mcolor in markers:
            markers_by_color[mcolor].append((mr, mc))
        for pat in patterns:
            pat_marker_cells = []
            for r, c in pat:
                color = grid[r][c]
                if color in markers_by_color:
                    pat_marker_cells.append((r, c, color))
            if not pat_marker_cells: continue
            pat_markers_by_color = defaultdict(list)
            for r, c, color in pat_marker_cells:
                pat_markers_by_color[color].append((r, c))
            for mcolor, mark_positions in markers_by_color.items():
                if mcolor not in pat_markers_by_color: continue
                pat_positions = pat_markers_by_color[mcolor]
                if len(mark_positions) == len(pat_positions):
                    mark_sorted = sorted(mark_positions)
                    pat_sorted = sorted(pat_positions)
                    dr = mark_sorted[0][0] - (pat_sorted[0][0] - pat_r_range[0])
                    dc = mark_sorted[0][1] - pat_sorted[0][1]
                    match = True
                    for k in range(1, len(mark_sorted)):
                        expected_r = pat_sorted[k][0] - pat_r_range[0] + dr
                        expected_c = pat_sorted[k][1] + dc
                        if (expected_r, expected_c) != mark_sorted[k]:
                            match = False
                            break
                    if match:
                        for pr, pc in pat:
                            nr = pr - pat_r_range[0] + dr
                            nc = pc + dc
                            if 0 <= nr < out_rows and 0 <= nc < out_cols:
                                output[nr][nc] = grid[pr][pc]
                        break
        return output
    else:
        half1_bg = Counter(grid[r][0] for r in range(rows)).most_common(1)[0][0]
        half2_bg = Counter(grid[r][v_split] for r in range(rows)).most_common(1)[0][0]
        half1_nonbg = sum(1 for r in range(rows) for c in range(v_split) if grid[r][c] != half1_bg)
        half2_nonbg = sum(1 for r in range(rows) for c in range(v_split, cols) if grid[r][c] != half2_bg)
        if half1_nonbg > half2_nonbg:
            pat_c_range = (0, v_split)
            pat_bg = half1_bg
            mark_c_range = (v_split, cols)
            mark_bg = half2_bg
        else:
            pat_c_range = (v_split, cols)
            pat_bg = half2_bg
            mark_c_range = (0, v_split)
            mark_bg = half1_bg
        out_rows = rows
        out_cols = mark_c_range[1] - mark_c_range[0]
        output = [[mark_bg]*out_cols for _ in range(out_rows)]
        visited = set()
        patterns = []
        for r in range(rows):
            for c in range(pat_c_range[0], pat_c_range[1]):
                if grid[r][c] != pat_bg and (r,c) not in visited:
                    cells = []
                    stack = [(r,c)]
                    while stack:
                        cr, cc = stack.pop()
                        if (cr,cc) in visited: continue
                        visited.add((cr,cc))
                        cells.append((cr,cc))
                        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<rows and pat_c_range[0]<=nc<pat_c_range[1] and grid[nr][nc]!=pat_bg and (nr,nc) not in visited:
                                stack.append((nr,nc))
                    patterns.append(cells)
        markers = []
        for r in range(rows):
            for c in range(mark_c_range[0], mark_c_range[1]):
                if grid[r][c] != mark_bg:
                    markers.append((r, c - mark_c_range[0], grid[r][c]))
        markers_by_color = defaultdict(list)
        for mr, mc, mcolor in markers:
            markers_by_color[mcolor].append((mr, mc))
        for pat in patterns:
            pat_marker_cells = []
            for r, c in pat:
                color = grid[r][c]
                if color in markers_by_color:
                    pat_marker_cells.append((r, c, color))
            if not pat_marker_cells: continue
            pat_markers_by_color = defaultdict(list)
            for r, c, color in pat_marker_cells:
                pat_markers_by_color[color].append((r, c))
            for mcolor, mark_positions in markers_by_color.items():
                if mcolor not in pat_markers_by_color: continue
                pat_positions = pat_markers_by_color[mcolor]
                if len(mark_positions) == len(pat_positions):
                    mark_sorted = sorted(mark_positions)
                    pat_sorted = sorted(pat_positions)
                    dr = mark_sorted[0][0] - pat_sorted[0][0]
                    dc = mark_sorted[0][1] - (pat_sorted[0][1] - pat_c_range[0])
                    match = True
                    for k in range(1, len(mark_sorted)):
                        expected_r = pat_sorted[k][0] + dr
                        expected_c = pat_sorted[k][1] - pat_c_range[0] + dc
                        if (expected_r, expected_c) != mark_sorted[k]:
                            match = False
                            break
                    if match:
                        for pr, pc in pat:
                            nr = pr + dr
                            nc = (pc - pat_c_range[0]) + dc
                            if 0 <= nr < out_rows and 0 <= nc < out_cols:
                                output[nr][nc] = grid[pr][pc]
                        break
        return output"""

# ============ e73095fd ============
solutions["e73095fd"] = """def solve(grid):
    rows, cols = len(grid), len(grid[0])
    output = [row[:] for row in grid]
    fives = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                fives.add((r,c))
    for r1 in range(rows):
        for r2 in range(r1+2, rows):
            for c1 in range(cols):
                for c2 in range(c1+2, cols):
                    top_ok = all((r1, c) in fives for c in range(c1, c2+1))
                    if not top_ok: continue
                    bot_ok = all((r2, c) in fives for c in range(c1, c2+1))
                    if not bot_ok: continue
                    left_ok = all((r, c1) in fives for r in range(r1, r2+1))
                    if not left_ok: continue
                    right_ok = all((r, c2) in fives for r in range(r1, r2+1))
                    if not right_ok: continue
                    for r in range(r1+1, r2):
                        for c in range(c1+1, c2):
                            if output[r][c] == 0:
                                output[r][c] = 4
    c = cols - 1
    for r1 in range(rows):
        if (r1, c) not in fives: continue
        for r2 in range(r1+2, rows):
            if (r2, c) not in fives: continue
            if not all((r, c-1) in fives for r in range(r1, r2+1)): continue
            if all(grid[r][c] == 0 for r in range(r1+1, r2)):
                for r in range(r1+1, r2):
                    output[r][c] = 4
    c = 0
    for r1 in range(rows):
        if (r1, c) not in fives: continue
        for r2 in range(r1+2, rows):
            if (r2, c) not in fives: continue
            if not all((r, c+1) in fives for r in range(r1, r2+1)): continue
            if all(grid[r][c] == 0 for r in range(r1+1, r2)):
                for r in range(r1+1, r2):
                    output[r][c] = 4
    r = 0
    for c1 in range(cols):
        if (r, c1) not in fives: continue
        for c2 in range(c1+2, cols):
            if (r, c2) not in fives: continue
            if not all((r+1, c) in fives for c in range(c1, c2+1)): continue
            if all(grid[r][c] == 0 for c in range(c1+1, c2)):
                for c in range(c1+1, c2):
                    output[r][c] = 4
    r = rows - 1
    for c1 in range(cols):
        if (r, c1) not in fives: continue
        for c2 in range(c1+2, cols):
            if (r, c2) not in fives: continue
            if not all((r-1, c) in fives for c in range(c1, c2+1)): continue
            if all(grid[r][c] == 0 for c in range(c1+1, c2)):
                for c in range(c1+1, c2):
                    output[r][c] = 4
    changed = True
    while changed:
        changed = False
        for r in range(rows):
            for c in range(cols):
                if output[r][c] == 0:
                    enclosed = True
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if output[nr][nc] == 0:
                                enclosed = False
                                break
                    if enclosed:
                        output[r][c] = 4
                        changed = True
    return output"""

# ============ eb281b96 ============
solutions["eb281b96"] = """def solve(grid):
    n = len(grid)
    out_rows = 4*n - 3
    period = list(range(n)) + list(range(n-2, 0, -1))
    indices = []
    while len(indices) < out_rows:
        indices.extend(period)
    indices = indices[:out_rows]
    return [grid[i][:] for i in indices]"""

# ============ Verify all solutions ============
passing = {}
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
        print(f"PASS: {task_id}")
        passing[task_id] = code

# Save
with open("data/arc_python_solutions_b33.json", "w") as f:
    json.dump(passing, f, indent=2)
print(f"\nSaved {len(passing)} solutions")
