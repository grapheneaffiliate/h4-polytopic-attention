import json

solutions = {}

solutions["eb5a1d5d"] = """def solve(grid):
    rows, cols = len(grid), len(grid[0])
    bg = grid[0][0]
    color_boxes = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != bg:
                if v not in color_boxes:
                    color_boxes[v] = [r, c, r, c]
                else:
                    color_boxes[v][0] = min(color_boxes[v][0], r)
                    color_boxes[v][1] = min(color_boxes[v][1], c)
                    color_boxes[v][2] = max(color_boxes[v][2], r)
                    color_boxes[v][3] = max(color_boxes[v][3], c)
    sorted_colors = sorted(color_boxes.keys(), key=lambda v: (color_boxes[v][2]-color_boxes[v][0]+1)*(color_boxes[v][3]-color_boxes[v][1]+1), reverse=True)
    all_layers = [bg] + sorted_colors
    n = len(all_layers)
    size = 2*n - 1
    output = [[0]*size for _ in range(size)]
    for i, color in enumerate(all_layers):
        for r in range(i, size-i):
            for c in range(i, size-i):
                output[r][c] = color
    return output"""

solutions["ec883f72"] = """def solve(grid):
    rows, cols = len(grid), len(grid[0])
    output = [row[:] for row in grid]
    colors = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != 0:
                if v not in colors:
                    colors[v] = []
                colors[v].append((r, c))
    if len(colors) < 2:
        return output
    color_list = list(colors.keys())
    def fill_ratio(cells):
        rr = [r for r, c in cells]
        cc = [c for r, c in cells]
        r1, c1, r2, c2 = min(rr), min(cc), max(rr), max(cc)
        area = (r2-r1+1) * (c2-c1+1)
        return len(cells) / area if area > 0 else 0
    c0, c1_val = color_list[0], color_list[1]
    fr0 = fill_ratio(colors[c0])
    fr1 = fill_ratio(colors[c1_val])
    if fr0 > fr1:
        inner_color = c0
        enc_color = c1_val
    else:
        inner_color = c1_val
        enc_color = c0
    enc_cells = set(colors[enc_color])
    inner_cells = set(colors[inner_color])
    corner_cells = []
    for r, c in enc_cells:
        adj = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            if (r+dr, c+dc) in enc_cells:
                adj.append((dr, dc))
        if len(adj) == 2:
            d1, d2 = adj
            if d1[0] * d2[0] + d1[1] * d2[1] == 0:
                esc_dr = -(d1[0] + d2[0])
                esc_dc = -(d1[1] + d2[1])
                nr, nc = r + esc_dr, c + esc_dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) not in enc_cells and (nr, nc) not in inner_cells:
                        corner_cells.append((r, c, esc_dr, esc_dc))
    for cr, cc, dr, dc in corner_cells:
        r, c = cr + dr, cc + dc
        while 0 <= r < rows and 0 <= c < cols:
            if output[r][c] == 0:
                output[r][c] = inner_color
            r += dr
            c += dc
    return output"""

solutions["ecdecbb3"] = """def solve(grid):
    rows, cols = len(grid), len(grid[0])
    output = [row[:] for row in grid]
    h_lines = sorted([r for r in range(rows) if all(grid[r][c] == 8 for c in range(cols))])
    v_lines = sorted([c for c in range(cols) if all(grid[r][c] == 8 for r in range(rows))])
    dots = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                dots.append((r, c))
    connections = []
    for dr, dc in dots:
        above = [lr for lr in h_lines if lr < dr]
        below = [lr for lr in h_lines if lr > dr]
        if above:
            connections.append(((dr, dc), 'h', above[-1]))
        if below:
            connections.append(((dr, dc), 'h', below[0]))
        left = [lc for lc in v_lines if lc < dc]
        right = [lc for lc in v_lines if lc > dc]
        if left:
            connections.append(((dr, dc), 'v', left[-1]))
        if right:
            connections.append(((dr, dc), 'v', right[0]))
    intersections = []
    for (dr, dc), ltype, lpos in connections:
        if ltype == 'h':
            intersections.append((lpos, dc))
        else:
            intersections.append((dr, lpos))
    for ir, ic in intersections:
        for rr in range(ir - 1, ir + 2):
            for cc in range(ic - 1, ic + 2):
                if 0 <= rr < rows and 0 <= cc < cols and output[rr][cc] == 0:
                    output[rr][cc] = 8
    for (dr, dc), ltype, lpos in connections:
        if ltype == 'h':
            col = dc
            output[lpos][col] = 2
            if dr < lpos:
                for r in range(dr + 1, lpos):
                    if output[r][col] != 8:
                        output[r][col] = 2
            else:
                for r in range(lpos + 1, dr):
                    if output[r][col] != 8:
                        output[r][col] = 2
        else:
            row = dr
            output[row][lpos] = 2
            if dc < lpos:
                for c in range(dc + 1, lpos):
                    if output[row][c] != 8:
                        output[row][c] = 2
            else:
                for c in range(lpos + 1, dc):
                    if output[row][c] != 8:
                        output[row][c] = 2
    return output"""

solutions["f1cefba8"] = """def solve(grid):
    rows, cols = len(grid), len(grid[0])
    output = [row[:] for row in grid]
    bg = 0
    frame_cells = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                v = grid[r][c]
                if v not in frame_cells:
                    frame_cells[v] = []
                frame_cells[v].append((r, c))
    colors = sorted(frame_cells.keys(), key=lambda v: len(frame_cells[v]), reverse=True)
    frame_color = colors[0]
    fill_color = colors[1]
    all_cells = frame_cells[frame_color] + frame_cells[fill_color]
    struct_min_r = min(r for r, c in all_cells)
    struct_max_r = max(r for r, c in all_cells)
    struct_min_c = min(c for r, c in all_cells)
    struct_max_c = max(c for r, c in all_cells)
    from collections import Counter
    row_counts = Counter(r for r, c in frame_cells[fill_color])
    col_counts = Counter(c for r, c in frame_cells[fill_color])
    max_row_count = max(row_counts.values())
    max_col_count = max(col_counts.values())
    main_fill_rows = [r for r, cnt in row_counts.items() if cnt >= max_row_count - 1]
    main_fill_cols = [c for c, cnt in col_counts.items() if cnt >= max_col_count - 1]
    inner_min_r = min(main_fill_rows)
    inner_max_r = max(main_fill_rows)
    inner_min_c = min(main_fill_cols)
    inner_max_c = max(main_fill_cols)
    notch_cols = set()
    notch_rows = set()
    for r, c in frame_cells[fill_color]:
        if r < inner_min_r or r > inner_max_r:
            notch_cols.add(c)
        if c < inner_min_c or c > inner_max_c:
            notch_rows.add(r)
    for r in range(struct_min_r, struct_max_r + 1):
        for c in range(struct_min_c, struct_max_c + 1):
            output[r][c] = frame_color
    for r in range(inner_min_r, inner_max_r + 1):
        for c in range(inner_min_c, inner_max_c + 1):
            output[r][c] = fill_color
    for nc in notch_cols:
        for r in range(inner_min_r, inner_max_r + 1):
            output[r][nc] = frame_color
    for nr in notch_rows:
        for c in range(inner_min_c, inner_max_c + 1):
            output[nr][c] = frame_color
    for nc in notch_cols:
        for r in range(struct_min_r, struct_max_r + 1):
            output[r][nc] = frame_color
    for nr in notch_rows:
        for c in range(struct_min_c, struct_max_c + 1):
            output[nr][c] = frame_color
    for nc in notch_cols:
        for r in range(0, struct_min_r):
            output[r][nc] = fill_color
        for r in range(struct_max_r + 1, rows):
            output[r][nc] = fill_color
    for nr in notch_rows:
        for c in range(0, struct_min_c):
            output[nr][c] = fill_color
        for c in range(struct_max_c + 1, cols):
            output[nr][c] = fill_color
    return output"""

solutions["f15e1fac"] = """def solve(grid):
    rows, cols = len(grid), len(grid[0])
    output = [[0]*cols for _ in range(rows)]
    eights = []
    twos = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8:
                eights.append((r, c))
            elif grid[r][c] == 2:
                twos.append((r, c))
    for r, c in twos:
        output[r][c] = 2
    eight_rows = set(r for r, c in eights)
    eight_cols = set(c for r, c in eights)
    two_rows = set(r for r, c in twos)
    two_cols = set(c for r, c in twos)
    if len(eight_rows) == 1:
        pattern_cols = sorted(c for r, c in eights)
        marker_col = list(two_cols)[0]
        breakpoints = sorted(r for r, c in twos)
        if marker_col <= cols // 2:
            shift = 1
        else:
            shift = -1
        seg_starts = [0] + breakpoints
        seg_ends = breakpoints + [rows]
        for seg_idx in range(len(seg_starts)):
            seg_start = seg_starts[seg_idx]
            seg_end = seg_ends[seg_idx]
            shifted_cols = [pc + seg_idx * shift for pc in pattern_cols]
            for r in range(seg_start, seg_end):
                for sc in shifted_cols:
                    if 0 <= sc < cols:
                        output[r][sc] = 8
    elif len(eight_cols) == 1:
        pattern_rows = sorted(r for r, c in eights)
        marker_row = list(two_rows)[0]
        breakpoints = sorted(c for r, c in twos)
        shift = -1
        seg_starts = [0] + breakpoints
        seg_ends = breakpoints + [cols]
        for seg_idx in range(len(seg_starts)):
            seg_c_start = seg_starts[seg_idx]
            seg_c_end = seg_ends[seg_idx]
            for pr in pattern_rows:
                shifted_row = pr + seg_idx * shift
                if 0 <= shifted_row < rows:
                    for c in range(seg_c_start, seg_c_end):
                        output[shifted_row][c] = 8
    for r, c in twos:
        output[r][c] = 2
    return output"""

solutions["fcc82909"] = """def solve(grid):
    rows, cols = len(grid), len(grid[0])
    output = [row[:] for row in grid]
    visited = [[False]*cols for _ in range(rows)]
    for r in range(rows-1):
        for c in range(cols-1):
            if (grid[r][c] != 0 and grid[r][c+1] != 0 and
                grid[r+1][c] != 0 and grid[r+1][c+1] != 0 and
                not visited[r][c]):
                visited[r][c] = visited[r][c+1] = visited[r+1][c] = visited[r+1][c+1] = True
                colors = {grid[r][c], grid[r][c+1], grid[r+1][c], grid[r+1][c+1]}
                shadow_len = len(colors)
                for dr in range(1, shadow_len + 1):
                    nr = r + 1 + dr
                    if nr < rows:
                        output[nr][c] = 3
                        output[nr][c+1] = 3
    return output"""

# Verify all solutions
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

# Save passing solutions
with open("data/arc_python_solutions_b34.json", "w") as f:
    json.dump(passing, f, indent=2)
print(f"\nSaved {len(passing)} solutions")
