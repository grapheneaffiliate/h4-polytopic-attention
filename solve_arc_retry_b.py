import json

solutions = {}

solutions["5daaa586"] = """def solve(grid):
    from collections import Counter
    import copy
    rows, cols = len(grid), len(grid[0])
    vert_lines = []
    for c in range(cols):
        col_vals = [grid[r][c] for r in range(rows)]
        cnt = Counter(col_vals)
        dominant = cnt.most_common(1)[0]
        if dominant[0] != 0 and dominant[1] > rows * 0.8:
            vert_lines.append((c, dominant[0]))
    left_c, left_color = vert_lines[0]
    right_c, right_color = vert_lines[1]
    horiz_lines = []
    for r in range(rows):
        region = grid[r][left_c+1:right_c]
        unique = set(region)
        if len(unique) == 1 and unique.pop() != 0:
            horiz_lines.append((r, grid[r][left_c+1]))
    top_r, top_color = horiz_lines[0]
    bot_r, bot_color = horiz_lines[1]
    content_colors = Counter()
    for r in range(top_r+1, bot_r):
        for c in range(left_c+1, right_c):
            v = grid[r][c]
            if v != 0:
                content_colors[v] += 1
    content_color = content_colors.most_common(1)[0][0]
    ih = bot_r - top_r + 1
    iw = right_c - left_c + 1
    out = []
    for r in range(top_r, bot_r + 1):
        out.append([grid[r][c] for c in range(left_c, right_c + 1)])
    for r in range(1, ih - 1):
        for c in range(1, iw - 1):
            if out[r][c] == content_color:
                out[r][c] = 0
    if content_color == bot_color:
        for c in range(left_c+1, right_c):
            positions = [r for r in range(top_r+1, bot_r) if grid[r][c] == content_color]
            if positions:
                topmost = min(positions)
                for r in range(topmost, bot_r):
                    out[r - top_r][c - left_c] = content_color
    elif content_color == top_color:
        for c in range(left_c+1, right_c):
            positions = [r for r in range(top_r+1, bot_r) if grid[r][c] == content_color]
            if positions:
                bottommost = max(positions)
                for r in range(top_r+1, bottommost+1):
                    out[r - top_r][c - left_c] = content_color
    elif content_color == right_color:
        for r in range(top_r+1, bot_r):
            positions = [c for c in range(left_c+1, right_c) if grid[r][c] == content_color]
            if positions:
                leftmost = min(positions)
                for c in range(leftmost, right_c):
                    out[r - top_r][c - left_c] = content_color
    elif content_color == left_color:
        for r in range(top_r+1, bot_r):
            positions = [c for c in range(left_c+1, right_c) if grid[r][c] == content_color]
            if positions:
                rightmost = max(positions)
                for c in range(left_c+1, rightmost+1):
                    out[r - top_r][c - left_c] = content_color
    return out"""

solutions["6aa20dc0"] = """def solve(grid):
    import copy
    from collections import Counter
    rows, cols = len(grid), len(grid[0])
    cnt = Counter()
    for row in grid:
        cnt.update(row)
    bg = cnt.most_common(1)[0][0]
    non_bg = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                non_bg[(r,c)] = grid[r][c]
    visited = set()
    components = []
    for (r,c) in non_bg:
        if (r,c) in visited:
            continue
        comp = []
        queue = [(r,c)]
        visited.add((r,c))
        while queue:
            cr, cc = queue.pop(0)
            comp.append((cr, cc, non_bg[(cr,cc)]))
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = cr+dr, cc+dc
                    if (nr,nc) in non_bg and (nr,nc) not in visited:
                        visited.add((nr,nc))
                        queue.append((nr,nc))
        components.append(comp)
    main_shape = None
    isolated = []
    for comp in components:
        colors = set(v for _,_,v in comp)
        if len(colors) >= 3:
            main_shape = comp
        else:
            isolated.append(comp)
    if main_shape is None:
        for comp in components:
            colors = set(v for _,_,v in comp)
            if len(colors) >= 2:
                main_shape = comp
                break
    shape_by_color = {}
    for r, c, v in main_shape:
        shape_by_color.setdefault(v, []).append((r,c))
    color_counts = {v: len(cells) for v, cells in shape_by_color.items()}
    middle_color = max(color_counts, key=color_counts.get)
    other_colors = sorted([v for v in shape_by_color if v != middle_color])
    c1, c2 = other_colors[0], other_colors[1]
    shape_c1 = shape_by_color[c1][0]
    shape_c2 = shape_by_color[c2][0]
    orig_vec = (shape_c2[0] - shape_c1[0], shape_c2[1] - shape_c1[1])
    iso_groups = {}
    for comp in isolated:
        color = comp[0][2]
        iso_groups.setdefault(color, []).append(comp)
    c1_isos = iso_groups.get(c1, [])
    c2_isos = iso_groups.get(c2, [])
    out = copy.deepcopy(grid)
    used_c2 = set()
    for c1_comp in c1_isos:
        c1_tl = (min(r for r,c,v in c1_comp), min(c for r,c,v in c1_comp))
        c1_size = int(len(c1_comp) ** 0.5 + 0.5)
        best_dist = float('inf')
        best_idx = -1
        for idx, c2_comp in enumerate(c2_isos):
            if idx in used_c2: continue
            c2_size = int(len(c2_comp) ** 0.5 + 0.5)
            if c2_size != c1_size: continue
            c2_tl = (min(r for r,c,v in c2_comp), min(c for r,c,v in c2_comp))
            dist = abs(c2_tl[0]-c1_tl[0]) + abs(c2_tl[1]-c1_tl[1])
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx == -1: continue
        used_c2.add(best_idx)
        c2_comp = c2_isos[best_idx]
        c2_tl = (min(r for r,c,v in c2_comp), min(c for r,c,v in c2_comp))
        block_size = c1_size
        iso_vec = (c2_tl[0] - c1_tl[0], c2_tl[1] - c1_tl[1])
        sr = iso_vec[0] // block_size
        sc = iso_vec[1] // block_size
        transforms = [
            (1, 0, 0, 1), (-1, 0, 0, -1), (0, 1, -1, 0), (0, -1, 1, 0),
            (1, 0, 0, -1), (-1, 0, 0, 1), (0, 1, 1, 0), (0, -1, -1, 0),
        ]
        found = None
        for a, b, cc, d in transforms:
            tr = a * orig_vec[0] + b * orig_vec[1]
            tc = cc * orig_vec[0] + d * orig_vec[1]
            if tr == sr and tc == sc:
                found = (a, b, cc, d)
                break
        if found is None: continue
        a, b, c_t, d = found
        for r, co, v in main_shape:
            if v == middle_color:
                dr = r - shape_c1[0]
                dc = co - shape_c1[1]
                tr = a * dr + b * dc
                tc = c_t * dr + d * dc
                for br in range(block_size):
                    for bc in range(block_size):
                        nr = c1_tl[0] + tr * block_size + br
                        nc = c1_tl[1] + tc * block_size + bc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            out[nr][nc] = middle_color
    return out"""

solutions["6d58a25d"] = """def solve(grid):
    import copy
    from collections import Counter
    rows, cols = len(grid), len(grid[0])
    cnt = Counter()
    for row in grid:
        for v in row:
            if v != 0: cnt[v] += 1
    funnel_color = None
    scatter_color = None
    for v in cnt:
        cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == v]
        if len(cells) == 10:
            funnel_color = v
    scatter_color = [v for v in cnt if v != funnel_color][0]
    funnel_cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == funnel_color]
    funnel_rows = {}
    for r, c in funnel_cells:
        funnel_rows.setdefault(r, set()).add(c)
    bottom_row = max(funnel_rows.keys())
    bottom_cols = funnel_rows[bottom_row]
    gap_left = min(bottom_cols) + 1
    gap_right = max(bottom_cols) - 1
    gap_by_row = {}
    for r in sorted(funnel_rows.keys()):
        fcols = funnel_rows[r]
        if len(fcols) >= 2:
            gaps = set(c for c in range(min(fcols)+1, max(fcols)) if c not in fcols)
            if gaps:
                gap_by_row[r] = gaps
    scatter_cols = set()
    for r in range(bottom_row+1, rows):
        for c in range(gap_left, gap_right+1):
            if grid[r][c] == scatter_color:
                scatter_cols.add(c)
    out = copy.deepcopy(grid)
    for sc in scatter_cols:
        start_row = None
        for r in sorted(gap_by_row.keys()):
            if sc in gap_by_row[r]:
                start_row = r
                break
        if start_row is None:
            start_row = bottom_row + 1
        for r in range(start_row, rows):
            out[r][sc] = scatter_color
    return out"""

solutions["73251a56"] = """def solve(grid):
    rows, cols = len(grid), len(grid[0])
    N = max(max(row) for row in grid)
    V = None
    for i in range(min(rows, cols)):
        if grid[i][i] != 0:
            V = grid[i][i]
            break
    if V is None:
        V = grid[0][0]
    V_adj = V - 1 if V > 1 else N
    out = [[0]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            m = min(r, c)
            d = abs(r - c)
            if d == 0:
                val = V
            elif d <= m + 1:
                val = V_adj
            else:
                group_size = m + 2
                start_d = m + 2
                group = (d - start_d) // group_size
                val = (V + group - 1) % N + 1
            out[r][c] = val
    return out"""

solutions["7df24a62"] = """def solve(grid):
    import copy
    rows, cols = len(grid), len(grid[0])
    stamp_cells = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in (0, 4):
                stamp_cells.add((r, c))
    sr1 = min(r for r,c in stamp_cells)
    sc1 = min(c for r,c in stamp_cells)
    sr2 = max(r for r,c in stamp_cells)
    sc2 = max(c for r,c in stamp_cells)
    sh, sw = sr2-sr1+1, sc2-sc1+1
    stamp = []
    for r in range(sr1, sr2+1):
        row = []
        for c in range(sc1, sc2+1):
            row.append(grid[r][c])
        stamp.append(row)
    def rotate90(arr):
        h = len(arr)
        w = len(arr[0])
        return [[arr[h-1-c][r] for c in range(h)] for r in range(w)]
    def fliph(arr):
        return [row[::-1] for row in arr]
    all_orients = []
    cur = [row[:] for row in stamp]
    for k in range(4):
        all_orients.append(cur)
        all_orients.append(fliph(cur))
        cur = rotate90(cur)
    unique_orients = []
    seen_patterns = set()
    for arr in all_orients:
        h = len(arr)
        w = len(arr[0])
        fours = tuple(sorted((r,c) for r in range(h) for c in range(w) if arr[r][c] == 4))
        key = (h, w, fours)
        if key not in seen_patterns:
            seen_patterns.add(key)
            unique_orients.append((h, w, fours, arr))
    outside_4s = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 4 and not (sr1 <= r <= sr2 and sc1 <= c <= sc2):
                outside_4s.add((r, c))
    out = copy.deepcopy(grid)
    placed = set()
    for h, w, fours, arr in unique_orients:
        for ref_dr, ref_dc in fours:
            for fr, fc in outside_4s:
                tl_r = fr - ref_dr
                tl_c = fc - ref_dc
                all_match = True
                for dr, dc in fours:
                    if (tl_r + dr, tl_c + dc) not in outside_4s:
                        all_match = False
                        break
                if not all_match:
                    continue
                placement_key = (tl_r, tl_c, h, w)
                if placement_key in placed:
                    continue
                placed.add(placement_key)
                for dr in range(h):
                    for dc in range(w):
                        r2, c2 = tl_r + dr, tl_c + dc
                        if 0 <= r2 < rows and 0 <= c2 < cols:
                            if grid[r2][c2] == 0:
                                out[r2][c2] = arr[dr][dc]
    return out"""

solutions["a64e4611"] = """def solve(grid):
    import copy
    rows, cols = len(grid), len(grid[0])
    threshold = rows - 5
    high_zero = [c for c in range(cols) if sum(1 for r in range(rows) if grid[r][c] == 0) >= threshold]
    if not high_zero:
        return grid
    runs = []
    s = high_zero[0]
    for i in range(1, len(high_zero)):
        if high_zero[i] != high_zero[i-1] + 1:
            runs.append((s, high_zero[i-1]))
            s = high_zero[i]
    runs.append((s, high_zero[-1]))
    longest = max(runs, key=lambda x: x[1]-x[0]+1)
    vc_start = longest[0] + 1
    vc_end = longest[1] - 1
    vert_rows = [r for r in range(rows) if all(grid[r][c] == 0 for c in range(vc_start, vc_end+1))]
    vr_runs = []
    s = vert_rows[0]
    for i in range(1, len(vert_rows)):
        if vert_rows[i] != vert_rows[i-1] + 1:
            vr_runs.append((s, vert_rows[i-1]))
            s = vert_rows[i]
    vr_runs.append((s, vert_rows[-1]))
    longest_vr = max(vr_runs, key=lambda x: x[1]-x[0]+1)
    vr_start = longest_vr[0] + (1 if longest_vr[0] > 0 else 0)
    vr_end = longest_vr[1] - (1 if longest_vr[1] < rows-1 else 0)
    left_cols = range(0, vc_start)
    right_cols = range(vc_end+1, cols)
    right_ext_rows = []
    left_ext_rows = []
    for r in range(vr_start, vr_end+1):
        right_all_zero = all(grid[r][c] == 0 for c in right_cols)
        left_all_zero = all(grid[r][c] == 0 for c in left_cols)
        if right_all_zero:
            right_ext_rows.append(r)
        if left_all_zero:
            left_ext_rows.append(r)
    def shrink_runs(row_list):
        if not row_list:
            return []
        rns = []
        s2 = row_list[0]
        for i in range(1, len(row_list)):
            if row_list[i] != row_list[i-1] + 1:
                rns.append((s2, row_list[i-1]))
                s2 = row_list[i]
        rns.append((s2, row_list[-1]))
        result = []
        for start, end in rns:
            if end - start >= 2:
                result.extend(range(start+1, end))
        return result
    right_ext = shrink_runs(right_ext_rows)
    left_ext = shrink_runs(left_ext_rows)
    out = copy.deepcopy(grid)
    for r in range(vr_start, vr_end+1):
        for c in range(vc_start, vc_end+1):
            if grid[r][c] == 0:
                out[r][c] = 3
    for r in right_ext:
        for c in range(vc_end+1, cols):
            if grid[r][c] == 0:
                out[r][c] = 3
    for r in left_ext:
        for c in range(0, vc_start):
            if grid[r][c] == 0:
                out[r][c] = 3
    return out"""

# Verify
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

with open("data/arc_python_solutions_retry_b.json", "w") as f:
    json.dump(solutions, f, indent=2)
print(f"Saved {len(solutions)} solutions")
