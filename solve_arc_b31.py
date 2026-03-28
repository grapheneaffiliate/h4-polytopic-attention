import json

solutions = {}

solutions["90c28cc7"] = """def solve(grid):
    rows, cols = len(grid), len(grid[0])
    min_r, max_r, min_c, max_c = rows, 0, cols, 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    col_boundaries = set()
    for r in range(min_r, max_r+1):
        for c in range(min_c+1, max_c+1):
            if grid[r][c] != grid[r][c-1]:
                col_boundaries.add(c)
    row_boundaries = set()
    for c in range(min_c, max_c+1):
        for r in range(min_r+1, max_r+1):
            if grid[r][c] != grid[r-1][c]:
                row_boundaries.add(r)
    col_breaks = sorted([min_c] + list(col_boundaries) + [max_c+1])
    row_breaks = sorted([min_r] + list(row_boundaries) + [max_r+1])
    out = []
    for ri in range(len(row_breaks)-1):
        row = []
        for ci in range(len(col_breaks)-1):
            row.append(grid[row_breaks[ri]][col_breaks[ci]])
        out.append(row)
    return out"""

solutions["91714a58"] = """def solve(grid):
    rows, cols = len(grid), len(grid[0])
    best = None
    best_area = 0
    for val in range(1, 10):
        for r1 in range(rows):
            for c1 in range(cols):
                if grid[r1][c1] != val:
                    continue
                max_c2 = cols - 1
                for r2 in range(r1, rows):
                    if grid[r2][c1] != val:
                        break
                    c2 = c1
                    while c2 + 1 <= max_c2 and grid[r2][c2+1] == val:
                        c2 += 1
                    max_c2 = min(max_c2, c2)
                    area = (r2 - r1 + 1) * (max_c2 - c1 + 1)
                    if area > best_area:
                        best_area = area
                        best = (val, r1, r2, c1, max_c2)
    output = [[0]*cols for _ in range(rows)]
    if best:
        val, r1, r2, c1, c2 = best
        for r in range(r1, r2+1):
            for c in range(c1, c2+1):
                output[r][c] = val
    return output"""

solutions["97a05b5b"] = """def solve(grid):
    rows, cols = len(grid), len(grid[0])
    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2 and (r,c) not in visited:
                comp = set()
                queue = [(r,c)]
                while queue:
                    cr,cc = queue.pop(0)
                    if (cr,cc) in visited or grid[cr][cc] != 2:
                        continue
                    visited.add((cr,cc))
                    comp.add((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc = cr+dr,cc+dc
                        if 0<=nr<rows and 0<=nc<cols:
                            queue.append((nr,nc))
                components.append(comp)
    main_comp = max(components, key=len)
    mr1 = min(r for r,c in main_comp)
    mr2 = max(r for r,c in main_comp)
    mc1 = min(c for r,c in main_comp)
    mc2 = max(c for r,c in main_comp)
    rh = mr2-mr1+1
    rw = mc2-mc1+1
    rect = [[grid[r+mr1][c+mc1] for c in range(rw)] for r in range(rh)]
    all_zeros = set()
    for r in range(rh):
        for c in range(rw):
            if rect[r][c] == 0:
                all_zeros.add((r,c))
    patch_visited = set()
    patch_grids = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and (r,c) not in patch_visited:
                if mr1 <= r <= mr2 and mc1 <= c <= mc2:
                    continue
                patch = {}
                queue = [(r,c)]
                while queue:
                    cr,cc = queue.pop(0)
                    if (cr,cc) in patch_visited or grid[cr][cc] == 0:
                        continue
                    if mr1 <= cr <= mr2 and mc1 <= cc <= mc2:
                        continue
                    patch_visited.add((cr,cc))
                    patch[(cr,cc)] = grid[cr][cc]
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc = cr+dr,cc+dc
                        if 0<=nr<rows and 0<=nc<cols:
                            queue.append((nr,nc))
                if patch:
                    pr1 = min(r for r,c in patch)
                    pc1 = min(c for r,c in patch)
                    pr2 = max(r for r,c in patch)
                    pc2 = max(c for r,c in patch)
                    g = [[2]*(pc2-pc1+1) for _ in range(pr2-pr1+1)]
                    for (rr,cc), v in patch.items():
                        g[rr-pr1][cc-pc1] = v
                    patch_grids.append(g)
    def get_variants(g):
        variants = []
        cur = g
        for _ in range(4):
            variants.append([row[:] for row in cur])
            variants.append([row[::-1] for row in cur])
            ch, cw = len(cur), len(cur[0])
            cur = [[cur[ch-1-c][r] for c in range(ch)] for r in range(cw)]
        unique = []
        seen = set()
        for v in variants:
            key = tuple(tuple(row) for row in v)
            if key not in seen:
                seen.add(key)
                unique.append(v)
        return unique
    output = [row[:] for row in rect]
    for r,c in all_zeros:
        output[r][c] = 2
    remaining_zeros = set(all_zeros)
    used = set()
    placements = []
    for pi, pg in enumerate(patch_grids):
        for variant in get_variants(pg):
            vh, vw = len(variant), len(variant[0])
            twos = [(r,c) for r in range(vh) for c in range(vw) if variant[r][c] == 2]
            non2s = [(r,c,variant[r][c]) for r in range(vh) for c in range(vw) if variant[r][c] != 2]
            for tz_r, tz_c in twos:
                for z_r, z_c in all_zeros:
                    off_r = z_r - tz_r
                    off_c = z_c - tz_c
                    ok = True
                    mapped_zeros = set()
                    for tr, tc in twos:
                        rr, cc = tr+off_r, tc+off_c
                        if (rr,cc) not in all_zeros:
                            ok = False
                            break
                        mapped_zeros.add((rr,cc))
                    if not ok:
                        continue
                    for nr, nc, nv in non2s:
                        rr, cc = nr+off_r, nc+off_c
                        if rr < 0 or rr >= rh or cc < 0 or cc >= rw:
                            ok = False
                            break
                        if rect[rr][cc] != 2:
                            ok = False
                            break
                    if ok:
                        min_bd = float('inf')
                        for nr, nc, nv in non2s:
                            rr, cc = nr+off_r, nc+off_c
                            d = min(rr, rh-1-rr, cc, rw-1-cc)
                            min_bd = min(min_bd, d)
                        placements.append((min_bd, pi, variant, off_r, off_c, mapped_zeros))
                    break
    placements.sort(key=lambda x: -x[0])
    for score, pi, variant, off_r, off_c, mapped_zeros in placements:
        if pi in used:
            continue
        if not mapped_zeros.issubset(remaining_zeros):
            continue
        vh, vw = len(variant), len(variant[0])
        for r in range(vh):
            for c in range(vw):
                if variant[r][c] != 2:
                    output[r+off_r][c+off_c] = variant[r][c]
        remaining_zeros -= mapped_zeros
        used.add(pi)
    return output"""

solutions["98cf29f8"] = """def solve(grid):
    rows, cols = len(grid), len(grid[0])
    from collections import Counter
    colors = Counter()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                colors[grid[r][c]] += 1
    color_list = list(colors.keys())
    color_info = {}
    for col_val in color_list:
        positions = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == col_val]
        min_r = min(p[0] for p in positions)
        max_r = max(p[0] for p in positions)
        min_c = min(p[1] for p in positions)
        max_c = max(p[1] for p in positions)
        area = (max_r-min_r+1)*(max_c-min_c+1)
        is_rect = len(positions) == area
        color_info[col_val] = {
            'positions': set(positions), 'count': len(positions),
            'bbox': (min_r, max_r, min_c, max_c), 'area': area, 'is_rect': is_rect
        }
    anchor_color = None
    mover_color = None
    for col_val in color_list:
        if color_info[col_val]['is_rect']:
            anchor_color = col_val
        else:
            mover_color = col_val
    if anchor_color is None or mover_color is None:
        sorted_colors = sorted(color_list, key=lambda c: color_info[c]['count'], reverse=True)
        anchor_color = sorted_colors[0]
        mover_color = sorted_colors[1] if len(sorted_colors) > 1 else sorted_colors[0]
    mover_positions = color_info[mover_color]['positions']
    best_rect = None
    best_rect_area = 0
    mr1, mr2, mc1, mc2 = color_info[mover_color]['bbox']
    for r1 in range(mr1, mr2+1):
        for c1 in range(mc1, mc2+1):
            if (r1, c1) not in mover_positions:
                continue
            max_c2 = mc2
            for r2 in range(r1, mr2+1):
                if (r2, c1) not in mover_positions:
                    break
                c2 = c1
                while c2 + 1 <= max_c2 and (r2, c2+1) in mover_positions:
                    c2 += 1
                max_c2 = min(max_c2, c2)
                area = (r2 - r1 + 1) * (max_c2 - c1 + 1)
                if area > best_rect_area:
                    best_rect_area = area
                    best_rect = (r1, r2, c1, max_c2)
    rect_r1, rect_r2, rect_c1, rect_c2 = best_rect
    rect_h = rect_r2 - rect_r1 + 1
    rect_w = rect_c2 - rect_c1 + 1
    rect_set = set()
    for r in range(rect_r1, rect_r2+1):
        for c in range(rect_c1, rect_c2+1):
            rect_set.add((r,c))
    line_positions = mover_positions - rect_set
    anc_r1, anc_r2, anc_c1, anc_c2 = color_info[anchor_color]['bbox']
    if len(line_positions) > 0:
        if all(c == list(line_positions)[0][1] for r,c in line_positions):
            if rect_r1 > anc_r2:
                new_r1 = anc_r2 + 1
                new_c1 = rect_c1
            else:
                new_r1 = anc_r1 - rect_h
                new_c1 = rect_c1
        else:
            if rect_c1 > anc_c2:
                new_r1 = rect_r1
                new_c1 = anc_c2 + 1
            else:
                new_r1 = rect_r1
                new_c1 = anc_c1 - rect_w
    output = [[0]*cols for _ in range(rows)]
    for r in range(anc_r1, anc_r2+1):
        for c in range(anc_c1, anc_c2+1):
            output[r][c] = anchor_color
    for dr in range(rect_h):
        for dc in range(rect_w):
            output[new_r1+dr][new_c1+dc] = mover_color
    return output"""

solutions["9aec4887"] = """def solve(grid):
    rows, cols = len(grid), len(grid[0])
    eight_pos = []
    non_eight = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8:
                eight_pos.append((r,c))
            elif grid[r][c] != 0:
                if grid[r][c] not in non_eight:
                    non_eight[grid[r][c]] = []
                non_eight[grid[r][c]].append((r,c))
    frame_colors = {}
    for color, positions in non_eight.items():
        rs = [p[0] for p in positions]
        cs = [p[1] for p in positions]
        if len(set(rs)) == 1:
            frame_colors[color] = ('h', min(rs), min(cs), max(cs))
        elif len(set(cs)) == 1:
            frame_colors[color] = ('v', min(rs), max(rs), min(cs))
    h_colors = [(c, info) for c, info in frame_colors.items() if info[0] == 'h']
    v_colors = [(c, info) for c, info in frame_colors.items() if info[0] == 'v']
    h_colors.sort(key=lambda x: x[1][1])
    v_colors.sort(key=lambda x: x[1][3])
    top_color = h_colors[0][0]
    bottom_color = h_colors[1][0]
    left_color = v_colors[0][0]
    right_color = v_colors[1][0]
    pat_min_r = min(p[0] for p in eight_pos)
    pat_min_c = min(p[1] for p in eight_pos)
    pat_max_r = max(p[0] for p in eight_pos)
    pat_max_c = max(p[1] for p in eight_pos)
    pat_h = pat_max_r - pat_min_r + 1
    pat_w = pat_max_c - pat_min_c + 1
    n = pat_h
    out_h = n + 2
    out_w = pat_w + 2
    output = [[0] * out_w for _ in range(out_h)]
    for c in range(1, out_w - 1):
        output[0][c] = top_color
    for c in range(1, out_w - 1):
        output[out_h-1][c] = bottom_color
    for r in range(1, out_h - 1):
        output[r][0] = left_color
    for r in range(1, out_h - 1):
        output[r][out_w-1] = right_color
    for r, c in eight_pos:
        pr = r - pat_min_r
        pc = c - pat_min_c
        if pr == pc or pr + pc == n - 1:
            output[pr+1][pc+1] = 8
        elif pr < pc and pr + pc < n - 1:
            output[pr+1][pc+1] = top_color
        elif pr < pc and pr + pc > n - 1:
            output[pr+1][pc+1] = right_color
        elif pr >= pc and pr + pc < n - 1:
            output[pr+1][pc+1] = left_color
        elif pr >= pc and pr + pc > n - 1:
            output[pr+1][pc+1] = bottom_color
        else:
            output[pr+1][pc+1] = 8
    return output"""

solutions["9d9215db"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    output = [[0]*cols for _ in range(rows)]
    values = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                dr = min(r, rows-1-r)
                dc = min(c, cols-1-c)
                values[(dr, dc)] = grid[r][c]
    for (dr, dc), val in values.items():
        if dr == dc:
            for rr in [dr, rows-1-dr]:
                for cc in [dc, cols-1-dc]:
                    output[rr][cc] = val
        elif dr < dc:
            for cc in range(dc, cols-dc, 2):
                output[dr][cc] = val
                output[rows-1-dr][cc] = val
        else:
            for rr in range(dr, rows-dr, 2):
                output[rr][dc] = val
                output[rr][cols-1-dc] = val
    return output"""

solutions["9ecd008a"] = """def solve(grid):
    rows, cols = len(grid), len(grid[0])
    zeros = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 0]
    min_r = min(r for r,c in zeros)
    min_c = min(c for r,c in zeros)
    max_r = max(r for r,c in zeros)
    max_c = max(c for r,c in zeros)
    out_h = max_r - min_r + 1
    out_w = max_c - min_c + 1
    output = [[0]*out_w for _ in range(out_h)]
    for r,c in zeros:
        mr = rows - 1 - r
        mc = cols - 1 - c
        output[r-min_r][c-min_c] = grid[mr][mc]
    return output"""

solutions["9f236235"] = """def solve(grid):
    rows, cols = len(grid), len(grid[0])
    sep_color = None
    for r in range(rows):
        if all(grid[r][c] == grid[r][0] for c in range(cols)) and grid[r][0] != 0:
            sep_color = grid[r][0]
            break
    sep_rows = [r for r in range(rows) if all(grid[r][c] == sep_color for c in range(cols))]
    sep_cols = [c for c in range(cols) if all(grid[r][c] == sep_color for r in range(rows))]
    row_ranges = []
    prev = 0
    for sr in sep_rows:
        if sr > prev:
            row_ranges.append((prev, sr-1))
        prev = sr + 1
    if prev < rows:
        row_ranges.append((prev, rows-1))
    col_ranges = []
    prev = 0
    for sc in sep_cols:
        if sc > prev:
            col_ranges.append((prev, sc-1))
        prev = sc + 1
    if prev < cols:
        col_ranges.append((prev, cols-1))
    n_rows = len(row_ranges)
    n_cols = len(col_ranges)
    cell_grid = [[0]*n_cols for _ in range(n_rows)]
    for ri, (r1,r2) in enumerate(row_ranges):
        for ci, (c1,c2) in enumerate(col_ranges):
            val = grid[r1][c1]
            if val != 0 and val != sep_color:
                cell_grid[ri][ci] = val
    output = [row[::-1] for row in cell_grid]
    return output"""

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

# Save
with open("data/arc_python_solutions_b31.json", "w") as f:
    json.dump(solutions, f, indent=2)
print(f"Saved {len(solutions)} solutions")
