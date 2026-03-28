import json

solutions = {}

# ============================================================
# 0a938d79 - Two colored dots create repeating stripes
# ============================================================
solutions["0a938d79"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    output = [[0]*cols for _ in range(rows)]

    points = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                points.append((r, c, grid[r][c]))

    r1, c1, v1 = points[0]
    r2, c2, v2 = points[1]

    dr = abs(r2 - r1)
    dc = abs(c2 - c1)

    if dc == 0 or dr > dc:
        if dc > 0:
            # Vertical stripes
            if c1 < c2:
                first_c, first_v = c1, v1
                second_c, second_v = c2, v2
            else:
                first_c, first_v = c2, v2
                second_c, second_v = c1, v1
            spacing = second_c - first_c
            for c in range(cols):
                if c >= first_c:
                    rel = (c - first_c) % (spacing * 2)
                    if rel == 0:
                        for r in range(rows):
                            output[r][c] = first_v
                    elif rel == spacing:
                        for r in range(rows):
                            output[r][c] = second_v
            return output
        # Horizontal stripes
        if r1 < r2:
            first_r, first_v = r1, v1
            second_r, second_v = r2, v2
        else:
            first_r, first_v = r2, v2
            second_r, second_v = r1, v1
        spacing = second_r - first_r
        for r in range(rows):
            if r >= first_r:
                rel = (r - first_r) % (spacing * 2)
                if rel == 0:
                    for c in range(cols):
                        output[r][c] = first_v
                elif rel == spacing:
                    for c in range(cols):
                        output[r][c] = second_v
    else:
        # dr <= dc: horizontal stripes
        if r1 < r2:
            first_r, first_v = r1, v1
            second_r, second_v = r2, v2
        else:
            first_r, first_v = r2, v2
            second_r, second_v = r1, v1
        spacing = second_r - first_r
        if spacing == 0:
            # Same row -> vertical stripes
            if c1 < c2:
                first_c, first_v = c1, v1
                second_c, second_v = c2, v2
            else:
                first_c, first_v = c2, v2
                second_c, second_v = c1, v1
            sp = second_c - first_c
            for c in range(cols):
                if c >= first_c:
                    rel = (c - first_c) % (sp * 2)
                    if rel == 0:
                        for r in range(rows):
                            output[r][c] = first_v
                    elif rel == sp:
                        for r in range(rows):
                            output[r][c] = second_v
        else:
            for r in range(rows):
                if r >= first_r:
                    rel = (r - first_r) % (spacing * 2)
                    if rel == 0:
                        for c in range(cols):
                            output[r][c] = first_v
                    elif rel == spacing:
                        for c in range(cols):
                            output[r][c] = second_v
    return output
"""

# ============================================================
# 0b148d64 - Four quadrants, output the unique-colored one
# ============================================================
solutions["0b148d64"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    sep_rows = set()
    for r in range(rows):
        if all(grid[r][c] == 0 for c in range(cols)):
            sep_rows.add(r)

    sep_cols = set()
    for c in range(cols):
        if all(grid[r][c] == 0 for r in range(rows)):
            sep_cols.add(c)

    sorted_sep_rows = sorted(sep_rows)
    row_groups = []
    if sorted_sep_rows:
        start = sorted_sep_rows[0]
        for i in range(1, len(sorted_sep_rows)):
            if sorted_sep_rows[i] != sorted_sep_rows[i-1] + 1:
                row_groups.append((start, sorted_sep_rows[i-1]))
                start = sorted_sep_rows[i]
        row_groups.append((start, sorted_sep_rows[-1]))

    sorted_sep_cols = sorted(sep_cols)
    col_groups = []
    if sorted_sep_cols:
        start = sorted_sep_cols[0]
        for i in range(1, len(sorted_sep_cols)):
            if sorted_sep_cols[i] != sorted_sep_cols[i-1] + 1:
                col_groups.append((start, sorted_sep_cols[i-1]))
                start = sorted_sep_cols[i]
        col_groups.append((start, sorted_sep_cols[-1]))

    row_ranges = []
    prev = 0
    for rg in row_groups:
        if prev < rg[0]:
            row_ranges.append((prev, rg[0] - 1))
        prev = rg[1] + 1
    if prev < rows:
        row_ranges.append((prev, rows - 1))

    col_ranges = []
    prev = 0
    for cg in col_groups:
        if prev < cg[0]:
            col_ranges.append((prev, cg[0] - 1))
        prev = cg[1] + 1
    if prev < cols:
        col_ranges.append((prev, cols - 1))

    regions = []
    for rr in row_ranges:
        for cr in col_ranges:
            region = []
            colors = set()
            for r in range(rr[0], rr[1]+1):
                row = []
                for c in range(cr[0], cr[1]+1):
                    row.append(grid[r][c])
                    if grid[r][c] != 0:
                        colors.add(grid[r][c])
                region.append(row)
            regions.append((region, colors))

    from collections import Counter
    color_count = Counter()
    for region, colors in regions:
        for c in colors:
            color_count[c] += 1

    unique_color = None
    for c, cnt in color_count.items():
        if cnt == 1:
            unique_color = c
            break

    if unique_color is None:
        unique_color = min(color_count, key=color_count.get)

    for region, colors in regions:
        if unique_color in colors:
            return region

    return regions[0][0]
"""

# ============================================================
# 0dfd9992 - Fill holes in repeating wallpaper pattern
# ============================================================
solutions["0dfd9992"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]

    for pr in range(1, rows):
        for pc in range(1, cols):
            pattern = [[None]*pc for _ in range(pr)]
            ok = True
            for r in range(rows):
                for c in range(cols):
                    pr_idx = r % pr
                    pc_idx = c % pc
                    if grid[r][c] != 0:
                        if pattern[pr_idx][pc_idx] is None:
                            pattern[pr_idx][pc_idx] = grid[r][c]
                        elif pattern[pr_idx][pc_idx] != grid[r][c]:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                all_filled = all(pattern[r][c] is not None for r in range(pr) for c in range(pc))
                if all_filled:
                    for r in range(rows):
                        for c in range(cols):
                            output[r][c] = pattern[r % pr][c % pc]
                    return output
    return output
"""

# ============================================================
# 1a07d186 - Lines with stray dots absorbed to nearest matching line
# ============================================================
solutions["1a07d186"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    h_lines = {}
    for r in range(rows):
        vals = set(grid[r])
        if len(vals) == 1 and 0 not in vals:
            h_lines[r] = grid[r][0]

    v_lines = {}
    for c in range(cols):
        vals = set(grid[r][c] for r in range(rows))
        if len(vals) == 1 and 0 not in vals:
            v_lines[c] = grid[0][c]

    color_to_hline = {}
    color_to_vline = {}
    for r, color in h_lines.items():
        color_to_hline[color] = r
    for c, color in v_lines.items():
        color_to_vline[color] = c

    output = [[0]*cols for _ in range(rows)]
    for r, color in h_lines.items():
        for c in range(cols):
            output[r][c] = color
    for c, color in v_lines.items():
        for r in range(rows):
            output[r][c] = color

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and r not in h_lines and c not in v_lines:
                dot_color = grid[r][c]

                if dot_color in color_to_vline:
                    lc = color_to_vline[dot_color]
                    if c > lc:
                        output[r][lc + 1] = dot_color
                    else:
                        output[r][lc - 1] = dot_color
                elif dot_color in color_to_hline:
                    lr = color_to_hline[dot_color]
                    if r > lr:
                        output[lr + 1][c] = dot_color
                    else:
                        output[lr - 1][c] = dot_color

    return output
"""

# ============================================================
# 1c786137 - Extract interior of rectangular border
# ============================================================
solutions["1c786137"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    for border_color in set(grid[r][c] for r in range(rows) for c in range(cols)) - {0}:
        min_r = min_c = float('inf')
        max_r = max_c = -1
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == border_color:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)

        top_ok = all(grid[min_r][c] == border_color for c in range(min_c, max_c + 1))
        bot_ok = all(grid[max_r][c] == border_color for c in range(min_c, max_c + 1))
        left_ok = all(grid[r][min_c] == border_color for r in range(min_r, max_r + 1))
        right_ok = all(grid[r][max_c] == border_color for r in range(min_r, max_r + 1))

        if top_ok and bot_ok and left_ok and right_ok:
            interior = []
            for r in range(min_r + 1, max_r):
                row = []
                for c in range(min_c + 1, max_c):
                    row.append(grid[r][c])
                interior.append(row)
            return interior
    return grid
"""

# ============================================================
# 1e32b0e9 - 3x3 grid sections, union of shapes with divider color fill
# ============================================================
solutions["1e32b0e9"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]

    div_color = None
    div_rows = []
    div_cols = []

    for r in range(rows):
        if len(set(grid[r])) == 1 and grid[r][0] != 0:
            div_rows.append(r)
            div_color = grid[r][0]

    for c in range(cols):
        col_vals = set(grid[r][c] for r in range(rows))
        if len(col_vals) == 1 and list(col_vals)[0] == div_color:
            div_cols.append(c)

    all_row_bounds = [0] + div_rows + [rows]
    all_col_bounds = [0] + div_cols + [cols]

    sec_rows = []
    for i in range(len(all_row_bounds) - 1):
        rs = all_row_bounds[i]
        re = all_row_bounds[i+1]
        if rs in div_rows:
            rs += 1
        if re - 1 in div_rows:
            re -= 1
        if rs < re:
            sec_rows.append((rs, re))

    sec_cols = []
    for i in range(len(all_col_bounds) - 1):
        cs = all_col_bounds[i]
        ce = all_col_bounds[i+1]
        if cs in div_cols:
            cs += 1
        if ce - 1 in div_cols:
            ce -= 1
        if cs < ce:
            sec_cols.append((cs, ce))

    sec_h = sec_rows[0][1] - sec_rows[0][0]
    sec_w = sec_cols[0][1] - sec_cols[0][0]

    union_mask = [[False]*sec_w for _ in range(sec_h)]

    section_data = []
    for sr, (rs, re) in enumerate(sec_rows):
        for sc, (cs, ce) in enumerate(sec_cols):
            sec = []
            for r in range(rs, re):
                row = []
                for c in range(cs, ce):
                    row.append(grid[r][c])
                    if grid[r][c] != 0:
                        union_mask[r-rs][c-cs] = True
                sec.append(row)
            section_data.append((sr, sc, sec, rs, cs))

    for sr, sc, sec, rs, cs in section_data:
        for r in range(sec_h):
            for c in range(sec_w):
                if union_mask[r][c]:
                    if sec[r][c] == 0:
                        output[rs+r][cs+c] = div_color

    return output
"""

# ============================================================
# 1f85a75f - Find compact colored rectangle in noisy grid
# ============================================================
solutions["1f85a75f"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    from collections import Counter

    cc = Counter()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                cc[grid[r][c]] += 1

    for color in sorted(cc, key=cc.get):
        positions = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color:
                    positions.append((r, c))

        min_r = min(p[0] for p in positions)
        max_r = max(p[0] for p in positions)
        min_c = min(p[1] for p in positions)
        max_c = max(p[1] for p in positions)

        bbox_area = (max_r - min_r + 1) * (max_c - min_c + 1)
        density = len(positions) / bbox_area
        if density > 0.4 and bbox_area <= rows * cols / 4:
            result = []
            for r in range(min_r, max_r + 1):
                row = []
                for c in range(min_c, max_c + 1):
                    if grid[r][c] == color:
                        row.append(color)
                    else:
                        row.append(0)
                result.append(row)
            return result
    return grid
"""

# ============================================================
# 045e512c - Shape with arrow dots, replicate shape in arrow direction
# ============================================================
solutions["045e512c"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]

    visited = [[False]*cols for _ in range(rows)]
    def flood(r, c):
        stack = [(r,c)]
        comp = []
        while stack:
            cr, cc = stack.pop()
            if cr<0 or cr>=rows or cc<0 or cc>=cols: continue
            if visited[cr][cc] or grid[cr][cc] == 0: continue
            visited[cr][cc] = True
            comp.append((cr,cc,grid[cr][cc]))
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((cr+dr,cc+dc))
        return comp

    components = []
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != 0:
                comp = flood(r, c)
                if comp:
                    components.append(comp)

    from collections import defaultdict
    color_comps = defaultdict(list)
    for comp in components:
        color = comp[0][2]
        color_comps[color].append(comp)

    main_color = max(color_comps, key=lambda c: sum(len(comp) for comp in color_comps[c]))
    main_cells = []
    for comp in color_comps[main_color]:
        main_cells.extend(comp)

    min_r = min(c[0] for c in main_cells)
    max_r = max(c[0] for c in main_cells)
    min_c = min(c[1] for c in main_cells)
    max_c = max(c[1] for c in main_cells)
    shape_h = max_r - min_r + 1
    shape_w = max_c - min_c + 1

    shape_cells = [(r - min_r, c - min_c) for r, c, v in main_cells]
    shape_cr = (min_r + max_r) / 2
    shape_cc = (min_c + max_c) / 2

    arrow_groups = []
    for color, comps in color_comps.items():
        if color == main_color:
            continue
        if len(comps) == 1:
            arrow_groups.append(comps[0])
        else:
            centers = []
            for comp in comps:
                cr = sum(c[0] for c in comp) / len(comp)
                cc = sum(c[1] for c in comp) / len(comp)
                centers.append((cr, cc))

            directions = []
            for cr, cc in centers:
                dr = cr - shape_cr
                dc = cc - shape_cc
                dr_d = 0 if abs(dr) < 0.5 else (1 if dr > 0 else -1)
                dc_d = 0 if abs(dc) < 0.5 else (1 if dc > 0 else -1)
                directions.append((dr_d, dc_d))

            if len(set(directions)) == 1:
                merged = []
                for comp in comps:
                    merged.extend(comp)
                arrow_groups.append(merged)
            else:
                for comp in comps:
                    arrow_groups.append(comp)

    for arrow_cells in arrow_groups:
        arrow_color = arrow_cells[0][2]
        ar = sum(c[0] for c in arrow_cells) / len(arrow_cells)
        ac = sum(c[1] for c in arrow_cells) / len(arrow_cells)

        dr = ar - shape_cr
        dc = ac - shape_cc

        step_r = 0
        step_c = 0

        if abs(dr) > 0.5:
            if dr > 0:
                arrow_min_r = min(c[0] for c in arrow_cells)
                step_r = arrow_min_r - min_r
            else:
                arrow_max_r = max(c[0] for c in arrow_cells)
                step_r = -(max_r - arrow_max_r)

        if abs(dc) > 0.5:
            if dc > 0:
                arrow_min_c = min(c[1] for c in arrow_cells)
                step_c = arrow_min_c - min_c
            else:
                arrow_max_c = max(c[1] for c in arrow_cells)
                step_c = -(max_c - arrow_max_c)

        if step_r == 0 and step_c == 0:
            continue

        k = 1
        while True:
            placed = False
            for sr, sc in shape_cells:
                nr = min_r + sr + k * step_r
                nc = min_c + sc + k * step_c
                if 0 <= nr < rows and 0 <= nc < cols:
                    output[nr][nc] = arrow_color
                    placed = True
            if not placed:
                break
            k += 1

    return output
"""

# ============================================================
# 06df4c85 - Grid cells with colored blocks that fill between same-color cells
# ============================================================
solutions["06df4c85"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]

    grid_color = None
    grid_rows_set = set()
    for r in range(rows):
        vals = set(grid[r])
        if len(vals) == 1 and 0 not in vals:
            grid_rows_set.add(r)
            grid_color = grid[r][0]

    if grid_color is None:
        return output

    non_grid_rows = [r for r in range(rows) if r not in grid_rows_set]

    grid_cols_set = set()
    for c in range(cols):
        if all(grid[r][c] == grid_color for r in non_grid_rows):
            grid_cols_set.add(c)

    def get_bands(indices, total):
        non_div = sorted(set(range(total)) - indices)
        bands = []
        if not non_div:
            return bands
        start = non_div[0]
        for i in range(1, len(non_div)):
            if non_div[i] != non_div[i-1] + 1:
                bands.append((start, non_div[i-1]))
                start = non_div[i]
        bands.append((start, non_div[-1]))
        return bands

    row_bands = get_bands(grid_rows_set, rows)
    col_bands = get_bands(grid_cols_set, cols)

    cell_colors = {}

    for ci, (rs, re) in enumerate(row_bands):
        for cj, (cs, ce) in enumerate(col_bands):
            colors = set()
            for r in range(rs, re+1):
                for c in range(cs, ce+1):
                    v = grid[r][c]
                    if v != grid_color and v != 0:
                        colors.add(v)
            if colors:
                cell_colors[(ci, cj)] = list(colors)[0]

    from collections import defaultdict
    color_cells = defaultdict(list)
    for (ci, cj), color in cell_colors.items():
        color_cells[color].append((ci, cj))

    filled_cells = dict(cell_colors)

    for color, cells in color_cells.items():
        row_groups = defaultdict(list)
        col_groups = defaultdict(list)
        for ci, cj in cells:
            row_groups[ci].append(cj)
            col_groups[cj].append(ci)

        for ci, cj_list in row_groups.items():
            if len(cj_list) >= 2:
                min_cj = min(cj_list)
                max_cj = max(cj_list)
                for cj in range(min_cj, max_cj + 1):
                    if (ci, cj) not in filled_cells:
                        filled_cells[(ci, cj)] = color

        for cj, ci_list in col_groups.items():
            if len(ci_list) >= 2:
                min_ci = min(ci_list)
                max_ci = max(ci_list)
                for ci in range(min_ci, max_ci + 1):
                    if (ci, cj) not in filled_cells:
                        filled_cells[(ci, cj)] = color

    for (ci, cj), color in filled_cells.items():
        if (ci, cj) in cell_colors:
            continue
        rs, re = row_bands[ci]
        cs, ce = col_bands[cj]

        src = None
        for (sci, scj), sc in cell_colors.items():
            if sc == color:
                src = (sci, scj)
                break
        if src is None:
            continue
        src_rs, src_re = row_bands[src[0]]
        src_cs, src_ce = col_bands[src[1]]

        for dr in range(re - rs + 1):
            for dc in range(ce - cs + 1):
                sr = src_rs + dr
                sc2 = src_cs + dc
                if sr <= src_re and sc2 <= src_ce:
                    if grid[sr][sc2] != grid_color and grid[sr][sc2] != 0:
                        output[rs + dr][cs + dc] = color

    return output
"""

# ============================================================
# Verify all solutions
# ============================================================
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
with open("data/arc_python_solutions_b25.json", "w") as f:
    json.dump(passing, f, indent=2)
print(f"\nSaved {len(passing)} solutions")
