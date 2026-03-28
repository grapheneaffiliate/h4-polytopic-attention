import json

solutions = {}

# 272f95fa: Grid divided by 8-lines into 3x3 sections, fill with specific colors
solutions["272f95fa"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    h_lines = []
    v_lines = []
    for r in range(rows):
        if all(grid[r][c] == 8 for c in range(cols)):
            h_lines.append(r)
    for c in range(cols):
        if all(grid[r][c] == 8 for r in range(rows)):
            v_lines.append(c)
    row_bands = [(0, h_lines[0]-1), (h_lines[0]+1, h_lines[1]-1), (h_lines[1]+1, rows-1)]
    col_bands = [(0, v_lines[0]-1), (v_lines[0]+1, v_lines[1]-1), (v_lines[1]+1, cols-1)]
    color_map = {
        (0,0): 0, (0,1): 2, (0,2): 0,
        (1,0): 4, (1,1): 6, (1,2): 3,
        (2,0): 0, (2,1): 1, (2,2): 0,
    }
    for ri, (r1, r2) in enumerate(row_bands):
        for ci, (c1, c2) in enumerate(col_bands):
            color = color_map[(ri, ci)]
            for r in range(r1, r2+1):
                for c in range(c1, c2+1):
                    if grid[r][c] == 0:
                        output[r][c] = color
    return output"""

# 29ec7d0e: Repeating tiled pattern with 0s as holes, fill in the 0s
solutions["29ec7d0e"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    for rp in range(1, rows+1):
        for cp in range(1, cols+1):
            valid = True
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != 0:
                        for r2 in range(rows):
                            for c2 in range(cols):
                                if grid[r2][c2] != 0 and r % rp == r2 % rp and c % cp == c2 % cp:
                                    if grid[r][c] != grid[r2][c2]:
                                        valid = False
                                        break
                            if not valid:
                                break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                tile = {}
                for r in range(rows):
                    for c in range(cols):
                        if grid[r][c] != 0:
                            tile[(r % rp, c % cp)] = grid[r][c]
                if len(tile) == rp * cp:
                    for r in range(rows):
                        for c in range(cols):
                            key = (r % rp, c % cp)
                            if key in tile:
                                output[r][c] = tile[key]
                    return output
    return output"""

# 32597951: 8-bordered region, replace non-0 non-8 cells inside with 3
solutions["32597951"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    eight_cells = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8:
                eight_cells.add((r, c))
    if not eight_cells:
        return output
    rs = [c[0] for c in eight_cells]
    cs = [c[1] for c in eight_cells]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    for r in range(rmin, rmax + 1):
        for c in range(cmin, cmax + 1):
            if grid[r][c] not in (0, 8):
                output[r][c] = 3
    return output"""

# 3345333e: Shape with rectangular overlay, remove overlay and restore shape symmetry
solutions["3345333e"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    color_cells = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != 0:
                if v not in color_cells:
                    color_cells[v] = []
                color_cells[v].append((r, c))
    if len(color_cells) < 2:
        return output
    def is_rectangle(cells):
        if not cells:
            return False
        rs = [c[0] for c in cells]
        cs = [c[1] for c in cells]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)
        return len(cells) == (rmax - rmin + 1) * (cmax - cmin + 1)
    overlay_color = None
    shape_color = None
    for color, cells in color_cells.items():
        if is_rectangle(cells):
            overlay_color = color
        else:
            shape_color = color
    if overlay_color is None or shape_color is None:
        return output
    overlay_cells = set(color_cells[overlay_color])
    shape_cells = set(color_cells[shape_color])
    for r, c in overlay_cells:
        output[r][c] = 0
    shape_cs = [c[1] for c in shape_cells]
    shape_rs = [c[0] for c in shape_cells]
    c_center = (min(shape_cs) + max(shape_cs)) / 2.0
    r_center = (min(shape_rs) + max(shape_rs)) / 2.0
    lr_valid = True
    for r, c in shape_cells:
        mirror_c = round(2 * c_center - c)
        if 0 <= mirror_c < cols:
            if (r, mirror_c) not in shape_cells and (r, mirror_c) not in overlay_cells:
                lr_valid = False
                break
    tb_valid = True
    for r, c in shape_cells:
        mirror_r = round(2 * r_center - r)
        if 0 <= mirror_r < rows:
            if (mirror_r, c) not in shape_cells and (mirror_r, c) not in overlay_cells:
                tb_valid = False
                break
    if lr_valid:
        for r, c in shape_cells:
            mirror_c = round(2 * c_center - c)
            if 0 <= mirror_c < cols and (r, mirror_c) in overlay_cells:
                output[r][mirror_c] = shape_color
    if tb_valid:
        for r, c in shape_cells:
            mirror_r = round(2 * r_center - r)
            if 0 <= mirror_r < rows and (mirror_r, c) in overlay_cells:
                output[mirror_r][c] = shape_color
    if lr_valid and tb_valid:
        for r, c in shape_cells:
            mirror_r = round(2 * r_center - r)
            mirror_c = round(2 * c_center - c)
            if 0 <= mirror_r < rows and 0 <= mirror_c < cols and (mirror_r, mirror_c) in overlay_cells:
                output[mirror_r][mirror_c] = shape_color
    return output"""

# 36fdfd69: Clusters of 2-cells, fill bbox of each cluster with 4
solutions["36fdfd69"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    two_cells = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                two_cells.add((r, c))
    parent = {}
    for cell in two_cells:
        parent[cell] = cell
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b
    two_list = list(two_cells)
    for i in range(len(two_list)):
        for j in range(i+1, len(two_list)):
            r1, c1 = two_list[i]
            r2, c2 = two_list[j]
            dr = abs(r1 - r2)
            dc = abs(c1 - c2)
            if max(dr, dc) <= 2:
                rmin, rmax = min(r1, r2), max(r1, r2)
                cmin, cmax = min(c1, c2), max(c1, c2)
                all_nonzero = True
                for r in range(rmin, rmax + 1):
                    for c in range(cmin, cmax + 1):
                        if grid[r][c] == 0:
                            all_nonzero = False
                            break
                    if not all_nonzero:
                        break
                if all_nonzero:
                    union(two_list[i], two_list[j])
            elif max(dr, dc) <= 3 and min(dr, dc) <= 1:
                rmin, rmax = min(r1, r2), max(r1, r2)
                cmin, cmax = min(c1, c2), max(c1, c2)
                all_nonzero = True
                for r in range(rmin, rmax + 1):
                    for c in range(cmin, cmax + 1):
                        if grid[r][c] == 0:
                            all_nonzero = False
                            break
                    if not all_nonzero:
                        break
                if all_nonzero:
                    union(two_list[i], two_list[j])
    from collections import defaultdict
    groups = defaultdict(set)
    for cell in two_cells:
        groups[find(cell)].add(cell)
    for cluster in groups.values():
        rs = [c[0] for c in cluster]
        cs = [c[1] for c in cluster]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)
        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                if grid[r][c] != 2:
                    output[r][c] = 4
    return output"""

# 39e1d7f9: Grid divided into cells, stamp pattern applied at dot positions
solutions["39e1d7f9"] = """def solve(grid):
    rows, cols = len(grid), len(grid[0])
    output = [row[:] for row in grid]
    sep_color = None
    for test_color in range(10):
        h_count = sum(1 for r in range(rows) if all(grid[r][c] == test_color for c in range(cols)))
        v_count = sum(1 for c in range(cols) if all(grid[r][c] == test_color for r in range(rows)))
        if h_count >= 2 and v_count >= 2:
            sep_color = test_color
            break
    if sep_color is None:
        return output
    h_lines = [r for r in range(rows) if all(grid[r][c] == sep_color for c in range(cols))]
    v_lines = [c for c in range(cols) if all(grid[r][c] == sep_color for r in range(rows))]
    row_bands = []
    prev = 0
    for hl in h_lines:
        if hl > prev:
            row_bands.append((prev, hl - 1))
        prev = hl + 1
    if prev < rows:
        row_bands.append((prev, rows - 1))
    col_bands = []
    prev = 0
    for vl in v_lines:
        if vl > prev:
            col_bands.append((prev, vl - 1))
        prev = vl + 1
    if prev < cols:
        col_bands.append((prev, cols - 1))
    n_rows = len(row_bands)
    n_cols = len(col_bands)
    cell_colors = {}
    for ri in range(n_rows):
        for ci in range(n_cols):
            r1, r2 = row_bands[ri]
            c1, c2 = col_bands[ci]
            colors = set()
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    if grid[r][c] not in (0, sep_color):
                        colors.add(grid[r][c])
            if colors:
                cell_colors[(ri, ci)] = list(colors)[0]
            else:
                cell_colors[(ri, ci)] = 0
    nonzero_cells = {k: v for k, v in cell_colors.items() if v != 0}
    visited = set()
    components = []
    for cell in nonzero_cells:
        if cell in visited:
            continue
        comp = {}
        queue = [cell]
        while queue:
            cr, cc = queue.pop(0)
            if (cr, cc) in visited:
                continue
            visited.add((cr, cc))
            comp[(cr, cc)] = nonzero_cells[(cr, cc)]
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr + dr, cc + dc
                if (nr, nc) in nonzero_cells and (nr, nc) not in visited:
                    queue.append((nr, nc))
        components.append(comp)
    stamp_comp = max(components, key=len)
    dots = [list(comp.keys())[0] for comp in components if len(comp) == 1]
    if not dots:
        return output
    dot_color = cell_colors[dots[0]]
    stamp_center = None
    for pos, color in stamp_comp.items():
        if color == dot_color:
            stamp_center = pos
            break
    if stamp_center is None:
        return output
    stamp_pattern = {}
    for pos, color in stamp_comp.items():
        dr = pos[0] - stamp_center[0]
        dc = pos[1] - stamp_center[1]
        stamp_pattern[(dr, dc)] = color
    for dot in dots:
        for (dr, dc), color in stamp_pattern.items():
            nr, nc = dot[0] + dr, dot[1] + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols:
                r1, r2 = row_bands[nr]
                c1, c2 = col_bands[nc]
                for r in range(r1, r2 + 1):
                    for c in range(c1, c2 + 1):
                        if grid[r][c] == 0:
                            output[r][c] = color
    return output"""

# 2dd70a9a: Two colored markers connected by L/U path of color 3
solutions["2dd70a9a"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    markers = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in (0, 8):
                color = grid[r][c]
                if color not in markers:
                    markers[color] = []
                markers[color].append((r, c))
    colors = list(markers.keys())
    cells_a = set(markers[colors[0]])
    cells_b = set(markers[colors[1]])
    def get_max_ext(start_r, start_c, dr, dc):
        ext = []
        nr, nc = start_r + dr, start_c + dc
        while 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            ext.append((nr, nc))
            nr += dr
            nc += dc
        return ext
    best_path = None
    best_score = None
    for d_a in [(-1,0),(1,0),(0,-1),(0,1)]:
        d_b = (-d_a[0], -d_a[1])
        for ra, ca in cells_a:
            a_full = get_max_ext(ra, ca, d_a[0], d_a[1])
            if not a_full:
                continue
            for rb, cb in cells_b:
                b_full = get_max_ext(rb, cb, d_b[0], d_b[1])
                if not b_full:
                    continue
                for ai in range(len(a_full)):
                    ap = a_full[ai]
                    for bi in range(len(b_full)):
                        bp = b_full[bi]
                        if d_a[0] == 0:
                            if ap[1] != bp[1]:
                                continue
                            rmin, rmax = min(ap[0], bp[0]), max(ap[0], bp[0])
                            connector = []
                            valid = True
                            for rr in range(rmin + 1, rmax):
                                if grid[rr][ap[1]] == 0:
                                    connector.append((rr, ap[1]))
                                else:
                                    valid = False
                                    break
                        else:
                            if ap[0] != bp[0]:
                                continue
                            cmin, cmax = min(ap[1], bp[1]), max(ap[1], bp[1])
                            connector = []
                            valid = True
                            for cc in range(cmin + 1, cmax):
                                if grid[ap[0]][cc] == 0:
                                    connector.append((ap[0], cc))
                                else:
                                    valid = False
                                    break
                        if not valid:
                            continue
                        connector_len = len(connector)
                        a_len = ai + 1
                        path = set(a_full[:ai+1] + b_full[:bi+1] + connector)
                        total_len = len(path)
                        score = (connector_len, total_len, a_len)
                        if best_score is None or score < best_score:
                            best_path = path
                            best_score = score
    if best_path:
        for r, c in best_path:
            output[r][c] = 3
        return output
    best_score = None
    for d in [(-1,0),(1,0),(0,-1),(0,1)]:
        for ra, ca in cells_a:
            a_full = get_max_ext(ra, ca, d[0], d[1])
            if not a_full:
                continue
            for rb, cb in cells_b:
                b_full = get_max_ext(rb, cb, d[0], d[1])
                if not b_full:
                    continue
                if len(a_full) <= len(b_full):
                    pivot = a_full[-1]
                else:
                    pivot = b_full[-1]
                if d[0] == 0:
                    target_col = pivot[1]
                    if len(a_full) <= len(b_full):
                        a_used = a_full
                        b_used = None
                        for bi, bpt in enumerate(b_full):
                            if bpt[1] == target_col:
                                b_used = b_full[:bi+1]
                                bp = bpt
                                break
                        if b_used is None:
                            continue
                        ap = pivot
                    else:
                        b_used = b_full
                        a_used = None
                        for ai, apt in enumerate(a_full):
                            if apt[1] == target_col:
                                a_used = a_full[:ai+1]
                                ap = apt
                                break
                        if a_used is None:
                            continue
                        bp = pivot
                else:
                    target_row = pivot[0]
                    if len(a_full) <= len(b_full):
                        a_used = a_full
                        b_used = None
                        for bi, bpt in enumerate(b_full):
                            if bpt[0] == target_row:
                                b_used = b_full[:bi+1]
                                bp = bpt
                                break
                        if b_used is None:
                            continue
                        ap = pivot
                    else:
                        b_used = b_full
                        a_used = None
                        for ai, apt in enumerate(a_full):
                            if apt[0] == target_row:
                                a_used = a_full[:ai+1]
                                ap = apt
                                break
                        if a_used is None:
                            continue
                        bp = pivot
                if d[0] == 0:
                    if ap[1] != bp[1]:
                        continue
                    rmin, rmax = min(ap[0], bp[0]), max(ap[0], bp[0])
                    connector = []
                    valid = True
                    for rr in range(rmin + 1, rmax):
                        if grid[rr][ap[1]] == 0:
                            connector.append((rr, ap[1]))
                        else:
                            valid = False
                            break
                else:
                    if ap[0] != bp[0]:
                        continue
                    cmin, cmax = min(ap[1], bp[1]), max(ap[1], bp[1])
                    connector = []
                    valid = True
                    for cc in range(cmin + 1, cmax):
                        if grid[ap[0]][cc] == 0:
                            connector.append((ap[0], cc))
                        else:
                            valid = False
                            break
                if not valid:
                    continue
                connector_len = len(connector)
                path = set(list(a_used) + list(b_used) + connector)
                total_len = len(path)
                score = (connector_len, total_len)
                if best_score is None or score < best_score:
                    best_path = path
                    best_score = score
    if best_path:
        for r, c in best_path:
            output[r][c] = 3
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

# Save only passing solutions
with open("data/arc_python_solutions_b26.json", "w") as f:
    json.dump(passing, f, indent=2)
print(f"\nSaved {len(passing)} solutions")
