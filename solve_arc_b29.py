import json

solutions = {}

# === 6cdd2623 ===
solutions["6cdd2623"] = r"""def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    color_positions = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != 0:
                color_positions.setdefault(v, []).append((r, c))
    cross_color = min(color_positions, key=lambda c: len(color_positions[c]))
    positions_set = set(color_positions[cross_color])
    h_rows = set()
    for r in range(rows):
        if (r, 0) in positions_set and (r, cols-1) in positions_set:
            h_rows.add(r)
    v_cols = set()
    for c in range(cols):
        if (0, c) in positions_set and (rows-1, c) in positions_set:
            v_cols.add(c)
    output = [[0]*cols for _ in range(rows)]
    for r in h_rows:
        for c in range(cols):
            output[r][c] = cross_color
    for c in v_cols:
        for r in range(rows):
            output[r][c] = cross_color
    return output
"""

# === 6cf79266 ===
solutions["6cf79266"] = r"""def solve(grid):
    import copy
    rows = len(grid)
    cols = len(grid[0])
    output = copy.deepcopy(grid)

    # Find all NxN all-zero blocks for N from large to small
    # Find the right block size and fill all non-overlapping instances
    for N in range(min(rows, cols), 1, -1):
        blocks = []
        for r1 in range(rows - N + 1):
            for c1 in range(cols - N + 1):
                all_zero = True
                for r in range(r1, r1+N):
                    for c in range(c1, c1+N):
                        if grid[r][c] != 0:
                            all_zero = False
                            break
                    if not all_zero:
                        break
                if all_zero:
                    blocks.append((r1, c1))
        if blocks:
            # Group overlapping blocks and take one representative per group
            def overlaps(b1, b2):
                r1, c1 = b1
                r2, c2 = b2
                return not (r1+N <= r2 or r2+N <= r1 or c1+N <= c2 or c2+N <= c1)
            groups = []
            for b in blocks:
                merged = False
                for g in groups:
                    if any(overlaps(b, gb) for gb in g):
                        g.append(b)
                        merged = True
                        break
                if not merged:
                    groups.append([b])
            reps = []
            for g in groups:
                g.sort()
                reps.append(g[0])
            for r1, c1 in reps:
                for r in range(r1, r1+N):
                    for c in range(c1, c1+N):
                        output[r][c] = 1
            return output
    return output
"""

# === 6ecd11f4 ===
solutions["6ecd11f4"] = r"""def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    from collections import Counter

    counts = Counter(v for r in grid for v in r if v != 0)
    block_color = counts.most_common(1)[0][0]

    # Find all non-zero cells and connected components
    nonzero = set((r,c) for r in range(rows) for c in range(cols) if grid[r][c] != 0)
    visited = set()
    components = []
    for r,c in nonzero:
        if (r,c) in visited:
            continue
        comp = set()
        q = [(r,c)]
        while q:
            cr,cc = q.pop(0)
            if (cr,cc) in visited:
                continue
            visited.add((cr,cc))
            comp.add((cr,cc))
            for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr,nc = cr+dr, cc+dc
                if (nr,nc) in nonzero and (nr,nc) not in visited:
                    q.append((nr,nc))
        components.append(comp)

    # Find legend component: small component with multiple non-block colors
    components.sort(key=len, reverse=True)
    legend_comp = None
    for comp in components:
        colors = set(grid[r][c] for r,c in comp)
        non_block = colors - {block_color}
        if len(non_block) >= 2 and len(comp) < 50:
            legend_comp = comp
            break
    if legend_comp is None:
        for comp in components:
            colors = set(grid[r][c] for r,c in comp)
            non_block = colors - {block_color}
            if len(non_block) >= 1 and len(comp) < 50:
                legend_comp = comp
                break

    if legend_comp is None:
        return [[0]]

    lr = [r for r,c in legend_comp]
    lc = [c for r,c in legend_comp]
    lr1, lr2 = min(lr), max(lr)
    lc1, lc2 = min(lc), max(lc)
    legend_rows = lr2 - lr1 + 1
    legend_cols = lc2 - lc1 + 1

    legend = []
    for r in range(lr1, lr2+1):
        row = []
        for c in range(lc1, lc2+1):
            row.append(grid[r][c])
        legend.append(row)

    # Block cells: all block_color cells NOT in legend area
    block_cells = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == block_color and not (lr1 <= r <= lr2 and lc1 <= c <= lc2):
                block_cells.add((r,c))

    if not block_cells:
        return [[0]*legend_cols]*legend_rows

    all_block_r = [r for r,c in block_cells]
    all_block_c = [c for r,c in block_cells]
    br1, br2 = min(all_block_r), max(all_block_r)
    bc1, bc2 = min(all_block_c), max(all_block_c)
    total_h = br2 - br1 + 1
    total_w = bc2 - bc1 + 1

    presence = []
    for bi in range(legend_rows):
        row = []
        for bj in range(legend_cols):
            cr = br1 + int((bi + 0.5) * total_h / legend_rows)
            cc = bc1 + int((bj + 0.5) * total_w / legend_cols)
            row.append((cr, cc) in block_cells)
        presence.append(row)

    output = []
    for i in range(legend_rows):
        row = []
        for j in range(legend_cols):
            if presence[i][j]:
                row.append(legend[i][j])
            else:
                row.append(0)
        output.append(row)
    return output
"""

# === 72322fa7 ===
solutions["72322fa7"] = r"""def solve(grid):
    import copy
    rows = len(grid)
    cols = len(grid[0])
    output = copy.deepcopy(grid)

    all_nz = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                all_nz.add((r,c))

    # Use 8-connectivity for finding components
    visited = set()
    components = []
    for r,c in all_nz:
        if (r,c) in visited:
            continue
        comp = set()
        queue = [(r,c)]
        while queue:
            cr,cc = queue.pop(0)
            if (cr,cc) in visited:
                continue
            visited.add((cr,cc))
            comp.add((cr,cc))
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr,nc = cr+dr, cc+dc
                    if (nr,nc) in all_nz and (nr,nc) not in visited:
                        queue.append((nr,nc))
        components.append(comp)

    # Templates: multi-color components
    templates = []
    other_comps = []
    for comp in components:
        colors = set(grid[r][c] for r,c in comp)
        if len(colors) >= 2:
            templates.append(comp)
        else:
            other_comps.append(comp)

    # Extract template patterns
    template_patterns = []
    for tmpl in templates:
        colors_count = {}
        for r,c in tmpl:
            v = grid[r][c]
            colors_count[v] = colors_count.get(v, 0) + 1

        center_color = min(colors_count, key=lambda c: colors_count[c])
        surround_color = [c for c in colors_count if c != center_color][0]

        center_positions = [(r,c) for r,c in tmpl if grid[r][c] == center_color]
        cr, cc = center_positions[0]

        offsets = []
        for r,c in tmpl:
            offsets.append((r-cr, c-cc, grid[r][c]))

        template_patterns.append({
            'center_color': center_color,
            'surround_color': surround_color,
            'offsets': offsets,
        })

    # For each non-template component, try to match and complete
    for comp in other_comps:
        comp_color = grid[list(comp)[0][0]][list(comp)[0][1]]
        comp_list = list(comp)

        for tp in template_patterns:
            if comp_color == tp['center_color'] and len(comp) == 1:
                # Single center cell - place full template
                r, c = comp_list[0]
                for dr, dc, v in tp['offsets']:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if output[nr][nc] == 0:
                            output[nr][nc] = v
                break
            elif comp_color == tp['surround_color']:
                # Multiple surround cells - find center and complete
                surround_offsets = [(dr,dc) for dr,dc,v in tp['offsets'] if v == tp['surround_color']]

                # For each possible alignment: try placing center at different positions
                for ref_r, ref_c in comp_list:
                    for s_dr, s_dc in surround_offsets:
                        # If ref cell is at offset (s_dr, s_dc) from center
                        potential_cr = ref_r - s_dr
                        potential_cc = ref_c - s_dc

                        # Check if all surround offsets match existing cells
                        match = True
                        for s_dr2, s_dc2 in surround_offsets:
                            nr = potential_cr + s_dr2
                            nc = potential_cc + s_dc2
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if grid[nr][nc] != tp['surround_color']:
                                    match = False
                                    break
                            else:
                                match = False
                                break

                        if match:
                            # Complete the pattern
                            for dr2, dc2, v2 in tp['offsets']:
                                nr = potential_cr + dr2
                                nc = potential_cc + dc2
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    if output[nr][nc] == 0:
                                        output[nr][nc] = v2
                            break
                    else:
                        continue
                    break
                break

    return output
"""

# === 776ffc46 ===
solutions["776ffc46"] = r"""def solve(grid):
    import copy
    rows = len(grid)
    cols = len(grid[0])
    output = copy.deepcopy(grid)

    five_cells = set((r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 5)
    visited = set()
    boxes = []
    for r,c in five_cells:
        if (r,c) in visited:
            continue
        comp = set()
        queue = [(r,c)]
        while queue:
            cr,cc = queue.pop(0)
            if (cr,cc) in visited:
                continue
            visited.add((cr,cc))
            comp.add((cr,cc))
            for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr,nc = cr+dr,cc+dc
                if (nr,nc) in five_cells and (nr,nc) not in visited:
                    queue.append((nr,nc))
        boxes.append(comp)

    template_shape = None
    template_color = None

    for box in boxes:
        br1 = min(r for r,c in box)
        br2 = max(r for r,c in box)
        bc1 = min(c for r,c in box)
        bc2 = max(c for r,c in box)

        inner_cells = []
        for r in range(br1+1, br2):
            for c in range(bc1+1, bc2):
                if grid[r][c] != 0 and grid[r][c] != 5:
                    inner_cells.append((r, c))

        if not inner_cells:
            continue

        ir1 = min(r for r,c in inner_cells)
        ir2 = max(r for r,c in inner_cells)
        ic1 = min(c for r,c in inner_cells)
        ic2 = max(c for r,c in inner_cells)

        top_pad = ir1 - (br1 + 1)
        bot_pad = (br2 - 1) - ir2
        left_pad = ic1 - (bc1 + 1)
        right_pad = (bc2 - 1) - ic2

        if top_pad > 0 and bot_pad > 0 and left_pad > 0 and right_pad > 0:
            min_r = min(r for r,c in inner_cells)
            min_c = min(c for r,c in inner_cells)
            template_shape = set((r - min_r, c - min_c) for r,c in inner_cells)
            template_color = grid[inner_cells[0][0]][inner_cells[0][1]]
            break

    if template_shape is None:
        return output

    box_regions = set()
    for box in boxes:
        br1 = min(r for r,c in box)
        br2 = max(r for r,c in box)
        bc1 = min(c for r,c in box)
        bc2 = max(c for r,c in box)
        for r in range(br1, br2+1):
            for c in range(bc1, bc2+1):
                box_regions.add((r,c))

    free_cells = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and grid[r][c] != 5 and (r,c) not in box_regions:
                free_cells.add((r,c))

    vis = set()
    for r,c in free_cells:
        if (r,c) in vis:
            continue
        comp = set()
        queue = [(r,c)]
        while queue:
            cr,cc = queue.pop(0)
            if (cr,cc) in vis:
                continue
            vis.add((cr,cc))
            comp.add((cr,cc))
            for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr,nc = cr+dr, cc+dc
                if (nr,nc) in free_cells and (nr,nc) not in vis:
                    queue.append((nr,nc))

        min_r = min(r for r,c in comp)
        min_c = min(c for r,c in comp)
        comp_norm = set((r-min_r, c-min_c) for r,c in comp)

        if comp_norm == template_shape:
            for r,c in comp:
                output[r][c] = template_color

    return output
"""

# === 780d0b14 ===
solutions["780d0b14"] = r"""def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    from collections import Counter

    sep_rows = set(r for r in range(rows) if all(grid[r][c] == 0 for c in range(cols)))
    sep_cols = set(c for c in range(cols) if all(grid[r][c] == 0 for r in range(rows)))

    row_ranges = []
    start = None
    for r in range(rows):
        if r not in sep_rows:
            if start is None:
                start = r
        else:
            if start is not None:
                row_ranges.append((start, r))
                start = None
    if start is not None:
        row_ranges.append((start, rows))

    col_ranges = []
    start = None
    for c in range(cols):
        if c not in sep_cols:
            if start is None:
                start = c
        else:
            if start is not None:
                col_ranges.append((start, c))
                start = None
    if start is not None:
        col_ranges.append((start, cols))

    output = []
    for r1, r2 in row_ranges:
        row = []
        for c1, c2 in col_ranges:
            counts = Counter()
            for r in range(r1, r2):
                for c in range(c1, c2):
                    v = grid[r][c]
                    if v != 0:
                        counts[v] += 1
            if counts:
                row.append(counts.most_common(1)[0][0])
            else:
                row.append(0)
        output.append(row)
    return output
"""

# === 7b6016b9 ===
solutions["7b6016b9"] = r"""def solve(grid):
    import copy
    rows = len(grid)
    cols = len(grid[0])
    output = copy.deepcopy(grid)

    outside = set()
    queue = []
    for r in range(rows):
        for c in [0, cols-1]:
            if grid[r][c] == 0 and (r,c) not in outside:
                outside.add((r,c))
                queue.append((r,c))
    for c in range(cols):
        for r in [0, rows-1]:
            if grid[r][c] == 0 and (r,c) not in outside:
                outside.add((r,c))
                queue.append((r,c))

    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr,nc) not in outside and grid[nr][nc] == 0:
                outside.add((nr,nc))
                queue.append((nr,nc))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                output[r][c] = 3 if (r,c) in outside else 2
    return output
"""

# === 7837ac64 ===
solutions["7837ac64"] = r"""def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    from collections import Counter

    counts = Counter(v for r in grid for v in r)
    bg = counts.most_common(1)[0][0]

    if bg == 0:
        tile_color = max((c for c in counts if c != 0), key=lambda c: counts[c])
    else:
        tile_color = bg

    # Find grid line rows and columns
    full_rows = [r for r in range(rows) if all(grid[r][c] == tile_color for c in range(cols))]
    special_rows = [r for r in range(rows) if any(grid[r][c] != tile_color and grid[r][c] != 0 for c in range(cols))]
    all_grid_rows = sorted(set(full_rows + special_rows))

    full_cols = [c for c in range(cols) if all(grid[r][c] == tile_color for r in range(rows))]
    special_cols = [c for c in range(cols) if any(grid[r][c] != tile_color and grid[r][c] != 0 for r in range(rows))]
    all_grid_cols = sorted(set(full_cols + special_cols))

    # Get intersection colors (non-tile values at grid line crossings)
    int_grid = {}
    for ri, r in enumerate(all_grid_rows):
        for ci, c in enumerate(all_grid_cols):
            v = grid[r][c]
            if v != tile_color:
                int_grid[(ri, ci)] = v

    if not int_grid:
        return [[0]]

    gis = [gi for gi, gj in int_grid]
    gjs = [gj for gi, gj in int_grid]
    gi_min, gi_max = min(gis), max(gis)
    gj_min, gj_max = min(gjs), max(gjs)

    n_gi = gi_max - gi_min + 1
    n_gj = gj_max - gj_min + 1

    # Build 4x4 (or NxN) sub-grid of special values
    sub = [[0]*n_gj for _ in range(n_gi)]
    for (gi, gj), v in int_grid.items():
        sub[gi - gi_min][gj - gj_min] = v

    # Apply 2x2 sliding window: output (i,j) = value if all 4 cells same, else 0
    out_rows = n_gi - 1
    out_cols = n_gj - 1
    output = [[0]*out_cols for _ in range(out_rows)]
    for i in range(out_rows):
        for j in range(out_cols):
            vals = [sub[i][j], sub[i][j+1], sub[i+1][j], sub[i+1][j+1]]
            if vals[0] == vals[1] == vals[2] == vals[3]:
                output[i][j] = vals[0]
            else:
                output[i][j] = 0
    return output
"""

# Verify all solutions
print("Testing solutions...")
passing = {}
for task_id, code in list(solutions.items()):
    try:
        with open(f"data/arc1/{task_id}.json") as f:
            data = json.load(f)
        exec(code)
        all_pass = True
        for i, pair in enumerate(data["train"]):
            result = solve(pair["input"])
            if result != pair["output"]:
                print(f"FAIL: {task_id} train {i}")
                for r in range(min(len(result), len(pair['output']))):
                    for c in range(min(len(result[0]), len(pair['output'][0]))):
                        if result[r][c] != pair['output'][r][c]:
                            print(f"  First diff at ({r},{c}): got {result[r][c]}, expected {pair['output'][r][c]}")
                            break
                    else:
                        continue
                    break
                if len(result) != len(pair['output']) or (len(result) > 0 and len(result[0]) != len(pair['output'][0])):
                    print(f"  Size mismatch: {len(result)}x{len(result[0]) if result else 0} vs {len(pair['output'])}x{len(pair['output'][0])}")
                all_pass = False
        if all_pass:
            print(f"PASS: {task_id}")
            passing[task_id] = code
    except Exception as e:
        print(f"ERROR: {task_id}: {e}")
        import traceback
        traceback.print_exc()

# Save only passing solutions
with open("data/arc_python_solutions_b29.json", "w") as f:
    json.dump(passing, f, indent=2)
print(f"\nSaved {len(passing)} solutions")
