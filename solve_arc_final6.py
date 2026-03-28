import json
from collections import Counter, defaultdict

solutions = {}

# ============================================================
# 5daaa586: Grid with separator lines forming bordered section.
# Scattered cells "waterfall" toward the matching border.
# ============================================================
solutions["5daaa586"] = """
def solve(grid):
    rows, cols = len(grid), len(grid[0])
    from collections import Counter

    h_seps = []
    for r in range(rows):
        counts = Counter(grid[r][c] for c in range(cols) if grid[r][c] != 0)
        if counts:
            color, count = counts.most_common(1)[0]
            if count >= cols * 0.7:
                h_seps.append((r, color))

    v_seps = []
    for c in range(cols):
        counts = Counter(grid[r][c] for r in range(rows) if grid[r][c] != 0)
        if counts:
            color, count = counts.most_common(1)[0]
            if count >= rows * 0.7:
                v_seps.append((c, color))

    h_seps.sort()
    v_seps.sort()

    r1, h1_color = h_seps[0]
    r2, h2_color = h_seps[1]
    c1, v1_color = v_seps[0]
    c2, v2_color = v_seps[1]

    center = [[grid[r][c] for c in range(c1, c2+1)] for r in range(r1, r2+1)]
    out_rows = len(center)
    out_cols = len(center[0])

    border_dirs = {
        h1_color: 'up',
        h2_color: 'down',
        v1_color: 'left',
        v2_color: 'right'
    }

    scatter_color = None
    gravity_dir = None
    for r in range(1, out_rows-1):
        for c in range(1, out_cols-1):
            v = center[r][c]
            if v != 0 and v in border_dirs:
                scatter_color = v
                gravity_dir = border_dirs[v]
                break
        if scatter_color:
            break

    result = [row[:] for row in center]

    if gravity_dir == 'down':
        for c in range(1, out_cols-1):
            first_row = None
            for r in range(1, out_rows-1):
                if center[r][c] == scatter_color:
                    if first_row is None:
                        first_row = r
            if first_row is not None:
                for r in range(first_row, out_rows-1):
                    result[r][c] = scatter_color

    elif gravity_dir == 'up':
        for c in range(1, out_cols-1):
            last_row = None
            for r in range(out_rows-2, 0, -1):
                if center[r][c] == scatter_color:
                    if last_row is None:
                        last_row = r
            if last_row is not None:
                for r in range(1, last_row+1):
                    result[r][c] = scatter_color

    elif gravity_dir == 'right':
        for r in range(1, out_rows-1):
            first_col = None
            for c in range(1, out_cols-1):
                if center[r][c] == scatter_color:
                    if first_col is None:
                        first_col = c
            if first_col is not None:
                for c in range(first_col, out_cols-1):
                    result[r][c] = scatter_color

    elif gravity_dir == 'left':
        for r in range(1, out_rows-1):
            last_col = None
            for c in range(out_cols-2, 0, -1):
                if center[r][c] == scatter_color:
                    if last_col is None:
                        last_col = c
            if last_col is not None:
                for c in range(1, last_col+1):
                    result[r][c] = scatter_color

    return result
"""

# ============================================================
# 50846271: Crosses made of 2s in a 0/5 grid.
# 5s on cross arms become 8. Arms extend to global max arm length.
# ============================================================
solutions["50846271"] = """
def solve(grid):
    grid = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    twos = set((r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 2)
    if not twos:
        return grid

    from collections import defaultdict

    # Cluster 2-positions by proximity
    remaining = set(twos)
    clusters = []
    while remaining:
        start = remaining.pop()
        cluster = {start}
        queue = [start]
        while queue:
            r, c = queue.pop(0)
            for r2, c2 in list(remaining):
                if abs(r-r2) + abs(c-c2) <= 4:
                    cluster.add((r2, c2))
                    remaining.discard((r2, c2))
                    queue.append((r2, c2))
        clusters.append(cluster)

    all_crosses = []

    for cluster in clusters:
        row_twos = defaultdict(set)
        col_twos = defaultdict(set)
        for r, c in cluster:
            row_twos[r].add(c)
            col_twos[c].add(r)

        h_rows = {r: sorted(cs) for r, cs in row_twos.items() if len(cs) >= 2}
        v_cols = {c: sorted(rs) for c, rs in col_twos.items() if len(rs) >= 2}

        found = False
        if h_rows and v_cols:
            best = None
            for r in h_rows:
                for c in v_cols:
                    h_min, h_max = h_rows[r][0], h_rows[r][-1]
                    v_min, v_max = v_cols[c][0], v_cols[c][-1]
                    if h_min <= c <= h_max and v_min <= r <= v_max:
                        left = c - h_min
                        right = h_max - c
                        up = r - v_min
                        down = v_max - r
                        max_arm = max(left, right, up, down)
                        score = len(h_rows[r]) + len(v_cols[c])
                        if best is None or score > best[0]:
                            best = (score, r, c, max_arm)
            if best:
                all_crosses.append((best[1], best[2], best[3]))
                found = True

        if not found:
            if h_rows:
                best_r = max(h_rows, key=lambda r: len(h_rows[r]))
                h_min, h_max = h_rows[best_r][0], h_rows[best_r][-1]
                min_max_arm = float('inf')
                best_c = h_min
                for c in h_rows[best_r]:
                    arm = max(c - h_min, h_max - c)
                    if arm < min_max_arm:
                        min_max_arm = arm
                        best_c = c
                all_crosses.append((best_r, best_c, min_max_arm))
            elif v_cols:
                best_c = max(v_cols, key=lambda c: len(v_cols[c]))
                v_min, v_max = v_cols[best_c][0], v_cols[best_c][-1]
                min_max_arm = float('inf')
                best_r = v_min
                for r in v_cols[best_c]:
                    arm = max(r - v_min, v_max - r)
                    if arm < min_max_arm:
                        min_max_arm = arm
                        best_r = r
                all_crosses.append((best_r, best_c, min_max_arm))

    global_max = max(arm for _, _, arm in all_crosses) if all_crosses else 0

    for center_r, center_c, _ in all_crosses:
        for c in range(center_c - global_max, center_c + global_max + 1):
            if 0 <= c < cols and grid[center_r][c] == 5:
                grid[center_r][c] = 8
        for r in range(center_r - global_max, center_r + global_max + 1):
            if 0 <= r < rows and grid[r][center_c] == 5:
                grid[r][center_c] = 8

    return grid
"""

# ============================================================
# 6aa20dc0: Template shape with fill color and 2 marker colors.
# External marker blocks at various scales. Fill scaled template.
# ============================================================
solutions["6aa20dc0"] = """
def solve(grid):
    grid = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])
    from collections import Counter, defaultdict

    c = Counter(grid[r][c] for r in range(rows) for c in range(cols))
    bg = c.most_common(1)[0][0]

    visited = set()
    components = []
    for r in range(rows):
        for cc in range(cols):
            if grid[r][cc] != bg and (r, cc) not in visited:
                comp = []
                stack = [(r, cc)]
                while stack:
                    cr, ccc = stack.pop()
                    if (cr, ccc) in visited:
                        continue
                    if 0 <= cr < rows and 0 <= ccc < cols and grid[cr][ccc] != bg:
                        visited.add((cr, ccc))
                        comp.append((cr, ccc))
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                stack.append((cr+dr, ccc+dc))
                components.append(comp)

    template_comp = None
    other_comps = []
    for comp in components:
        colors = set(grid[r][cc] for r, cc in comp)
        if len(colors) >= 2:
            template_comp = comp
        else:
            other_comps.append(comp)

    if template_comp is None:
        return grid

    t_min_r = min(r for r, c in template_comp)
    t_max_r = max(r for r, c in template_comp)
    t_min_c = min(c for r, c in template_comp)
    t_max_c = max(c for r, c in template_comp)
    t_h = t_max_r - t_min_r + 1
    t_w = t_max_c - t_min_c + 1

    template = [[bg]*t_w for _ in range(t_h)]
    for r, cc in template_comp:
        template[r - t_min_r][cc - t_min_c] = grid[r][cc]

    t_colors = Counter()
    for row in template:
        for v in row:
            if v != bg:
                t_colors[v] += 1

    fill_color = t_colors.most_common(1)[0][0]
    marker_colors = [color for color, cnt in t_colors.items() if cnt == 1]
    if len(marker_colors) != 2:
        return grid

    marker_A, marker_B = marker_colors
    pos_A = pos_B = None
    for r in range(t_h):
        for cc in range(t_w):
            if template[r][cc] == marker_A:
                pos_A = (r, cc)
            elif template[r][cc] == marker_B:
                pos_B = (r, cc)

    t_vec = (pos_B[0] - pos_A[0], pos_B[1] - pos_A[1])

    color_comps = defaultdict(list)
    for comp in other_comps:
        color = grid[comp[0][0]][comp[0][1]]
        color_comps[color].append(comp)

    a_comps = color_comps.get(marker_A, [])
    b_comps = color_comps.get(marker_B, [])

    rotations = [
        (1, 0, 0, 1), (0, -1, 1, 0), (-1, 0, 0, -1), (0, 1, -1, 0),
        (1, 0, 0, -1), (-1, 0, 0, 1), (0, 1, 1, 0), (0, -1, -1, 0),
    ]

    for a_comp in a_comps:
        a_h = max(r for r, c in a_comp) - min(r for r, c in a_comp) + 1
        a_w = max(c for r, c in a_comp) - min(c for r, c in a_comp) + 1
        scale = max(a_h, a_w)
        a_min_r = min(r for r, c in a_comp)
        a_min_c = min(c for r, c in a_comp)

        for b_comp in b_comps:
            b_h = max(r for r, c in b_comp) - min(r for r, c in b_comp) + 1
            b_w = max(c for r, c in b_comp) - min(c for r, c in b_comp) + 1
            if max(b_h, b_w) != scale:
                continue
            b_min_r = min(r for r, c in b_comp)
            b_min_c = min(c for r, c in b_comp)
            ext_vec = (b_min_r - a_min_r, b_min_c - a_min_c)

            for a_r, b_r, c_r, d_r in rotations:
                rot_r = a_r * t_vec[0] + b_r * t_vec[1]
                rot_c = c_r * t_vec[0] + d_r * t_vec[1]
                if abs(scale * rot_r - ext_vec[0]) < 1.5 and abs(scale * rot_c - ext_vec[1]) < 1.5:
                    for tr in range(t_h):
                        for tc in range(t_w):
                            if template[tr][tc] != fill_color:
                                continue
                            dr_t = tr - pos_A[0]
                            dc_t = tc - pos_A[1]
                            rot_dr = a_r * dr_t + b_r * dc_t
                            rot_dc = c_r * dr_t + d_r * dc_t
                            for sr in range(scale):
                                for sc in range(scale):
                                    gr = a_min_r + rot_dr * scale + sr
                                    gc = a_min_c + rot_dc * scale + sc
                                    if 0 <= gr < rows and 0 <= gc < cols:
                                        if grid[gr][gc] == bg:
                                            grid[gr][gc] = fill_color
                    break

    return grid
"""

# ============================================================
# e5062a87: Binary grid (0/5) with 2-pattern. Stamp the pattern
# at all valid all-0 positions. Use 5-bridge priority, then edge distance.
# ============================================================
solutions["e5062a87"] = """
def solve(grid):
    grid = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    twos = sorted((r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 2)
    anchor = twos[0]
    rel_shape = [(r-anchor[0], c-anchor[1]) for r,c in twos]
    two_set = set(twos)

    placements = []
    for r in range(rows):
        for c in range(cols):
            pos = [(r+dr, c+dc) for dr, dc in rel_shape]
            if all(0 <= pr < rows and 0 <= pc < cols and grid[pr][pc] == 0
                   for pr, pc in pos):
                has_bridge = False
                for pr, pc in pos:
                    for dr2, dc2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = pr+dr2, pc+dc2
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 5:
                            for dr3, dc3 in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr2, nc2 = nr+dr3, nc+dc3
                                if (nr2, nc2) in two_set:
                                    has_bridge = True

                min_edge_dist = min(
                    min(pr, rows-1-pr, pc, cols-1-pc) for pr, pc in pos
                )
                placements.append((min_edge_dist, r, c, pos, has_bridge))

    bridge_placements = [p for p in placements if p[4]]
    if bridge_placements:
        use_placements = sorted(bridge_placements)
    else:
        use_placements = sorted(placements)

    changed = True
    while changed:
        changed = False
        for _, r, c, pos, _ in use_placements:
            if all(grid[pr][pc] == 0 for pr, pc in pos):
                for pr, pc in pos:
                    grid[pr][pc] = 2
                changed = True

    return grid
"""

# ============================================================
# Verify and save
# ============================================================
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

with open("data/arc_python_solutions_final6.json", "w") as f:
    json.dump(solutions, f, indent=2)
print(f"Saved {len(solutions)} solutions")
