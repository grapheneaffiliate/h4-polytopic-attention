import json

solutions = {}

# Task 0e206a2e: Template shapes with scattered anchor points
# The input has template shapes (connected components with fill color) and
# scattered isolated colored points. Each template has a fill color and anchor colors.
# The scattered points define new positions for the anchor colors.
# Find the rotation/reflection that maps template anchors to scattered anchors,
# then draw the template at the new position.
solutions["0e206a2e"] = """def solve(grid):
    from collections import deque
    rows = len(grid)
    cols = len(grid[0])

    visited = [[False]*cols for _ in range(rows)]
    components = []

    def bfs(sr, sc):
        q = deque([(sr, sc)])
        visited[sr][sc] = True
        cells = [(sr, sc, grid[sr][sc])]
        while q:
            r, c = q.popleft()
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != 0:
                        visited[nr][nc] = True
                        q.append((nr, nc))
                        cells.append((nr, nc, grid[nr][nc]))
        return cells

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                comp = bfs(r, c)
                components.append(comp)

    templates = []
    scattered_points = []

    for comp in components:
        colors = {}
        for r, c, v in comp:
            colors[v] = colors.get(v, 0) + 1
        max_count = max(colors.values())
        if max_count >= 3:
            templates.append(comp)
        else:
            for r, c, v in comp:
                scattered_points.append((r, c, v))

    output = [[0]*cols for _ in range(rows)]
    used_scattered = set()

    for td_cells in templates:
        colors = {}
        for r, c, v in td_cells:
            colors[v] = colors.get(v, 0) + 1
        fill_color = max(colors, key=colors.get)
        anchor_colors = {v for v in colors if v != fill_color}

        cp = {}
        for r, c, v in td_cells:
            if v not in cp:
                cp[v] = []
            cp[v].append((r, c))

        found_match = False
        for center_color in sorted(anchor_colors):
            if found_match:
                break
            if len(cp.get(center_color, [])) != 1:
                continue
            cr, cc = cp[center_color][0]

            rel = {}
            for r, c, v in td_cells:
                rel[(r-cr, c-cc)] = v

            other_anchors = {}
            for (dr, dc), v in rel.items():
                if v != fill_color and v != center_color:
                    if v not in other_anchors:
                        other_anchors[v] = []
                    other_anchors[v].append((dr, dc))

            for idx, (iso_r, iso_c, iso_v) in enumerate(scattered_points):
                if iso_v != center_color or idx in used_scattered:
                    continue

                transforms = [
                    lambda r, c: (r, c),
                    lambda r, c: (r, -c),
                    lambda r, c: (-r, c),
                    lambda r, c: (-r, -c),
                    lambda r, c: (c, r),
                    lambda r, c: (c, -r),
                    lambda r, c: (-c, r),
                    lambda r, c: (-c, -r),
                ]

                for tfn in transforms:
                    match = True
                    matched_indices = {idx}

                    for v, positions in other_anchors.items():
                        for (dr, dc) in positions:
                            tr, tc = tfn(dr, dc)
                            ar, ac = tr + iso_r, tc + iso_c
                            found = False
                            for idx2, (ir, ic, iv) in enumerate(scattered_points):
                                if idx2 not in used_scattered and idx2 not in matched_indices and ir == ar and ic == ac and iv == v:
                                    found = True
                                    matched_indices.add(idx2)
                                    break
                            if not found:
                                match = False
                                break
                        if not match:
                            break

                    if match:
                        for (dr, dc), v in rel.items():
                            tr, tc = tfn(dr, dc)
                            ar, ac = tr + iso_r, tc + iso_c
                            if 0 <= ar < rows and 0 <= ac < cols:
                                output[ar][ac] = v
                        used_scattered |= matched_indices
                        found_match = True
                        break

                if found_match:
                    break

    return output"""

# Task 264363fd: Stamp pattern on rectangles
# The input has a background color, rectangular regions of a fill color with markers,
# and a small "stamp" pattern. The stamp is centered on each marker inside rectangles.
# Cross lines extend through the marker along the rectangle using the stamp's arm color.
solutions["264363fd"] = """def solve(grid):
    from collections import Counter, deque
    rows = len(grid)
    cols = len(grid[0])

    all_colors = Counter()
    for row in grid:
        for v in row:
            all_colors[v] += 1
    bg = all_colors.most_common(1)[0][0]

    visited = [[False]*cols for _ in range(rows)]
    components = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                q = deque([(r,c)])
                visited[r][c] = True
                comp = [(r,c)]
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] != bg:
                            visited[nr][nc] = True
                            q.append((nr,nc))
                            comp.append((nr,nc))
                components.append(comp)

    rectangles = []
    stamp_comp = None

    for comp in components:
        colors = Counter(grid[r][c] for r,c in comp)
        fill = colors.most_common(1)[0][0]
        fill_count = colors.most_common(1)[0][1]

        if fill_count / len(comp) < 0.95 and len(comp) < 50:
            stamp_comp = comp
        else:
            marker_cells = [(r,c) for r,c in comp if grid[r][c] != fill]
            all_r = [r for r,c in comp]
            all_c = [c for r,c in comp]
            rect_bounds = (min(all_r), min(all_c), max(all_r), max(all_c))
            rectangles.append({
                'fill': fill,
                'markers': marker_cells,
                'bounds': rect_bounds,
                'cells': set((r,c) for r,c in comp)
            })

    if stamp_comp is None:
        return grid

    sr = [r for r,c in stamp_comp]
    sc = [c for r,c in stamp_comp]
    center_r = (min(sr) + max(sr)) // 2
    center_c = (min(sc) + max(sc)) // 2

    stamp_pattern = {}
    for r,c in stamp_comp:
        stamp_pattern[(r - center_r, c - center_c)] = grid[r][c]

    stamp_center_color = grid[center_r][center_c]

    v_arm_color = None
    v_arm_extent = 0
    for (dr, dc), v in stamp_pattern.items():
        if dc == 0 and v != stamp_center_color:
            v_arm_color = v
            v_arm_extent = max(v_arm_extent, abs(dr))

    h_arm_color = None
    h_arm_extent = 0
    for (dr, dc), v in stamp_pattern.items():
        if dr == 0 and v != stamp_center_color:
            h_arm_color = v
            h_arm_extent = max(h_arm_extent, abs(dc))

    output = [row[:] for row in grid]

    for r,c in stamp_comp:
        output[r][c] = bg

    for rect in rectangles:
        r_min, c_min, r_max, c_max = rect['bounds']

        for mr, mc in rect['markers']:
            if v_arm_color is not None and v_arm_extent >= 2:
                for r in range(r_min, r_max + 1):
                    if (r, mc) in rect['cells']:
                        output[r][mc] = v_arm_color

            if h_arm_color is not None and h_arm_extent >= 2:
                for c in range(c_min, c_max + 1):
                    if (mr, c) in rect['cells']:
                        output[mr][c] = h_arm_color

            for (dr, dc), v in stamp_pattern.items():
                nr, nc = mr + dr, mc + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) in rect['cells']:
                    output[nr][nc] = v

    return output"""

# Task 5c2c9af4: Concentric rectangles from 3 collinear points
# The 3 input points define a center (middle point) and step distance.
# Concentric rectangles are drawn centered on the center point,
# at distances step*k for k=1,2,3,... until off-grid.
solutions["5c2c9af4"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    points = []
    color = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                points.append((r, c))
                color = grid[r][c]

    points.sort()
    center_r, center_c = points[1]
    step_r = points[1][0] - points[0][0]
    step_c = points[1][1] - points[0][1]
    step = max(abs(step_r), abs(step_c))

    output = [[0]*cols for _ in range(rows)]
    output[center_r][center_c] = color

    for k in range(1, 50):
        top = center_r - step * k
        bot = center_r + step * k
        left = center_c - step * k
        right = center_c + step * k

        if bot < 0 or top >= rows:
            if left >= cols or right < 0:
                break

        if 0 <= top < rows:
            for c in range(max(0, left), min(cols, right + 1)):
                output[top][c] = color
        if 0 <= bot < rows:
            for c in range(max(0, left), min(cols, right + 1)):
                output[bot][c] = color
        if 0 <= left < cols:
            for r in range(max(0, top), min(rows, bot + 1)):
                output[r][left] = color
        if 0 <= right < cols:
            for r in range(max(0, top), min(rows, bot + 1)):
                output[r][right] = color

    return output"""

# Verify all solutions
for task_id, code in solutions.items():
    with open(f"data/arc1/{task_id}.json") as f:
        task_data = json.load(f)
    exec(code)
    all_pass = True
    for i, pair in enumerate(task_data["train"]):
        result = solve(pair["input"])
        if result != pair["output"]:
            print(f"FAIL: {task_id} train {i}")
            all_pass = False
    if all_pass:
        print(f"PASS: {task_id}")

with open("data/arc_python_solutions_retry_a.json", "w") as f:
    json.dump(solutions, f, indent=2)
print(f"Saved {len(solutions)} solutions")
