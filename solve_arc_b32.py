import json

PY_DIR = "data/arc1"

solutions = {}

solutions["b27ca6d3"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    twos = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                twos.add((r,c))
    visited = set()
    groups = []
    for cell in twos:
        if cell in visited:
            continue
        group = set()
        queue = [cell]
        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            group.add(curr)
            r, c = curr
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if (nr, nc) in twos and (nr, nc) not in visited:
                    queue.append((nr, nc))
        groups.append(group)
    output = [row[:] for row in grid]
    for group in groups:
        if len(group) < 2:
            continue
        min_r = min(r for r,c in group)
        max_r = max(r for r,c in group)
        min_c = min(c for r,c in group)
        max_c = max(c for r,c in group)
        br1 = max(0, min_r - 1)
        br2 = min(rows - 1, max_r + 1)
        bc1 = max(0, min_c - 1)
        bc2 = min(cols - 1, max_c + 1)
        for r in range(br1, br2 + 1):
            for c in range(bc1, bc2 + 1):
                if r == br1 or r == br2 or c == bc1 or c == bc2:
                    if output[r][c] == 0:
                        output[r][c] = 3
    return output"""

solutions["b775ac94"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    nonzero = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                nonzero[(r,c)] = grid[r][c]
    visited = set()
    blobs = []
    for cell in nonzero:
        if cell in visited:
            continue
        blob = set()
        queue = [cell]
        while queue:
            curr = queue.pop()
            if curr in visited:
                continue
            visited.add(curr)
            blob.add(curr)
            r, c = curr
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r+dr, c+dc
                    if (nr,nc) in nonzero and (nr,nc) not in visited:
                        queue.append((nr,nc))
        blobs.append(blob)
    output = [row[:] for row in grid]
    for blob in blobs:
        colors = {}
        for cell in blob:
            c = nonzero[cell]
            if c not in colors:
                colors[c] = set()
            colors[c].add(cell)
        if len(colors) == 1:
            continue
        main_color = max(colors, key=lambda c: len(colors[c]))
        main_cells = colors[main_color]
        min_r = min(r for r,c in main_cells)
        max_r = max(r for r,c in main_cells)
        min_c = min(c for r,c in main_cells)
        max_c = max(c for r,c in main_cells)
        for conn_color, conn_cells in colors.items():
            if conn_color == main_color:
                continue
            for cr, cc in conn_cells:
                right_of = cc > max_c
                left_of = cc < min_c
                below = cr > max_r
                above = cr < min_r
                if right_of and not below and not above:
                    axis_c = max_c + 0.5
                    for mr, mc in main_cells:
                        nc = int(2 * axis_c - mc)
                        if 0 <= nc < cols and output[mr][nc] == 0:
                            output[mr][nc] = conn_color
                elif left_of and not below and not above:
                    axis_c = min_c - 0.5
                    for mr, mc in main_cells:
                        nc = int(2 * axis_c - mc)
                        if 0 <= nc < cols and output[mr][nc] == 0:
                            output[mr][nc] = conn_color
                elif below and not right_of and not left_of:
                    axis_r = max_r + 0.5
                    for mr, mc in main_cells:
                        nr = int(2 * axis_r - mr)
                        if 0 <= nr < rows and output[nr][mc] == 0:
                            output[nr][mc] = conn_color
                elif above and not right_of and not left_of:
                    axis_r = min_r - 0.5
                    for mr, mc in main_cells:
                        nr = int(2 * axis_r - mr)
                        if 0 <= nr < rows and output[nr][mc] == 0:
                            output[nr][mc] = conn_color
                elif right_of and below:
                    pr = max_r + 0.5
                    pc = max_c + 0.5
                    for mr, mc in main_cells:
                        nr = int(2 * pr - mr)
                        nc = int(2 * pc - mc)
                        if 0 <= nr < rows and 0 <= nc < cols and output[nr][nc] == 0:
                            output[nr][nc] = conn_color
                elif left_of and below:
                    pr = max_r + 0.5
                    pc = min_c - 0.5
                    for mr, mc in main_cells:
                        nr = int(2 * pr - mr)
                        nc = int(2 * pc - mc)
                        if 0 <= nr < rows and 0 <= nc < cols and output[nr][nc] == 0:
                            output[nr][nc] = conn_color
                elif right_of and above:
                    pr = min_r - 0.5
                    pc = max_c + 0.5
                    for mr, mc in main_cells:
                        nr = int(2 * pr - mr)
                        nc = int(2 * pc - mc)
                        if 0 <= nr < rows and 0 <= nc < cols and output[nr][nc] == 0:
                            output[nr][nc] = conn_color
                elif left_of and above:
                    pr = min_r - 0.5
                    pc = min_c - 0.5
                    for mr, mc in main_cells:
                        nr = int(2 * pr - mr)
                        nc = int(2 * pc - mc)
                        if 0 <= nr < rows and 0 <= nc < cols and output[nr][nc] == 0:
                            output[nr][nc] = conn_color
    return output"""

solutions["b782dc8a"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    markers = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in (0, 8):
                markers[(r,c)] = grid[r][c]
    center = None
    max_neighbors = 0
    for (r,c), v in markers.items():
        n = 0
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            if (r+dr, c+dc) in markers:
                n += 1
        if n > max_neighbors:
            max_neighbors = n
            center = (r, c)
    center_color = markers[center]
    arm_colors = set(v for v in markers.values() if v != center_color)
    arm_color = arm_colors.pop() if arm_colors else center_color
    output = [row[:] for row in grid]
    from collections import deque
    visited = {}
    queue = deque()
    visited[center] = 0
    queue.append((center, 0))
    for (r,c), v in markers.items():
        if (r,c) != center:
            visited[(r,c)] = 1
            queue.append(((r,c), 1))
    while queue:
        (r,c), dist = queue.popleft()
        color = center_color if dist % 2 == 0 else arm_color
        output[r][c] = color
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr,nc) not in visited and grid[nr][nc] == 0:
                visited[(nr,nc)] = dist + 1
                queue.append(((nr,nc), dist + 1))
    return output"""

solutions["b8825c91"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    four_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 4:
                four_cells.append((r,c))
    cr = (rows - 1) / 2
    cc = (cols - 1) / 2
    output = [row[:] for row in grid]
    for r, c in four_cells:
        sr = int(2 * cr - r)
        sc = int(2 * cc - c)
        if 0 <= sr < rows and 0 <= sc < cols and grid[sr][sc] != 4:
            output[r][c] = grid[sr][sc]
    return output"""

solutions["c1d99e64"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    zero_rows = []
    for r in range(rows):
        if all(grid[r][c] == 0 for c in range(cols)):
            zero_rows.append(r)
    zero_cols = []
    for c in range(cols):
        if all(grid[r][c] == 0 for r in range(rows)):
            zero_cols.append(c)
    for r in zero_rows:
        for c in range(cols):
            output[r][c] = 2
    for c in zero_cols:
        for r in range(rows):
            output[r][c] = 2
    return output"""

solutions["c3f564a4"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    non_zero_vals = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                non_zero_vals.add(grid[r][c])
    N = len(non_zero_vals)
    off = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                off = (grid[r][c] - 1 - (r + c)) % N
                break
        if off is not None:
            break
    output = []
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append(((r + c + off) % N) + 1)
        output.append(row)
    return output"""

solutions["c444b776"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    four_rows = []
    for r in range(rows):
        if all(grid[r][c] == 4 for c in range(cols)):
            four_rows.append(r)
    four_cols = []
    for c in range(cols):
        if all(grid[r][c] == 4 for r in range(rows)):
            four_cols.append(c)
    row_ranges = []
    prev = 0
    for r in sorted(four_rows):
        if r > prev:
            row_ranges.append((prev, r))
        prev = r + 1
    if prev < rows:
        row_ranges.append((prev, rows))
    col_ranges = []
    prev = 0
    for c in sorted(four_cols):
        if c > prev:
            col_ranges.append((prev, c))
        prev = c + 1
    if prev < cols:
        col_ranges.append((prev, cols))
    def get_pattern(r_start, r_end, c_start, c_end):
        pat = []
        for r in range(r_start, r_end):
            row = []
            for c in range(c_start, c_end):
                row.append(grid[r][c])
            pat.append(row)
        return pat
    def has_content(pat):
        for row in pat:
            for v in row:
                if v != 0:
                    return True
        return False
    source_pat = None
    for rs, re in row_ranges:
        for cs, ce in col_ranges:
            pat = get_pattern(rs, re, cs, ce)
            if has_content(pat):
                source_pat = pat
                break
        if source_pat:
            break
    output = [row[:] for row in grid]
    for rs, re in row_ranges:
        for cs, ce in col_ranges:
            h = re - rs
            w = ce - cs
            if h == len(source_pat) and w == len(source_pat[0]):
                for r in range(h):
                    for c in range(w):
                        output[rs + r][cs + c] = source_pat[r][c]
    return output"""

solutions["c909285e"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    from collections import Counter
    all_vals = Counter()
    for r in range(rows):
        for c in range(cols):
            all_vals[grid[r][c]] += 1
    for color in all_vals:
        cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == color]
        if len(cells) < 8 or len(cells) > 200:
            continue
        min_r = min(r for r,c in cells)
        max_r = max(r for r,c in cells)
        min_c = min(c for r,c in cells)
        max_c = max(c for r,c in cells)
        border_cells = set()
        for r in range(min_r, max_r+1):
            for c in range(min_c, max_c+1):
                if r == min_r or r == max_r or c == min_c or c == max_c:
                    border_cells.add((r,c))
        if set(cells) == border_cells:
            output = []
            for r in range(min_r, max_r+1):
                row = []
                for c in range(min_c, max_c+1):
                    row.append(grid[r][c])
                output.append(row)
            return output
    return grid"""

solutions["ce602527"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    bg = grid[0][0]
    shapes = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                v = grid[r][c]
                if v not in shapes:
                    shapes[v] = set()
                shapes[v].add((r,c))
    shape_info = {}
    for color, cells in shapes.items():
        min_r = min(r for r,c in cells)
        max_r = max(r for r,c in cells)
        min_c = min(c for r,c in cells)
        max_c = max(c for r,c in cells)
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        binary = []
        for r in range(min_r, max_r+1):
            row = []
            for c in range(min_c, max_c+1):
                row.append(1 if (r,c) in cells else 0)
            binary.append(row)
        row_types = [1 if all(v == 1 for v in row) else 0 for row in binary]
        shape_info[color] = {'cells': cells, 'h': h, 'w': w, 'binary': binary, 'row_types': row_types, 'size': len(cells)}
    large_c = max(shape_info, key=lambda c: shape_info[c]['size'])
    large = shape_info[large_c]
    binary = large['binary']
    h = large['h']
    row_groups = []
    i_r = 0
    while i_r < h:
        j_r = i_r + 1
        while j_r < h and binary[j_r] == binary[i_r]:
            j_r += 1
        is_full = 1 if all(v == 1 for v in binary[i_r]) else 0
        row_groups.append(is_full)
        i_r = j_r
    smalls = {c: v for c, v in shape_info.items() if c != large_c}
    for sc, sinfo in smalls.items():
        rt = sinfo['row_types']
        if len(rt) >= len(row_groups) and rt[:len(row_groups)] == row_groups:
            binary_s = sinfo['binary']
            output = []
            for row in binary_s:
                out_row = [sc if v == 1 else bg for v in row]
                output.append(out_row)
            return output
    for sc, sinfo in smalls.items():
        rt = sinfo['row_types']
        if len(rt) >= len(row_groups) and rt[-len(row_groups):] == row_groups:
            binary_s = sinfo['binary']
            output = []
            for row in binary_s:
                out_row = [sc if v == 1 else bg for v in row]
                output.append(out_row)
            return output
    return grid"""

solutions["d07ae81c"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    from collections import Counter
    vals = Counter()
    for r in range(rows):
        for c in range(cols):
            vals[grid[r][c]] += 1
    region_colors = [v for v, _ in vals.most_common(2)]
    markers = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in region_colors:
                markers.append((r, c, grid[r][c]))
    color_map = {}
    for mr, mc, mcolor in markers:
        neighbor_counts = Counter()
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = mr+dr, mc+dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr,nc) != (mr,mc):
                    v = grid[nr][nc]
                    if v in region_colors:
                        neighbor_counts[v] += 1
        if neighbor_counts:
            home_region = neighbor_counts.most_common(1)[0][0]
            color_map[home_region] = mcolor
    output = [row[:] for row in grid]
    for mr, mc, mcolor in markers:
        for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            r, c = mr + dr, mc + dc
            while 0 <= r < rows and 0 <= c < cols:
                orig = grid[r][c]
                if orig in color_map:
                    output[r][c] = color_map[orig]
                r += dr
                c += dc
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
with open("data/arc_python_solutions_b32.json", "w") as f:
    json.dump(passing, f, indent=2)
print(f"\nSaved {len(passing)} solutions")
