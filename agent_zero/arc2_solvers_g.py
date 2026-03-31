"""ARC-AGI-2 solvers - batch G."""

from collections import deque, Counter


def solve_d8e07eb2(grid):
    """Grid with top reference, middle 4x4 pattern grid, separator rows of 6s.
    Match top patterns to middle positions by normalized shape. Draw 3-borders."""
    grid = [row[:] for row in grid]
    H = len(grid)
    W = len(grid[0])

    # Find 6-rows (separators)
    six_rows = [r for r in range(H) if all(grid[r][c] == 6 for c in range(W))]

    mid_start = six_rows[0] + 1
    mid_end = six_rows[1] - 1
    top_start = 0
    top_end = six_rows[0] - 1
    bot_start = six_rows[1] + 1
    bot_end = H - 1

    row_groups = [(mid_start + 2 + 5 * i, mid_start + 4 + 5 * i) for i in range(4)]
    col_groups = [(2 + 5 * j, 4 + 5 * j) for j in range(4)]
    top_pattern_rows = (top_start + 1, top_start + 3)

    def normalize(pat):
        return tuple(tuple(0 if v != 8 else 8 for v in row) for row in pat)

    def get_pattern(g, rstart, rend, cstart, cend):
        return tuple(
            tuple(g[r][c] for c in range(cstart, cend + 1))
            for r in range(rstart, rend + 1)
        )

    top_pats = []
    for ci, (cs, ce) in enumerate(col_groups):
        pat = get_pattern(grid, top_pattern_rows[0], top_pattern_rows[1], cs, ce)
        top_pats.append(normalize(pat))

    matches = []
    for ti, tp in enumerate(top_pats):
        for ri, (rs, re) in enumerate(row_groups):
            for ci, (cs, ce) in enumerate(col_groups):
                pat = get_pattern(grid, rs, re, cs, ce)
                if normalize(pat) == tp:
                    matches.append((ti, ri, ci))

    border_cells = set()
    for ti, ri, ci in matches:
        r_start_band = mid_start + 1 + 5 * ri
        r_end_band = mid_start + 5 + 5 * ri
        c_start_band = 1 + 5 * ci
        c_end_band = 5 + 5 * ci
        for r in range(r_start_band, r_end_band + 1):
            for c in range(c_start_band, c_end_band + 1):
                border_cells.add((r, c))

    for r, c in border_cells:
        if grid[r][c] == 8:
            grid[r][c] = 3

    match_rows = set(ri for ti, ri, ci in matches)
    match_cols = set(ci for ti, ri, ci in matches)
    all_same = (len(match_rows) == 1 and len(matches) == 4) or \
               (len(match_cols) == 1 and len(matches) == 4)

    if all_same:
        for r in range(top_start, top_end + 1):
            for c in range(W):
                if grid[r][c] == 8:
                    grid[r][c] = 3
        for r in range(bot_start, bot_end + 1):
            for c in range(W):
                grid[r][c] = 3
    else:
        for r in range(bot_start, bot_end + 1):
            for c in range(W):
                grid[r][c] = 2

    return grid


def solve_e3721c99(grid):
    """Reference strip defines color mapping by hole count.
    Each blob in main area gets colored based on number of internal hole groups."""
    grid = [row[:] for row in grid]
    H, W = len(grid), len(grid[0])
    blob_color = 5

    # Find separator lines
    h_sep = None
    h_sep_color = None
    for r in range(H):
        non_zero_non_blob = [grid[r][c] for c in range(W)
                             if grid[r][c] != 0 and grid[r][c] != blob_color]
        if non_zero_non_blob:
            mc = Counter(non_zero_non_blob).most_common(1)[0]
            if mc[1] >= 8:
                h_sep = r
                h_sep_color = mc[0]
                break

    v_sep = None
    v_sep_color = None
    if h_sep_color:
        for c in range(W):
            count = 0
            for r in range(H):
                if grid[r][c] == h_sep_color:
                    count += 1
                else:
                    break
            if count >= 3 and c < W // 2:
                v_sep = c
                v_sep_color = h_sep_color
                break

    if h_sep is None:
        for c in range(W):
            non_zero_non_blob = [grid[r][c] for r in range(H)
                                 if grid[r][c] != 0 and grid[r][c] != blob_color]
            if non_zero_non_blob:
                mc = Counter(non_zero_non_blob).most_common(1)[0]
                if mc[1] >= 3:
                    v_sep = c
                    v_sep_color = mc[0]
                    h_sep_color = mc[0]
                    for r in range(H):
                        count = sum(1 for cc in range(W) if grid[r][cc] == v_sep_color)
                        if count >= 8:
                            h_sep = r
                            break
                    break

    if h_sep is None and v_sep is None:
        return grid

    ref_rows = [1, 2, 3]
    ref_cols = list(range(v_sep)) if v_sep is not None else list(range(W))

    non_zero_cols = set()
    for r in ref_rows:
        for c in ref_cols:
            if grid[r][c] != 0:
                non_zero_cols.add(c)

    if not non_zero_cols:
        return grid

    cols = sorted(non_zero_cols)
    groups = []
    start = cols[0]
    prev = cols[0]
    for c in cols[1:]:
        if c > prev + 1:
            groups.append((start, prev))
            start = c
        prev = c
    groups.append((start, prev))

    seg_colors = {}
    for idx, (cs, ce) in enumerate(groups):
        pattern = []
        color = None
        for r in ref_rows:
            row = []
            for c in range(cs, ce + 1):
                v = grid[r][c]
                row.append(v)
                if v != 0 and color is None:
                    color = v
            pattern.append(row)

        w = ce - cs + 1
        h = len(ref_rows)
        internal_zeros = 0
        for ri in range(h):
            for ci in range(w):
                if pattern[ri][ci] == 0:
                    if ri > 0 and ri < h - 1 and ci > 0 and ci < w - 1:
                        internal_zeros += 1

        seg_colors[internal_zeros] = color

    visited = set()
    blobs = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == blob_color and (r, c) not in visited:
                queue = deque([(r, c)])
                visited.add((r, c))
                blob = set()
                blob.add((r, c))
                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited and grid[nr][nc] == blob_color:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                            blob.add((nr, nc))
                blobs.append(blob)

    for blob in blobs:
        rmin = min(r for r, c in blob)
        rmax = max(r for r, c in blob)
        cmin = min(c for r, c in blob)
        cmax = max(c for r, c in blob)

        zero_cells = set()
        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                if (r, c) not in blob:
                    zero_cells.add((r, c))

        boundary_connected = set()
        queue = deque()
        for r in range(rmin, rmax + 1):
            for c in [cmin, cmax]:
                if (r, c) in zero_cells:
                    boundary_connected.add((r, c))
                    queue.append((r, c))
        for c in range(cmin, cmax + 1):
            for r in [rmin, rmax]:
                if (r, c) in zero_cells and (r, c) not in boundary_connected:
                    boundary_connected.add((r, c))
                    queue.append((r, c))
        while queue:
            cr, cc = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if rmin <= nr <= rmax and cmin <= nc <= cmax and (nr, nc) in zero_cells and (nr, nc) not in boundary_connected:
                    boundary_connected.add((nr, nc))
                    queue.append((nr, nc))

        holes = zero_cells - boundary_connected

        hole_groups = 0
        hole_visited = set()
        for h in holes:
            if h not in hole_visited:
                hole_groups += 1
                hq = deque([h])
                hole_visited.add(h)
                while hq:
                    hr, hc = hq.popleft()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = hr + dr, hc + dc
                        if (nr, nc) in holes and (nr, nc) not in hole_visited:
                            hole_visited.add((nr, nc))
                            hq.append((nr, nc))

        new_color = seg_colors.get(hole_groups, 0)
        for r, c in blob:
            grid[r][c] = new_color

    return grid


ARC2_SOLVERS_G = {
    "d8e07eb2": solve_d8e07eb2,
    "e3721c99": solve_e3721c99,
}
