"""ARC-AGI-2 custom solvers batch B."""

import copy
import math
from collections import deque, Counter


def solve_65b59efc(grid):
    """Grid with pattern templates, layout map, and color markers.

    The input has 3 row sections separated by rows of 5s:
    1. Pattern row: 3x3 (or NxN) template patterns in each column group
    2. Layout row: shows where each pattern goes in the output grid
    3. Marker row: replacement colors for each column's pattern

    The output places patterns at positions defined by the layout,
    with original colors replaced by marker colors.
    """
    H, W = len(grid), len(grid[0])

    # Find separator rows (containing 5s mixed with 0s)
    sep_rows = []
    for r in range(H):
        has_5 = any(grid[r][c] == 5 for c in range(W))
        all_5_or_0 = all(grid[r][c] in (0, 5) for c in range(W))
        if has_5 and all_5_or_0:
            sep_rows.append(r)

    # Find separator columns
    sep_cols = []
    for c in range(W):
        has_5 = any(grid[r][c] == 5 for r in range(H))
        all_5_or_0 = all(grid[r][c] in (0, 5) for r in range(H))
        if has_5 and all_5_or_0:
            sep_cols.append(c)

    def get_groups(seps, total):
        groups = []
        prev = -1
        for s in seps:
            if s > prev + 1:
                groups.append((prev + 1, s - 1))
            prev = s
        if prev < total - 1:
            groups.append((prev + 1, total - 1))
        return groups

    row_groups = get_groups(sep_rows, H)
    col_groups = get_groups(sep_cols, W)

    pattern_rg = row_groups[0]
    layout_rg = row_groups[1]
    marker_rg = row_groups[2] if len(row_groups) > 2 else None

    cell_h = pattern_rg[1] - pattern_rg[0] + 1
    cell_w = col_groups[0][1] - col_groups[0][0] + 1

    # Extract patterns from pattern row
    pattern_by_col = {}
    for ci, (c1, c2) in enumerate(col_groups):
        r1, r2 = pattern_rg
        block = []
        for r in range(r1, r2 + 1):
            row = []
            for c in range(c1, c2 + 1):
                row.append(grid[r][c])
            block.append(row)
        colors = set(v for row in block for v in row if v != 0)
        if len(colors) == 1:
            color = colors.pop()
        elif len(colors) > 1:
            color = max(colors, key=lambda cv: sum(1 for row in block for v in row if v == cv))
        else:
            color = 0
        norm = [[1 if v == color else 0 for v in row] for row in block]
        pattern_by_col[ci] = (color, norm)

    # Extract markers
    markers = {}
    if marker_rg:
        for ci, (c1, c2) in enumerate(col_groups):
            for r in range(marker_rg[0], marker_rg[1] + 1):
                for c in range(c1, c2 + 1):
                    if grid[r][c] != 0:
                        markers[ci] = grid[r][c]

    # Extract layout
    layout = {}
    lr1, lr2 = layout_rg
    for ci, (c1, c2) in enumerate(col_groups):
        for local_r in range(cell_h):
            for local_c in range(cell_w):
                r = lr1 + local_r
                c = c1 + local_c
                if r <= lr2 and c <= c2:
                    v = grid[r][c]
                    if v != 0 and v != 5:
                        layout[(local_r, local_c)] = v

    # Build color->column mapping
    color_to_col = {}
    for ci, (color, norm) in pattern_by_col.items():
        if color != 0:
            color_to_col[color] = ci

    # Build output grid
    out_h = cell_h * cell_h
    out_w = cell_w * cell_w
    result = [[0] * out_w for _ in range(out_h)]

    for lr in range(cell_h):
        for lc in range(cell_w):
            if (lr, lc) in layout:
                orig_color = layout[(lr, lc)]
                if orig_color in color_to_col:
                    col_idx = color_to_col[orig_color]
                    _, norm = pattern_by_col[col_idx]
                    marker_color = markers.get(col_idx, orig_color)

                    for pr in range(cell_h):
                        for pc in range(cell_w):
                            out_r = lr * cell_h + pr
                            out_c = lc * cell_w + pc
                            if out_r < out_h and out_c < out_w:
                                result[out_r][out_c] = marker_color if norm[pr][pc] else 0

    return result


def solve_dd6b8c4b(grid):
    """3x3 core with 3s and center 2, surrounded by 6-frame, with 9s outside.

    Count 9s reachable from the core (through non-6 cells).
    Fill that many core cells with 9 in reading order.
    Remove those 9s (replace with 7 background).
    For n>9 case, use BFS distance from core to determine removal priority.
    """
    grid = copy.deepcopy(grid)
    H, W = len(grid), len(grid[0])
    bg = 7

    # Find center (2)
    center = None
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 2:
                center = (r, c)
    if center is None:
        return grid
    cr, cc = center

    core = set()
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            core.add((cr + dr, cc + dc))

    sixes = set()
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 6:
                sixes.add((r, c))

    # Flood fill from core through non-6 cells
    reachable = set()
    q = deque()
    for cell in core:
        reachable.add(cell)
        q.append(cell)

    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in sixes and (nr, nc) not in reachable:
                reachable.add((nr, nc))
                q.append((nr, nc))

    nines_in_comp = [(r, c) for r, c in reachable if grid[r][c] == 9 and (r, c) not in core]
    n = len(nines_in_comp)

    # Reading order for core cells
    core_order = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            core_order.append((cr + dr, cc + dc))

    n_fill = min(n, 9)
    for i in range(n_fill):
        r, c = core_order[i]
        grid[r][c] = 9

    if n <= 9:
        for r, c in nines_in_comp:
            grid[r][c] = bg
    else:
        # BFS from core boundary through non-9, non-6 cells
        # When we reach a 9, record its BFS distance
        # Nines blocked by other nines have higher distance (shielded)
        nine_set = set(nines_in_comp)
        bfs_dist = {}
        bfs_q = deque()
        bfs_visited = set()

        for cell_r, cell_c in core:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cell_r + dr, cell_c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in core and (nr, nc) not in sixes:
                    if (nr, nc) not in bfs_visited:
                        bfs_visited.add((nr, nc))
                        bfs_q.append((nr, nc, 1))

        while bfs_q:
            r, c, d = bfs_q.popleft()
            if (r, c) in nine_set:
                if (r, c) not in bfs_dist:
                    bfs_dist[(r, c)] = d
                continue  # Don't BFS through 9s
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in core and (nr, nc) not in sixes and (nr, nc) not in bfs_visited:
                    bfs_visited.add((nr, nc))
                    bfs_q.append((nr, nc, d + 1))

        removable = [
            (bfs_dist.get(rc, float('inf')), math.sqrt((rc[0] - cr) ** 2 + (rc[1] - cc) ** 2), rc)
            for rc in nines_in_comp
        ]
        removable.sort()

        for _, _, (r, c) in removable[:n_fill]:
            grid[r][c] = bg

    return grid


ARC2_SOLVERS_B = {
    "65b59efc": solve_65b59efc,
    "dd6b8c4b": solve_dd6b8c4b,
}
