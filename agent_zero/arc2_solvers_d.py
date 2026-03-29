"""ARC-AGI-2 custom solvers batch D."""
from collections import Counter, deque


def solve_800d221b(grid):
    """Wall cells get recolored by nearest corner region's dominant color.

    The grid has a background color, a wall color, and corner pattern regions.
    The wall forms a fractal-like structure with a 3x3 ring at its center.
    Each wall cell gets recolored by BFS from the nearest corner region.
    The 3x3 ring border (8 cells surrounding the center) stays as wall.
    """
    H = len(grid)
    W = len(grid[0])
    colors = Counter()
    for r in range(H):
        for c in range(W):
            colors[grid[r][c]] += 1
    bg = colors.most_common(1)[0][0]
    wall = colors.most_common(2)[1][0]

    wall_set = set((r, c) for r in range(H) for c in range(W) if grid[r][c] == wall)

    # Find 3x3 ring center (where all 8 neighbors are wall)
    ring_center = None
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            border = [(r + dr, c + dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (dr, dc) != (0, 0)]
            if all(b in wall_set for b in border):
                ring_center = (r, c)
                break
        if ring_center:
            break

    # The 8 border cells stay as wall (not the center)
    ring_border = set()
    if ring_center:
        rr, rc = ring_center
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if (dr, dc) != (0, 0):
                    ring_border.add((rr + dr, rc + dc))

    # Find corner regions (non-bg, non-wall connected components)
    visited = [[False] * W for _ in range(H)]
    regions = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and grid[r][c] != wall and not visited[r][c]:
                comp = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] != bg and grid[nr][nc] != wall:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                region_colors = Counter(grid[r][c] for r, c in comp)
                dom_color = region_colors.most_common(1)[0][0]
                regions.append((comp, dom_color))

    # BFS through ALL wall cells (including ring border for path finding)
    # But only recolor non-ring-border cells
    dist = [[float('inf')] * W for _ in range(H)]
    assigned = [[None] * W for _ in range(H)]
    q = deque()

    for comp, dom_color in regions:
        for r, c in comp:
            dist[r][c] = 0
            assigned[r][c] = dom_color
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == wall and dist[nr][nc] == float('inf'):
                    dist[nr][nc] = 1
                    assigned[nr][nc] = dom_color
                    q.append((nr, nc))

    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == wall and dist[nr][nc] == float('inf'):
                dist[nr][nc] = dist[r][c] + 1
                assigned[nr][nc] = assigned[r][c]
                q.append((nr, nc))

    result = [row[:] for row in grid]
    for r in range(H):
        for c in range(W):
            if grid[r][c] == wall and (r, c) not in ring_border and assigned[r][c] is not None:
                result[r][c] = assigned[r][c]
    return result


def solve_4a21e3da(grid):
    """Reflect 7-cluster across 2-ray lines to grid corners.

    The grid has background 1s, a cluster of 7s, and one or two 2-markers on edges.
    Each 2 on the edge defines a ray going into the grid.
    The ray passes through the 7-cluster, replacing 1s with 2s (7s on the ray stay).
    The cluster is split by rays and each piece is pushed to the nearest corner.
    """
    H = len(grid)
    W = len(grid[0])

    twos = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == 2]
    sevens = set((r, c) for r in range(H) for c in range(W) if grid[r][c] == 7)

    if not sevens:
        return [row[:] for row in grid]

    # Determine rays
    rays = []
    for r, c in twos:
        if r == 0:
            rays.append(('down', c))
        elif r == H - 1:
            rays.append(('up', c))
        elif c == 0:
            rays.append(('right', r))
        elif c == W - 1:
            rays.append(('left', r))

    result = [[1] * W for _ in range(H)]

    v_ray_col = None
    v_ray_dir = None
    h_ray_row = None
    h_ray_dir = None

    for direction, coord in rays:
        if direction in ('down', 'up'):
            v_ray_col = coord
            v_ray_dir = direction
        else:
            h_ray_row = coord
            h_ray_dir = direction

    # Draw 2-lines through the cluster
    if v_ray_col is not None:
        col = v_ray_col
        ray_sevens = [r for r in range(H) if (r, col) in sevens]
        if ray_sevens:
            if v_ray_dir == 'down':
                start, end = 0, max(ray_sevens)
            else:
                start, end = min(ray_sevens), H - 1
            for r in range(start, end + 1):
                if (r, col) in sevens:
                    result[r][col] = 7
                else:
                    result[r][col] = 2
        else:
            if v_ray_dir == 'down':
                result[0][col] = 2
            else:
                result[H - 1][col] = 2

    if h_ray_row is not None:
        row = h_ray_row
        ray_sevens = [c for c in range(W) if (row, c) in sevens]
        if ray_sevens:
            if h_ray_dir == 'left':
                start, end = min(ray_sevens), W - 1
            else:
                start, end = 0, max(ray_sevens)
            for c in range(start, end + 1):
                if (row, c) in sevens:
                    result[row][c] = 7
                else:
                    result[row][c] = 2
        else:
            if h_ray_dir == 'left':
                result[row][W - 1] = 2
            else:
                result[row][0] = 2

    # Identify on-ray 7s (they stay in place)
    on_ray = set()
    if v_ray_col is not None:
        for r, c in sevens:
            if c == v_ray_col:
                on_ray.add((r, c))
    if h_ray_row is not None:
        for r, c in sevens:
            if r == h_ray_row:
                on_ray.add((r, c))

    off_ray = sevens - on_ray

    # Group by quadrant relative to rays
    groups = {}
    for r, c in off_ray:
        v_side = None
        h_side = None
        if v_ray_col is not None:
            v_side = 'L' if c < v_ray_col else 'R'
        if h_ray_row is not None:
            h_side = 'T' if r < h_ray_row else 'B'
        groups.setdefault((v_side, h_side), []).append((r, c))

    for key, cells in groups.items():
        v_side, h_side = key
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]

        # Determine if this group should be placed
        should_place = True
        if v_ray_col is not None and h_ray_row is not None:
            v_front = (v_ray_dir == 'down' and h_side == 'T') or (v_ray_dir == 'up' and h_side == 'B')
            h_front = (h_ray_dir == 'left' and v_side == 'R') or (h_ray_dir == 'right' and v_side == 'L')
            should_place = v_front or h_front

        if not should_place:
            continue

        # Row shift
        if v_ray_col is not None and h_ray_row is None:
            if v_ray_dir == 'down':
                row_shift = -min(rows)
            else:
                row_shift = (H - 1) - max(rows)
        elif h_ray_row is not None:
            if h_side == 'T':
                row_shift = -min(rows)
            else:
                row_shift = (H - 1) - max(rows)
        else:
            row_shift = 0

        # Col shift
        if h_ray_row is not None and v_ray_col is None:
            if h_ray_dir == 'left':
                col_shift = (W - 1) - max(cols)
            else:
                col_shift = -min(cols)
        elif v_ray_col is not None:
            if v_side == 'L':
                col_shift = -min(cols)
            else:
                col_shift = (W - 1) - max(cols)
        else:
            col_shift = 0

        for r, c in cells:
            nr, nc = r + row_shift, c + col_shift
            if 0 <= nr < H and 0 <= nc < W:
                result[nr][nc] = 7

    return result


ARC2_SOLVERS_D = {
    "800d221b": solve_800d221b,
    "4a21e3da": solve_4a21e3da,
}
