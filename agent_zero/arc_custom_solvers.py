"""
Custom ARC solvers for specific tasks.
Each solver is a function that takes a grid (list of lists of ints) and returns the output grid.
"""

from collections import Counter, deque, defaultdict


def solve_de1cd16c(grid):
    """Grid divided into regions of different bg colors with marker dots.
    Output = bg color of region with the most markers."""
    h, w = len(grid), len(grid[0])
    flat = [grid[r][c] for r in range(h) for c in range(w)]
    counts = Counter(flat)
    marker_color = min(counts, key=lambda c: counts[c])
    region_markers = Counter()
    for r in range(h):
        for c in range(w):
            if grid[r][c] == marker_color:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != marker_color:
                        region_markers[grid[nr][nc]] += 1
                        break
    best_region = region_markers.most_common(1)[0][0]
    return [[best_region]]


def solve_239be575(grid):
    """Two 2x2 blocks of color 2 in a grid of 0s and 8s.
    Output [[8]] if blocks can be connected via 8-cells, else [[0]]."""
    h, w = len(grid), len(grid[0])
    blocks = []
    for r in range(h - 1):
        for c in range(w - 1):
            if grid[r][c] == 2 and grid[r][c + 1] == 2 and grid[r + 1][c] == 2 and grid[r + 1][c + 1] == 2:
                blocks.append((r, c))
    b1, b2 = blocks[0], blocks[1]
    start = {(b1[0], b1[1]), (b1[0], b1[1] + 1), (b1[0] + 1, b1[1]), (b1[0] + 1, b1[1] + 1)}
    target = {(b2[0], b2[1]), (b2[0], b2[1] + 1), (b2[0] + 1, b2[1]), (b2[0] + 1, b2[1] + 1)}
    visited = set(start)
    queue = deque(start)
    while queue:
        r, c = queue.popleft()
        if (r, c) in target:
            return [[8]]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                if grid[nr][nc] == 8 or grid[nr][nc] == 2:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return [[0]]


def solve_27a28665(grid):
    """3x3 binary pattern classification. Lookup shape -> output number."""
    binary = tuple(tuple(1 if c != 0 else 0 for c in row) for row in grid)
    patterns = {
        ((1, 1, 0), (1, 0, 1), (0, 1, 0)): 1,
        ((1, 0, 1), (0, 1, 0), (1, 0, 1)): 2,
        ((0, 1, 1), (0, 1, 1), (1, 0, 0)): 3,
        ((0, 1, 0), (1, 1, 1), (0, 1, 0)): 6,
    }
    lookup = {}
    for pat, out in patterns.items():
        g = [list(row) for row in pat]
        for _ in range(4):
            key = tuple(tuple(r) for r in g)
            lookup[key] = out
            flipped = tuple(tuple(r[::-1]) for r in g)
            lookup[flipped] = out
            h, w = len(g), len(g[0])
            g = [[g[h - 1 - r][c] for r in range(h)] for c in range(w)]
    if binary in lookup:
        return [[lookup[binary]]]
    return [[0]]


def solve_25ff71a9(grid):
    """Shift grid down by one row."""
    return [[0] * len(grid[0])] + [row[:] for row in grid[:-1]]


def solve_48d8fb45(grid):
    """5 marks position above a 3x3 region to extract.
    Output = 3x3 region at (5_row+1, 5_col-1) to (5_row+3, 5_col+1)."""
    h, w = len(grid), len(grid[0])
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 5:
                result = []
                for dr in range(1, 4):
                    row = []
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            row.append(grid[nr][nc])
                        else:
                            row.append(0)
                    result.append(row)
                return result
    return [[0] * 3] * 3


def solve_137eaa0f(grid):
    """Multiple 5-anchored fragments. Each 5 has colored neighbors.
    Overlay all fragments onto a 3x3 grid with 5 at center."""
    h, w = len(grid), len(grid[0])
    result = [[0] * 3 for _ in range(3)]
    result[1][1] = 5
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 5:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0 and grid[nr][nc] != 5:
                            result[1 + dr][1 + dc] = grid[nr][nc]
    return result


def solve_39a8645d(grid):
    """Multiple colored clusters (8-connected). Find the shape that appears
    most frequently. Output that shape's 3x3 representation."""
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    clusters = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                cells = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                                visited[nr][nc] = True
                                q.append((nr, nc))
                clusters.append((color, cells))
    shape_counts = Counter()
    shape_to_patch = {}
    for color, cells in clusters:
        min_r = min(r for r, c in cells)
        min_c = min(c for r, c in cells)
        sk = tuple(sorted((r - min_r, c - min_c) for r, c in cells))
        shape_counts[sk] += 1
        if sk not in shape_to_patch:
            max_r = max(r for r, c in cells)
            max_c = max(c for r, c in cells)
            bh, bw = max_r - min_r + 1, max_c - min_c + 1
            patch = [[0] * bw for _ in range(bh)]
            for r, c in cells:
                patch[r - min_r][c - min_c] = color
            shape_to_patch[sk] = patch
    best = shape_counts.most_common(1)[0][0]
    return shape_to_patch[best]


def solve_4be741c5(grid):
    """Grid has color bands (horizontal or vertical). Output = sequence of colors
    sorted by average position."""
    h, w = len(grid), len(grid[0])
    color_cols = defaultdict(list)
    color_rows = defaultdict(list)
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            color_cols[v].append(c)
            color_rows[v].append(r)
    colors = list(color_cols.keys())
    avg_col = {c: sum(color_cols[c]) / len(color_cols[c]) for c in colors}
    avg_row = {c: sum(color_rows[c]) / len(color_rows[c]) for c in colors}
    col_spread = max(avg_col.values()) - min(avg_col.values())
    row_spread = max(avg_row.values()) - min(avg_row.values())
    if col_spread >= row_spread:
        sorted_colors = sorted(colors, key=lambda c: avg_col[c])
        return [sorted_colors]
    else:
        sorted_colors = sorted(colors, key=lambda c: avg_row[c])
        return [[c] for c in sorted_colors]


def solve_5ad4f10b(grid):
    """Large block of one color forms a 3x3 arrangement of sub-blocks.
    Output: 3x3 grid where filled sub-blocks get the scatter color, empty get 0."""
    h, w = len(grid), len(grid[0])
    colors = set(grid[r][c] for r in range(h) for c in range(w) if grid[r][c] != 0)
    best_block_color = None
    best_cluster_size = 0
    for color in colors:
        visited = set()
        for r in range(h):
            for c in range(w):
                if grid[r][c] == color and (r, c) not in visited:
                    cluster_size = 0
                    q = deque([(r, c)])
                    visited.add((r, c))
                    while q:
                        cr, cc = q.popleft()
                        cluster_size += 1
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and grid[nr][nc] == color:
                                visited.add((nr, nc))
                                q.append((nr, nc))
                    if cluster_size > best_cluster_size:
                        best_cluster_size = cluster_size
                        best_block_color = color
    scatter_color = [c for c in colors if c != best_block_color][0]
    block_cells = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == best_block_color]
    min_r = min(r for r, c in block_cells)
    max_r = max(r for r, c in block_cells)
    min_c = min(c for r, c in block_cells)
    max_c = max(c for r, c in block_cells)
    sub_h = (max_r - min_r + 1) // 3
    sub_w = (max_c - min_c + 1) // 3
    result = [[0] * 3 for _ in range(3)]
    for sr in range(3):
        for sc in range(3):
            r_start = min_r + sr * sub_h
            c_start = min_c + sc * sub_w
            filled = all(grid[r_start + dr][c_start + dc] == best_block_color
                         for dr in range(sub_h) for dc in range(sub_w))
            if filled:
                result[sr][sc] = scatter_color
    return result


def solve_780d0b14(grid):
    """Grid divided by separator rows/cols of all-0s into rectangular regions.
    Each region has a dominant color. Output = grid of dominant colors."""
    h, w = len(grid), len(grid[0])
    sep_rows = [r for r in range(h) if all(grid[r][c] == 0 for c in range(w))]
    sep_cols = [c for c in range(w) if all(grid[r][c] == 0 for r in range(h))]
    row_bands = []
    prev = 0
    for sr in sep_rows:
        if sr > prev:
            row_bands.append((prev, sr - 1))
        prev = sr + 1
    if prev < h:
        row_bands.append((prev, h - 1))
    col_bands = []
    prev = 0
    for sc in sep_cols:
        if sc > prev:
            col_bands.append((prev, sc - 1))
        prev = sc + 1
    if prev < w:
        col_bands.append((prev, w - 1))
    result = []
    for r_start, r_end in row_bands:
        row = []
        for c_start, c_end in col_bands:
            cells = [grid[r][c] for r in range(r_start, r_end + 1)
                     for c in range(c_start, c_end + 1) if grid[r][c] != 0]
            row.append(Counter(cells).most_common(1)[0][0] if cells else 0)
        result.append(row)
    return result


CUSTOM_SOLVERS = {
    "de1cd16c": solve_de1cd16c,
    "239be575": solve_239be575,
    "27a28665": solve_27a28665,
    "25ff71a9": solve_25ff71a9,
    "48d8fb45": solve_48d8fb45,
    "137eaa0f": solve_137eaa0f,
    "39a8645d": solve_39a8645d,
    "4be741c5": solve_4be741c5,
    "5ad4f10b": solve_5ad4f10b,
    "780d0b14": solve_780d0b14,
}
