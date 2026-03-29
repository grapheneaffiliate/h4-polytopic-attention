"""ARC-AGI-2 custom solvers batch E."""

from collections import Counter, deque


def solve_7b0280bc(grid):
    """Shortest path network: find two endpoint blocks, recolor shortest path."""
    R = len(grid)
    C = len(grid[0])

    colors = Counter()
    for r in range(R):
        for c in range(C):
            colors[grid[r][c]] += 1
    bg = colors.most_common(1)[0][0]

    non_bg = {v: cnt for v, cnt in colors.items() if v != bg}
    sorted_colors = sorted(non_bg.items(), key=lambda x: x[1])
    endpoint_color = sorted_colors[0][0]
    connector_color = sorted_colors[1][0]
    path_color = sorted_colors[2][0]

    blocks = []
    used = set()
    for r in range(R - 1):
        for c in range(C - 1):
            v = grid[r][c]
            if v in (connector_color, endpoint_color) and (r, c) not in used:
                if grid[r + 1][c] == v and grid[r][c + 1] == v and grid[r + 1][c + 1] == v:
                    blocks.append((r, c, v))
                    used.update([(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)])

    block_cells_map = {}
    all_block_cells = set()
    block_at = {}
    for idx, (r, c, v) in enumerate(blocks):
        cells = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
        block_cells_map[idx] = cells
        all_block_cells.update(cells)
        for cell in cells:
            block_at[cell] = idx

    path_cells = {(r, c) for r in range(R) for c in range(C) if grid[r][c] != bg and (r, c) not in all_block_cells}
    dirs8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    adj = {i: set() for i in range(len(blocks))}
    for idx in range(len(blocks)):
        starts = set()
        for cell in block_cells_map[idx]:
            r, c = cell
            for dr, dc in dirs8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C and (nr, nc) in path_cells:
                    starts.add((nr, nc))

        visited = set(starts)
        queue = deque(starts)
        while queue:
            cr, cc = queue.popleft()
            for dr, dc in dirs8:
                nr, nc = cr + dr, cc + dc
                if (nr, nc) in visited:
                    continue
                if 0 <= nr < R and 0 <= nc < C:
                    if (nr, nc) in path_cells:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
                    elif (nr, nc) in all_block_cells:
                        other = block_at[(nr, nc)]
                        if other != idx:
                            adj[idx].add(other)

    endpoints = [i for i, (r, c, v) in enumerate(blocks) if v == endpoint_color]

    start, end = endpoints[0], endpoints[1]
    parent = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == end:
            break
        for neighbor in adj[node]:
            if neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)

    path_blocks = []
    node = end
    while node is not None:
        path_blocks.append(node)
        node = parent[node]
    path_blocks.reverse()

    result = [row[:] for row in grid]

    for block_idx in path_blocks:
        r, c, v = blocks[block_idx]
        if v == connector_color:
            for cell in block_cells_map[block_idx]:
                result[cell[0]][cell[1]] = 3

    for i in range(len(path_blocks) - 1):
        b1, b2 = path_blocks[i], path_blocks[i + 1]

        starts1 = set()
        for cell in block_cells_map[b1]:
            r, c = cell
            for dr, dc in dirs8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C and (nr, nc) in path_cells:
                    starts1.add((nr, nc))

        visited1 = set(starts1)
        queue = deque(starts1)
        while queue:
            cr, cc = queue.popleft()
            for dr, dc in dirs8:
                nr, nc = cr + dr, cc + dc
                if (nr, nc) in visited1:
                    continue
                if 0 <= nr < R and 0 <= nc < C:
                    if (nr, nc) in path_cells:
                        visited1.add((nr, nc))
                        queue.append((nr, nc))
                    elif (nr, nc) in all_block_cells:
                        pass

        starts2 = set()
        for cell in block_cells_map[b2]:
            r, c = cell
            for dr, dc in dirs8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C and (nr, nc) in path_cells:
                    starts2.add((nr, nc))

        visited2 = set(starts2)
        queue = deque(starts2)
        while queue:
            cr, cc = queue.popleft()
            for dr, dc in dirs8:
                nr, nc = cr + dr, cc + dc
                if (nr, nc) in visited2:
                    continue
                if 0 <= nr < R and 0 <= nc < C:
                    if (nr, nc) in path_cells:
                        visited2.add((nr, nc))
                        queue.append((nr, nc))
                    elif (nr, nc) in all_block_cells:
                        pass

        edge_cells = visited1 & visited2
        for cell in edge_cells:
            result[cell[0]][cell[1]] = 5

    return result


def solve_7c66cb00(grid):
    """Move shapes from background area into matching colored bands."""
    R = len(grid)
    C = len(grid[0])

    bg = Counter(grid[0]).most_common(1)[0][0]

    bg_rows = set()
    for r in range(R):
        if all(grid[r][c] == bg for c in range(C)):
            bg_rows.add(r)

    bands = []
    in_band = False
    band_start = -1
    for r in range(R):
        is_bg_row = r in bg_rows
        if not is_bg_row and not in_band:
            row_colors = set(grid[r])
            if bg not in row_colors:
                band_start = r
                in_band = True
        elif is_bg_row and in_band:
            bands.append((band_start, r - 1))
            in_band = False
    if in_band:
        bands.append((band_start, R - 1))

    band_info = []
    band_by_fill = {}
    for start, end in bands:
        border_color = grid[start][0]
        fill_counts = Counter()
        for r in range(start, end + 1):
            for c in range(1, C - 1):
                if grid[r][c] != border_color:
                    fill_counts[grid[r][c]] += 1
        fill_color = fill_counts.most_common(1)[0][0] if fill_counts else border_color
        band_info.append((start, end, fill_color, border_color))
        band_by_fill[fill_color] = (start, end, fill_color, border_color)

    band_rows_set = set()
    for start, end, _, _ in band_info:
        for r in range(start, end + 1):
            band_rows_set.add(r)

    color_cells = {}
    for r in range(R):
        if r in band_rows_set:
            continue
        for c in range(C):
            if grid[r][c] != bg:
                color = grid[r][c]
                if color not in color_cells:
                    color_cells[color] = set()
                color_cells[color].add((r, c))

    dirs8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    all_components = []

    for color, cells in color_cells.items():
        remaining = cells.copy()
        while remaining:
            start = remaining.pop()
            comp = {start}
            queue = deque([start])
            while queue:
                cr, cc = queue.popleft()
                for dr, dc in dirs8:
                    nr, nc = cr + dr, cc + dc
                    if (nr, nc) in remaining:
                        remaining.discard((nr, nc))
                        comp.add((nr, nc))
                        queue.append((nr, nc))
            all_components.append((color, comp))

    result = [row[:] for row in grid]

    for color, cells in color_cells.items():
        for r, c in cells:
            result[r][c] = bg

    for color, comp in all_components:
        if color not in band_by_fill:
            continue
        band_start, band_end, band_fill, band_border = band_by_fill[color]

        max_row = max(r for r, c in comp)
        row_offset = band_end - max_row

        for r, c in comp:
            new_r = r + row_offset
            if band_start <= new_r <= band_end:
                result[new_r][c] = band_border

    return result


ARC2_SOLVERS_E = {
    "7b0280bc": solve_7b0280bc,
    "7c66cb00": solve_7c66cb00,
}
