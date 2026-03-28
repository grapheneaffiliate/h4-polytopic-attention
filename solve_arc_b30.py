import json

solutions = {}

# ============ 80af3007 ============
# Fractal pattern: input has 3x3 meta-grid of 3xN blocks of 5s.
# Output is 9x9 where cell (r,c) = 5 if meta[r//3][c//3] AND meta[r%3][c%3], else 0.
solutions["80af3007"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find bounding box of 5s
    min_r = min_c = float('inf')
    max_r = max_c = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    h = max_r - min_r + 1
    w = max_c - min_c + 1
    ch = h // 3
    cw = w // 3

    # Build meta-grid
    meta = [[0]*3 for _ in range(3)]
    for mr in range(3):
        for mc in range(3):
            if grid[min_r + mr*ch][min_c + mc*cw] == 5:
                meta[mr][mc] = 1

    # Build output: fractal self-similarity
    output = [[0]*9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            if meta[r//3][c//3] and meta[r%3][c%3]:
                output[r][c] = 5

    return output"""

# ============ 83302e8f ============
# Grid with line pattern (like tic-tac-toe). Some line segments have 0 gaps.
# Fill cells: 3 if no border gaps, 4 if any border gap. Preserve line structure.
solutions["83302e8f"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    from collections import Counter
    cnt = Counter(grid[r][c] for r in range(rows) for c in range(cols))
    line_color = max((c for c in cnt if c != 0), key=lambda c: cnt[c])

    # Find grid line rows and columns
    line_rows = []
    for r in range(rows):
        count = sum(1 for c in range(cols) if grid[r][c] == line_color)
        if count > cols * 0.5:
            line_rows.append(r)

    line_cols = []
    for c in range(cols):
        count = sum(1 for r in range(rows) if grid[r][c] == line_color)
        if count > rows * 0.5:
            line_cols.append(c)

    # Define cell boundaries
    all_rows = [-1] + line_rows + [rows]
    all_cols = [-1] + line_cols + [cols]

    # First, determine fill color for each cell region
    cell_colors = {}
    for ri in range(len(all_rows) - 1):
        for ci in range(len(all_cols) - 1):
            r_start = all_rows[ri] + 1
            r_end = all_rows[ri + 1]
            c_start = all_cols[ci] + 1
            c_end = all_cols[ci + 1]

            if r_start >= r_end or c_start >= c_end:
                continue

            has_gap = False

            if all_rows[ri] >= 0:
                r = all_rows[ri]
                for c in range(c_start, c_end):
                    if grid[r][c] != line_color:
                        has_gap = True
                        break

            if not has_gap and all_rows[ri + 1] < rows:
                r = all_rows[ri + 1]
                for c in range(c_start, c_end):
                    if grid[r][c] != line_color:
                        has_gap = True
                        break

            if not has_gap and all_cols[ci] >= 0:
                c = all_cols[ci]
                for r in range(r_start, r_end):
                    if grid[r][c] != line_color:
                        has_gap = True
                        break

            if not has_gap and all_cols[ci + 1] < cols:
                c = all_cols[ci + 1]
                for r in range(r_start, r_end):
                    if grid[r][c] != line_color:
                        has_gap = True
                        break

            fill_color = 4 if has_gap else 3
            cell_colors[(ri, ci)] = fill_color

    output = [row[:] for row in grid]

    # Fill cell interiors
    for (ri, ci), fill_color in cell_colors.items():
        r_start = all_rows[ri] + 1
        r_end = all_rows[ri + 1]
        c_start = all_cols[ci] + 1
        c_end = all_cols[ci + 1]

        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                output[r][c] = fill_color

    # Fill gap cells on lines with 4
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and output[r][c] == 0:
                output[r][c] = 4

    return output"""

# ============ 855e0971 ============
# Grid has color bands. Each band has one 0 cell.
# The 0 extends as a vertical/horizontal stripe through all rows/cols of that band.
solutions["855e0971"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    # Check row-based bands (each row has one dominant color)
    row_colors = []
    for r in range(rows):
        colors = set(grid[r]) - {0}
        if len(colors) == 1:
            row_colors.append(colors.pop())
        else:
            row_colors.append(None)

    col_colors = []
    for c in range(cols):
        colors = set(grid[r][c] for r in range(rows)) - {0}
        if len(colors) == 1:
            col_colors.append(colors.pop())
        else:
            col_colors.append(None)

    if all(c is not None for c in row_colors):
        # Row-based bands
        bands = []
        i = 0
        while i < rows:
            color = row_colors[i]
            j = i
            while j < rows and row_colors[j] == color:
                j += 1
            bands.append((i, j, color))
            i = j

        for start, end, color in bands:
            zeros = []
            for r in range(start, end):
                for c in range(cols):
                    if grid[r][c] == 0:
                        zeros.append((r, c))
            for _, c in zeros:
                for r in range(start, end):
                    out[r][c] = 0
        return out

    if all(c is not None for c in col_colors):
        # Column-based bands
        bands = []
        i = 0
        while i < cols:
            color = col_colors[i]
            j = i
            while j < cols and col_colors[j] == color:
                j += 1
            bands.append((i, j, color))
            i = j

        for start, end, color in bands:
            zeros = []
            for c in range(start, end):
                for r in range(rows):
                    if grid[r][c] == 0:
                        zeros.append((r, c))
            for r, _ in zeros:
                for c in range(start, end):
                    out[r][c] = 0
        return out

    return out"""

# ============ 8731374e ============
# Rectangle of one color (bg) embedded in noisy grid with a few anomaly (fg) cells.
# Output: rectangle extracted, anomaly positions generate full row+column lines of fg.
solutions["8731374e"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    from collections import Counter

    # Find all rectangular regions and pick the one with most uniform color
    # We need to find the rectangle dimensions first by trying all possible sizes
    # Actually, find the rectangle by looking for the largest rectangular area
    # where cells are predominantly one color

    # Strategy: try to find a rectangular region with high uniformity
    # For each possible color, find the largest rectangle of that color

    # Count all colors
    all_colors = Counter(grid[r][c] for r in range(rows) for c in range(cols))

    # Try each color as potential bg_color
    best_rect = None
    best_score = 0
    best_bg = None

    for bg_color in all_colors:
        # Find contiguous rows/cols where this color appears frequently
        # Scan for rectangular region
        for sr in range(rows):
            for sc in range(cols):
                if grid[sr][sc] != bg_color:
                    continue
                # Try expanding from this point
                for er in range(sr + 3, min(sr + 20, rows + 1)):
                    for ec in range(sc + 3, min(sc + 20, cols + 1)):
                        h = er - sr
                        w = ec - sc
                        count = sum(1 for r in range(sr, er) for c in range(sc, ec) if grid[r][c] == bg_color)
                        total = h * w
                        # Need high percentage (> 80%) and decent size
                        if count > total * 0.8 and count > best_score and total >= 20:
                            # Check that anomaly color is consistent
                            anomaly_colors = set()
                            for r in range(sr, er):
                                for c in range(sc, ec):
                                    if grid[r][c] != bg_color:
                                        anomaly_colors.add(grid[r][c])
                            if len(anomaly_colors) <= 1 and anomaly_colors:
                                best_score = count
                                best_rect = (sr, sc, er, ec)
                                best_bg = bg_color

    if best_rect is None:
        return grid

    sr, sc, er, ec = best_rect
    oh = er - sr
    ow = ec - sc
    bg = best_bg

    # Find anomalies
    anomalies = []
    fg = None
    for r in range(sr, er):
        for c in range(sc, ec):
            if grid[r][c] != bg:
                anomalies.append((r - sr, c - sc))
                fg = grid[r][c]

    if fg is None:
        return [[bg]*ow for _ in range(oh)]

    # Build output with row+col lines through anomaly positions
    output = [[bg]*ow for _ in range(oh)]
    for ar, ac in anomalies:
        for c in range(ow):
            output[ar][c] = fg
        for r in range(oh):
            output[r][ac] = fg

    return output"""

# ============ 8a004b2b ============
# 4 corner markers define rectangle. Colored blocks inside. Key pattern outside.
# Expand key pattern by block_size and place within frame.
solutions["8a004b2b"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find 4-corner positions
    fours = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 4]
    min_r4 = min(r for r, c in fours)
    max_r4 = max(r for r, c in fours)
    min_c4 = min(c for r, c in fours)
    max_c4 = max(c for r, c in fours)

    frame_h = max_r4 - min_r4 + 1
    frame_w = max_c4 - min_c4 + 1

    # Get colored blocks inside frame
    blocks = {}
    for r in range(min_r4, max_r4 + 1):
        for c in range(min_c4, max_c4 + 1):
            v = grid[r][c]
            if v not in (0, 4):
                if v not in blocks:
                    blocks[v] = []
                blocks[v].append((r - min_r4, c - min_c4))

    # Determine block_size
    block_size = 1
    for color, positions in blocks.items():
        min_br = min(r for r, c in positions)
        max_br = max(r for r, c in positions)
        block_size = max(block_size, max_br - min_br + 1)
        break

    # Get key pattern (non-0, non-4 cells outside frame)
    key_cells = []
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != 0 and v != 4 and (r < min_r4 or r > max_r4 or c < min_c4 or c > max_c4):
                key_cells.append((r, c, v))

    if not key_cells:
        return grid

    min_kr = min(r for r, c, v in key_cells)
    min_kc = min(c for r, c, v in key_cells)
    max_kr = max(r for r, c, v in key_cells)
    max_kc = max(c for r, c, v in key_cells)

    key_h = max_kr - min_kr + 1
    key_w = max_kc - min_kc + 1
    key_grid = [[0] * key_w for _ in range(key_h)]
    for r, c, v in key_cells:
        key_grid[r - min_kr][c - min_kc] = v

    # Find offset by matching first block position to key position
    first_color = list(blocks.keys())[0]
    first_pos = blocks[first_color][0]

    for kr in range(key_h):
        for kc in range(key_w):
            if key_grid[kr][kc] == first_color:
                offset_r = first_pos[0] - kr * block_size
                offset_c = first_pos[1] - kc * block_size

                # Build prediction with this offset
                pred = [[0] * frame_w for _ in range(frame_h)]
                pred[0][0] = 4
                pred[0][frame_w - 1] = 4
                pred[frame_h - 1][0] = 4
                pred[frame_h - 1][frame_w - 1] = 4

                for kr2 in range(key_h):
                    for kc2 in range(key_w):
                        if key_grid[kr2][kc2] != 0:
                            for dr in range(block_size):
                                for dc in range(block_size):
                                    pr = offset_r + kr2 * block_size + dr
                                    pc = offset_c + kc2 * block_size + dc
                                    if 0 <= pr < frame_h and 0 <= pc < frame_w:
                                        pred[pr][pc] = key_grid[kr2][kc2]

                # Verify corners match
                if (pred[0][0] == 4 and pred[0][frame_w-1] == 4 and
                    pred[frame_h-1][0] == 4 and pred[frame_h-1][frame_w-1] == 4):
                    return pred

    return grid"""

# ============ 8e1813be ============
# Grid has a block of 5s and colored lines. Output stacks colored lines compactly.
solutions["8e1813be"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find 5-block
    five_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 5]
    if not five_cells:
        return grid

    min_r5 = min(r for r, c in five_cells)
    max_r5 = max(r for r, c in five_cells)
    min_c5 = min(c for r, c in five_cells)
    max_c5 = max(c for r, c in five_cells)
    block_h = max_r5 - min_r5 + 1
    block_w = max_c5 - min_c5 + 1

    # Find all unique non-0, non-5 colors
    colors = set()
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v not in (0, 5):
                colors.add(v)

    all_lines = []

    for color in colors:
        positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == color]
        row_set = set(r for r, c in positions)
        col_set = set(c for r, c in positions)

        if len(row_set) <= 2:
            avg_row = sum(row_set) / len(row_set)
            all_lines.append(('h', avg_row, color))
        elif len(col_set) <= 2:
            avg_col = sum(col_set) / len(col_set)
            all_lines.append(('v', avg_col, color))

    all_lines.sort(key=lambda x: x[1])

    h_only = [c for d, p, c in all_lines if d == 'h']
    v_only = [c for d, p, c in all_lines if d == 'v']

    if v_only and not h_only:
        return [[v_only[c] for c in range(len(v_only))] for _ in range(block_h)]
    elif h_only and not v_only:
        return [[h_only[r]] * block_w for r in range(len(h_only))]
    else:
        return [[h_only[r]] * block_w for r in range(len(h_only))]"""

# ============ 8eb1be9a ============
# Pattern in middle of grid, tile it to fill entire grid.
solutions["8eb1be9a"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find non-zero rows
    nz_rows = [r for r in range(rows) if any(v != 0 for v in grid[r])]
    pattern = [grid[r] for r in nz_rows]
    ph = len(pattern)
    start = nz_rows[0]

    output = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        idx = (r - start) % ph
        output[r] = pattern[idx][:]

    return output"""

# ============ 8efcae92 ============
# Multiple rectangles of 1s with some 2s. Output the rectangle with most 2s.
solutions["8efcae92"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    visited = [[False] * cols for _ in range(rows)]
    rects = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                queue = [(r, c)]
                visited[r][c] = True
                cells = [(r, c)]
                while queue:
                    cr, cc = queue.pop(0)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != 0:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                            cells.append((nr, nc))

                min_r = min(r for r, c in cells)
                max_r = max(r for r, c in cells)
                min_c = min(c for r, c in cells)
                max_c = max(c for r, c in cells)

                rect = []
                for rr in range(min_r, max_r + 1):
                    row = []
                    for cc in range(min_c, max_c + 1):
                        row.append(grid[rr][cc])
                    rect.append(row)

                twos = sum(1 for rr in rect for v in rr if v == 2)
                rects.append((rect, twos))

    rects.sort(key=lambda x: x[1], reverse=True)
    return rects[0][0]"""

# ============ 890034e9 ============
# Bordered rectangle (border of one color, interior of 0s) exists in input.
# A copy is placed at another location where there's a matching all-0 interior.
solutions["890034e9"] = """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    from collections import Counter
    cnt = Counter(grid[r][c] for r in range(rows) for c in range(cols))

    for color in cnt:
        if color == 0:
            continue

        cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == color]
        if len(cells) < 8:
            continue

        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)
        h = max_r - min_r + 1
        w = max_c - min_c + 1

        if h < 3 or w < 3:
            continue

        is_border = True
        for c in range(min_c, max_c + 1):
            if grid[min_r][c] != color or grid[max_r][c] != color:
                is_border = False
                break
        if not is_border:
            continue

        for r in range(min_r, max_r + 1):
            if grid[r][min_c] != color or grid[r][max_c] != color:
                is_border = False
                break
        if not is_border:
            continue

        for r in range(min_r + 1, max_r):
            for c in range(min_c + 1, max_c):
                if grid[r][c] != 0:
                    is_border = False
                    break
            if not is_border:
                break
        if not is_border:
            continue

        rect_color = color
        rect_r, rect_c = min_r, min_c
        rect_h, rect_w = h, w

        output = [row[:] for row in grid]

        for sr in range(rows - rect_h + 1):
            for sc in range(cols - rect_w + 1):
                if sr == rect_r and sc == rect_c:
                    continue

                all_zero = True
                for r in range(sr + 1, sr + rect_h - 1):
                    for c in range(sc + 1, sc + rect_w - 1):
                        if grid[r][c] != 0:
                            all_zero = False
                            break
                    if not all_zero:
                        break

                if all_zero:
                    for r in range(rect_h):
                        for c in range(rect_w):
                            if grid[rect_r + r][rect_c + c] == rect_color:
                                output[sr + r][sc + c] = rect_color
                    return output

        return output

    return grid"""

# ============ Verify all solutions ============
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

# Save only passing solutions
passing = {}
for task_id, code in solutions.items():
    with open(f"data/arc1/{task_id}.json") as f:
        data = json.load(f)
    exec(code)
    all_pass = True
    for i, pair in enumerate(data["train"]):
        result = solve(pair["input"])
        if result != pair["output"]:
            all_pass = False
            break
    if all_pass:
        passing[task_id] = code

with open("data/arc_python_solutions_b30.json", "w") as f:
    json.dump(passing, f, indent=2)
print(f"Saved {len(passing)} solutions")
