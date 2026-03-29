from collections import Counter, defaultdict


def solve_90c28cc7(grid):
    """Grid of colored rectangles on black background -> color map."""
    rows = len(grid)
    cols = len(grid[0])

    min_r, max_r, min_c, max_c = rows, 0, cols, 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    row_boundaries = [min_r]
    for r in range(min_r + 1, max_r + 1):
        different = False
        for c in range(min_c, max_c + 1):
            if grid[r][c] != grid[r - 1][c]:
                different = True
                break
        if different:
            row_boundaries.append(r)
    row_boundaries.append(max_r + 1)

    col_boundaries = [min_c]
    for c in range(min_c + 1, max_c + 1):
        different = False
        for r in range(min_r, max_r + 1):
            if grid[r][c] != grid[r][c - 1]:
                different = True
                break
        if different:
            col_boundaries.append(c)
    col_boundaries.append(max_c + 1)

    out_rows = len(row_boundaries) - 1
    out_cols = len(col_boundaries) - 1
    result = []
    for i in range(out_rows):
        row = []
        for j in range(out_cols):
            r = (row_boundaries[i] + row_boundaries[i + 1]) // 2
            c = (col_boundaries[j] + col_boundaries[j + 1]) // 2
            row.append(grid[r][c])
        result.append(row)
    return result


def solve_9ecd008a(grid):
    """Grid with 4-fold symmetry and a 3x3 patch of 0s. Fill 0s using 180-degree rotation."""
    n = len(grid)
    zeros = []
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 0:
                zeros.append((r, c))

    filled = [row[:] for row in grid]
    for r, c in zeros:
        filled[r][c] = grid[n - 1 - r][n - 1 - c]

    min_r = min(r for r, c in zeros)
    min_c = min(c for r, c in zeros)
    max_r = max(r for r, c in zeros)
    max_c = max(c for r, c in zeros)

    patch = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            row.append(filled[r][c])
        patch.append(row)
    return patch


def solve_7837ac64(grid):
    """Grid with separator lines and colored intersections. Output uses 4-corner rule."""
    rows = len(grid)
    cols = len(grid[0])

    sep_color = None
    for r in range(rows):
        if len(set(grid[r])) == 1 and grid[r][0] != 0:
            sep_color = grid[r][0]
            break

    non_std = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and grid[r][c] != sep_color:
                non_std[(r, c)] = grid[r][c]

    row_lines = sorted(set(r for r, c in non_std))
    col_lines = sorted(set(c for r, c in non_std))

    out_rows = len(row_lines) - 1
    out_cols = len(col_lines) - 1
    result = [[0] * out_cols for _ in range(out_rows)]
    for i in range(out_rows):
        for j in range(out_cols):
            corners = [
                non_std.get((row_lines[i], col_lines[j]), 0),
                non_std.get((row_lines[i], col_lines[j + 1]), 0),
                non_std.get((row_lines[i + 1], col_lines[j]), 0),
                non_std.get((row_lines[i + 1], col_lines[j + 1]), 0),
            ]
            non_zero = [c for c in corners if c != 0]
            if len(non_zero) == 4 and len(set(non_zero)) == 1:
                result[i][j] = non_zero[0]

    return result


def solve_ce602527(grid):
    """Three shapes on grid: big shape, two small shapes. Output = small shape matching big orientation."""
    rows = len(grid)
    cols = len(grid[0])
    bg = grid[0][0]

    color_positions = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                color_positions[grid[r][c]].append((r, c))

    shapes = {}
    for color, positions in color_positions.items():
        min_r = min(r for r, c in positions)
        max_r = max(r for r, c in positions)
        min_c = min(c for r, c in positions)
        max_c = max(c for r, c in positions)
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        pattern = []
        for r in range(min_r, max_r + 1):
            row = []
            for c in range(min_c, max_c + 1):
                row.append(1 if grid[r][c] == color else 0)
            pattern.append(row)
        shapes[color] = {"h": h, "w": w, "pattern": pattern, "bbox": h * w}

    sorted_by_bbox = sorted(shapes.items(), key=lambda x: x[1]["bbox"], reverse=True)
    big_color = sorted_by_bbox[0][0]
    big = shapes[big_color]
    small_shapes = sorted_by_bbox[1:]

    big_orient = 1 if big["h"] > big["w"] else (-1 if big["w"] > big["h"] else 0)

    candidates = []
    for sc, ss in small_shapes:
        orient = 1 if ss["h"] > ss["w"] else (-1 if ss["w"] > ss["h"] else 0)
        orient_match = orient == big_orient
        candidates.append((sc, ss, orient_match))

    matching = [c for c in candidates if c[2]]

    if len(matching) == 1:
        chosen_color = matching[0][0]
        chosen = matching[0][1]
    else:
        best = None
        best_score = -1
        for sc, ss, _ in candidates:
            score = sum(ss["pattern"][0])
            if score > best_score:
                best_score = score
                best = (sc, ss)
        chosen_color = best[0]
        chosen = best[1]

    result = []
    for r in range(chosen["h"]):
        row = []
        for c in range(chosen["w"]):
            row.append(chosen_color if chosen["pattern"][r][c] == 1 else bg)
        result.append(row)

    return result


def solve_1190e5a7(grid):
    """Grid divided by separator lines. Output = grid of bg color sized by number of row/col regions."""
    rows = len(grid)
    cols = len(grid[0])

    sep = None
    bg = None
    for r in range(rows):
        vals = set(grid[r])
        if len(vals) == 1:
            sep = grid[r][0]
            break
    if sep is None:
        for c in range(cols):
            vals = set(grid[r][c] for r in range(rows))
            if len(vals) == 1:
                sep = list(vals)[0]
                break

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != sep:
                bg = grid[r][c]
                break
        if bg is not None:
            break

    row_regions = 0
    in_region = False
    for r in range(rows):
        is_sep = all(grid[r][c] == sep for c in range(cols))
        if not is_sep:
            if not in_region:
                row_regions += 1
                in_region = True
        else:
            in_region = False

    col_regions = 0
    in_region = False
    for c in range(cols):
        is_sep = all(grid[r][c] == sep for r in range(rows))
        if not is_sep:
            if not in_region:
                col_regions += 1
                in_region = True
        else:
            in_region = False

    return [[bg] * col_regions for _ in range(row_regions)]


def solve_6ecd11f4(grid):
    """Large shape of blocks + small key grid. Output = key masked by block pattern."""
    rows = len(grid)
    cols = len(grid[0])

    color_count = Counter()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color_count[grid[r][c]] += 1

    shape_color = color_count.most_common(1)[0][0]

    key_cells = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and grid[r][c] != shape_color:
                key_cells[(r, c)] = grid[r][c]

    key_min_r = min(r for r, c in key_cells)
    key_max_r = max(r for r, c in key_cells)
    key_min_c = min(c for r, c in key_cells)
    key_max_c = max(c for r, c in key_cells)

    for r in range(key_min_r, key_max_r + 1):
        for c in range(key_min_c, key_max_c + 1):
            if grid[r][c] == shape_color:
                key_cells[(r, c)] = shape_color

    key_h = key_max_r - key_min_r + 1
    key_w = key_max_c - key_min_c + 1

    key_grid = [[0] * key_w for _ in range(key_h)]
    for (r, c), v in key_cells.items():
        key_grid[r - key_min_r][c - key_min_c] = v

    shape_cells = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == shape_color and not (
                key_min_r <= r <= key_max_r and key_min_c <= c <= key_max_c
            ):
                shape_cells.add((r, c))

    if not shape_cells:
        return key_grid

    shape_min_r = min(r for r, c in shape_cells)
    shape_max_r = max(r for r, c in shape_cells)
    shape_min_c = min(c for r, c in shape_cells)
    shape_max_c = max(c for r, c in shape_cells)
    shape_h = shape_max_r - shape_min_r + 1
    shape_w = shape_max_c - shape_min_c + 1

    bh = shape_h // key_h
    bw = shape_w // key_w

    block_grid = [[0] * key_w for _ in range(key_h)]
    for bi in range(key_h):
        for bj in range(key_w):
            br_start = shape_min_r + bi * bh
            bc_start = shape_min_c + bj * bw
            filled = True
            for dr in range(bh):
                for dc in range(bw):
                    if (br_start + dr, bc_start + dc) not in shape_cells:
                        filled = False
                        break
                if not filled:
                    break
            block_grid[bi][bj] = 1 if filled else 0

    result = [[0] * key_w for _ in range(key_h)]
    for r in range(key_h):
        for c in range(key_w):
            if block_grid[r][c] == 1:
                result[r][c] = key_grid[r][c]

    return result


def solve_9f236235(grid):
    """Grid divided by separator lines into colored cells. Output = cell color grid, mirrored horizontally."""
    rows = len(grid)
    cols = len(grid[0])

    sep_color = None
    for r in range(rows):
        if len(set(grid[r])) == 1:
            sep_color = grid[r][0]
            break
    if sep_color is None:
        for c in range(cols):
            col_vals = set(grid[r][c] for r in range(rows))
            if len(col_vals) == 1:
                sep_color = grid[0][c]
                break

    sep_rows = [
        r for r in range(rows) if all(grid[r][c] == sep_color for c in range(cols))
    ]
    sep_cols = [
        c for c in range(cols) if all(grid[r][c] == sep_color for r in range(rows))
    ]

    row_bands = []
    prev = 0
    for sr in sep_rows:
        if sr > prev:
            row_bands.append((prev, sr - 1))
        prev = sr + 1
    if prev < rows:
        row_bands.append((prev, rows - 1))

    col_bands = []
    prev = 0
    for sc in sep_cols:
        if sc > prev:
            col_bands.append((prev, sc - 1))
        prev = sc + 1
    if prev < cols:
        col_bands.append((prev, cols - 1))

    result = []
    for rb_start, rb_end in row_bands:
        row = []
        for cb_start, cb_end in reversed(col_bands):
            color = grid[rb_start][cb_start]
            row.append(color)
        result.append(row)

    return result


def solve_28bf18c6(grid):
    """Extract non-zero shape and duplicate it horizontally."""
    rows = len(grid)
    cols = len(grid[0])

    min_r, max_r, min_c, max_c = rows, 0, cols, 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    pattern = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            row.append(grid[r][c])
        pattern.append(row)

    result = []
    for row in pattern:
        result.append(row + row)

    return result


def solve_49d1d64f(grid):
    """Add border: top/bottom rows are [0]+row+[0], middle rows have first/last col duplicated."""
    h = len(grid)

    result = []
    result.append([0] + grid[0] + [0])
    for r in range(h):
        result.append([grid[r][0]] + grid[r] + [grid[r][-1]])
    result.append([0] + grid[-1] + [0])

    return result


def solve_1f85a75f(grid):
    """Find the small cluster of a unique color in a noisy grid, extract its pattern."""
    rows = len(grid)
    cols = len(grid[0])

    color_count = Counter()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color_count[grid[r][c]] += 1

    sorted_colors = color_count.most_common()

    for color, count in reversed(sorted_colors):
        positions = [
            (r, c) for r in range(rows) for c in range(cols) if grid[r][c] == color
        ]
        if not positions:
            continue

        min_r = min(r for r, c in positions)
        max_r = max(r for r, c in positions)
        min_c = min(c for r, c in positions)
        max_c = max(c for r, c in positions)
        h = max_r - min_r + 1
        w = max_c - min_c + 1

        if h <= 10 and w <= 10:
            result = []
            for r in range(min_r, max_r + 1):
                row = []
                for c in range(min_c, max_c + 1):
                    row.append(color if grid[r][c] == color else 0)
                result.append(row)
            return result

    return [[0]]


CUSTOM_SOLVERS_B = {
    "90c28cc7": solve_90c28cc7,
    "9ecd008a": solve_9ecd008a,
    "7837ac64": solve_7837ac64,
    "ce602527": solve_ce602527,
    "1190e5a7": solve_1190e5a7,
    "6ecd11f4": solve_6ecd11f4,
    "9f236235": solve_9f236235,
    "28bf18c6": solve_28bf18c6,
    "49d1d64f": solve_49d1d64f,
    "1f85a75f": solve_1f85a75f,
}
