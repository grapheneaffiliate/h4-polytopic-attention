#!/usr/bin/env python3
import json
import copy
from collections import Counter, defaultdict, deque

def solve_e1baa8a4(grid):
    rows, cols = len(grid), len(grid[0])
    row_bands = [0]
    for r in range(1, rows):
        if grid[r][0] != grid[r-1][0]:
            row_bands.append(r)
    col_bands = [0]
    for c in range(1, cols):
        if grid[0][c] != grid[0][c-1]:
            col_bands.append(c)
    out_rows = len(row_bands)
    out_cols = len(col_bands)
    result = [[0]*out_cols for _ in range(out_rows)]
    for ri, rb in enumerate(row_bands):
        for ci, cb in enumerate(col_bands):
            result[ri][ci] = grid[rb][cb]
    return result

def solve_e633a9e5(grid):
    out = [[0]*5 for _ in range(5)]
    row_ranges = [(0,2),(2,3),(3,5)]
    col_ranges = [(0,2),(2,3),(3,5)]
    for r in range(3):
        for c in range(3):
            val = grid[r][c]
            r_start, r_end = row_ranges[r]
            c_start, c_end = col_ranges[c]
            for rr in range(r_start, r_end):
                for cc in range(c_start, c_end):
                    out[rr][cc] = val
    return out

def solve_e133d23d(grid):
    out = [[0]*3 for _ in range(3)]
    for r in range(3):
        for c in range(3):
            left = grid[r][c]
            right = grid[r][c+4]
            if left != 0 or right != 0:
                out[r][c] = 2
    return out

def solve_e345f17b(grid):
    rows = len(grid)
    cols = len(grid[0]) // 2
    out = [[0]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and grid[r][c+cols] == 0:
                out[r][c] = 4
    return out

def solve_e3fe1151(grid):
    rows, cols = len(grid), len(grid[0])
    cr, cc = rows//2, cols//2
    # Collect non-7 values from all quadrants
    quads = [
        (range(0,cr), range(0,cc)),
        (range(0,cr), range(cc+1,cols)),
        (range(cr+1,rows), range(0,cc)),
        (range(cr+1,rows), range(cc+1,cols))
    ]
    all_non7 = []
    quad_info = []
    for rr, ccr in quads:
        vals = []
        seven_pos = None
        for r in rr:
            for c in ccr:
                if grid[r][c] == 7:
                    seven_pos = (r,c)
                else:
                    vals.append(grid[r][c])
                    all_non7.append(grid[r][c])
        quad_info.append((vals, seven_pos))

    # Target multiset: total of 16 values (4 per quad), 12 known + 4 unknown
    # All quads must have the same multiset
    # Try: for TL quad, find the replacement that makes the multiset work for all
    out = [row[:] for row in grid]

    # Get possible target multisets: TL has 3 known vals + 1 unknown
    tl_vals = quad_info[0][0]
    # Try each possible value (from the set of all non-7 values)
    candidates = set(all_non7)
    for v in candidates:
        target = sorted(tl_vals + [v])
        # Check if all other quads can match this target
        ok = True
        replacements = {}
        for i, (vals, pos) in enumerate(quad_info):
            needed = list(target)
            for val in vals:
                if val in needed:
                    needed.remove(val)
                else:
                    ok = False
                    break
            if not ok:
                break
            if len(needed) != 1:
                ok = False
                break
            replacements[i] = needed[0]
        if ok:
            for i, (vals, pos) in enumerate(quad_info):
                if pos and i in replacements:
                    out[pos[0]][pos[1]] = replacements[i]
            return out
    return out

def solve_e57337a4(grid):
    rows, cols = len(grid), len(grid[0])
    bg = grid[0][0]
    out = [[bg]*3 for _ in range(3)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                br = r // 5
                bc = c // 5
                out[br][bc] = 0
    return out

def solve_e21a174a(grid):
    rows, cols = len(grid), len(grid[0])
    groups = []
    i = 0
    while i < rows:
        row_colors = set(grid[i][c] for c in range(cols)) - {0}
        if not row_colors:
            i += 1
            continue
        color = max(row_colors, key=lambda v: sum(1 for c in range(cols) if grid[i][c] == v))
        group_rows = []
        while i < rows:
            rc = set(grid[i][c] for c in range(cols)) - {0}
            if not rc:
                break
            c2 = max(rc, key=lambda v: sum(1 for c in range(cols) if grid[i][c] == v))
            if c2 != color:
                break
            group_rows.append(grid[i][:])
            i += 1
        groups.append(group_rows)
    groups.reverse()
    out = [[0]*cols for _ in range(rows)]
    row_idx = 0
    for r in range(rows):
        rc = set(grid[r][c] for c in range(cols)) - {0}
        if rc:
            break
        row_idx = r + 1
    curr = row_idx
    for group in groups:
        for row_data in group:
            out[curr] = row_data[:]
            curr += 1
    return out

def solve_e74e1818(grid):
    rows_n, cols_n = len(grid), len(grid[0])
    groups = []
    i = 0
    while i < rows_n:
        row_colors = set(grid[i][c] for c in range(cols_n)) - {0}
        if not row_colors:
            i += 1
            continue
        color = max(row_colors, key=lambda v: sum(1 for c in range(cols_n) if grid[i][c] == v))
        group_rows = []
        while i < rows_n:
            rc = set(grid[i][c] for c in range(cols_n)) - {0}
            if not rc:
                break
            c2 = max(rc, key=lambda v: sum(1 for c in range(cols_n) if grid[i][c] == v))
            if c2 != color:
                break
            group_rows.append(i)
            i += 1
        groups.append((color, group_rows))
    out = [row[:] for row in grid]
    for color, group_rows in groups:
        reversed_rows = group_rows[::-1]
        for orig, rev in zip(group_rows, reversed_rows):
            out[orig] = grid[rev][:]
    return out

def solve_da2b0fe3(grid):
    rows, cols = len(grid), len(grid[0])
    nonzero = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] != 0]
    if not nonzero:
        return grid
    min_r = min(r for r,c in nonzero)
    max_r = max(r for r,c in nonzero)
    min_c = min(c for r,c in nonzero)
    max_c = max(c for r,c in nonzero)
    out = [row[:] for row in grid]
    for r in range(min_r, max_r + 1):
        if all(grid[r][c] == 0 for c in range(min_c, max_c + 1)):
            for c in range(cols):
                out[r][c] = 3
            return out
    for c in range(min_c, max_c + 1):
        if all(grid[r][c] == 0 for r in range(min_r, max_r + 1)):
            for r in range(rows):
                out[r][c] = 3
            return out
    return out

def solve_e0fb7511(grid):
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    zeros = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                zeros.add((r, c))
    pairs = set()
    for r, c in zeros:
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if (nr, nc) in zeros:
                pairs.add((r, c))
                pairs.add((nr, nc))
    for r, c in pairs:
        out[r][c] = 8
    return out

def solve_dc1df850(grid):
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if out[nr][nc] == 0:
                                out[nr][nc] = 1
    return out

def solve_e7dd8335(grid):
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    ones = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    if not ones:
        return out
    min_r = min(r for r,c in ones)
    max_r = max(r for r,c in ones)
    mid_r = (min_r + max_r) / 2
    for r, c in ones:
        if r > mid_r:
            out[r][c] = 2
    return out

def solve_e7a25a18(grid):
    rows, cols = len(grid), len(grid[0])
    twos = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 2]
    if not twos:
        return grid
    min_r = min(r for r,c in twos)
    max_r = max(r for r,c in twos)
    min_c = min(c for r,c in twos)
    max_c = max(c for r,c in twos)
    int_r1, int_r2 = min_r + 1, max_r - 1
    int_c1, int_c2 = min_c + 1, max_c - 1
    int_h = int_r2 - int_r1 + 1
    int_w = int_c2 - int_c1 + 1
    blocks = {}
    for r in range(int_r1, int_r2 + 1):
        for c in range(int_c1, int_c2 + 1):
            v = grid[r][c]
            if v != 0 and v != 2:
                if v not in blocks:
                    blocks[v] = []
                blocks[v].append((r, c))
    all_color_cells = [(r,c) for color, cells in blocks.items() for r,c in cells]
    if not all_color_cells:
        return grid
    pat_r1 = min(r for r,c in all_color_cells)
    pat_r2 = max(r for r,c in all_color_cells)
    pat_c1 = min(c for r,c in all_color_cells)
    pat_c2 = max(c for r,c in all_color_cells)
    pat_h = pat_r2 - pat_r1 + 1
    pat_w = pat_c2 - pat_c1 + 1
    scale_r = int_h / pat_h
    scale_c = int_w / pat_w
    out_h = int(pat_h * scale_r)
    out_w = int(pat_w * scale_c)
    result = [[2] * (out_w + 2) for _ in range(out_h + 2)]
    for r in range(pat_h):
        for c in range(pat_w):
            src_r = pat_r1 + r
            src_c = pat_c1 + c
            color = grid[src_r][src_c]
            if color == 0:
                color = 0
                for cc, cells in blocks.items():
                    if (src_r, src_c) in [(rr,ccc) for rr,ccc in cells]:
                        color = cc
                        break
            for dr in range(int(scale_r)):
                for dc in range(int(scale_c)):
                    result[1 + int(r * scale_r) + dr][1 + int(c * scale_c) + dc] = color
    for c in range(out_w + 2):
        result[0][c] = 2
        result[out_h + 1][c] = 2
    for r in range(out_h + 2):
        result[r][0] = 2
        result[r][out_w + 1] = 2
    return result

def solve_e7b06bea(grid):
    rows, cols = len(grid), len(grid[0])
    # Find the 5-marker rows
    five_rows = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                five_rows.add(r)
    five_count = len(five_rows)

    # Find pattern digits from right side
    right_cols = []
    for c in range(cols-1, -1, -1):
        vals = set(grid[r][c] for r in range(rows)) - {0, 5}
        if vals:
            right_cols.append(c)
        else:
            break
    right_cols.sort()
    if not right_cols:
        return grid

    sample_row = 0
    for r in range(rows):
        if all(grid[r][c] != 0 for c in right_cols):
            sample_row = r
            break

    pattern = [grid[sample_row][c] for c in right_cols]
    out_col = right_cols[0] - 1
    seg_len = five_count

    out = [row[:] for row in grid]
    for r in range(rows):
        for c in right_cols:
            out[r][c] = 0

    pat_idx = 0
    for start_r in range(0, rows, seg_len):
        if pat_idx >= len(pattern):
            pat_idx = 0
        digit = pattern[pat_idx]
        for r in range(start_r, min(start_r + seg_len, rows)):
            out[r][out_col] = digit
        pat_idx += 1

    return out

def solve_ded97339(grid):
    """Pairs of 8s sharing row/col: connect all such pairs with lines of 8."""
    rows, cols = len(grid), len(grid[0])
    eights = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8:
                eights.append((r, c))
    out = [row[:] for row in grid]
    # Connect all pairs sharing a row or column
    for i in range(len(eights)):
        for j in range(i+1, len(eights)):
            r1, c1 = eights[i]
            r2, c2 = eights[j]
            if r1 == r2:
                for c in range(min(c1,c2), max(c1,c2)+1):
                    out[r1][c] = 8
            elif c1 == c2:
                for r in range(min(r1,r2), max(r1,r2)+1):
                    out[r][c1] = 8
    return out

def solve_e4075551(grid):
    """Four colored dots at edges + one interior dot. Draw rectangle with border colors and cross of 5s."""
    rows, cols = len(grid), len(grid[0])
    dots = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                dots.append((r, c, grid[r][c]))
    if len(dots) != 5:
        return grid

    rs = [r for r,c,v in dots]
    cs = [c for r,c,v in dots]
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)

    dot_map = {(r,c): v for r,c,v in dots}

    # Find edge dots (on boundary of bounding box) and interior dot
    interior_dot = None
    top_color = bot_color = left_color = right_color = None

    for r,c,v in dots:
        if r == min_r: top_color = v
        if r == max_r: bot_color = v
        if c == min_c: left_color = v
        if c == max_c: right_color = v

    for r,c,v in dots:
        if r != min_r and r != max_r and c != min_c and c != max_c:
            interior_dot = (r,c,v)

    if not interior_dot or not all([top_color, bot_color, left_color, right_color]):
        return grid

    out = [[0]*cols for _ in range(rows)]
    # Draw left/right edges first
    for r in range(min_r + 1, max_r):
        out[r][min_c] = left_color
        out[r][max_c] = right_color
    # Then top/bottom edges (overwrite corners with row color)
    for c in range(min_c, max_c + 1):
        out[min_r][c] = top_color
        out[max_r][c] = bot_color

    ir, ic = interior_dot[0], interior_dot[1]
    for c in range(min_c + 1, max_c):
        if out[ir][c] == 0:
            out[ir][c] = 5
    for r in range(min_r + 1, max_r):
        if out[r][ic] == 0:
            out[r][ic] = 5
    out[ir][ic] = interior_dot[2]

    return out

def solve_e4941b18(grid):
    """5-rectangle with 2 and 8 markers adjacent. 2 goes to 8's pos, 8 goes to opposite corner of 5-rect +1 outward."""
    rows, cols = len(grid), len(grid[0])
    two_pos = eight_pos = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2: two_pos = (r,c)
            elif grid[r][c] == 8: eight_pos = (r,c)
    if not two_pos or not eight_pos:
        return grid

    fives = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 5]
    if not fives:
        return grid
    five_min_r = min(r for r,c in fives)
    five_max_r = max(r for r,c in fives)
    five_min_c = min(c for r,c in fives)
    five_max_c = max(c for r,c in fives)

    out = [row[:] for row in grid]
    out[two_pos[0]][two_pos[1]] = 7
    out[eight_pos[0]][eight_pos[1]] = 2

    # Find which corner of the 5-rect the 2 is adjacent to
    # 2 is near the top edge if two_pos[0] <= five_min_r, or bottom if >= five_max_r
    # 2 is near left if two_pos[1] <= five_min_c, or right if >= five_max_c
    near_top = two_pos[0] <= five_min_r
    near_left = two_pos[1] <= five_min_c + (five_max_c - five_min_c) / 2

    # Opposite corner
    if near_top:
        opp_r = five_max_r
    else:
        opp_r = five_min_r
    if near_left:
        opp_c = five_max_c + 1  # +1 outward to the right
    else:
        opp_c = five_min_c - 1  # -1 outward to the left

    if 0 <= opp_r < rows and 0 <= opp_c < cols:
        out[opp_r][opp_c] = 8

    return out

def solve_dce56571(grid):
    """Non-8 shape: find weighted center row/col and draw full line."""
    rows, cols = len(grid), len(grid[0])
    non8 = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 8:
                non8.append((r, c, grid[r][c]))
    if not non8:
        return grid
    color = non8[0][2]
    rs = [r for r,c,v in non8]
    cs = [c for r,c,v in non8]
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)
    r_span = max_r - min_r
    c_span = max_c - min_c

    # Find the center of mass
    avg_r = sum(rs) / len(rs)
    avg_c = sum(cs) / len(cs)

    out = [[8]*cols for _ in range(rows)]

    # Draw line at the integer closest to center of mass
    if c_span >= r_span:
        # Wider shape -> draw horizontal line
        fill_r = round(avg_r)
        for c in range(cols):
            out[fill_r][c] = color
    else:
        fill_c = round(avg_c)
        for r in range(rows):
            out[r][fill_c] = color

    return out

def solve_e7639916(grid):
    """Multiple 8s on grid -> draw bounding rectangle with 1-borders, 8s stay."""
    rows, cols = len(grid), len(grid[0])
    eights = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8:
                eights.append((r, c))
    if len(eights) < 2:
        return grid

    out = [row[:] for row in grid]
    min_r = min(r for r,c in eights)
    max_r = max(r for r,c in eights)
    min_c = min(c for r,c in eights)
    max_c = max(c for r,c in eights)

    # Draw rectangle outline with 1s
    for c in range(min_c, max_c + 1):
        if out[min_r][c] == 0: out[min_r][c] = 1
        if out[max_r][c] == 0: out[max_r][c] = 1
    for r in range(min_r, max_r + 1):
        if out[r][min_c] == 0: out[r][min_c] = 1
        if out[r][max_c] == 0: out[r][max_c] = 1

    return out

def solve_db7260a4(grid):
    """U-shape of 2s with a 1 dot. If 1 is above the U opening, fill interior with 1s. Otherwise fill a border row."""
    rows, cols = len(grid), len(grid[0])
    one_r = one_c = -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                one_r, one_c = r, c

    twos = set((r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 2)
    if not twos:
        return grid

    min_r2 = min(r for r,c in twos)
    max_r2 = max(r for r,c in twos)
    min_c2 = min(c for r,c in twos)
    max_c2 = max(c for r,c in twos)

    out = [row[:] for row in grid]
    out[one_r][one_c] = 0

    # Find interior cells of the U (0-cells within bounding box not part of 2-border)
    interior = set()
    for r in range(min_r2, max_r2+1):
        for c in range(min_c2, max_c2+1):
            if (r,c) not in twos:
                interior.add((r,c))

    # Check if the 1's column/row intersects with interior cells
    col_interior = [(r,c) for r,c in interior if c == one_c]
    row_interior = [(r,c) for r,c in interior if r == one_r]

    if col_interior and one_r < min_r2:
        # 1 above, column hits interior -> fill interior
        for r,c in interior:
            out[r][c] = 1
    elif col_interior and one_r > max_r2:
        for r,c in interior:
            out[r][c] = 1
    elif row_interior and one_c < min_c2:
        for r,c in interior:
            out[r][c] = 1
    elif row_interior and one_c > max_c2:
        for r,c in interior:
            out[r][c] = 1
    else:
        # 1 doesn't enter the U -> fill entire border row/col
        if one_r <= min_r2:
            for c in range(cols):
                out[rows-1][c] = 1
        elif one_r >= max_r2:
            for c in range(cols):
                out[0][c] = 1
        elif one_c <= min_c2:
            for r in range(rows):
                out[r][cols-1] = 1
        else:
            for r in range(rows):
                out[r][0] = 1
    return out

def solve_d968ffd4(grid):
    """Two colored rectangles on background. Expand toward each other filling gap."""
    rows, cols = len(grid), len(grid[0])
    bg = grid[0][0]

    # Find colored rectangles
    visited = [[False]*cols for _ in range(rows)]
    rects = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                color = grid[r][c]
                stack = [(r,c)]
                cells = []
                while stack:
                    cr2, cc2 = stack.pop()
                    if cr2<0 or cr2>=rows or cc2<0 or cc2>=cols: continue
                    if visited[cr2][cc2] or grid[cr2][cc2] != color: continue
                    visited[cr2][cc2] = True
                    cells.append((cr2, cc2))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        stack.append((cr2+dr, cc2+dc))
                mr = min(r2 for r2,c2 in cells)
                xr = max(r2 for r2,c2 in cells)
                mc = min(c2 for r2,c2 in cells)
                xc = max(c2 for r2,c2 in cells)
                rects.append((color, mr, xr, mc, xc))

    if len(rects) != 2:
        return grid

    out = [row[:] for row in grid]
    c1, mr1, xr1, mc1, xc1 = rects[0]
    c2, mr2, xr2, mc2, xc2 = rects[1]

    h1, w1 = xr1-mr1+1, xc1-mc1+1
    h2, w2 = xr2-mr2+1, xc2-mc2+1

    # Determine if gap is horizontal or vertical
    # Horizontal gap: rects side by side
    r_overlap = not (xr1 < mr2 or xr2 < mr1)
    c_overlap = not (xc1 < mc2 or xc2 < mc1)

    if r_overlap and not c_overlap:
        # Horizontal gap
        if mc1 < mc2:
            left_c, left_mr, left_xr, left_mc, left_xc = c1, mr1, xr1, mc1, xc1
            right_c, right_mr, right_xr, right_mc, right_xc = c2, mr2, xr2, mc2, xc2
            left_w, left_h = w1, h1
            right_w, right_h = w2, h2
        else:
            left_c, left_mr, left_xr, left_mc, left_xc = c2, mr2, xr2, mc2, xc2
            right_c, right_mr, right_xr, right_mc, right_xc = c1, mr1, xr1, mc1, xc1
            left_w, left_h = w2, h2
            right_w, right_h = w1, h1

        gap = right_mc - left_xc - 1
        if gap % 2 == 0:
            left_expand = gap // 2
            right_expand = gap // 2
        else:
            left_expand = gap // 2
            right_expand = gap // 2
            # middle column stays bg

        # Expand left rect rightward
        for r in range(left_mr, left_xr+1):
            for c in range(left_xc+1, left_xc+1+left_expand):
                if c < cols:
                    out[r][c] = left_c
        # Expand right rect leftward
        for r in range(right_mr, right_xr+1):
            for c in range(right_mc-right_expand, right_mc):
                if c >= 0:
                    out[r][c] = right_c

        # Taper: expand above and below by 1 row, reducing width by rect_width
        for d in range(1, max(rows, cols)):
            any_placed = False
            for side in ['left', 'right']:
                if side == 'left':
                    core_mr, core_xr = left_mr, left_xr
                    core_mc, core_xc = left_mc, left_xc + left_expand
                    orig_w = left_w
                    color = left_c
                else:
                    core_mr, core_xr = right_mr, right_xr
                    core_mc, core_xc = right_mc - right_expand, right_xc
                    orig_w = right_w
                    color = right_c

                # Above
                r_above = core_mr - d
                if r_above >= 0:
                    taper_mc = core_mc + d * orig_w
                    taper_xc = core_xc - d * orig_w
                    if taper_mc <= taper_xc:
                        for c in range(taper_mc, taper_xc+1):
                            if 0 <= c < cols:
                                out[r_above][c] = color
                                any_placed = True
                # Below
                r_below = core_xr + d
                if r_below < rows:
                    taper_mc = core_mc + d * orig_w
                    taper_xc = core_xc - d * orig_w
                    if taper_mc <= taper_xc:
                        for c in range(taper_mc, taper_xc+1):
                            if 0 <= c < cols:
                                out[r_below][c] = color
                                any_placed = True
            if not any_placed:
                break

    elif c_overlap and not r_overlap:
        # Vertical gap
        if mr1 < mr2:
            top_c, top_mr, top_xr, top_mc, top_xc = c1, mr1, xr1, mc1, xc1
            bot_c, bot_mr, bot_xr, bot_mc, bot_xc = c2, mr2, xr2, mc2, xc2
            top_w, top_h = w1, h1
            bot_w, bot_h = w2, h2
        else:
            top_c, top_mr, top_xr, top_mc, top_xc = c2, mr2, xr2, mc2, xc2
            bot_c, bot_mr, bot_xr, bot_mc, bot_xc = c1, mr1, xr1, mc1, xc1
            top_w, top_h = w2, h2
            bot_w, bot_h = w1, h1

        gap = bot_mr - top_xr - 1
        top_expand = gap // 2
        bot_expand = gap // 2

        for c in range(top_mc, top_xc+1):
            for r in range(top_xr+1, top_xr+1+top_expand):
                if r < rows:
                    out[r][c] = top_c
        for c in range(bot_mc, bot_xc+1):
            for r in range(bot_mr-bot_expand, bot_mr):
                if r >= 0:
                    out[r][c] = bot_c

        for d in range(1, max(rows, cols)):
            any_placed = False
            for side in ['top', 'bot']:
                if side == 'top':
                    core_mr, core_xr = top_mr, top_xr + top_expand
                    core_mc, core_xc = top_mc, top_xc
                    orig_h = top_h
                    color = top_c
                else:
                    core_mr, core_xr = bot_mr - bot_expand, bot_xr
                    core_mc, core_xc = bot_mc, bot_xc
                    orig_h = bot_h
                    color = bot_c

                c_left = core_mc - d
                if c_left >= 0:
                    taper_mr = core_mr + d * orig_h
                    taper_xr = core_xr - d * orig_h
                    if taper_mr <= taper_xr:
                        for r in range(taper_mr, taper_xr+1):
                            if 0 <= r < rows:
                                out[r][c_left] = color
                                any_placed = True
                c_right = core_xc + d
                if c_right < cols:
                    taper_mr = core_mr + d * orig_h
                    taper_xr = core_xr - d * orig_h
                    if taper_mr <= taper_xr:
                        for r in range(taper_mr, taper_xr+1):
                            if 0 <= r < rows:
                                out[r][c_right] = color
                                any_placed = True
            if not any_placed:
                break

    return out

def solve_e41c6fd3(grid):
    """Shapes with same structure at different heights -> align bottoms to same row."""
    rows, cols = len(grid), len(grid[0])
    visited = [[False]*cols for _ in range(rows)]
    shapes = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                stack = [(r,c)]
                cells = []
                while stack:
                    cr2, cc2 = stack.pop()
                    if cr2<0 or cr2>=rows or cc2<0 or cc2>=cols: continue
                    if visited[cr2][cc2] or grid[cr2][cc2] == 0: continue
                    visited[cr2][cc2] = True
                    cells.append((cr2, cc2, grid[cr2][cc2]))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        stack.append((cr2+dr, cc2+dc))
                shapes.append(cells)

    if not shapes:
        return grid

    shape_info = []
    for shape in shapes:
        min_r = min(r for r,c,v in shape)
        max_r = max(r for r,c,v in shape)
        h = max_r - min_r + 1
        shape_info.append((shape, min_r, max_r, h))

    # All shapes have same height. Target: align all bottoms.
    # The target bottom is the bottom of the shape with the maximum bottom row,
    # BUT constrained so all shapes fit within the grid.
    shape_h = shape_info[0][3]
    # Target bottom: maximize so all fit. The limiting factor is: target_bottom - shape_h + 1 >= 0
    # and target_bottom < rows. So: target_bottom = rows - 1 may push shapes above grid.
    # Choose: target_bottom such that all shapes align nicely.
    # From examples: target = bottom of the bottommost shape that fits within grid.
    # Actually simplest: target = the bottom of the second shape (which is typically already
    # at the right position).
    # But from train 0: shapes at rows 2-5, 6-9, 8-11, 10-13.
    # Output: all at 6-9. Target bottom = 9.
    # That's the bottom of shape 1 (which is the earliest shape at bottom half).

    # Actually: the target bottom = rows - 1 - (shape_h - 1) might work for some.
    # rows=14, shape_h=4. rows-1=13, 13-3=10. Not 9.

    # Looking at train 1: shapes at rows 1-5, 2-6, 3-7, 7-11.
    # All have different heights! 1:5rows, 2:5rows, 3:5rows, 4:5rows.
    # Output all at rows 1-5. target_bottom=5.

    # From train 2: shapes all shift down too.
    # Maybe: target_bottom = bottom of the shape that already has the lowest position
    # minus some factor.

    # Simplest: all shapes align to the bottom row of the lowest shape.
    # But train 0: lowest shape bottom = 13, but output target = 9.

    # WAIT: let me recheck. In train 0 output, shapes are at rows 6-9.
    # In train 1 output (13x23),
    # In train 2 output (16x30), shapes align at rows 2-5.
    # Shape positions vary widely.

    # Let me just look: in each case, the shapes get "gravity" dropped to the bottom,
    # with the constraint that they don't go below the grid.
    # Or: they all align to the position of the tallest shape.

    # ACTUALLY: they all align to the BOTTOM of the grid minus rows needed.
    # Train 0: rows=14. Shapes 4 rows tall. Bottom of last content = row 9.
    # But rows-1=13, and 13-4+1=10. That doesn't give 9.

    # Maybe they align to the row where all shapes' bottoms coincide minimally.
    # The shape at the lowest position determines the target, and all others move to match.
    # But shape at rows 10-13 would need to move UP (to row 9). Shapes moving up is unusual.

    # Actually I realize: ALL shapes in the output end at the SAME row.
    # For train 0: all end at row 9.
    # bottom_rows = [5, 9, 11, 13]. median = 10. Not 9.
    # But 9 is the bottom of the 2nd shape. Maybe it's the median of sorted bottoms?
    # Or: the bottom that minimizes total movement.

    # Another theory: 9 = 13 - 4 = max_bottom - shape_h.
    # Let me check: shape_h=4. max_bottom=13. 13-4=9? Yes!
    # So: target_bottom = max(max_r for all shapes) - shape_h?
    # That's equivalent to: the top of the bottommost shape.

    # Check train 1 (13x23): shapes have heights... let me not overanalyze.
    # Let me try: target_bottom = max(max_r) - shape_h

    max_max_r = max(max_r for _,_,max_r,_ in shape_info)
    target_bottom = max_max_r - shape_h

    # Actually no, that gives top=target_bottom-shape_h+1=max_max_r-2*shape_h+1.
    # For train 0: 13-4=9. Bottom=9. Top=9-4+1=6. Output is rows 6-9. ✓!

    out = [[0]*cols for _ in range(rows)]
    for shape, min_r, max_r, h in shape_info:
        shift = target_bottom - max_r
        for r, c, v in shape:
            nr = r + shift
            if 0 <= nr < rows:
                out[nr][c] = v
    return out

def solve_e48_e048c9ed(grid):
    """Bars of varying lengths with sentinel value. Output value = (length-1)^2 % 10."""
    rows, cols = len(grid), len(grid[0])

    # Find the sentinel (isolated non-zero value not part of any bar)
    # Find bars (horizontal sequences of same non-zero value)
    bars = []
    sentinel = None
    sentinel_pos = None

    for r in range(rows):
        c = 0
        while c < cols:
            if grid[r][c] != 0:
                # Find extent of this bar
                color = grid[r][c]
                start_c = c
                while c < cols and grid[r][c] == color:
                    c += 1
                length = c - start_c
                if length == 1:
                    # Check if isolated (no same-color neighbors)
                    isolated = True
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, start_c+dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == color:
                            isolated = False
                    if isolated:
                        sentinel = color
                        sentinel_pos = (r, start_c)
                        continue
                bars.append((r, start_c, length, color))
            else:
                c += 1

    if sentinel_pos is None or not bars:
        return grid

    out = [row[:] for row in grid]
    sent_r, sent_c = sentinel_pos

    for bar_r, bar_start, bar_len, bar_color in bars:
        val = ((bar_len - 1) ** 2) % 10
        out[bar_r][sent_c] = val

    return out

def solve_e619ca6e(grid):
    """Seed rectangle that spawns copies diagonally, growing in a zigzag pattern."""
    rows, cols = len(grid), len(grid[0])

    # Find the initial rectangle (non-zero block)
    non_zero = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                non_zero.append((r, c))

    if not non_zero:
        return grid

    min_r = min(r for r,c in non_zero)
    max_r = max(r for r,c in non_zero)
    min_c = min(c for r,c in non_zero)
    max_c = max(c for r,c in non_zero)

    block_h = max_r - min_r + 1
    block_w = max_c - min_c + 1
    color = grid[min_r][min_c]

    out = [row[:] for row in grid]

    # The block spawns copies that go diagonally in alternating directions
    # Starting from the seed, each new copy is offset diagonally
    # Direction alternates: down-right, down-left, down-right, etc.

    # The pattern: from the bottom edge of the current block, a new block appears
    # offset diagonally. The diagonal direction depends on block dimensions.

    # Looking at examples: the copies form a zigzag pattern going down
    cur_r = max_r + 1  # start below the seed
    cur_c_left = min_c
    cur_c_right = max_c

    # Determine initial diagonal direction based on seed position
    # The next block goes to the left and down
    direction = -1  # -1 = going left, +1 = going right

    # Alternate: each new block is placed at opposite side
    # The block width determines the horizontal step

    # From examples: after the seed, new blocks appear diag-left, then diag-right
    # The offset pattern: block shifts by (block_h, -block_w) then (block_h, +block_w) etc.
    # But also the connecting pattern (the 3s between blocks) needs to be placed

    # Actually from example 0: seed at rows 9-10, cols 6-10.
    # Next copy: rows 11-12, cols 1-5 (shifted left)
    # Then: rows 13-14, cols 11-15... hmm
    # Actually the blocks are "3" and they expand outward in a zigzag

    # This is a complex spatial recursion. Let me just place blocks in the zigzag.
    cr, cc = min_r, min_c  # current block top-left
    # Each step: place block, then move diagonally
    step = 0
    while True:
        # Place current block
        for r in range(cr, cr + block_h):
            for c in range(cc, cc + block_w):
                if 0 <= r < rows and 0 <= c < cols:
                    out[r][c] = color

        # Next block position
        if step % 2 == 0:
            next_r = cr + block_h
            next_c = cc - block_w - (block_w)
        else:
            next_r = cr + block_h
            next_c = cc + block_w + (block_w)

        if next_r >= rows or next_c < 0 or next_c + block_w > cols:
            break

        cr, cc = next_r, next_c
        step += 1

    return out

def solve_e2092e0c(grid):
    """Find 5555 row/block and expand it in a specific pattern."""
    rows, cols = len(grid), len(grid[0])
    # Find the row containing 5555 pattern
    five_row = -1
    for r in range(rows):
        row_vals = grid[r]
        count5 = sum(1 for v in row_vals if v == 5)
        if count5 >= 4:
            five_row = r
            break

    if five_row < 0:
        return grid

    # The 5s define a shape. Copy it and reflect to fill gaps.
    # This pattern is about reflecting the 5-block through the grid.

    # Actually: look at the full pattern. The 5555 row acts as a mirror/copy source.
    # For the row of 5s, look at numbers in the same row - they define positions.
    # Then expand using those numbers.

    out = [row[:] for row in grid]

    # Find the 5-block: consecutive 5s in a specific row
    five_cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 5]
    if not five_cells:
        return grid

    # Find the bounding box of 5s
    min_r5 = min(r for r,c in five_cells)
    max_r5 = max(r for r,c in five_cells)
    min_c5 = min(c for r,c in five_cells)
    max_c5 = max(c for r,c in five_cells)

    # The 5-block is a rectangle. Find what's adjacent and copy it.
    # Looking at examples: the 5-row acts as a separator. Above and below it,
    # there are numbers. The pattern copies the numbers to fill a rectangular region.

    return grid  # skip for now

def solve_e84fef15(grid):
    """Tiled grid with one tile modified. Find the difference."""
    rows, cols = len(grid), len(grid[0])

    # Grid is tiled with a repeating pattern separated by 3s
    # Find the tile size by looking for separator rows/cols
    sep_rows = []
    for r in range(rows):
        if all(grid[r][c] == 3 for c in range(cols)):
            sep_rows.append(r)

    sep_cols = []
    for c in range(cols):
        if all(grid[r][c] == 3 for r in range(rows)):
            sep_cols.append(c)

    if not sep_rows or not sep_cols:
        return grid

    # Extract tiles
    tile_h = sep_rows[0]
    tile_w = sep_cols[0]

    # Extract all tiles
    tiles = []
    r_starts = [0] + [s+1 for s in sep_rows]
    c_starts = [0] + [s+1 for s in sep_cols]

    for rs in r_starts:
        for cs in c_starts:
            if rs + tile_h <= rows and cs + tile_w <= cols:
                tile = []
                for r in range(rs, rs + tile_h):
                    row = []
                    for c in range(cs, cs + tile_w):
                        row.append(grid[r][c])
                    tile.append(tuple(row))
                tiles.append((rs, cs, tuple(tile)))

    if not tiles:
        return grid

    # Find the most common tile (the reference)
    tile_counter = Counter(t for _,_,t in tiles)
    ref_tile = tile_counter.most_common(1)[0][0]

    # Find the modified tile
    result = [[8]*tile_w for _ in range(tile_h)]
    for rs, cs, tile in tiles:
        if tile != ref_tile:
            # Find differences
            for r in range(tile_h):
                for c in range(tile_w):
                    if tile[r][c] != ref_tile[r][c]:
                        result[r][c] = 1
                    else:
                        result[r][c] = ref_tile[r][c]
            break

    # Actually: the output is the tile_h x tile_w result showing the difference
    # with 1 replacing the changed cells and 8 for unchanged
    # Wait, looking at examples more carefully - the output marks changed cells
    # with a specific pattern.

    # Actually from the examples: the output tile has 8s where the tile matches reference,
    # and 1s where it differs.
    for r in range(tile_h):
        for c in range(tile_w):
            if ref_tile[r][c] == result[r][c]:
                pass  # keep

    return [list(row) for row in result]

def solve_e78887d1(grid):
    """Multiple 3xN blocks separated by 0-rows. XOR or combine blocks."""
    rows, cols = len(grid), len(grid[0])

    # Find blocks separated by all-0 rows
    blocks = []
    i = 0
    while i < rows:
        if all(grid[i][c] == 0 for c in range(cols)):
            i += 1
            continue
        block_start = i
        block_rows = []
        while i < rows and not all(grid[i][c] == 0 for c in range(cols)):
            block_rows.append(grid[i][:])
            i += 1
        blocks.append(block_rows)

    if not blocks:
        return grid

    # Output is 3 rows (fixed height based on examples)
    # XOR/combine the blocks
    # Looking at examples: blocks are combined by some operation to produce a single 3-row output

    # From ex3: 4 blocks of 3 rows each -> output 3 rows. The output matches block 0.
    # So maybe: the output is the block that appears only once (unique)?
    # Or: the block that's different from all others?

    # Check: ex0 has 2 blocks. Output matches... let me check.
    # This needs more analysis. Skip for now.
    return grid

def solve_e5790162(grid):
    """From 3 (start), draw path of 3s toward 6/8 (stop). Path goes horizontal first, then turns."""
    rows, cols = len(grid), len(grid[0])

    # Find key positions
    three_pos = six_pos = eight_pos = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3:
                three_pos = (r, c)
            elif grid[r][c] == 6:
                six_pos = (r, c)
            elif grid[r][c] == 8:
                eight_pos = (r, c)

    if three_pos is None:
        return grid

    out = [row[:] for row in grid]

    # The 3 draws a path. Looking at examples:
    # 3 draws horizontal line toward 8, then vertical toward 6
    # Or: 3 goes horizontal to the column of 8, then turns vertical toward 6

    # Ex0: 3 at (2,0), 8 at (2,4), 6 at (0,3)? Actually:
    # Input: 000000/000000/300080/000000/000000/000000
    # 3 at (2,0), 8 at (2,4).
    # No 6? Wait: 300080 = 3,0,0,0,8,0. That's 3 and 8.
    # But I see 6 somewhere... hmm the task data shows:
    # Row 2: 300080. Wait: 3,0,0,0,8,0. No 6.
    # Let me recheck. Oh wait the input is:
    # 000000/000000/300080/000000/000000/000000
    # 3 at (2,0), 8 at (2,4)
    # Output: 000300/000300/333380/000000/000000/000000
    # 3s fill from (2,0) to left of 8, then go UP in col 3.

    # So 3 draws a horizontal line from its position toward 8, stopping before 8.
    # Then continues vertically from the turn point.

    # The rule seems to be: from 3's position, go horizontally toward 8 (stopping 1 before).
    # Then go vertically from the last horizontal cell, continuing until hitting the grid edge.

    # Multiple 3-8 pairs in some inputs? Let me check ex1.
    # Ex1: 3 at (3,0), 6 at (3,4), 8 at (6,3)
    # Output: ... /333360/000300/000333/000800/000000
    # 3 goes right from (3,0) to (3,3) [before 6], then down from (3,3) to... (5,3)?
    # Then right from (5,3) to (5,5)? No: output row 5: 000333.

    # Hmm, it looks like: 3 draws from its position toward the nearest target (6 or 8),
    # making an L-shaped path. At the bend point, it continues in the perpendicular direction.

    # Actually, maybe the 3 seeks toward all 8s and 6s sequentially.
    # This is like drawing an L-path from 3 to each target.

    # I'll implement: for each pair (3, target), draw L-path with 3s.
    targets = []
    if six_pos: targets.append(six_pos)
    if eight_pos: targets.append(eight_pos)

    # For 6-targets: draw from 3 towards them
    # For 8-targets: 8 is the endpoint

    # Actually let me look at it more simply. From 3's starting position:
    # Go horizontal to align with the 8's column (or stop before 8's row)
    # Then go vertical toward 8 (or the edge).

    # For each adjacent pair (3 shares row/col with an 8 or 6):
    # Draw path from 3 to that point, then continue perpendicular

    # This task is quite varied. Skip for now.
    return grid

def solve_e45ef808(grid):
    """Diagonal boundary between 1s and 6s. Add 9 and 4 marks along it."""
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # Row 0 is all 0s (separator).
    # The grid below has 1s and 6s forming regions with a staircase boundary.
    # Find the boundary: transitions between 1 and 6

    # For each column, find the row where 1->6 transition happens
    boundary_points = []
    for c in range(cols):
        for r in range(1, rows-1):
            if grid[r][c] == 1 and grid[r+1][c] == 6:
                boundary_points.append((r, c))
                break

    if not boundary_points:
        return grid

    # The staircase goes from one side to another.
    # Find the two "endpoints" of the staircase (where boundary starts and ends column-wise)
    # Place 9 as a vertical line at the leftmost boundary column
    # Place 4 as a vertical line at the rightmost boundary column (or something similar)

    # From examples: 9 is placed in the 1-region on one side of the staircase,
    # and 4 on the other side. They form vertical lines.

    # In ex0: boundary goes from around col 4-5 (bottom) to col 8-9 (top-right area)
    # 9 at col 4, 4 at col 8 for rows 1-8

    # Actually: the boundary column at the bottom of the staircase -> 9
    # The boundary column at the top of the staircase -> 4

    min_boundary_col = min(c for r,c in boundary_points)
    max_boundary_col = max(c for r,c in boundary_points)

    # Find the corresponding rows
    nine_col = min_boundary_col
    four_col = max_boundary_col

    # Hmm, from example 0:
    # Boundary points: transitions at various rows/cols.
    # Let me try a different approach: find the 'step' positions.

    # The staircase has steps. Each step is where the boundary moves right/down.
    # 9 marks one step and 4 marks another.

    # Actually from the example outputs, it seems like:
    # 9 is placed in the COLUMN where the first 6 appears (leftmost 6 column)
    # in the 1-region rows
    # 4 is placed in the column where the transition from all-1 rows to mixed rows happens

    # I'll skip this task for now.
    return grid

def solve_df8cc377(grid):
    """Some grid transformation..."""
    return grid  # placeholder

def solve_e760a62e(grid):
    """Grid divided by 8-lines into cells. Colored dots spread to fill their row/column band."""
    rows, cols = len(grid), len(grid[0])

    # Find 8-rows and 8-cols (separator lines)
    sep_rows = []
    for r in range(rows):
        if all(grid[r][c] == 8 for c in range(cols)):
            sep_rows.append(r)

    sep_cols = []
    for c in range(cols):
        if all(grid[r][c] == 8 for r in range(rows)):
            sep_cols.append(c)

    # Find non-0, non-8 dots
    dots = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and grid[r][c] != 8:
                dots.append((r, c, grid[r][c]))

    if not dots or not sep_rows or not sep_cols:
        return grid

    out = [row[:] for row in grid]

    # Each dot fills its entire cell-band (row band between separators)
    # Find which cell-row each dot is in
    for r, c, v in dots:
        # Find the row-band (between two sep_rows)
        band_top = 0
        band_bot = rows - 1
        for sr in sep_rows:
            if sr < r:
                band_top = sr + 1
            elif sr > r:
                band_bot = sr - 1
                break

        # Find the col-band
        band_left = 0
        band_right = cols - 1
        for sc in sep_cols:
            if sc < c:
                band_left = sc + 1
            elif sc > c:
                band_right = sc - 1
                break

        # Fill the entire row-band in the column-band with the color
        for rr in range(band_top, band_bot + 1):
            for cc in range(band_left, band_right + 1):
                if out[rr][cc] == 0:
                    out[rr][cc] = v

    return out

def solve_e734a0e8(grid):
    """Grid divided by 0-lines into blocks. Non-7 patterns in blocks propagate to all blocks."""
    rows, cols = len(grid), len(grid[0])

    sep_rows = []
    for r in range(rows):
        if all(grid[r][c] == 0 for c in range(cols)):
            sep_rows.append(r)

    sep_cols = []
    for c in range(cols):
        if all(grid[r][c] == 0 for r in range(rows)):
            sep_cols.append(c)

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

    bh = row_bands[0][1] - row_bands[0][0] + 1
    bw = col_bands[0][1] - col_bands[0][0] + 1

    # Find the main pattern block (most non-7 cells)
    best_block = None
    best_count = 0
    for ri, (r1, r2) in enumerate(row_bands):
        for ci, (c1, c2) in enumerate(col_bands):
            count = sum(1 for dr in range(bh) for dc in range(bw) if grid[r1+dr][c1+dc] != 7)
            if count > best_count:
                best_count = count
                best_block = (ri, ci, r1, c1)

    if not best_block:
        return grid

    _, _, pr1, pc1 = best_block
    pattern = [[grid[pr1+dr][pc1+dc] for dc in range(bw)] for dr in range(bh)]

    out = [row[:] for row in grid]

    # Find blocks with any non-7 content and copy the pattern there
    for ri, (r1, r2) in enumerate(row_bands):
        for ci, (c1, c2) in enumerate(col_bands):
            has_non7 = any(grid[r1+dr][c1+dc] != 7 for dr in range(bh) for dc in range(bw))
            if has_non7:
                for dr in range(bh):
                    for dc in range(bw):
                        out[r1+dr][c1+dc] = pattern[dr][dc]

    return out

def solve_de493100(grid):
    """Large symmetric grid with 7-block masking region. Reconstruct the masked part."""
    rows, cols = len(grid), len(grid[0])

    # Find the 7-region (masked area)
    sevens = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 7:
                sevens.add((r, c))

    if not sevens:
        return grid

    min_r7 = min(r for r,c in sevens)
    max_r7 = max(r for r,c in sevens)
    min_c7 = min(c for r,c in sevens)
    max_c7 = max(c for r,c in sevens)

    out = [row[:] for row in grid]

    # The grid has point symmetry (180 rotation) or reflection symmetry
    # Try to reconstruct 7-cells using the symmetric counterpart

    # Try horizontal reflection (left-right)
    center_c = (cols - 1) / 2
    for r, c in sevens:
        mirror_c = int(2 * center_c - c)
        if 0 <= mirror_c < cols and grid[r][mirror_c] != 7:
            out[r][c] = grid[r][mirror_c]

    # Check if any 7s remain
    remaining = [(r,c) for r,c in sevens if out[r][c] == 7]
    if remaining:
        # Try vertical reflection
        center_r = (rows - 1) / 2
        for r, c in remaining:
            mirror_r = int(2 * center_r - r)
            if 0 <= mirror_r < rows and grid[mirror_r][c] != 7:
                out[r][c] = grid[mirror_r][c]

    remaining = [(r,c) for r,c in sevens if out[r][c] == 7]
    if remaining:
        # Try 180 rotation
        for r, c in remaining:
            mirror_r = int(2 * center_r - r)
            mirror_c = int(2 * center_c - c)
            if 0 <= mirror_r < rows and 0 <= mirror_c < cols and grid[mirror_r][mirror_c] != 7:
                out[r][c] = grid[mirror_r][mirror_c]

    return out

# Extract output from masked 7-region
def solve_de493100_extract(grid):
    """Large grid with 7-masked area. Output = just the reconstructed tile."""
    rows, cols = len(grid), len(grid[0])

    sevens = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 7:
                sevens.add((r,c))

    if not sevens:
        return grid

    min_r7 = min(r for r,c in sevens)
    max_r7 = max(r for r,c in sevens)
    min_c7 = min(c for r,c in sevens)
    max_c7 = max(c for r,c in sevens)

    # Reconstruct using symmetry
    center_r = (rows - 1) / 2
    center_c = (cols - 1) / 2

    reconstructed = {}
    for r, c in sevens:
        # Try multiple symmetries
        for mr, mc in [
            (r, int(2*center_c - c)),  # h-reflect
            (int(2*center_r - r), c),  # v-reflect
            (int(2*center_r - r), int(2*center_c - c)),  # 180-rotate
        ]:
            if 0 <= mr < rows and 0 <= mc < cols and grid[mr][mc] != 7:
                reconstructed[(r,c)] = grid[mr][mc]
                break

    # Extract the 7-region as output
    h = max_r7 - min_r7 + 1
    w = max_c7 - min_c7 + 1
    result = [[0]*w for _ in range(h)]
    for r in range(min_r7, max_r7 + 1):
        for c in range(min_c7, max_c7 + 1):
            if (r,c) in reconstructed:
                result[r - min_r7][c - min_c7] = reconstructed[(r,c)]
            elif (r,c) in sevens:
                result[r - min_r7][c - min_c7] = 0  # fallback
            else:
                result[r - min_r7][c - min_c7] = grid[r][c]

    return result

# ========== Register and test ==========
TASK_SOLVERS = {}
def register(task_id, solver):
    TASK_SOLVERS[task_id] = solver

register('e133d23d', solve_e133d23d)
register('e345f17b', solve_e345f17b)
register('e633a9e5', solve_e633a9e5)
register('e1baa8a4', solve_e1baa8a4)
register('dc1df850', solve_dc1df850)
register('e3fe1151', solve_e3fe1151)
register('e57337a4', solve_e57337a4)
register('e21a174a', solve_e21a174a)
register('e74e1818', solve_e74e1818)
register('da2b0fe3', solve_da2b0fe3)
register('e0fb7511', solve_e0fb7511)
register('db7260a4', solve_db7260a4)
register('e7dd8335', solve_e7dd8335)
register('d968ffd4', solve_d968ffd4)
register('e7639916', solve_e7639916)
register('e7a25a18', solve_e7a25a18)
register('e7b06bea', solve_e7b06bea)
register('ded97339', solve_ded97339)
register('e4075551', solve_e4075551)
register('e41c6fd3', solve_e41c6fd3)
register('dce56571', solve_dce56571)
register('e4941b18', solve_e4941b18)
register('e734a0e8', solve_e734a0e8)
register('de493100', solve_de493100_extract)
register('e760a62e', solve_e760a62e)

def test_solver(task_id, solver):
    data = json.load(open(f'data/arc2/{task_id}.json'))
    for i, pair in enumerate(data['train']):
        try:
            result = solver(pair['input'])
            if result != pair['output']:
                return False, i
        except Exception as e:
            return False, i
    return True, -1

def main():
    task_ids = ['d94c3b52','d968ffd4','da2b0fe3','da6e95e5','db118e2a','db615bd4','db7260a4',
                'dbc1a6ce','dc1df850','dc2aa30b','dc2e9a9d','dc46ea44','dce56571','dd2401ed',
                'de493100','ded97339','df8cc377','df978a02','df9fd884','e048c9ed','e0fb7511',
                'e133d23d','e1baa8a4','e1d2900e','e2092e0c','e21a174a','e345f17b','e39e9282',
                'e3f79277','e3fe1151','e4075551','e41c6fd3','e45ef808','e4888269','e4941b18',
                'e57337a4','e5790162','e5c44e8f','e619ca6e','e633a9e5','e681b708','e69241bd',
                'e6de6e8f','e729b7be','e734a0e8','e74e1818','e760a62e','e7639916','e78887d1',
                'e7a25a18','e7b06bea','e7dd8335','e84fef15','e872b94a']

    solutions = {}

    for task_id in task_ids:
        if task_id not in TASK_SOLVERS:
            continue

        solver = TASK_SOLVERS[task_id]
        passed, failed_idx = test_solver(task_id, solver)

        if passed:
            print(f'{task_id}: PASS')
            data = json.load(open(f'data/arc2/{task_id}.json'))
            task_solutions = []
            for test in data['test']:
                result = solver(test['input'])
                task_solutions.append(result)
            solutions[task_id] = task_solutions
        else:
            print(f'{task_id}: FAIL at train {failed_idx}')

    with open('data/arc2_solutions_train_af.json', 'w') as f:
        json.dump(solutions, f)

    print(f'\nTotal passing: {len(solutions)}/{len(task_ids)}')

if __name__ == '__main__':
    main()
