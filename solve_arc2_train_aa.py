#!/usr/bin/env python3
"""ARC-AGI solver for batch aa (first 54 tasks)."""

import json
import copy
from collections import Counter, deque, defaultdict

BASE = "C:/Users/atchi/h4-polytopic-attention/data/arc2"

def load_task(tid):
    with open(f"{BASE}/{tid}.json") as f:
        return json.load(f)

def test_solver(tid, solve_fn):
    task = load_task(tid)
    for i, pair in enumerate(task['train']):
        inp = [row[:] for row in pair['input']]
        expected = pair['output']
        try:
            result = solve_fn(inp)
        except Exception as e:
            return False
        if result != expected:
            return False
    return True

def get_test_outputs(tid, solve_fn):
    task = load_task(tid)
    outputs = []
    for pair in task['test']:
        inp = [row[:] for row in pair['input']]
        result = solve_fn(inp)
        outputs.append(result)
    return outputs

# ============================================================
# HELPERS
# ============================================================
def find_connected_components(grid, target_color, R, C):
    visited = [[False]*C for _ in range(R)]
    components = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] == target_color and not visited[r][c]:
                comp = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and grid[nr][nc] == target_color:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append(comp)
    return components

def find_all_components(grid, R, C, ignore={0}):
    visited = [[False]*C for _ in range(R)]
    components = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] not in ignore and not visited[r][c]:
                color = grid[r][c]
                comp = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append((color, comp))
    return components

def count_enclosed_regions(comp_set, min_r, max_r, min_c, max_c):
    visited = {}
    enclosed = 0
    for r in range(min_r, max_r+1):
        for c in range(min_c, max_c+1):
            if (r,c) not in comp_set and (r,c) not in visited:
                q = deque([(r,c)])
                visited[(r,c)] = True
                cells = [(r,c)]
                touches_edge = False
                while q:
                    cr, cc = q.popleft()
                    if cr == min_r or cr == max_r or cc == min_c or cc == max_c:
                        touches_edge = True
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if min_r <= nr <= max_r and min_c <= nc <= max_c and (nr,nc) not in comp_set and (nr,nc) not in visited:
                            visited[(nr,nc)] = True
                            q.append((nr,nc))
                            cells.append((nr,nc))
                if not touches_edge:
                    enclosed += 1
    return enclosed

# ============================================================
# 00576224 - 2x2 tile 3x3, alternate rows reversed
# ============================================================
def solve_00576224(grid):
    R, C = len(grid), len(grid[0])
    out = [[0]*(C*3) for _ in range(R*3)]
    for br in range(3):
        for bc in range(3):
            for r in range(R):
                for c in range(C):
                    if br % 2 == 0:
                        out[br*R + r][bc*C + c] = grid[r][c]
                    else:
                        out[br*R + r][bc*C + c] = grid[r][C-1-c]
    return out

# ============================================================
# 007bbfb7 - Self-outer-product
# ============================================================
def solve_007bbfb7(grid):
    R, C = len(grid), len(grid[0])
    out = [[0]*(C*C) for _ in range(R*R)]
    for r in range(R):
        for c in range(C):
            if grid[r][c] != 0:
                for r2 in range(R):
                    for c2 in range(C):
                        if grid[r2][c2] != 0:
                            out[r*R + r2][c*C + c2] = grid[r2][c2]
    return out

# ============================================================
# 009d5c81 - 8-shape recolored based on 1-pattern identity
# ============================================================
def solve_009d5c81(grid):
    R, C = len(grid), len(grid[0])
    out = [[0]*C for _ in range(R)]

    ones = [(r,c) for r in range(R) for c in range(C) if grid[r][c] == 1]
    if not ones:
        return [row[:] for row in grid]

    min_r = min(r for r,c in ones)
    max_r = max(r for r,c in ones)
    min_c = min(c for r,c in ones)
    max_c = max(c for r,c in ones)

    pattern = frozenset((r-min_r, c-min_c) for r,c in ones)

    # Determine enclosed regions
    comp_set = set(ones)
    enc = count_enclosed_regions(comp_set, min_r, max_r, min_c, max_c)

    # Pattern -> color mapping
    n = len(ones)
    if enc >= 1:
        color = 7
    elif n == 5:
        color = 2
    else:
        color = 3

    for r in range(R):
        for c in range(C):
            if grid[r][c] == 8:
                out[r][c] = color
    return out

# ============================================================
# 00d62c1b - Fill enclosed areas in 3-shapes with 4
# ============================================================
def solve_00d62c1b(grid):
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    visited = [[False]*C for _ in range(R)]
    q = deque()
    for r in range(R):
        for c in range(C):
            if (r == 0 or r == R-1 or c == 0 or c == C-1) and grid[r][c] == 0:
                if not visited[r][c]:
                    visited[r][c] = True
                    q.append((r,c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and grid[nr][nc] == 0:
                visited[nr][nc] = True
                q.append((nr, nc))
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 0 and not visited[r][c]:
                out[r][c] = 4
    return out

# ============================================================
# 00dbd492 - Rectangle frames of 2s: fill interior based on size
# ============================================================
def solve_00dbd492(grid):
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r1 in range(R):
        for c1 in range(C):
            if grid[r1][c1] != 2:
                continue
            for r2 in range(r1+2, R):
                for c2 in range(c1+2, C):
                    is_frame = True
                    for c in range(c1, c2+1):
                        if grid[r1][c] != 2 or grid[r2][c] != 2:
                            is_frame = False
                            break
                    if not is_frame:
                        continue
                    for r in range(r1, r2+1):
                        if grid[r][c1] != 2 or grid[r][c2] != 2:
                            is_frame = False
                            break
                    if not is_frame:
                        continue
                    interior_has_2 = False
                    for r in range(r1+1, r2):
                        for c in range(c1+1, c2):
                            if grid[r][c] == 2:
                                interior_has_2 = True
                    if not interior_has_2:
                        continue
                    interior_size = (r2 - r1 - 1) * (c2 - c1 - 1)
                    if interior_size <= 9:
                        fill_color = 8
                    elif interior_size <= 25:
                        fill_color = 4
                    else:
                        fill_color = 3
                    for r in range(r1+1, r2):
                        for c in range(c1+1, c2):
                            if out[r][c] == 0:
                                out[r][c] = fill_color
    return out

# ============================================================
# 03560426 - Stack blocks diagonally
# ============================================================
def solve_03560426(grid):
    R, C = len(grid), len(grid[0])
    visited = [[False]*C for _ in range(R)]
    blocks = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                comp = []
                q = deque([(r,c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                min_r = min(cr for cr,cc in comp)
                max_r = max(cr for cr,cc in comp)
                min_c = min(cc for cr,cc in comp)
                max_c = max(cc for cr,cc in comp)
                blocks.append((color, min_r, max_r, min_c, max_c, len(comp)))
    blocks.sort(key=lambda b: (b[3], b[1]))
    out = [[0]*C for _ in range(R)]
    cur_r = 0
    cur_c = 0
    for color, min_r, max_r, min_c, max_c, size in blocks:
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        for r in range(h):
            for c in range(w):
                if cur_r + r < R and cur_c + c < C:
                    out[cur_r + r][cur_c + c] = color
        cur_r += h - 1
        cur_c += w - 1
    return out

# ============================================================
# 0520fde7 - Split at 5-column, AND both halves -> 2
# ============================================================
def solve_0520fde7(grid):
    R = len(grid)
    col5 = None
    for c in range(len(grid[0])):
        if all(grid[r][c] == 5 for r in range(R)):
            col5 = c
            break
    left = [row[:col5] for row in grid]
    right = [row[col5+1:] for row in grid]
    C2 = len(left[0])
    out = [[0]*C2 for _ in range(R)]
    for r in range(R):
        for c in range(C2):
            if left[r][c] != 0 and right[r][c] != 0:
                out[r][c] = 2
    return out

# ============================================================
# 0692e18c - Self-outer-product with inverted grid
# ============================================================
def solve_0692e18c(grid):
    R, C = len(grid), len(grid[0])
    out = [[0]*(C*C) for _ in range(R*R)]
    nz = None
    for r in range(R):
        for c in range(C):
            if grid[r][c] != 0:
                nz = grid[r][c]
    for r in range(R):
        for c in range(C):
            if grid[r][c] != 0:
                for r2 in range(R):
                    for c2 in range(C):
                        if grid[r2][c2] == 0:
                            out[r*R + r2][c*C + c2] = nz
    return out

# ============================================================
# 08ed6ac7 - Color columns by height rank (tallest=1)
# ============================================================
def solve_08ed6ac7(grid):
    R, C = len(grid), len(grid[0])
    out = [[0]*C for _ in range(R)]
    # Find columns with 5s and their heights
    col_heights = []
    for c in range(C):
        height = sum(1 for r in range(R) if grid[r][c] == 5)
        if height > 0:
            col_heights.append((height, c))
    # Sort by height descending
    col_heights.sort(key=lambda x: -x[0])
    # Assign colors 1,2,3,4...
    for idx, (height, c) in enumerate(col_heights):
        color = idx + 1
        for r in range(R):
            if grid[r][c] == 5:
                out[r][c] = color
    return out

# ============================================================
# 0a2355a6 - Color shapes by enclosed region count
# ============================================================
def solve_0a2355a6(grid):
    R, C = len(grid), len(grid[0])
    out = [[0]*C for _ in range(R)]
    visited = [[False]*C for _ in range(R)]
    components = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 8 and not visited[r][c]:
                comp = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and grid[nr][nc] == 8:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append(comp)

    enc_to_color = {1: 1, 2: 3, 3: 2, 4: 4, 5: 5}

    for comp in components:
        comp_set = set(comp)
        rs = [r for r,c in comp]
        cs = [c for r,c in comp]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        enc = count_enclosed_regions(comp_set, min_r, max_r, min_c, max_c)
        color = enc_to_color.get(enc, enc)
        for r, c in comp:
            out[r][c] = color
    return out

# ============================================================
# 0b17323b - Diagonal dots, continue pattern with color 2
# ============================================================
def solve_0b17323b(grid):
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    ones = [(r, c) for r in range(R) for c in range(C) if grid[r][c] == 1]
    if len(ones) < 2:
        return grid
    ones.sort()
    dr = ones[1][0] - ones[0][0]
    dc = ones[1][1] - ones[0][1]
    last = ones[-1]
    r, c = last[0] + dr, last[1] + dc
    while 0 <= r < R and 0 <= c < C:
        out[r][c] = 2
        r += dr
        c += dc
    return out

# ============================================================
# 0bb8deee - Grid divided by colored lines into 4 quadrants, extract and tile
# ============================================================
def solve_0bb8deee(grid):
    R, C = len(grid), len(grid[0])

    h_line = v_line = None
    div_color = None
    for r in range(R):
        vals = set(grid[r])
        if len(vals) == 1 and 0 not in vals:
            h_line = r
            div_color = grid[r][0]
            break

    if div_color is None:
        return grid

    for c in range(C):
        if all(grid[r][c] == div_color for r in range(R)):
            v_line = c
            break

    if v_line is None:
        return grid

    def extract_shape(r_start, r_end, c_start, c_end):
        cells = []
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                if grid[r][c] != 0 and grid[r][c] != div_color:
                    cells.append((r - r_start, c - c_start, grid[r][c]))
        return cells

    def normalize(cells):
        if not cells:
            return cells
        min_r = min(r for r,c,v in cells)
        min_c = min(c for r,c,v in cells)
        return [(r-min_r, c-min_c, v) for r,c,v in cells]

    tl = normalize(extract_shape(0, h_line, 0, v_line))
    tr = normalize(extract_shape(0, h_line, v_line+1, C))
    bl = normalize(extract_shape(h_line+1, R, 0, v_line))
    br = normalize(extract_shape(h_line+1, R, v_line+1, C))

    def shape_dims(cells):
        if not cells:
            return 0, 0
        return max(r for r,c,v in cells)+1, max(c for r,c,v in cells)+1

    max_h = max(shape_dims(s)[0] for s in [tl, tr, bl, br] if s)
    max_w = max(shape_dims(s)[1] for s in [tl, tr, bl, br] if s)

    out = [[0]*(max_w*2) for _ in range(max_h*2)]

    for quad, off_r, off_c in [(tl,0,0),(tr,0,max_w),(bl,max_h,0),(br,max_h,max_w)]:
        for r,c,v in quad:
            out[off_r+r][off_c+c] = v
    return out

# ============================================================
# 0becf7df - Swap colors pairwise based on 2x2 key
# ============================================================
def solve_0becf7df(grid):
    R, C = len(grid), len(grid[0])
    key_tl = grid[0][0]
    key_tr = grid[0][1]
    key_bl = grid[1][0]
    key_br = grid[1][1]
    mapping = {key_tl: key_tr, key_tr: key_tl, key_bl: key_br, key_br: key_bl, 0: 0}
    out = [row[:] for row in grid]
    for r in range(R):
        for c in range(C):
            if r <= 1 and c <= 1:
                continue  # skip the key itself
            if grid[r][c] in mapping:
                out[r][c] = mapping[grid[r][c]]
    return out

# ============================================================
# 0c786b71 - Flip vertically, then mirror each row
# ============================================================
def solve_0c786b71(grid):
    R, C = len(grid), len(grid[0])
    flipped = grid[::-1]
    out = []
    for r in range(R):
        row = flipped[r]
        out.append(row[::-1] + row[:])
    for r in range(R):
        row = flipped[R-1-r]
        out.append(row[::-1] + row[:])
    return out

# ============================================================
# 0c9aba6e - Two halves separated by 7-row, NOR -> 8
# ============================================================
def solve_0c9aba6e(grid):
    R = len(grid)
    C = len(grid[0])
    sep = None
    for r in range(R):
        if all(grid[r][c] == 7 for c in range(C)):
            sep = r
            break
    top = grid[:sep]
    bottom = grid[sep+1:]
    outR = len(top)
    out = [[0]*C for _ in range(outR)]
    for r in range(outR):
        for c in range(C):
            t = (top[r][c] != 0) if r < len(top) else False
            b = (bottom[r][c] != 0) if r < len(bottom) else False
            if not t and not b:
                out[r][c] = 8
    return out

# ============================================================
# 0ca9ddb6 - Cross pattern around dots (1->7 cardinal, 2->4 diagonal)
# ============================================================
def solve_0ca9ddb6(grid):
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 1:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < R and 0 <= nc < C and out[nr][nc] == 0:
                        out[nr][nc] = 7
            elif grid[r][c] == 2:
                for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < R and 0 <= nc < C and out[nr][nc] == 0:
                        out[nr][nc] = 4
    return out

# ============================================================
# 0d3d703e - Color mapping (complement-like)
# ============================================================
def solve_0d3d703e(grid):
    mapping = {1:5, 2:6, 3:4, 4:3, 5:1, 6:2, 8:9, 9:8, 0:0, 7:7}
    return [[mapping[v] for v in row] for row in grid]

# ============================================================
# 0d87d2a6 - Connect two 1-dots with a vertical/horizontal line, recolor overlapping 2-shapes
# ============================================================
def solve_0d87d2a6(grid):
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    ones = [(r, c) for r in range(R) for c in range(C) if grid[r][c] == 1]
    if len(ones) < 2:
        return grid

    # Draw lines between each pair sharing row or column
    for i in range(len(ones)):
        for j in range(i+1, len(ones)):
            r1, c1 = ones[i]
            r2, c2 = ones[j]
            if c1 == c2:
                for r in range(min(r1,r2), max(r1,r2)+1):
                    out[r][c1] = 1
            elif r1 == r2:
                for c in range(min(c1,c2), max(c1,c2)+1):
                    out[r1][c] = 1

    # Recolor 2-components that overlap with any 1-line
    visited = [[False]*C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 2 and not visited[r][c]:
                comp = []
                q = deque([(r,c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and grid[nr][nc] == 2:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                overlaps = any(out[cr][cc] == 1 for cr, cc in comp)
                if overlaps:
                    for cr, cc in comp:
                        out[cr][cc] = 1
    return out

# ============================================================
# 0f63c0b9 - Colored dots define horizontal bands
# ============================================================
def solve_0f63c0b9(grid):
    R, C = len(grid), len(grid[0])
    dots = sorted([(r, c, grid[r][c]) for r in range(R) for c in range(C) if grid[r][c] != 0])

    zone_bounds = []
    for i in range(len(dots)-1):
        mid = (dots[i][0] + dots[i+1][0]) / 2
        zone_bounds.append(mid)

    out = [[0]*C for _ in range(R)]

    for r in range(R):
        color = None
        for di, (dr, dc, dv) in enumerate(dots):
            if di == 0 and (not zone_bounds or r <= zone_bounds[0]):
                color = dv
                break
            elif di == len(dots)-1:
                color = dv
                break
            elif zone_bounds[di-1] < r <= zone_bounds[di]:
                color = dv
                break
        if color is None:
            color = dots[-1][2]

        is_dot_row = any(dr == r for dr,dc,dv in dots)
        is_edge_row = (r == 0 or r == R-1)

        if is_dot_row or is_edge_row:
            for c in range(C):
                out[r][c] = color
        else:
            out[r][0] = color
            out[r][C-1] = color
    return out

# ============================================================
# 12997ef3 - Template shape + colored dots -> replicate in each color
# ============================================================
def solve_12997ef3(grid):
    R, C = len(grid), len(grid[0])
    ones = [(r,c) for r in range(R) for c in range(C) if grid[r][c] == 1]
    dots = [(r,c,grid[r][c]) for r in range(R) for c in range(C) if grid[r][c] not in (0, 1)]
    if not ones or not dots:
        return grid
    min_r = min(r for r,c in ones)
    max_r = max(r for r,c in ones)
    min_c = min(c for r,c in ones)
    max_c = max(c for r,c in ones)
    template = [(r - min_r, c - min_c) for r, c in ones]
    tH = max_r - min_r + 1
    tW = max_c - min_c + 1
    dots.sort()
    dot_rows = [r for r,c,v in dots]
    dot_cols = [c for r,c,v in dots]

    if len(set(dot_rows)) == 1:
        outW = tW * len(dots)
        outH = tH
        out = [[0]*outW for _ in range(outH)]
        for di, (dr, dc, color) in enumerate(sorted(dots, key=lambda x: x[1])):
            for tr, tc in template:
                out[tr][di*tW + tc] = color
        return out
    elif len(set(dot_cols)) == 1:
        outW = tW
        outH = tH * len(dots)
        out = [[0]*outW for _ in range(outH)]
        for di, (dr, dc, color) in enumerate(sorted(dots, key=lambda x: x[0])):
            for tr, tc in template:
                out[di*tH + tr][tc] = color
        return out
    else:
        outW = tW * len(dots)
        outH = tH
        out = [[0]*outW for _ in range(outH)]
        for di, (dr, dc, color) in enumerate(sorted(dots, key=lambda x: x[1])):
            for tr, tc in template:
                out[tr][di*tW + tc] = color
        return out

# ============================================================
# 140c817e - Cross pattern at dots on colored bg
# ============================================================
def solve_140c817e(grid):
    R, C = len(grid), len(grid[0])
    bg = Counter(grid[r][c] for r in range(R) for c in range(C)).most_common(1)[0][0]
    dots = [(r,c) for r in range(R) for c in range(C) if grid[r][c] != bg]
    out = [row[:] for row in grid]
    for r,c in dots:
        for rr in range(R):
            out[rr][c] = 1
        for cc in range(C):
            out[r][cc] = 1
        out[r][c] = 2
    for r,c in dots:
        for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < R and 0 <= nc < C and out[nr][nc] == bg:
                out[nr][nc] = 3
    return out

# ============================================================
# 14b8e18c - Add 2s at outer corners of rectangular borders
# ============================================================
def solve_14b8e18c(grid):
    R, C = len(grid), len(grid[0])
    bg = 7
    out = [row[:] for row in grid]

    visited = [[False]*C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg and not visited[r][c]:
                color = grid[r][c]
                comp = []
                q = deque([(r,c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            q.append((nr, nc))

                comp_set = set(comp)
                rs = [r2 for r2,c2 in comp]
                cs = [c2 for r2,c2 in comp]
                min_r, max_r = min(rs), max(rs)
                min_c, max_c = min(cs), max(cs)
                h = max_r - min_r + 1
                w = max_c - min_c + 1

                if h < 2 or w < 2:
                    continue

                is_border = True
                for c2 in range(min_c, max_c+1):
                    if (min_r, c2) not in comp_set or (max_r, c2) not in comp_set:
                        is_border = False; break
                if is_border:
                    for r2 in range(min_r, max_r+1):
                        if (r2, min_c) not in comp_set or (r2, max_c) not in comp_set:
                            is_border = False; break

                if not is_border:
                    continue

                is_solid = (len(comp) == h * w)

                if not is_solid or (h == w):
                    corners = [
                        (min_r-1, min_c), (min_r-1, max_c),
                        (min_r, min_c-1), (min_r, max_c+1),
                        (max_r, min_c-1), (max_r, max_c+1),
                        (max_r+1, min_c), (max_r+1, max_c),
                    ]
                    for nr, nc in corners:
                        if 0 <= nr < R and 0 <= nc < C:
                            out[nr][nc] = 2
    return out

# ============================================================
# 15696249 - 3x3 -> 9x9, tile based on uniform row/col
# ============================================================
def solve_15696249(grid):
    R, C = len(grid), len(grid[0])
    out = [[0]*(C*3) for _ in range(R*3)]
    for r in range(R):
        if len(set(grid[r])) == 1:
            for bc in range(3):
                for ri in range(R):
                    for ci in range(C):
                        out[r*R + ri][bc*C + ci] = grid[ri][ci]
            return out
    for c in range(C):
        col_vals = [grid[r][c] for r in range(R)]
        if len(set(col_vals)) == 1:
            for br in range(3):
                for ri in range(R):
                    for ci in range(C):
                        out[br*R + ri][c*C + ci] = grid[ri][ci]
            return out
    return out

# ============================================================
# 178fcbfb - Color 1,3 -> H-lines, Color 2 -> V-line
# ============================================================
def solve_178fcbfb(grid):
    R, C = len(grid), len(grid[0])
    dots = [(r,c,grid[r][c]) for r in range(R) for c in range(C) if grid[r][c] != 0]
    out = [[0]*C for _ in range(R)]
    for r,c,v in dots:
        if v == 2:
            for rr in range(R):
                out[rr][c] = 2
    for r,c,v in dots:
        if v in (1, 3):
            for cc in range(C):
                out[r][cc] = v
    return out

# ============================================================
# 17cae0c1 - 3x9 grid, 3 blocks of 3x3, identify pattern -> color
# ============================================================
def solve_17cae0c1(grid):
    R = len(grid)
    C = len(grid[0])

    def block_to_color(block):
        flat = [block[r][c] for r in range(3) for c in range(3)]
        count = sum(1 for x in flat if x != 0)
        if count == 1:
            return 4
        elif count == 3:
            if flat[0] and flat[1] and flat[2]:
                return 6
            elif flat[6] and flat[7] and flat[8]:
                return 1
            else:
                return 9
        elif count == 8:
            return 3
        else:
            return count

    out = [[0]*C for _ in range(R)]
    for bc in range(3):
        block = []
        for r in range(3):
            row = []
            for c in range(3):
                row.append(grid[r][bc*3 + c])
            block.append(row)
        color = block_to_color(block)
        for r in range(3):
            for c in range(3):
                out[r][bc*3 + c] = color
    return out

# ============================================================
# 195ba7dc - Split at 2-column, OR both halves -> 1
# ============================================================
def solve_195ba7dc(grid):
    R = len(grid)
    C = len(grid[0])
    col2 = None
    for c in range(C):
        if all(grid[r][c] == 2 for r in range(R)):
            col2 = c
            break
    left = [row[:col2] for row in grid]
    right = [row[col2+1:] for row in grid]
    outC = len(left[0])
    out = [[0]*outC for _ in range(R)]
    for r in range(R):
        for c in range(outC):
            if left[r][c] == 7 or right[r][c] == 7:
                out[r][c] = 1
    return out

# ============================================================
# 0e671a1a - Draw L-shaped paths between pairs of colored dots
# ============================================================
def solve_0e671a1a(grid):
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    dots = {grid[r][c]: (r,c) for r in range(R) for c in range(C) if grid[r][c] != 0}

    if 4 not in dots or 3 not in dots or 2 not in dots:
        return grid

    src = dots[4]
    d3 = dots[3]
    d2 = dots[2]
    sr, sc = src

    # Path to 3 (H-first): along row src_r to col d3_c, then col d3_c to row d3_r
    tr, tc = d3
    step_c = 1 if tc > sc else -1
    for c in range(sc + step_c, tc + step_c, step_c):
        if out[sr][c] == 0:
            out[sr][c] = 5
    step_r = 1 if tr > sr else -1
    for r in range(sr + step_r, tr, step_r):
        if out[r][tc] == 0:
            out[r][tc] = 5

    # Path to 2 (V-first): along col src_c to row d2_r, then row d2_r to col d2_c
    tr2, tc2 = d2
    step_r = 1 if tr2 > sr else -1
    for r in range(sr + step_r, tr2 + step_r, step_r):
        if out[r][sc] == 0:
            out[r][sc] = 5
    step_c = 1 if tc2 > sc else -1
    for c in range(sc + step_c, tc2, step_c):
        if out[tr2][c] == 0:
            out[tr2][c] = 5

    return out

# ============================================================
# 137f0df0 - Fill gaps between 5-blocks: interior gaps -> 2, outer gaps -> 1
# ============================================================
def solve_137f0df0(grid):
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    rows_with_5 = sorted(set(r for r in range(R) for c in range(C) if grid[r][c] == 5))
    cols_with_5 = sorted(set(c for r in range(R) for c in range(C) if grid[r][c] == 5))

    min_r5, max_r5 = min(rows_with_5), max(rows_with_5)
    min_c5, max_c5 = min(cols_with_5), max(cols_with_5)

    rows_5_set = set(rows_with_5)
    cols_5_set = set(cols_with_5)
    gap_rows = set(r for r in range(min_r5, max_r5+1) if r not in rows_5_set)
    gap_cols = set(c for c in range(min_c5, max_c5+1) if c not in cols_5_set)

    for r in range(R):
        for c in range(C):
            if grid[r][c] != 0:
                continue
            r_in_5 = r in rows_5_set
            c_in_5 = c in cols_5_set
            r_in_gap = r in gap_rows
            c_in_gap = c in gap_cols

            if r_in_5 and c_in_gap:
                out[r][c] = 2
            elif c_in_5 and r_in_gap:
                out[r][c] = 2
            elif r_in_gap and c_in_gap:
                out[r][c] = 2
            elif r_in_gap and not c_in_5:
                out[r][c] = 1
            elif c_in_gap and not r_in_5:
                out[r][c] = 1
    return out

# ============================================================
# 18286ef8 - Move 9 within 5-block toward 6, replace 6 with 9
# ============================================================
def solve_18286ef8(grid):
    import math
    R, C = len(grid), len(grid[0])
    nine_pos = six_pos = None
    five_block = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 9: nine_pos = (r, c)
            elif grid[r][c] == 6: six_pos = (r, c)
            elif grid[r][c] == 5: five_block.append((r, c))
    if not nine_pos or not six_pos:
        return [row[:] for row in grid]
    dy = six_pos[0] - nine_pos[0]
    dx = six_pos[1] - nine_pos[1]
    target_angle = math.atan2(dy, dx)
    best = None
    best_diff = float('inf')
    for r, c in five_block:
        dr = r - nine_pos[0]
        dc = c - nine_pos[1]
        if abs(dr) <= 1 and abs(dc) <= 1 and (dr != 0 or dc != 0):
            angle = math.atan2(dr, dc)
            diff = abs(angle - target_angle)
            if diff > math.pi: diff = 2*math.pi - diff
            if diff < best_diff:
                best_diff = diff
                best = (r, c)
    out = [row[:] for row in grid]
    out[nine_pos[0]][nine_pos[1]] = 5
    if best:
        out[best[0]][best[1]] = 9
    out[six_pos[0]][six_pos[1]] = 9
    return out

# ============================================================
# Test all solvers
# ============================================================

solvers = {
    "00576224": solve_00576224,
    "007bbfb7": solve_007bbfb7,
    "009d5c81": solve_009d5c81,
    "00d62c1b": solve_00d62c1b,
    "00dbd492": solve_00dbd492,
    "03560426": solve_03560426,
    "0520fde7": solve_0520fde7,
    "0692e18c": solve_0692e18c,
    "08ed6ac7": solve_08ed6ac7,
    "0a2355a6": solve_0a2355a6,
    "0b17323b": solve_0b17323b,
    "0bb8deee": solve_0bb8deee,
    "0becf7df": solve_0becf7df,
    "0c786b71": solve_0c786b71,
    "0c9aba6e": solve_0c9aba6e,
    "0ca9ddb6": solve_0ca9ddb6,
    "0d3d703e": solve_0d3d703e,
    "0d87d2a6": solve_0d87d2a6,
    "0e671a1a": solve_0e671a1a,
    "0f63c0b9": solve_0f63c0b9,
    "12997ef3": solve_12997ef3,
    "137f0df0": solve_137f0df0,
    "140c817e": solve_140c817e,
    "14b8e18c": solve_14b8e18c,
    "15696249": solve_15696249,
    "178fcbfb": solve_178fcbfb,
    "17cae0c1": solve_17cae0c1,
    "195ba7dc": solve_195ba7dc,
    "18286ef8": solve_18286ef8,
}

solutions = {}
for tid, solver in solvers.items():
    try:
        if test_solver(tid, solver):
            outputs = get_test_outputs(tid, solver)
            solutions[tid] = outputs
            print(f"PASS: {tid}")
        else:
            print(f"FAIL: {tid}")
    except Exception as e:
        print(f"ERROR: {tid}: {e}")

out_path = "C:/Users/atchi/h4-polytopic-attention/data/arc2_solutions_train_aa.json"
with open(out_path, 'w') as f:
    json.dump(solutions, f)

print(f"\nTotal passing: {len(solutions)}/{len(solvers)} solvers tested")
print(f"Saved to {out_path}")
