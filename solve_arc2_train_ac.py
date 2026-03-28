import json
import os
import copy
from collections import Counter, defaultdict

DATA_DIR = "data/arc2"
OUTPUT_FILE = "data/arc2_solutions_train_ac.json"

TASK_IDS = "5783df64,5792cb4d,57edb29d,5833af48,58743b76,58c02a16,58e15b12,59341089,5a5a2103,5a719d11,5ad8a7c0,5adee1b2,5af49b42,5b37cb25,5b526a93,5b692c0f,5b6cbef5,5bd6f4ac,5d2a5c43,5d588b4d,5e6bbc0b,5ecac7f7,5ffb2104,60a26a3e,60c09cac,60d73be6,6150a2bd,6165ea8f,626c0bcc,62ab2642,62b74c02,6350f1f4,639f5a19,642248e4,642d658d,64a7c07e,652646ff,668eec9a,66ac4c3b,66e6c45b,66f2d22f,67636eac,67a3c6ac,67c52801,689c358e,68b16354,68b67ca3,68bc2e87,692cd3b6,695367ec,696d4842,69889d6e,6a11f6da,6a980be1".split(",")

def load_task(tid):
    with open(os.path.join(DATA_DIR, f"{tid}.json")) as f:
        return json.load(f)

def test_solve(task, solve_fn):
    for pair in task["train"]:
        try:
            result = solve_fn(pair["input"])
            if result != pair["output"]:
                return False
        except Exception:
            return False
    return True

def get_test_outputs(task, solve_fn):
    outputs = []
    for pair in task["test"]:
        try:
            outputs.append(solve_fn(pair["input"]))
        except Exception:
            return None
    return outputs

# ========== SOLVER FUNCTIONS ==========

def solve_5783df64(grid):
    """9x9/6x6 -> 3x3: each non-zero value in a block, extract by block position"""
    R, C = len(grid), len(grid[0])
    bh, bw = R // 3, C // 3
    out = [[0]*3 for _ in range(3)]
    for r in range(R):
        for c in range(C):
            if grid[r][c] != 0:
                br, bc = r // bh, c // bw
                out[br][bc] = grid[r][c]
    return out

def solve_5792cb4d(grid):
    """Reverse values along the path of non-bg cells, positions stay same"""
    bg = 8
    R, C = len(grid), len(grid[0])
    cells = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg:
                cells.append((r, c))
    if not cells:
        return [row[:] for row in grid]
    # Order cells by path connectivity (BFS/DFS from endpoint)
    # Find endpoints: cells with only 1 neighbor
    adj = defaultdict(list)
    cell_set = set(cells)
    for r, c in cells:
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if (nr, nc) in cell_set:
                adj[(r,c)].append((nr,nc))
    # Find endpoint (degree 1)
    endpoints = [c for c in cells if len(adj[c]) <= 1]
    if not endpoints:
        endpoints = [cells[0]]
    # BFS to order
    start = endpoints[0]
    visited = set()
    ordered = []
    queue = [start]
    visited.add(start)
    while queue:
        node = queue.pop(0)
        ordered.append(node)
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    # Get values in order
    vals = [grid[r][c] for r, c in ordered]
    vals_rev = vals[::-1]
    out = [[bg]*C for _ in range(R)]
    for i, (r, c) in enumerate(ordered):
        out[r][c] = vals_rev[i]
    return out

def solve_68b67ca3(grid):
    """6x6 -> 3x3: sample every other row and column"""
    out = [[0]*3 for _ in range(3)]
    for r in range(3):
        for c in range(3):
            out[r][c] = grid[r*2][c*2]
    return out

def solve_6150a2bd(grid):
    """3x3 -> 3x3: rotate 180"""
    R, C = len(grid), len(grid[0])
    return [[grid[R-1-r][C-1-c] for c in range(C)] for r in range(R)]

def solve_68b16354(grid):
    """Flip vertically"""
    return grid[::-1]

def solve_66e6c45b(grid):
    """4x4: move inner 2x2 to corners"""
    out = [[0]*4 for _ in range(4)]
    out[0][0] = grid[1][1]
    out[0][3] = grid[1][2]
    out[3][0] = grid[2][1]
    out[3][3] = grid[2][2]
    return out

def solve_60c09cac(grid):
    """Scale up by 2x"""
    R, C = len(grid), len(grid[0])
    out = [[0]*(C*2) for _ in range(R*2)]
    for r in range(R):
        for c in range(C):
            out[r*2][c*2] = grid[r][c]
            out[r*2][c*2+1] = grid[r][c]
            out[r*2+1][c*2] = grid[r][c]
            out[r*2+1][c*2+1] = grid[r][c]
    return out

def solve_67a3c6ac(grid):
    """Reverse each row (horizontal flip)"""
    return [row[::-1] for row in grid]

def solve_5d2a5c43(grid):
    """Left/right halves separated by column of 1s, OR them -> color 8"""
    R, C = len(grid), len(grid[0])
    sep = -1
    for c in range(C):
        if all(grid[r][c] == 1 for r in range(R)):
            sep = c
            break
    left = [row[:sep] for row in grid]
    right = [row[sep+1:] for row in grid]
    w = len(left[0])
    out = [[0]*w for _ in range(R)]
    for r in range(R):
        for c in range(w):
            if left[r][c] != 0 or right[r][c] != 0:
                out[r][c] = 8
    return out

def solve_59341089(grid):
    """3x3 -> 3x12: each row -> reversed+original palindrome, repeated 2x"""
    out = []
    for r in range(len(grid)):
        row = grid[r]
        palindrome = row[::-1] + row
        out.append(palindrome * 2)
    return out

def solve_5b6cbef5(grid):
    """4x4 -> 16x16: fractal - non-zero cells get pattern, zero cells get all-zero block"""
    R, C = len(grid), len(grid[0])
    out = [[0]*(C*C) for _ in range(R*R)]
    for br in range(R):
        for bc in range(C):
            if grid[br][bc] != 0:
                for r in range(R):
                    for c in range(C):
                        out[br*R+r][bc*C+c] = grid[r][c]
    return out

def solve_60d73be6(grid):
    """Reflect across axis lines - mirror non-bg content across cross"""
    R, C = len(grid), len(grid[0])
    bg = 7
    hline = vline = -1
    for r in range(R):
        vals = [grid[r][c] for c in range(C)]
        if len(set(vals)) == 1 and vals[0] != bg:
            hline = r
            break
    for c in range(C):
        vals = [grid[r][c] for r in range(R)]
        if len(set(vals)) == 1 and vals[0] != bg:
            vline = c
            break

    out = [row[:] for row in grid]
    if hline >= 0 and vline >= 0:
        for r in range(R):
            if r == hline: continue
            for c in range(C):
                if c == vline: continue
                if grid[r][c] != bg:
                    mc = 2*vline - c
                    mr = 2*hline - r
                    if 0 <= mc < C and mc != vline:
                        out[r][mc] = grid[r][c]
                    if 0 <= mr < R and mr != hline:
                        out[mr][c] = grid[r][c]
                    if 0 <= mr < R and 0 <= mc < C:
                        out[mr][mc] = grid[r][c]
    elif hline >= 0:
        for r in range(R):
            if r == hline: continue
            for c in range(C):
                mr = 2*hline - r
                if 0 <= mr < R and mr != hline and grid[r][c] != bg:
                    out[mr][c] = grid[r][c]
    elif vline >= 0:
        for r in range(R):
            for c in range(C):
                if c == vline: continue
                mc = 2*vline - c
                if 0 <= mc < C and mc != vline and grid[r][c] != bg:
                    out[r][mc] = grid[r][c]
    return out

def solve_62b74c02(grid):
    """Pattern on left, zero-fill rest -> copy pattern to right side, fill middle with edge color"""
    R, C = len(grid), len(grid[0])
    # Find pattern width
    pw = 0
    for c in range(C):
        if any(grid[r][c] != 0 for r in range(R)):
            pw = c + 1
    out = [row[:] for row in grid]
    for r in range(R):
        fill_color = grid[r][0]
        for c in range(pw, C - pw):
            out[r][c] = fill_color
        # Copy (not mirror) pattern to right side
        for c in range(pw):
            out[r][C - pw + c] = grid[r][c]
    return out

def solve_58743b76(grid):
    """2x2 color key in corner, replace marker color based on quadrant position"""
    R, C = len(grid), len(grid[0])
    bg = 8
    # Find 2x2 key
    key = None
    key_r = key_c = -1
    corners = [(0, C-2), (0, 0), (R-2, 0), (R-2, C-2)]
    for kr, kc in corners:
        if (grid[kr][kc] != bg and grid[kr][kc+1] != bg and
            grid[kr+1][kc] != bg and grid[kr+1][kc+1] != bg):
            key = [[grid[kr][kc], grid[kr][kc+1]], [grid[kr+1][kc], grid[kr+1][kc+1]]]
            key_r, key_c = kr, kc
            break
    if not key:
        return grid

    # Find main area bounds (non-bg, non-key)
    main_cells = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg and not (key_r <= r <= key_r+1 and key_c <= c <= key_c+1):
                main_cells.append((r, c))
    if not main_cells:
        return grid
    min_r = min(r for r,c in main_cells)
    max_r = max(r for r,c in main_cells)
    min_c = min(c for r,c in main_cells)
    max_c = max(c for r,c in main_cells)
    mid_r = (min_r + max_r) / 2
    mid_c = (min_c + max_c) / 2

    out = [row[:] for row in grid]
    for r, c in main_cells:
        v = grid[r][c]
        if v != 0:
            qr = 0 if r < mid_r else 1
            qc = 0 if c < mid_c else 1
            out[r][c] = key[qr][qc]
    return out

def solve_5af49b42(grid):
    """Sequences at edges, scatter points -> place sequence at point position"""
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    edge_seqs = []
    for row_idx in [0, R-1]:
        row = grid[row_idx]
        seq = []
        for c in range(C):
            if row[c] != 0:
                seq.append((c, row[c]))
            else:
                if seq:
                    edge_seqs.append(([v for _,v in seq], True))
                    seq = []
        if seq:
            edge_seqs.append(([v for _,v in seq], True))
    for col_idx in [0, C-1]:
        col = [grid[r][col_idx] for r in range(R)]
        seq = []
        for r in range(R):
            if col[r] != 0:
                seq.append((r, col[r]))
            else:
                if seq:
                    edge_seqs.append(([v for _,v in seq], False))
                    seq = []
        if seq:
            edge_seqs.append(([v for _,v in seq], False))
    color_to_seq = {}
    for seq_vals, is_horiz in edge_seqs:
        for v in seq_vals:
            if v not in color_to_seq:
                color_to_seq[v] = (seq_vals, is_horiz)
    edge_rows = set()
    for row_idx in [0, R-1]:
        if any(grid[row_idx][c] != 0 for c in range(C)):
            edge_rows.add(row_idx)
    edge_cols = set()
    for col_idx in [0, C-1]:
        if any(grid[r][col_idx] != 0 for r in range(R)):
            edge_cols.add(col_idx)
    for r in range(R):
        if r in edge_rows:
            continue
        for c in range(C):
            if c in edge_cols:
                continue
            if grid[r][c] != 0:
                color = grid[r][c]
                if color in color_to_seq:
                    seq_vals, is_horiz = color_to_seq[color]
                    idx = seq_vals.index(color)
                    if is_horiz:
                        sc = c - idx
                        for si, sv in enumerate(seq_vals):
                            nc = sc + si
                            if 0 <= nc < C:
                                out[r][nc] = sv
                    else:
                        sr = r - idx
                        for si, sv in enumerate(seq_vals):
                            nr = sr + si
                            if 0 <= nr < R:
                                out[nr][c] = sv
    return out

def solve_6a11f6da(grid):
    """15x5 -> 5x5: Three 5x5 blocks stacked. Priority: block2 > block0 > block1"""
    R, C = len(grid), len(grid[0])
    h = R // 3
    blocks = [grid[i*h:(i+1)*h] for i in range(3)]
    out = [[0]*C for _ in range(h)]
    for r in range(h):
        for c in range(C):
            v0 = blocks[0][r][c]
            v1 = blocks[1][r][c]
            v2 = blocks[2][r][c]
            if v2 != 0:
                out[r][c] = v2
            elif v0 != 0:
                out[r][c] = v0
            elif v1 != 0:
                out[r][c] = v1
    return out

def solve_5bd6f4ac(grid):
    """9x9 -> 3x3: always the top-right block (0,2)"""
    return [[grid[r][c] for c in range(6, 9)] for r in range(3)]

def solve_69889d6e(grid):
    """Draw diagonal line from 2 going up-right (or toward edge), with double-width trail"""
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    pos1 = pos2 = None
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 1:
                pos1 = (r, c)
            elif grid[r][c] == 2:
                pos2 = (r, c)

    if pos2 is None:
        return grid

    r2, c2 = pos2
    # Determine direction: up-right diagonal from pos2
    # Draw the line going up from pos2 at 45 degrees to the right
    # At pos2: single 2. Then going up-right, each step places 2 at (r, c-1) and (r, c)

    # If pos1 exists, draw only up to pos1, then stop
    end_r = pos1[0] if pos1 else 0

    r, c = r2 - 1, c2 + 1
    while r >= 0 and c < C:
        # Place two cells: (r, c-1) and (r, c)
        if c - 1 >= 0:
            out[r][c-1] = 2
        out[r][c] = 2
        if pos1 and r == end_r:
            break
        r -= 1
        c += 1

    return out

def solve_5ffb2104(grid):
    """Gravity: push each connected component right until it hits wall or another component"""
    R, C = len(grid), len(grid[0])
    visited = [[False]*C for _ in range(R)]
    components = []

    def bfs(sr, sc):
        queue = [(sr, sc)]
        visited[sr][sc] = True
        cells = [(sr, sc)]
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and grid[nr][nc] != 0:
                    visited[nr][nc] = True
                    queue.append((nr, nc))
                    cells.append((nr, nc))
        return cells

    for r in range(R):
        for c in range(C):
            if grid[r][c] != 0 and not visited[r][c]:
                components.append(bfs(r, c))

    # Sort by max_c descending (rightmost first - these block others)
    components.sort(key=lambda comp: -max(c for r, c in comp))

    out = [[0]*C for _ in range(R)]
    for comp in components:
        max_shift = C
        for r, c in comp:
            space = 0
            for nc in range(c+1, C):
                if out[r][nc] != 0:
                    break
                space += 1
            max_shift = min(max_shift, space)
        for r, c in comp:
            nc = c + max_shift
            if 0 <= nc < C:
                out[r][nc] = grid[r][c]
    return out

def solve_64a7c07e(grid):
    """Move each connected component of 8s to the right by its width"""
    R, C = len(grid), len(grid[0])
    visited = [[False]*C for _ in range(R)]
    components = []

    def bfs(sr, sc):
        queue = [(sr, sc)]
        visited[sr][sc] = True
        cells = [(sr, sc)]
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and grid[nr][nc] != 0:
                    visited[nr][nc] = True
                    queue.append((nr, nc))
                    cells.append((nr, nc))
        return cells

    for r in range(R):
        for c in range(C):
            if grid[r][c] != 0 and not visited[r][c]:
                components.append(bfs(r, c))

    out = [[0]*C for _ in range(R)]
    for cells in components:
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)
        width = max_c - min_c + 1
        for r, c in cells:
            nc = c + width
            if 0 <= nc < C:
                out[r][nc] = grid[r][c]
    return out

def solve_66f2d22f(grid):
    """4x14 -> 4x7: XOR of left and right halves, mark with 5"""
    R, C = len(grid), len(grid[0])
    half = C // 2
    out = [[0]*half for _ in range(R)]
    for r in range(R):
        for c in range(half):
            l = 1 if grid[r][c] != 0 else 0
            ri = 1 if grid[r][c+half] != 0 else 0
            if l == 0 and ri == 0:
                out[r][c] = 5
            else:
                out[r][c] = 0
    return out

def solve_5ad8a7c0(grid):
    """Diamond of 2s: fill between 2s on rows where gap equals neighbor row gaps"""
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    # Get left/right positions of 2s per row
    row_info = []
    for r in range(R):
        positions = [c for c in range(C) if grid[r][c] == 2]
        if len(positions) >= 2:
            row_info.append((r, positions[0], positions[-1]))
        else:
            row_info.append((r, -1, -1))

    # Fill row if it has a "concave" section relative to the diamond
    # Fill between 2s if the gap does NOT monotonically change (flat section)
    for i, (r, left, right) in enumerate(row_info):
        if left < 0:
            continue
        gap = right - left
        # Check neighbors: if adjacent row has same gap, fill both
        fill = False
        for j, (r2, l2, r2r) in enumerate(row_info):
            if j == i:
                continue
            if l2 >= 0 and abs(j - i) == 1:
                gap2 = r2r - l2
                if gap == gap2:
                    fill = True
                    break
        if fill:
            for c in range(left, right + 1):
                out[r][c] = 2

    return out

def solve_68bc2e87(grid):
    """Nested rectangles -> list of border colors from outside in"""
    R, C = len(grid), len(grid[0])
    bg = 8
    colors = []
    # Scan inward finding rectangle borders
    r_min, r_max, c_min, c_max = 0, R-1, 0, C-1
    while r_min <= r_max and c_min <= c_max:
        found = None
        for r in range(r_min, r_max+1):
            for c in range(c_min, c_max+1):
                if grid[r][c] != bg:
                    found = grid[r][c]
                    break
            if found:
                break
        if not found:
            r_min += 1; r_max -= 1; c_min += 1; c_max -= 1
            continue
        if found not in colors:
            colors.append(found)
        # Find bounds of this color
        rr_min = rr_max = cc_min = cc_max = None
        for r in range(r_min, r_max+1):
            for c in range(c_min, c_max+1):
                if grid[r][c] == found:
                    if rr_min is None or r < rr_min: rr_min = r
                    if rr_max is None or r > rr_max: rr_max = r
                    if cc_min is None or c < cc_min: cc_min = c
                    if cc_max is None or c > cc_max: cc_max = c
        r_min = rr_min + 1
        r_max = rr_max - 1
        c_min = cc_min + 1
        c_max = cc_max - 1
    return [[c] for c in colors]

def solve_642d658d(grid):
    """Large grid -> 1x1: find the most frequent non-zero color"""
    R, C = len(grid), len(grid[0])
    counts = Counter()
    for r in range(R):
        for c in range(C):
            v = grid[r][c]
            if v != 0:
                counts[v] += 1
    if counts:
        return [[counts.most_common(1)[0][0]]]
    return [[0]]

def solve_695367ec(grid):
    """Small grid -> 15x15: tile with the pattern"""
    R, C = len(grid), len(grid[0])
    # Input 2x2 -> 15x15, 3x3 -> 15x15, 4x4 -> 15x15
    # The pattern: make a 15x15 grid where every Nth row/col has the color
    # For 2x2 input [[8,8],[8,8]]: every 3rd row/col is 8, rest is 0
    # The output has a grid pattern with period (R+1)? No...
    # 2x2 input: period 3 (rows 2,5,8,11,14 are all 8), others have 8 at cols 2,5,8,11,14
    # Actually: columns 2,5,8,11,14 (period 3) are fully 8, and rows 2,5,8,11,14 same
    # That's a grid with spacing 3 = input_size + 1? 2+1=3 yes
    # For 3x3: spacing would be 4? Let me check
    # For 4x4: spacing would be 5? 15/5=3 cells... need to verify

    # Actually looking at 2x2 -> 15: spacing = input_size + 1 = 3
    # Grid lines at positions 2, 5, 8, 11, 14 (i.e., n*3 + 2 for n=0..4)
    # Wait: 2,5,8,11,14 -> differences of 3. Starting from 2.
    # For general: starting at R, spacing R+1

    # For 2x2 [[8,8],[8,8]]:
    # Colors at grid lines
    s = R + 1  # spacing (for 2x2, s=3)
    color = grid[0][0]  # assume all same color
    out_size = 15
    out = [[0]*out_size for _ in range(out_size)]

    # Grid lines: horizontal at r*s + (R-1), vertical at same
    # For 2x2: at 2, 5, 8, 11, 14
    grid_positions = [i * s + (R - 1) for i in range(out_size)]
    grid_positions = [p for p in grid_positions if p < out_size]

    # Actually let me just check: for [[8,8],[8,8]], output rows 0,1,3,4,6,7,9,10,12,13 have
    # 8 at columns 2,5,8,11,14 only. Rows 2,5,8,11,14 are all 8.
    for r in range(out_size):
        for c in range(out_size):
            if r in grid_positions or c in grid_positions:
                out[r][c] = color
    return out

def solve_67c52801(grid):
    """Drop colored objects (gravity) onto the bottom static structure"""
    R, C = len(grid), len(grid[0])
    # Find the static structure at bottom (largest connected non-zero component touching bottom)
    # Find all connected components
    visited = [[False]*C for _ in range(R)]
    components = []

    def bfs(sr, sc):
        v = grid[sr][sc]
        queue = [(sr, sc)]
        visited[sr][sc] = True
        cells = [(sr, sc)]
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and grid[nr][nc] != 0:
                    visited[nr][nc] = True
                    queue.append((nr, nc))
                    cells.append((nr, nc))
        return cells

    for r in range(R):
        for c in range(C):
            if grid[r][c] != 0 and not visited[r][c]:
                cells = bfs(r, c)
                components.append(cells)

    if not components:
        return grid

    # Find the static "floor" component (touching bottom row, or the largest)
    floor_comp = None
    for comp in components:
        if any(r == R-1 for r, c in comp):
            if floor_comp is None or len(comp) > len(floor_comp):
                floor_comp = comp

    if floor_comp is None:
        return grid

    floor_set = set((r,c) for r,c in floor_comp)

    # The floor has holes (columns where floor has 0s in its top row)
    # Find floor bounding box
    floor_min_r = min(r for r,c in floor_comp)
    floor_max_r = max(r for r,c in floor_comp)

    # The other components need to drop into the holes in the floor
    out = [[0]*C for _ in range(R)]

    # Place floor
    for r, c in floor_comp:
        out[r][c] = grid[r][c]

    # For each non-floor component, drop it down
    for comp in components:
        if comp is floor_comp:
            continue
        # This is a floating object - drop it into a hole in the floor
        comp_min_r = min(r for r,c in comp)
        comp_max_r = max(r for r,c in comp)
        comp_min_c = min(c for r,c in comp)
        comp_max_c = max(c for r,c in comp)

        # Find where this object should land
        # Drop it down: find the highest floor cell directly below any cell of this component
        # The object needs to drop until it sits just above the floor or in a gap
        # Find the gap in the floor that matches this object's column span
        # For each column in the object, find the topmost floor cell below
        max_drop = R
        for r, c in comp:
            # Find first occupied cell below this cell
            for nr in range(r+1, R+1):
                if nr == R or (nr, c) in floor_set:
                    drop = nr - r - 1
                    max_drop = min(max_drop, drop)
                    break

        for r, c in comp:
            nr = r + max_drop
            if 0 <= nr < R:
                out[nr][c] = grid[r][c]

    return out

def solve_695367ec_v2(grid):
    """Small grid -> 15x15: create grid lines of the color"""
    R, C = len(grid), len(grid[0])
    color = None
    for r in range(R):
        for c in range(C):
            if grid[r][c] != 0:
                color = grid[r][c]
                break
        if color: break
    if not color:
        return [[0]*15 for _ in range(15)]

    # Period = R + 1 for square input
    s = R + 1
    out = [[0]*15 for _ in range(15)]
    for r in range(15):
        for c in range(15):
            if r % s >= s - R or c % s >= s - R:
                out[r][c] = color
    return out

def solve_5a5a2103(grid):
    """Grid divided by color 3 lines into cells. Each cell row has a color + pattern template. Tile pattern with each cell's color."""
    R, C = len(grid), len(grid[0])
    sep_color = 3
    # Find horizontal and vertical separator lines
    h_seps = []
    v_seps = []
    for r in range(R):
        if all(grid[r][c] == sep_color for c in range(C)):
            h_seps.append(r)
    for c in range(C):
        if all(grid[r][c] == sep_color for r in range(R)):
            v_seps.append(c)

    # Extract cell blocks
    h_bounds = [-1] + h_seps + [R]
    v_bounds = [-1] + v_seps + [C]

    def get_block(ri, ci):
        r_start = h_bounds[ri] + 1
        r_end = h_bounds[ri+1]
        c_start = v_bounds[ci] + 1
        c_end = v_bounds[ci+1]
        block = []
        for r in range(r_start, r_end):
            row = []
            for c in range(c_start, c_end):
                row.append(grid[r][c])
            block.append(row)
        return block, r_start, c_start

    n_row_blocks = len(h_seps) + 1
    n_col_blocks = len(v_seps) + 1

    # Find the template pattern (non-zero block in a specific row)
    # And find the color for each row
    # In the first column, each row-block has a colored square (like 4, 2, 8, 1)
    # Find one cell with a non-trivial pattern (not just a solid square)
    template = None
    template_ri = template_ci = -1
    for ri in range(n_row_blocks):
        for ci in range(n_col_blocks):
            block, _, _ = get_block(ri, ci)
            # Check if block has non-zero values that are not all the same
            vals = set()
            for row in block:
                for v in row:
                    if v != 0:
                        vals.add(v)
            if len(vals) > 1:
                template = block
                template_ri = ri
                template_ci = ci
                break
        if template:
            break

    if not template:
        return grid

    # The template has a pattern shape in one color (e.g., 6)
    # Each row-block's color replaces the template shape
    out = [row[:] for row in grid]

    # Get row colors (from the first column of cells)
    row_colors = []
    for ri in range(n_row_blocks):
        block, _, _ = get_block(ri, 0)
        color = 0
        for row in block:
            for v in row:
                if v != 0:
                    color = v
                    break
            if color:
                break
        row_colors.append(color)

    # The template pattern: extract the shape (positions of the template color)
    template_color = None
    for row in template:
        for v in row:
            if v != 0 and v not in row_colors:
                template_color = v
                break
        if template_color:
            break

    if not template_color:
        # Template color might be one of the row colors
        # Find the non-row-color in the template
        all_template_vals = set()
        for row in template:
            for v in row:
                if v != 0:
                    all_template_vals.add(v)
        non_row = all_template_vals - set(row_colors)
        if non_row:
            template_color = non_row.pop()

    if not template_color:
        return grid

    # Get template shape positions
    bh = len(template)
    bw = len(template[0])
    shape = []
    for r in range(bh):
        for c in range(bw):
            if template[r][c] == template_color:
                shape.append((r, c))

    # For each cell, apply the pattern
    for ri in range(n_row_blocks):
        color = row_colors[ri]
        for ci in range(n_col_blocks):
            block, r_start, c_start = get_block(ri, ci)
            # Clear and apply
            for r in range(bh):
                for c in range(bw):
                    rr = r_start + r
                    cc = c_start + c
                    if 0 <= rr < R and 0 <= cc < C:
                        out[rr][cc] = 0
            for r, c in shape:
                rr = r_start + r
                cc = c_start + c
                if 0 <= rr < R and 0 <= cc < C:
                    out[rr][cc] = color
    return out

def solve_6a980be1(grid):
    """Swap 0 and 8 in the grid"""
    R, C = len(grid), len(grid[0])
    out = [[0]*C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 0:
                out[r][c] = 8
            elif grid[r][c] == 8:
                out[r][c] = 0
            else:
                out[r][c] = grid[r][c]
    return out


def solve_60a26a3e(grid):
    """Connect diamonds (cross patterns) with lines of color 1"""
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    diamonds = []
    for r in range(1, R-1):
        for c in range(1, C-1):
            if (grid[r][c] == 0 and grid[r-1][c] != 0 and grid[r+1][c] != 0 and
                grid[r][c-1] != 0 and grid[r][c+1] != 0):
                diamonds.append((r, c))
    by_row = {}
    by_col = {}
    for r, c in diamonds:
        by_row.setdefault(r, []).append(c)
        by_col.setdefault(c, []).append(r)
    for row, cols in by_row.items():
        cols.sort()
        for i in range(len(cols) - 1):
            c1 = cols[i] + 1
            c2 = cols[i+1] - 1
            for c in range(c1 + 1, c2):
                out[row][c] = 1
    for col, rows in by_col.items():
        rows.sort()
        for i in range(len(rows) - 1):
            r1 = rows[i] + 1
            r2 = rows[i+1] - 1
            for r in range(r1 + 1, r2):
                out[r][col] = 1
    return out


def solve_642248e4(grid):
    """Markers near borders: place border color one step toward nearest border"""
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    borders = {}
    for r in range(R):
        if all(grid[r][c] == grid[r][0] and grid[r][c] != 0 for c in range(C)):
            borders[('r', r)] = grid[r][0]
    for c in range(C):
        if all(grid[r][c] == grid[0][c] and grid[r][c] != 0 for r in range(R)):
            borders[('c', c)] = grid[0][c]
    border_colors = set(borders.values())
    markers = []
    for r in range(R):
        for c in range(C):
            v = grid[r][c]
            if v != 0 and v not in border_colors:
                markers.append((r, c, v))
    for mr, mc, mv in markers:
        best_dist = float('inf')
        best_color = 0
        best_dr = best_dc = 0
        for (btype, bidx), bcolor in borders.items():
            if btype == 'r':
                dist = abs(mr - bidx)
                if dist < best_dist:
                    best_dist = dist
                    best_color = bcolor
                    best_dr = -1 if bidx < mr else 1
                    best_dc = 0
            else:
                dist = abs(mc - bidx)
                if dist < best_dist:
                    best_dist = dist
                    best_color = bcolor
                    best_dr = 0
                    best_dc = -1 if bidx < mc else 1
        nr, nc = mr + best_dr, mc + best_dc
        if 0 <= nr < R and 0 <= nc < C:
            out[nr][nc] = best_color
    return out


def solve_695367ec(grid):
    """NxN grid -> 15x15: create grid lines with period N+1"""
    N = len(grid)
    color = grid[0][0]
    s = N + 1
    out = [[0]*15 for _ in range(15)]
    for r in range(15):
        for c in range(15):
            if r % s == N or c % s == N:
                out[r][c] = color
    return out

# ========== SOLVER REGISTRY ==========

solvers = {
    "5783df64": solve_5783df64,
    "5792cb4d": solve_5792cb4d,
    "68b67ca3": solve_68b67ca3,
    "6150a2bd": solve_6150a2bd,
    "68b16354": solve_68b16354,
    "66e6c45b": solve_66e6c45b,
    "60c09cac": solve_60c09cac,
    "67a3c6ac": solve_67a3c6ac,
    "5d2a5c43": solve_5d2a5c43,
    "59341089": solve_59341089,
    "5b6cbef5": solve_5b6cbef5,
    "60d73be6": solve_60d73be6,
    "62b74c02": solve_62b74c02,
    "58743b76": solve_58743b76,
    "5af49b42": solve_5af49b42,
    "6a11f6da": solve_6a11f6da,
    "5bd6f4ac": solve_5bd6f4ac,
    "5ffb2104": solve_5ffb2104,
    "64a7c07e": solve_64a7c07e,
    "66f2d22f": solve_66f2d22f,
    "67c52801": solve_67c52801,
    "60a26a3e": solve_60a26a3e,
    "642248e4": solve_642248e4,
    "695367ec": solve_695367ec,
}

# ========== GENERIC SOLVERS ==========

generic_solvers = [
    lambda g: [row[:] for row in g],  # identity
    lambda g: [[g[len(g)-1-r][len(g[0])-1-c] for c in range(len(g[0]))] for r in range(len(g))],  # rotate 180
    lambda g: [row[::-1] for row in g],  # flip horizontal
    lambda g: g[::-1],  # flip vertical
    lambda g: [[g[r][c] for r in range(len(g))] for c in range(len(g[0]))],  # transpose
]

def main():
    solutions = {}

    for tid in TASK_IDS:
        task = load_task(tid)
        solved = False

        if tid in solvers:
            fn = solvers[tid]
            if test_solve(task, fn):
                outputs = get_test_outputs(task, fn)
                if outputs:
                    solutions[tid] = outputs
                    solved = True
                    print(f"  {tid}: SOLVED (specific)")

        if not solved:
            for gfn in generic_solvers:
                t0 = task["train"][0]
                try:
                    result = gfn(t0["input"])
                    if len(result) != len(t0["output"]) or len(result[0]) != len(t0["output"][0]):
                        continue
                except:
                    continue
                if test_solve(task, gfn):
                    outputs = get_test_outputs(task, gfn)
                    if outputs:
                        solutions[tid] = outputs
                        solved = True
                        print(f"  {tid}: SOLVED (generic)")
                    break

        if not solved:
            print(f"  {tid}: UNSOLVED")

    print(f"\nTotal solved: {len(solutions)}/{len(TASK_IDS)}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(solutions, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
