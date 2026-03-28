import json
import copy
from collections import Counter, defaultdict

DATA = "C:/Users/atchi/h4-polytopic-attention/data/arc1"
OUT = "C:/Users/atchi/h4-polytopic-attention/data/arc_python_solutions_b20.json"

def load(tid):
    with open(f"{DATA}/{tid}.json") as f:
        return json.load(f)

# ========== c3e719e8 ==========
# 3x3->9x9: place input at block positions where input cell == most frequent value
def solve_c3e719e8(grid):
    h, w = len(grid), len(grid[0])
    out = [[0]*w*3 for _ in range(h*3)]
    vals = [v for row in grid for v in row]
    most_common = Counter(vals).most_common(1)[0][0]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == most_common:
                for dr in range(h):
                    for dc in range(w):
                        out[r*h+dr][c*w+dc] = grid[dr][dc]
    return out

# ========== c8cbb738 ==========
# Multiple sets of 4 non-bg dots in grid, each forming a rectangle.
# Output overlays all sets centered, using their relative offsets from their own centers.
def solve_c8cbb738(grid):
    h, w = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]

    # Group non-bg points by color
    color_points = defaultdict(list)
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg:
                color_points[grid[r][c]].append((r, c))

    # For each color, compute center and relative offsets
    all_offsets = []  # list of (color, dr, dc) relative to each color's own center
    max_dr = 0
    max_dc = 0

    for color, pts in color_points.items():
        center_r = sum(r for r, c in pts) / len(pts)
        center_c = sum(c for r, c in pts) / len(pts)
        for r, c in pts:
            dr = round(r - center_r)
            dc = round(c - center_c)
            all_offsets.append((color, dr, dc))
            max_dr = max(max_dr, abs(dr))
            max_dc = max(max_dc, abs(dc))

    out_h = max_dr * 2 + 1
    out_w = max_dc * 2 + 1
    out = [[bg] * out_w for _ in range(out_h)]

    cr = out_h // 2
    cc = out_w // 2

    for color, dr, dc in all_offsets:
        r, c = cr + dr, cc + dc
        if 0 <= r < out_h and 0 <= c < out_w:
            out[r][c] = color

    return out

# ========== c9e6f938 ==========
# 3x3 -> 3x6: concatenate with horizontal mirror
def solve_c9e6f938(grid):
    return [row + row[::-1] for row in grid]

# ========== c9f8e694 ==========
# Replace all 5s with the value from column 0 of same row
def solve_c9f8e694(grid):
    out = copy.deepcopy(grid)
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 5:
                out[r][c] = grid[r][0]
    return out

# ========== caa06a1f ==========
# Checkerboard with border of fill color -> complete checkerboard, shifted by 1
def solve_caa06a1f(grid):
    h, w = len(grid), len(grid[0])

    # Find the checkerboard pattern and period
    # The border color fills some rows/cols
    border_color = grid[h-1][w-1]

    # Find pattern from first row (before border)
    pattern_row = []
    for c in range(w):
        if grid[0][c] == border_color:
            break
        pattern_row.append(grid[0][c])

    # Find period
    period = 0
    for p in range(1, len(pattern_row)+1):
        pat = pattern_row[:p]
        ok = True
        for i in range(len(pattern_row)):
            if pattern_row[i] != pat[i % p]:
                ok = False
                break
        if ok:
            period = p
            break

    # Find row shift (how much the pattern shifts per row)
    row_shift = 0
    if h > 1 and grid[1][0] != border_color:
        for s in range(period):
            if grid[1][0] == pattern_row[s % period]:
                row_shift = s
                break

    # Rule: output[r][c] = pattern[(c + input_row_offset[r] + 1) % period]
    # Detect input row offsets from the actual input data
    pat = pattern_row[:period]

    # Build a lookup for each row's offset from the input
    row_offsets = []
    for r in range(h):
        if grid[r][0] != border_color:
            # Find offset
            for off in range(period):
                if grid[r][0] == pat[off]:
                    row_offsets.append(off)
                    break
            else:
                row_offsets.append(0)
        else:
            row_offsets.append(None)  # border row

    # For border rows, extrapolate the offset pattern
    # Find the cycle of offsets from non-border rows
    non_border_offsets = [o for o in row_offsets if o is not None]
    # Find the row-offset period
    row_period = 1
    for rp in range(1, len(non_border_offsets)+1):
        pat_check = non_border_offsets[:rp]
        if all(non_border_offsets[i] == pat_check[i % rp] for i in range(len(non_border_offsets))):
            row_period = rp
            break
    offset_cycle = non_border_offsets[:row_period]

    out = [[0]*w for _ in range(h)]
    for r in range(h):
        if row_offsets[r] is not None:
            off = row_offsets[r]
        else:
            off = offset_cycle[r % row_period]
        for c in range(w):
            idx = (c + off + 1) % period
            out[r][c] = pat[idx]

    return out

# ========== cbded52d ==========
# 8x8 grid divided into 3x3 blocks of 2x2 cells by 0-separators
# Propagate non-1 values to same local position in blocks sharing block-row or block-col
def solve_cbded52d(grid):
    h, w = len(grid), len(grid[0])
    out = copy.deepcopy(grid)

    block_rows = [0, 3, 6]
    block_cols = [0, 3, 6]

    # Extract block values
    def get_block(bi, bj):
        br, bc = block_rows[bi], block_cols[bj]
        return [[out[br+dr][bc+dc] for dc in range(2)] for dr in range(2)]

    def set_block_cell(bi, bj, dr, dc, val):
        br, bc = block_rows[bi], block_cols[bj]
        out[br+dr][bc+dc] = val

    # Iterate to convergence
    changed = True
    while changed:
        changed = False
        # For each local position and each non-1 value
        for dr in range(2):
            for dc in range(2):
                # Collect which blocks have non-1 values at this local position
                val_blocks = defaultdict(set)
                for bi in range(3):
                    for bj in range(3):
                        v = get_block(bi, bj)[dr][dc]
                        if v != 0 and v != 1:
                            val_blocks[v].add((bi, bj))

                for v, blocks in val_blocks.items():
                    # Check block-rows: if 2+ blocks in same row have this value
                    for row_idx in range(3):
                        row_blocks = [(bi,bj) for bi,bj in blocks if bi == row_idx]
                        if len(row_blocks) >= 2:
                            # Propagate to all blocks in this row
                            for bj in range(3):
                                br, bc = block_rows[row_idx], block_cols[bj]
                                if out[br+dr][bc+dc] == 1:
                                    out[br+dr][bc+dc] = v
                                    changed = True

                    # Check block-cols
                    for col_idx in range(3):
                        col_blocks = [(bi,bj) for bi,bj in blocks if bj == col_idx]
                        if len(col_blocks) >= 2:
                            for bi in range(3):
                                br, bc = block_rows[bi], block_cols[col_idx]
                                if out[br+dr][bc+dc] == 1:
                                    out[br+dr][bc+dc] = v
                                    changed = True

    return out

# ========== cce03e0d ==========
# 3x3 -> 9x9: place input at block (r,c) where grid[r][c] == 2
def solve_cce03e0d(grid):
    h, w = len(grid), len(grid[0])
    out = [[0]*w*3 for _ in range(h*3)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 2:
                for dr in range(h):
                    for dc in range(w):
                        out[r*h+dr][c*w+dc] = grid[dr][dc]
    return out

# ========== cdecee7f ==========
# 10x10 sparse -> 3x3: collect non-zero values sorted by column, snake-fill into 3x3
def solve_cdecee7f(grid):
    h, w = len(grid), len(grid[0])
    vals = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                vals.append((c, r, grid[r][c]))
    vals.sort()
    values = [v[2] for v in vals]

    out = [[0]*3 for _ in range(3)]
    idx = 0
    for r in range(3):
        cols = range(3) if r % 2 == 0 else range(2, -1, -1)
        for c in cols:
            if idx < len(values):
                out[r][c] = values[idx]
                idx += 1
    return out

# ========== ce22a75a ==========
# 9x9 grid with 5s marking centers of 3x3 blocks -> fill those blocks with 1
def solve_ce22a75a(grid):
    h, w = len(grid), len(grid[0])
    out = [[0]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 5:
                br = (r // 3) * 3
                bc = (c // 3) * 3
                for dr in range(3):
                    for dc in range(3):
                        out[br+dr][bc+dc] = 1
    return out

# ========== ce4f8723 ==========
# OR of top and bottom 4x4 grids -> 3 where either is non-zero
def solve_ce4f8723(grid):
    top = [grid[r] for r in range(4)]
    bot = [grid[r] for r in range(5, 9)]
    out = [[0]*4 for _ in range(4)]
    for r in range(4):
        for c in range(4):
            if top[r][c] != 0 or bot[r][c] != 0:
                out[r][c] = 3
    return out

# ========== ce9e57f2 ==========
# Columns of 2s: top ceil(h/2) stay 2, bottom floor(h/2) become 8
def solve_ce9e57f2(grid):
    h, w = len(grid), len(grid[0])
    out = copy.deepcopy(grid)

    for c in range(w):
        # Find cells with 2 in this column
        twos = [(r, c) for r in range(h) if grid[r][c] == 2]
        if not twos:
            continue

        height = len(twos)
        num_twos = (height + 1) // 2  # ceil(h/2)

        # Top num_twos stay as 2, rest become 8
        for i, (r, _) in enumerate(twos):
            if i >= num_twos:
                out[r][c] = 8

    return out

# ========== cf98881b ==========
# 4x14 -> 4x4: overlay three 4x4 sections with priority 4 > 9 > 1
def solve_cf98881b(grid):
    A = [row[0:4] for row in grid]
    B = [row[5:9] for row in grid]
    C = [row[10:14] for row in grid]
    out = [[0]*4 for _ in range(4)]
    for r in range(4):
        for c in range(4):
            if A[r][c] == 4:
                out[r][c] = 4
            elif B[r][c] == 9:
                out[r][c] = 9
            elif C[r][c] == 1:
                out[r][c] = 1
    return out

# ========== d037b0a7 ==========
# Non-zero values propagate downward in their column
def solve_d037b0a7(grid):
    h, w = len(grid), len(grid[0])
    out = copy.deepcopy(grid)
    for c in range(w):
        val = 0
        for r in range(h):
            if grid[r][c] != 0:
                val = grid[r][c]
            if val != 0:
                out[r][c] = val
    return out

# ========== d06dbe63 ==========
# 13x13 with single 8: draw zigzag staircase of 5s above and below
# Pattern: alternating A (simple step) and B (step + horizontal move of 2)
def solve_d06dbe63(grid):
    h, w = len(grid), len(grid[0])
    out = copy.deepcopy(grid)

    # Find 8
    r8, c8 = 0, 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 8:
                r8, c8 = r, c

    # Upward zigzag: alternating A (up 1) and B (up 1 + right 2)
    cr, cc = r8, c8
    phase = 'A'
    while True:
        if phase == 'A':
            nr = cr - 1
            if nr < 0:
                break
            cr = nr
            out[cr][cc] = 5
            phase = 'B'
        else:  # B
            nr = cr - 1
            if nr < 0:
                break
            cr = nr
            out[cr][cc] = 5
            # Right 2
            moved = 0
            for i in range(1, 3):
                nc = cc + i
                if nc < w:
                    out[cr][nc] = 5
                    moved = i
            if moved < 2:
                break  # incomplete horizontal step, stop
            cc = cc + moved
            phase = 'A'

    # Downward zigzag: alternating A (down 1) and B (down 1 + left 2)
    cr, cc = r8, c8
    phase = 'A'
    while True:
        if phase == 'A':
            nr = cr + 1
            if nr >= h:
                break
            cr = nr
            out[cr][cc] = 5
            phase = 'B'
        else:  # B
            nr = cr + 1
            if nr >= h:
                break
            cr = nr
            out[cr][cc] = 5
            # Left 2
            moved = 0
            for i in range(1, 3):
                nc = cc - i
                if nc >= 0:
                    out[cr][nc] = 5
                    moved = i
            if moved < 2:
                break  # incomplete horizontal step, stop
            cc = cc - moved
            phase = 'A'

    return out

# ========== d13f3404 ==========
# 3x3 -> 6x6: diagonal tiling of the input
def solve_d13f3404(grid):
    h, w = len(grid), len(grid[0])
    oh, ow = h*2, w*2
    out = [[0]*ow for _ in range(oh)]
    for d in range(oh):
        for r in range(h):
            for c in range(w):
                nr, nc = r + d, c + d
                if 0 <= nr < oh and 0 <= nc < ow and grid[r][c] != 0:
                    out[nr][nc] = grid[r][c]
    return out

# ========== d22278a0 ==========
# Grid with colored dots in corners -> each grows a recursive zigzag spiral toward center
def solve_d22278a0(grid):
    h, w = len(grid), len(grid[0])
    out = [[0]*w for _ in range(h)]

    corners = {}
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                corners[(r, c)] = grid[r][c]

    corner_list = list(corners.keys())

    # Ownership by Manhattan distance
    ownership = [[None]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            min_dist = float('inf')
            owner = None
            tied = False
            for (cr, cc) in corner_list:
                d = abs(r - cr) + abs(c - cc)
                if d < min_dist:
                    min_dist = d
                    owner = (cr, cc)
                    tied = False
                elif d == min_dist:
                    tied = True
            if not tied:
                ownership[r][c] = owner

    for (r0, c0), color in corners.items():
        dr = 1 if r0 == 0 else -1
        dc = 1 if c0 == 0 else -1

        # Determine column step: 1 if same-row partner exists, 2 otherwise
        same_row_partners = [c for r, c in corner_list if r == r0 and c != c0]
        step_c = 1 if same_row_partners else 2

        # Number of zigzag levels
        if same_row_partners:
            partner_c = same_row_partners[0]
            half_w = (abs(partner_c - c0) + 1) // 2
        else:
            half_w = (w + 1) // 2

        half_h = (h + 1) // 2

        pattern = [[False]*w for _ in range(h)]

        def draw_pattern(sh, sw):
            """Recursive zigzag.
            sh = row levels remaining (base at row sh-1 from corner)
            sw = col levels remaining (spine at col (sw-1)*step_c from corner)
            """
            if sw <= 0:
                return

            spine_c = c0 + (sw - 1) * step_c * dc
            if not (0 <= spine_c < w):
                return

            if sh <= 0:
                # No more base rows, but still draw spine at corner row only
                if 0 <= r0 < h:
                    pattern[r0][spine_c] = True
                sw_red = 2 if step_c == 1 else 1
                draw_pattern(sh, sw - sw_red)
                return

            base_r = r0 + (sh - 1) * dr
            if not (0 <= base_r < h):
                return

            # Draw spine: from r0 to base_r at spine_c
            for r in range(min(r0, base_r), max(r0, base_r) + 1):
                if 0 <= r < h:
                    pattern[r][spine_c] = True

            # Draw base bar: contiguous from c0, width = sh physical cols
            bar_end_c = c0 + (sh - 1) * dc
            for c in range(min(c0, bar_end_c), max(c0, bar_end_c) + 1):
                if 0 <= c < w:
                    pattern[base_r][c] = True

            sw_red = 2 if step_c == 1 else 1
            draw_pattern(sh - 2, sw - sw_red)

            # Extended bars every 2 rows beyond base
            ext_r = base_r + 2 * dr
            while 0 <= ext_r < h:
                for c in range(min(c0, bar_end_c), max(c0, bar_end_c) + 1):
                    if 0 <= c < w:
                        pattern[ext_r][c] = True
                ext_r += 2 * dr

        draw_pattern(half_h, half_w)

        for r in range(h):
            for c in range(w):
                if pattern[r][c] and ownership[r][c] == (r0, c0):
                    out[r][c] = color

    return out

# ========== d23f8c26 ==========
# Keep only the center column values
def solve_d23f8c26(grid):
    h, w = len(grid), len(grid[0])
    center_c = w // 2
    out = [[0]*w for _ in range(h)]
    for r in range(h):
        out[r][center_c] = grid[r][center_c]
    return out

# ========== d2abd087 ==========
# Replace 5s: shapes with width > height -> 2, else 1. Special cases for L-shapes.
def solve_d2abd087(grid):
    h, w = len(grid), len(grid[0])
    out = [[0]*w for _ in range(h)]

    visited = [[False]*w for _ in range(h)]

    def flood_fill(r0, c0):
        stack = [(r0, c0)]
        cells = []
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= h or c < 0 or c >= w:
                continue
            if visited[r][c] or grid[r][c] != 5:
                continue
            visited[r][c] = True
            cells.append((r, c))
            for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                stack.append((nr, nc))
        return cells

    def classify_shape(cells):
        # Shapes with exactly 6 cells -> color 2, all others -> color 1
        if len(cells) == 6:
            return 2
        return 1

    shapes = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 5 and not visited[r][c]:
                cells = flood_fill(r, c)
                if cells:
                    shapes.append(cells)

    for cells in shapes:
        color = classify_shape(cells)
        for r, c in cells:
            out[r][c] = color

    return out

# ========== d364b489 ==========
# Each 1 gets decorated: 2 above, 8 below, 7 left, 6 right
def solve_d364b489(grid):
    h, w = len(grid), len(grid[0])
    out = copy.deepcopy(grid)
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 1:
                if r > 0: out[r-1][c] = 2
                if r < h-1: out[r+1][c] = 8
                if c > 0: out[r][c-1] = 7
                if c < w-1: out[r][c+1] = 6
    return out

# ========== d406998b ==========
# 3xN with 5s: even cols -> 5 or 3, odd cols -> opposite, based on N parity
def solve_d406998b(grid):
    h, w = len(grid), len(grid[0])
    out = copy.deepcopy(grid)
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 5:
                if w % 2 == 0:
                    if c % 2 == 1:
                        out[r][c] = 3
                else:
                    if c % 2 == 0:
                        out[r][c] = 3
    return out

# ==========================================
# Solve all tasks and verify
# ==========================================
solvers = {
    'c3e719e8': solve_c3e719e8,
    'c8cbb738': solve_c8cbb738,
    'c9e6f938': solve_c9e6f938,
    'c9f8e694': solve_c9f8e694,
    'caa06a1f': solve_caa06a1f,
    'cbded52d': solve_cbded52d,
    'cce03e0d': solve_cce03e0d,
    'cdecee7f': solve_cdecee7f,
    'ce22a75a': solve_ce22a75a,
    'ce4f8723': solve_ce4f8723,
    'ce9e57f2': solve_ce9e57f2,
    'cf98881b': solve_cf98881b,
    'd037b0a7': solve_d037b0a7,
    'd06dbe63': solve_d06dbe63,
    'd13f3404': solve_d13f3404,
    'd22278a0': solve_d22278a0,
    'd23f8c26': solve_d23f8c26,
    'd2abd087': solve_d2abd087,
    'd364b489': solve_d364b489,
    'd406998b': solve_d406998b,
}

all_results = {}
total_pass = 0
total_fail = 0

for tid, solver in solvers.items():
    data = load(tid)
    all_pass = True
    for split in ['train', 'test']:
        for i, pair in enumerate(data[split]):
            pred = solver(pair['input'])
            expected = pair['output']
            if pred != expected:
                all_pass = False
                print(f"FAIL {tid} {split}[{i}]")
                if len(expected) <= 5 and len(expected[0]) <= 10:
                    print(f"  Expected: {expected}")
                    print(f"  Got:      {pred}")
                else:
                    for r in range(min(len(expected), len(pred))):
                        if r < len(pred) and expected[r] != pred[r]:
                            print(f"  Row {r} exp: {expected[r]}")
                            print(f"  Row {r} got: {pred[r]}")
                            break
                    if len(expected) != len(pred):
                        print(f"  Size mismatch: {len(expected)}x{len(expected[0])} vs {len(pred)}x{len(pred[0])}")

    if all_pass:
        print(f"PASS {tid}")
        total_pass += 1
    else:
        total_fail += 1

    test_preds = []
    for pair in data['test']:
        test_preds.append(solver(pair['input']))
    all_results[tid] = test_preds

print(f"\nTotal: {total_pass} pass, {total_fail} fail")

with open(OUT, 'w') as f:
    json.dump(all_results, f)
print(f"Saved to {OUT}")
