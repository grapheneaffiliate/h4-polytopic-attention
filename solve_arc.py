import json
import os

BASE = "C:/Users/atchi/h4-polytopic-attention/data/arc1"
OUT_FILE = "C:/Users/atchi/h4-polytopic-attention/data/arc_python_solutions_b0.json"

if os.path.exists(OUT_FILE):
    with open(OUT_FILE) as f:
        solutions = json.load(f)
else:
    solutions = {}

def load_task(tid):
    with open(f"{BASE}/{tid}.json") as f:
        return json.load(f)

def test_solve(task, solve_fn):
    for i, pair in enumerate(task["train"]):
        result = solve_fn(pair["input"])
        if result != pair["output"]:
            print(f"  FAIL pair {i}")
            exp = pair["output"]
            for r in range(max(len(result), len(exp))):
                if r >= len(result):
                    print(f"    row {r}: MISSING (exp {exp[r]})")
                elif r >= len(exp):
                    print(f"    row {r}: EXTRA   (got {result[r]})")
                elif result[r] != exp[r]:
                    print(f"    row {r}: got {result[r]}")
                    print(f"           exp {exp[r]}")
            return False
    return True

# 017c7c7b: Each column is periodic; extend from 6 to 9 rows with 1->2
def solve_017c7c7b(grid):
    h, w = len(grid), len(grid[0])
    oh = h + h // 2
    out = [[0]*w for _ in range(oh)]
    for c in range(w):
        col = [grid[r][c] for r in range(h)]
        best_period = h
        for period in range(1, h+1):
            if all(col[j] == col[j % period] for j in range(h)):
                best_period = period
                break
        for r in range(oh):
            v = col[r % best_period]
            out[r][c] = 2 if v == 1 else v
    return out

task = load_task("017c7c7b")
if test_solve(task, solve_017c7c7b):
    print("017c7c7b: PASS")
    solutions["017c7c7b"] = """def solve(grid):
    h, w = len(grid), len(grid[0])
    oh = h + h // 2
    out = [[0]*w for _ in range(oh)]
    for c in range(w):
        col = [grid[r][c] for r in range(h)]
        best_period = h
        for period in range(1, h+1):
            if all(col[j] == col[j % period] for j in range(h)):
                best_period = period
                break
        for r in range(oh):
            v = col[r % best_period]
            out[r][c] = 2 if v == 1 else v
    return out"""
else:
    print("017c7c7b: FAIL")

# 025d127b: Parallelogram shapes get shifted right by 1 (except bottom row), clipped to max_c
def solve_025d127b(grid):
    h, w = len(grid), len(grid[0])
    out = [[0]*w for _ in range(h)]
    color_cells = {}
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                color = grid[r][c]
                if color not in color_cells:
                    color_cells[color] = []
                color_cells[color].append((r, c))
    for color, cells in color_cells.items():
        max_r = max(r for r, c in cells)
        max_c = max(c for r, c in cells)
        for r, c in cells:
            if r == max_r:
                out[r][c] = color
            else:
                new_c = min(c + 1, max_c)
                out[r][new_c] = color
    return out

task = load_task("025d127b")
if test_solve(task, solve_025d127b):
    print("025d127b: PASS")
    solutions["025d127b"] = """def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [[0]*w for _ in range(h)]
    color_cells = {}
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                color = grid[r][c]
                if color not in color_cells:
                    color_cells[color] = []
                color_cells[color].append((r, c))
    for color, cells in color_cells.items():
        max_r = max(r for r, c in cells)
        max_c = max(c for r, c in cells)
        for r, c in cells:
            if r == max_r:
                out[r][c] = color
            else:
                new_c = min(c + 1, max_c)
                out[r][new_c] = color
    return out"""
else:
    print("025d127b: FAIL")

# 05269061
def solve_05269061(grid):
    h, w = len(grid), len(grid[0])
    colors = [0, 0, 0]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                idx = (r + c) % 3
                colors[idx] = grid[r][c]
    out = [[0]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            out[r][c] = colors[(r + c) % 3]
    return out

task = load_task("05269061")
if test_solve(task, solve_05269061):
    print("05269061: PASS")
    solutions["05269061"] = """def solve(grid):
    h, w = len(grid), len(grid[0])
    colors = [0, 0, 0]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                idx = (r + c) % 3
                colors[idx] = grid[r][c]
    out = [[0]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            out[r][c] = colors[(r + c) % 3]
    return out"""
else:
    print("05269061: FAIL")

# 05f2a901
def solve_05f2a901(grid):
    h, w = len(grid), len(grid[0])
    r2, c2, r8, c8 = [], [], [], []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 2:
                r2.append(r); c2.append(c)
            elif grid[r][c] == 8:
                r8.append(r); c8.append(c)
    r2_min, r2_max = min(r2), max(r2)
    c2_min, c2_max = min(c2), max(c2)
    r8_min, r8_max = min(r8), max(r8)
    c8_min, c8_max = min(c8), max(c8)
    dr, dc = 0, 0
    if r2_max < r8_min:
        dr = r8_min - r2_max - 1
    elif r2_min > r8_max:
        dr = r8_max - r2_min + 1
    if c2_max < c8_min:
        dc = c8_min - c2_max - 1
    elif c2_min > c8_max:
        dc = c8_max - c2_min + 1
    out = [[0]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 8:
                out[r][c] = 8
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 2:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr][nc] = 2
    return out

task = load_task("05f2a901")
if test_solve(task, solve_05f2a901):
    print("05f2a901: PASS")
    solutions["05f2a901"] = """def solve(grid):
    h, w = len(grid), len(grid[0])
    r2, c2, r8, c8 = [], [], [], []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 2:
                r2.append(r); c2.append(c)
            elif grid[r][c] == 8:
                r8.append(r); c8.append(c)
    r2_min, r2_max = min(r2), max(r2)
    c2_min, c2_max = min(c2), max(c2)
    r8_min, r8_max = min(r8), max(r8)
    c8_min, c8_max = min(c8), max(c8)
    dr, dc = 0, 0
    if r2_max < r8_min:
        dr = r8_min - r2_max - 1
    elif r2_min > r8_max:
        dr = r8_max - r2_min + 1
    if c2_max < c8_min:
        dc = c8_min - c2_max - 1
    elif c2_min > c8_max:
        dc = c8_max - c2_min + 1
    out = [[0]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 8:
                out[r][c] = 8
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 2:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr][nc] = 2
    return out"""
else:
    print("05f2a901: FAIL")

# 09629e4f
def solve_09629e4f(grid):
    h, w = len(grid), len(grid[0])
    block_rows = [(0,3), (4,7), (8,11)]
    block_cols = [(0,3), (4,7), (8,11)]
    key_bi, key_bj = -1, -1
    for bi, (r1, r2) in enumerate(block_rows):
        for bj, (c1, c2) in enumerate(block_cols):
            has8 = any(grid[r][c] == 8 for r in range(r1, r2) for c in range(c1, c2))
            if not has8:
                key_bi, key_bj = bi, bj
    kr1, kr2 = block_rows[key_bi]
    kc1, kc2 = block_cols[key_bj]
    out = [[0]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 5:
                out[r][c] = 5
    for r in range(kr1, kr2):
        for c in range(kc1, kc2):
            v = grid[r][c]
            if v != 0 and v != 5:
                lr, lc = r - kr1, c - kc1
                br1, br2 = block_rows[lr]
                bc1, bc2 = block_cols[lc]
                for rr in range(br1, br2):
                    for cc in range(bc1, bc2):
                        out[rr][cc] = v
    return out

task = load_task("09629e4f")
if test_solve(task, solve_09629e4f):
    print("09629e4f: PASS")
    solutions["09629e4f"] = """def solve(grid):
    h, w = len(grid), len(grid[0])
    block_rows = [(0,3), (4,7), (8,11)]
    block_cols = [(0,3), (4,7), (8,11)]
    key_bi, key_bj = -1, -1
    for bi, (r1, r2) in enumerate(block_rows):
        for bj, (c1, c2) in enumerate(block_cols):
            has8 = any(grid[r][c] == 8 for r in range(r1, r2) for c in range(c1, c2))
            if not has8:
                key_bi, key_bj = bi, bj
    kr1, kr2 = block_rows[key_bi]
    kc1, kc2 = block_cols[key_bj]
    out = [[0]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 5:
                out[r][c] = 5
    for r in range(kr1, kr2):
        for c in range(kc1, kc2):
            v = grid[r][c]
            if v != 0 and v != 5:
                lr, lc = r - kr1, c - kc1
                br1, br2 = block_rows[lr]
                bc1, bc2 = block_cols[lc]
                for rr in range(br1, br2):
                    for cc in range(bc1, bc2):
                        out[rr][cc] = v
    return out"""
else:
    print("09629e4f: FAIL")

# 0962bcdd
def solve_0962bcdd(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                center_color = grid[r][c]
                arm_positions = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
                arm_colors = set()
                valid = True
                for nr, nc in arm_positions:
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0 and grid[nr][nc] != center_color:
                        arm_colors.add(grid[nr][nc])
                    else:
                        if not (0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0):
                            valid = False
                if valid and len(arm_colors) == 1:
                    arm_color = arm_colors.pop()
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if dr == 0 and dc == 0:
                                    out[nr][nc] = center_color
                                elif dr == 0 or dc == 0:
                                    out[nr][nc] = arm_color
                                elif abs(dr) == abs(dc):
                                    out[nr][nc] = center_color
    return out

task = load_task("0962bcdd")
if test_solve(task, solve_0962bcdd):
    print("0962bcdd: PASS")
    solutions["0962bcdd"] = """def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                center_color = grid[r][c]
                arm_positions = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
                arm_colors = set()
                valid = True
                for nr, nc in arm_positions:
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0 and grid[nr][nc] != center_color:
                        arm_colors.add(grid[nr][nc])
                    else:
                        if not (0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0):
                            valid = False
                if valid and len(arm_colors) == 1:
                    arm_color = arm_colors.pop()
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if dr == 0 and dc == 0:
                                    out[nr][nc] = center_color
                                elif dr == 0 or dc == 0:
                                    out[nr][nc] = arm_color
                                elif abs(dr) == abs(dc):
                                    out[nr][nc] = center_color
    return out"""
else:
    print("0962bcdd: FAIL")

# 10fcaaa3
def solve_10fcaaa3(grid):
    h, w = len(grid), len(grid[0])
    oh, ow = h * 2, w * 2
    tiled = [[grid[r % h][c % w] for c in range(ow)] for r in range(oh)]
    out = [row[:] for row in tiled]
    for r in range(oh):
        for c in range(ow):
            if tiled[r][c] == 0:
                for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < oh and 0 <= nc < ow and tiled[nr][nc] != 0:
                        out[r][c] = 8
                        break
    return out

task = load_task("10fcaaa3")
if test_solve(task, solve_10fcaaa3):
    print("10fcaaa3: PASS")
    solutions["10fcaaa3"] = """def solve(grid):
    h, w = len(grid), len(grid[0])
    oh, ow = h * 2, w * 2
    tiled = [[grid[r % h][c % w] for c in range(ow)] for r in range(oh)]
    out = [row[:] for row in tiled]
    for r in range(oh):
        for c in range(ow):
            if tiled[r][c] == 0:
                for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < oh and 0 <= nc < ow and tiled[nr][nc] != 0:
                        out[r][c] = 8
                        break
    return out"""
else:
    print("10fcaaa3: FAIL")

# 11852cab
def solve_11852cab(grid):
    h, w = len(grid), len(grid[0])
    cells = [(r, c, grid[r][c]) for r in range(h) for c in range(w) if grid[r][c] != 0]
    cr = round(sum(r for r, c, v in cells) / len(cells))
    cc = round(sum(c for r, c, v in cells) / len(cells))
    out = [row[:] for row in grid]
    for r, c, v in cells:
        dr, dc = r - cr, c - cc
        for _ in range(4):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and out[nr][nc] == 0:
                out[nr][nc] = v
            dr, dc = dc, -dr
    return out

task = load_task("11852cab")
if test_solve(task, solve_11852cab):
    print("11852cab: PASS")
    solutions["11852cab"] = """def solve(grid):
    h, w = len(grid), len(grid[0])
    cells = [(r, c, grid[r][c]) for r in range(h) for c in range(w) if grid[r][c] != 0]
    cr = round(sum(r for r, c, v in cells) / len(cells))
    cc = round(sum(c for r, c, v in cells) / len(cells))
    out = [row[:] for row in grid]
    for r, c, v in cells:
        dr, dc = r - cr, c - cc
        for _ in range(4):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and out[nr][nc] == 0:
                out[nr][nc] = v
            dr, dc = dc, -dr
    return out"""
else:
    print("11852cab: FAIL")

# 1190e5a7
def solve_1190e5a7(grid):
    h, w = len(grid), len(grid[0])
    colors = set(v for row in grid for v in row)
    sep_color = None
    bg_color = None
    for color in colors:
        full_rows = [r for r in range(h) if all(grid[r][c] == color for c in range(w))]
        full_cols = [c for c in range(w) if all(grid[r][c] == color for r in range(h))]
        if full_rows or full_cols:
            sep_color = color
    for color in colors:
        if color != sep_color:
            bg_color = color
    row_groups = 0
    in_group = False
    for r in range(h):
        is_sep = all(grid[r][c] == sep_color for c in range(w))
        if not is_sep:
            if not in_group:
                row_groups += 1
                in_group = True
        else:
            in_group = False
    col_groups = 0
    in_group = False
    for c in range(w):
        is_sep = all(grid[r][c] == sep_color for r in range(h))
        if not is_sep:
            if not in_group:
                col_groups += 1
                in_group = True
        else:
            in_group = False
    return [[bg_color] * col_groups for _ in range(row_groups)]

task = load_task("1190e5a7")
if test_solve(task, solve_1190e5a7):
    print("1190e5a7: PASS")
    solutions["1190e5a7"] = """def solve(grid):
    h, w = len(grid), len(grid[0])
    colors = set(v for row in grid for v in row)
    sep_color = None
    bg_color = None
    for color in colors:
        full_rows = [r for r in range(h) if all(grid[r][c] == color for c in range(w))]
        full_cols = [c for c in range(w) if all(grid[r][c] == color for r in range(h))]
        if full_rows or full_cols:
            sep_color = color
    for color in colors:
        if color != sep_color:
            bg_color = color
    row_groups = 0
    in_group = False
    for r in range(h):
        is_sep = all(grid[r][c] == sep_color for c in range(w))
        if not is_sep:
            if not in_group:
                row_groups += 1
                in_group = True
        else:
            in_group = False
    col_groups = 0
    in_group = False
    for c in range(w):
        is_sep = all(grid[r][c] == sep_color for r in range(h))
        if not is_sep:
            if not in_group:
                col_groups += 1
                in_group = True
        else:
            in_group = False
    return [[bg_color] * col_groups for _ in range(row_groups)]"""
else:
    print("1190e5a7: FAIL")

# 137eaa0f
def solve_137eaa0f(grid):
    h, w = len(grid), len(grid[0])
    fives = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 5]
    cells = [(r, c, grid[r][c]) for r in range(h) for c in range(w) if grid[r][c] not in (0, 5)]
    out = [[0]*3 for _ in range(3)]
    out[1][1] = 5
    for r, c, v in cells:
        best_dist = float('inf')
        best_five = None
        for fr, fc in fives:
            d = abs(r - fr) + abs(c - fc)
            if d < best_dist:
                best_dist = d
                best_five = (fr, fc)
        dr = r - best_five[0]
        dc = c - best_five[1]
        or_, oc = 1 + dr, 1 + dc
        if 0 <= or_ < 3 and 0 <= oc < 3:
            out[or_][oc] = v
    return out

task = load_task("137eaa0f")
if test_solve(task, solve_137eaa0f):
    print("137eaa0f: PASS")
    solutions["137eaa0f"] = """def solve(grid):
    h, w = len(grid), len(grid[0])
    fives = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 5]
    cells = [(r, c, grid[r][c]) for r in range(h) for c in range(w) if grid[r][c] not in (0, 5)]
    out = [[0]*3 for _ in range(3)]
    out[1][1] = 5
    for r, c, v in cells:
        best_dist = float('inf')
        best_five = None
        for fr, fc in fives:
            d = abs(r - fr) + abs(c - fc)
            if d < best_dist:
                best_dist = d
                best_five = (fr, fc)
        dr = r - best_five[0]
        dc = c - best_five[1]
        or_, oc = 1 + dr, 1 + dc
        if 0 <= or_ < 3 and 0 <= oc < 3:
            out[or_][oc] = v
    return out"""
else:
    print("137eaa0f: FAIL")

# Save
with open(OUT_FILE, 'w') as f:
    json.dump(solutions, f, indent=2)
print(f"\nSaved {len(solutions)} solutions to {OUT_FILE}")
