import json
import copy

# ============================================================
# Task: a61f2674
# Columns with 5s from bottom. Tallest column -> color 1, shortest -> color 2.
# ============================================================
def solve_a61f2674(grid):
    rows = len(grid)
    cols = len(grid[0])
    heights = {}
    for c in range(cols):
        h = sum(1 for r in range(rows) if grid[r][c] == 5)
        if h > 0:
            heights[c] = h

    max_h = max(heights.values())
    min_h = min(heights.values())

    out = [[0]*cols for _ in range(rows)]
    for c, h in heights.items():
        if h == max_h:
            color = 1
        elif h == min_h:
            color = 2
        else:
            color = 0
        if color != 0:
            for r in range(rows - h, rows):
                out[r][c] = color
    return out

# ============================================================
# Task: a65b410d
# Row of 2s, triangle of 3s above (expanding), triangle of 1s below (shrinking)
# ============================================================
def solve_a65b410d(grid):
    rows = len(grid)
    cols = len(grid[0])
    out = [[0]*cols for _ in range(rows)]

    two_row = -1
    two_len = 0
    for r in range(rows):
        count = sum(1 for c in range(cols) if grid[r][c] == 2)
        if count > 0:
            two_row = r
            two_len = count
            break

    for c in range(two_len):
        out[two_row][c] = 2

    for d in range(1, two_row + 1):
        length = two_len + d
        for c in range(min(length, cols)):
            out[two_row - d][c] = 3

    for d in range(1, rows - two_row):
        length = two_len - d
        if length <= 0:
            break
        for c in range(length):
            out[two_row + d][c] = 1

    return out

# ============================================================
# Task: a68b268e
# 9x9 grid divided by row/col 4 into 4 quadrants with colors 7,4,8,6.
# Overlay: where 7 exists use 7, else try 4, 8, 6.
# ============================================================
def solve_a68b268e(grid):
    tl = [[grid[r][c] for c in range(4)] for r in range(4)]
    tr = [[grid[r][c] for c in range(5,9)] for r in range(4)]
    bl = [[grid[r][c] for c in range(4)] for r in range(5,9)]
    br = [[grid[r][c] for c in range(5,9)] for r in range(5,9)]

    out = [[0]*4 for _ in range(4)]
    for r in range(4):
        for c in range(4):
            if tl[r][c] != 0:
                out[r][c] = tl[r][c]
            elif tr[r][c] != 0:
                out[r][c] = tr[r][c]
            elif bl[r][c] != 0:
                out[r][c] = bl[r][c]
            elif br[r][c] != 0:
                out[r][c] = br[r][c]
            else:
                out[r][c] = 0
    return out

# ============================================================
# Task: a740d043
# Extract bounding box of non-1 cells, replace 1s with 0 inside.
# ============================================================
def solve_a740d043(grid):
    rows = len(grid)
    cols = len(grid[0])

    min_r, max_r, min_c, max_c = rows, 0, cols, 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 1:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    out = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            val = grid[r][c]
            row.append(0 if val == 1 else val)
        out.append(row)
    return out

# ============================================================
# Task: a78176bb
# Diagonal with 5-triangle. Remove 5s, add parallel diagonal(s).
# ============================================================
def solve_a78176bb(grid):
    rows = len(grid)
    cols = len(grid[0])
    out = [[0]*cols for _ in range(rows)]

    diag_color = 0
    diag_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and grid[r][c] != 5:
                diag_color = grid[r][c]
                diag_cells.append((r, c))

    offsets = [c - r for r, c in diag_cells]
    orig_offset = offsets[0]

    pos_dists = []
    neg_dists = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                dist = (c - r) - orig_offset
                if dist > 0:
                    pos_dists.append(dist)
                elif dist < 0:
                    neg_dists.append(abs(dist))

    offsets_to_draw = [orig_offset]
    if pos_dists:
        offsets_to_draw.append(orig_offset + max(pos_dists) + 2)
    if neg_dists:
        offsets_to_draw.append(orig_offset - max(neg_dists) - 2)

    for r in range(rows):
        for off in offsets_to_draw:
            c = r + off
            if 0 <= c < cols:
                out[r][c] = diag_color

    return out

# ============================================================
# Task: a79310a0
# Shape of 8s shifted down by 1 row, recolored to 2.
# ============================================================
def solve_a79310a0(grid):
    rows = len(grid)
    cols = len(grid[0])
    out = [[0]*cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8:
                if r + 1 < rows:
                    out[r + 1][c] = 2
    return out

# ============================================================
# Task: a85d4709
# 3x3 grid, one 5 per row. Col position -> color. Fill row.
# col 0 -> 2, col 1 -> 4, col 2 -> 3
# ============================================================
def solve_a85d4709(grid):
    color_map = {0: 2, 1: 4, 2: 3}
    out = []
    for r in range(3):
        for c in range(3):
            if grid[r][c] == 5:
                out.append([color_map[c]] * 3)
                break
    return out

# ============================================================
# Task: a87f7484
# Multiple 3x3 blocks. Find the unique one (unique pattern).
# ============================================================
def solve_a87f7484(grid):
    rows = len(grid)
    cols = len(grid[0])

    blocks = []
    if rows % 3 == 0 and cols == 3:
        for i in range(rows // 3):
            block = [row[:] for row in grid[i*3:(i+1)*3]]
            blocks.append(block)
    elif rows == 3 and cols % 3 == 0:
        for i in range(cols // 3):
            block = [[grid[r][i*3+c] for c in range(3)] for r in range(3)]
            blocks.append(block)

    def pattern(block):
        return tuple(tuple(1 if cell != 0 else 0 for cell in row) for row in block)

    patterns = [pattern(b) for b in blocks]
    for i, p in enumerate(patterns):
        count = sum(1 for q in patterns if q == p)
        if count == 1:
            return blocks[i]

    max_fill = -1
    best = None
    for b in blocks:
        fill = sum(1 for row in b for cell in row if cell != 0)
        if fill > max_fill:
            max_fill = fill
            best = b
    return best

# ============================================================
# Task: a8c38be5
# Scattered 3x3 blocks with colored patterns in 5-backgrounds.
# Arrange into 9x9 grid based on shape orientation.
# ============================================================
def solve_a8c38be5(grid):
    rows = len(grid)
    cols = len(grid[0])

    blocks = []

    for r in range(rows - 2):
        for c in range(cols - 2):
            colors = set()
            has_five = False
            all_valid = True
            for dr in range(3):
                for dc in range(3):
                    v = grid[r+dr][c+dc]
                    if v == 0:
                        all_valid = False
                        break
                    if v == 5:
                        has_five = True
                    else:
                        colors.add(v)
                if not all_valid:
                    break

            if all_valid and len(colors) == 1 and has_five:
                color = colors.pop()
                block = [[grid[r+dr][c+dc] for dc in range(3)] for dr in range(3)]
                blocks.append((color, block, r, c))

    def classify_block(block):
        colored_r = []
        colored_c = []
        for r in range(3):
            for c in range(3):
                if block[r][c] != 5:
                    colored_r.append(r)
                    colored_c.append(c)

        avg_r = sum(colored_r) / len(colored_r)
        avg_c = sum(colored_c) / len(colored_c)

        if avg_r < 0.8:
            rp = 0
        elif avg_r > 1.2:
            rp = 2
        else:
            rp = 1

        if avg_c < 0.8:
            cp = 0
        elif avg_c > 1.2:
            cp = 2
        else:
            cp = 1

        return (rp, cp)

    out = [[5]*9 for _ in range(9)]

    for color, block, orig_r, orig_c in blocks:
        rp, cp = classify_block(block)
        for dr in range(3):
            for dc in range(3):
                out[rp*3 + dr][cp*3 + dc] = block[dr][dc]

    return out

# ============================================================
# Task: a9f96cdd
# Single 2 in grid. Replace with 0. Diagonal neighbors get colors:
# top-left: 3, top-right: 6, bottom-left: 8, bottom-right: 7
# ============================================================
def solve_a9f96cdd(grid):
    rows = len(grid)
    cols = len(grid[0])
    out = [[0]*cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                if r-1 >= 0 and c-1 >= 0:
                    out[r-1][c-1] = 3
                if r-1 >= 0 and c+1 < cols:
                    out[r-1][c+1] = 6
                if r+1 < rows and c-1 >= 0:
                    out[r+1][c-1] = 8
                if r+1 < rows and c+1 < cols:
                    out[r+1][c+1] = 7
                return out
    return out

# ============================================================
# Main
# ============================================================
solvers = {
    'a61f2674': solve_a61f2674,
    'a65b410d': solve_a65b410d,
    'a68b268e': solve_a68b268e,
    'a740d043': solve_a740d043,
    'a78176bb': solve_a78176bb,
    'a79310a0': solve_a79310a0,
    'a85d4709': solve_a85d4709,
    'a87f7484': solve_a87f7484,
    'a8c38be5': solve_a8c38be5,
    'a9f96cdd': solve_a9f96cdd,
}

results = {}
all_pass = True

for task_id, solver in solvers.items():
    with open(f'C:/Users/atchi/h4-polytopic-attention/data/arc1/{task_id}.json') as f:
        data = json.load(f)

    task_pass = True

    for i, pair in enumerate(data['train']):
        pred = solver(pair['input'])
        expected = pair['output']
        if pred != expected:
            print(f"FAIL {task_id} train[{i}]")
            for j, (er, pr) in enumerate(zip(expected, pred)):
                if er != pr:
                    print(f"  row {j}: expected {er}")
                    print(f"          got      {pr}")
            task_pass = False
            all_pass = False

    for i, pair in enumerate(data['test']):
        pred = solver(pair['input'])
        if 'output' in pair:
            expected = pair['output']
            if pred != expected:
                print(f"FAIL {task_id} test[{i}]")
                for j, (er, pr) in enumerate(zip(expected, pred)):
                    if er != pr:
                        print(f"  row {j}: expected {er}")
                        print(f"          got      {pr}")
                task_pass = False
                all_pass = False

    if task_pass:
        print(f"PASS {task_id}")

    results[task_id] = {
        'solve': f'solve_{task_id}',
        'test_output': [solver(p['input']) for p in data['test']]
    }

print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

with open('C:/Users/atchi/h4-polytopic-attention/data/arc_python_solutions_b16.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved to data/arc_python_solutions_b16.json")
