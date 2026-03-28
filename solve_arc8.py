import json
from collections import Counter

def load_task(task_id):
    with open(f'data/arc1/{task_id}.json') as f:
        return json.load(f)

def test_solve(task_id, solve_fn):
    task = load_task(task_id)
    all_pass = True
    for i, pair in enumerate(task['train']):
        out = solve_fn(pair['input'])
        match = out == pair['output']
        print(f"  {task_id} train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
    return all_pass

solutions = {}

# ===================== Task 1190e5a7 =====================
print("=" * 50)
print("Task 1190e5a7")

# The grid is divided into cells. The task says "find smallest rectangle cell dimensions".
# But the actual cell sizes don't directly match the output.
#
# Let me re-examine. Looking at the grids:
#
# Train 0: The grid has separator rows at [2] and separator cols at [1,10,13].
# This creates a 2x4 grid of cells:
#   Row-band 0 (rows 0-1): cells at widths [1, 8, 2, 1]
#   Row-band 1 (rows 3-14): cells at widths [1, 8, 2, 1]
# Total: 2 row-bands x 4 column-bands = 8 cells
#
# Output: 2x4 -> matches the GRID DIMENSIONS (2 row-bands x 4 col-bands)!
# And it's filled with background color.
#
# Train 1: sep_rows=[3,9], sep_cols=[4]
# Row-bands: [3, 5, 1] -> 3 row-bands
# Col-bands: [4, 6] -> 2 col-bands
# Output: 3x2 -> YES! 3 row-bands x 2 col-bands!
#
# Train 2: sep_rows=[2,7,16,21,23], sep_cols=[6,21,23,25]
# Row-bands: [2,4,8,4,1,3] -> 6 row-bands
# Col-bands: [6,14,1,1,1] -> 5 col-bands
# Output: 6x5 -> YES! 6 row-bands x 5 col-bands!
#
# So the output is NOT the smallest cell, it's the number of cells in each dimension!
# The output grid has dimensions (number of row-bands) x (number of col-bands),
# filled with the background color.

def solve_1190e5a7(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find separator value
    sep = None
    for r in range(rows):
        if len(set(grid[r])) == 1:
            sep = grid[r][0]
            break

    bg = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != sep:
                bg = grid[r][c]
                break
        if bg is not None:
            break

    # Find separator rows
    sep_rows = []
    for r in range(rows):
        if all(grid[r][c] == sep for c in range(cols)):
            sep_rows.append(r)

    # Find separator cols
    sep_cols = []
    for c in range(cols):
        if all(grid[r][c] == sep for r in range(rows)):
            sep_cols.append(c)

    # Count row-bands and col-bands
    all_rows = sorted(set([-1] + sep_rows + [rows]))
    all_cols = sorted(set([-1] + sep_cols + [cols]))

    n_row_bands = 0
    for i in range(len(all_rows)-1):
        gap = all_rows[i+1] - all_rows[i] - 1
        if gap > 0:
            n_row_bands += 1

    n_col_bands = 0
    for i in range(len(all_cols)-1):
        gap = all_cols[i+1] - all_cols[i] - 1
        if gap > 0:
            n_col_bands += 1

    return [[bg] * n_col_bands for _ in range(n_row_bands)]

if test_solve('1190e5a7', solve_1190e5a7):
    solutions['1190e5a7'] = {
        "code": """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    sep = None
    for r in range(rows):
        if len(set(grid[r])) == 1:
            sep = grid[r][0]
            break
    bg = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != sep:
                bg = grid[r][c]
                break
        if bg is not None:
            break
    sep_rows = []
    for r in range(rows):
        if all(grid[r][c] == sep for c in range(cols)):
            sep_rows.append(r)
    sep_cols = []
    for c in range(cols):
        if all(grid[r][c] == sep for r in range(rows)):
            sep_cols.append(c)
    all_rows = sorted(set([-1] + sep_rows + [rows]))
    all_cols = sorted(set([-1] + sep_cols + [cols]))
    n_row_bands = sum(1 for i in range(len(all_rows)-1) if all_rows[i+1] - all_rows[i] - 1 > 0)
    n_col_bands = sum(1 for i in range(len(all_cols)-1) if all_cols[i+1] - all_cols[i] - 1 > 0)
    return [[bg] * n_col_bands for _ in range(n_row_bands)]""",
        "description": "Count number of row and column bands in grid divided by separator lines, output grid of that size"
    }
    print("  SAVED!")

# ===================== Task 10fcaaa3 =====================
print("\n" + "=" * 50)
print("Task 10fcaaa3: Tile and add 8s between colored cells")

# Let me look at the examples more carefully.
# The output is 2x the input dimensions.
# Looking at example 4 (simplest): input 4x4 with single colored cell at (1,1)=2.
# Output 8x8: the input is tiled 2x2, and 8s appear between the colored cells.
#
# In the output:
# Colored cells at: (1,1)=2, (1,5)=2, (5,1)=2, (5,5)=2
# 8s at: all (r,c) where r%4==0 and c%4==0 -> (0,0),(0,4),(4,0),(4,4) and
#         also at (0,2),(0,4),(0,6),(2,0),(2,2),(2,4),(2,6),(4,0)...
# Actually 8s at: row 0 cols 0,2,4,6. Row 2 cols 0,2,4,6. Row 4 cols 0,2,4,6. Row 6 cols 0,2,4,6.
# That's all even rows, even cols.
# But row 3 has no 8s: [0,0,0,0,0,0,0,0].

# Wait, row 3 in input is [0,0,0,0] and in output also [0,0,0,0,0,0,0,0]. All zeros.
# But row 0 in input is [0,0,0,0] and output is [8,0,8,0,8,0,8,0]. Has 8s!
# What's special about row 0 vs row 3?
#
# In the input, colored cell is at (1,1). The "anti" position of (1,1) in period-4 tiling
# would be... hmm.
#
# OK let me think about this differently. For each cell in the OUTPUT, we need to decide
# if it's 0, 8, or a colored value.
#
# The colored cells tile with 2x repetition: (r,c) = colored if (r%input_h, c%input_w) has color.
# The 8s fill positions that are "between" colored cells, possibly on the reflection boundaries.
#
# The reflection boundaries in a 2x tiling would be at the midpoints.
# For input height H, the horizontal reflection axis is at row H (between the two tiles).
# For input width W, vertical axis at col W.
#
# 8 is placed at (r,c) if: NOT a colored cell AND the cell is on a "grid line" defined
# by connecting the colored cells.
#
# Actually, maybe it's simpler: for each colored cell at (r0,c0) in the tiled grid,
# place 8s at all positions that differ from (r0,c0) by either (dr,0) or (0,dc) where
# dr and dc are odd? Or: 8 at positions reachable from any colored cell by moving
# in a specific pattern.
#
# Let me look at example 3 more carefully:
# Input: 5x3 with colored cells at (1,1)=4 and (4,0)=4
# Output: 10x6
# Let me check where 8s appear.

task10 = load_task('10fcaaa3')

for ti, pair in enumerate(task10['train']):
    inp = pair['input']
    out = pair['output']
    h, w = len(inp), len(inp[0])
    oh, ow = len(out), len(out[0])
    print(f"\n  Train {ti}: {h}x{w} -> {oh}x{ow}")

    # Find colored cells in input
    colored = []
    for r in range(h):
        for c in range(w):
            if inp[r][c] != 0:
                colored.append((r, c, inp[r][c]))
    print(f"    Colored cells: {colored}")

    # Tiled colored cells
    tiled_colored = set()
    for r, c, v in colored:
        for tr in range(2):
            for tc in range(2):
                tiled_colored.add((r + tr*h, c + tc*w))

    # Find 8 positions in output
    eights = []
    for r in range(oh):
        for c in range(ow):
            if out[r][c] == 8:
                eights.append((r, c))

    print(f"    8 positions: {eights}")
    print(f"    Tiled colored: {tiled_colored}")

    # For each 8 position, compute distance to nearest colored cell
    for r, c in eights:
        min_dist = min(abs(r-cr) + abs(c-cc) for cr, cc in tiled_colored)
        # Also compute chebyshev
        min_cheb = min(max(abs(r-cr), abs(c-cc)) for cr, cc in tiled_colored)
        # Check relationship with colored cells
        # Is (r,c) on a line connecting two colored cells?
        pass

    # Let me try: for each 0-cell in output, check if it's on the perpendicular bisector
    # of two colored cells.
    # Or: 8 at (r,c) if it's equidistant to two different colored cells (Voronoi boundary).

    # Actually let me just look at the patterns:
    print(f"    Output grid:")
    for r in range(oh):
        print(f"      {out[r]}")

# I think the pattern might be:
# For each pair of colored cells (in the tiled grid), draw 8s along the
# perpendicular bisector line between them.
# The perpendicular bisector of two points forms a line where
# manhattan distance to both is equal.

# Actually looking at example 1: two colored cells at (1,1) and (1,1+4)=(1,5) (tiled).
# Also (1,1) and (1+2,1)=(3,1) tiled? No, input is 2x4, so tiled is (1,1),(1,5),(3,1),(3,5)?
# Wait, input is 2x4, output is 4x8. Colored at (1,1)=5.
# Tiled colored: (1,1),(1,5),(3,1),(3,5).

# 8 positions: (0,0),(0,2),(0,4),(0,6),(2,0),(2,2),(2,4),(2,6)
# Row 0: all even cols. Row 2: all even cols.
# These are at rows 0 and 2 (even rows). Row 1 and 3 have colored cells.

# The 8s are at positions that are at Chebyshev distance 1 from some colored cell
# but NOT adjacent orthogonally (only diagonally)?
# (0,0) to (1,1): Chebyshev 1, diagonal. YES.
# (0,2) to (1,1): Chebyshev 1, diagonal. YES.
# (2,0) to (1,1): Chebyshev 1, diagonal. YES.
# (2,0) to (3,1): Chebyshev 1, diagonal. YES.
# (2,2) to (1,1): Chebyshev 1, diagonal. YES. Also to (3,1): Chebyshev 1. Also to (1,5)? No.

# But in example 2, row 1 is ALL 8s: [8,8,8,8,8,8,8,8]
# (1,0) to nearest colored cell: input colored at (0,2) and (2,1).
# Tiled colored: (0,2),(0,6),(2,1),(2,5),(3,2),(3,6),(5,1),(5,5)
# (1,0) to (0,2): manhattan=3, chebyshev=2. Not adjacent diagonally.
# (1,0) to (2,1): manhattan=2, chebyshev=1. YES diagonal.
# So (1,0) is diagonally adjacent to (2,1). So 8 there. OK.
# (1,3) to nearest: (0,2) manhattan=2 cheb=1 (diagonal). (2,1) manhattan=3. So yes, diagonal to (0,2).
# (1,4) to nearest: (0,2) dist=3, (2,5) dist=2 cheb=1 diagonal. Yes.
# So it seems ALL 8-positions are diagonal neighbors of colored cells.

# But some diagonal neighbors of colored cells DON'T get 8:
# (0,1) is diagonal to (1,0)? No, (1,0) is not colored. (0,1) diagonal to what colored?
# In example 2: colored at (0,2),(0,6),(2,1),(2,5),(3,2),(3,6),(5,1),(5,5)
# (0,1): diagonal neighbors are (1,0) and (1,2). Neither is colored.
# Adjacent orthogonally: (0,0)=0, (0,2)=6, (1,1)=8. Wait output(0,1) is 0, not 8.
# So (0,1) is NOT diagonally adjacent to any colored cell. Good.

# Let me verify: for example 2, are ALL 8-positions diagonal neighbors of colored cells?
# And all non-8, non-colored positions NOT diagonal neighbors?

# This seems promising. Let me implement and test.

def solve_10fcaaa3(grid):
    h = len(grid)
    w = len(grid[0])
    oh, ow = 2*h, 2*w

    # Find colored cells
    colored = set()
    color_map = {}
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                color_map[(r, c)] = grid[r][c]

    # Tile 2x2
    result = [[0]*ow for _ in range(oh)]
    tiled_colored = set()
    for (r, c), v in color_map.items():
        for tr in range(2):
            for tc in range(2):
                nr, nc = r + tr*h, c + tc*w
                result[nr][nc] = v
                tiled_colored.add((nr, nc))

    # Place 8s at diagonal neighbors of colored cells (if not already colored)
    for cr, cc in tiled_colored:
        for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = cr+dr, cc+dc
            if 0 <= nr < oh and 0 <= nc < ow and result[nr][nc] == 0:
                result[nr][nc] = 8

    return result

for ti, pair in enumerate(task10['train']):
    out = solve_10fcaaa3(pair['input'])
    match = out == pair['output']
    print(f"  10fcaaa3 train {ti}: {'PASS' if match else 'FAIL'}")
    if not match:
        # Show differences
        for r in range(len(out)):
            for c in range(len(out[0])):
                if r < len(pair['output']) and c < len(pair['output'][0]):
                    if out[r][c] != pair['output'][r][c]:
                        print(f"    ({r},{c}): got {out[r][c]} expected {pair['output'][r][c]}")

if test_solve('10fcaaa3', solve_10fcaaa3):
    solutions['10fcaaa3'] = {
        "code": """def solve(grid):
    h = len(grid)
    w = len(grid[0])
    oh, ow = 2*h, 2*w
    color_map = {}
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                color_map[(r, c)] = grid[r][c]
    result = [[0]*ow for _ in range(oh)]
    tiled_colored = set()
    for (r, c), v in color_map.items():
        for tr in range(2):
            for tc in range(2):
                nr, nc = r + tr*h, c + tc*w
                result[nr][nc] = v
                tiled_colored.add((nr, nc))
    for cr, cc in tiled_colored:
        for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = cr+dr, cc+dc
            if 0 <= nr < oh and 0 <= nc < ow and result[nr][nc] == 0:
                result[nr][nc] = 8
    return result""",
        "description": "Tile input 2x2 and place 8s at diagonal neighbors of colored cells"
    }
    print("  SAVED!")

# ===================== Task 09629e4f (another attempt) =====================
print("\n" + "=" * 50)
print("Task 09629e4f: Another approach")

# Let me look at it as a transformation of a 9x9 grid (ignoring separator rows/cols).
# The input 11x11 grid has separator rows at r=3,7 and cols at c=3,7.
# This gives a 9x9 inner grid.
# Each cell in the 9x9 has a value from {0, 2, 3, 4, 6, 8}.
# The output maps each 3x3 block to a single color.

# Let me try: for each 3x3 block in the 9x9, look at the 9 cell positions.
# The positions of the 4 output colors {2,3,4,6} might form a pattern.
# And the position of 8 (or 0) might indicate which color wins.

# New idea: In each 3x3 block, the 5 non-zero cells have colors {2,3,4,6,8}.
# 8 always appears at some specific position. The position of 8 in the block
# tells us which color would occupy that position if 8 weren't there.
# The output color is the color that 8 "replaces" or "points to".

# But actually, I already checked this (the "8 position" approach) and it didn't work.

# Let me try: for each block, look at which cells are 0. The 0-cell positions
# define a pattern that maps to the output.

# Actually, let me try a completely different approach. Maybe the grid as a whole
# has a pattern where each 3x3 block represents a "vote" for a color,
# and the block that gets the most votes from its row and column wins.

# Or maybe: think of it as 9 Sudoku-like puzzles. Each color needs to appear
# exactly once in each row of blocks and each column of blocks.

# In train 0 output:
# [[2, 0, 0], [0, 4, 3], [6, 0, 0]]
# Column 0: [2, 0, 6]. Column 1: [0, 4, 0]. Column 2: [0, 3, 0].
# Row 0: [2, 0, 0]. Row 1: [0, 4, 3]. Row 2: [6, 0, 0].
# Each non-zero value appears exactly once. And exactly 4 non-zeros.

# Could the non-zero blocks be where a specific color appears at its "correct"
# row OR column position?

# I wonder if the answer is simpler than I think. Maybe:
# For each block, there's exactly one color c such that c appears at
# position (r,c) in the TRANSPOSED block coordinates.
# No wait, let me try: the output at block (R,C) is the color that appears
# at cell (C,R) within that block (transposed block coordinates).

task09 = load_task('09629e4f')
print("\nTesting: output = color at cell (bc, br) within block (br, bc)")
for ti in range(4):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']
    print(f"  Train {ti}:")
    all_match = True
    for br in range(3):
        for bc in range(3):
            rs, cs = br*4, bc*4
            val = inp[rs+bc][cs+br]  # (bc, br) within block
            out_val = out[rs][cs]
            match = val == out_val
            if not match:
                all_match = False
            if out_val != 0 or val != 0:
                print(f"    Block ({br},{bc}): cell({bc},{br})={val}, output={out_val}, {'OK' if match else 'MISMATCH'}")
    print(f"    All match: {all_match}")

# Save solutions
print("\n" + "=" * 50)
print(f"\nTotal solutions: {len(solutions)}")

try:
    with open('data/arc_python_solutions.json') as f:
        existing = json.load(f)
except:
    existing = {}

existing.update(solutions)
with open('data/arc_python_solutions.json', 'w') as f:
    json.dump(existing, f, indent=2)
print(f"Saved {len(existing)} total solutions to data/arc_python_solutions.json")
