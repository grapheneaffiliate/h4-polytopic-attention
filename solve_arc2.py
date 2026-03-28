import json

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
            print(f"    Expected: {pair['output']}")
            print(f"    Got:      {out}")
            all_pass = False
    return all_pass

solutions = {}

# ===================== Task 025d127b =====================
print("=" * 50)
print("Task 025d127b")

# The parallelogram needs to become more rectangular.
# Looking at the outputs more carefully:
# Train 1 (8-shape): the output makes the bottom of the left edge and the top of the right edge
# match, creating a trapezoid that's more symmetric.
#
# Actually I think the rule is: make the shape symmetric about its horizontal center axis.
# Let me check:
# Train 1 (8-shape only, no 2-shape):
# Input rows 1-5:
#   Row 1: cols 1,2,3,4,5 (top edge)
#   Row 2: cols 1, 6
#   Row 3: cols 2, 7
#   Row 4: cols 3, 8
#   Row 5: cols 4,5,6,7,8 (bottom edge)
#
# Output rows 1-5:
#   Row 1: cols 2,3,4,5,6 (top edge - shifted right)
#   Row 2: cols 2, 7
#   Row 3: cols 3, 8
#   Row 4: cols 4, 8 (right edge became vertical)
#   Row 5: cols 4,5,6,7,8 (bottom edge - same)
#
# If symmetric about center (row 3):
#   Row 1 should mirror row 5: row 5 = cols 4-8, so row 1 should be cols 4-8? No, it's 2-6.
# Not simply symmetric.
#
# Let me think about it as: the shape is a parallelogram that gets its right edge made vertical.
# Input right edge: row 1 col 5, row 2 col 6, row 3 col 7, row 4 col 8, row 5 col 8
# Output right edge: row 1 col 6, row 2 col 7, row 3 col 8, row 4 col 8, row 5 col 8
# The right edge shifted up by one step: the col 8 starts at row 3 instead of row 4.
#
# Input left edge: row 1 col 1, row 2 col 1, row 3 col 2, row 4 col 3, row 5 col 4
# Output left edge: row 1 col 2, row 2 col 2, row 3 col 3, row 4 col 4, row 5 col 4
# The left edge also shifted: col 4 starts at row 4 instead of row 5.
#
# So the right side became more vertical (the max-col position extends one row higher).
# And the left side also became more vertical (the min-col-at-bottom position extends one row higher).
#
# For train 0 (6-shape):
# Input: rows 1-5
#   Left edge: 1,1,2,3,4 -> Output: 2,2,3,4,4 (shift right 1, except bottom stays)
#   Right edge: 3,4,5,6,6 -> Output: 4,5,6,6,6 (shift right 1, except bottom stays)
# Same pattern!
#
# For train 0 (2-shape):
# Input rows 7-9:
#   Left edge: 2,2,3 -> Output: 3,3,3
#   Right edge: 4,5,5 -> Output: 5,5,5
# Left shifted by 1 except bottom. Right shifted by 1 except bottom.
# BUT row 9 left was 3 and stays 3. Row 9 right was 5 and stays 5. Bottom stays.
# Row 8 left 2->3, right 5->5 (already at max). Row 7 left 2->3, right 4->5.
#
# Hmm wait, for the 2-shape: max_col=5 (bottom right).
# Row 7 right was 4 -> +1 = 5 = max_col. OK.
# Row 8 right was 5 -> already at max, stays 5. But wait, row 8 was the MIDDLE row.
# The 2-shape only has 3 rows, and row 9 is bottom.
# Row 8 left 2->3, right: input had cells at cols 2 and 5 (wait, let me recheck)

# Input 2-shape:
# Row 7: [0,0,2,2,2,0,0,0,0] cols 2,3,4
# Row 8: [0,0,2,0,0,2,0,0,0] cols 2,5
# Row 9: [0,0,0,2,2,2,0,0,0] cols 3,4,5
#
# Output:
# Row 7: [0,0,0,2,2,2,0,0,0] cols 3,4,5
# Row 8: [0,0,0,2,0,2,0,0,0] cols 3,5
# Row 9: [0,0,0,2,2,2,0,0,0] cols 3,4,5
#
# So the 2-shape went from parallelogram to rectangle! cols 3-5 for all rows.
# In the input, left edge was 2,2,3. In output, 3,3,3 (all equal to bottom left).
# Right edge was 4,5,5. In output, 5,5,5 (all equal to bottom right).
#
# But for the top edge, row 7 had cols 2,3,4 -> became 3,4,5. Shifted right by 1.
# The interior cells also shifted. So it's not just edges, it's ALL cells.
#
# Every cell in the shape (except bottom row) shifts right by 1.
# But in my previous code, I had max_col cap. Let me check why it failed.
#
# For the 6-shape in train 0:
# Row 1 input: [0,6,6,6,0,0,0,0,0] cols 1,2,3
# Expected output row 1: [0,0,6,6,6,0,0,0,0] cols 2,3,4
# My code: each cell shifts right by 1 IF c+1 <= max_col.
# max_col for 6-shape = max(6 in bottom row cols 4,5,6) = 6.
# Col 1+1=2 <= 6 YES. Col 2+1=3 <= 6 YES. Col 3+1=4 <= 6 YES.
# So output should be cols 2,3,4. That's correct.
#
# But my code gave wrong output. Let me check: when cells overlap on shift,
# maybe the issue is that flood_fill with 4-connectivity doesn't work for
# shapes with diagonal connections?

# The 6-shape has cells at (1,1),(1,2),(1,3), (2,1),(2,4), (3,2),(3,5), (4,3),(4,6), (5,4),(5,5),(5,6)
# (1,1) and (2,1) are connected. But (2,4) is NOT connected to (1,3) by 4-connectivity!
# (1,3) and (2,4) are diagonal neighbors. So flood fill would split the shape!

# I need to use 8-connectivity for flood fill (include diagonals).
# Or better, treat all cells of the same color as one component.

# Actually looking at the shapes, they're all outlines of parallelograms.
# The outline has diagonal edges that aren't 4-connected.
# Let me just group by color instead of flood fill.

def solve_025d127b(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [[0]*cols for _ in range(rows)]

    # Group cells by color
    color_cells = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != 0:
                if v not in color_cells:
                    color_cells[v] = []
                color_cells[v].append((r, c))

    for color, cells in color_cells.items():
        bottom_row = max(r for r, c in cells)
        max_col = max(c for r, c in cells)

        for r, c in cells:
            if r == bottom_row:
                result[r][c] = color
            else:
                new_c = c + 1 if c + 1 <= max_col else c
                result[r][new_c] = color

    return result

if test_solve('025d127b', solve_025d127b):
    solutions['025d127b'] = {
        "code": """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [[0]*cols for _ in range(rows)]
    color_cells = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != 0:
                if v not in color_cells:
                    color_cells[v] = []
                color_cells[v].append((r, c))
    for color, cells in color_cells.items():
        bottom_row = max(r for r, c in cells)
        max_col = max(c for r, c in cells)
        for r, c in cells:
            if r == bottom_row:
                result[r][c] = color
            else:
                new_c = c + 1 if c + 1 <= max_col else c
                result[r][new_c] = color
    return result""",
        "description": "Shift parallelogram shapes right by 1 (except bottom row) to reduce shear, capped at max column"
    }
    print("  SAVED!")

# ===================== Task 09629e4f =====================
print("\n" + "=" * 50)
print("Task 09629e4f")

# Grid divided into 3x3 blocks by rows/cols of 5s.
# Each block contains colored cells. Output fills each block with one color or 0.
# Need to figure out which color wins.
#
# Let me look at multiple training examples to find the pattern.

task09 = load_task('09629e4f')

for ti, pair in enumerate(task09['train']):
    inp = pair['input']
    out = pair['output']
    print(f"\n  Train {ti}:")
    for br in range(3):
        for bc in range(3):
            r_start = br * 4
            c_start = bc * 4
            colors = {}
            for r in range(r_start, r_start + 3):
                for c in range(c_start, c_start + 3):
                    v = inp[r][c]
                    if v != 0 and v != 5:
                        colors[v] = colors.get(v, 0) + 1
            out_val = out[r_start][c_start]
            # Check what color appears at what positions
            block = []
            for r in range(r_start, r_start + 3):
                row = []
                for c in range(c_start, c_start + 3):
                    row.append(inp[r][c])
                block.append(row)
            print(f"    Block ({br},{bc}): {block} -> {out_val}")

# ===================== Task 0962bcdd =====================
print("\n" + "=" * 50)
print("Task 0962bcdd: Cross pattern expansion")
# Each cross pattern (center + 4 arms of color A, center color B) gets expanded.
# The expansion adds diagonal elements: B radiates outward diagonally,
# and A extends along the arms.

task0962 = load_task('0962bcdd')

def solve_0962bcdd(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find cross patterns: center cell with 4 orthogonal neighbors of same color
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                continue
            center_color = grid[r][c]
            # Check if this is a center of a cross (has arm_color in all 4 directions)
            if (r > 0 and r < rows-1 and c > 0 and c < cols-1):
                up = grid[r-1][c]
                down = grid[r+1][c]
                left = grid[r][c-1]
                right = grid[r][c+1]
                if up == down == left == right and up != 0 and up != center_color:
                    arm_color = up
                    # This is a cross center. Expand it.
                    # Add arm_color extending further along arms
                    # Add center_color on diagonals
                    for dist in range(1, max(rows, cols)):
                        # Extend arms
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r + dr*dist, c + dc*dist
                            if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] == 0:
                                result[nr][nc] = arm_color
                        # Add center_color on diagonals
                        for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                            nr, nc = r + dr*dist, c + dc*dist
                            if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] == 0:
                                result[nr][nc] = center_color

    return result

if test_solve('0962bcdd', solve_0962bcdd):
    solutions['0962bcdd'] = {
        "code": """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                continue
            center_color = grid[r][c]
            if (r > 0 and r < rows-1 and c > 0 and c < cols-1):
                up = grid[r-1][c]
                down = grid[r+1][c]
                left = grid[r][c-1]
                right = grid[r][c+1]
                if up == down == left == right and up != 0 and up != center_color:
                    arm_color = up
                    for dist in range(1, max(rows, cols)):
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r + dr*dist, c + dc*dist
                            if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] == 0:
                                result[nr][nc] = arm_color
                        for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                            nr, nc = r + dr*dist, c + dc*dist
                            if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] == 0:
                                result[nr][nc] = center_color
    return result""",
        "description": "Expand cross patterns: extend arms and add center color on diagonals"
    }
    print("  SAVED!")

# ===================== Task 10fcaaa3 =====================
print("\n" + "=" * 50)
print("Task 10fcaaa3: Tile pattern with 8s between colored cells")
# Input has colored non-zero cells on a background of 0.
# Output doubles the grid dimensions and fills with 8s between colored cell copies.
# The colored cells appear to tile, and 8s fill the spaces between them.

# Let me look more carefully at the pattern.
# Input 1: 2x4, Output: 4x8 (doubled dimensions)
# Input [[0,0,0,0],[0,5,0,0]] -> Output [[8,0,8,0,8,0,8,0],[0,5,0,0,0,5,0,0],[8,0,8,0,8,0,8,0],[0,5,0,0,0,5,0,0]]
# So the output tiles the input 2x2 times, and in the positions where the
# original had 0, alternate with 8 based on some checkerboard pattern.
# Actually: the output is 2x the input in each dimension.
# Rows 0,2 of output: 8,0 pattern replaces 0s
# Rows 1,3: original row 1 repeated twice

# Wait: output[0] = [8,0,8,0,8,0,8,0] - this is input row 0 = [0,0,0,0] but with 0s replaced by checkerboard 8s
# output[1] = [0,5,0,0,0,5,0,0] - input row 1 = [0,5,0,0] tiled twice
# output[2] = [8,0,8,0,8,0,8,0] - same as row 0
# output[3] = [0,5,0,0,0,5,0,0] - same as row 1

# So the output tiles the input 2x2 and then replaces 0s with 8 in a checkerboard pattern?
# Let me check: output[0][0] = 8, [0][1] = 0, [0][2] = 8, [0][3] = 0...
# That's checkerboard with 8 at even columns.
# output[1][0] = 0, [1][1] = 5, [1][2] = 0, [1][3] = 0...
# Original input[1] = [0,5,0,0]. So tiled gives [0,5,0,0,0,5,0,0].
# No 8s inserted here because the original had specific non-zero cells?
#
# Actually, I think the rule is:
# 1. Tile the input 2x2
# 2. For any cell that is 0 in the tiled version AND both (row+col) is even (checkerboard), place 8
#
# Wait, output[0][0]=8: row=0, col=0, sum=0 (even). Tiled value=0. -> 8. YES.
# output[0][1]=0: row=0, col=1, sum=1 (odd). Tiled value=0. -> stays 0.
# output[1][0]=0: row=1, col=0, sum=1 (odd). Tiled value=0. -> stays 0.
# output[1][1]=5: tiled value=5 (non-zero). -> stays 5.
#
# Hmm but what about output[1][2]=0? row=1, col=2, sum=3 (odd). Tiled value=0. -> stays 0.
# And output[1][4]=0? Tiled value = input[1][4%4]=input[1][0]=0. sum=5 (odd). -> stays 0.
# output[1][5]=5? Tiled = input[1][5%4]=input[1][1]=5. stays 5. YES.

# Let me check training example 2:
# Input: 3x4, [[0,0,6,0],[0,0,0,0],[0,6,0,0]]
# Output: 6x8
# Expected: [[0,0,6,0,0,0,6,0],[8,8,8,8,8,8,8,8],[0,6,0,8,0,6,0,8],[8,0,6,0,8,0,6,0],[8,8,8,8,8,8,8,8],[0,6,0,0,0,6,0,0]]
#
# Row 0 output: [0,0,6,0,0,0,6,0] - tiled input row 0 = [0,0,6,0,0,0,6,0]. No 8s.
# Row 1 output: [8,8,8,8,8,8,8,8] - all 8s. Input row 1 is all 0s.
# Row 2 output: [0,6,0,8,0,6,0,8] - tiled row 2 = [0,6,0,0,0,6,0,0] but cols 3,7 became 8.
#
# For row 2, (r,c) = (2,3): tiled value = input[2][3]=0. (2+3)=5 odd -> should stay 0?
# But output is 8! So my checkerboard theory is wrong.

# Let me reconsider. The 8s seem to trace paths between the colored cells.
# In example 2, colored cells at (0,2)=6 and (2,1)=6.
# The line from (0,2) to (2,1) goes diagonally through (1,1.5).
# The 8s fill the space between them.

# Actually let me look at example 3: input [[0,0,0],[0,4,0],[0,0,0],[0,0,0],[4,0,0]]
# Output 10x6:
# [[8,0,8,8,0,8],[0,4,0,0,4,0],[8,0,8,8,0,8],[0,8,8,0,8,0],[4,0,0,4,0,0],
#  [8,8,8,8,8,8],[0,4,0,0,4,0],[8,0,8,8,0,8],[0,8,8,0,8,0],[4,0,0,4,0,0]]
#
# The output tiles 2x2 and places 8s. Let me think about what the 8 placement rule is.
#
# In the tiled grid (before 8s), the colored cells are at specific positions.
# The 8s connect these colored cells along shortest Manhattan paths?
#
# Or maybe: for each 0 in the tiled grid, check if it's adjacent (including diag) to
# a colored cell via some path. The 8s mark positions that are equidistant or
# on the Voronoi boundaries.
#
# Actually, I think the pattern might be simpler:
# For each pair of colored cells in the tiled grid, draw 8s on the diagonal/straight
# path between them.
#
# Let me try another approach: maybe the 8s appear at positions where
# (manhattan_distance to nearest colored cell) is odd, or some function of position.

# Actually, let me look at it as: for each 0 cell in the tiled output,
# place 8 if it's on the "boundary" between two colored cells in some sense.

# Hmm, let me look at example 4 which is simpler:
# Input [[0,0,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]], single colored cell at (1,1)
# Output [[8,0,8,0,8,0,8,0],[0,2,0,0,0,2,0,0],[8,0,8,0,8,0,8,0],[0,0,0,0,0,0,0,0],
#          [8,0,8,0,8,0,8,0],[0,2,0,0,0,2,0,0],[8,0,8,0,8,0,8,0],[0,0,0,0,0,0,0,0]]
#
# Tiled colored cells at (1,1),(1,5),(5,1),(5,5) (value 2).
# 8s appear at: all positions where row is even AND col is even AND cell is 0 in tiled grid.
# Row 0: cols 0,2,4,6 = 8. Row 2: cols 0,2,4,6 = 8. Row 3: no 8s. Row 4: cols 0,2,4,6 = 8.
#
# (0,0): even+even=even -> 8. (0,1): even+odd -> no. (1,0): odd+even -> no.
#
# But for example 2, row 1 is all 8s: [8,8,8,8,8,8,8,8].
# Row 1 is odd. So it can't be "even row AND even col" pattern.
# Row 1 in example 2: input row 1 is [0,0,0,0] (all zeros).
# The colored cells connect from (0,2) through (2,1).
# Row 1 has 8s everywhere. That's different from example 4.

# I think the pattern relates to how the colored cells connect to each other.
# Let me look at it differently:
# For each pair of adjacent colored cells (in the tiled grid), draw 8s
# on the lines connecting them.
# The direction between (0,2) and (2,1) is diagonal down-left.
# Between them: (1,1.5) -> places 8 at (1,1) and (1,2)? But output has ALL of row 1 as 8.

# I'm going to try a different theory: the 8s form the shortest path between
# all colored cells in the tiled grid. Like a minimum spanning tree of 8s.

# Or maybe: 8 replaces 0 wherever the chebyshev distance to the nearest
# colored cell equals a certain value.

# Let me compute distances for example 4:
# Colored cells at (1,1),(1,5),(5,1),(5,5)
# (0,0): chebyshev dist to (1,1) = 1. -> 8
# (0,1): chebyshev to (1,1) = 1. -> 0 (not 8)
# Hmm, both distance 1 but different result. Not chebyshev distance.

# Manhattan: (0,0) to (1,1) = 2. (0,1) to (1,1) = 1.
# (0,0) -> 8, manhattan=2 (even). (0,1) -> 0, manhattan=1 (odd).
# (0,2) -> 8, manhattan to (1,1)=2. (0,4) -> 8, manhattan to (1,5)=2.
# (1,0) -> 0, manhattan to (1,1)=1.
# (3,0) -> 0, manhattan to (1,1)=3 or to (5,1)=3. Odd.
# (3,3) -> 0, manhattan to nearest (1,1)=4 or (5,5)=4 or (1,5)=4 or (5,1)=4. Even. But output is 0.
# So it's not just manhattan distance parity.

# Let me try: (row + col) % 2 == 0 AND the cell is 0 in tiled grid -> place 8?
# Ex 4: (0,0): 0+0=0 even, cell=0 -> 8. YES.
# (0,1): 0+1=1 odd -> no 8. YES.
# (3,0): 3+0=3 odd -> no 8. Output=0. YES.
# (3,3): 3+3=6 even, cell=0 -> should be 8. But output=0!
# NOPE.

# OK let me look at output row 3 of example 4: [0,0,0,0,0,0,0,0]. All zeros.
# And row 4: [8,0,8,0,8,0,8,0]. 8s at even cols.
# So rows 0,2,4,6 have 8s at even cols. Rows 1,3,5,7 have no 8s.
# BUT colored cells are at rows 1,5 cols 1,5.
# So: 8 at (r,c) if r is even AND c is even AND cell is 0.
# (3,3): r=3 odd -> no 8. Correct!
# This works for example 4.

# But example 2:
# Row 1 (odd): [8,8,8,8,8,8,8,8]. All 8s. But row 1 is odd!
# So "row even AND col even" doesn't work for example 2.

# Maybe the rule depends on the positions of the colored cells.
# In example 4, colored cell at (1,1) in the base pattern.
# In example 2, colored cells at (0,2) and (2,1).
# The 8s trace the connection between them?

# Let me think about it as vectors. Each colored cell creates a "wave" of 8s.
# The 8s appear at positions that are reflections/translations of the colored cells.

# Actually, I think I should look at this differently.
# The input has colored cells at specific positions.
# Between each pair of colored cells, draw a line of 8s.
# The line goes horizontally, vertically, or diagonally.

# For example 2: cells at (0,2) and (2,1).
# Direction from (0,2) to (2,1): (2,-1), so diagonal.
# The 8s appear along the diagonal between them.
# But row 1 has ALL 8s, not just the diagonal.

# Let me try yet another approach. Maybe the non-zero cells define a lattice,
# and the 8s fill certain positions in that lattice.

# For the tiled grid, colored cells repeat. Between them, 8s fill the "cross"
# pattern centered between pairs.

# Actually, I just realized: the output is NOT simply the input tiled 2x2.
# Let me recompute. Example 2:
# Input: 3 rows, 4 cols. Output: 6 rows, 8 cols.
# Row 2 output: [0,6,0,8,0,6,0,8]
# Tiled row 2 = input[2] * 2 = [0,6,0,0,0,6,0,0]
# But output[2] = [0,6,0,8,0,6,0,8]
# The 0s at cols 3 and 7 became 8.

# What if the colored cells define vectors, and 8s appear at positions
# reachable by those vectors from a colored cell?
# Cell (0,2): vector to (2,1) is (2,-1). So from (0,2), go (2,-1) -> (2,1).
# And from (2,1), go (2,-1) -> (4,0). From (0,2) go (-2,1) -> (-2,3)=invalid.
# From (2,1) go (-2,1) -> (0,2). This traces the diagonal.

# The 8s in example 2 row 1: all 8. That's positions (1,0)-(1,7).
# How does a diagonal from (0,2) to (2,1) give all of row 1?
# Step from (0,2) in direction (1,-0.5) gives (1, 1.5) which rounds to (1,1) and (1,2)?
# But we need all of (1,0)-(1,7).

# I think the actual rule might be: for each 0 cell in the output,
# place 8 if it lies between two colored cells (in any direction).
# "Between" meaning on the straight line connecting them.
# But with tiling, every point is between some pair.

# Let me try the SIMPLEST possible rule: tile 2x2, replace all remaining 0s with 8.
# Example 4: output has rows 3 and 7 as all 0s. So not all 0s become 8.

# OK, I think the 8s represent the "Voronoi boundary" between colored cells,
# or more precisely, the positions where you're equidistant from two colored cells.

# Let me just move on and come back to this one later.
# Skip 10fcaaa3 for now.

# ===================== Task 11852cab =====================
print("\n" + "=" * 50)
print("Task 11852cab: Diamond pattern symmetry completion")
# Input has a diamond/cross shape with some cells filled.
# Output completes the symmetry (rotational symmetry).
# The pattern has 8s on the border and colored cells inside.
# Some corners are missing and need to be filled by symmetry.

task11 = load_task('11852cab')
for i, pair in enumerate(task11['train']):
    print(f"\n  Train {i}:")
    print("  Input:")
    for r in pair['input']:
        print(f"    {r}")
    print("  Output:")
    for r in pair['output']:
        print(f"    {r}")

# Train 0:
# Input has a diamond shape with 8s and 2s and 3s.
# There's a pattern: 8 on edges, colored inside. The top-left corner is missing (0).
# Output fills in: row 1 has 3 at (1,2) and (1,6), row 5 has 3 at (5,2) and (5,6).
# The pattern is a diamond with 4-fold symmetry (rotate 90 degrees).
#
# Actually looking more carefully at the pattern:
# The diamond has 8s forming a border and 2s on the cross axes.
# Some positions have a unique color (3 for train 0, 4 for train 1).
# The input has some zeros that should be the unique color.
# The output fills those zeros with the correct color to complete the symmetry.

# For train 0:
# Input:  row 1: [0,0,3,0,8,0,0,0,0,0] -> output row 1: same (but check...)
# Actually let me compare input and output:
# Input row 1: [0,0,3,0,8,0,0,0,0,0]
# Output row 1: [0,0,3,0,8,0,3,0,0,0]
# Added 3 at (1,6)!
# Input row 5: [0,0,0,0,8,0,0,0,0,0]
# Output row 5: [0,0,3,0,8,0,3,0,0,0]
# Added 3 at (5,2) and (5,6)

# The pattern has 4 symmetry positions. The non-8, non-0 value (like 3)
# appears at symmetric positions around the center of the diamond.
# The center is at the intersection of the 8-cross.

# For train 0, the diamond center seems to be at the center of the 8 cross.
# 8s are at: row 1 col 4, row 2 col 3 and 5, row 3 col 2 and 4 and 6, etc.
# Actually looking at the 8 positions:
# (1,4), (2,3),(2,5), (3,2),(3,4),(3,6), (4,3),(4,5), (5,4)
# This is a diamond/cross pattern centered at (3,4).
# Actually the 8s form a diamond border around center (3,4).

# The 2s are at: (2,4),(3,3),(3,5),(4,4) - these form a smaller cross at the center.
# Actually input (3,4)=3 and (3,3)=8... let me re-read.

# Input train 0:
# Row 0: [0,0,0,0,0,0,0,0,0,0] - empty
# Row 1: [0,0,3,0,8,0,0,0,0,0]
# Row 2: [0,0,0,2,0,2,0,0,0,0]
# Row 3: [0,0,8,0,3,0,8,0,0,0]
# Row 4: [0,0,0,2,0,2,0,0,0,0]
# Row 5: [0,0,0,0,8,0,0,0,0,0]

# So the diamond pattern:
# center = (3,4) with value 3
# Cross arms (distance 1): 2s at (2,3),(2,5),(4,3),(4,5)
# Cross arms (distance 2): 8s at (1,4),(3,2),(3,6),(5,4)
# Corners (distance 1 diag): 3 at (1,2) in input, should also be at (1,6),(5,2),(5,6)
#
# Wait, (1,2) has value 3 in input. The center (3,4) also has 3.
# (1,6) should also be 3 by symmetry. Output confirms: (1,6)=3.
# (5,2) and (5,6) should be 3. Output confirms: (5,2)=3, (5,6)=3.
# But in input, (5,2) and (5,6) and (1,6) were 0. Output fills them.

# So the rule is: the pattern has 4-fold rotational symmetry around its center.
# Any missing (0) positions that should have a value by symmetry get filled.

# For train 1:
# Input:
# Row 1: [0,0,0,8,0,8,0,8,0,0]
# Row 2: [0,0,0,0,4,0,0,0,0,0] -> should be at (2,4) and (2,6)?
# Row 3: [0,0,0,8,0,1,0,8,0,0]
# Row 4: [0,0,0,0,0,0,0,0,0,0]
# Row 5: [0,0,0,8,0,8,0,8,0,0]
#
# Output:
# Row 1: same
# Row 2: [0,0,0,0,4,0,4,0,0,0] -> added 4 at (2,6)
# Row 3: same
# Row 4: [0,0,0,0,4,0,4,0,0,0] -> added 4s
# Row 5: same

# So the center is (3,5) with value 1.
# 8s form border: (1,3),(1,5),(1,7),(3,3),(3,7),(5,3),(5,5),(5,7)
# 4s should be at (2,4),(2,6),(4,4),(4,6) by 4-fold symmetry.
# Input had 4 only at (2,4). Output fills (2,6),(4,4),(4,6).

# Great! The rule is: find the cross/diamond pattern, identify its center,
# and fill missing positions using 4-fold rotational symmetry.

# For train 2:
# Row 1: [0,0,0,8,0,8,0,8,0,0]
# Row 2: [0,0,0,0,4,0,0,0,0,0]
# Row 3: [0,0,0,8,0,1,0,8,0,0]
# ...
# The center is at the unique non-8 non-0 value that's not at a diagonal position.
# Actually the center is the cell with value that's different from both 8 and the other color.
# In train 0: center has value 3 (same as diagonal). Actually 3 is the center AND the diagonal.
# The 2s are the cross arms, 8s are the outer border.
# In train 1: center has value 1, 4s are diagonals.

# Let me think about this more carefully. The pattern is:
# A cross/plus shape with 8 at the tips, 2 or 4 at adjacent positions,
# and a center color. The 4-fold symmetry fills in missing values.

# Implementation:
# 1. Find the non-zero region
# 2. Find the center of the pattern
# 3. For each non-zero cell, rotate 90, 180, 270 around center and fill

def solve_11852cab(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find all non-zero cells
    non_zero = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                non_zero[(r,c)] = grid[r][c]

    if not non_zero:
        return result

    # Find center: average of all non-zero positions
    # Or find the 8-cross center
    all_r = [r for r,c in non_zero]
    all_c = [c for r,c in non_zero]
    center_r = (min(all_r) + max(all_r)) / 2
    center_c = (min(all_c) + max(all_c)) / 2

    # Apply 4-fold symmetry
    for (r, c), val in non_zero.items():
        # Rotate around center by 90, 180, 270
        dr = r - center_r
        dc = c - center_c
        rotations = [
            (center_r - dc, center_c + dr),  # 90
            (center_r - dr, center_c - dc),  # 180
            (center_r + dc, center_c - dr),  # 270
        ]
        for nr, nc in rotations:
            nr, nc = int(round(nr)), int(round(nc))
            if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] == 0:
                result[nr][nc] = val

    return result

if test_solve('11852cab', solve_11852cab):
    solutions['11852cab'] = {
        "code": """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]
    non_zero = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                non_zero[(r,c)] = grid[r][c]
    if not non_zero:
        return result
    all_r = [r for r,c in non_zero]
    all_c = [c for r,c in non_zero]
    center_r = (min(all_r) + max(all_r)) / 2
    center_c = (min(all_c) + max(all_c)) / 2
    for (r, c), val in non_zero.items():
        dr = r - center_r
        dc = c - center_c
        rotations = [
            (center_r - dc, center_c + dr),
            (center_r - dr, center_c - dc),
            (center_r + dc, center_c - dr),
        ]
        for nr, nc in rotations:
            nr, nc = int(round(nr)), int(round(nc))
            if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] == 0:
                result[nr][nc] = val
    return result""",
        "description": "Complete 4-fold rotational symmetry of diamond/cross pattern"
    }
    print("  SAVED!")

# ===================== Task 1190e5a7 =====================
print("\n" + "=" * 50)
print("Task 1190e5a7: Grid with lines, find smallest rectangle cell dimensions")
# Grid divided by lines of a separator color. Output is the dimensions of the smallest cell.
# The grid has horizontal and vertical lines of one color dividing it into cells.
# Output is a single-color grid of the dimensions of the smallest cell.

task1190 = load_task('1190e5a7')
for i, pair in enumerate(task1190['train']):
    inp = pair['input']
    out = pair['output']
    print(f"  Train {i}: input {len(inp)}x{len(inp[0])}, output {len(out)}x{len(out[0])}")
    # Find the separator color (the one that forms full rows/columns)
    bg = inp[0][0]  # background color
    # Find separator: appears in full rows or full columns
    # Check which rows are all one color
    row_colors = set()
    for r in range(len(inp)):
        if all(inp[r][c] == inp[r][0] for c in range(len(inp[0]))) and inp[r][0] != bg:
            row_colors.add(inp[r][0])
    print(f"    Background: {bg}, separator candidates from rows: {row_colors}")
    print(f"    Output: {out}")

# Train 0: 15x15 grid with 7 as separator, 3 as background.
# Horizontal separator rows: some rows are all 7s
# Vertical separator cols: some cols are all 7s
# Output: 4x4 grid of all 3s. This means the smallest cell in the grid is 4x4.

# Train 1: 11x11, background 1, separator 8.
# Output: 2x2 of 1s. Smallest cell is 2x2? Let me check.
# Actually output is [[1,1],[1,1],[1,1]] -> 3x2.

# Train 2: 27x27 with separators.
# Output: 6x6 of all 3s.

# So the rule is: find the grid division, measure each cell's dimensions,
# and output a grid of background color with the smallest cell's dimensions.
# But what defines "smallest"? The unique cell that has different dimensions.

# Actually: looking at the grid structure, each grid is divided into cells of varying sizes.
# The output represents the most common cell size, or perhaps there's one cell size that
# appears most often or uniquely.

# Let me reconsider. Looking at train 0: the grid is divided into cells.
# The separator rows/cols divide it into a grid of cells with different sizes.
# Some are larger (like the big 10x10 cell on the left) and some smaller.
# The output = smallest cell dimensions filled with background.

# Implementation:
# 1. Find separator value
# 2. Find positions of separator rows and separator columns
# 3. Compute the cell dimensions between consecutive separators
# 4. Find the smallest rectangle dimensions
# 5. Output that size filled with background color

def solve_1190e5a7(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find separator color: the color that forms complete rows
    bg = None
    sep = None
    for r in range(rows):
        row_vals = set(grid[r])
        if len(row_vals) == 1:
            # Full row of one color
            if sep is None:
                sep = grid[r][0]
            elif grid[r][0] != sep:
                # Multiple candidates, keep the one that appears in full rows
                pass

    # Find background: the other color
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != sep:
                bg = grid[r][c]
                break
        if bg is not None:
            break

    # Find separator row positions
    sep_rows = [-1]  # virtual separator before row 0
    for r in range(rows):
        if all(grid[r][c] == sep for c in range(cols)):
            sep_rows.append(r)
    sep_rows.append(rows)  # virtual separator after last row

    # Find separator col positions
    sep_cols = [-1]
    for c in range(cols):
        if all(grid[r][c] == sep for r in range(rows)):
            sep_cols.append(c)
    sep_cols.append(cols)

    # Compute cell heights and widths
    heights = []
    for i in range(len(sep_rows) - 1):
        h = sep_rows[i+1] - sep_rows[i] - 1
        if h > 0:
            heights.append(h)

    widths = []
    for i in range(len(sep_cols) - 1):
        w = sep_cols[i+1] - sep_cols[i] - 1
        if w > 0:
            widths.append(w)

    # Find the smallest/most common dimensions
    # Actually: find the unique smallest cell
    min_h = min(heights) if heights else 1
    min_w = min(widths) if widths else 1

    # Create output
    result = [[bg] * min_w for _ in range(min_h)]
    return result

if test_solve('1190e5a7', solve_1190e5a7):
    solutions['1190e5a7'] = {
        "code": """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    sep = None
    bg = None
    for r in range(rows):
        row_vals = set(grid[r])
        if len(row_vals) == 1:
            sep = grid[r][0]
            break
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != sep:
                bg = grid[r][c]
                break
        if bg is not None:
            break
    sep_rows = [-1]
    for r in range(rows):
        if all(grid[r][c] == sep for c in range(cols)):
            sep_rows.append(r)
    sep_rows.append(rows)
    sep_cols = [-1]
    for c in range(cols):
        if all(grid[r][c] == sep for r in range(rows)):
            sep_cols.append(c)
    sep_cols.append(cols)
    heights = []
    for i in range(len(sep_rows) - 1):
        h = sep_rows[i+1] - sep_rows[i] - 1
        if h > 0:
            heights.append(h)
    widths = []
    for i in range(len(sep_cols) - 1):
        w = sep_cols[i+1] - sep_cols[i] - 1
        if w > 0:
            widths.append(w)
    min_h = min(heights) if heights else 1
    min_w = min(widths) if widths else 1
    result = [[bg] * min_w for _ in range(min_h)]
    return result""",
        "description": "Find the smallest cell in a grid divided by separator lines, output its dimensions filled with background"
    }
    print("  SAVED!")

# ===================== Task 137eaa0f =====================
print("\n" + "=" * 50)
print("Task 137eaa0f: Assemble fragments around 5-marked center")
# Input: 11x11 grid with scattered colored fragments and 5-markers.
# Output: 3x3 grid assembled from the fragments.
# The 5s mark positions where fragments connect to the center.

task137 = load_task('137eaa0f')
for i, pair in enumerate(task137['train']):
    print(f"\n  Train {i}:")
    print("  Input:")
    for r in pair['input']:
        print(f"    {r}")
    print("  Output:")
    for r in pair['output']:
        print(f"    {r}")

# Train 0:
# Input has fragments:
# (1,6),(1,7)=6  and (2,3)=5,(2,7)=5 and (3,2),(3,3)=4
# (7,6)=7 and (8,5)=5,(8,6)=7
# Output: [[6,6,7],[0,5,7],[4,4,0]]
#
# The 5s mark the center of the 3x3 output.
# Each fragment is positioned relative to a 5, and the 5 is the center.
# So fragments are assembled around the 5-center.
#
# Fragment 1: (1,6)=6,(1,7)=6 and (2,7)=5. The 5 at (2,7) connects them.
# Relative to 5 at (2,7): (1,6) is (-1,-1), (1,7) is (-1,0).
# So in output: center is (1,1) (the 5), (-1,-1) = (0,0)=6, (-1,0)=(0,1)=6. YES!
#
# Fragment 2: (2,3)=5 and (3,2)=4,(3,3)=4
# Relative to 5 at (2,3): (3,2) is (+1,-1), (3,3) is (+1,0).
# Output: (1+1,1-1)=(2,0)=4, (1+1,1+0)=(2,1)=4. YES!
#
# Fragment 3: (7,6)=7 and (8,5)=5,(8,6)=7
# Relative to 5 at (8,5): (7,6) is (-1,+1), (8,6) is (0,+1).
# Output: (1-1,1+1)=(0,2)=7, (1+0,1+1)=(1,2)=7. YES!
#
# So the rule is:
# 1. Find all 5-cells and their surrounding non-zero, non-5 cells
# 2. Each 5 represents the center (1,1) of a 3x3 grid
# 3. The offsets from the 5 to nearby colored cells map to offsets in the 3x3 output
# 4. Combine all fragments into the 3x3 output

def solve_137eaa0f(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]  # center is always 5

    # Find all 5-cells
    fives = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                fives.append((r, c))

    # For each 5, find neighboring non-zero non-5 cells and map to output
    for fr, fc in fives:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = fr + dr, fc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    val = grid[nr][nc]
                    if val != 0 and val != 5:
                        # Map to output: center of output is (1,1)
                        or_, oc = 1 + dr, 1 + dc
                        if 0 <= or_ < 3 and 0 <= oc < 3:
                            result[or_][oc] = val

    return result

if test_solve('137eaa0f', solve_137eaa0f):
    solutions['137eaa0f'] = {
        "code": """def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
    fives = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                fives.append((r, c))
    for fr, fc in fives:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = fr + dr, fc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    val = grid[nr][nc]
                    if val != 0 and val != 5:
                        or_, oc = 1 + dr, 1 + dc
                        if 0 <= or_ < 3 and 0 <= oc < 3:
                            result[or_][oc] = val
    return result""",
        "description": "Assemble colored fragments around 5-marked centers into a 3x3 output grid"
    }
    print("  SAVED!")

# Save all solutions
print("\n" + "=" * 50)
print(f"\nTotal solutions: {len(solutions)}")

# Load existing solutions if any
try:
    with open('data/arc_python_solutions.json') as f:
        existing = json.load(f)
except:
    existing = {}

existing.update(solutions)
with open('data/arc_python_solutions.json', 'w') as f:
    json.dump(existing, f, indent=2)
print(f"Saved {len(existing)} total solutions to data/arc_python_solutions.json")
