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
            for r in range(min(len(out), len(pair['output']))):
                for c in range(min(len(out[0]) if out else 0, len(pair['output'][0]))):
                    if out[r][c] != pair['output'][r][c]:
                        print(f"    ({r},{c}): got {out[r][c]} expected {pair['output'][r][c]}")
            if len(out) != len(pair['output']):
                print(f"    Row count: got {len(out)} expected {len(pair['output'])}")
            all_pass = False
    return all_pass

solutions = {}

# ===================== Task 0962bcdd =====================
print("=" * 50)
print("Task 0962bcdd: Cross pattern - add center_color at diagonal distance 2")
# Looking at the expected output more carefully:
# Input cross at (3,2) center=2, arms=7:
#   (2,2)=7, (3,1)=7, (3,3)=7, (4,2)=7, (3,2)=2
# Expected output adds:
#   (1,2)=7, (2,1)=7, (2,3)=2(!), etc.
# Wait, the output keeps the original cross AND adds:
# Row 1: 2 at (1,0) and (1,4)
# Row 2: 2 at (2,1) and (2,3)
# Row 3: 7 at (3,0) and (3,4) (extending arms)
# Row 4: 2 at (4,1) and (4,3)
# Row 5: 2 at (5,0) and (5,4) and 7 at (5,2) and (5,6) and (5,8) and (5,10)

# Actually, the correct output for cross 1 at center (3,2):
# Row 1: (1,0)=2, (1,2)=7, (1,4)=2
# Row 2: (2,1)=2, (2,2)=7, (2,3)=2
# Row 3: (3,0)=7, (3,1)=7, (3,2)=2, (3,3)=7, (3,4)=7
# Row 4: (4,1)=2, (4,2)=7, (4,3)=2
# Row 5: (5,0)=2, (5,2)=7, (5,4)=2

# So the cross is expanded: the arm color (7) extends one more in each direction,
# AND the center color (2) appears at diagonal positions at distance 1 AND 2.

# Pattern: the original cross has arm_color at distance 1 orthogonally from center.
# The expansion adds:
# - arm_color at distance 2 orthogonally (extending arms by 1)
# - center_color at distance 1 diagonally (the 4 diagonal neighbors)
# - center_color at distance 2 diagonally (further diagonal)

# Let me verify with cross 2 at center (7,8):
# center=2, arms=7
# Input: (6,8)=7, (7,7)=7, (7,9)=7, (8,8)=7, (7,8)=2
# Expected output around it:
# (5,6)=2, (5,8)=7, (5,10)=2
# (6,7)=2, (6,8)=7, (6,9)=2
# (7,6)=7, (7,7)=7, (7,8)=2, (7,9)=7, (7,10)=7
# (8,7)=2, (8,8)=7, (8,9)=2
# (9,6)=2, (9,8)=7, (9,10)=2

# Yes! The pattern is:
# Center stays center_color
# Orthogonal dist 1: arm_color (original arms)
# Orthogonal dist 2: arm_color (new - extend arms)
# Diagonal dist 1: center_color (new)
# Diagonal dist 2: center_color (new)

# But looking at the expected output:
# (5,6)=2: that's (-2,-2) from center (7,8). Manhattan dist=4, Chebyshev dist=2. Diagonal dist 2.
# (5,8)=7: that's (-2,0). Orthogonal dist 2.
# (5,10)=2: that's (-2,+2). Diagonal dist 2.
# (6,7)=2: that's (-1,-1). Diagonal dist 1.
# (6,9)=2: that's (-1,+1). Diagonal dist 1.
# (7,6)=7: that's (0,-2). Orthogonal dist 2.
# (7,10)=7: that's (0,+2). Orthogonal dist 2.
# (9,6)=2: that's (+2,-2). Diagonal dist 2.
# (9,10)=2: that's (+2,+2). Diagonal dist 2.

# Great! So the pattern around each cross center:
# dist 2 orthogonal: arm_color
# dist 1 diagonal: center_color
# dist 2 diagonal: center_color
# Original cross kept.

def solve_0962bcdd(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                continue
            center_color = grid[r][c]
            if r > 0 and r < rows-1 and c > 0 and c < cols-1:
                up = grid[r-1][c]
                down = grid[r+1][c]
                left = grid[r][c-1]
                right = grid[r][c+1]
                if up == down == left == right and up != 0 and up != center_color:
                    arm_color = up
                    # Extend arms by 1 (distance 2 orthogonal)
                    for dr, dc in [(-2,0),(2,0),(0,-2),(0,2)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            result[nr][nc] = arm_color
                    # Center color at diagonal distance 1
                    for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            result[nr][nc] = center_color
                    # Center color at diagonal distance 2
                    for dr, dc in [(-2,-2),(-2,2),(2,-2),(2,2)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < rows and 0 <= nc < cols:
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
            if r > 0 and r < rows-1 and c > 0 and c < cols-1:
                up = grid[r-1][c]
                down = grid[r+1][c]
                left = grid[r][c-1]
                right = grid[r][c+1]
                if up == down == left == right and up != 0 and up != center_color:
                    arm_color = up
                    for dr, dc in [(-2,0),(2,0),(0,-2),(0,2)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            result[nr][nc] = arm_color
                    for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            result[nr][nc] = center_color
                    for dr, dc in [(-2,-2),(-2,2),(2,-2),(2,2)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            result[nr][nc] = center_color
    return result""",
        "description": "Expand cross patterns: extend arms by 1, add center color at diagonal positions"
    }
    print("  SAVED!")

# ===================== Task 1190e5a7 =====================
print("\n" + "=" * 50)
print("Task 1190e5a7: Grid divided by separator lines - find smallest cell")

# The issue is my separator detection. Let me debug.
task1190 = load_task('1190e5a7')

for ti, pair in enumerate(task1190['train']):
    inp = pair['input']
    rows = len(inp)
    cols = len(inp[0])
    out = pair['output']
    print(f"\n  Train {ti}: {rows}x{cols} -> {len(out)}x{len(out[0])}")

    # Find which value forms complete rows
    bg_val = None
    sep_val = None
    for r in range(rows):
        vals = set(inp[r])
        if len(vals) == 1:
            sep_val = inp[r][0]
            break

    # Find which rows and cols are separators
    sep_rows = []
    for r in range(rows):
        if all(inp[r][c] == sep_val for c in range(cols)):
            sep_rows.append(r)

    sep_cols = []
    for c in range(cols):
        if all(inp[r][c] == sep_val for r in range(rows)):
            sep_cols.append(c)

    # Background is the other value
    for r in range(rows):
        for c in range(cols):
            if inp[r][c] != sep_val:
                bg_val = inp[r][c]
                break
        if bg_val is not None:
            break

    print(f"    Separator: {sep_val}, Background: {bg_val}")
    print(f"    Sep rows: {sep_rows}")
    print(f"    Sep cols: {sep_cols}")

    # Compute cell sizes
    all_rows = [-1] + sep_rows + [rows]
    all_cols = [-1] + sep_cols + [cols]

    heights = []
    for i in range(len(all_rows)-1):
        h = all_rows[i+1] - all_rows[i] - 1
        if h > 0:
            heights.append(h)

    widths = []
    for i in range(len(all_cols)-1):
        w = all_cols[i+1] - all_cols[i] - 1
        if w > 0:
            widths.append(w)

    print(f"    Heights: {heights}")
    print(f"    Widths: {widths}")
    print(f"    Expected: {len(out)}x{len(out[0])}")

    # The output is the SMALLEST height x SMALLEST width?
    # Train 0: heights=[2,10,2], widths=[1,8,4] -> min_h=2, min_w=1 -> 2x1. Expected 2x4. WRONG.
    # So it's not simply min of each.
    # Maybe: the output dimensions are the most common h x most common w?
    # Or the second smallest?
    # Or: the smallest unique cell that's not a 1-wide sliver?

    # Actually, looking at train 0 more carefully:
    # Heights: [2, 10, 2]. Min = 2.
    # Widths: [1, 8, 4]. Min = 1.
    # But expected output is 2x4.
    # What if the width corresponds to the SECOND smallest? 4 is the second smallest.
    # Or what about: the output is the cell with the smallest AREA?
    # 2x1=2, 2x8=16, 2x4=8, 10x1=10, 10x8=80, 10x4=40. Min area = 2 (2x1).
    # But expected is 2x4. Not min area.

    # Hmm, maybe the "smallest cell" means something else.
    # Looking at the grid structure: each cell has different sizes.
    # The output corresponds to one specific cell. Which one?

    # Train 1: heights=[3,5,1]. widths=[4,6]. Expected 3x2. But 2 is not in widths!
    # Wait, I might have wrong sep detection. Let me re-check.

# ===================== Task 09629e4f =====================
print("\n" + "=" * 50)
print("Task 09629e4f: Grid blocks - determine which color fills each block")

# Looking at the block data again:
# Each 3x3 block has 5 different colors from {2,3,4,6,8} and 4 zeros.
# Each block has cells at specific positions.
# The output color for each block seems to relate to which color is MISSING.
# Actually count=4 means one color missing, count=5 means all present.
#
# Wait, blocks in train 0:
# Block (0,0): colors {2,3,4,6} count=4, missing 8 -> output 2
# Block (0,1): colors {2,3,4,6,8} count=5 -> output 0
# Block (1,1): colors {2,3,4,6,8} count=5 -> output 4
#
# So count doesn't directly determine it. With all 5 colors present, output can be 0 or non-zero.
#
# Maybe the position of each color within the block matters.
# Like, which color is at position (0,0) of the block.
#
# Block (0,0) train 0: [[2,0,0],[0,4,3],[6,0,0]] -> (0,0)=2. Output=2. MATCH!
# Block (0,1) train 0: [[0,6,2],[4,0,8],[3,0,0]] -> (0,0)=0. Output=0. MATCH!
# Block (1,1) train 0: [[6,2,0],[0,0,4],[3,8,0]] -> (0,0)=6. Output=4. NO MATCH.
#
# Not (0,0). Let me try other positions.
# Block (0,0): output=2. Cell at (0,0)=2. YES.
# Block (1,1): output=4. Cell at? Let's check all positions:
#   (0,0)=6, (0,1)=2, (0,2)=0, (1,0)=0, (1,1)=0, (1,2)=4, (2,0)=3, (2,1)=8, (2,2)=0
#   4 is at position (1,2). Output=4.
#
# Maybe the rule is: find the color that appears in a specific position relative
# to a pattern. Or the color that appears only once and is special.
#
# Let me look at what's unique about the output color in each block.
# For block (0,0) train 0: [[2,0,0],[0,4,3],[6,0,0]]
# The non-zero cells: (0,0)=2, (1,1)=4, (1,2)=3, (2,0)=6
# Is 2 special? It's at position (0,0).
# For block (1,1) train 0: [[6,2,0],[0,0,4],[3,8,0]]
# Output=4, at position (1,2).
# For block (1,2) train 0: [[0,4,8],[6,0,0],[0,3,2]]
# Output=3, at position (2,1).
# For block (2,0) train 0: [[0,3,6],[2,0,0],[8,0,4]]
# Output=6, at position (0,2).

# Positions of output colors: (0,0), (1,2), (2,1), (0,2)
# These are at the corners or edges.
# The blocks with output 0 have 0 at what position?

# Block (0,1) output=0: [[0,6,2],[4,0,8],[3,0,0]]
# Block (0,2) output=0: [[0,0,4],[3,0,6],[8,0,2]]
# Block (1,0) output=0: [[3,8,0],[0,0,4],[6,2,0]]
# Block (2,1) output=0: [[0,2,0],[4,0,8],[6,3,0]]
# Block (2,2) output=0: [[0,6,0],[0,0,8],[2,3,4]]

# Hmm, I need another approach. Let me check if there's a Sudoku-like pattern.
# In a 3x3 grid of blocks, maybe each row of blocks and each column of blocks
# has each color exactly once (like a Sudoku with colors instead of numbers).

# Train 0 outputs:
# Row 0: 2, 0, 0
# Row 1: 0, 4, 3
# Row 2: 6, 0, 0
# So in block-row 0, only color 2 appears. Block-row 1 has 4 and 3. Block-row 2 has 6.
# In block-col 0: 2, 0, 6. Block-col 1: 0, 4, 0. Block-col 2: 0, 3, 0.
# Not Sudoku-like. Only some blocks get a color.

# What if: each block that gets a color has ALL 9 cells filled with that color?
# And the rest are 0?
# The pattern of non-zero output blocks is:
# Train 0: (0,0)=2, (1,1)=4, (1,2)=3, (2,0)=6
# Train 1: (0,2)=2, (1,1)=3, (2,0)=4, (2,2)=6
# Train 2: (0,1)=3, (1,1)=6, (1,2)=4, (2,0)=2
# Train 3: (0,1)=4, (1,1)=2, (1,2)=6, (2,1)=3

# These are always exactly 4 non-zero blocks. And they always use colors {2,3,4,6}.
# Color 8 is never an output. Interesting.

# So the question is: which 4 blocks (out of 9) get which of the 4 colors {2,3,4,6}?
# And color 8 is the "filler" that doesn't determine anything.

# Let me look at positions of each color across all blocks:
# In train 0:
# Color 2 appears in blocks: (0,0) at (0,0), (0,1) at (0,2), (0,2) at (2,2),
#   (1,0) at (2,1), (1,1) at (0,1), (1,2) at (2,2), (2,0) at (1,0), (2,1) at (0,1), (2,2) at (2,0)
# That's a lot. Color 2 in the original appears exactly once per block (since each block
# has 4-5 non-zero cells from 5 colors with each appearing once).
#
# Each color appears exactly once per block. And the output picks one of the 4 colors
# {2,3,4,6} for some blocks and 0 for others.
#
# What if: in each input block, all 5 colors (or 4 of 5) occupy specific cell positions.
# The block positions form a pattern, and the color at a specific position determines output.

# Let me look at this from a different angle. Maybe each block's color is determined
# by which cell position within the block is empty (has 0).
# Block (0,0) train 0: [[2,0,0],[0,4,3],[6,0,0]]
# Zero positions: (0,1),(0,2),(1,0),(2,1),(2,2). Non-zero: (0,0)=2, (1,1)=4, (1,2)=3, (2,0)=6
# Only 4 non-zero cells. The 5th color (8) is missing.
# Output = 2 (color at position 0,0).

# Block (1,1) train 0: [[6,2,0],[0,0,4],[3,8,0]]
# Non-zero: (0,0)=6, (0,1)=2, (1,2)=4, (2,0)=3, (2,1)=8. 5 non-zero cells.
# Output = 4 (color at position 1,2).

# What determines which cell's color becomes the output?
# Maybe it's the cell position that's unique/special in some way across all blocks.
# Or maybe the color value at a specific (row, col) within the block.

# Wait, let me check if the output equals the value at position (br, bc) within the block,
# where (br, bc) is the block's position in the 3x3 grid of blocks.
# Block (0,0): value at cell (0,0) = 2. Output = 2. YES!
# Block (0,1): value at cell (0,1) = 6. Output = 0. NO!
# Block (1,1): value at cell (1,1) = 0. Output = 4. NO!

# What about (bc, br)?
# Block (0,0): cell (0,0) = 2. Output = 2. YES.
# Block (1,1): cell (1,1) = 0. Output = 4. NO.

# Not that simple. Let me think about what determines which cell position matters.

# Actually, maybe we should look at the position of each color within each block
# and find a pattern that's consistent across blocks.
# Color 2 in block (0,0) is at position (0,0).
# Color 4 in block (1,1) is at position (1,2).
# Color 3 in block (1,2) is at position (2,1).
# Color 6 in block (2,0) is at position (0,2).

# Is there a relationship between the block position and the cell position of the output color?
# Block (0,0) -> cell (0,0): same
# Block (1,1) -> cell (1,2): different
# Block (1,2) -> cell (2,1): different
# Block (2,0) -> cell (0,2): different

# Hmm. Let me try looking at each color's position across ALL blocks.
# Color 2 positions across all blocks in train 0:
task09 = load_task('09629e4f')
pair = task09['train'][0]
inp = pair['input']
out = pair['output']

print("\n  Detailed analysis train 0:")
for color in [2, 3, 4, 6, 8]:
    print(f"  Color {color} positions (block_r, block_c, cell_r, cell_c):")
    for br in range(3):
        for bc in range(3):
            rs = br * 4
            cs = bc * 4
            for r in range(3):
                for c in range(3):
                    if inp[rs+r][cs+c] == color:
                        print(f"    block({br},{bc}) cell({r},{c})")

# Maybe the rule is: for each color, if it appears at the SAME cell position
# across all blocks, that's its "home" position. The block where the color
# is at its home position gets that color as output.

# Or: think of it as a 9x9 grid. Each color appears 9 times (once per block).
# The positions form some pattern.

# Actually wait - in the original 11x11 grid, positions of color 2:
# Let me check
for color in [2, 3, 4, 6, 8]:
    positions = []
    for r in range(11):
        for c in range(11):
            if inp[r][c] == color:
                positions.append((r, c))
    print(f"  Color {color}: {positions}")

print("\n  Output blocks:")
for br in range(3):
    row = []
    for bc in range(3):
        row.append(out[br*4][bc*4])
    print(f"    {row}")

# Let me check: for each non-zero output block, what is the position of the output color
# within that block, and is there a pattern?
print("\n  Output color positions within blocks:")
for br in range(3):
    for bc in range(3):
        rs = br * 4
        cs = bc * 4
        out_color = out[rs][cs]
        if out_color == 0:
            continue
        for r in range(3):
            for c in range(3):
                if inp[rs+r][cs+c] == out_color:
                    print(f"    Block ({br},{bc}) output={out_color} at cell ({r},{c})")

# Checking all training pairs
for ti in range(4):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']
    print(f"\n  Train {ti} output color positions:")
    for br in range(3):
        for bc in range(3):
            rs = br * 4
            cs = bc * 4
            out_color = out[rs][cs]
            if out_color == 0:
                continue
            for r in range(3):
                for c in range(3):
                    if inp[rs+r][cs+c] == out_color:
                        print(f"    Block ({br},{bc}) output={out_color} at cell ({r},{c}), block_diag={br==bc}")
