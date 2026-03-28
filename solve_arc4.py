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
            all_pass = False
    return all_pass

solutions = {}

# ===================== Task 09629e4f =====================
print("=" * 50)
print("Task 09629e4f")

# Let me look at this differently. The 4 output colors are always from {2,3,4,6}.
# Color 8 is a "noise" color. The output blocks have exactly 4 non-zero values.
#
# Maybe the rule has to do with the relative ordering of colors within each block.
# Or maybe it's about which cell position is "correct" for each color.
#
# Let me try: for each block, remove 8s and 0s. Then check if the remaining 4 colors
# form some specific spatial arrangement.
#
# Another idea: look at the full 3x3 grid layout. Each cell in the original grid
# can be mapped to (block_row, block_col, cell_row, cell_col).
# Maybe each color has a "target" cell position, and the block where the color
# is at its target position gets that color in the output.
#
# For train 0:
# Color 2 output block = (0,0), cell position in that block = (0,0)
# Color 4 output block = (1,1), cell position = (1,2)
# Color 3 output block = (1,2), cell position = (2,1)
# Color 6 output block = (2,0), cell position = (0,2)
#
# These cell positions are: (0,0), (1,2), (2,1), (0,2)
# Do these correspond to anything about the block positions?
# Block (0,0) cell (0,0): both are (0,0). Hmm.
# Block (1,1) cell (1,2): different.
# Block (1,2) cell (2,1): swapped with flip.
# Block (2,0) cell (0,2): r and c swapped.
#
# What if the "target" cell = (block_col, block_row)? (transposed block position)
# Block (0,0) -> target (0,0). Color 2 at cell (0,0). YES!
# Block (1,1) -> target (1,1). Color 4 at cell (1,2). NO!
# Block (1,2) -> target (2,1). Color 3 at cell (2,1). YES!
# Block (2,0) -> target (0,2). Color 6 at cell (0,2). YES!
# 3 out of 4 match. Close but not quite.
#
# For block (1,1): target would be (1,1). Which color is at (1,1)? It's 0 (empty).
# So this theory fails for (1,1).
#
# What if: for non-diagonal blocks, target = (bc, br). For diagonal blocks, something else?
# Block (0,0) is diagonal: color 2 at (0,0). Target = (0,0). YES.
# Block (1,1) is diagonal: color 4 at (1,2). Target = ???
# In train 1, block (1,1) output=3 at cell (1,1).
# In train 2, block (1,1) output=6 at cell (1,2).
# Not consistent for diagonal blocks either.

# Let me try yet another approach: maybe each block represents a mapping from
# position to color, and we need to find which mapping is consistent.

# Actually, let me look at the problem from a completely different angle.
# Maybe the blocks that get a non-zero output are those where
# a specific COLOR appears at a SPECIFIC POSITION, and the non-zero blocks
# form a pattern.

# Output pattern across all training examples:
# Train 0: non-zero at (0,0), (1,1), (1,2), (2,0) -> anti-diagonal + off
# Train 1: non-zero at (0,2), (1,1), (2,0), (2,2) -> main diagonal corners + center
# Train 2: non-zero at (0,1), (1,1), (1,2), (2,0)
# Train 3: non-zero at (0,1), (1,1), (1,2), (2,1)
# Block (1,1) always non-zero.

# Maybe the zero blocks all have the same count of colors, or some other property.
# Let me count non-zero non-5 cells per block (excluding 8):
task09 = load_task('09629e4f')

for ti in range(4):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']
    print(f"\n  Train {ti}:")
    for br in range(3):
        for bc in range(3):
            rs, cs = br*4, bc*4
            block = [[inp[rs+r][cs+c] for c in range(3)] for r in range(3)]
            # Count non-zero non-5 cells (also non-8?)
            count_non8 = sum(1 for r in range(3) for c in range(3) if block[r][c] not in [0, 5, 8])
            count_all = sum(1 for r in range(3) for c in range(3) if block[r][c] not in [0, 5])
            has_8 = any(block[r][c] == 8 for r in range(3) for c in range(3))
            out_val = out[rs][cs]
            # What is the value that appears twice or in a special position?
            vals = {}
            for r in range(3):
                for c in range(3):
                    v = block[r][c]
                    if v not in [0, 5]:
                        vals[v] = vals.get(v, 0) + 1
            print(f"    Block ({br},{bc}): has_8={has_8}, count={count_all}, count_no8={count_non8}, vals={vals}, out={out_val}")

# Looking at this: blocks with output 0 have 5 colors (including 8) in them.
# Blocks with output non-zero might have only 4 colors (missing 8)?
# Let's check: Block (0,0) train 0: has_8=False! Output=2.
# Block (0,1) train 0: has_8=True. Output=0.
# Block (1,1) train 0: has_8=True. Output=4.
# Hmm, (1,1) has 8 but still gets output 4. So it's not about having/missing 8.

# Actually wait - block (0,0) has no 8 -> output non-zero.
# But block (1,1) has 8 -> also non-zero. So having/not having 8 is not the determinant.

# Let me look at the actual number of non-zero cells:
# Block (0,0): 4 non-zero cells (no 8), output=2
# Block (0,1): 5 non-zero cells (has 8), output=0
# Block (1,1): 5 non-zero cells (has 8), output=4

# No clear pattern from counts.

# New idea: maybe the answer is about WHICH color each block shares with its neighbors.
# Or: the 3x3 block grid acts like a Latin square.

# Let me look at the NON-ZERO output values and their positions more carefully.
# In all training pairs, exactly 4 blocks are non-zero. The values are always {2,3,4,6}.
# The 5 zero blocks form a pattern.

# Let me check if there's a consistent mapping: for each cell position (r,c) in a block,
# there's a specific color that "belongs" there. If a block has the right color at
# the right position for the block's location, it gets that color.

# I think the actual key might be something about pairs of colors or about matching
# colors across blocks in the same row/column.

# Let me try: for each pair of adjacent blocks (same row or same column),
# which color appears in the "shared edge" position?

# Actually, let me try something much simpler. Maybe the output for each block
# is the color that appears at the same position in ALL blocks of that row or column.
# Or: the color that appears at position (r,c) in block (R,C) where R,C,r,c satisfy
# some relationship.

# I'm going to step back and try a brute-force approach:
# For each block, try all possible functions of the block contents to predict the output.

# After much analysis, let me try: the output color is the one that does NOT appear
# in a specific paired block. Or maybe: each block's output color is determined by
# some cross-referencing with other blocks.

# Let me try something completely different: maybe the colors form a multiplication
# table or group operation table. The block at (br, bc) contains the "product" of
# the row-header and column-header.

# Actually, I just realized: maybe each row of blocks has certain colors, and
# each column has certain colors, and the output is determined by which color
# is unique to that block's row-column intersection.

# Let me check: what colors appear in each block-row and block-column?
for ti in range(1):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']
    print(f"\n  Train {ti} block-row colors:")
    for br in range(3):
        all_colors = set()
        for bc in range(3):
            rs, cs = br*4, bc*4
            for r in range(3):
                for c in range(3):
                    v = inp[rs+r][cs+c]
                    if v not in [0, 5]:
                        all_colors.add(v)
        print(f"    Row {br}: {sorted(all_colors)}")
    print(f"  Block-col colors:")
    for bc in range(3):
        all_colors = set()
        for br in range(3):
            rs, cs = br*4, bc*4
            for r in range(3):
                for c in range(3):
                    v = inp[rs+r][cs+c]
                    if v not in [0, 5]:
                        all_colors.add(v)
        print(f"    Col {bc}: {sorted(all_colors)}")

# Each row and column has all 5 colors. Not helpful.

# Let me try yet another approach: maybe there's something about the POSITION
# of the 0 cells (empty cells) within each block.
for ti in range(1):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']
    print(f"\n  Train {ti} zero positions within blocks:")
    for br in range(3):
        for bc in range(3):
            rs, cs = br*4, bc*4
            zeros = []
            for r in range(3):
                for c in range(3):
                    if inp[rs+r][cs+c] == 0:
                        zeros.append((r, c))
            out_val = out[rs][cs]
            print(f"    Block ({br},{bc}): zeros={zeros}, out={out_val}")

# Hmm maybe the zero positions form a shape, and the shape determines the output.
# Or: the non-zero positions form a specific pattern.

# Let me try the simplest possible thing: for each block, read the colors in
# reading order (left to right, top to bottom), skip 0s.
# Then check if the first/last/middle color is the output.
for ti in range(4):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']
    print(f"\n  Train {ti} first non-zero vs output:")
    for br in range(3):
        for bc in range(3):
            rs, cs = br*4, bc*4
            vals = []
            for r in range(3):
                for c in range(3):
                    v = inp[rs+r][cs+c]
                    if v not in [0, 5]:
                        vals.append(v)
            out_val = out[rs][cs]
            if out_val != 0:
                print(f"    Block ({br},{bc}): vals={vals}, out={out_val}, first={vals[0] if vals else None}")

# OK I need to think about this differently. Let me look at what values go INTO
# the output and what pattern they form.
#
# The 3x3 output grid always has exactly 4 non-zero cells (from {2,3,4,6})
# and 5 zero cells. Block (1,1) is always non-zero.
#
# What if the 4 non-zero blocks are determined by which blocks don't have
# a certain property, like having all 5 colors present?
for ti in range(4):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']
    print(f"\n  Train {ti}: blocks with <5 colors vs output:")
    for br in range(3):
        for bc in range(3):
            rs, cs = br*4, bc*4
            vals = set()
            for r in range(3):
                for c in range(3):
                    v = inp[rs+r][cs+c]
                    if v not in [0, 5]:
                        vals.add(v)
            out_val = out[rs][cs]
            missing = {2,3,4,6,8} - vals
            print(f"    Block ({br},{bc}): colors={sorted(vals)}, missing={sorted(missing)}, out={out_val}")
