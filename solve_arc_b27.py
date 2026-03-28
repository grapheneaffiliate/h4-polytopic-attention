import json

PYTHON = r"C:\Users\atchi\AppData\Local\Programs\Python\Python311\python.exe"

solutions = {}

# 469497ad: 5x5 grid -> scaled output. Each cell in 5x5 grid is scaled by a factor.
# The 4x4 top-left contains a shape (with 0s and a colored block), last col has border colors, last row has border colors.
# The corner (4,4) has a color. The output scales each element by a factor.
# Looking more carefully: the grid has a 4x4 region (rows 0-3, cols 0-3) with 0s and a colored rectangle,
# then column 4 has border colors and row 4 has border colors, and cell (4,4) is a corner color.
# The output is the entire 5x5 scaled up where each original cell becomes an NxN block.
# Scale factor seems to be related to the size of the colored rectangle inside.
# Train 0: 2x2 block of 8s at (1,1)-(2,2). Scale=2. Output is 10x10 (5*2).
# The 0-region (4x4) becomes 8x8, with the 8-block becoming 4x4, diagonal of 2s in the 0 area.
# Train 1: 2x2 block of 4s at (1,0)-(2,1). Scale=3. Output is 15x15 (5*3).
# Train 2: 2x2 block of 1s at (1,1)-(2,2). Scale=4. Output is 20x20 (5*4).
# Actually let me reconsider the scale factor. Train 0: output 10x10 = 5*2. Train 1: 15x15 = 5*3. Train 2: 20x20 = 5*4.
# What determines the scale?
# Train 0: corner is 3, scale 2. Train 1: corner is 6, scale 3. Train 2: corner is 4, scale 4.
# Hmm, that doesn't directly map. Let me look at the border colors.
# Train 0: col4 = [3,3,3,3,3], row4 = [3,3,3,3,3]. All same color 3. Scale = 2.
# Train 1: col4 = [7,7,6,6,6], row4 = [7,7,6,6,6]. Two colors. Scale = 3.
# Train 2: col4 = [9,9,3,3,4], row4 = [9,9,3,3,4]. Three colors. Scale = 4.
# Number of distinct border colors = scale - 1? No: 1 -> 2, 2 -> 3, 3 -> 4. So scale = num_distinct_border_colors + 1.
# Actually the border row/col specify the output structure. Each border cell indicates the color of a block in the output border.
# Let me think differently. The scale factor for the inner 4x4 is: how many rows of each border color?
# Train 0: border is all 3. The 3-colored section is 2 rows/cols. So scale = 2.
# Train 1: border is [7,7,6,6,6]. 7 appears 2 times, 6 appears 3 times. Total = 5. Scale = 3 (each cell in 4x4 becomes 3x3).
# Hmm, output is 15x15. Inner 4x4 -> 4*3 = 12. Border 1 cell -> 1*3 = 3. Total = 15. Yes scale=3.
# The border pattern: col4 = [7,7,6,6,6]. In the output, the rightmost columns are:
# rows 0-5: 7, rows 6-8: 6, rows 9-11: 6, rows 12-14: 6.
# Wait, looking at output train 1 col 12-14 for all rows:
# rows 0-3: 7,7,7 ; rows 4-5: 7,7,7; rows 6-8: 6,6,6; rows 9-11: 6,6,6; rows 12-14: 6,6,6.
# Actually col 12 for rows: [7,7,7,7,7,7,6,6,6,6,6,6,6,6,6] - that's rows 0-5 = 7 (6 rows), rows 6-14 = 6 (9 rows).
# Hmm, scale = 3: each of the 5 input cells in the border becomes 3 output cells.
# input col4: [7,7,6,6,6] -> scaled: 7*3, 7*3, 6*3, 6*3, 6*3 = 6 7s then 9 6s. That matches!

# So the rule: scale factor = output_size / input_size. But we don't know output_size from input alone.
# The inner colored rectangle determines scale? Train 0: 2x2 rect, scale 2. Train 1: 2x2 rect, scale 3. Train 2: 2x2 rect, scale 4.
# All have 2x2 rects... so that's not it.
# Let me count unique colors: Train 0 border has 1 unique color (3), scale 2. Train 1 has 2 (7,6), scale 3. Train 2 has 3 (9,3,4), scale 4.
# Scale = number of distinct border colors + 1. Yes!
#
# For the diagonal of 2s: In train 0, the 0-areas (not the colored block, not the border) get a diagonal of color 2.
# The diagonal goes from corner to the colored block in each quadrant.
# Actually looking more carefully at train 2 output:
# The top-left 16x16 (the scaled 4x4 region) has:
# - A large 1-block (8x8) in the center
# - 0s around it
# - A diagonal of 2s from corners to the 1-block
#
# Let me reconsider the structure. The 5x5 input has:
# - A 4x4 "main" area (top-left)
# - Border info in last row and last column
# - Corner value
#
# The main area has a colored rectangle and 0s. In the output:
# - Each cell is scaled by factor N
# - The 0-cells in the main area get a diagonal pattern of color 2
# - The diagonal goes from the outer corner towards the colored block
#
# This is getting complex. Let me code it step by step.

solutions["469497ad"] = r"""def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    # Grid is 5x5
    # Find the colored block in the 4x4 region
    main = [row[:4] for row in grid[:4]]
    border_col = [grid[r][4] for r in range(4)]
    border_row = grid[4][:4]
    corner = grid[4][4]

    # Count distinct border colors
    border_colors = set(border_col + border_row + [corner])
    scale = len(border_colors) + 1

    # Build scaled output
    out_rows = 5 * scale
    out_cols = 5 * scale
    output = [[0]*out_cols for _ in range(out_rows)]

    # Fill border regions (last row block and last col block)
    for r in range(4):
        for dr in range(scale):
            for c_out in range(4*scale, 5*scale):
                output[r*scale + dr][c_out] = border_col[r]
    for c in range(4):
        for dc in range(scale):
            for r_out in range(4*scale, 5*scale):
                output[r_out][c*scale + dc] = border_row[c]
    # Corner
    for r_out in range(4*scale, 5*scale):
        for c_out in range(4*scale, 5*scale):
            output[r_out][c_out] = corner

    # Fill main 4x4 scaled region
    for r in range(4):
        for c in range(4):
            val = main[r][c]
            if val != 0:
                for dr in range(scale):
                    for dc in range(scale):
                        output[r*scale + dr][c*scale + dc] = val

    # Now add diagonal of 2s in the 0-regions of the main area
    # The diagonal connects corners to the colored block
    # Find the colored block bounds
    colored_cells = [(r,c) for r in range(4) for c in range(4) if main[r][c] != 0]
    min_r = min(r for r,c in colored_cells)
    max_r = max(r for r,c in colored_cells)
    min_c = min(c for r,c in colored_cells)
    max_c = max(c for r,c in colored_cells)

    # The diagonal of 2s goes in the 0-regions
    # Top-left quadrant: diagonal from (0,0) to (min_r*scale, min_c*scale)
    # We need to figure out the diagonal pattern
    # Looking at train 0 (scale=2): diagonal at (0,0),(1,1) and (0,7),(1,6) etc.
    # In train 0: block is at rows 1-2, cols 1-2 (in 4x4). Scale=2.
    # Scaled block: rows 2-5, cols 2-5.
    # Top-left corner: 0-area is rows 0-1, cols 0-1. Diagonal: (0,0)=2, (1,1)=2
    # Top-right corner: 0-area is rows 0-1, cols 6-7. Diagonal: (0,7)=2, (1,6)=2
    # Bottom-left: rows 6-7, cols 0-1. Diagonal: (6,1)=2, (7,0)=2
    # Bottom-right: rows 6-7, cols 6-7. Diagonal: (6,6)=2, (7,7)=2

    # For each 0-cell in the 4x4 grid, we place a diagonal of 2 within its scaled block
    # The direction of the diagonal depends on position relative to the colored block
    for r in range(4):
        for c in range(4):
            if main[r][c] == 0:
                # Determine diagonal direction
                # If above-left of block: top-left to bottom-right diagonal
                # If above-right: top-right to bottom-left
                # If below-left: bottom-left to top-right
                # If below-right: bottom-right to top-left
                # The diagonal goes from the outer corner of this cell's scaled block
                # toward the colored block

                # Determine relative position
                above = r < min_r
                below = r > max_r
                left = c < min_c
                right = c > max_c

                if above and left:
                    # diagonal from top-left to bottom-right
                    for d in range(scale):
                        output[r*scale + d][c*scale + d] = 2
                elif above and right:
                    # diagonal from top-right to bottom-left
                    for d in range(scale):
                        output[r*scale + d][c*scale + (scale-1-d)] = 2
                elif below and left:
                    # diagonal from bottom-left to top-right
                    for d in range(scale):
                        output[r*scale + (scale-1-d)][c*scale + d] = 2
                elif below and right:
                    # diagonal from bottom-right to top-left
                    for d in range(scale):
                        output[r*scale + d][c*scale + d] = 2
                elif above and not left and not right:
                    # directly above - no diagonal? Let's check
                    pass
                elif below and not left and not right:
                    pass
                elif left and not above and not below:
                    pass
                elif right and not above and not below:
                    pass

    return output
"""

# Let me verify 469497ad more carefully by looking at train 1
# Train 1: input col4=[7,7,6,6,6], row4=[7,7,6,6,6], corner=6
# border_colors = {7,6} -> scale = 3
# Main 4x4: [[0,0,0,0],[4,4,0,0],[4,4,0,0],[0,0,0,0]]
# Colored block at rows 1-2, cols 0-1
# Output 15x15.
# For cell (0,0): above=True, left=False (min_c=0, c=0 so not left). Hmm.
# Actually cell (0,0) is above the block (r=0 < min_r=1) and at same col (c=0 = min_c=0)
# In output, rows 0-2, cols 0-2 should have diagonal from (0,2) to (2,0)? Let me check.
# Output row 0: [0,0,0,0,0,0,0,0,2,0,0,0,7,7,7]
# So (0,8)=2. Cell (0,2) in 4x4, scaled: cols 6-8. But (0,8)=2 is at col 8 = cell(0,2)*3+2.
# Cell (0,0) in output rows 0-2 cols 0-2: all 0s.
# Cell (0,1) rows 0-2 cols 3-5: all 0s.
# Cell (0,2) rows 0-2 cols 6-8: (0,8)=2, (1,7)=2, (2,6)=2 - that's a top-right-to-bottom-left diagonal
# Cell (0,2) is above block (r=0 < 1) and right of block (c=2 > 1). So above and right -> diagonal from top-right to bottom-left. Correct!
# Cell (0,3): (0,11)=0, (1,11)=0, (2,11)=0. All 0s. Cell (0,3) is above and right - should have diagonal?
# Output: row 0 cols 9-11: [0,0,0]. No diagonal. Hmm.
# Wait, output row 0: [0,0,0,0,0,0,0,0,2,0,0,0,7,7,7]
# Cell (0,3) scaled is cols 9-11. All 0. But (0,3) is above (r=0<1) and right (c=3>1).
# Let me check cell(3,0): below (r=3>2) and left? No, min_c=0, so c=0 is not left.
# In output rows 9-11 cols 0-2:
# Row 9: [0,0,0,0,0,0,2,...] -> (9,6)=2. That's cell(3,2).
# Row 10: (10,7)=2. Cell(3,2) rows 9-11 cols 6-8: (9,6)=2,(10,7)=2,(11,8)=2.
# Cell (3,2) is below and right -> diagonal top-left to bottom-right. Yes!
# But cell(3,0) rows 9-11, cols 0-2: all 0. Cell(3,0) is below (r=3>2), same col as block. No diagonal.
# Cell(3,3) rows 9-11, cols 9-11: all 0. Also no diagonal.
# So the diagonal only appears when the cell is STRICTLY in a corner relative to the block.
# Cell(0,0) = above, same col range. No diagonal. Cell(0,2) = above and right (c > max_c). Has diagonal.
# Cell(0,3) = above and right. Should have diagonal but doesn't!
# Hmm let me re-examine. Output row 0: [0,0,0,0,0,0,0,0,2,0,0,0,7,7,7]
# col 0-2: cell(0,0), col 3-5: cell(0,1), col 6-8: cell(0,2), col 9-11: cell(0,3)
# Cell(0,2) has diagonal. Cell(0,3) doesn't. Why?
# Cell(0,2) is diagonally adjacent to the colored block (block at r1-2,c0-1; cell(0,2) touches at (1,1) diagonal).
# Maybe only cells that are diagonally adjacent to the colored block get diagonals.

# Let me reconsider: the diagonal is drawn from the corner of the 4x4 inner area toward the block.
# It goes through all 0-cells that are on that diagonal path.
# Train 0: block at (1,1)-(2,2). Diagonal from (0,0): cells (0,0). Diagonal from (0,3): cells (0,3).
# Diagonal from (3,0): cells (3,0). Diagonal from (3,3): cells (3,3).
# Each corner has exactly one cell to the block.
# Train 1: block at (1,0)-(2,1).
# Top-right of block: cells on diagonal from corner (0,3) to block...
# (0,3), (0,2) are above and to the right. The diagonal from (0,3) goes (0,3)->(1,2) but (1,2) is 0, not block adjacent...
# Actually just the cell diagonally adjacent: (0,2) is diag adjacent to block corner (1,1). So only (0,2) gets diagonal.
# And (3,2) is diag adjacent to (2,1). So only (3,2) gets diagonal.
# What about other cells? They stay 0.

# So the rule is: only cells that are DIAGONALLY ADJACENT to the colored block get a diagonal of 2s.
# Let me verify with train 2. Block at (1,1)-(2,2) same as train 0 but scale=4.
# Diag adjacent cells: (0,0), (0,3), (3,0), (3,3).
# Output for train 2: row 0: [2,0,0,0,0,...,0,0,0,0,0,0,0,0,0,2,9,9,9,9]
# Cell(0,0): (0,0)=2, (1,1)=2, (2,2)=2, (3,3)=2 -> top-left to bottom-right diagonal in 4x4 scaled block. Yes!
# Cell(0,3): cols 12-15. (0,15)=2,(1,14)=2,(2,13)=2,(3,12)=2 -> top-right to bottom-left. Yes!
# Cell(3,0): rows 12-15, cols 0-3. (12,3)=2,(13,2)=2,(14,1)=2,(15,0)=2 -> bottom-right to top-left?
# Actually (12,3)=2: row 12 col 3. That's top-right of cell(3,0)'s block. Going to (15,0) bottom-left.
# Cell(3,0) is below-left of block -> diagonal from bottom-left to top-right? (15,0)->(12,3)? That's the same.
# Let me check: row 12 = [0,0,0,2,...], row 13 = [0,0,2,...], row 14 = [0,2,...], row 15 = [2,...].
# (12,3)=2, (13,2)=2, (14,1)=2, (15,0)=2. Yes, diagonal from top-right to bottom-left within the block.
# For cell below-left: the diagonal points from the inner corner (closest to block) to the outer corner.
# Actually the direction is: the diagonal corner closest to the block has the 2.
# For (0,0) top-left of block: bottom-right of cell block is closest to main block -> diagonal goes from top-left to bottom-right. (0,0) to (3,3).
# For (0,3) top-right: bottom-left closest -> top-right to bottom-left. (0,15) to (3,12).
# For (3,0) bottom-left: top-right closest -> bottom-left to top-right. Same as top-right to bottom-left direction.
# For (3,3) bottom-right: top-left closest -> diagonal from top-left to bottom-right. (12,12) to (15,15).
# Output: row 12 col 12-15: check... row 12: [...,0,0,0,0,2,0,0,0,3,3,3,3], col 12 for row 12 = 2?
# Let me re-read train 2 output row 12: [0,0,0,2,0,0,0,0,0,0,0,0,2,0,0,0,3,3,3,3]
# (12,12)=2. (13,13)=2? Row 13: [0,0,2,0,0,0,0,0,0,0,0,0,0,2,0,0,3,3,3,3]. (13,13)=2. Yes!
# (14,14)=2? Row 14: [0,2,0,0,0,0,0,0,0,0,0,0,0,0,2,0,3,3,3,3]. (14,14)=2. Yes!
# (15,15)=2? Row 15: [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,3,3,3,3]. (15,15)=2. Yes!
# So cell(3,3) has diagonal from top-left to bottom-right. Correct!

# Great! So the rule is clear. Let me also handle train 1 properly.
# Train 1 block at rows 1-2, cols 0-1. Diagonal adjacent cells:
# (0,2) - above-right of block corner (1,1)
# (3,2) - below-right of block corner (2,1)
# No top-left diagonal cell because block starts at col 0
# No bottom-left diagonal cell because block starts at col 0

# Now rewrite the solution properly.
solutions["469497ad"] = r"""def solve(grid):
    main = [row[:4] for row in grid[:4]]
    border_col = [grid[r][4] for r in range(4)]
    border_row = grid[4][:4]
    corner = grid[4][4]

    border_colors = set(border_col + border_row + [corner])
    scale = len(border_colors) + 1

    out_size = 5 * scale
    output = [[0]*out_size for _ in range(out_size)]

    # Fill border column (col 4 scaled)
    for r in range(4):
        for dr in range(scale):
            for dc in range(scale):
                output[r*scale + dr][4*scale + dc] = border_col[r]
    # Fill border row (row 4 scaled)
    for c in range(4):
        for dr in range(scale):
            for dc in range(scale):
                output[4*scale + dr][c*scale + dc] = border_row[c]
    # Corner
    for dr in range(scale):
        for dc in range(scale):
            output[4*scale + dr][4*scale + dc] = corner

    # Fill main colored cells
    colored_cells = set()
    for r in range(4):
        for c in range(4):
            if main[r][c] != 0:
                colored_cells.add((r,c))
                for dr in range(scale):
                    for dc in range(scale):
                        output[r*scale + dr][c*scale + dc] = main[r][c]

    # Find block bounds
    min_r = min(r for r,c in colored_cells)
    max_r = max(r for r,c in colored_cells)
    min_c = min(c for r,c in colored_cells)
    max_c = max(c for r,c in colored_cells)

    # Add diagonal of 2s in cells diagonally adjacent to the block
    for r in range(4):
        for c in range(4):
            if main[r][c] != 0:
                continue
            # Check if diagonally adjacent to block
            is_diag = False
            if r == min_r - 1 and c == min_c - 1:
                # top-left: diagonal from (0,0) to (scale-1,scale-1)
                for d in range(scale):
                    output[r*scale+d][c*scale+d] = 2
            elif r == min_r - 1 and c == max_c + 1:
                # top-right: diagonal from (0,scale-1) to (scale-1,0)
                for d in range(scale):
                    output[r*scale+d][c*scale+(scale-1-d)] = 2
            elif r == max_r + 1 and c == min_c - 1:
                # bottom-left: diagonal from (scale-1,0) to (0,scale-1)
                for d in range(scale):
                    output[r*scale+(scale-1-d)][c*scale+d] = 2
            elif r == max_r + 1 and c == max_c + 1:
                # bottom-right: diagonal from (0,0) to (scale-1,scale-1)
                for d in range(scale):
                    output[r*scale+d][c*scale+d] = 2

    return output
"""


# 46f33fce: 10x10 grid with scattered non-zero values -> 20x20 output
# Each non-zero value at position (r,c) becomes a 4x4 block at position (r*2, c*2) in the output
# Actually looking more carefully: input is 10x10, output is 20x20.
# Each non-zero pixel at (r,c) in input -> 4x4 block of that color at (r*2, c*2)?
# Let me check: Train 0: value 2 at (1,1). Output: rows 0-3, cols 0-3 are all 2. That's (0,0) to (3,3).
# Actually (1*2, 1*2) = (2,2). But block is at (0,0)-(3,3). Hmm, that's centered around (2,2): (0,0) to (3,3).
# Range (1*2-1, 1*2-1) to (1*2+2, 1*2+2) = (1,1) to (4,4)? No, output has 2s at (0,0)-(3,3).
# Let me just look: positions of non-zero values in train 0 input:
# (1,1)=2, (3,1)=4, (3,3)=1, (5,5)=3, (7,7)=4, (9,7)=2, (9,9)=3
# Output blocks:
# 2 at (0,0)-(3,3), 4 at (4,0)-(7,3) and 1 at (4,4)-(7,7), 3 at (8,8)-(11,11),
# 4 at (12,12)-(15,15), 2 at (16,12)-(19,15) and (16,16)-(19,19)=3
# Hmm wait. Let me look at row 16-19:
# Row 16: [0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2]
# Row 17-19 same.
# So at (9,7)=2 -> block at cols 12-15 rows 16-19. And (9,9)=3 -> block at cols 16-19 rows 16-19.
# (9*2, 7*2) = (18, 14). Block from (16,12) to (19,15). That's (18-2, 14-2) to (18+1, 14+1).
# Hmm not clean. Let me try: each input pixel at (r,c) maps to a 4x4 block with top-left at (r*2, c*2).
# (1,1) -> top-left (2,2), block (2,2)-(5,5). But output has 2 at (0,0)-(3,3). Doesn't match.
# Try: block centered at (r*2, c*2) i.e. (r*2-1, c*2-1) to (r*2+2, c*2+2).
# (1,1) -> (1,1) to (4,4). But output is (0,0)-(3,3).
# Try just r*2-2: (1*2-2, 1*2-2) = (0,0) to (3,3). That works!
# (3,1)=4: (3*2-2, 1*2-2) = (4,0) to (7,3). Output rows 4-7, cols 0-3: yes, 4,4,4,4.
# (3,3)=1: (4,4) to (7,7). Output: 1,1,1,1. Yes!
# (5,5)=3: (8,8) to (11,11). Output: 3,3,3,3. Yes!
# (7,7)=4: (12,12) to (15,15). Output: 4,4,4,4. Yes!
# (9,7)=2: (16,12) to (19,15). Yes! (9,9)=3: (16,16) to (19,19). Yes!
# Formula: block top-left = (r*2-2, c*2-2), size 4x4.
# But wait, what about (9,7): 9*2-2=16, 7*2-2=12. Block (16,12)-(19,15). Check row 16: cols 12-19 = [2,2,2,2,2,2,2,2].
# Hmm, that's 8 values not 4. Because (9,7) and ... no, cols 12-15 = 2 and cols 16-19 = 3 (from (9,9)=3). Let me recheck.
# Row 16: [0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2] - indices 12-19 are all listed.
# Hmm 2,2,2,2,2,2,2,2. But (9,9)=3 should give cols 16-19 = 3.
# Wait, train 2 output row 16: [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,4,4,4,4]
# That's train 2 (third example). For train 0:
# Row 16: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,3,3,3]
# (9,9)=3: block at (16,16)-(19,19). Row 16 cols 16-19 = 3,3,3,3. Yes.
# And (9,7)=2: block at (16,12)-(19,15). Row 16 cols 12-15 = 0,0,0,0? That doesn't match!
# Let me re-read. Train 0 output row 16: [0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2]
# Wait, that's from train 1 (second example), not train 0.
# Let me re-examine. I initially read the outputs from the file.
# Train 0 input row 9: [0,0,0,0,0,0,0,0,0,3]
# So (9,9)=3 only. No (9,7).
# Train 1 input row 9: [0,0,0,0,0,0,0,2,0,2]
# So (9,7)=2 and (9,9)=2. Block (16,12)-(19,15) and (16,16)-(19,19) all 2. That gives row 16 cols 12-19 = 2,2,2,2,2,2,2,2. Yes!

# OK so the formula is: for each non-zero pixel at (r,c), fill a 4x4 block from (r*2-2, c*2-2) to (r*2+1, c*2+1) in the 20x20 output.
# Hmm let me double-check with a different value. Train 1: (7,9)=8.
# Block: (12,16) to (15,19). Output row 12: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,8,8,8]. cols 16-19 = 8. Yes!

# Simpler way to think: each cell at odd positions (1,3,5,7,9) in input maps to output.
# Actually the non-zero values are always at odd row, odd col? Let me check:
# Train 0: (1,1), (3,1), (3,3), (5,5), (7,7), (9,9) - all odd. Yes.
# Train 1: (1,1), (1,3), (3,3), (7,9), (9,7), (9,9) - all odd. Yes.
# So mapping: input (r,c) where r,c are odd -> output block at ((r-1)*2, (c-1)*2) to ((r-1)*2+3, (c-1)*2+3).
# (1,1): (0,0)-(3,3). (3,3): (4,4)-(7,7). (5,5): (8,8)-(11,11). Yes!

solutions["46f33fce"] = r"""def solve(grid):
    H, W = len(grid), len(grid[0])
    out = [[0]*(W*2) for _ in range(H*2)]
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0:
                # Place 4x4 block at (r*2-2, c*2-2)
                tr, tc = r*2-2, c*2-2
                for dr in range(4):
                    for dc in range(4):
                        rr, cc = tr+dr, tc+dc
                        if 0 <= rr < H*2 and 0 <= cc < W*2:
                            out[rr][cc] = grid[r][c]
    return out
"""


# 47c1f68c: Grid divided by a cross (horizontal and vertical lines of a color).
# The shape is in one quadrant. Output removes the cross lines and creates 4-fold reflections.
# Train 0: 11x11 grid. Cross at row 5 (all 2) and col 5 (all 2). Shape in top-left quadrant (0-4, 0-4).
# Shape (color 1): (1,1), (2,0), (2,1), (3,1), (3,2).
# Output is 10x10 (cross lines removed).
# The shape gets reflected: original in top-left, horizontal mirror in top-right, vertical mirror in bottom-left, both in bottom-right.
# But the shape color changes to the cross color (2).
# Output: top-left has the shape in color 2. Top-right has it mirrored horizontally in color 2.
# Bottom has vertical mirrors.
# Let me verify: Train 0 output row 0: [0,0,0,0,0,0,0,0,0,0] - all 0.
# Row 1: [0,2,0,0,0,0,0,0,2,0]. Shape at (1,1)=2 and mirrored at (1,8)=2.
# Mirror across col 4.5: col 1 -> col 8 (1 from left -> 1 from right in 10 cols: 10-1-1=8). Yes!
# Row 2: [2,2,0,0,0,0,0,0,2,2]. (2,0)=2,(2,1)=2 and mirrored (2,9)=2,(2,8)=2. Yes!
# Bottom half mirrors vertically. Row 6 = mirror of row 3: [0,2,2,0,0,0,0,2,2,0]. Row 3: [0,2,2,0,0,0,0,2,2,0]. Yes!
# Row 7 = mirror of row 2: [2,2,0,0,0,0,0,0,2,2]. Yes!
# Row 8 = mirror of row 1: [0,2,0,0,0,0,0,0,2,0]. Yes!

# So the output size is (H-1)x(W-1) where the cross lines are removed.
# The shape is reflected in 4 quadrants, and the color becomes the cross color.

# Train 1: 9x9 grid. Cross at row 4 (all 8) and col 4 (all 8). Shape in top-left (0-3, 0-3) with color 3.
# Shape: (0,0)=3, (0,2)=3, (1,0)=3, (1,1)=3, (2,0)=3.
# Output 8x8 with color 8 for the shape.

# Train 2: 7x7 grid. Cross at row 3 (all 4) and col 3 (all 4). Shape in top-left (0-2, 0-2) with color 2.
# Shape: (0,0)=2, (1,1)=2, (1,2)=2, (2,1)=2.
# Output 6x6 with color 4.

solutions["47c1f68c"] = r"""def solve(grid):
    H, W = len(grid), len(grid[0])
    # Find cross lines
    cross_row = -1
    cross_col = -1
    cross_color = 0
    for r in range(H):
        if all(grid[r][c] != 0 for c in range(W)):
            cross_row = r
            cross_color = grid[r][0]
            break
    for c in range(W):
        if all(grid[r][c] != 0 for r in range(H)):
            cross_col = c
            break

    # Extract shape from the quadrant that has it (non-zero, non-cross-color)
    shape = []
    for r in range(H):
        if r == cross_row:
            continue
        for c in range(W):
            if c == cross_col:
                continue
            if grid[r][c] != 0 and grid[r][c] != cross_color:
                # Map to quadrant coordinates
                qr = r if r < cross_row else r - 1
                qc = c if c < cross_col else c - 1
                shape.append((qr, qc))

    out_H = H - 1
    out_W = W - 1
    output = [[0]*out_W for _ in range(out_H)]

    # Place shape in 4 quadrants with cross_color
    for qr, qc in shape:
        # Original (top-left assumed)
        output[qr][qc] = cross_color
        # Horizontal mirror
        output[qr][out_W - 1 - qc] = cross_color
        # Vertical mirror
        output[out_H - 1 - qr][qc] = cross_color
        # Both mirrors
        output[out_H - 1 - qr][out_W - 1 - qc] = cross_color

    return output
"""


# 484b58aa: Repeating pattern with 0-holes. Output fills in the 0s to complete the repeating pattern.
# The grid has a tiled pattern with some rectangular regions set to 0.
# The output is the complete tiled pattern without 0-holes.
# Need to detect the tile period and fill in.

solutions["484b58aa"] = r"""def solve(grid):
    import copy
    H, W = len(grid), len(grid[0])
    out = copy.deepcopy(grid)

    # Find the repeating period by trying all (pr, pc) combinations
    # Prioritize smaller periods
    def try_period(pr, pc):
        tile = [[None]*pc for _ in range(pr)]
        for r in range(H):
            for c in range(W):
                if grid[r][c] != 0:
                    tr, tc = r % pr, c % pc
                    if tile[tr][tc] is None:
                        tile[tr][tc] = grid[r][c]
                    elif tile[tr][tc] != grid[r][c]:
                        return None
        # Check all tile cells filled
        for r in range(pr):
            for c in range(pc):
                if tile[r][c] is None:
                    return None
        return tile

    # Try periods in order of total area (smallest first)
    best_tile = None
    best_pr = None
    best_pc = None
    candidates = []
    for pr in range(1, H+1):
        for pc in range(1, W+1):
            candidates.append((pr * pc, pr, pc))
    candidates.sort()

    for _, pr, pc in candidates:
        tile = try_period(pr, pc)
        if tile is not None:
            best_tile = tile
            best_pr = pr
            best_pc = pc
            break

    if best_tile is not None:
        for r in range(H):
            for c in range(W):
                out[r][c] = best_tile[r % best_pr][c % best_pc]

    return out
"""


# 48d8fb45: Grid with 5 marker, and shapes of a single color.
# The 5 marks which shape is the "key". Output is the overlap/intersection pattern.
# Looking at train 0: shapes of color 1. One has 5 nearby. There are 3 shapes.
# Output is 3x3. The shape near 5 is the "query" and the others are compared.
# Actually, let me look more carefully.
# Train 0: 10x10 grid. Color 1 shapes and one color 5.
# 5 is at (2,3). Nearby shape: (3,3)=1, (4,2)=1, (4,3)=1, (4,4)=1, (5,3)=1, (5,4)=1.
# That's a cross-like shape. Other shapes:
# Shape at (1,8)=1,(2,7)=1,(2,8)=1,(3,8)=1 - an L
# Shape at (7,6)=1,(7,7)=1,(8,5)=1,(8,6)=1,(8,7)=1,(9,6)=1,(9,7)=1.
# Output is [[0,1,0],[1,1,1],[0,1,1]].
# The shape near 5 has bounding box 3x3 (rows 3-5, cols 2-4):
# Row 3: [0,1,0]; Row 4: [1,1,1]; Row 5: [0,1,1]. That matches the output!
# So the output is the shape that's adjacent to the marker 5, extracted as its bounding box.
# But what about the other shapes? Maybe they're distractors?
# Actually wait, there might be a relationship. Let me check train 1.
# Train 1: 5 at (1,7). Nearby shape (color 4): need to check.
# (2,6)=4,(2,7)=4,(3,7)=4,(3,8)=4,(4,7)=4. That's NOT the shape near 5.
# Wait, (1,7)=5. Adjacent cells: (2,7)=4, (2,6)=4.
# Other shapes: (2,2)=4,(3,1)=4,(3,3)=4,(4,1)=4,(4,2)=4,(4,3)=4,(5,2)=4,(5,3)=4.
# Output: [[4,4,0],[0,0,4],[0,4,0]].
# Shape near 5 bounding box: rows 2-4, cols 6-8:
# Row 2: [4,4,0]; Row 3: [0,4,4]; Row 4: [0,4,0]. That gives [[4,4,0],[0,4,4],[0,4,0]].
# But output is [[4,4,0],[0,0,4],[0,4,0]]. Different at (1,1): output has 0, shape has 4.
# So it's not simply extracting the shape near 5.
#
# Hmm, so there's some kind of overlay/XOR between shapes.
# Let me reconsider. Maybe the shape near 5 is the one to subtract from, and the overlapping parts are removed.
# Or maybe it's: find the shape NOT adjacent to 5 that matches in form, and output the other non-matching one.
#
# Actually let me re-examine. In train 0:
# Shape A (near 5): rows 3-5, cols 2-4: [[0,1,0],[1,1,1],[0,1,1]]
# Shape B (top-right): rows 1-3, cols 7-8. BBox 3x2: [[0,1],[1,1],[0,1]]. Hmm, 3 rows 2 cols.
# Actually (1,8)=1, (2,7)=1, (2,8)=1, (3,8)=1. BBox rows 1-3, cols 7-8:
# Row 1: [0,1]; Row 2: [1,1]; Row 3: [0,1]. That's 3x2.
# Shape C (bottom): rows 7-9, cols 5-7:
# Row 7: [0,1,1]; Row 8: [1,1,1]; Row 9: [0,1,1]. That's 3x3.
# Output = [[0,1,0],[1,1,1],[0,1,1]] which is shape A. So the output IS shape A (near 5)!
# Let me recheck train 1.
# Shape near 5: 5 is at (1,7). Adjacent to (2,7) which has color 4.
# The shape containing (2,7): (2,6)=4, (2,7)=4, (3,7)=4, (3,8)=4, (4,7)=4.
# BBox rows 2-4, cols 6-8: [[4,4,0],[0,4,4],[0,4,0]].
# Output: [[4,4,0],[0,0,4],[0,4,0]]. Different!
#
# Shape B: (2,2)=4,(3,1)=4,(3,3)=4,(4,1)=4,(4,2)=4,(4,3)=4,(5,2)=4,(5,3)=4.
# BBox rows 2-5, cols 1-3: [[0,4,0],[4,0,4],[4,4,4],[0,4,4]]. 4 rows, 3 cols.
#
# Hmm that doesn't match either. Let me reconsider.
#
# Maybe there are exactly 2 other shapes that together form the "parts" and the marker shape shows which part.
# Or maybe: the 5 marks the position within a shape, and the shape pieces are combined differently.
#
# Let me look at train 2: 5 at (4,7).
# Shape near 5: what's at adjacent cells? (4,7)=5. (5,7)=2,(5,8)=2,(6,6)=2,(6,7)=2,(7,7)=2.
# BBox rows 5-7, cols 6-8: [[0,2,2],[2,2,0],[0,2,0]].
# Output: [[0,2,2],[2,2,0],[0,2,0]]. Matches!
#
# So train 0 and 2 the output IS the shape near 5. But train 1 is different?
# Let me recheck train 1 more carefully.
# Input:
# Row 0: all 0
# Row 1: [0,0,0,0,0,0,0,5,0,0] -> 5 at (1,7)
# Row 2: [0,0,0,0,0,0,4,4,0,0] -> 4 at (2,6),(2,7)
# Row 3: [0,0,4,0,0,0,0,0,4,0] -> 4 at (3,2),(3,8)
# Row 4: [0,4,0,4,0,0,0,4,0,0] -> 4 at (4,1),(4,3),(4,7)
# Row 5: [0,0,4,4,0,0,0,0,0,0] -> 4 at (5,2),(5,3)
# Rows 6-9: all 0.
#
# Shape near 5 at (1,7): Connected component containing (2,7):
# (2,7) -> neighbors (2,6)=4, (3,8)=4? (3,8) is not adjacent to (2,7).
# BFS from (2,7): (2,7), (2,6). Neighbors of (2,6): only (2,7) already visited.
# So shape A = {(2,6),(2,7)} only?
# That seems small. Maybe connectivity includes diagonals? Or maybe the 5 itself is part of the shape?
#
# Or perhaps: the shapes are NOT connected components. Instead, there are exactly 3 shapes identified by
# some other criterion, and the output is a specific combination.
#
# Actually, I think I may have miscounted. Let me look again:
# Shape 1 (near 5): cells connected to 5's neighbors:
# 5 at (1,7). Below: (2,7)=4. Then (2,6)=4. Are there more connected? (3,7)? Grid (3,7)=0. (3,6)? Grid row 3: [0,0,4,0,0,0,0,0,4,0]. (3,6)=0. So shape 1 = {(2,6),(2,7)}.
# Shape 2: {(3,2),(4,1),(4,3),(5,2),(5,3)}. Connected? (3,2)->(4,1)? Not adjacent. (3,2)->(4,3)? Not adjacent. (3,2)->(5,2)? Not adjacent directly. Hmm.
# Maybe (3,2) and (4,3) are separate parts? Let me check 4-connectivity:
# (3,2): neighbors (2,2)=0, (4,2)=0, (3,1)=0, (3,3)=0. Isolated!
# (4,1): neighbors (3,1)=0, (5,1)=0, (4,0)=0, (4,2)=0. Isolated!
# (4,3): neighbors (3,3)=0, (5,3)=4!, (4,2)=0, (4,4)=0. Connected to (5,3).
# (5,2): neighbors (4,2)=0, (6,2)=0, (5,1)=0, (5,3)=4!. Connected to (5,3).
# (5,3): neighbors (4,3)=4, (5,2)=4. So cluster: {(4,3),(5,2),(5,3)}.
# Remaining: {(3,2)}, {(4,1)}, {(3,8)}, {(4,7)}.
# Wait, (4,7)=4. And (3,8)=4.
# (3,8) neighbors: (2,8)=0, (4,8)=0, (3,7)=0, (3,9)=0. Isolated.
# (4,7) neighbors: (3,7)=0, (5,7)=0, (4,6)=0, (4,8)=0. Isolated.
#
# So we have many disconnected 4-cells. This doesn't seem right for my approach.
#
# OK maybe the rule is different. Let me look at it differently.
#
# Actually, I think the shapes overlap. There are 2 identical shapes at different positions, and the 5
# shows which part is the "answer". The output is the overlap (AND) of the two non-5 shapes,
# or it's the shape indicated by 5's position.
#
# Wait, actually I think the problem is: there are 3 shapes of the same size/bounding box.
# One has a 5 marker. The other two, when overlaid, produce the marked shape.
# OR: two of the shapes, when XORed, give the third.
#
# Train 0:
# Output = [[0,1,0],[1,1,1],[0,1,1]] (shape A near 5)
# Shape B (3x2): [[0,1],[1,1],[0,1]] - doesn't have same bbox.
# Shape C (3x3): [[0,1,1],[1,1,1],[0,1,1]]
# Hmm, B is 3x2 and C is 3x3. A is 3x3.
# A XOR C: [[0,0,1],[0,0,0],[0,0,0]] - that's just one cell. Doesn't match B.
# A AND C: [[0,1,0],[1,1,1],[0,1,1]] = A. That means A is subset of C.
# C minus A: [[0,0,1],[0,0,0],[0,0,0]] - just (0,2).
#
# I think maybe the shapes in the grid need to be analyzed as "fragments" of a single 3x3 pattern,
# and the one marked with 5 is the complete one that overlaps both others.
#
# Actually, re-reading the task more carefully: maybe the output is always the shape closest to the 5 marker,
# extracted by bounding box. And I made an error for train 1.
#
# Train 1 output: [[4,4,0],[0,0,4],[0,4,0]].
# The shape near 5: {(2,6),(2,7)} with BBox [[4,4]] - just 1x2. Too small.
# What if the "shape near 5" includes cells that are close but not 4-connected?
# 5 at (1,7). Looking at all 4-cells near it: (2,6),(2,7),(3,8),(4,7).
# These form a cluster if we consider proximity. BBox rows 2-4, cols 6-8:
# Row 2: [4,4,0]; Row 3: [0,0,4]; Row 4: [0,4,0].
# Output: [[4,4,0],[0,0,4],[0,4,0]]. MATCHES!
#
# But they're not connected. So the "shape" is all colored cells that are closest to the 5.
# Or the 5 is placed within/above a shape and all cells of that shape form the output.
#
# Hmm, perhaps the shapes are identified differently. Let me think of it as:
# There are multiple copies of the same shape at different orientations/positions.
# The 5 picks which one. And we need to extract it.
#
# Actually, I bet the trick is simpler than I thought. Let me look at it as:
# The 5 marker sits at a specific position relative to a shape. The shape cells that are
# close/associated with the 5 form the output.
#
# Train 1: If I consider {(2,6),(2,7),(3,8),(4,7)} as one "group" and
# {(3,2),(4,1),(4,3),(5,2),(5,3)} as another "group":
# Group 1 BBox: rows 2-4, cols 6-8, 3x3: [[4,4,0],[0,0,4],[0,4,0]]
# Group 2 BBox: rows 3-5, cols 1-3, 3x3: [[4,0,0],[4,0,4],[0,4,4]]
# Output matches group 1. Group 1 is near 5.
#
# So the "groups" are determined by proximity to the 5 marker.
# A simple approach: find all non-zero, non-5 cells. The 5 is a marker.
# Group them by connectivity - but they're not connected.
# Maybe spatial clustering? Or maybe by identifying which ones can form the same bounding box size?
#
# Actually, the simplest approach for all 3 examples:
# 1. Find the 5 marker position
# 2. Find the non-5 colored cell closest to the 5
# 3. All colored cells within a certain radius form one shape
#
# OR: The shapes ARE connected if you include the 5 cell as part of the shape.
# Train 1: 5 at (1,7). If we include it: {(1,7),(2,6),(2,7)} connected? (1,7)-(2,7) yes. (2,7)-(2,6) yes.
# Then (3,8): connected to anything? (2,8)=0, (3,7)=0. No. And (4,7): (3,7)=0, (5,7)=0. No.
# So even including 5, only {(1,7),(2,6),(2,7)} are connected. Still missing (3,8) and (4,7).
#
# Different approach: maybe it's about the bounding box around the 5 and using a fixed 3x3 window.
# 5 at (1,7). Look at 3x3 below-left: rows 2-4, cols 6-8. That captures all 4 cells: (2,6),(2,7),(3,8),(4,7).
# Hmm, that's the output. But how to determine the window offset?
#
# For train 0: 5 at (2,3). Output shape is rows 3-5, cols 2-4. That's below-left of 5 too (well, below).
# For train 2: 5 at (4,7). Output shape is rows 5-7, cols 6-8. Below-left.
# So the pattern is: 5 marks the position ABOVE the shape. The shape is the 3x3 (or NxN) region
# starting one row below and one col to the left? Not exactly...
# 5 at (2,3) -> shape at (3,2)-(5,4). Row offset +1, col offset -1.
# 5 at (1,7) -> shape at (2,6)-(4,8). Row offset +1, col offset -1.
# 5 at (4,7) -> shape at (5,6)-(7,8). Row offset +1, col offset -1.
# Yes! The shape starts at (5_row+1, 5_col-1) and extends 3x3.
# And we just read the grid values in that 3x3 window.
#
# But actually, that just happens to be where the nearby shape cells are. The actual rule might be
# more general. Let me just use: find the shape closest to the 5 marker.

solutions["48d8fb45"] = r"""def solve(grid):
    H, W = len(grid), len(grid[0])
    # Find 5 position
    pos5 = None
    color = 0
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 5:
                pos5 = (r, c)
            elif grid[r][c] != 0:
                color = grid[r][c]

    r5, c5 = pos5

    # Find all non-zero, non-5 cells
    cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] != 0 and grid[r][c] != 5]

    # Find the cell closest to the 5 marker
    cells.sort(key=lambda rc: abs(rc[0]-r5) + abs(rc[1]-c5))

    # The closest cell tells us which group to extract
    # Use BFS/flood fill with the 5 as seed - find all cells near 5
    # Actually, just find the cluster nearest to 5
    # Group cells by connected components (8-connectivity? or spatial proximity?)

    # Approach: find connected components, then pick the one closest to 5
    from collections import deque
    visited = set()
    components = []
    for r, c in cells:
        if (r,c) in visited:
            continue
        # BFS with 8-connectivity
        comp = []
        queue = deque([(r,c)])
        visited.add((r,c))
        while queue:
            cr, cc = queue.popleft()
            comp.append((cr,cc))
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = cr+dr, cc+dc
                    if (nr,nc) not in visited and 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != 0 and grid[nr][nc] != 5:
                        visited.add((nr,nc))
                        queue.append((nr,nc))
        components.append(comp)

    # Pick component closest to 5
    best_comp = None
    best_dist = float('inf')
    for comp in components:
        d = min(abs(r-r5)+abs(c-c5) for r,c in comp)
        if d < best_dist:
            best_dist = d
            best_comp = comp

    # Extract bounding box
    min_r = min(r for r,c in best_comp)
    max_r = max(r for r,c in best_comp)
    min_c = min(c for r,c in best_comp)
    max_c = max(c for r,c in best_comp)

    output = [[0]*(max_c-min_c+1) for _ in range(max_r-min_r+1)]
    for r, c in best_comp:
        output[r-min_r][c-min_c] = grid[r][c]

    return output
"""


# 4938f0c2: Shape + connector (3) pattern. Shape is reflected 4 ways around a connector.
# Train 0: 30x30 grid. Shape made of 2s, connector of 3s. Output same size.
# The shape (L-shape of 2s) is adjacent to the 3-connector (2x2 block of 3s).
# In output, the shape is reflected/replicated to all 4 sides of the connector.
#
# Train 1: 10x10 grid. Shape of 2s and connector 3s (2x2).
# Input has shape at top-left, connector at (3,4)-(4,5).
# Output: shape reflected around connector in all 4 quadrants.
# Actually, input already has the 4 reflections! And output is same.
# Wait, train 2 input == output. Let me re-check.
# Train 0: input has shape only in top-left quadrant near connector. Output adds reflections.
# Train 1: input has shape in top-left. Output has all 4 reflections.
# Train 2: input already has all 4 reflections (and output = input).
#
# So the rule: find the 3-colored connector block, find the 2-colored shape,
# reflect it in all 4 quadrants around the connector.

solutions["4938f0c2"] = r"""def solve(grid):
    import copy
    H, W = len(grid), len(grid[0])
    output = copy.deepcopy(grid)

    # Find 3-cells (connector)
    threes = [(r,c) for r in range(H) for c in range(W) if grid[r][c] == 3]
    # Find 2-cells (shape)
    twos = [(r,c) for r in range(H) for c in range(W) if grid[r][c] == 2]

    if not threes or not twos:
        return output

    # Connector bounding box
    min_r3 = min(r for r,c in threes)
    max_r3 = max(r for r,c in threes)
    min_c3 = min(c for r,c in threes)
    max_c3 = max(c for r,c in threes)

    # Center of connector
    cr = (min_r3 + max_r3) / 2.0
    cc = (min_c3 + max_c3) / 2.0

    # Reflect each 2-cell in all 4 ways
    for r, c in twos:
        # Original
        output[r][c] = 2
        # Horizontal reflection (across vertical center of connector)
        nr, nc = r, int(2*cc - c)
        if 0 <= nr < H and 0 <= nc < W:
            output[nr][nc] = 2
        # Vertical reflection (across horizontal center of connector)
        nr, nc = int(2*cr - r), c
        if 0 <= nr < H and 0 <= nc < W:
            output[nr][nc] = 2
        # Both
        nr, nc = int(2*cr - r), int(2*cc - c)
        if 0 <= nr < H and 0 <= nc < W:
            output[nr][nc] = 2

    return output
"""


# 496994bd: Colored blocks at top of grid, reflected at bottom.
# Train 0: 10x3. Top has [2,2,2],[2,2,2],[3,3,3] then 0s. Output mirrors at bottom.
# The pattern at top gets reflected vertically at the bottom.
# Train 1: 10x5. Top has [2,2,2,2,2],[8,8,8,8,8] then 0s. Output mirrors at bottom.
# Rule: the non-zero rows at the top are mirrored at the bottom of the grid.

solutions["496994bd"] = r"""def solve(grid):
    import copy
    H, W = len(grid), len(grid[0])
    output = copy.deepcopy(grid)

    # Find the non-zero rows at top
    non_zero_rows = []
    for r in range(H):
        if any(grid[r][c] != 0 for c in range(W)):
            non_zero_rows.append(r)
        else:
            break

    # Mirror them at the bottom
    n = len(non_zero_rows)
    for i, r in enumerate(non_zero_rows):
        target_r = H - 1 - i
        output[target_r] = list(grid[r])

    return output
"""


# 49d1d64f: Input grid -> output has extra row on top/bottom and extra col on left/right.
# Train 0: 2x2 -> 4x4.
# Input: [[1,2],[3,8]]. Output: [[0,1,2,0],[1,1,2,2],[3,3,8,8],[0,3,8,0]].
# The original grid is in the center (rows 1-2, cols 1-2). Top row: left cell=0, then first row of input, right cell=0.
# Bottom row: left=0, then last row of input, right=0.
# Left column: top=0, then first col of input, bottom=0.
# Right column: top=0, then last col of input, bottom=0.
# Actually: output[0] = [0, grid[0][0], grid[0][1], 0] and output[3] = [0, grid[1][0], grid[1][1], 0]
# output[1] = [grid[0][0], grid[0][0], grid[0][1], grid[0][1]] - each cell duplicated
# output[2] = [grid[1][0], grid[1][0], grid[1][1], grid[1][1]]
#
# Train 1: 2x3 -> 4x5.
# Input: [[1,8,4],[8,3,8]]. Output: [[0,1,8,4,0],[1,1,8,4,4],[8,8,3,8,8],[0,8,3,8,0]].
# output[0] = [0, 1, 8, 4, 0] - 0, first row, 0
# output[1] = [1, 1, 8, 4, 4] - first col val, first row, last col val
# output[2] = [8, 8, 3, 8, 8] - first col val, second row, last col val
# output[3] = [0, 8, 3, 8, 0] - 0, last row, 0
#
# So the pattern is:
# - Add a border of 1 cell around the grid
# - Top border row: [0, grid[0][0..W-1], 0]
# - Bottom border row: [0, grid[H-1][0..W-1], 0]
# - Left border col: [0, grid[0][0], grid[1][0], ..., grid[H-1][0], 0]
# - Right border col: [0, grid[0][W-1], grid[1][W-1], ..., grid[H-1][W-1], 0]
# - Corners are 0
# - Interior is the original grid but each cell duplicated? No, it's just the original grid.
#
# Wait, output[1] = [1, 1, 8, 4, 4]. The middle is [1, 8, 4] = grid[0]. Left edge is grid[0][0]=1, right edge is grid[0][2]=4.
# So: output rows 1 to H are: [grid[r][0]] + grid[r] + [grid[r][W-1]]
# Top row: [0] + grid[0] + [0]
# Bottom row: [0] + grid[H-1] + [0]

solutions["49d1d64f"] = r"""def solve(grid):
    H, W = len(grid), len(grid[0])
    out_H = H + 2
    out_W = W + 2
    output = [[0]*out_W for _ in range(out_H)]

    # Top row
    for c in range(W):
        output[0][c+1] = grid[0][c]

    # Bottom row
    for c in range(W):
        output[out_H-1][c+1] = grid[H-1][c]

    # Middle rows
    for r in range(H):
        output[r+1][0] = grid[r][0]
        for c in range(W):
            output[r+1][c+1] = grid[r][c]
        output[r+1][out_W-1] = grid[r][W-1]

    return output
"""


# 4be741c5: Grid with colored bands/stripes. Output is a single row listing the band colors in order.
# Train 0: 14x16. Bands of colors 4, 2, 8 from left to right. Output: [[4,2,8]].
# Train 1: 9x7. Bands 2, 8, 5 from top to bottom. Output: [[2],[8],[5]].
# Train 2: 11x9. Bands 6, 4, 2, 3 from top to bottom. Output: [[6],[4],[2],[3]].
#
# The bands are rough/noisy at boundaries. The output lists the colors in order.
# If bands are horizontal -> output is a column. If vertical -> output is a row.
#
# Train 0: bands are vertical. Output is 1x3 = [[4,2,8]].
# Train 1: bands are horizontal. Output is 3x1 = [[2],[8],[5]].
#
# To determine band order: look at the dominant color in each row/column.
# For vertical bands: each column has a dominant color. Group consecutive columns by dominant color.
# For horizontal bands: each row has a dominant color.

solutions["4be741c5"] = r"""def solve(grid):
    H, W = len(grid), len(grid[0])
    from collections import Counter

    # Check if bands are horizontal or vertical
    # For horizontal: each row should be mostly one color
    # For vertical: each column should be mostly one color

    # Check rows
    row_colors = []
    for r in range(H):
        c = Counter(grid[r])
        row_colors.append(c.most_common(1)[0][0])

    # Check columns
    col_colors = []
    for c in range(W):
        vals = [grid[r][c] for r in range(H)]
        cnt = Counter(vals)
        col_colors.append(cnt.most_common(1)[0][0])

    # If all rows have same color -> vertical bands
    # If many different row colors -> horizontal bands
    row_unique = len(set(row_colors))
    col_unique = len(set(col_colors))

    if col_unique > row_unique:
        # Vertical bands - list distinct colors in column order
        bands = []
        prev = None
        for c in col_colors:
            if c != prev:
                bands.append(c)
                prev = c
        return [bands]
    else:
        # Horizontal bands
        bands = []
        prev = None
        for c in row_colors:
            if c != prev:
                bands.append(c)
                prev = c
        return [[b] for b in bands]
"""


# 4c4377d9: 3-row 4-col input. Output is 6 rows (input reversed + input).
# Train 0: Input [[9,9,5,9],[5,5,9,9],[9,5,9,9]].
# Output: [[9,5,9,9],[5,5,9,9],[9,9,5,9],[9,9,5,9],[5,5,9,9],[9,5,9,9]].
# That's input reversed (row 2, 1, 0) then input (row 0, 1, 2).
#
# Train 1: Input [[4,1,1,4],[1,1,1,1],[4,4,4,1]].
# Output: [[4,4,4,1],[1,1,1,1],[4,1,1,4],[4,1,1,4],[1,1,1,1],[4,4,4,1]].
# Reversed + original. Yes!

solutions["4c4377d9"] = r"""def solve(grid):
    H = len(grid)
    reversed_grid = list(reversed(grid))
    return [list(row) for row in reversed_grid] + [list(row) for row in grid]
"""


# Now let me write and test everything
print("Solutions defined for:", list(solutions.keys()))

# Verify
PYTHON_PATH = r"C:\Users\atchi\AppData\Local\Programs\Python\Python311\python.exe"

for task_id, code in solutions.items():
    with open(f"data/arc1/{task_id}.json") as f:
        data = json.load(f)
    exec(code)
    all_pass = True
    for i, pair in enumerate(data["train"]):
        result = solve(pair["input"])
        if result != pair["output"]:
            print(f"FAIL: {task_id} train {i}")
            # Show differences
            out = pair["output"]
            for r in range(min(len(result), len(out))):
                for c in range(min(len(result[0]) if result else 0, len(out[0]) if out else 0)):
                    if r < len(result) and c < len(result[r]) and r < len(out) and c < len(out[r]):
                        if result[r][c] != out[r][c]:
                            print(f"  Diff at ({r},{c}): got {result[r][c]}, expected {out[r][c]}")
            if len(result) != len(out):
                print(f"  Row count: got {len(result)}, expected {len(out)}")
            elif result and out and len(result[0]) != len(out[0]):
                print(f"  Col count: got {len(result[0])}, expected {len(out[0])}")
            all_pass = False
    if all_pass:
        print(f"PASS: {task_id}")

# Save passing solutions
passing = {}
for task_id, code in solutions.items():
    with open(f"data/arc1/{task_id}.json") as f:
        data = json.load(f)
    exec(code)
    all_pass = True
    for i, pair in enumerate(data["train"]):
        result = solve(pair["input"])
        if result != pair["output"]:
            all_pass = False
            break
    if all_pass:
        passing[task_id] = code

with open("data/arc_python_solutions_b27.json", "w") as f:
    json.dump(passing, f, indent=2)
print(f"\nSaved {len(passing)} solutions to data/arc_python_solutions_b27.json")
