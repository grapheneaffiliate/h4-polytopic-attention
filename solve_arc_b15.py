import json
import copy

# Task 99fa7670: Each non-zero pixel extends right to edge, then down to bottom-right
# Pattern: find each colored dot, draw line right to edge from dot row, then down right column to bottom
def solve_99fa7670(grid):
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    # Find all non-zero cells
    dots = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                dots.append((r, c, grid[r][c]))
    for r, c, v in dots:
        # Fill right from dot to right edge
        for cc in range(c, cols):
            out[r][cc] = v
        # Fill down on right edge from dot row to bottom (or next dot's row)
        # Find the next dot below
        next_r = rows  # default to bottom
        for r2, c2, v2 in dots:
            if r2 > r and r2 < next_r:
                next_r = r2
        for rr in range(r, next_r):
            out[rr][cols-1] = v
    return out

# Task 9af7a82c: Nested L-shapes / staircase unwinding
# Input is a grid with nested colored regions. Output lists colors from outside to inside,
# each color gets a column, height = count of that color's cells
def solve_9af7a82c(grid):
    rows, cols = len(grid), len(grid[0])
    # Find colors and their counts, ordered from most to least
    from collections import Counter
    counts = Counter()
    for r in range(rows):
        for c in range(cols):
            counts[grid[r][c]] += 1
    # Sort by count descending
    sorted_colors = sorted(counts.items(), key=lambda x: -x[1])
    # Number of distinct colors = number of output columns
    n_colors = len(sorted_colors)
    # Output height = count of most frequent color
    max_count = sorted_colors[0][1]
    out = [[0]*n_colors for _ in range(max_count)]
    for ci, (color, count) in enumerate(sorted_colors):
        for ri in range(count):
            out[ri][ci] = color
    return out

# Task 9edfc990: Flood fill from cells with value 1 through 0s. Reached 0s become 1.
def solve_9edfc990(grid):
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    # BFS from all cells containing value 1, spreading through 0s
    visited = [[False]*cols for _ in range(rows)]
    queue = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                visited[r][c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 0:
                visited[nr][nc] = True
                out[nr][nc] = 1
                queue.append((nr, nc))
    return out

# Task a1570a43: 3s define a rectangular frame. The 2-shape is shifted so its bounding box
# top-left aligns with (frame_r1+1, frame_c1+1), i.e., one cell inside the frame.
def solve_a1570a43(grid):
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    threes = []
    twos = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3:
                threes.append((r, c))
            elif grid[r][c] == 2:
                twos.append((r, c))

    frame_r1 = min(r for r, c in threes)
    frame_c1 = min(c for r, c in threes)
    shape_r1 = min(r for r, c in twos)
    shape_c1 = min(c for r, c in twos)

    dr = (frame_r1 + 1) - shape_r1
    dc = (frame_c1 + 1) - shape_c1

    # Clear old 2s, place new ones
    for r in range(rows):
        for c in range(cols):
            if out[r][c] == 2:
                out[r][c] = 0
    for r, c in twos:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out[nr][nc] = 2
    return out

# Task a2fd1cf0: Draw L-shaped path from point 2 to point 3 using color 8
# Path goes horizontal from 2 toward 3's column, then vertical to 3's row
def solve_a2fd1cf0(grid):
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    # Find 2 and 3 positions
    pos2 = pos3 = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                pos2 = (r, c)
            elif grid[r][c] == 3:
                pos3 = (r, c)
    r2, c2 = pos2
    r3, c3 = pos3
    # Draw horizontal line from 2 toward 3's column
    if c2 < c3:
        for c in range(c2+1, c3+1):
            out[r2][c] = 8
    else:
        for c in range(c3, c2):
            out[r2][c] = 8
    # Draw vertical line from 3's column toward 2's row
    if r2 < r3:
        for r in range(r2+1, r3):
            out[r][c3] = 8
    else:
        for r in range(r3+1, r2):
            out[r][c3] = 8
    return out

# Task a3325580: Sort colored blobs by size (cell count), output as columns
# Each blob's color appears as a column, sorted left to right by size ascending (smallest left? biggest left?)
# Looking at examples more carefully:
# Train 0: shapes are 4(5cells), 6(5cells), 8(4cells), 3(3cells) -> output 5x3 [4,6,8] each row
#   Wait, 4 has cells at (2,2),(3,2),(4,2),(4,3),(5,2) = 5 cells? No: (2,2),(3,2),(4,2),(4,3),(5,2) = 5
#   6 has (3,5),(3,6),(4,6),(5,5),(5,6) = 5 cells
#   8 has (1,7),(1,8),(1,9),(2,9),(3,9) = 5? No: [0,0,0,0,0,0,0,8,8,8], [0,0,0,0,0,0,0,0,0,8], [0,0,0,0,0,0,0,0,0,8]
#     8: (1,7),(1,8),(1,9),(2,9),(3,9) = 5 cells
#   3: (7,1),(7,2),(8,0),(8,1) = 4 cells? No wait: [0,3,3,0,...], [3,3,0,...] -> (7,1),(7,2),(8,0),(8,1) = 4 cells
#   Output is 5x3: columns 4,6,8. Height 5. 3 is excluded (smallest, only 4 cells)
#   Actually output height = 5 = max blob size? And number of columns = 3
#   But 3 blobs have 5 cells each (4,6,8) and blob 3 has 4 cells
#
# Train 1: 9(7cells),6(4cells),4(7cells) -> output 9x2 [9,4].
#   9: (3,1),(4,1),(4,2),(5,0),(5,1),(6,1),(7,1),(7,2),(8,1) = 9 cells
#   6: (3,5),(4,5),(5,4),(5,5),(5,6),(6,6) = 6? Let me recount
#   Actually the output height is determined by the max blob size, and the blobs are sorted by size descending?
#
# Let me reconsider: output rows = max size of included blobs, output cols = number of blobs sorted by size
# Actually: each shape's "height" (bounding box height) determines the output height
# No... Let me look at it differently.
#
# Train 3: shapes 8(3 cells), 6(2 cells), 4(2 cells) -> output 3x1 [8], height=3
# Train 4: shapes 2(3), 3(3) -> output 3x2 [2,3], height=3
# Train 5: shapes 1(3), 4(3?), 8(3) -> output 3x3, height=3
#
# So output height = max cells in any blob, output width = number of blobs with that max count
# No, train 0 has blobs of sizes 5,5,5,4 and output is 5x3 (excludes the 4-cell one)
# Train 1 has blobs of sizes 9,6,7 and output is 9x2... that excludes the 6-cell one
#
# Actually let me recount train 1:
# 9: rows 3-8 col 1 area: (3,1),(4,1),(4,2),(5,0),(5,1),(6,1),(7,1),(7,2),(8,1) = 9
# 6: (3,5),(4,5),(5,4),(5,5),(5,6),(6,6) = 6
# 4: (0,7),(0,8),(0,9),(1,7),(1,9),(2,9),(3,8),(3,9),(4,9) = 9
# Output: 9x2 with [9,4]. So the two biggest (both 9) are shown. 6 is excluded.
#
# So the rule is: find all blobs, sort by size descending, take the ones with the largest size
# Wait but train 0 has 3 blobs of size 5 and one of size 4, output has 3 columns = the 3 largest
# Train 1 has 2 blobs of size 9 and one of 6, output has 2 columns = the 2 largest
# Train 2: 7(3),2(4),3(3),4(2)
#   7: (1,0),(1,1),(1,2),(2,2) = 4? Let me recheck input:
#   [0,0,0,0,0,0,0,0,0,1],[7,7,7,0,0,2,2,0,0,1],[0,0,7,0,0,0,2,2,0,1],[0,0,0,0,0,0,2,0,0,1]
#   7: (1,0),(1,1),(1,2),(2,2) = 4
#   2: (1,5),(1,6),(2,6),(2,7),(3,6) = 5
#   1: (0,9),(1,9),(2,9),(3,9) = 4
#   3: (5,3),(6,3),(6,4) = 3
#   Output: 5x1 [2]. Height=5, only 1 column. The largest blob is 2 with 5 cells.
#
# So: find all blobs, the maximum blob size = output height.
# Blobs with that max size are included as columns.
# But train 0 has sizes 5,5,5,4 -> 3 columns (the three 5s). OK.
# But then how are they ordered left to right?
# Train 0: colors 4,6,8 in output. Positions: 4 is leftmost, 6 middle, 8 rightmost -> sorted by x position
# Train 1: colors 9,4. 9 is leftish, 4 is rightish -> sorted by x position
# Train 5: 1,4,8 -> 1 leftmost, 4 middle, 8 rightmost -> sorted by column position
#
# So: find blobs, get max size, keep blobs with max size, sort by average column position, output
def solve_a3325580(grid):
    rows, cols = len(grid), len(grid[0])
    # Find connected components by color
    visited = [[False]*cols for _ in range(rows)]
    blobs = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                # BFS
                cells = []
                queue = [(r, c)]
                visited[r][c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    cells.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                blobs.append((color, cells))

    # Find max blob size
    max_size = max(len(cells) for _, cells in blobs)

    # Keep blobs with max size
    max_blobs = [(color, cells) for color, cells in blobs if len(cells) == max_size]

    # Sort by average column position
    max_blobs.sort(key=lambda x: sum(c for _, c in x[1]) / len(x[1]))

    # Output: height = max_size, width = number of max blobs
    n_cols = len(max_blobs)
    out = []
    for ri in range(max_size):
        row = [b[0] for b in max_blobs]
        out.append(row)
    return out

# Task a3df8b1e: Bouncing ball - 1 starts at bottom-left, bounces off walls going up
# Width determines bounce pattern. Ball goes up-right, bounces off right wall, goes up-left, etc.
def solve_a3df8b1e(grid):
    rows, cols = len(grid), len(grid[0])
    out = [[0]*cols for _ in range(rows)]
    # Ball starts at bottom-left (rows-1, 0), moves upward
    # Position at row r from bottom
    # Pattern: position bounces between 0 and cols-1
    # At bottom (row rows-1), pos = 0
    # Going up, pos increases by 1 each row until hitting cols-1, then decreases
    pos = 0
    direction = 1  # 1 = moving right, -1 = moving left
    for r in range(rows-1, -1, -1):
        out[r][pos] = 1
        if r > 0:  # compute next position
            next_pos = pos + direction
            if next_pos >= cols:
                direction = -1
                next_pos = pos + direction
            elif next_pos < 0:
                direction = 1
                next_pos = pos + direction
            pos = next_pos
    return out

# Task a48eeaf7: 2x2 block stays, 5s are attracted toward the block
# Each 5 moves to be adjacent to the 2x2 block, at the closest position
# The 5s indicate corners/adjacent positions around the block
# Looking at train 0: block at (3,3)-(4,4), 5s at (0,3),(3,8),(7,7)
# Output: 5s at (2,3),(3,5),(5,5) - each 5 moved to be 1 cell away from the block
# The 5 at (0,3) moved to (2,3) - directly above the block
# The 5 at (3,8) moved to (3,5) - directly right of block
# The 5 at (7,7) moved to (5,5) - diagonally below-right of block? No, (5,5) is below-right corner
# Actually output has 5 at (2,3), (3,5), (5,5). Block is at rows 3-4, cols 3-4.
# (2,3) = directly above top-left of block
# (3,5) = directly right of top-right of block
# (5,5) = directly below-right (diagonal) of block... hmm
# Wait, re-reading output more carefully:
# O: [0,0,0,5,0,0,0,0,0,0] row 2 -> 5 at (2,3)
# O: [0,0,0,2,2,5,0,0,0,0] row 3 -> 5 at (3,5)
# O: [0,0,0,2,2,0,0,0,0,0] row 4
# O: [0,0,0,0,0,5,0,0,0,0] row 5 -> 5 at (5,5)
# So each 5 is projected onto the nearest edge of the block (extended)
# (0,3) -> project down to row just above block -> (2,3). Col stays same (within block col range)
# (3,8) -> project left to col just right of block -> (3,5). Row stays same (within block row range)
# (7,7) -> project to nearest corner: (5,5)? This is the bottom-right corner extended by 1
#
# Actually, it seems like each 5 is moved to the nearest point on the "border" around the 2x2 block
# The border is 1 cell away from the block.
# For (0,3): nearest border point going straight down = (2,3)
# For (3,8): nearest border point going straight left = (3,5)
# For (7,7): nearest border point... the block occupies (3,3)-(4,4).
#   Going from (7,7): need to reach border. Closest approach: move toward block diag
#   The projection is: clamp row to [2,5], clamp col to [2,5], then push to just outside
#   Actually, each 5 is projected toward the block. It moves in a straight line toward the 2x2 block
#   and stops 1 cell outside.
#
# Let me think differently. The 5 at (0,3) has col=3 which is within block cols [3,4].
# It's above the block (row 0 < row 3). So it moves to row 2 (one above block top).
# The 5 at (3,8) has row=3 which is within block rows [3,4].
# It's right of block (col 8 > col 4). So it moves to col 5 (one right of block right).
# The 5 at (7,7) is below-right. Both row and col are outside block.
# It should go to the corner: (5,5)? That's (block_bottom+1, block_right+1).
# But why not (5,4) or (4,5)?
#
# For diagonal cases, the 5 maps to the corner of the extended border.
# For axis-aligned cases, the 5 maps to the nearest point on that axis.
#
# Let me verify with train 1:
# Block at (2,5)-(3,6). 5s at (0,8),(3,1),(6,9),(8,5)
# Output: 5s at (1,7),(3,4),(4,7),(4,5)? Let me reread:
# O row 1: [0,0,0,0,0,0,0,5,0,0] -> 5 at (1,7)
# O row 3: [0,0,0,0,5,2,2,0,0,0] -> 5 at (3,4)
# O row 4: [0,0,0,0,0,5,0,5,0,0] -> 5s at (4,5) and (4,7)
#
# 5 at (0,8): above-right of block. Row < block top (2), col > block right (6).
#   -> corner: (1, 7) = (block_top-1, block_right+1). Yes!
# 5 at (3,1): left of block. Row 3 is within block rows [2,3]. Col 1 < block left 5.
#   -> (3, 4) = (3, block_left-1). Yes!
# 5 at (6,9): below-right. Row 6 > block bottom 3. Col 9 > block right 6.
#   -> corner: (4, 7) = (block_bottom+1, block_right+1). Yes!
# 5 at (8,5): below block. Col 5 within block cols [5,6]. Row 8 > block bottom 3.
#   -> (4, 5) = (block_bottom+1, 5). Yes!
def solve_a48eeaf7(grid):
    rows, cols = len(grid), len(grid[0])
    out = [[0]*cols for _ in range(rows)]

    # Find 2x2 block
    block_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                block_cells.append((r, c))
    br_min = min(r for r, c in block_cells)
    br_max = max(r for r, c in block_cells)
    bc_min = min(c for r, c in block_cells)
    bc_max = max(c for r, c in block_cells)

    # Place block
    for r, c in block_cells:
        out[r][c] = 2

    # Find 5s
    fives = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                fives.append((r, c))

    # Project each 5 onto border of block
    for r, c in fives:
        # Determine new position
        if r < br_min:
            nr = br_min - 1
        elif r > br_max:
            nr = br_max + 1
        else:
            nr = r

        if c < bc_min:
            nc = bc_min - 1
        elif c > bc_max:
            nc = bc_max + 1
        else:
            nc = c

        out[nr][nc] = 5

    return out

# Task a5f85a15: Diagonal lines alternate between original color and 4.
# Group cells by diagonal (same r-c value). Within each diagonal, sorted by row,
# cells alternate: keep original, replace with 4, keep, replace, ...
def solve_a5f85a15(grid):
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    # Group non-zero cells by diagonal (r-c)
    diags = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                key = r - c
                if key not in diags:
                    diags[key] = []
                diags[key].append((r, c))
    # Within each diagonal, alternate: keep, 4, keep, 4, ...
    for key, cells in diags.items():
        cells.sort()  # sort by row
        for i, (r, c) in enumerate(cells):
            if i % 2 == 1:
                out[r][c] = 4
    return out

# Task a61ba2ce: Four L-shaped pieces arranged into a 4x4 grid
# Each piece is an L-shape (2 cells). They get arranged into corners of a 4x4 output.
# Looking at train 0:
# 8 at (1,6),(1,7),(2,6) -> top-right corner piece: forms [[8,8],[8,0]]
# 2 at (3,1),(3,2),(4,2) -> top-left corner piece: forms [[2,2],[0,2]]
# 1 at (7,8),(8,7),(8,8) -> bottom-right: [[0,1],[1,1]]
# 3 at (9,3),(10,3),(10,4) -> bottom-left: [[3,0],[3,3]]
# Output: [[8,8,2,2],[8,0,0,2],[3,0,0,1],[3,3,1,1]]
# So the 4 pieces are placed in corners: top-left has 8, top-right has 2, bottom-left has 3, bottom-right has 1
#
# The pieces' spatial arrangement in the input determines their corner in the output.
# 8 is in top-right area of input -> goes to top-right of output... wait output has 8 top-left.
# Let me look at piece shapes:
# 8: [[8,8],[8,0]] - L pointing down-right
# 2: [[2,2],[0,2]] - L pointing down-left
# 1: [[0,1],[1,1]] - L pointing up-left
# 3: [[3,0],[3,3]] - L pointing up-right
# In output: 8(top-left), 2(top-right), 3(bottom-left), 1(bottom-right)
# The L orientation determines the corner!
# L pointing down-right (open at bottom-right) goes to top-left
# L pointing down-left (open at bottom-left) goes to top-right
# L pointing up-right (open at top-right) goes to bottom-left
# L pointing up-left (open at top-left) goes to bottom-right
#
# Train 1 verification:
# 8 at (1,8),(1,9),(2,9) -> [[8,8],[0,8]] L pointing down-left -> top-right
# 1 at (3,2),(3,3),(4,2) -> [[1,1],[1,0]] L pointing down-right -> top-left
# 2 at (5,8),(6,7),(6,8) -> [[0,2],[2,2]] L pointing up-left -> bottom-right
# 4 at (9,4),(10,4),(10,5) -> [[4,0],[4,4]] L pointing up-right -> bottom-left
# Output: [[1,1,8,8],[1,0,0,8],[4,0,0,2],[4,4,2,2]] Yes!

def solve_a61ba2ce(grid):
    rows, cols = len(grid), len(grid[0])
    # Find all colored blobs
    visited = [[False]*cols for _ in range(rows)]
    pieces = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                cells = []
                queue = [(r, c)]
                visited[r][c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    cells.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                pieces.append((color, cells))

    # For each piece, determine its L-shape orientation
    # Normalize to 2x2 bounding box
    corner_pieces = {}  # 'TL', 'TR', 'BL', 'BR' -> (color, 2x2 pattern)
    for color, cells in pieces:
        rmin = min(r for r, c in cells)
        cmin = min(c for r, c in cells)
        # Create 2x2 pattern
        pattern = [[0,0],[0,0]]
        for r, c in cells:
            pattern[r-rmin][c-cmin] = color

        # Determine which corner is empty (the 0 in the 2x2)
        if pattern[0][0] == 0:  # empty top-left -> L points up-left -> goes to bottom-right
            corner_pieces['BR'] = (color, pattern)
        elif pattern[0][1] == 0:  # empty top-right -> L points up-right -> goes to bottom-left
            corner_pieces['BL'] = (color, pattern)
        elif pattern[1][0] == 0:  # empty bottom-left -> L points down-left -> goes to top-right
            corner_pieces['TR'] = (color, pattern)
        elif pattern[1][1] == 0:  # empty bottom-right -> L points down-right -> goes to top-left
            corner_pieces['TL'] = (color, pattern)

    # Build 4x4 output
    out = [[0]*4 for _ in range(4)]
    if 'TL' in corner_pieces:
        p = corner_pieces['TL'][1]
        for r in range(2):
            for c in range(2):
                out[r][c] = p[r][c]
    if 'TR' in corner_pieces:
        p = corner_pieces['TR'][1]
        for r in range(2):
            for c in range(2):
                out[r][c+2] = p[r][c]
    if 'BL' in corner_pieces:
        p = corner_pieces['BL'][1]
        for r in range(2):
            for c in range(2):
                out[r+2][c] = p[r][c]
    if 'BR' in corner_pieces:
        p = corner_pieces['BR'][1]
        for r in range(2):
            for c in range(2):
                out[r+2][c+2] = p[r][c]

    return out


# Now test all solutions
solvers = {
    '99fa7670': solve_99fa7670,
    '9af7a82c': solve_9af7a82c,
    '9edfc990': solve_9edfc990,
    'a1570a43': solve_a1570a43,
    'a2fd1cf0': solve_a2fd1cf0,
    'a3325580': solve_a3325580,
    'a3df8b1e': solve_a3df8b1e,
    'a48eeaf7': solve_a48eeaf7,
    'a5f85a15': solve_a5f85a15,
    'a61ba2ce': solve_a61ba2ce,
}

results = {}
all_pass = True

for task_id, solver in solvers.items():
    data = json.load(open(f'C:/Users/atchi/h4-polytopic-attention/data/arc1/{task_id}.json'))
    task_pass = True

    # Test on training pairs
    for i, pair in enumerate(data['train']):
        predicted = solver(pair['input'])
        expected = pair['output']
        if predicted != expected:
            print(f"FAIL {task_id} train {i}")
            print(f"  Expected: {expected[:3]}...")
            print(f"  Got:      {predicted[:3]}...")
            task_pass = False
            all_pass = False

    # Test on test pairs
    for i, pair in enumerate(data['test']):
        predicted = solver(pair['input'])
        if 'output' in pair:
            expected = pair['output']
            if predicted != expected:
                print(f"FAIL {task_id} test {i}")
                print(f"  Expected: {expected[:3]}...")
                print(f"  Got:      {predicted[:3]}...")
                task_pass = False
                all_pass = False

    if task_pass:
        print(f"PASS {task_id}")

    # Store test outputs
    test_outputs = []
    for pair in data['test']:
        test_outputs.append(solver(pair['input']))
    results[task_id] = test_outputs

if all_pass:
    print("\nALL TESTS PASSED!")
else:
    print("\nSome tests failed.")

# Save results
with open('C:/Users/atchi/h4-polytopic-attention/data/arc_python_solutions_b15.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to data/arc_python_solutions_b15.json")
