import json

solutions = {}

# ============================================================
# 4c5c2cf0: There's a "connector" pattern (like an X with 4 dots) and a shape.
# The connector has cells at specific positions forming a symmetric pattern.
# The shape gets reflected/copied to each "arm" of the connector.
#
# Analysis: Two objects - a "shape" (color A) and a "connector" (color B, X-like pattern).
# The shape is adjacent to the connector. The shape gets reflected through each
# arm of the connector pattern.
#
# The connector is a 3x3-ish cross pattern with cells at corners and center absent,
# or similar. Looking more carefully:
# Connector pattern has 4 non-zero cells forming a diamond/X shape.
# The shape object gets stamped at each of the 4 reflected positions around the connector.

def solve_4c5c2cf0(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # Find all non-zero cells, separate into two color groups
    colors = {}
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                col = grid[r][c]
                if col not in colors:
                    colors[col] = []
                colors[col].append((r, c))

    color_list = list(colors.keys())

    # The connector has 4-fold symmetry (X/diamond pattern with center).
    # Check each color for this property.
    connector_color = None
    shape_color = None

    for col in color_list:
        cells = colors[col]
        rs = [r for r,c in cells]
        cs = [c for r,c in cells]
        cr = (min(rs) + max(rs)) / 2
        cc = (min(cs) + max(cs)) / 2
        # Check 4-fold symmetry: for each cell, all 4 reflections should exist
        cell_set = set(cells)
        symmetric = True
        for r, c in cells:
            # Check horizontal, vertical, and both reflections
            hr, hc = r, int(2*cc - c)
            vr, vc = int(2*cr - r), c
            br, bc = int(2*cr - r), int(2*cc - c)
            if (hr, hc) not in cell_set or (vr, vc) not in cell_set or (br, bc) not in cell_set:
                symmetric = False
                break
        if symmetric and len(cells) >= 4:
            connector_color = col
        else:
            shape_color = col

    if connector_color is None or shape_color is None:
        if len(colors[color_list[0]]) < len(colors[color_list[1]]):
            connector_color = color_list[0]
            shape_color = color_list[1]
        else:
            connector_color = color_list[1]
            shape_color = color_list[0]

    conn_cells = colors[connector_color]
    shape_cells = colors[shape_color]

    # Find connector center
    conn_rs = [r for r,c in conn_cells]
    conn_cs = [c for r,c in conn_cells]
    conn_cr = (min(conn_rs) + max(conn_rs)) / 2
    conn_cc = (min(conn_cs) + max(conn_cs)) / 2

    ccr = round(conn_cr)
    ccc = round(conn_cc)

    # For each shape cell, compute its reflection through the connector center
    for r, c in shape_cells:
        # Horizontal reflection
        nr, nc = r, 2*ccc - c
        if 0 <= nr < h and 0 <= nc < w and out[nr][nc] == 0:
            out[nr][nc] = shape_color
        # Vertical reflection
        nr, nc = 2*ccr - r, c
        if 0 <= nr < h and 0 <= nc < w and out[nr][nc] == 0:
            out[nr][nc] = shape_color
        # Both reflections
        nr, nc = 2*ccr - r, 2*ccc - c
        if 0 <= nr < h and 0 <= nc < w and out[nr][nc] == 0:
            out[nr][nc] = shape_color

    return out

# ============================================================
# 508bd3b6: Ball bouncing off wall
# There's a diagonal line of 8s heading toward a wall of 2s.
# The ball "bounces" - continues as 3s, reflecting off the wall.
# The direction reverses perpendicular to the wall.

def solve_508bd3b6(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # Find 8 cells (the arrow/diagonal line)
    eights = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 8:
                eights.append((r, c))
    eights.sort()

    # Direction of the 8-arrow
    dr = eights[1][0] - eights[0][0]
    dc = eights[1][1] - eights[0][1]
    # Normalize to unit steps
    dr = 1 if dr > 0 else (-1 if dr < 0 else 0)
    dc = 1 if dc > 0 else (-1 if dc < 0 else 0)

    # Find wall of 2s
    two_set = set()
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 2:
                two_set.add((r, c))

    # Determine wall orientation
    two_rows = set(r for r, c in two_set)
    two_cols = set(c for r, c in two_set)
    wall_is_horizontal = any(sum(1 for c2 in range(w) if (row, c2) in two_set) == w for row in two_rows)
    wall_is_vertical = any(sum(1 for r2 in range(h) if (r2, col) in two_set) == h for col in two_cols)

    # The ball continues from the tip of the 8-arrow in the arrow direction,
    # bouncing off the wall when it would enter a wall cell.
    # Find which end of the 8-line to continue from:
    # Continue in the direction that goes TOWARD the wall.
    # Test: does continuing from last 8 in arrow direction lead toward wall?
    last_8 = eights[-1]  # last in sorted order = farthest in +row direction
    first_8 = eights[0]

    # Try continuing from last_8 in direction (dr, dc)
    test_r, test_c = last_8[0] + dr, last_8[1] + dc
    goes_toward_wall_from_last = False
    for step in range(1, max(h, w)):
        tr, tc = last_8[0] + step * dr, last_8[1] + step * dc
        if (tr, tc) in two_set or tr < 0 or tr >= h or tc < 0 or tc >= w:
            if (tr, tc) in two_set:
                goes_toward_wall_from_last = True
            break

    if goes_toward_wall_from_last:
        start = last_8
        move_dr, move_dc = dr, dc
    else:
        start = first_8
        move_dr, move_dc = -dr, -dc

    # Draw path from start in direction (move_dr, move_dc) until hitting wall, then bounce
    r, c = start
    cur_dr, cur_dc = move_dr, move_dc
    while True:
        nr, nc = r + cur_dr, c + cur_dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            break
        if (nr, nc) in two_set:
            # Bounce: reverse the component perpendicular to wall
            if wall_is_horizontal:
                cur_dr = -cur_dr  # reverse vertical
            elif wall_is_vertical:
                cur_dc = -cur_dc  # reverse horizontal
            nr, nc = r + cur_dr, c + cur_dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                break
            if (nr, nc) in two_set:
                break
        if out[nr][nc] != 0:
            break
        out[nr][nc] = 3
        r, c = nr, nc

    return out

# ============================================================
# 50cb2852: Fill interior of rectangles with 8
# Each rectangle is a solid block of one color. Fill the interior (non-border) cells with 8.

def solve_50cb2852(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # Find connected components of non-zero cells
    visited = [[False]*w for _ in range(h)]

    def flood_fill(sr, sc, color):
        cells = []
        stack = [(sr, sc)]
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= h or c < 0 or c >= w:
                continue
            if visited[r][c] or grid[r][c] != color:
                continue
            visited[r][c] = True
            cells.append((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((r+dr, c+dc))
        return cells

    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                cells = flood_fill(r, c, color)
                if len(cells) < 4:
                    continue
                # Find bounding box
                rs = [r for r,c in cells]
                cs = [c for r,c in cells]
                min_r, max_r = min(rs), max(rs)
                min_c, max_c = min(cs), max(cs)
                # Fill interior with 8
                for ir in range(min_r+1, max_r):
                    for ic in range(min_c+1, max_c):
                        out[ir][ic] = 8

    return out

# ============================================================
# 5117e062: Multiple shapes, one has color 8 marker.
# Find the shape whose pattern matches the 8-marked shape (ignoring 8),
# and output that shape's color filling the 8-marked pattern.
#
# Analysis: There are several 3x3 shapes. One has an 8 cell marking it as "key".
# There are other shapes that are "patterns". We need to find which non-8, non-key
# shape has the same spatial pattern as the key shape (ignoring the 8 cell).
# Output is the 3x3 bounding box of that matched shape's color filling the key pattern.
#
# Wait, re-reading:
# Example 1: shapes are color 3 (has 8), color 4, color 2, color 6
# Color 4 pattern: (0,10),(1,9),(1,11),(2,10) = cross with 8 at (1,10)
# Actually color 4 has cells at specific positions AND an 8 cell.
# The 8 is inside the color 4 shape.
# Color 2 pattern: (4,4),(5,4),(5,5),(5,6),(6,5) = plus/cross
# Color 3 pattern: (0,2),(1,1)(1,2)(1,3),(2,1)(2,2)
# Color 6 pattern: (9,8)(9,9),(10,8)(10,10),(11,8)(11,9)(11,10)
# Output is [[0,4,0],[4,4,4],[0,4,0]] which is the color 4 pattern = cross
#
# So the 8-marked shape is the "question" - what color has same pattern as...
# Actually: the output is the pattern of color 2 (cross shape) but colored as 4.
# Color 2: (4,4),(5,4)(5,5)(5,6),(6,5) relative to bbox (4,4)-(6,6):
# row0: 2,0,0 -> no. Let me re-check.
# (4,4)=2, (5,4)=2,(5,5)=2,(5,6)=2, (6,5)=2
# Relative: (0,0),(1,0),(1,1),(1,2),(2,1) ->
# 1 0 0
# 1 1 1
# 0 1 0
# That's a plus/cross. Output [[0,4,0],[4,4,4],[0,4,0]] is also a cross but rotated 180.
# Wait output is: row0=[0,4,0], row1=[4,4,4], row2=[0,4,0] which IS a plus.
# And color 2 relative pattern is:
# (0,0)=1,(1,0)=1,(1,1)=1,(1,2)=1,(2,1)=1
# 1 0 0
# 1 1 1
# 0 1 0
# That's not the same as output. Let me re-examine.
#
# Hmm. Let me look at the 8-marked shape more carefully.
# Color 4 cells: (0,10),(1,9),(1,11),(2,10). The 8 is at (1,10).
# So the full shape including 8:
# (0,10)=4, (1,9)=4, (1,10)=8, (1,11)=4, (2,10)=4
# bbox rows 0-2, cols 9-11:
# 0 4 0      0 1 0
# 4 8 4  ->  1 X 1   (X = hole marked by 8)
# 0 4 0      0 1 0
# So it's a plus/cross with center marked.
#
# Now which OTHER shape has the same pattern when you consider the 8 as "present"?
# All shapes are 3x3 patterns. The 8 marks the "special" shape.
# Output color is 4 (the 8-marked shape's color).
# Output pattern is [[0,4,0],[4,4,4],[0,4,0]] - the plus pattern.
#
# So the task is: the 8-marked shape has a certain pattern (with 8 as part of it).
# Replace the 8 with the shape's color and output just that 3x3 pattern.
#
# Wait, that gives [[0,4,0],[4,4,4],[0,4,0]] for example 1. Yes!
# Let me verify example 2: Color 3 has 8 at (5,3).
# Color 3 cells: (4,3)(4,4),(5,2),(5,3)=8,(6,3)(6,4)
# bbox rows 4-6, cols 2-4:
# (4,2)=0,(4,3)=3,(4,4)=3 -> 0 3 3
# (5,2)=3,(5,3)=8,(5,4)=0 -> 3 8 0
# (6,2)=0,(6,3)=3,(6,4)=3 -> 0 3 3
# Replace 8 with 3: [[0,3,3],[3,3,0],[0,3,3]]
# Expected output: [[0,3,3],[3,3,0],[0,3,3]] ✓
#
# Example 3: Color 2 has 8 at (3,2).
# (2,1)(2,2)(2,3)=2, (3,2)=8, (4,1)(4,2)=2
# bbox rows 2-4, cols 1-3:
# 2 2 2
# 0 8 0
# 2 2 0
# Replace 8 with 2: [[2,2,2],[0,2,0],[2,2,0]]
# Expected: [[2,2,2],[0,2,0],[2,2,0]] ✓

def solve_5117e062(grid):
    h, w = len(grid), len(grid[0])

    # Find the 8 cell
    eight_r, eight_c = -1, -1
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 8:
                eight_r, eight_c = r, c

    # Find the color of the shape containing the 8
    # Look at neighbors of the 8 cell
    shape_color = 0
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        nr, nc = eight_r + dr, eight_c + dc
        if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0 and grid[nr][nc] != 8:
            shape_color = grid[nr][nc]
            break

    # Find all cells of this shape color + the 8 cell
    cells = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == shape_color or (r == eight_r and c == eight_c):
                cells.append((r, c))

    # Actually we need to find just the connected component containing the 8
    # Let's get the bounding box of cells near the 8
    # Find connected component of shape_color + 8
    visited = set()
    stack = [(eight_r, eight_c)]
    component = []
    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue
        if r < 0 or r >= h or c < 0 or c >= w:
            continue
        if grid[r][c] != shape_color and grid[r][c] != 8:
            continue
        visited.add((r, c))
        component.append((r, c))
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                stack.append((r+dr, c+dc))

    rs = [r for r, c in component]
    cs = [c for r, c in component]
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)

    # Build output: the shape pattern with 8 replaced by shape_color
    result = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            if grid[r][c] == shape_color or grid[r][c] == 8:
                row.append(shape_color)
            else:
                row.append(0)
        result.append(row)

    return result

# ============================================================
# 5168d44c: A 2x2 box moves along a trail of 3s.
# The box (color 2) sits at a position. There are 3s extending in a line.
# The box needs to move to a new position along the 3 trail.
#
# Looking more carefully at the examples:
# Example 1: 2x2 box at (2,0)-(4,2), 3s extend right on row 3.
# Trail: 3s at specific positions with gaps.
# Output: box moves to a new position.
#
# Actually re-reading: The box has a 3 inside it (the center cell in the direction).
# The trail of 3s leads to a position. The box needs to "hop" along the 3 trail
# to the furthest 3 position.
#
# Example 1 input: Box at rows 2-4, cols 0-2. 3s on row 3 at cols 3,5,7,9,11.
# Output: Box moved to cols 2-4 (shifted right by 2).
# Wait no. Let me look again.
#
# Input row 3: [2,3,2,3,0,3,0,3,0,3,0,3,0]
# The 2x2 box is 3x3 with 3 at center:
# rows 2-4, cols 0-2:
# 2 2 2
# 2 3 2
# 2 2 2
# And then 3s trail: col 3, 5, 7, 9, 11 on row 3
# Output row 3: [0,3,2,3,2,3,0,3,0,3,0,3,0]
# Box moved right by 2 to cols 2-4...
# Actually output:
# row 2: 0 0 2 2 2 0 0 0 0 0 0 0 0
# row 3: 0 3 2 3 2 3 0 3 0 3 0 3 0
# row 4: 0 0 2 2 2 0 0 0 0 0 0 0 0
# Box is at cols 2-4, rows 2-4. Moved right by 2.
#
# Example 2 input: Box at rows 3-5 cols 3-5, 3s on col 4 at rows 0,2,6,8,10,12
# Output: Box moved down by 2 to rows 5-7.
# Wait, output:
# row 3: 0 0 0 0 3 0 0
# row 4: 0 0 0 0 0 0 0
# row 5: 0 0 0 0 3 0 0
# row 6: 0 0 0 2 2 2 0
# row 7: 0 0 0 2 3 2 0 -- wait this doesn't match
#
# Let me re-read example 2 more carefully.
# Input:
# row 0: 0 0 3 0 0        -- 3 at col 2? No wait width is 7
# Actually the format: [[0,0,3,0,0,0,0], ...]
# Hmm wait this is 5168d44c. Let me re-check.
#
# Input example 2:
# [[0,0,0,0,3,0,0],  row 0
#  [0,0,0,0,0,0,0],  row 1
#  [0,0,0,0,3,0,0],  row 2
#  [0,0,0,2,2,2,0],  row 3
#  [0,0,0,2,3,2,0],  row 4
#  [0,0,0,2,2,2,0],  row 5
#  [0,0,0,0,3,0,0],  row 6
#  ...
#  [0,0,0,0,3,0,0],  row 12
# Output:
# row 0-4: 3s on col 4
# row 5: 0 0 0 2 2 2 0  -- wait no
#
# Output example 2:
# [[0,0,0,0,3,0,0],  row 0
#  [0,0,0,0,0,0,0],
#  [0,0,0,0,3,0,0],
#  [0,0,0,0,0,0,0],  -- box gone from here
#  [0,0,0,0,3,0,0],
#  [0,0,0,2,2,2,0],  row 5
#  [0,0,0,2,3,2,0],  row 6
#  [0,0,0,2,2,2,0],  row 7
#  ...
# Box moved from rows 3-5 to rows 5-7. Moved down by 2.
#
# Example 3 input: Box at rows 1-3, cols 1-3. 3 at (0,2),(4,2),(6,2)
# Output: Box at rows 3-5. Moved down by 2.
#
# So the pattern: the box hops one step along the 3-trail direction.
# The 3s are spaced 2 apart. The box moves by 2 in the trail direction.
#
# Wait, let me count: in example 1, trail 3s at cols 3,5,7,9,11. Gap is 2.
# Box center was at col 1, moved to col 3. That's moving to the first 3.
#
# In example 2, trail 3s at rows 0,2,6,8,10,12. Box center at row 4.
# The 3s adjacent to box (within trail): row 2 (before) and row 6 (after).
# Box moved to... center at row 6. So it moved toward the longer trail side?
# No - from rows 3-5 to rows 5-7, center from 4 to 6. It moved +2.
# The 3s below: rows 6,8,10,12 (4 of them). Above: rows 0,2 (2 of them).
# More 3s below, so it moved toward the longer trail.
#
# In example 3: Box center at (2,2). 3 above at (0,2), 3s below at (4,2),(6,2).
# More below (2) than above (1). Box moved down to center at (4,2).
#
# So: count 3s on each side of the trail. Move box toward the side with more 3s,
# by one hop (2 cells in the trail direction).

def solve_5168d44c(grid):
    h, w = len(grid), len(grid[0])
    out = [[0]*w for _ in range(h)]

    # Find the box (3x3 of 2s with 3 center)
    box_cells = []
    center_r, center_c = -1, -1
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 2:
                box_cells.append((r, c))

    # Find box center (the 3 inside the box)
    br = [r for r,c in box_cells]
    bc = [c for r,c in box_cells]
    center_r = (min(br) + max(br)) // 2
    center_c = (min(bc) + max(bc)) // 2

    # Find trail 3s (not the center)
    trail_3s = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 3 and (r, c) != (center_r, center_c):
                trail_3s.append((r, c))

    # Determine trail direction
    # All trail 3s should be on same row or same col as center
    same_row = all(r == center_r for r, c in trail_3s)
    same_col = all(c == center_c for r, c in trail_3s)

    if same_row:
        # Horizontal trail
        left = [c for r, c in trail_3s if c < center_c]
        right = [c for r, c in trail_3s if c > center_c]
        if len(right) > len(left):
            move_dr, move_dc = 0, 2
        else:
            move_dr, move_dc = 0, -2
    else:
        # Vertical trail
        above = [r for r, c in trail_3s if r < center_r]
        below = [r for r, c in trail_3s if r > center_r]
        if len(below) > len(above):
            move_dr, move_dc = 2, 0
        else:
            move_dr, move_dc = -2, 0

    # Place ALL 3s from input (trail + original center) - they all stay
    all_3s = trail_3s + [(center_r, center_c)]
    for r, c in all_3s:
        out[r][c] = 3

    # Place box (2-cells) at new position, overwriting 0s but not 3s at center
    for r, c in box_cells:
        nr = r + move_dr
        nc = c + move_dc
        if 0 <= nr < h and 0 <= nc < w:
            out[nr][nc] = 2

    # The new box center remains 3
    new_cr = center_r + move_dr
    new_cc = center_c + move_dc
    out[new_cr][new_cc] = 3

    return out

# ============================================================
# 539a4f51: Fractal/recursive pattern tiling
# Input is 5x5 with a pattern of colors and 0s. The diagonal has special structure.
# Output is 10x10 - double the size.
# The pattern: input has regions separated by a "diagonal" of specific colors.
# Each cell in a 2x2 meta-grid gets filled based on the pattern.
#
# Looking at example 1: Input 5x5, output 10x10
# Input: [[2,2,2,3,0],[2,2,2,3,0],[2,2,2,3,0],[3,3,3,3,0],[0,0,0,0,0]]
# The non-zero part is a 4x4 grid with a 3x3 block of 2s, bordered by 3s, and 0s.
# Output doubles this pattern recursively.
#
# The input defines a self-similar pattern. The 0 regions get replaced with
# copies of the full pattern, while the colored regions stay.
#
# Actually: The input is read as a set of "tiles". The colors on the diagonal
# define a hierarchy. The 0 region gets replaced with the smallest color tile.
#
# Let me look at it differently. The 5x5 input has colors along the diagonal:
# (0,0)-(2,2): color 2, (3,3): color 3, (4,4): color 0 (empty)
# But wait, (0,3)=3, (1,3)=3, (2,3)=3 and (3,0)=3,(3,1)=3,(3,2)=3
# So color 3 forms a cross/separator.
#
# Output 10x10: The pattern is doubled. Each quadrant:
# Top-left (0-3, 0-3): original 4x4 non-zero part
# Top-right (0-3, 4-7): filled with color 2
# Bottom-left (4-7, 0-3): filled with color 2
# Top-right and bottom-left use the top-left color (2)
# Bottom-right (4-7, 4-7): same pattern again but with 3
# Etc.
# Wait, output:
# [[2,2,2,3,2,2,2,3,2,2],
#  [2,2,2,3,2,2,2,3,2,2],
#  [2,2,2,3,2,2,2,3,2,2],
#  [3,3,3,3,2,2,2,3,2,2],
#  [2,2,2,2,2,2,2,3,2,2],
#  [2,2,2,2,2,2,2,3,2,2],
#  [2,2,2,2,2,2,2,3,2,2],
#  [3,3,3,3,3,3,3,3,2,2],
#  [2,2,2,2,2,2,2,2,2,2],
#  [2,2,2,2,2,2,2,2,2,2]]
#
# So the output is 10x10. The input diagonal values are: 2 (size 3), 3 (size 1), 0 (size 1).
# Actually the diagonal values from top-left corner outward define "layers":
# Layer 0 (innermost): color at (0,0) = 2, size 3x3
# Layer 1: color 3, size extends to 4x4 (adds 1 row/col of 3)
# Layer 2: color 0, size extends to 5x5 (adds 1 row/col of 0)
#
# In the output, the 0 region is replaced with the first color (2).
# The pattern recursively expands.
#
# I think the rule is: the input encodes a self-similar structure.
# The 0s in the input get filled with the top-left color.
# The output doubles the size, creating a fractal-like expansion.
#
# Let me think about this differently based on the examples.
#
# Example 3: Input [[2,3,4,1,6],[3,3,4,1,6],[4,4,4,1,6],[1,1,1,1,6],[6,6,6,6,6]]
# This is a lower-triangular pattern where each "step" of the diagonal uses a different color.
# Colors on diagonal: 2,3,4,1,6 (each 1x1 step)
# Output 10x10: doubles it, and each layer wraps around.
#
# The output seems to be: make a 10x10 grid where position (r,c) is determined by
# the "layer" = max number of complete "input pattern widths" that (r,c) is from the bottom-right.
#
# Actually, for each cell (r,c) in the output, we look at which "diagonal band" it falls in.
# The diagonal band is determined by the minimum of how far from the right edge and bottom edge.
# No wait...
#
# Let me look at example 3 output more carefully:
# Row 0: 2 3 4 1 6 2 3 4 1 6
# Row 5: 2 2 2 2 2 2 3 4 1 6
# Row 9: 6 6 6 6 6 6 6 6 6 6
#
# For cell (r,c): the value is determined by the diagonal from bottom-right.
# distance = min(9-r, 9-c)? No...
#
# (0,0)=2: dist from BR = min(9,9)=9
# (0,4)=6: dist from BR = min(9,5)=5
# (0,5)=2: dist from BR = min(9,4)=4
# (5,0)=2: dist from BR = min(4,9)=4
# (9,9)=6: dist = 0
# (8,8)=1: dist = 1 -> maps to 1? In input diagonal pos 3 = 1.
# Actually: (9,9)=6 -> input[4]=6, (8,8)=1 -> input[3]=1, (8,9)=6 -> input[?]
#
# Hmm. Let me think of it as: the output[r][c] depends on max(r,c) somehow.
# No. Let me look at the structure.
#
# Input encodes layers from top-left. In input:
# cell (r,c) = input[max(r,c)] when r != c in the lower-triangular sense.
# Actually input[r][c] = diagonal_color[max(r,c)] for the lower-triangular part.
#
# For the output 10x10, it seems like we're extending this pattern.
# Let me check: output[r][c] should be the color of layer max(... something).
#
# Looking at example 1:
# Input diagonal: positions 0-2 -> color 2, position 3 -> color 3, position 4 -> color 0 (but 0 means "fill with base")
#
# In output: each (r,c) gets a color based on how far from bottom-right:
# dist_r = 9-r, dist_c = 9-c, layer = min(dist_r, dist_c)
# layer 0: bottom-right 2 rows/cols -> (8-9, 8-9) -> color 2 (the innermost/first color)
# layer 1 (next 2): color 2
# layer 2 (next 3): color 2
# Hmm that doesn't work cleanly.
#
# Let me try: the input defines a recursive structure.
# Looking at the diagonals of the input:
# input[i][j] value depends on max(i,j), which gives the "level".
#
# For example 3 input, levels are 0,1,2,3,4 with colors 2,3,4,1,6.
# For the output (10x10), we double the size.
#
# Level for output cell (r,c):
# In example 3, output[5][0] = 2. That's at position where layer from bottom-right = min(4,9)=4.
# What maps to 2? In the input, level 0 = 2.
#
# output[0][0] = 2. Layer from BR = min(9,9)=9. Maps to level 0 = 2.
# output[0][4] = 6. Layer from BR = min(9,5)=5. Maps to level 4 = 6.
# output[5][5] = 2. Layer from BR = min(4,4)=4. Maps to level 0 = 2. But 4 should map to...
#
# Wait: 9 -> 2(level 0), 5 -> 6(level 4), 4 -> 2(level 0)
# It seems like it wraps: 9 mod 5 = 4 -> input level 4 = 6? But output is 2.
#
# Let me try a different approach. The output is the input pattern tiled/expanded:
# For each 0 in the input, we substitute a copy of the full pattern.
# But the input has 0s only in the last row and column.
#
# Actually, let me look at each example output as a 2x2 arrangement of 5x5 blocks:
# Example 1 output (10x10):
# Top-left 5x5: [[2,2,2,3,2],[2,2,2,3,2],[2,2,2,3,2],[3,3,3,3,2],[2,2,2,2,2]]
# Top-right 5x5: [[2,2,3,2,2],[2,2,3,2,2],[2,2,3,2,2],[2,2,3,2,2],[2,2,3,2,2]]
# Wait that doesn't split nicely.
#
# Let me just look at it as: each row i and col j in output maps to input[i % 5][j % 5]
# but with 0 replaced... No.
#
# OK let me try yet another approach. Looking at all 3 examples:
#
# Example 1: n=5. input diagonal colors by level (max(r,c)):
# 0->2, 1->2, 2->2, 3->3, 4->0
# So there are blocks: [2,2,2] then [3] then [0].
# Sizes: 3, 1, 1.
#
# Output is 10x10. The color at (r,c) = color of level min(r, c) in a
# rotated/expanded version... hmm.
#
# Let me just try: output[r][c] = input_color[min(9-r, 9-c)] where
# input_color maps distances to colors based on the input diagonal.
#
# For example 3: input diagonal = [2,3,4,1,6] (levels 0-4).
# Output: dist_from_BR = min(9-r, 9-c)
# dist 0 -> row/col 9 -> output should be 6 = input[4]. ✓ (output[9][9]=6, output[9][0]=6)
# dist 1 -> row/col 8 -> output should be 1 = input[3]. Check output[8][8]=1. But output[8][0]=1?
# output row 8: [1,1,1,1,1,1,1,1,1,6]. output[8][0]=1. ✓
# dist 4 -> min(9-r,9-c)=4, e.g., (5,5): output=2=input[0].
# But input[4]=6 and input[0]=2. So dist 4 maps to input[0]?
# That means dist d maps to input[(9-d) mod 5]? No, 9-4=5 mod 5=0 -> input[0]=2. ✓
# dist 0: (9-0) mod 5 = 4 -> input[4]=6 ✓
# dist 1: (9-1) mod 5 = 3 -> input[3]=1 ✓
# dist 5: e.g. (4,4): output=6. (9-5) mod 5 = 4 -> input[4]=6 ✓
# dist 9: (0,0): output=2. (9-9) mod 5 = 0 -> input[0]=2 ✓
#
# So for example 3: output[r][c] = input_diag[(n_out - 1 - min(n_out-1-r, n_out-1-c)) % n_in]
# = input_diag[max(r,c) % n_in]?
# Wait: min(9-r, 9-c) = 9 - max(r,c). So dist = 9 - max(r,c).
# Then: input_diag[(9 - dist) % 5] = input_diag[max(r,c) % 5].
#
# So output[r][c] = input_diag[max(r,c) % n] where n is input size!
#
# Let me verify with example 1:
# input_diag = [2,2,2,3,0]. But 0 means...
# output[r][c] = input_diag[max(r,c) % 5]
# (0,0): max=0, 0%5=0, input_diag[0]=2 ✓
# (0,4): max=4, 4%5=4, input_diag[4]=0. But output=6? No wait,
# example 1 output[0][4]=2. And input_diag[4]=0. That's wrong.
#
# Hmm. In example 1, the 0 in the input diagonal is at level 4.
# But the output doesn't have any 0s - 0s are replaced by the first color.
#
# So maybe: output[r][c] = input_diag[max(r,c) % n], but 0 is replaced by the innermost color?
# input_diag for ex1 = [2,2,2,3,0] -> replace 0 with 2 -> [2,2,2,3,2]
# (0,4): max=4, 4%5=4 -> 2 ✓
# (3,3): max=3, 3%5=3 -> 3 ✓
# (7,7): max=7, 7%5=2 -> 2 ✓
# (7,3): max=7, 7%5=2 -> 2. output[7][3]=3. ✗
#
# That doesn't work for all cells. The issue is that the input isn't just about
# the diagonal - the off-diagonal cells also encode information.
#
# Hmm. Let me think about this more carefully.
# In the input, cell (r,c) has a value. For the output, cell (R,C) maps to
# input[R % n][C % n], but with 0 replaced.
#
# Example 1: output[7][3] should be 3.
# 7%5=2, 3%5=3. input[2][3]=3. ✓!
# output[0][4]: 0%5=0, 4%5=4. input[0][4]=0 -> replace with 2. ✓!
# output[4][0]: 4%5=4, 0%5=0. input[4][0]=0 -> replace with 2. ✓!
# output[5][5]: 5%5=0, 5%5=0. input[0][0]=2. ✓!
# output[8][8]: 8%5=3, 8%5=3. input[3][3]=3. But output[8][8]=2.
# Hmm wait, let me check output[8][8] for example 1.
# output row 8: [2,2,2,2,2,2,2,2,2,2]. So output[8][8]=2.
# input[3][3]=3. That gives 3, not 2. ✗
#
# So simple modular tiling with 0-replacement doesn't work.
#
# Let me look at this problem from a fractal perspective.
# In example 3 (simplest - each diagonal entry is unique):
# Input: [[2,3,4,1,6],[3,3,4,1,6],[4,4,4,1,6],[1,1,1,1,6],[6,6,6,6,6]]
# This is: cell(r,c) = diagonal[max(r,c)] where diagonal = [2,3,4,1,6]
#
# Output:
# Row 0: 2 3 4 1 6 2 3 4 1 6 -> max(0,c): c for c=0..9
# But output[0][5]=2 and max(0,5)=5. diagonal[5]? No, diagonal only goes to 4.
# 5 maps to... the pattern wraps! 5->2(=diagonal[0]), 6->3(=diagonal[1]),...
# So output[r][c] = diagonal[max(r,c) % 5]?
# output[5][0]: max=5, 5%5=0 -> 2. ✓
# output[5][5]: max=5, 5%5=0 -> 2. ✓
# output[6][5]: max=6, 6%5=1 -> 3. output row 6: [3,3,3,3,3,3,3,4,1,6]. output[6][5]=3. ✓
# output[8][8]: max=8, 8%5=3 -> 1. output row 8: [1,1,1,1,1,1,1,1,1,6]. output[8][8]=1. ✓
# output[9][9]: max=9, 9%5=4 -> 6. ✓
# output[9][0]: max=9, 9%5=4 -> 6. ✓
#
# ✓✓✓ This works for example 3!
#
# But example 1 has non-unique diagonal entries and 0s.
# Input diagonal: [2,2,2,3,0]. Input: cell(r,c)=diagonal[max(r,c)]
# But 0 entries...
#
# For example 1 output, output[r][c] = diagonal[max(r,c) % 5], replacing 0 with...
# Let me check: for max(r,c)=4 (mod 5=4), diagonal[4]=0.
# output[4][0]=2, output[0][4]=2. So 0->2 (first color).
# max=8: 8%5=3 -> diagonal[3]=3. output[8][8]=2. But that should be 3!
#
# Hmm, for example 1 output row 8: 2,2,2,2,2,2,2,2,2,2 (all 2s).
# max(8,c) for c=0..9: 8,8,8,8,8,8,8,8,8,9.
# 8%5=3 -> should be 3 but output=2. Doesn't work!
#
# So the simple max(r,c)%n doesn't work for example 1.
#
# OK, for example 1 the input is NOT simply diagonal[max(r,c)].
# Input: [[2,2,2,3,0],[2,2,2,3,0],[2,2,2,3,0],[3,3,3,3,0],[0,0,0,0,0]]
# This has a 3x3 block of 2s (rows 0-2, cols 0-2), then col 3 is all 3s (rows 0-3),
# row 3 is all 3s (cols 0-3), and the last row/col are 0s.
#
# For example 2:
# Input: [[1,1,4,6,0],[1,1,4,6,0],[4,4,4,6,0],[6,6,6,6,0],[0,0,0,0,0]]
# Block structure: 2x2 of 1s, then 4-column, then 6-row/col, then 0-row/col.
# Diagonal: 1,1,4,6,0. Sizes from top-left: 2 (for 1), 1 (for 4), 1 (for 6), 1 (for 0).
#
# Let me think of the input as encoding "levels" based on position:
# level(r,c) = the color at (r,c) in the input.
# For the output: we tile the input pattern but replace 0 with the appropriate color.
#
# Actually I think the answer is: the output grid has size 2*n x 2*n where n=5 (so 10x10).
# For output cell (R,C), we determine the "level" differently.
#
# The input defines a fractal rule. The 5x5 input has blocks:
# For example 1: blocks at sizes 3,1,1 with colors 2,3,0.
# The 0-block gets replaced with the pattern, recursively.
#
# The output at cell (R,C):
# Step 1: Find the "block" that (R,C) falls in within the 10x10 grid.
# The 10x10 grid is divided the same way as the 5x5 but doubled.
# Wait no, the output is exactly 10x10 = 2*5.
#
# Let me try yet another approach. The output is the input pattern with each 0
# replaced by a "zoomed out" version of the non-zero pattern.
#
# For example 1: non-zero part is 4x4 (rows 0-3, cols 0-3):
# 2 2 2 3
# 2 2 2 3
# 2 2 2 3
# 3 3 3 3
# Zero part is row 4 and col 4.
# Output 10x10: the first 4 rows/cols are the original pattern.
# Then the 0-row (row 4) expands to...
#
# OK I'm going to take a more systematic approach.
# Let me extract the "diagonal structure" from the input.

# From the input, identify the nested rectangles:
# Each level k defines a rectangle from (0,0) to (s_k-1, s_k-1) where s_k is the cumulative size.
# The color at level k fills the L-shaped border of that rectangle.

# For example 1: levels are:
# Level 0: color 2, size 3 (fills 3x3 block)
# Level 1: color 3, size 1 (L-shaped border from 3x3 to 4x4)
# Level 2: color 0, size 1 (L-shaped border from 4x4 to 5x5)

# For the output (10x10), we double everything:
# Starting from output size 10, work inward.
# The outermost level (color 0->fill with base) has... hmm.

# I think the output rule is:
# Given the input nxn grid, the output is 2n x 2n.
# For output cell (R,C):
# - Compute the "effective level" based on the block structure
# - The 0-cells in the input pattern get replaced by the smallest (base) color

# Let me just hardcode based on the structural observation:
# The input has a lower-triangular staircase pattern.
# For the output, we reconstruct this staircase but with doubled dimensions.
#
# For a given (R,C) in the output:
# The "depth" from the bottom-right corner determines the color.
# depth(R,C) = min(2n-1-R, 2n-1-C) for output size 2n.
# Then we map this depth back to the input's color structure.
#
# In the input, the depth from bottom-right determines color:
# depth d maps to: input[n-1-d][n-1-d]... wait no.
# In the input, cell (r,c) = color of max(r,c) level.
# So the "band" is determined by b = max(r,c), and color = input[b][b] (diagonal).
# But that's not right because input[0][0]=input[1][1]=input[2][2]=2 in example 1.
#
# Actually, in example 1: input[r][c] = depends on max(r,c):
# max(r,c)=0 -> 2, =1 -> 2, =2 -> 2, =3 -> 3, =4 -> 0
#
# For the output, maybe:
# depth from bottom-right = min(2n-1-R, 2n-1-C)
# This depth maps to a color based on the input structure.
# But we need to figure out how depths map.
#
# In the input, the depths from bottom-right are:
# For n=5, max depth = n-1 = 4. depth(r,c) = min(4-r, 4-c) = 4-max(r,c).
# depth 4 -> max(r,c)=0 -> color 2
# depth 3 -> max(r,c)=1 -> color 2
# depth 2 -> max(r,c)=2 -> color 2
# depth 1 -> max(r,c)=3 -> color 3
# depth 0 -> max(r,c)=4 -> color 0
#
# For output, max depth = 2n-1 = 9.
# Now the key question: how do output depths map to input depths?
#
# Attempt: output_depth -> input_depth as:
# We have "block sizes" from the input diagonal.
# In example 1: the diagonal is [2,2,2,3,0].
# The distinct levels from inside out (highest depth first):
# Depth 4,3,2 -> color 2 (3 cells wide)
# Depth 1 -> color 3 (1 cell wide)
# Depth 0 -> color 0 (1 cell wide)
# Sizes: [3, 1, 1]
# Colors: [2, 3, 0]
#
# For output: the sizes double. New sizes: [6, 2, 2]? Total should be 10 (=2*5).
# Hmm, 6+2+2=10. That works!
# So output depth 9,8,7,6,5,4 -> color 2 (6 cells)
# Output depth 3,2 -> color 3 (2 cells)
# Output depth 1,0 -> color 0 -> fill with 2 (base color)
#
# Let me verify:
# output[0][0]: depth=min(9,9)=9 -> color 2 ✓
# output[8][8]: depth=min(1,1)=1 -> color 0 -> 2 ✓ (output is 2)
# output[7][3]: depth=min(2,6)=2 -> color 3. output[7][3]=3 ✓
# output[7][7]: depth=min(2,2)=2 -> color 3. output[7][7]=3 ✓
# output[3][3]: depth=min(6,6)=6 -> color 2. But output[3][3]=3.
# 3%5=3 and input[3][3]=3. Hmm.
# depth 6 should map to color 2 in my scheme. But output is 3. ✗
#
# So doubling the sizes doesn't work either.
#
# I need to think about this differently. Let me look at the outputs more carefully.
#
# Example 3 output (which worked with max(r,c)%5):
# output[r][c] = input[max(r,c) % 5][min(r,c) % 5]? No that's overcomplicated.
# Since input[r][c] = diagonal[max(r,c)], and output[r][c] = diagonal[max(r,c) % 5],
# this is equivalent to: output[R][C] = input[max(R,C) % n][max(R,C) % n] (diagonal value).
# But since input values in example 3 only depend on max(r,c), this is also
# output[R][C] = input[R % n][C % n] when both R%n and C%n map correctly...
# Actually no, output[R][C] just depends on max(R,C) % n.
#
# For example 1 output, let's check: is output[R][C] = input[R%5][C%5] with 0->2?
# output[3][3]: input[3][3]=3 -> 3 ✓
# output[7][3]: input[2][3]=3 -> 3 ✓
# output[7][7]: input[2][2]=2 -> 2. But expected output[7][7]=3. ✗ Hmm wait:
# Output row 7: [3,3,3,3,3,3,3,3,2,2]. output[7][7]=3.
# input[2][2]=2 != 3. So input[R%5][C%5] doesn't work.
#
# Hmm. output[7][7]=3. R%5=2, C%5=2. input[2][2]=2. Doesn't match.
# What about input[max(R,C)%5][min(R,C)%5]? max=7, min=7. Both %5=2. input[2][2]=2. Still no.
#
# Let me look at all of row 7 of example 1 output: [3,3,3,3,3,3,3,3,2,2]
# These are: 3,3,3,3,3,3,3,3,2,2. The 3s go from col 0-7, then 2s at 8-9.
# In terms of "what determines this":
# Row 7, col 7: the "level" from bottom-right is min(2,2)=2. From my earlier attempt that gave color 3.
# But with doubled sizes [6,2,2] it gave depth 2 -> color 3. Let me recheck my depth mapping.
#
# Sizes [3,1,1] -> doubled [6,2,2]. Cumulative from outside:
# 0-1 (depth 0-1): size 2, color 0 -> base 2
# 2-3 (depth 2-3): size 2, color 3
# 4-9 (depth 4-9): size 6, color 2
#
# output[7][7]: depth = min(9-7, 9-7) = 2 -> color 3 ✓!
# output[3][3]: depth = min(9-3, 9-3) = 6 -> color 2 ✓!
# output[7][3]: depth = min(9-7, 9-3) = min(2,6) = 2 -> color 3 ✓!
# output[8][8]: depth = 1 -> color 0 -> base 2 ✓!
#
# Let me verify more:
# output[3][7]: depth = min(6,2) = 2 -> color 3. Output row 3: [2,2,2,3,2,2,2,3,2,2].
# output[3][7] = 3. ✓!
# output[4][4]: depth = min(5,5) = 5 -> color 2. Output row 4: [2,2,2,2,2,2,2,3,2,2].
# output[4][4] = 2. ✓!
# output[4][7]: depth = min(5,2) = 2 -> color 3. output[4][7] = 3. ✓!
#
# Let me now verify example 2:
# Input diagonal: [1,1,4,6,0].
# Sizes from inside out: 2 (for 1), 1 (for 4), 1 (for 6), 1 (for 0).
# Wait, the cumulative sizes are 2,3,4,5.
# So level sizes: 2,1,1,1. Colors: 1,4,6,0.
# Doubled sizes: 4,2,2,2. Total = 10. ✓
#
# Depth mapping (from outside, depth 0):
# depth 0-1: size 2, color 0 -> base 1
# depth 2-3: size 2, color 6
# depth 4-5: size 2, color 4
# depth 6-9: size 4, color 1
#
# output[0][0]: depth=9 -> color 1 ✓
# output[9][9]: depth=0 -> color 0 -> 1 ✓
# output[2][2]: depth=min(7,7)=7 -> color 1. Output row 2: [4,4,4,6,1,1,4,6,1,1].
# output[2][2] = 4. But depth 7 -> color 1. ✗!
#
# Hmm. Depth 7 maps to color 1 in my scheme (size 4 block, depths 6-9).
# But output[2][2] = 4.
# depth of (2,2) = min(9-2, 9-2) = 7. My mapping says color 1, but actual is 4.
#
# Something's wrong. Let me re-examine.
#
# Maybe I should not double the sizes but instead just tile?
#
# Example 2 output:
# Row 0: 1 1 4 6 1 1 4 6 1 1
# Row 2: 4 4 4 6 1 1 4 6 1 1
# Row 6: 4 4 4 4 4 4 4 6 1 1
# Row 7: 6 6 6 6 6 6 6 6 1 1
# Row 9: 1 1 1 1 1 1 1 1 1 1
#
# So: output[r][c] in example 2:
# (0,0)=1, (0,2)=4, (0,3)=6, (0,4)=1, (0,6)=4, (0,7)=6, (0,8)=1
# (2,0)=4, (2,3)=6, (2,4)=1, (2,6)=4
# (6,0)=4, (6,6)=4, (6,7)=6
# (7,0)=6, (7,7)=6, (7,8)=1
# (9,0)=1, (9,9)=1
#
# Pattern: output[r][c] depends on max(r,c). Let's check:
# max(0,0)=0 -> 1, max(0,2)=2 -> 4, max(0,3)=3 -> 6, max(0,4)=4 -> 1
# max(2,0)=2 -> 4, max(6,6)=6 -> 4, max(7,7)=7 -> 6
# max(9,9)=9 -> 1
#
# So: 0->1, 2->4, 3->6, 4->1, 6->4, 7->6, 9->1
# The pattern repeats with period... 0->1, 1->1, 2->4, 3->6, 4->1, 5->1, 6->4, 7->6, 8->1, 9->1
#
# Check max(0,1)=1: output[0][1]=1 ✓ (1->1)
# max(8,0)=8: output[8][0]=1 ✓ (8->1)
#
# So the mapping by max(r,c): 0->1, 1->1, 2->4, 3->6, 4->1, 5->1, 6->4, 7->6, 8->1, 9->1
#
# This is the input diagonal [1,1,4,6] repeating! Period 4 (not 5)!
# 0%4=0->1, 1%4=1->1, 2%4=2->4, 3%4=3->6, 4%4=0->1, 5%4=1->1, 6%4=2->4, 7%4=3->6, 8%4=0->1, 9%4=1->1
# ✓✓✓!
#
# So the period is n-1 (=4 for 5x5 input), not n!
# And we use the first n-1 diagonal values (excluding the last one which is 0).
#
# Let me verify example 1:
# Input diagonal: [2,2,2,3,0]. First 4: [2,2,2,3]. Period=4.
# output[r][c] = diagonal[max(r,c) % 4]
# (7,7): max=7, 7%4=3 -> 3 ✓
# (3,3): max=3, 3%4=3 -> 3 ✓
# (4,4): max=4, 4%4=0 -> 2 ✓
# (0,4): max=4, 4%4=0 -> 2 ✓
# (8,8): max=8, 8%4=0 -> 2 ✓
# ✓✓✓!
#
# Verify example 3:
# Input diagonal: [2,3,4,1,6]. Period = 4 (first 4): [2,3,4,1].
# (0,4): max=4, 4%4=0 -> 2. But output[0][4]=6. ✗!
#
# Hmm, example 3 has no 0 in the last diagonal position (it's 6, not 0).
# In example 3, the last diagonal is 6 (non-zero), so the "period" is 5.
#
# Wait, but examples 1 and 2 have 0 in the last position, and example 3 doesn't.
# In example 3 ALL cells are non-zero. There's no recursive expansion needed.
# The output just tiles with period n=5.
#
# For examples 1 and 2, the last row/col is 0. Period = n-1 = 4 (excluding the 0 layer).
# For example 3, all non-zero. Period = n = 5.
#
# Actually wait, I need to reconsider. In examples 1 and 2, the 0 fills are
# the outermost layer. In example 3, the outermost layer is 6 (non-zero).
#
# So the rule is:
# 1. Extract the diagonal colors from the input.
# 2. Remove trailing 0s to get the "core" diagonal colors.
# 3. The output is 2n x 2n (where n = input size).
# 4. output[r][c] = core_diagonal[max(r,c) % len(core_diagonal)]
#
# Wait but n_out = 2*n for examples 1,2 and also for example 3.
# Actually ALL outputs are 10x10 for 5x5 inputs. So n_out = 2*n.
#
# With this rule:
# Example 3: core_diagonal = [2,3,4,1,6] (length 5, no trailing 0s).
# output[0][4] = core[4%5] = core[4] = 6 ✓
# output[4][0] = core[4%5] = 6. Output row 4: [6,6,6,6,6,2,3,4,1,6]. output[4][0]=6 ✓
# output[5][5] = core[5%5] = core[0] = 2 ✓
# ✓✓✓!
#
# This works! Final rule:
# - Get diagonal from input
# - Strip trailing 0s
# - Output is 2n x 2n
# - output[r][c] = stripped_diagonal[max(r,c) % len(stripped_diagonal)]

def solve_539a4f51(grid):
    n = len(grid)
    # Get diagonal
    diag = [grid[i][i] for i in range(n)]
    # Strip trailing 0s
    while diag and diag[-1] == 0:
        diag.pop()
    period = len(diag)
    out_size = 2 * n
    out = [[0]*out_size for _ in range(out_size)]
    for r in range(out_size):
        for c in range(out_size):
            out[r][c] = diag[max(r, c) % period]
    return out

# ============================================================
# 53b68214: Extend/tile a pattern to fill a 10-row grid
# Input is a pattern (some rows with colored cells) on a 10-wide grid with fewer rows than 10.
# Output is 10x10, extending the pattern by repeating it.
#
# Example 1: Input 6 rows, output 10 rows. The pattern continues.
# The staircase pattern: 1s form a staircase going right and down.
# Input: rows 0-5, pattern repeats in rows 6-9.
#
# Example 2: Input 5 rows of vertical line at col 2. Output 10 rows, same line extended.
#
# Example 3: Input 8 rows with zigzag pattern (period seems to be ~3 or 5 rows).
# Output extends to 10 rows continuing the pattern.
#
# The pattern: the input has h rows and 10 cols. Output is 10x10.
# The extra rows continue the periodic pattern of the input.
#
# For example 3, input is 8 rows. The pattern has period 3 (rows 0-2 is one cycle,
# rows 3-5 another, rows 6-7 is partial). Output adds 2 more rows (8-9) to complete.
#
# Actually, looking more carefully:
# Example 1: pattern is a staircase that shifts right by 2 and down by 2 each step.
# Input rows: 3 pairs of rows, each pair shifts right. Output continues for 2 more pairs.
#
# Example 3: zigzag period appears to be 3 rows. Input 8 rows = 2 full periods + 2 extra.
# Output 10 rows = 2 full periods + 1 period + 1 remaining. Hmm, 10/3 is not clean.
#
# Actually the period for example 3 is 5 rows:
# Row 0: 0 2 0 ...
# Row 1: 0 2 0 ...
# Row 2: 2 0 2 ...
# Row 3: 0 2 0 ...
# Row 4: 0 2 0 ...
# Row 5: 2 0 2 ... (same as row 2)
# So period = 3 (rows 0,1,2 repeat as 3,4,5 and 6,7).
# Output rows 8,9 = continuation: row 8 like row 2 (8%3=2), row 9 like row 0 (9%3=0).
#
# Output row 8: [2,0,2,0,...] and row 9: [0,2,0,...].
# Actual output row 8: [2,0,2,0,0,0,0,0,0,0] and row 9: [0,2,0,0,...].
# ✓!
#
# So the rule is: find the period of the input rows, then extend to 10 rows.
#
# But wait, example 1 has a translating pattern, not a simple row-period.
# Rows 0-1: cols 0-2 have 1s
# Rows 2-3: cols 2-4 have 1s
# Rows 4-5: cols 4-6 have 1s
# Each block shifts right by 2. The "period" in the row pattern is not simple repetition
# of the same rows - it's a progression.
#
# Hmm. Let me look at this differently. Maybe the pattern is just:
# repeat the last N rows cyclically to fill 10 rows.
# But that wouldn't capture the translation.
#
# Actually for example 1:
# The output rows 6-9 continue the staircase. The non-zero cells shift by +2 each 2 rows.
# So the rule might be: extrapolate the pattern, understanding the "velocity" of features.
#
# Alternatively: find the minimal repeating unit considering both temporal and spatial shift.
#
# A simpler approach: find the period p such that row[i] relates to row[i-p] by some shift.
# For example 1, p=2 and shift = (+2 cols). So row[6] = row[4] shifted right by 2.
# For example 2, p=1 (or any, since all rows are identical).
# For example 3, p=3 and shift = 0.
#
# Actually, for example 1: what if we just look at the difference between consecutive
# non-zero cell positions? The pattern moves right by 2 every 2 rows, and
# down by 2 every 2 rows. If we keep extending, we get the output.
#
# Let me think about this more generally. The key observation is:
# The input is always shorter than 10 rows (height < 10, width = 10).
# The output is always 10x10.
# We need to find how to extend the rows.
#
# Simple approach: find period p of the row pattern, extend.
# For shifting patterns, look at differences.
#
# Let me try: find the smallest p such that for all rows i and i+p that both exist,
# there's a consistent column shift s (possibly 0).
# Then extend by applying shift s every p rows.

def solve_53b68214(grid):
    h = len(grid)
    w = len(grid[0])
    target_h = w  # output is always 10x10 (width x width)

    if h >= target_h:
        return grid[:target_h]

    # Find period and shift
    best_p = None
    best_s = None

    for p in range(1, h):
        # Check if all rows i and i+p have consistent shift
        shifts = []
        valid = True
        for i in range(h - p):
            row_a = grid[i]
            row_b = grid[i + p]
            # Find shift between row_a and row_b
            # Find non-zero positions
            nz_a = [c for c in range(w) if row_a[c] != 0]
            nz_b = [c for c in range(w) if row_b[c] != 0]
            if len(nz_a) == 0 and len(nz_b) == 0:
                shifts.append(0)
                continue
            if len(nz_a) != len(nz_b):
                valid = False
                break
            if len(nz_a) == 0:
                shifts.append(0)
                continue
            s = nz_b[0] - nz_a[0]
            # Verify all positions shift by s
            if all(nz_b[j] - nz_a[j] == s for j in range(len(nz_a))):
                # Also verify colors match
                if all(row_a[nz_a[j]] == row_b[nz_b[j]] for j in range(len(nz_a))):
                    shifts.append(s)
                else:
                    valid = False
                    break
            else:
                valid = False
                break

        if valid and len(set(shifts)) <= 1:
            best_p = p
            best_s = shifts[0] if shifts else 0
            break

    if best_p is None:
        best_p = h
        best_s = 0

    # Extend to target_h rows
    out = [row[:] for row in grid]
    for i in range(h, target_h):
        # Row i = row (i - best_p) shifted by best_s
        src = i - best_p
        new_row = [0] * w
        for c in range(w):
            src_c = c - best_s
            if 0 <= src_c < w:
                new_row[c] = out[src][src_c]
        out.append(new_row)

    return out

# ============================================================
# 543a7ed5: Rectangles on 8-background with 6-borders.
# Add a 3-border around each rectangle, and fill interior of hollow rectangles with 4.
#
# The rectangles are made of 6 cells on an 8 background.
# Some rectangles are solid 6-blocks, some are hollow (6 border with 8 interior).
# Output: each rectangle gets a 3-colored border (1 cell thick) around it.
# Hollow rectangles get their interior filled with 4.

def solve_543a7ed5(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # Find connected components of 6 cells
    visited = [[False]*w for _ in range(h)]

    def flood_fill(sr, sc):
        cells = []
        stack = [(sr, sc)]
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= h or c < 0 or c >= w:
                continue
            if visited[r][c] or grid[r][c] != 6:
                continue
            visited[r][c] = True
            cells.append((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((r+dr, c+dc))
        return cells

    rectangles = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 6 and not visited[r][c]:
                cells = flood_fill(r, c)
                rectangles.append(cells)

    for cells in rectangles:
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)

        # Check if hollow (interior has 8 cells)
        has_interior = False
        for r in range(min_r + 1, max_r):
            for c in range(min_c + 1, max_c):
                if grid[r][c] == 8:
                    has_interior = True
                    break
            if has_interior:
                break

        # Fill interior with 4 if hollow
        if has_interior:
            for r in range(min_r + 1, max_r):
                for c in range(min_c + 1, max_c):
                    if grid[r][c] == 8:
                        out[r][c] = 4

        # Add 3-border around the rectangle
        for r in range(min_r - 1, max_r + 2):
            for c in range(min_c - 1, max_c + 2):
                if 0 <= r < h and 0 <= c < w:
                    if out[r][c] == 8:
                        # Check if adjacent to the rectangle bbox
                        if r == min_r - 1 or r == max_r + 1 or c == min_c - 1 or c == max_c + 1:
                            out[r][c] = 3

    return out

# ============================================================
# 54d82841: U-shaped patterns that "drop" a marker (color 4) at the opening.
# Each U-shape (open-bottom shape like ⊓) has an opening at the bottom.
# A color 4 marker is placed at the bottom center of each U.
#
# Looking at examples:
# Example 1: Two U-shapes (open at bottom):
# Shape 1 at rows 0-1, cols 1-3: [[6,6,6],[6,0,6]] - opens downward
# Shape 2 at rows 2-3, cols 5-7: [[6,6,6],[6,0,6]] - opens downward
# Output: same but row 7 (last row) has 4s at cols 2 and 6.
# The 4 appears at the bottom of the grid, in the column of the opening.
#
# It's like gravity - the opening of the U-shape drops a ball (4) that falls
# to the bottom of the grid.
#
# Actually looking more carefully:
# Shape 1: [[0,6,6,6,0,...],[0,6,0,6,0,...]] at rows 0-1. Opening at (1,2).
# Shape 2: [[...,6,6,6,...],[...,6,0,6,...]] at rows 2-3. Opening at (3,6).
# Output adds 4 at (7,2) and (7,6) (bottom row, column of opening).
#
# Example 2: [[0,8,8,8,0,...],[0,8,0,8,6,6,6],[0,0,0,0,6,0,6],[...]]
# Shape 1: 8-colored, rows 0-1, cols 1-3. Opening at (1,2).
# Shape 2: 6-colored, rows 1-2, cols 4-6. Opening at (2,5) or (3,5).
# Actually (1,4)=6,(1,5)=6,(1,6)=6, (2,4)=6,(2,5)=0,(2,6)=6. Opening at (2,5).
# Output: (4,2)=4 and (4,5)=4.
#
# Wait, the output is the same grid but with 4s at the last row in the opening columns.
# And the 4 is placed in the very last row of the grid.
#
# Actually let me re-examine. The output has the SAME grid dimensions.
# The 4 appears on the BOTTOM row (last row) at the column of each U-shape's opening.
#
# The "opening" of a U-shape is the gap inside the U (the cell that's 0 between the arms).

def solve_54d82841(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # Find U-shapes: look for patterns like [X,X,X] on top and [X,0,X] below
    # (or rotated versions)
    # The shapes are ⊓-shaped (3 cells on top, 2 cells below with gap)

    # Find all non-zero colored shapes
    visited = [[False]*w for _ in range(h)]

    def flood_fill(sr, sc, color):
        cells = []
        stack = [(sr, sc)]
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= h or c < 0 or c >= w:
                continue
            if visited[r][c] or grid[r][c] != color:
                continue
            visited[r][c] = True
            cells.append((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((r+dr, c+dc))
        return cells

    shapes = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                cells = flood_fill(r, c, color)
                shapes.append((color, cells))

    # For each shape, find the "opening" - the 0-cell that's enclosed on 3 sides
    for color, cells in shapes:
        cell_set = set(cells)
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)

        # Check for opening: find 0-cells within/adjacent to bbox that have 3 colored neighbors
        # Actually the U-shape has a specific gap. Let me find it.
        # The opening is a cell within the bbox that is 0 and surrounded by shape cells on 3 sides.
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if (r, c) not in cell_set and grid[r][c] == 0:
                    # Count adjacent shape cells
                    adj = 0
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r + dr, c + dc
                        if (nr, nc) in cell_set:
                            adj += 1
                    if adj >= 3:
                        # This is the opening - place 4 at bottom of grid in this column
                        out[h-1][c] = 4

    return out

# ============================================================
# 54d9e175: Grid with 5-separators dividing into cells. Each cell has a center color.
# The center color (1,2,3,4) maps to a fill color (6,7,8,9) = center + 5.
# Each section gets filled with the mapped color.

def solve_54d9e175(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # The grid is divided by 5-separators into sections
    # Find separator rows and columns
    sep_rows = []
    sep_cols = []

    for r in range(h):
        if all(grid[r][c] == 5 for c in range(w)):
            sep_rows.append(r)

    for c in range(w):
        if all(grid[r][c] == 5 for r in range(h)):
            sep_cols.append(c)

    # Define sections as regions between separators
    row_ranges = []
    prev = 0
    for sr in sep_rows:
        if sr > prev:
            row_ranges.append((prev, sr))
        prev = sr + 1
    if prev < h:
        row_ranges.append((prev, h))

    col_ranges = []
    prev = 0
    for sc in sep_cols:
        if sc > prev:
            col_ranges.append((prev, sc))
        prev = sc + 1
    if prev < w:
        col_ranges.append((prev, w))

    # For each section, find the center color and fill with color + 5
    for r_start, r_end in row_ranges:
        for c_start, c_end in col_ranges:
            # Find the non-zero, non-5 color in this section
            center_color = 0
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    if grid[r][c] != 0 and grid[r][c] != 5:
                        center_color = grid[r][c]
                        break
                if center_color != 0:
                    break

            if center_color != 0:
                fill_color = center_color + 5
                for r in range(r_start, r_end):
                    for c in range(c_start, c_end):
                        if grid[r][c] != 5:
                            out[r][c] = fill_color

    return out

# ============================================================
# Now test and save all solutions
# ============================================================

task_ids = [
    '4c5c2cf0', '508bd3b6', '50cb2852', '5117e062', '5168d44c',
    '539a4f51', '53b68214', '543a7ed5', '54d82841', '54d9e175'
]

solvers = {
    '4c5c2cf0': solve_4c5c2cf0,
    '508bd3b6': solve_508bd3b6,
    '50cb2852': solve_50cb2852,
    '5117e062': solve_5117e062,
    '5168d44c': solve_5168d44c,
    '539a4f51': solve_539a4f51,
    '53b68214': solve_53b68214,
    '543a7ed5': solve_543a7ed5,
    '54d82841': solve_54d82841,
    '54d9e175': solve_54d9e175,
}

import inspect

results = {}
all_pass = True

for task_id in task_ids:
    with open(f'data/arc1/{task_id}.json') as f:
        task = json.load(f)

    solver = solvers[task_id]
    passed = True

    for i, pair in enumerate(task['train']):
        inp = pair['input']
        expected = pair['output']
        try:
            got = solver(inp)
            if got != expected:
                passed = False
                print(f"FAIL {task_id} train[{i}]")
                # Show differences
                for r in range(max(len(got), len(expected))):
                    if r >= len(got):
                        print(f"  Row {r}: MISSING (expected {expected[r]})")
                    elif r >= len(expected):
                        print(f"  Row {r}: EXTRA {got[r]}")
                    elif got[r] != expected[r]:
                        print(f"  Row {r}: got {got[r]}")
                        print(f"       exp {expected[r]}")
        except Exception as e:
            passed = False
            print(f"ERROR {task_id} train[{i}]: {e}")
            import traceback
            traceback.print_exc()

    if passed:
        print(f"PASS {task_id}")
        source = inspect.getsource(solver)
        results[task_id] = source
    else:
        all_pass = False

# Save results
with open('data/arc_python_solutions_b7.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)}/{len(task_ids)} solutions")
