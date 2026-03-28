import json
import copy

def solve_d43fd935(grid):
    """3x3 block has 2 colors on edges. Each color draws a line from block edge to nearest same-color pixel."""
    g = [row[:] for row in grid]
    R, C = len(g), len(g[0])
    # Find 3x3 block position
    br, bc = -1, -1
    for r in range(R-1):
        for c in range(C-1):
            if g[r][c] == 3 and g[r][c+1] == 3 and g[r+1][c] == 3 and g[r+1][c+1] == 3:
                br, bc = r, c
                break
    # Find non-zero non-3 pixels
    pixels = []
    for r in range(R):
        for c in range(C):
            if g[r][c] != 0 and g[r][c] != 3:
                pixels.append((r, c, g[r][c]))
    # For each color, find pixels and determine which edge of block to extend from
    # The block occupies br,bc to br+1,bc+1
    # Top edge: row br, cols bc to bc+1
    # Bottom edge: row br+1, cols bc to bc+1
    # Left edge: rows br to br+1, col bc
    # Right edge: rows br to br+1, col bc+1

    # Group pixels by color
    from collections import defaultdict
    color_pixels = defaultdict(list)
    for r, c, v in pixels:
        color_pixels[v].append((r, c))

    for color, plist in color_pixels.items():
        for pr, pc in plist:
            # Determine direction from block center to pixel
            block_center_r = br + 0.5
            block_center_c = bc + 0.5

            # Check if pixel is roughly above/below (same column range) or left/right
            # Try to connect: find nearest block edge cell that aligns

            # If pixel is in same column as block left or right col
            if pc == bc or pc == bc + 1:
                # vertical connection
                col = pc
                if pr < br:  # above
                    for rr in range(pr+1, br):
                        g[rr][col] = color
                elif pr > br + 1:  # below
                    for rr in range(br+2, pr):
                        g[rr][col] = color
            elif pr == br or pr == br + 1:
                # horizontal connection
                row = pr
                if pc < bc:  # left
                    for cc in range(pc+1, bc):
                        g[row][cc] = color
                elif pc > bc + 1:  # right
                    for cc in range(bc+2, pc):
                        g[row][cc] = color
            else:
                # Need L-shaped path: go to nearest block edge
                # Determine which side of block is closest
                # Try top/bottom edge sharing column, or left/right edge sharing row
                pass

    # Actually, let me re-analyze: each non-3 non-0 color draws a line from block edge
    # In train[0]: color 1 at (0,0), (3,8), (9,4). Output shows line from (3,4) to (3,8) with 1s (right of block on row 3)
    # And color 6 at (1,8), (6,7), (7,6), (8,2). Output shows column of 6s at col 2 from row 5 to row 8
    # So: for each color, lines extend from the block edges
    # The top-left of block is at (3,2).
    # Row 3 (top row of block), from col 4 (bc+2) to col 8: filled with 1
    # Col 2 (left col of block), from row 5 (br+2) to row 8: filled with 6

    # Re-examining: The block is at rows 3-4, cols 2-3
    # The edge cells of the block connect outward in their respective directions
    # Top-right corner (3,3) -> extends right: fills (3,4) to (3,8) with 1 (matching pixel at (3,8))
    # Bottom-left corner (4,2) -> extends down: fills (5,2) to (8,2) with 6 (matching pixel at (8,2))

    # Actually let me think more carefully about the rule.
    # For each isolated pixel of a given color, draw a line from the nearest block edge toward that pixel.
    # The line goes from the block edge (not the pixel) toward the pixel, filling intermediate cells.

    # Let me re-examine train[0]:
    # Block at (3,2)-(4,3)
    # Color 1 pixels: (0,0), (3,8), (9,4)
    # Output: row 3 gets 1s from col 4 to col 8 (connecting block row 3 right side to pixel (3,8))
    # (0,0) and (9,4) stay as-is, no line drawn to them

    # Color 6 pixels: (1,8), (6,7), (7,6), (8,2)
    # Output: col 2 gets 6s from row 5 to row 8 (connecting block col 2 bottom to pixel (8,2))
    # Other 6 pixels stay as-is

    # So only ONE pixel per color gets connected? The one that's aligned with a block edge?
    # (3,8) is on row 3 which is block's top row -> extends right
    # (8,2) is on col 2 which is block's left col -> extends down

    # Rule: for each non-3 colored pixel that shares a row or column with a block edge,
    # draw a line from the block edge to that pixel

    g = [row[:] for row in grid]  # reset

    for color, plist in color_pixels.items():
        for pr, pc in plist:
            # Check if pixel shares row with block rows or column with block cols
            if pr == br or pr == br + 1:
                # Same row as block
                row = pr
                if pc > bc + 1:
                    for cc in range(bc + 2, pc):
                        g[row][cc] = color
                elif pc < bc:
                    for cc in range(pc + 1, bc):
                        g[row][cc] = color
            elif pc == bc or pc == bc + 1:
                # Same column as block
                col = pc
                if pr > br + 1:
                    for rr in range(br + 2, pr):
                        g[rr][col] = color
                elif pr < br:
                    for rr in range(pr + 1, br):
                        g[rr][col] = color

    return g

def solve_d4469b4b(grid):
    """5x5 grid with single color. Count pixels in each row/column quadrant to determine cross shape."""
    # Looking at the pattern: color 1 -> plus/cross, color 2 -> T-up, color 3 -> L-bottom-right
    # Actually let me count: which 3x3 output shape maps to which input color?
    # Color 1: plus shape (0,5,0/5,5,5/0,5,0)
    # Color 2: T-up shape (5,5,5/0,5,0/0,5,0)
    # Color 3: L-bottom (0,0,5/0,0,5/5,5,5)

    # Count non-zero in each quadrant of 5x5
    # Let me think about what determines the shape...
    # train[0]: color 2, 10 pixels -> T-up (5,5,5/0,5,0/0,5,0)
    # train[1]: color 1, 7 pixels -> plus (0,5,0/5,5,5/0,5,0)
    # train[2]: color 3, 9 pixels -> L-bottom-right (0,0,5/0,0,5/5,5,5)
    # train[3]: color 1, 10 pixels -> plus
    # train[4]: color 2, 14 pixels -> T-up
    # train[5]: color 2, 13 pixels -> T-up
    # train[6]: color 3, 8 pixels -> L-bottom-right

    # It's just based on the color!
    # Color 1 -> plus, Color 2 -> T-up, Color 3 -> L-bottom-right

    color = 0
    for row in grid:
        for v in row:
            if v != 0:
                color = v
                break
        if color: break

    if color == 1:
        return [[0,5,0],[5,5,5],[0,5,0]]
    elif color == 2:
        return [[5,5,5],[0,5,0],[0,5,0]]
    elif color == 3:
        return [[0,0,5],[0,0,5],[5,5,5]]
    return [[0,0,0],[0,0,0],[0,0,0]]

def solve_d4a91cb9(grid):
    """Two points (8 and 2). Draw L-shaped path with 4s from 8 to 2."""
    g = [row[:] for row in grid]
    R, C = len(g), len(g[0])
    # Find 8 and 2
    r8, c8, r2, c2 = -1, -1, -1, -1
    for r in range(R):
        for c in range(C):
            if g[r][c] == 8:
                r8, c8 = r, c
            elif g[r][c] == 2:
                r2, c2 = r, c

    # Draw L-shape: go vertically from 8 then horizontally to 2
    # Column of 8, go toward row of 2
    if r8 < r2:
        for rr in range(r8 + 1, r2):
            g[rr][c8] = 4
        # Then horizontal on row of 2
        if c8 < c2:
            for cc in range(c8, c2):
                g[r2][cc] = 4
        else:
            for cc in range(c2 + 1, c8 + 1):
                g[r2][cc] = 4
    else:
        for rr in range(r2 + 1, r8):
            g[rr][c8] = 4
        if c8 < c2:
            for cc in range(c8, c2):
                g[r2][cc] = 4
        else:
            for cc in range(c2 + 1, c8 + 1):
                g[r2][cc] = 4

    return g

def solve_d4f3cd78(grid):
    """Rectangle of 5s with one gap. Fill interior with 8, extend 8 through gap to edge."""
    g = [row[:] for row in grid]
    R, C = len(g), len(g[0])

    # Find bounding box of 5s
    min_r, max_r, min_c, max_c = R, 0, C, 0
    for r in range(R):
        for c in range(C):
            if g[r][c] == 5:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    # Find the gap in the border
    gap_r, gap_c = -1, -1
    # Check top edge
    for c in range(min_c, max_c + 1):
        if g[min_r][c] == 0:
            gap_r, gap_c = min_r, c
    # Check bottom edge
    for c in range(min_c, max_c + 1):
        if g[max_r][c] == 0:
            gap_r, gap_c = max_r, c
    # Check left edge
    for r in range(min_r, max_r + 1):
        if g[r][min_c] == 0:
            gap_r, gap_c = r, min_c
    # Check right edge
    for r in range(min_r, max_r + 1):
        if g[r][max_c] == 0:
            gap_r, gap_c = r, max_c

    # Fill interior with 8
    for r in range(min_r + 1, max_r):
        for c in range(min_c + 1, max_c):
            if g[r][c] == 0:
                g[r][c] = 8

    # Extend 8 through gap
    # Replace gap with 8
    g[gap_r][gap_c] = 8

    # Determine direction of gap and extend
    if gap_r == min_r:  # top edge gap -> extend upward
        for r in range(0, min_r):
            g[r][gap_c] = 8
    elif gap_r == max_r:  # bottom edge gap -> extend downward
        for r in range(max_r + 1, R):
            g[r][gap_c] = 8
    elif gap_c == min_c:  # left edge gap -> extend leftward
        for c in range(0, min_c):
            g[gap_r][c] = 8
    elif gap_c == max_c:  # right edge gap -> extend rightward
        for c in range(max_c + 1, C):
            g[gap_r][c] = 8

    return g

def solve_d5d6de2d(grid):
    """Rectangles of 2s. Output: fill interior of each rectangle with 3, remove border of 2s.
    Small rectangles (like 2x2) just disappear."""
    g = [[0]*len(grid[0]) for _ in range(len(grid))]
    R, C = len(grid), len(grid[0])

    # Find connected components of 2s
    visited = [[False]*C for _ in range(R)]

    def flood_fill(r, c):
        stack = [(r, c)]
        cells = []
        while stack:
            rr, cc = stack.pop()
            if rr < 0 or rr >= R or cc < 0 or cc >= C:
                continue
            if visited[rr][cc] or grid[rr][cc] != 2:
                continue
            visited[rr][cc] = True
            cells.append((rr, cc))
            stack.extend([(rr+1,cc),(rr-1,cc),(rr,cc+1),(rr,cc-1)])
        return cells

    for r in range(R):
        for c in range(C):
            if grid[r][c] == 2 and not visited[r][c]:
                cells = flood_fill(r, c)
                # Find bounding box
                rows = [x[0] for x in cells]
                cols = [x[1] for x in cells]
                min_r, max_r = min(rows), max(rows)
                min_c, max_c = min(cols), max(cols)

                # Interior = everything inside the bounding box excluding border
                h = max_r - min_r + 1
                w = max_c - min_c + 1

                # Check if it's a proper rectangle border (has interior)
                inner_h = h - 2
                inner_w = w - 2

                if inner_h > 0 and inner_w > 0:
                    # Fill interior with 3
                    for rr in range(min_r + 1, max_r):
                        for cc in range(min_c + 1, max_c):
                            g[rr][cc] = 3
                # If no interior (like 2x2 block), output nothing (stays 0)

    return g

def solve_d631b094(grid):
    """3x3 grid. Count non-zero cells, output 1-row with that many cells of that color."""
    count = 0
    color = 0
    for row in grid:
        for v in row:
            if v != 0:
                count += 1
                color = v
    return [[color] * count]

def solve_d687bc17(grid):
    """Rectangle border with 4 different colors on edges. Interior colored pixels move to nearest border edge."""
    g = [row[:] for row in grid]
    R, C = len(g), len(g[0])

    # Find the border colors and positions
    # Top row, bottom row, left col, right col form the border
    # Corners are 0
    # Top edge color
    top_color = grid[0][1]
    bottom_color = grid[R-1][1]
    left_color = grid[1][0]
    right_color = grid[1][C-1]

    # Interior is rows 1..R-2, cols 1..C-2
    # Find interior colored pixels
    interior_pixels = []
    for r in range(1, R-1):
        for c in range(1, C-1):
            if grid[r][c] != 0:
                interior_pixels.append((r, c, grid[r][c]))

    # Clear interior
    for r in range(1, R-1):
        for c in range(1, C-1):
            g[r][c] = 0

    # For each interior pixel, move it to the nearest wall of matching color
    for r, c, color in interior_pixels:
        if color == top_color:
            # Move to top wall: place at (1, c) - adjacent to top border
            g[1][c] = top_color
        elif color == bottom_color:
            # Move to bottom wall: place at (R-2, c)
            g[R-2][c] = bottom_color
        elif color == left_color:
            # Move to left wall: place at (r, 1) - adjacent to left border
            g[r][1] = left_color
        elif color == right_color:
            # Move to right wall: place at (r, C-2)
            g[r][C-2] = right_color

    return g

def solve_d6ad076f(grid):
    """Two rectangles. Fill the gap between them with 8 at intersection of their interior ranges."""
    g = [row[:] for row in grid]
    R, C = len(g), len(g[0])

    from collections import defaultdict
    color_cells = defaultdict(list)
    for r in range(R):
        for c in range(C):
            if g[r][c] != 0:
                color_cells[g[r][c]].append((r, c))

    colors = list(color_cells.keys())
    if len(colors) != 2:
        return g

    def bbox(cells):
        rows = [c[0] for c in cells]
        cols = [c[1] for c in cells]
        return min(rows), max(rows), min(cols), max(cols)

    bb1 = bbox(color_cells[colors[0]])
    bb2 = bbox(color_cells[colors[1]])

    r1min, r1max, c1min, c1max = bb1
    r2min, r2max, c2min, c2max = bb2

    # Interior ranges (exclude borders of each rectangle)
    int1_rmin, int1_rmax = r1min + 1, r1max - 1
    int1_cmin, int1_cmax = c1min + 1, c1max - 1
    int2_rmin, int2_rmax = r2min + 1, r2max - 1
    int2_cmin, int2_cmax = c2min + 1, c2max - 1

    # Intersection of interior row ranges
    int_rmin = max(int1_rmin, int2_rmin)
    int_rmax = min(int1_rmax, int2_rmax)
    # Intersection of interior col ranges
    int_cmin = max(int1_cmin, int2_cmin)
    int_cmax = min(int1_cmax, int2_cmax)

    # Determine gap direction and fill
    # If rects are separated vertically (gap in rows)
    if r1max < r2min or r2max < r1min:
        # Vertical gap
        gap_rmin = min(r1max, r2max) + 1
        gap_rmax = max(r1min, r2min) - 1
        for r in range(gap_rmin, gap_rmax + 1):
            for c in range(int_cmin, int_cmax + 1):
                g[r][c] = 8
    # If rects are separated horizontally (gap in cols)
    elif c1max < c2min or c2max < c1min:
        # Horizontal gap
        gap_cmin = min(c1max, c2max) + 1
        gap_cmax = max(c1min, c2min) - 1
        for r in range(int_rmin, int_rmax + 1):
            for c in range(gap_cmin, gap_cmax + 1):
                g[r][c] = 8

    return g

def solve_d89b689b(grid):
    """4 colored pixels at corners (relative to 2x2 block of 8s). Map each to its quadrant of the 8 block."""
    g = [[0]*len(grid[0]) for _ in range(len(grid))]
    R, C = len(g), len(g[0])

    # Find 8-block
    br, bc = -1, -1
    for r in range(R-1):
        for c in range(C-1):
            if grid[r][c] == 8 and grid[r][c+1] == 8 and grid[r+1][c] == 8 and grid[r+1][c+1] == 8:
                br, bc = r, c
                break

    # Find the 4 non-8 non-0 pixels
    pixels = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] != 0 and grid[r][c] != 8:
                pixels.append((r, c, grid[r][c]))

    # Map each pixel to quadrant: top-left, top-right, bottom-left, bottom-right
    # relative to 8-block center
    center_r = br + 0.5
    center_c = bc + 0.5

    for r, c, color in pixels:
        if r < center_r and c < center_c:
            g[br][bc] = color  # top-left
        elif r < center_r and c > center_c:
            g[br][bc+1] = color  # top-right
        elif r > center_r and c < center_c:
            g[br+1][bc] = color  # bottom-left
        elif r > center_r and c > center_c:
            g[br+1][bc+1] = color  # bottom-right

    return g

def solve_d8c310e9(grid):
    """Pattern in bottom rows extends rightward to fill the 15-wide grid."""
    g = [row[:] for row in grid]
    R, C = len(g), len(g[0])

    # Find the bottom row (most filled) to determine the period
    bottom_row = None
    for r in range(R - 1, -1, -1):
        if any(grid[r][c] != 0 for c in range(C)):
            bottom_row = r
            break

    if bottom_row is None:
        return g

    # Get the non-zero portion of the bottom row
    br = grid[bottom_row]
    rmax = max(c for c in range(C) if br[c] != 0)
    existing = br[:rmax + 1]

    # Find the period: smallest p such that all non-zero values match modulo p
    # and the base (first p values) are all non-zero
    best_period = len(existing)
    for p in range(1, len(existing)):
        base = existing[:p]
        # Base must be all non-zero
        if any(v == 0 for v in base):
            continue
        valid = True
        for i in range(p, len(existing)):
            if existing[i] != 0 and existing[i] != base[i % p]:
                valid = False
                break
            if existing[i] == 0 and base[i % p] != 0:
                # Check if the zero is expected (upper rows may have zeros in pattern)
                pass  # Allow zeros in upper rows
        if valid:
            best_period = p
            break

    base = existing[:best_period]

    # Extend all pattern rows using this period
    for r in range(R):
        if not any(grid[r][c] != 0 for c in range(C)):
            continue
        row = grid[r]
        row_base = row[:best_period]
        for c in range(C):
            g[r][c] = row_base[c % best_period]

    return g

def solve_d90796e8(grid):
    """Where 3 and 2 are adjacent (including diagonal?), replace both with 8. Keep 5 and isolated 3,2."""
    # Actually looking more carefully:
    # When 3 and 2 are adjacent (4-connected), replace with 8 and 0.
    # train[0]: (0,0)=3, (0,1)=2 -> (0,0)=8, (0,1)=0
    # train[1]: (1,2)=3, (1,3)=2 -> (1,2)=8, (1,3)=0
    #           (3,1)=3, (4,1)=2 -> (3,1)=8, (4,1)=0
    #           Actually: train[1] output (3,1)=8 and (4,1)=0
    # So when 3 and 2 are adjacent, replace the pair with 8 (at 3's position) and 0 (at 2's position)

    g = [row[:] for row in grid]
    R, C = len(g), len(g[0])

    # Find adjacent 3-2 pairs
    pairs = []
    used = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 3:
                # Check 4-connected neighbors for 2
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 2:
                        if (r,c) not in used and (nr,nc) not in used:
                            pairs.append(((r,c),(nr,nc)))
                            used.add((r,c))
                            used.add((nr,nc))

    for (r3,c3), (r2,c2) in pairs:
        g[r3][c3] = 8
        g[r2][c2] = 0

    return g

def solve_d9f24cd1(grid):
    """Bottom row 2s mark wall positions. Walls go up. 5 on wall column deflects right with L-turn."""
    g = [[0]*len(grid[0]) for _ in range(len(grid))]
    R, C = len(g), len(g[0])

    bottom_2s = [c for c in range(C) if grid[R-1][c] == 2]

    fives = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 5:
                fives.add((r, c))
                g[r][c] = 5

    for wc in bottom_2s:
        col = wc
        r = R - 1
        while r >= 0:
            if (r, col) in fives:
                if r + 1 < R:
                    g[r + 1][col + 1] = 2
                col = col + 1
                g[r][col] = 2
                r -= 1
            else:
                g[r][col] = 2
                r -= 1

    return g

def _old_solve_d9f24cd1(grid):
    """UNUSED"""
    # For each 2 in bottom row, extend upward
    # The wall goes up from bottom. When it reaches a row with a 5, the 5 determines
    # the new column position.

    # Looking at train[0]: bottom 2s at cols 1, 4, 6
    # 5 at (3,6) and (5,2)
    # Output: col 1 wall goes straight up (no 5 in that column region)
    #         col 4 wall goes straight up
    #         col 6 wall: goes up from row 9 to row 4, then at row 3 the 5 is at (3,6) so wall shifts to col 7 from row 3 up
    # Wait, output col 6: rows 4-9 have 2 at col 6, row 3 has 5 at col 6 and 2 at col 7
    # So: wall goes up, when it hits a 5, the 5 stays and the wall continues from col+1

    # Actually: looking at output for train[0]:
    # col 1: 2 at rows 0-9 (all) -> no 5 affects it
    # col 4: 2 at rows 0-9 -> no, output row 0 col 4 is 2... but 5 at (5,2)
    # Wait let me reread.

    # train[0] output:
    # Row 0: [0,2,0,0,2,0,0,2,0,0] -> 2s at cols 1,4,7
    # Row 9: [0,2,0,0,2,0,2,0,0,0] -> 2s at cols 1,4,6
    # Bottom row (input): 2s at cols 1,4,6

    # So col 6 wall becomes col 7 at some point
    # 5 at (3,6): row 3, col 6
    # At row 3: the 5 is at col 6, and the wall shifts from col 6 to col 7
    # Below row 3 (rows 4-9): 2 at col 6
    # Above row 3 (rows 0-2): 2 at col 7

    # 5 at (5,2): row 5, col 2
    # But there's no wall near col 2... the wall at col 1 is nearby
    # At row 5: 5 at col 2, wall at col 1 shifts to col 2 at row 5
    # Below row 5 (rows 6-9): 2 at col 1
    # At row 5: 5 at col 2, 2 at col 2 (shifted)
    # Above row 5 (rows 0-4): 2 at col 2? No...
    # Output row 5: [0,2,5,0,2,0,2,2,0,0] -> wait the output is wrong

    # Let me re-read: the output for train[0]:
    # [0,2,0,0,2,0,0,2,0,0]  row 0
    # [0,2,0,0,2,0,0,2,0,0]  row 1
    # [0,2,0,0,2,0,0,2,0,0]  row 2
    # [0,2,0,0,2,0,5,2,0,0]  row 3 - 5 at col 6, 2 at col 7
    # [0,2,0,0,2,0,2,2,0,0]  row 4 - 2 at cols 1,4,6,7
    # Hmm row 4 has 2 at both col 6 and 7?

    # Actually I need to re-read the actual data more carefully. Let me just implement from scratch.

    # Reset and implement based on the pattern I see
    g = [[0]*C for _ in range(R)]

    # Place 5s
    for r, c in fives:
        g[r][c] = 5

    # For each wall (2 in bottom row), trace upward
    # Each wall starts at bottom row and goes up
    # When it reaches a 5, the wall column shifts to the 5's column
    # Then continues upward from there

    # Actually looking more carefully:
    # Each 5 is associated with the nearest wall (by column proximity in bottom row)
    # The 5 "pulls" the wall to its column at that row

    # For each bottom-2 column, find any 5 that affects it
    # A 5 at (r5, c5) affects the wall whose bottom column is nearest

    # Let's assign each 5 to the nearest wall
    five_to_wall = {}
    for r5, c5 in fives:
        best_wall = min(bottom_2s, key=lambda wc: abs(wc - c5))
        five_to_wall[(r5, c5)] = best_wall

    # For each wall, build path from bottom to top
    for wc in bottom_2s:
        # Find 5s affecting this wall
        my_fives = [(r5, c5) for (r5, c5), w in five_to_wall.items() if w == wc]
        my_fives.sort(key=lambda x: -x[0])  # sort by row descending (bottom to top)

        col = wc
        prev_shift_row = R  # start from bottom

        for r5, c5 in my_fives:
            # Draw wall from prev_shift_row-1 down to r5+1 at current col
            for r in range(r5 + 1, prev_shift_row):
                if r < R:
                    g[r][col] = 2
            # At row r5, the 5 stays, wall shifts to col+1 in direction of 5
            col = c5 + (1 if c5 >= col else -1) if c5 != col else col + 1
            prev_shift_row = r5

        # Draw remaining wall from prev_shift_row-1 to row 0
        for r in range(0, prev_shift_row):
            g[r][col] = 2

    return g

def solve_d9fac9be(grid):
    """Find the color inside the bordered box. The box is made of one color,
    and contains a different color inside. Output that contained color."""
    R, C = len(grid), len(grid[0])

    # Find rectangular frames (3x3 blocks of one color with a different color in center)
    from collections import defaultdict

    # Look for a bordered region
    for r in range(R - 2):
        for c in range(C - 2):
            # Check for 3x3 frame
            border_color = grid[r][c]
            if border_color == 0:
                continue
            # Check all border cells
            border_cells = [(r,c),(r,c+1),(r,c+2),(r+1,c),(r+1,c+2),(r+2,c),(r+2,c+1),(r+2,c+2)]
            if all(grid[br][bc] == border_color for br, bc in border_cells):
                center = grid[r+1][c+1]
                if center != 0 and center != border_color:
                    return [[center]]

    # Try larger frames
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 0:
                continue
            border_color = grid[r][c]
            # Try to find a rectangle of this color
            # Find extent
            for r2 in range(r + 2, R):
                for c2 in range(c + 2, C):
                    # Check if (r,c)-(r2,c2) forms a frame of border_color
                    valid = True
                    # Top and bottom edges
                    for cc in range(c, c2 + 1):
                        if grid[r][cc] != border_color or grid[r2][cc] != border_color:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Left and right edges
                    for rr in range(r, r2 + 1):
                        if grid[rr][c] != border_color or grid[rr][c2] != border_color:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Check interior has a different non-zero color
                    for rr in range(r + 1, r2):
                        for cc in range(c + 1, c2):
                            v = grid[rr][cc]
                            if v != 0 and v != border_color:
                                return [[v]]

    return [[0]]

def solve_dae9d2b5(grid):
    """3x6 grid split into left 3x3 (color 4) and right 3x3 (color 3).
    Output: OR of the two halves, using color 6."""
    R = len(grid)
    C = len(grid[0]) // 2
    result = [[0]*C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if grid[r][c] != 0 or grid[r][c + C] != 0:
                result[r][c] = 6
    return result

def solve_db3e9e38(grid):
    """Column of 7s creates a triangle/diamond pattern extending outward."""
    g = [[0]*len(grid[0]) for _ in range(len(grid))]
    R, C = len(g), len(g[0])

    # Find the column of 7s
    col_7 = -1
    seven_rows = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 7:
                col_7 = c
                seven_rows.append(r)

    if not seven_rows:
        return g

    # The 7s form a vertical line. The bottom of the 7s is the apex.
    # Triangle extends upward from bottom 7, alternating 7 and 8

    top_7 = min(seven_rows)
    bottom_7 = max(seven_rows)
    height = bottom_7 - top_7 + 1  # number of 7s

    # The pattern: from bottom_7, each row going up adds one more cell on each side
    # At bottom_7: just the 7
    # At bottom_7 - 1: 8,7,8 (width 3 centered on col_7)... wait

    # Let me check train[0]: col 3, rows 0-3 (4 sevens), bottom at row 3
    # Output:
    # Row 0: 8,7,8,7,8,7,8  -> 7 cells centered on col 3
    # Row 1: 0,7,8,7,8,7,0  -> 5 cells centered on col 3
    # Row 2: 0,0,8,7,8,0,0  -> 3 cells centered on col 3
    # Row 3: 0,0,0,7,0,0,0  -> just the 7

    # So from bottom (row 3) upward:
    # Row 3: width 1 (just 7)
    # Row 2: width 3 (8,7,8)
    # Row 1: width 5 (7,8,7,8,7)
    # Row 0: width 7 (8,7,8,7,8,7,8)

    # Pattern: at distance d from bottom (d=0 is bottom_7):
    # Width = 2*d + 1, centered on col_7
    # Values alternate: center is 7 if d is even... wait
    # d=0: 7
    # d=1: 8,7,8 (center 7)
    # d=2: 7,8,7,8,7 (center 7)
    # d=3: 8,7,8,7,8,7,8 (center 8? no, center is 7)

    # Wait row 0 center (col 3) = 7. So center is always 7.
    # From center outward at each row: 7, then alternating 8,7,8,7...
    # But d=0: just 7
    # d=1: 8,7,8 -> from center: 7, left=8, right=8
    # d=2: 7,8,7,8,7 -> from center: 7, left1=8, right1=8, left2=7, right2=7
    # d=3: 8,7,8,7,8,7,8 -> from center: 7, left1=8, right1=8, left2=7, right2=7, left3=8, right3=8
    # Wait that doesn't match. Row 0 = [8,7,8,7,8,7,8]. Center col 3 = 7.
    # Positions: col0=8, col1=7, col2=8, col3=7, col4=8, col5=7, col6=8
    # So at distance k from center: if k even -> 7, if k odd -> 8
    # That matches all rows. And the width at distance d from bottom is 2*d+1.

    for d in range(height):
        r = bottom_7 - d
        for k in range(-d, d+1):
            c = col_7 + k
            if 0 <= c < C:
                if abs(k) % 2 == 0:
                    g[r][c] = 7
                else:
                    g[r][c] = 8

    return g

def solve_db93a21d(grid):
    """Multiple 9-rectangles. Each gets a 3-border (thickness = min(H,W)//2).
    1-lines extend outward from each border in the rect's row/col range."""
    R, C = len(grid), len(grid[0])
    g = [[0]*C for _ in range(R)]

    # Find rectangles of 9s
    visited = [[False]*C for _ in range(R)]
    rects = []

    for r in range(R):
        for c in range(C):
            if grid[r][c] == 9 and not visited[r][c]:
                stack = [(r, c)]
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    if rr < 0 or rr >= R or cc < 0 or cc >= C:
                        continue
                    if visited[rr][cc] or grid[rr][cc] != 9:
                        continue
                    visited[rr][cc] = True
                    cells.append((rr, cc))
                    stack.extend([(rr+1,cc),(rr-1,cc),(rr,cc+1),(rr,cc-1)])

                rows_list = [x[0] for x in cells]
                cols_list = [x[1] for x in cells]
                rects.append((min(rows_list), max(rows_list), min(cols_list), max(cols_list)))

    # For each rect, compute border thickness and draw 1-lines, then 3-border, then 9s
    # Priority: 9 > 3 > 1 > 0

    # Step 1: Draw 1-lines (lowest priority, drawn first)
    # 1-lines extend ONLY downward from each border bottom, in rect's col range
    for rmin, rmax, cmin, cmax in rects:
        h = rmax - rmin + 1
        w = cmax - cmin + 1
        t = max(h, w) // 2

        border_rmax = rmax + t

        for r in range(min(R, border_rmax + 1), R):
            for c in range(cmin, cmax + 1):
                if 0 <= c < C:
                    g[r][c] = 1

    # Step 2: Draw 3-borders (overwrite 1s)
    for rmin, rmax, cmin, cmax in rects:
        h = rmax - rmin + 1
        w = cmax - cmin + 1
        t = max(h, w) // 2

        border_rmin = rmin - t
        border_rmax = rmax + t
        border_cmin = cmin - t
        border_cmax = cmax + t

        for r in range(max(0, border_rmin), min(R, border_rmax + 1)):
            for c in range(max(0, border_cmin), min(C, border_cmax + 1)):
                # Only fill 3 if not inside the 9-rect
                if not (rmin <= r <= rmax and cmin <= c <= cmax):
                    g[r][c] = 3

    # Step 3: Draw 9-rects (highest priority)
    for rmin, rmax, cmin, cmax in rects:
        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                g[r][c] = 9

    return g

def solve_dc433765(grid):
    """3 (green) and 4 (yellow) pixels. Move 3 one step closer to 4. 4 stays fixed.
    If same column, move 3 one step toward 4 vertically. If same row, move toward horizontally.
    Otherwise move diagonally (both closer)."""
    g = [[0]*len(grid[0]) for _ in range(len(grid))]
    R, C = len(g), len(g[0])

    # Find 3 and 4
    r3, c3, r4, c4 = -1, -1, -1, -1
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 3:
                r3, c3 = r, c
            elif grid[r][c] == 4:
                r4, c4 = r, c

    # Move 3 one step closer to 4
    dr = 0 if r3 == r4 else (1 if r4 > r3 else -1)
    dc = 0 if c3 == c4 else (1 if c4 > c3 else -1)

    g[r3 + dr][c3 + dc] = 3
    g[r4][c4] = 4

    return g

def solve_ddf7fa4f(grid):
    """Top row has colored markers. Rectangles of 5 below get colored based on nearest top marker."""
    g = [row[:] for row in grid]
    R, C = len(g), len(g[0])

    # Find markers in top row
    markers = []
    for c in range(C):
        if grid[0][c] != 0:
            markers.append((c, grid[0][c]))

    # Find rectangles of 5
    visited = [[False]*C for _ in range(R)]

    for r in range(R):
        for c in range(C):
            if grid[r][c] == 5 and not visited[r][c]:
                # BFS
                stack = [(r, c)]
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    if rr < 0 or rr >= R or cc < 0 or cc >= C:
                        continue
                    if visited[rr][cc] or grid[rr][cc] != 5:
                        continue
                    visited[rr][cc] = True
                    cells.append((rr, cc))
                    stack.extend([(rr+1,cc),(rr-1,cc),(rr,cc+1),(rr,cc-1)])

                # Find center of rectangle
                rows_list = [x[0] for x in cells]
                cols_list = [x[1] for x in cells]
                center_c = (min(cols_list) + max(cols_list)) / 2

                # Find nearest marker
                best_marker = min(markers, key=lambda m: abs(m[0] - center_c))
                color = best_marker[1]

                # Color the rectangle
                for rr, cc in cells:
                    g[rr][cc] = color

    return g

def solve_e179c5f4(grid):
    """Grid of zeros with 1 in bottom-left. Fill with 8, place 1 bouncing diagonally."""
    R, C = len(grid), len(grid[0])
    g = [[8]*C for _ in range(R)]

    # 1 starts at bottom-left (R-1, 0) and bounces diagonally
    # Direction: initially going up. In width=2: bounces between col 0 and 1
    # Width 2: pattern is col 0,1,0,1... going up from bottom, alternating
    # Width 3: 0,1,2,1,0,1,2... (bouncing)
    # Width 4: 0,1,2,3,2,1,0,1,2,3...
    # Width 5: 0,1,2,3,4,3,2,1,0,1...

    # Generate bounce pattern for columns
    if C == 1:
        for r in range(R):
            g[r][0] = 1
        return g

    # Create bounce sequence
    period = 2 * (C - 1)
    bounce = list(range(C)) + list(range(C-2, 0, -1))

    # Starting from bottom row (R-1) at position 0 in bounce
    for r in range(R):
        row_from_bottom = R - 1 - r
        col = bounce[row_from_bottom % period]
        g[r][col] = 1

    return g

def solve_e21d9049(grid):
    """Cross pattern: center pixel with arms extending in 4 directions.
    The arm patterns repeat along the cross."""
    g = [[0]*len(grid[0]) for _ in range(len(grid))]
    R, C = len(g), len(g[0])

    # Find the cross center and arms
    # The cross has a center pixel and arms in 4 directions
    # Find the non-zero cluster
    nz = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] != 0:
                nz.append((r, c, grid[r][c]))

    if not nz:
        return g

    # Find the center (intersection of horizontal and vertical arms)
    rows = [x[0] for x in nz]
    cols = [x[1] for x in nz]

    # The center is at the intersection of the row with multiple pixels and the column with multiple pixels
    from collections import Counter
    row_counts = Counter(rows)
    col_counts = Counter(cols)

    # The horizontal arm row and vertical arm column
    h_row = max(row_counts, key=row_counts.get)
    v_col = max(col_counts, key=col_counts.get)

    center_val = grid[h_row][v_col]

    # Get horizontal arm sequence (left to right from center)
    h_arm = []
    for c in sorted(set(cols)):
        if grid[h_row][c] != 0:
            h_arm.append((c, grid[h_row][c]))

    # Get vertical arm sequence (top to bottom from center)
    v_arm = []
    for r in sorted(set(rows)):
        if grid[r][v_col] != 0:
            v_arm.append((r, grid[r][v_col]))

    # The horizontal arm defines a repeating sequence
    # Extract the sequence of colors going right from center
    h_right = []
    for c, val in h_arm:
        if c >= v_col:
            h_right.append(val)

    # The pattern repeats: center, right1, right2, ...
    # For left: it reverses (or mirrors)

    # Looking at train[0]: center at (4,2), value 2
    # H arm: (4,0)=8, (4,1)=3, (4,2)=2
    # So left arm is 8,3 and center is 2
    # H-right sequence from center: [2]
    # H-left: (4,1)=3, (4,0)=8

    # Output row 4: 8,3,2,8,3,2,8,3,2,8,3
    # So the repeating unit is [8,3,2] going rightward (reversed left arm + center)
    # Period = 3 (length of the full arm including center)

    # V arm: (2,2)=8, (3,2)=3, (4,2)=2
    # Up arm from center: 3,8 (going up)
    # Output col 2: row 0=3, row 1=2, row 2=8, row 3=3, row 4=2, row 5=8, row 6=3, row 7=2, row 8=8, row 9=3, row 10=2, row 11=8
    # From center (row 4) going up: 2(r4), 3(r3), 8(r2), 3(r1)... wait that's cycling [2,3,8]
    # From center going down: 2(r4), 8(r5), 3(r6), 2(r7), 8(r8), 3(r9), 2(r10), 8(r11)
    # Wait: going down from center the sequence is [2, 8, 3, 2, 8, 3...]
    # Going up: [2, 3, 8, 2, 3, 8...] wait no: r4=2, r3=3, r2=8, r1=2, r0=3
    # Up sequence: 2, 3, 8, 2, 3 -> repeating [2, 3, 8]
    # Down sequence: 2, 8, 3, 2, 8, 3, 2, 8 -> repeating [2, 8, 3]
    # So up = [center, arm_up_1, arm_up_2, ...] and down = [center, arm_down_1, ...]
    # arm_up = [3, 8] (the vertical arm colors going up from center)
    # arm_down = reversed arm_up? [8, 3]

    # The up-arm from center through the defined pixels gives the sequence.
    # This sequence repeats cyclically.

    # Let me extract the arm sequences properly
    # Horizontal: get all pixels in h_row, sorted by column
    h_pixels = []
    for c in range(C):
        if grid[h_row][c] != 0:
            h_pixels.append((c, grid[h_row][c]))
    h_pixels.sort()

    # The horizontal sequence (the pattern defined by the input cross arm)
    # From leftmost to center to rightmost
    h_seq = [v for c, v in h_pixels]
    # The right-going sequence repeats this pattern
    h_period = len(h_seq)

    # Vertical: get all pixels in v_col, sorted by row
    v_pixels = []
    for r in range(R):
        if grid[r][v_col] != 0:
            v_pixels.append((r, grid[r][v_col]))
    v_pixels.sort()
    v_seq = [v for r, v in v_pixels]
    v_period = len(v_seq)

    # Find center index in sequences
    h_center_idx = next(i for i, (c, v) in enumerate(h_pixels) if c == v_col)
    v_center_idx = next(i for i, (r, v) in enumerate(v_pixels) if r == h_row)

    # Fill horizontal line
    for c in range(C):
        # Distance from center
        offset = c - v_col
        # Map to sequence index
        idx = (h_center_idx + offset) % h_period
        g[h_row][c] = h_seq[idx]

    # Fill vertical line
    for r in range(R):
        offset = r - h_row
        idx = (v_center_idx + offset) % v_period
        g[r][v_col] = v_seq[idx]

    return g

# Now let's build and test

solutions = {}
task_solvers = {
    'd43fd935': solve_d43fd935,
    'd4469b4b': solve_d4469b4b,
    'd4a91cb9': solve_d4a91cb9,
    'd4f3cd78': solve_d4f3cd78,
    'd5d6de2d': solve_d5d6de2d,
    'd631b094': solve_d631b094,
    'd687bc17': solve_d687bc17,
    'd6ad076f': solve_d6ad076f,
    'd89b689b': solve_d89b689b,
    'd8c310e9': solve_d8c310e9,
    'd90796e8': solve_d90796e8,
    'd9f24cd1': solve_d9f24cd1,
    'd9fac9be': solve_d9fac9be,
    'dae9d2b5': solve_dae9d2b5,
    'db3e9e38': solve_db3e9e38,
    'db93a21d': solve_db93a21d,
    'dc433765': solve_dc433765,
    'ddf7fa4f': solve_ddf7fa4f,
    'e179c5f4': solve_e179c5f4,
    'e21d9049': solve_e21d9049,
}

results = {}
for task_id, solver in task_solvers.items():
    with open(f'data/arc1/{task_id}.json') as f:
        data = json.load(f)

    all_pass = True
    for split in ['train', 'test']:
        for i, pair in enumerate(data[split]):
            try:
                output = solver(pair['input'])
                expected = pair['output']
                if output != expected:
                    all_pass = False
                    print(f"FAIL {task_id} {split}[{i}]")
                    # Show first difference
                    for r in range(min(len(output), len(expected))):
                        if r < len(output) and r < len(expected):
                            if output[r] != expected[r]:
                                print(f"  Row {r}: got {output[r]}")
                                print(f"  Row {r}: exp {expected[r]}")
                                break
                    if len(output) != len(expected):
                        print(f"  Size mismatch: got {len(output)} rows, exp {len(expected)} rows")
                else:
                    print(f"PASS {task_id} {split}[{i}]")
            except Exception as e:
                all_pass = False
                print(f"ERROR {task_id} {split}[{i}]: {e}")

    results[task_id] = all_pass

    # Generate solutions for test cases
    test_outputs = []
    for pair in data['test']:
        test_outputs.append(solver(pair['input']))
    solutions[task_id] = test_outputs

# Save solutions
with open('data/arc_python_solutions_b22.json', 'w') as f:
    json.dump(solutions, f)

print("\n=== SUMMARY ===")
passed = sum(1 for v in results.values() if v)
print(f"Passed: {passed}/{len(results)}")
for task_id, ok in sorted(results.items()):
    print(f"  {task_id}: {'PASS' if ok else 'FAIL'}")
