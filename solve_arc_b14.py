import json
import copy

solutions = {}

# === 928ad970 ===
# Pattern: There's a small rectangle of color C, and 4 dots of color 5 around it.
# Draw a larger rectangle (using color C) connecting the 4 dots (just inside them).
# The inner rectangle stays, outer rectangle is drawn as border.
def solve_928ad970(grid):
    grid = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])

    # Find the rectangle (non-zero, non-5 cells)
    color = 0
    rect_cells = []
    dot_positions = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                dot_positions.append((r, c))
            elif grid[r][c] != 0:
                color = grid[r][c]
                rect_cells.append((r, c))

    # Find bounding box of dots
    dot_rows = [p[0] for p in dot_positions]
    dot_cols = [p[1] for p in dot_positions]

    # The outer rectangle should be drawn between the dots
    # Looking at examples: the rectangle is drawn 1 cell inside from the dots
    top_dot = min(dot_rows)
    bottom_dot = max(dot_rows)
    left_dot = min(dot_cols)
    right_dot = max(dot_cols)

    # Draw outer rectangle border 1 cell inside from dots
    top = top_dot + 1
    bottom = bottom_dot - 1
    left = left_dot + 1
    right = right_dot - 1

    # Draw top and bottom edges
    for c in range(left, right + 1):
        grid[top][c] = color
        grid[bottom][c] = color
    # Draw left and right edges
    for r in range(top, bottom + 1):
        grid[r][left] = color
        grid[r][right] = color

    # Clear the interior of outer rect (except inner rect)
    for r in range(top + 1, bottom):
        for c in range(left + 1, right):
            if (r, c) not in set(rect_cells):
                grid[r][c] = 0

    return grid

solutions['928ad970'] = solve_928ad970

# === 93b581b8 ===
# 2x2 pattern at some position. Each corner value gets reflected to opposite diagonal corner area.
# The 2x2 block has values [TL, TR, BL, BR].
# Opposite corner (relative to 2x2 center) gets 2x2 fill of the diagonally opposite value.
def solve_93b581b8(grid):
    grid = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])

    # Find the 2x2 block
    br, bc = -1, -1
    for r in range(rows - 1):
        for c in range(cols - 1):
            if grid[r][c] != 0 and grid[r][c+1] != 0 and grid[r+1][c] != 0 and grid[r+1][c+1] != 0:
                br, bc = r, c
                break
        if br >= 0:
            break

    tl = grid[br][bc]
    tr = grid[br][bc+1]
    bl = grid[br+1][bc]
    bri = grid[br+1][bc+1]

    # The corners of the grid (relative to block) get filled with the diagonally opposite value
    # Top-left corner area gets bottom-right value (bri), 2x2
    # Top-right corner area gets bottom-left value (bl), 2x2
    # Bottom-left corner area gets top-right value (tr), 2x2
    # Bottom-right corner area gets top-left value (tl), 2x2

    # Looking at example 1: block at (2,2)-(3,3), values 9,3,7,8
    # Output: top-left (0,0) 2x2 = 8(BR), top-right (0,4) 2x2 = 7(BL),
    #         bottom-left (4,0) 2x2 = 3(TR), bottom-right (4,4) 2x2 = 9(TL)
    # Positions: TL area = (br-2, bc-2), TR area = (br-2, bc+2), BL area = (br+2, bc-2), BR area = (br+2, bc+2)

    # Top-left 2x2: rows [br-2, br-1], cols [bc-2, bc-1] -> value = bri (diagonal opposite)
    positions = [
        (br - 2, bc - 2, bri),   # top-left gets bottom-right
        (br - 2, bc + 2, bl),    # top-right gets bottom-left
        (br + 2, bc - 2, tr),    # bottom-left gets top-right
        (br + 2, bc + 2, tl),    # bottom-right gets top-left
    ]

    for r0, c0, val in positions:
        for dr in range(2):
            for dc in range(2):
                r2, c2 = r0 + dr, c0 + dc
                if 0 <= r2 < rows and 0 <= c2 < cols:
                    grid[r2][c2] = val

    return grid

solutions['93b581b8'] = solve_93b581b8

# === 941d9a10 ===
# Grid divided by lines of 5s into rectangular cells.
# The smallest cell (by area) gets filled with 1, next with 2, next with 3.
# Looking more carefully: cells are sorted by area, smallest=1, next=2, next=3
def solve_941d9a10(grid):
    grid = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])

    # Find horizontal and vertical lines of 5
    h_lines = [r for r in range(rows) if all(grid[r][c] == 5 for c in range(cols))]
    v_lines = [c for c in range(cols) if all(grid[r][c] == 5 for r in range(rows))]

    h_bounds = [-1] + h_lines + [rows]
    v_bounds = [-1] + v_lines + [cols]

    # Build cell grid
    n_cell_rows = len(h_bounds) - 1
    n_cell_cols = len(v_bounds) - 1

    cell_grid = {}
    for i in range(n_cell_rows):
        for j in range(n_cell_cols):
            r1, r2 = h_bounds[i] + 1, h_bounds[i + 1]
            c1, c2 = v_bounds[j] + 1, v_bounds[j + 1]
            if r1 < r2 and c1 < c2:
                cell_grid[(i, j)] = (r1, r2, c1, c2)

    # Color 3 diagonal cells: top-left(0,0)=1, center=2, bottom-right=3
    colored = [
        (0, 0, 1),
        (n_cell_rows // 2, n_cell_cols // 2, 2),
        (n_cell_rows - 1, n_cell_cols - 1, 3),
    ]

    for ci, cj, color in colored:
        if (ci, cj) in cell_grid:
            r1, r2, c1, c2 = cell_grid[(ci, cj)]
            for r in range(r1, r2):
                for c in range(c1, c2):
                    if grid[r][c] == 0:
                        grid[r][c] = color

    return grid

solutions['941d9a10'] = solve_941d9a10

# === 94f9d214 ===
# Two 4x4 grids stacked (top=3s pattern, bottom=1s pattern).
# Output is 4x4 where cell=2 if input has 3 XOR 1 (not both, not neither... wait)
# Looking: output has 2 where neither 3 nor 1. Let me check.
# Ex1: input top [0,0,0,0],[0,3,3,0],[0,0,0,0],[3,0,0,3], bottom [0,0,0,1],[1,0,1,1],[1,1,1,1],[0,1,0,1]
# output: [2,2,2,0],[0,0,0,0],[0,0,0,0],[0,0,2,0]
# Cell (0,0): top=0, bot=0 -> out=2. Cell(0,3): top=0, bot=1 -> out=0. Cell(1,0): top=0, bot=1 -> out=0
# Cell(1,1): top=3, bot=0 -> out=0. So: 2 where both are 0!
def solve_94f9d214(grid):
    # Top half is 3s pattern, bottom half is 1s pattern
    # Output: 4x4, cell=2 where both top and bottom are 0
    h = len(grid) // 2
    w = len(grid[0])
    output = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            top_val = grid[r][c]
            bot_val = grid[r + h][c]
            if top_val == 0 and bot_val == 0:
                output[r][c] = 2
    return output

solutions['94f9d214'] = solve_94f9d214

# === 952a094c ===
# Rectangle bordered by color C with 4 colored dots in corners inside.
# Output: remove dots from inside, place them outside at diagonal positions.
# The inner values are cleared, dots placed at corners outside the rectangle.
# Each corner dot goes to the diagonally opposite outside corner.
def solve_952a094c(grid):
    grid = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])

    # Find the rectangle border
    border_cells = []
    border_color = 0
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != 0:
                border_cells.append((r, c, v))

    # Find the rectangle: find the most common non-zero color
    from collections import Counter
    colors = [v for _, _, v in border_cells]
    color_counts = Counter(colors)
    border_color = color_counts.most_common(1)[0][0]

    # Find rectangle bounds
    border_positions = [(r, c) for r, c, v in border_cells if v == border_color]
    rmin = min(r for r, c in border_positions)
    rmax = max(r for r, c in border_positions)
    cmin = min(c for r, c in border_positions)
    cmax = max(c for r, c in border_positions)

    # Find corner dots inside the rectangle
    inner_dots = {}
    for r, c, v in border_cells:
        if v != border_color and rmin < r < rmax and cmin < c < cmax:
            inner_dots[(r, c)] = v

    # Identify corners: top-left, top-right, bottom-left, bottom-right
    inner_r = [r for r, c in inner_dots]
    inner_c = [c for r, c in inner_dots]

    top_r = min(inner_r)
    bot_r = max(inner_r)
    left_c = min(inner_c)
    right_c = max(inner_c)

    tl = inner_dots.get((top_r, left_c), 0)
    tr = inner_dots.get((top_r, right_c), 0)
    bl = inner_dots.get((bot_r, left_c), 0)
    br = inner_dots.get((bot_r, right_c), 0)

    # Clear inner dots and all interior non-border content
    for r in range(rmin + 1, rmax):
        for c in range(cmin + 1, cmax):
            grid[r][c] = 0

    # Place dots outside at diagonally opposite corners
    # TL inner -> placed at (rmin-1, cmin-1) but as diagonal opposite value
    # Looking at example 1: box at (2,6)x(3,7), inner: (3,4)=4, (3,5)=3, (5,4)=2, (5,5)=6
    # Output: (1,2)=6, (1,7)=2, (7,2)=3, (7,7)=4
    # So TL(4) goes to BR outside, TR(3) goes to BL outside, BL(2) goes to TR outside, BR(6) goes to TL outside
    # That means each corner value goes to the diagonally opposite outside position

    grid[rmin - 1][cmin - 1] = br  # outside TL gets BR value
    grid[rmin - 1][cmax + 1] = bl  # outside TR gets BL value
    grid[rmax + 1][cmin - 1] = tr  # outside BL gets TR value
    grid[rmax + 1][cmax + 1] = tl  # outside BR gets TL value

    return grid

solutions['952a094c'] = solve_952a094c

# === 9565186b ===
# 3x3 grid with various colors. Cells that are NOT the most common color
# and NOT 8 get replaced by 5. Actually let me look again.
# Ex1: [2,2,2],[2,1,8],[2,8,8] -> [2,2,2],[2,5,5],[2,5,5]
# The most common color is 2 (5 cells). Other cells (1,8,8,8) become 5.
# Ex2: [1,1,1],[8,1,3],[8,2,2] -> [1,1,1],[5,1,5],[5,5,5]
# Most common is 1 (4 cells). Others become 5.
# Ex3: [2,2,2],[8,8,2],[2,2,2] -> same but 8,8 become 5,5
# Ex4: [3,3,8],[4,4,4],[8,1,1] -> [5,5,5],[4,4,4],[5,5,5]
# Most common is 4 (3). But 3 has 2, 8 has 2, 1 has 2. All non-4 become 5.
# So: find the most frequent color, keep it, replace everything else with 5.
def solve_9565186b(grid):
    grid = copy.deepcopy(grid)
    from collections import Counter
    flat = [grid[r][c] for r in range(len(grid)) for c in range(len(grid[0]))]
    most_common = Counter(flat).most_common(1)[0][0]
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] != most_common:
                grid[r][c] = 5
    return grid

solutions['9565186b'] = solve_9565186b

# === 95990924 ===
# Each 2x2 block of 5s gets corner markers: 1 at top-left-1, 2 at top-right+1, 3 at bottom-left-1, 4 at bottom-right+1
def solve_95990924(grid):
    grid = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])

    # Find all 2x2 blocks of 5
    blocks = []
    visited = set()
    for r in range(rows - 1):
        for c in range(cols - 1):
            if (grid[r][c] == 5 and grid[r][c+1] == 5 and
                grid[r+1][c] == 5 and grid[r+1][c+1] == 5 and
                (r, c) not in visited):
                blocks.append((r, c))
                visited.update([(r,c),(r,c+1),(r+1,c),(r+1,c+1)])

    # For each block, place corner markers
    for r, c in blocks:
        # Top-left corner: (r-1, c-1) = 1
        if r-1 >= 0 and c-1 >= 0:
            grid[r-1][c-1] = 1
        # Top-right corner: (r-1, c+2) = 2
        if r-1 >= 0 and c+2 < cols:
            grid[r-1][c+2] = 2
        # Bottom-left corner: (r+2, c-1) = 3
        if r+2 < rows and c-1 >= 0:
            grid[r+2][c-1] = 3
        # Bottom-right corner: (r+2, c+2) = 4
        if r+2 < rows and c+2 < cols:
            grid[r+2][c+2] = 4

    return grid

solutions['95990924'] = solve_95990924

# === 963e52fc ===
# Pattern rows get doubled in width by continuing the pattern.
# Non-pattern rows (all 0) also get doubled.
# The pattern repeats/continues to double width.
def solve_963e52fc(grid):
    rows = len(grid)
    cols = len(grid[0])
    new_cols = cols * 2
    output = []

    for r in range(rows):
        row = grid[r]
        if all(v == 0 for v in row):
            output.append([0] * new_cols)
        else:
            # Find the repeating unit in the row
            # The pattern repeats - extend it to double width
            # Need to find the period of the pattern
            new_row = []
            for c in range(new_cols):
                # Try to continue the pattern
                # Find period
                new_row.append(row[c % cols] if c < cols else 0)

            # Actually, looking at examples more carefully:
            # The pattern continues seamlessly. Let me find the period.
            # Ex1: [2,8,2,8,2,8] -> [2,8,2,8,2,8,2,8,2,8,2,8] - period 2
            # Ex2: [2,3,3,2,3,3,2] -> [2,3,3,2,3,3,2,3,3,2,3,3,2,3] - period 3
            # Ex3: [1,2,2,1,2,2,1,2] -> [1,2,2,1,2,2,1,2,2,1,2,2,1,2,2,1] - period 3
            #       [2,1,2,2,1,2,2,1] -> [2,1,2,2,1,2,2,1,2,2,1,2,2,1,2,2] - period 3

            # Find the minimal period
            best_period = None
            for p in range(1, cols + 1):
                valid = True
                for c in range(cols):
                    if row[c] != row[c % p]:
                        valid = False
                        break
                if valid:
                    best_period = p
                    break

            new_row = [row[c % best_period] for c in range(new_cols)]
            output.append(new_row)

    return output

solutions['963e52fc'] = solve_963e52fc

# === 97999447 ===
# Each non-zero cell starts a pattern: value, 5, value, 5, ... extending to the right
def solve_97999447(grid):
    grid = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])

    # Find non-zero cells
    dots = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                dots.append((r, c, grid[r][c]))

    for r, c, val in dots:
        # Fill rightward: val, 5, val, 5, ...
        for i, cc in enumerate(range(c, cols)):
            if i == 0:
                grid[r][cc] = val
            elif i % 2 == 0:
                grid[r][cc] = val
            else:
                grid[r][cc] = 5

    return grid

solutions['97999447'] = solve_97999447

# === 995c5fa3 ===
# Three 4x4 panels separated by column of 0s. Each panel may have a 2x2 hole of 0s.
# Panel with no hole -> output row of 2. Panel ordering maps to output position.
# Actually looking more carefully:
# Each panel is 4 cols wide, separated by single col of 0.
# Panels: cols 0-3, 5-8, 10-13
# A panel is "full" (all 5s) or has a 2x2 hole.
# Hole position maps to a color:
#   - top-left hole (rows 1-2, left cols) -> ?
# Let me map holes to output colors:
# Ex1: panel1=full, panel2=hole at rows1-2 cols1-2(relative), panel3=hole at rows1-2 cols0,3?
# Let me re-examine.
# Panel structure: 4 rows x 4 cols of 5s, with possible 2x2 hole
# Ex1 panels:
# P1: all 5s -> full
# P2: [5,5,5,5],[5,0,0,5],[5,0,0,5],[5,5,5,5] -> center hole
# P3: [5,5,5,5],[0,5,5,0],[0,5,5,0],[5,5,5,5] -> side holes (cols 0,3)
# Wait P3: cols 10-13: [5,5,5,5],[0,5,5,0],[0,5,5,0],[5,5,5,5]
# Output: [2,2,2],[8,8,8],[3,3,3]
# Hmm. The panels order left-to-right maps to some ranking.
# Let me look at hole positions more carefully.
# Hole positions in 4x4 grid (ignoring border):
#   center: rows 1-2, cols 1-2
#   top-right: rows 0-1, cols 2-3? No, let me look at actual data.
#
# Ex1: P1=full, P2=center(r1-2,c1-2), P3=left-right edges(r1-2,c0+c3)
# Actually P3: row1=[0,5,5,0], row2=[0,5,5,0] - holes at (1,0),(1,3),(2,0),(2,3)
# That's not a 2x2 block. Let me re-examine.
#
# Ex2: P1=[5,5,5,5],[0,5,5,0],[0,5,5,0],[5,5,5,5], P2=all 5s, P3=[5,5,5,5],[5,5,5,5],[5,0,0,5],[5,0,0,5]
# Output: [3,3,3],[4,4,4],[2,2,2]
#
# So the hole POSITION determines the color:
# No hole -> 2
# Center hole (rows 1-2, cols 1-2) -> 8?
# In ex1: P2 has center hole -> output row 2 (middle) = 8
# In ex3: P1 has hole at rows 1-2, cols 1-2 -> output row 1 = 8. Yes!
#
# Holes at rows 1-2, cols 1-2 (center) -> 8
# Holes at rows 1-2, cols 0,3 (sides) -> 3? In ex1 P3 -> row3 = 3
# Holes at rows 0-1, cols 0,3? Let me check ex2 P1.
# P1 ex2: rows 1-2, cols 0,3 -> output row1 = 3. Hmm same pattern.
# Wait ex2 output: [3,3,3],[4,4,4],[2,2,2]. P1 has side holes -> 3, P2 full -> ?, P3 has bottom hole -> ?
#
# Actually let me reconsider. Maybe:
# full panel -> 2
# hole position determines color differently.
# Let me categorize all holes:
# Ex1: P2 center hole -> 8, P3 side holes -> 3
# Ex2: P1 side holes -> 3, P3 bottom-center hole (r2-3,c1-2) -> 4 (but r0-1 are full, r2-3 have holes)
# Wait P3 ex2: [5,5,5,5],[5,5,5,5],[5,0,0,5],[5,0,0,5] -> hole at rows 2-3, cols 1-2 -> 4
# Ex3: P1 has hole rows 1-2, cols 1-2 -> 8. P3 has hole rows 2-3, cols 1-2?
# P3 ex3: [5,5,5,5],[5,5,5,5],[5,0,0,5],[5,0,0,5] -> rows 2-3, cols 1-2 -> 4
# But output [8,8,8],[2,2,2],[4,4,4] - P1->8, P2->2(full), P3->4. Matches!
# Ex4: P1=full->2, P2 center hole->8... wait
# P2 ex4: [5,5,5,5],[5,5,5,5],[5,0,0,5],[5,0,0,5] -> rows 2-3, cols 1-2 -> that should be 4
# Output: [2,2,2],[4,4,4],[2,2,2]. P1 full->2, P2->4, P3 full->2. Yes!
#
# So: full=2, center(r1-2,c1-2)=8, bottom(r2-3,c1-2)=4, sides(r1-2,c0&c3)=3
# Let me verify ex2 P1: [5,5,5,5],[0,5,5,0],[0,5,5,0],[5,5,5,5] -> holes at r1-2, c0 and c3 -> 3.
# Output row1=3. Yes!

def solve_995c5fa3(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Extract panels (separated by columns of 0)
    # Find separator columns
    sep_cols = []
    for c in range(cols):
        if all(grid[r][c] == 0 for r in range(rows)):
            sep_cols.append(c)

    # Extract panel column ranges
    panel_ranges = []
    starts = [0] + [s + 1 for s in sep_cols]
    ends = [s for s in sep_cols] + [cols]
    for s, e in zip(starts, ends):
        if s < e:
            panel_ranges.append((s, e))

    output = []
    for s, e in panel_ranges:
        # Extract panel
        panel = []
        for r in range(rows):
            panel.append([grid[r][c] for c in range(s, e)])

        # Check if full (all 5s)
        is_full = all(panel[r][c] == 5 for r in range(rows) for c in range(e - s))
        if is_full:
            output.append([2, 2, 2])
            continue

        # Find hole positions
        holes = []
        for r in range(rows):
            for c in range(e - s):
                if panel[r][c] == 0:
                    holes.append((r, c))

        hole_rows = set(r for r, c in holes)
        hole_cols = set(c for r, c in holes)

        w = e - s  # panel width

        # Categorize hole
        if hole_rows == {1, 2} and hole_cols == {1, 2}:
            # Center hole
            output.append([8, 8, 8])
        elif hole_rows == {2, 3} and hole_cols == {1, 2}:
            # Bottom center hole
            output.append([4, 4, 4])
        elif hole_rows == {0, 1} and hole_cols == {1, 2}:
            # Top center hole
            # Need to check if this exists
            output.append([6, 6, 6])  # placeholder
        elif hole_cols == {0, w - 1}:
            # Side holes
            output.append([3, 3, 3])
        elif hole_rows == {0, 1} and hole_cols == {0, 1}:
            output.append([3, 3, 3])
        else:
            output.append([2, 2, 2])

    return output

solutions['995c5fa3'] = solve_995c5fa3

# ============ TESTING ============
import json

task_ids = ['928ad970', '93b581b8', '941d9a10', '94f9d214', '952a094c',
            '9565186b', '95990924', '963e52fc', '97999447', '995c5fa3']

results = {}
all_pass = True

for task_id in task_ids:
    with open(f'data/arc1/{task_id}.json', 'r') as f:
        task = json.load(f)

    solve_fn = solutions[task_id]
    passed = True

    # Test on training pairs
    for i, pair in enumerate(task['train']):
        output = solve_fn(pair['input'])
        expected = pair['output']
        if output != expected:
            print(f"FAIL {task_id} train[{i}]")
            # Show difference
            for r in range(min(len(output), len(expected))):
                for c in range(min(len(output[0]) if output else 0, len(expected[0]) if expected else 0)):
                    if r < len(output) and r < len(expected) and c < len(output[r]) and c < len(expected[r]):
                        if output[r][c] != expected[r][c]:
                            print(f"  ({r},{c}): got {output[r][c]}, expected {expected[r][c]}")
            if len(output) != len(expected):
                print(f"  rows: got {len(output)}, expected {len(expected)}")
            elif output and expected and len(output[0]) != len(expected[0]):
                print(f"  cols: got {len(output[0])}, expected {len(expected[0])}")
            passed = False
            all_pass = False

    # Test on test pairs
    for i, pair in enumerate(task['test']):
        output = solve_fn(pair['input'])
        if 'output' in pair:
            expected = pair['output']
            if output != expected:
                print(f"FAIL {task_id} test[{i}]")
                for r in range(min(len(output), len(expected))):
                    for c in range(min(len(output[0]) if output else 0, len(expected[0]) if expected else 0)):
                        if r < len(output) and r < len(expected) and c < len(output[r]) and c < len(expected[r]):
                            if output[r][c] != expected[r][c]:
                                print(f"  ({r},{c}): got {output[r][c]}, expected {expected[r][c]}")
                passed = False
                all_pass = False

    if passed:
        print(f"PASS {task_id}")

    results[task_id] = {
        'train': [solve_fn(p['input']) for p in task['train']],
        'test': [solve_fn(p['input']) for p in task['test']]
    }

print(f"\nAll pass: {all_pass}")

# Save results
output_path = 'data/arc_python_solutions_b14.json'
# Build output with solve functions as source code
import inspect
output_data = {}
for task_id in task_ids:
    solve_fn = solutions[task_id]
    with open(f'data/arc1/{task_id}.json', 'r') as f:
        task = json.load(f)
    output_data[task_id] = {
        'solve': inspect.getsource(solve_fn),
        'train_outputs': [solve_fn(p['input']) for p in task['train']],
        'test_outputs': [solve_fn(p['input']) for p in task['test']],
    }

with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"Saved to {output_path}")
