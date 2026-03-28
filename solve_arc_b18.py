import json
import copy

def solve_b527c5c6(grid):
    """Two rectangles of 3s, each has one 2. The 2 on the edge indicates direction
    to extend. Extension width = 2*rect_dim_along_edge - 1, centered on the 2."""
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    visited = [[False]*cols for _ in range(rows)]

    def flood_fill(r, c):
        stack = [(r, c)]
        cells = []
        while stack:
            cr, cc = stack.pop()
            if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                continue
            if visited[cr][cc] or grid[cr][cc] == 0:
                continue
            visited[cr][cc] = True
            cells.append((cr, cc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((cr+dr, cc+dc))
        return cells

    rectangles = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                cells = flood_fill(r, c)
                if cells:
                    rectangles.append(cells)

    for rect_cells in rectangles:
        two_pos = None
        for r, c in rect_cells:
            if grid[r][c] == 2:
                two_pos = (r, c)
                break
        if two_pos is None:
            continue

        min_r = min(r for r, c in rect_cells)
        max_r = max(r for r, c in rect_cells)
        min_c = min(c for r, c in rect_cells)
        max_c = max(c for r, c in rect_cells)

        tr, tc = two_pos
        rect_h = max_r - min_r + 1
        rect_w = max_c - min_c + 1

        if tr == min_r:  # top edge -> extend upward
            # Extension direction is vertical (up), edge is horizontal (top)
            # Perpendicular spread = 2 * rect_h - 1, centered on tc
            spread = rect_h - 1
            for r in range(min_r - 1, -1, -1):
                for c in range(tc - spread, tc + spread + 1):
                    if 0 <= c < cols:
                        if c == tc:
                            out[r][c] = 2
                        else:
                            out[r][c] = 3
        elif tr == max_r:  # bottom edge -> extend downward
            spread = rect_h - 1
            for r in range(max_r + 1, rows):
                for c in range(tc - spread, tc + spread + 1):
                    if 0 <= c < cols:
                        if c == tc:
                            out[r][c] = 2
                        else:
                            out[r][c] = 3
        elif tc == min_c:  # left edge -> extend leftward
            spread = rect_w - 1
            for c in range(min_c - 1, -1, -1):
                for r in range(tr - spread, tr + spread + 1):
                    if 0 <= r < rows:
                        if r == tr:
                            out[r][c] = 2
                        else:
                            out[r][c] = 3
        elif tc == max_c:  # right edge -> extend rightward
            spread = rect_w - 1
            for c in range(max_c + 1, cols):
                for r in range(tr - spread, tr + spread + 1):
                    if 0 <= r < rows:
                        if r == tr:
                            out[r][c] = 2
                        else:
                            out[r][c] = 3

    return out


def solve_b548a754(grid):
    """Rectangle with border color and inner color. An 8 dot indicates direction.
    Stretch the rectangle toward the 8, keeping border/inner pattern."""
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    eight_r, eight_c = None, None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8:
                eight_r, eight_c = r, c
                break

    rect_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and grid[r][c] != 8:
                rect_cells.append((r, c))

    min_r = min(r for r, c in rect_cells)
    max_r = max(r for r, c in rect_cells)
    min_c = min(c for r, c in rect_cells)
    max_c = max(c for r, c in rect_cells)

    border_color = grid[min_r][min_c]
    inner_color = None
    for r in range(min_r+1, max_r):
        for c in range(min_c+1, max_c):
            if grid[r][c] != border_color:
                inner_color = grid[r][c]
                break
        if inner_color:
            break

    out[eight_r][eight_c] = 0

    # Determine new bounding box
    new_min_r, new_max_r = min_r, max_r
    new_min_c, new_max_c = min_c, max_c

    if eight_r < min_r:
        new_min_r = eight_r
    elif eight_r > max_r:
        new_max_r = eight_r
    elif eight_c < min_c:
        new_min_c = eight_c
    elif eight_c > max_c:
        new_max_c = eight_c

    for r in range(new_min_r, new_max_r + 1):
        for c in range(new_min_c, new_max_c + 1):
            if r == new_min_r or r == new_max_r or c == new_min_c or c == new_max_c:
                out[r][c] = border_color
            else:
                out[r][c] = inner_color

    return out


def solve_b60334d2(grid):
    """Each 5 gets a plus pattern: diagonals get 5, orthogonals get 1, center becomes 0."""
    rows = len(grid)
    cols = len(grid[0])
    out = [[0]*cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if dr == 0 and dc == 0:
                                pass
                            elif dr == 0 or dc == 0:
                                out[nr][nc] = 1
                            else:
                                out[nr][nc] = 5

    return out


def solve_b6afb2da(grid):
    """Each rectangle of 5s: corners->1, border non-corners->4, interior->2."""
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    visited = [[False]*cols for _ in range(rows)]

    def flood_fill(r, c):
        stack = [(r, c)]
        cells = []
        while stack:
            cr, cc = stack.pop()
            if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                continue
            if visited[cr][cc] or grid[cr][cc] != 5:
                continue
            visited[cr][cc] = True
            cells.append((cr, cc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((cr+dr, cc+dc))
        return cells

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5 and not visited[r][c]:
                cells = flood_fill(r, c)
                min_r2 = min(r for r, c in cells)
                max_r2 = max(r for r, c in cells)
                min_c2 = min(c for r, c in cells)
                max_c2 = max(c for r, c in cells)

                for cr, cc in cells:
                    is_top = (cr == min_r2)
                    is_bot = (cr == max_r2)
                    is_left = (cc == min_c2)
                    is_right = (cc == max_c2)

                    is_corner = (is_top or is_bot) and (is_left or is_right)
                    is_border = is_top or is_bot or is_left or is_right

                    if is_corner:
                        out[cr][cc] = 1
                    elif is_border:
                        out[cr][cc] = 4
                    else:
                        out[cr][cc] = 2

    return out


def solve_b7249182(grid):
    """Two dots on same row or column. They extend toward each other,
    meeting with bracket shapes. Arm width perpendicular is always 2."""
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    dots = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                dots.append((r, c, grid[r][c]))

    r1, c1, color1 = dots[0]
    r2, c2, color2 = dots[1]

    if c1 == c2:
        # Same column - vertical arrangement
        if r1 > r2:
            r1, c1, color1, r2, c2, color2 = r2, c2, color2, r1, c1, color1

        dist = r2 - r1
        half = dist // 2
        half_w = 2

        # Color1 vertical line from r1 down
        bar1_r = r1 + half - 1
        for r in range(r1 + 1, bar1_r):
            out[r][c1] = color1
        # Horizontal bar for color1
        for c in range(c1 - half_w, c1 + half_w + 1):
            if 0 <= c < cols:
                out[bar1_r][c] = color1
        # Side row for color1
        side1_r = bar1_r + 1
        if 0 <= c1 - half_w < cols:
            out[side1_r][c1 - half_w] = color1
        if 0 <= c1 + half_w < cols:
            out[side1_r][c1 + half_w] = color1

        # Color2 vertical line from r2 up
        bar2_r = r2 - half + 1
        for r in range(r2 - 1, bar2_r, -1):
            out[r][c2] = color2
        # Horizontal bar for color2
        for c in range(c2 - half_w, c2 + half_w + 1):
            if 0 <= c < cols:
                out[bar2_r][c] = color2
        # Side row for color2
        side2_r = bar2_r - 1
        if 0 <= c2 - half_w < cols:
            out[side2_r][c2 - half_w] = color2
        if 0 <= c2 + half_w < cols:
            out[side2_r][c2 + half_w] = color2

    else:  # Same row
        if c1 > c2:
            r1, c1, color1, r2, c2, color2 = r2, c2, color2, r1, c1, color1

        dist = c2 - c1
        half = dist // 2
        half_w = 2

        # Color1 horizontal line from c1 right
        bar1_c = c1 + half - 1
        for c in range(c1 + 1, bar1_c):
            out[r1][c] = color1
        # Vertical bar for color1
        for r in range(r1 - half_w, r1 + half_w + 1):
            if 0 <= r < rows:
                out[r][bar1_c] = color1
        # Side col for color1
        side1_c = bar1_c + 1
        if 0 <= r1 - half_w < rows:
            out[r1 - half_w][side1_c] = color1
        if 0 <= r1 + half_w < rows:
            out[r1 + half_w][side1_c] = color1

        # Color2 horizontal line from c2 left
        bar2_c = c2 - half + 1
        for c in range(c2 - 1, bar2_c, -1):
            out[r2][c] = color2
        # Vertical bar for color2
        for r in range(r2 - half_w, r2 + half_w + 1):
            if 0 <= r < rows:
                out[r][bar2_c] = color2
        # Side col for color2
        side2_c = bar2_c - 1
        if 0 <= r2 - half_w < rows:
            out[r2 - half_w][side2_c] = color2
        if 0 <= r2 + half_w < rows:
            out[r2 + half_w][side2_c] = color2

    return out


def solve_b8cdaf2b(grid):
    """Bottom rows form a base pattern. The center color extends diagonally upward
    from the inner edges of the second-to-last row."""
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    bottom = grid[rows-1]
    second = grid[rows-2]

    center_c = cols // 2
    center_color = bottom[center_c]

    inner_cells = [(c, second[c]) for c in range(cols) if second[c] != 0]
    inner_left = min(c for c, _ in inner_cells)
    inner_right = max(c for c, _ in inner_cells)

    outer_left = min(c for c in range(cols) if bottom[c] != 0)
    outer_right = max(c for c in range(cols) if bottom[c] != 0)

    gap_left = inner_left - outer_left
    gap_right = outer_right - inner_right

    for step in range(1, gap_left + 1):
        r = rows - 2 - step
        c = inner_left - step
        if 0 <= r < rows and 0 <= c < cols:
            out[r][c] = center_color

    for step in range(1, gap_right + 1):
        r = rows - 2 - step
        c = inner_right + step
        if 0 <= r < rows and 0 <= c < cols:
            out[r][c] = center_color

    return out


def solve_b91ae062(grid):
    """3x3 input scaled up by factor = number of distinct non-zero colors."""
    rows = len(grid)
    cols = len(grid[0])

    distinct_colors = len(set(grid[r][c] for r in range(rows) for c in range(cols) if grid[r][c] != 0))
    scale = distinct_colors

    out_rows = rows * scale
    out_cols = cols * scale
    out = [[0]*out_cols for _ in range(out_rows)]

    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            for dr in range(scale):
                for dc in range(scale):
                    out[r*scale+dr][c*scale+dc] = val

    return out


def solve_b94a9452(grid):
    """Rectangle with border and inner colors. Extract and swap them."""
    rows = len(grid)
    cols = len(grid[0])

    rect_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                rect_cells.append((r, c))

    min_r = min(r for r, c in rect_cells)
    max_r = max(r for r, c in rect_cells)
    min_c = min(c for r, c in rect_cells)
    max_c = max(c for r, c in rect_cells)

    h = max_r - min_r + 1
    w = max_c - min_c + 1

    rect = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            row.append(grid[r][c])
        rect.append(row)

    border_color = rect[0][0]
    inner_color = None
    for r in range(h):
        for c in range(w):
            if rect[r][c] != border_color:
                inner_color = rect[r][c]
                break
        if inner_color:
            break

    out = []
    for r in range(h):
        row = []
        for c in range(w):
            if rect[r][c] == border_color:
                row.append(inner_color)
            else:
                row.append(border_color)
        out.append(row)

    return out


def solve_b9b7f026(grid):
    """Multiple filled rectangles. One has holes (0s inside). Output its color."""
    rows = len(grid)
    cols = len(grid[0])

    visited = [[False]*cols for _ in range(rows)]

    def flood_fill(r, c, color):
        stack = [(r, c)]
        cells = []
        while stack:
            cr, cc = stack.pop()
            if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                continue
            if visited[cr][cc] or grid[cr][cc] != color:
                continue
            visited[cr][cc] = True
            cells.append((cr, cc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((cr+dr, cc+dc))
        return cells

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                cells = flood_fill(r, c, color)

                min_r2 = min(cr for cr, cc in cells)
                max_r2 = max(cr for cr, cc in cells)
                min_c2 = min(cc for cr, cc in cells)
                max_c2 = max(cc for cr, cc in cells)

                expected_area = (max_r2 - min_r2 + 1) * (max_c2 - min_c2 + 1)
                actual_area = len(cells)

                if actual_area < expected_area:
                    return [[color]]

    return [[0]]


def solve_ba26e723(grid):
    """3-row pattern. Replace 4 with 6 at columns that are multiples of 3."""
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    for r in range(rows):
        for c in range(cols):
            if out[r][c] == 4 and c % 3 == 0:
                out[r][c] = 6

    return out


# Main execution
tasks = {
    'b527c5c6': solve_b527c5c6,
    'b548a754': solve_b548a754,
    'b60334d2': solve_b60334d2,
    'b6afb2da': solve_b6afb2da,
    'b7249182': solve_b7249182,
    'b8cdaf2b': solve_b8cdaf2b,
    'b91ae062': solve_b91ae062,
    'b94a9452': solve_b94a9452,
    'b9b7f026': solve_b9b7f026,
    'ba26e723': solve_ba26e723,
}

results = {}
all_pass = True

for tid, solver in tasks.items():
    with open(f'C:/Users/atchi/h4-polytopic-attention/data/arc1/{tid}.json') as f:
        data = json.load(f)

    task_pass = True

    all_pairs = [(p, 'train', i) for i, p in enumerate(data['train'])] + \
                [(p, 'test', i) for i, p in enumerate(data['test'])]

    for pair, split, idx in all_pairs:
        inp = pair['input']
        expected = pair['output']
        try:
            actual = solver(inp)
            if actual == expected:
                print(f"  {tid} {split}[{idx}]: PASS")
            else:
                print(f"  {tid} {split}[{idx}]: FAIL")
                if len(actual) != len(expected):
                    print(f"    Size mismatch: got {len(actual)}x{len(actual[0]) if actual else 0}, expected {len(expected)}x{len(expected[0])}")
                else:
                    for r in range(min(len(expected), 5)):
                        if r < len(actual) and actual[r] != expected[r]:
                            print(f"    Row {r}: got {actual[r]}")
                            print(f"           exp {expected[r]}")
                task_pass = False
        except Exception as e:
            print(f"  {tid} {split}[{idx}]: ERROR - {e}")
            import traceback
            traceback.print_exc()
            task_pass = False

    if task_pass:
        print(f"  {tid}: ALL PASS")
    else:
        print(f"  {tid}: SOME FAIL")
        all_pass = False

    results[tid] = {
        'solver': solver.__name__,
        'pass_all': task_pass
    }

print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
