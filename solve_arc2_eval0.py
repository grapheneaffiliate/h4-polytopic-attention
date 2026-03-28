import json
import copy

solutions = {}

# ============================================================
# 135a2760: Fix repeating patterns in bordered rectangular sections
# ============================================================
solutions["135a2760"] = '''
def solve(grid):
    import copy
    grid = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])
    bg = grid[0][0]

    def find_period(seq):
        n = len(seq)
        for p in range(1, n+1):
            votes = {}
            for i in range(n):
                k = i % p
                if k not in votes:
                    votes[k] = {}
                v = seq[i]
                votes[k][v] = votes[k].get(v, 0) + 1
            consensus = []
            for k in range(p):
                best_v = max(votes[k], key=votes[k].get)
                consensus.append(best_v)
            matches = sum(1 for i in range(n) if seq[i] == consensus[i % p])
            if matches >= n - 2:
                return p, consensus
        return n, seq

    def is_border_row(row):
        return set(row) == {bg, 2} or all(v == bg for v in row)

    for r in range(rows):
        row = grid[r]
        if is_border_row(row):
            continue
        twos = [c for c in range(cols) if row[c] == 2]
        if len(twos) < 2:
            continue
        left = twos[0]
        right = twos[-1]
        inner = row[left+1:right]
        if not inner:
            continue
        p, cons = find_period(inner)
        for i in range(len(inner)):
            grid[r][left+1+i] = cons[i % p]

    return grid
'''

# ============================================================
# 1ae2feb7: Extend row patterns to the right of a divider column
# Each row's left side has colored segments. Each segment of color C and length L
# generates a right-side pattern: C at position 0, then L-1 zeros, repeating.
# Segments applied left-to-right (later overwrites earlier).
# ============================================================
solutions["1ae2feb7"] = '''
def solve(grid):
    import copy
    grid = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])

    # Find divider column (vertical line of same non-zero value)
    divider_col = -1
    divider_val = -1
    for c in range(cols):
        col_vals = [grid[r][c] for r in range(rows)]
        if all(v == col_vals[0] for v in col_vals) and col_vals[0] != 0:
            divider_col = c
            divider_val = col_vals[0]
            break

    if divider_col == -1:
        return grid

    right_len = cols - divider_col - 1

    for r in range(rows):
        left = grid[r][:divider_col]
        if all(v == 0 for v in left):
            continue

        # Parse colored segments from left side
        segments = []
        i = 0
        while i < len(left):
            if left[i] != 0:
                color = left[i]
                length = 0
                while i < len(left) and left[i] == color:
                    length += 1
                    i += 1
                segments.append((color, length))
            else:
                i += 1

        # Apply each segment to right side
        right = [0] * right_len
        for color, length in segments:
            if length == 0:
                continue
            for j in range(right_len):
                if j % length == 0:
                    right[j] = color

        for j in range(right_len):
            grid[r][divider_col + 1 + j] = right[j]

    return grid
'''

# ============================================================
# 1818057f: Replace plus-shaped (cross) clusters of 4s with 8s
# ============================================================
solutions["1818057f"] = '''
def solve(grid):
    import copy
    grid = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])

    to_mark = set()
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if grid[r][c] == 4:
                if (grid[r-1][c] == 4 and grid[r+1][c] == 4 and
                    grid[r][c-1] == 4 and grid[r][c+1] == 4):
                    for dr, dc in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
                        to_mark.add((r+dr, c+dc))

    for r, c in to_mark:
        grid[r][c] = 8

    return grid
'''

# ============================================================
# 31f7f899: Vertical colored lines on a cross beam.
# Sort line extents ascending by column position (left to right).
# ============================================================
solutions["31f7f899"] = '''
def solve(grid):
    import copy
    grid = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])

    # Find beam row (row with 6s)
    beam_row = -1
    for r in range(rows):
        if 6 in grid[r]:
            beam_row = r
            break

    # Find vertical lines
    lines = {}
    for c in range(cols):
        color = None
        for r in range(rows):
            v = grid[r][c]
            if r != beam_row and v != 8 and v != 6:
                color = v
                break
        if color is None:
            continue
        extent_above = 0
        for r in range(beam_row-1, -1, -1):
            if grid[r][c] == color:
                extent_above += 1
            else:
                break
        extent_below = 0
        for r in range(beam_row+1, rows):
            if grid[r][c] == color:
                extent_below += 1
            else:
                break
        extent = max(extent_above, extent_below)
        if extent > 0 or grid[beam_row][c] == color:
            lines[c] = (color, extent)

    if not lines:
        return grid

    sorted_cols = sorted(lines.keys())
    extents = sorted([lines[c][1] for c in sorted_cols])

    for idx, c in enumerate(sorted_cols):
        old_color, old_ext = lines[c]
        new_ext = extents[idx]
        for r in range(rows):
            if r != beam_row and grid[r][c] == old_color:
                grid[r][c] = 8
        for d in range(1, new_ext + 1):
            if beam_row - d >= 0:
                grid[beam_row - d][c] = old_color
            if beam_row + d < rows:
                grid[beam_row + d][c] = old_color

    return grid
'''

# ============================================================
# 16de56c4: Row/column patterns with periodic tiling
# Each row (or column) with 2+ non-zero values defines a repeating pattern.
# Single color: tile with that color at the period.
# Two colors: if unique falls ON tiling grid -> unique replaces repeated.
#             if unique falls OFF grid -> repeated tiles, unique stays as override.
# ============================================================
solutions["16de56c4"] = '''
def solve(grid):
    import copy
    from collections import Counter
    grid = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])

    row_count = sum(1 for r in range(rows) if sum(1 for c in range(cols) if grid[r][c] != 0) >= 2)
    col_count = sum(1 for c in range(cols) if sum(1 for r in range(rows) if grid[r][c] != 0) >= 2)
    is_row = row_count >= col_count

    def process_line(nz, length, is_row_mode):
        if len(nz) < 2:
            return None
        colors = Counter(v for _, v in nz)

        if len(colors) == 1:
            color = nz[0][1]
            positions = sorted([p for p, v in nz])
            period = positions[1] - positions[0]
            start = positions[0]
            result = [0] * length
            for p in range(length):
                if (p - start) % period == 0:
                    result[p] = color
            return result
        elif len(colors) == 2:
            rep_colors = [c for c, cnt in colors.items() if cnt >= 2]
            uniq_colors = [c for c, cnt in colors.items() if cnt == 1]
            if not rep_colors or not uniq_colors:
                return None
            rep_color = rep_colors[0]
            uniq_color = uniq_colors[0]
            rep_pos = sorted([p for p, v in nz if v == rep_color])
            uniq_pos = [p for p, v in nz if v == uniq_color][0]
            period = rep_pos[1] - rep_pos[0]
            start = rep_pos[0]
            on_grid = (uniq_pos - start) % period == 0
            result = [0] * length
            max_pos = max(p for p, v in nz)
            if on_grid:
                if is_row_mode:
                    tile_range = range(start, max_pos + 1)
                else:
                    tile_range = range(length)
                for p in tile_range:
                    if (p - start) % period == 0:
                        result[p] = uniq_color
            else:
                for p in range(length):
                    if (p - start) % period == 0:
                        result[p] = rep_color
                result[uniq_pos] = uniq_color
            return result
        return None

    if is_row:
        for r in range(rows):
            nz = [(c, grid[r][c]) for c in range(cols) if grid[r][c] != 0]
            if len(nz) < 2:
                continue
            result = process_line(nz, cols, True)
            if result:
                grid[r] = result
    else:
        for c in range(cols):
            nz = [(r, grid[r][c]) for r in range(rows) if grid[r][c] != 0]
            if len(nz) < 2:
                continue
            result = process_line(nz, rows, False)
            if result:
                for r in range(rows):
                    grid[r][c] = result[r]

    return grid
'''

# ============================================================
# Testing and saving
# ============================================================
print("Testing solutions...")
passing = {}
for task_id, code in solutions.items():
    try:
        with open(f"data/arc2/{task_id}.json") as f:
            data = json.load(f)
        exec(code)
        all_pass = True
        for i, pair in enumerate(data["train"]):
            result = solve(pair["input"])
            if result != pair["output"]:
                print(f"FAIL: {task_id} train {i}")
                all_pass = False
                break
        if all_pass:
            print(f"PASS: {task_id}")
            passing[task_id] = code
    except Exception as e:
        print(f"ERROR: {task_id}: {e}")

# Save only passing solutions
with open("data/arc2_solutions_eval0.json", "w") as f:
    json.dump(passing, f, indent=2)
print(f"\nSaved {len(passing)} solutions")
