#!/usr/bin/env python3
"""ARC-AGI-2 retry solver for hard tasks."""
import json
import copy
import os
from collections import Counter, defaultdict

DATA_DIR = "data/arc2"
OUTPUT_FILE = "data/arc2_solutions_retry0.json"

TASK_IDS = "0934a4d8,136b0064,13e47133,142ca369,16b78196,195c6913,20270e3b,20a9e565,21897d95,221dfab4,247ef758,269e22fb,271d71e2,28a6681f,291dc1e1,2b83f449,2ba387bc,2c181942,2d0172a1,332f06d7,35ab12c3,36a08778,38007db0,3a25b0d8,3dc255db,3e6067c3".split(",")

def load_task(tid):
    with open(os.path.join(DATA_DIR, f"{tid}.json")) as f:
        return json.load(f)

def grid_eq(a, b):
    if len(a) != len(b): return False
    for r1, r2 in zip(a, b):
        if r1 != r2: return False
    return True

def test_solve(tid, solve_fn):
    task = load_task(tid)
    for i, pair in enumerate(task['train']):
        result = solve_fn(pair['input'])
        if not grid_eq(result, pair['output']):
            return False
    return True

def apply_solve(tid, solve_fn):
    task = load_task(tid)
    results = []
    for pair in task['test']:
        results.append(solve_fn(pair['input']))
    return results

# ============================================================
# 38007db0: Grid of repeated tiles. Per-cell odd-one-out.
# ============================================================
def solve_38007db0(grid):
    R, C = len(grid), len(grid[0])
    sep_val = grid[0][0]
    sep_cols = [c for c in range(C) if all(grid[r][c] == sep_val for r in range(R))]
    if len(sep_cols) < 2:
        return grid
    tile_w = sep_cols[1] - sep_cols[0] + 1

    result = []
    for r in range(R):
        row_result = []
        for dc in range(tile_w):
            candidates = []
            for cg_start in sep_cols[:-1]:
                c = cg_start + dc
                if c < C:
                    candidates.append(grid[r][c])
            cnt = Counter(candidates)
            if len(cnt) == 1:
                row_result.append(candidates[0])
            else:
                minority = cnt.most_common()[-1][0]
                row_result.append(minority)
        result.append(row_result)
    return result


# ============================================================
# 2ba387bc: Separate frame objects (with holes) and solid objects.
# Pair them and stack: frame left, solid right. Each takes 4 rows.
# ============================================================
def solve_2ba387bc(grid):
    R, C = len(grid), len(grid[0])
    bg = 0

    visited = [[False]*C for _ in range(R)]
    objects = []

    for r in range(R):
        for c in range(C):
            if not visited[r][c] and grid[r][c] != bg:
                stack = [(r, c)]
                cells = []
                while stack:
                    r2, c2 = stack.pop()
                    if r2 < 0 or r2 >= R or c2 < 0 or c2 >= C: continue
                    if visited[r2][c2]: continue
                    if grid[r2][c2] == bg: continue
                    visited[r2][c2] = True
                    cells.append((r2, c2, grid[r2][c2]))
                    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                        stack.append((r2+dr, c2+dc))
                if cells:
                    objects.append(cells)

    extracted = []
    for cells in objects:
        rmin = min(r for r, c, v in cells)
        rmax = max(r for r, c, v in cells)
        cmin = min(c for r, c, v in cells)
        cmax = max(c for r, c, v in cells)

        pattern = []
        for r in range(rmin, rmax+1):
            row = []
            for c in range(cmin, cmax+1):
                row.append(grid[r][c])
            pattern.append(row)

        h = rmax - rmin + 1
        w = cmax - cmin + 1
        has_frame = any(grid[r2][c2] == bg for r2 in range(rmin, rmax+1) for c2 in range(cmin, cmax+1))
        colors = [v for r, c, v in cells]
        dominant = Counter(colors).most_common(1)[0][0]

        extracted.append({
            'pattern': pattern, 'h': h, 'w': w,
            'color': dominant, 'rmin': rmin, 'cmin': cmin,
            'has_frame': has_frame
        })

    frames = sorted([o for o in extracted if o['has_frame']], key=lambda x: (x['rmin'], x['cmin']))
    solids = sorted([o for o in extracted if not o['has_frame']], key=lambda x: (x['rmin'], x['cmin']))

    n_pairs = max(len(frames), len(solids))
    result = []
    for i in range(n_pairs):
        frame_pat = frames[i]['pattern'] if i < len(frames) else [[0]*4 for _ in range(4)]
        solid_pat = solids[i]['pattern'] if i < len(solids) else [[0]*4 for _ in range(4)]

        for r in range(4):
            left = frame_pat[r] if r < len(frame_pat) else [0]*4
            right = solid_pat[r] if r < len(solid_pat) else [0]*4
            left = (left + [0]*4)[:4]
            right = (right + [0]*4)[:4]
            result.append(left + right)

    return result


# ============================================================
# 3e6067c3: Grid with boxed patterns and code row.
# Chain of connections between boxes.
# ============================================================
def solve_3e6067c3(grid):
    R, C = len(grid), len(grid[0])
    g = [row[:] for row in grid]
    bg = 8

    # Find code row
    code_row = None
    for r in range(R-1, -1, -1):
        non_bg = [c for c in range(C) if grid[r][c] != bg]
        unique = set(grid[r][c] for c in non_bg)
        if len(unique) >= 3:
            code_row = r
            break
    if code_row is None:
        return g

    # Count non-1, non-bg cell values (excluding code row)
    interior_vals = Counter()
    for r in range(R):
        if r == code_row or abs(r - code_row) <= 1:
            continue
        for c in range(C):
            v = grid[r][c]
            if v != 1 and v != bg:
                interior_vals[v] += 1

    # The interior fill value is the most common non-1, non-bg value
    fill_val = None
    skip_vals = {1, bg}
    if interior_vals:
        fill_val = interior_vals.most_common(1)[0][0]
        if interior_vals[fill_val] > sum(interior_vals.values()) * 0.3:
            skip_vals.add(fill_val)

    # Parse code: extract center colors from code row
    # The code row has center colors separated by a "filler" value
    # The filler is the most common value in the code row
    code_vals = [grid[code_row][c] for c in range(C)]
    code_val_counts = Counter(code_vals)
    code_filler = code_val_counts.most_common(1)[0][0]
    code_seq = [v for v in code_vals if v != code_filler and v != bg]

    # Find ALL boxes by looking for center marker cells
    all_boxes = []
    found_centers = set()

    for r in range(R):
        if r == code_row or abs(r - code_row) <= 1:
            continue
        for c in range(C):
            v = grid[r][c]
            if v not in skip_vals and (r, c) not in found_centers:
                # Found a potential center cell. Find all connected cells of same color.
                stack = [(r, c)]
                center_cells = []
                visited_cc = set()
                while stack:
                    r2, c2 = stack.pop()
                    if r2 < 0 or r2 >= R or c2 < 0 or c2 >= C: continue
                    if (r2, c2) in visited_cc: continue
                    if grid[r2][c2] != v: continue
                    visited_cc.add((r2, c2))
                    center_cells.append((r2, c2))
                    found_centers.add((r2, c2))
                    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                        stack.append((r2+dr, c2+dc))

                if not center_cells:
                    continue

                center_rmin = min(r2 for r2, c2 in center_cells)
                center_rmax = max(r2 for r2, c2 in center_cells)
                center_cmin = min(c2 for r2, c2 in center_cells)
                center_cmax = max(c2 for r2, c2 in center_cells)

                # Find the surrounding 1-border box
                # Expand outward from center cells until we hit the 1-border
                rmin = center_rmin
                while rmin > 0 and grid[rmin-1][center_cmin] == 1:
                    rmin -= 1
                rmax = center_rmax
                while rmax < R-1 and grid[rmax+1][center_cmin] == 1:
                    rmax += 1
                cmin = center_cmin
                while cmin > 0 and grid[center_rmin][cmin-1] == 1:
                    cmin -= 1
                cmax = center_cmax
                while cmax < C-1 and grid[center_rmin][cmax+1] == 1:
                    cmax += 1

                center_rows = sorted(set(r2 for r2, c2 in center_cells))
                center_cols = sorted(set(c2 for r2, c2 in center_cells))
                cr = (center_rmin + center_rmax) // 2
                cc = (center_cmin + center_cmax) // 2

                all_boxes.append({
                    'color': v,
                    'rmin': rmin, 'rmax': rmax,
                    'cmin': cmin, 'cmax': cmax,
                    'cr': cr, 'cc': cc,
                    'center_rows': center_rows,
                    'center_cols': center_cols,
                })

    # Group boxes by color
    boxes_by_color = defaultdict(list)
    for box in all_boxes:
        boxes_by_color[box['color']].append(box)

    # Trace the chain: start from the first color, find nearest matching box at each step
    if not code_seq or code_seq[0] not in boxes_by_color:
        return g

    # Find starting box: the first box with the first color
    # Pick the one with smallest row, then smallest col
    current_box = min(boxes_by_color[code_seq[0]], key=lambda b: (b['cr'], b['cc']))
    used_boxes = {id(current_box)}

    for i in range(len(code_seq) - 1):
        a_color = code_seq[i]
        b_color = code_seq[i + 1]

        # Find nearest box with b_color that hasn't been used yet
        candidates = [b for b in boxes_by_color.get(b_color, []) if id(b) not in used_boxes]
        if not candidates:
            continue

        # Pick nearest by Manhattan distance from current box center
        target_box = min(candidates,
                        key=lambda b: abs(b['cr'] - current_box['cr']) + abs(b['cc'] - current_box['cc']))
        used_boxes.add(id(target_box))

        a = current_box
        b = target_box

        # Draw connection from a to b using a's color
        if abs(b['cr'] - a['cr']) > abs(b['cc'] - a['cc']):
            # Vertical connection
            col_lo = min(a['center_cols'])
            col_hi = max(a['center_cols'])
            if b['cr'] > a['cr']:
                for r2 in range(a['rmax'] + 1, b['rmin']):
                    for c2 in range(col_lo, col_hi + 1):
                        g[r2][c2] = a['color']
            else:
                for r2 in range(b['rmax'] + 1, a['rmin']):
                    for c2 in range(col_lo, col_hi + 1):
                        g[r2][c2] = a['color']
        else:
            # Horizontal connection
            row_lo = min(a['center_rows'])
            row_hi = max(a['center_rows'])
            if b['cc'] > a['cc']:
                for c2 in range(a['cmax'] + 1, b['cmin']):
                    for r2 in range(row_lo, row_hi + 1):
                        g[r2][c2] = a['color']
            else:
                for c2 in range(b['cmax'] + 1, a['cmin']):
                    for r2 in range(row_lo, row_hi + 1):
                        g[r2][c2] = a['color']

        current_box = target_box

    return g


# ============================================================
# 36a08778: Extend lines from endpoints of colored segments to borders
# ============================================================
def solve_36a08778(grid):
    R, C = len(grid), len(grid[0])
    g = [row[:] for row in grid]
    bg = 7

    # Find non-7 segments (colored horizontal lines and 6-columns/67-patterns)
    # There are 6-lines (vertical columns of 6) and colored horizontal segments
    # The 6-lines extend downward/upward to connect to colored segments

    # Find all 6-cells that are single-cell columns (vertical lines)
    # and colored horizontal segments (2222, etc.)

    # Actually, looking at the training data:
    # The grid has 6 and 7 as main values, plus colored segments (2222, etc.)
    # 6 forms vertical lines connecting to colored segments.
    # The transformation extends 6-lines to connect all segments into a tree.

    # Approach: find all non-7 cells. The 6s and colored segments form a tree.
    # Extend the tree by growing 6-lines from segment endpoints toward other segments.

    # Find all non-7 connected components
    visited = [[False]*C for _ in range(R)]
    components = []

    def flood(sr, sc):
        stack = [(sr, sc)]
        cells = set()
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= R or c < 0 or c >= C: continue
            if (r, c) in cells: continue
            if grid[r][c] == bg: continue
            cells.add((r, c))
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                stack.append((r+dr, c+dc))
        return cells

    for r in range(R):
        for c in range(C):
            if not visited[r][c] and grid[r][c] != bg:
                cells = flood(r, c)
                for cr, cc in cells:
                    visited[cr][cc] = True
                components.append(cells)

    # The segments are horizontal runs of non-6, non-7 values.
    # The 6-columns are vertical runs of 6.
    # Need to connect them.

    # For each component, find its colored segments (non-6, non-7)
    # and 6-columns.

    # Hmm, actually looking at the outputs, it seems like each colored segment
    # grows a vertical 6-line downward until it meets another segment.

    # Let me try a different approach: identify the "tree" structure.
    # Each colored segment (horizontal line of same non-7, non-6 value)
    # has a 6-line going from one of its endpoints downward/upward.
    # In the output, these 6-lines extend to connect to the next segment.

    # The rule seems to be:
    # 1. Find 6-columns (vertical pairs of adjacent 6 values or 6-7-6 patterns)
    # 2. Extend each 6-column downward until it reaches a colored segment
    # 3. At the colored segment, turn horizontally

    # Actually the simplest interpretation from the training data:
    # Find each 6-cell that has only bg (7) neighbors on one side.
    # Extend the 6 in that direction until it hits another non-7 cell.
    # At that cell, draw the appropriate connection.

    # Let me try iterative growth: repeatedly find 6-cells at the boundary
    # of the non-7 region and grow them toward nearby colored segments.

    # Simpler: for each colored segment, find the nearest 6-column
    # and extend the 6-column to reach this segment.

    # I think the actual rule is:
    # There's a "root" segment at the top. 6-lines grow downward.
    # At each level, a new colored segment is reached and a 6-line
    # turns into the segment's column.

    # This is getting complex. Let me look for a pattern:
    # The 6s always form a VERTICAL connection line between two colored segments.
    # The connection goes from the endpoint of one segment DOWN to the start of the next.

    # For each pair of vertically adjacent segments (one above, one below),
    # a vertical 6-line connects them.

    # In the output, additional 6-lines are added to create a tree structure
    # where all segments are connected.

    # Let me just try: grow 6s vertically from existing 6-cells until they hit
    # a non-7, non-6 cell or the grid boundary.

    changed = True
    while changed:
        changed = False
        for r in range(R):
            for c in range(C):
                if g[r][c] == 6:
                    # Try to extend downward
                    if r + 1 < R and g[r+1][c] == bg:
                        # Check if there's a non-7 cell below in this column
                        has_target = False
                        for r2 in range(r+1, R):
                            if g[r2][c] != bg:
                                has_target = True
                                break
                        if has_target:
                            g[r+1][c] = 6
                            changed = True
                    # Try to extend upward
                    if r - 1 >= 0 and g[r-1][c] == bg:
                        has_target = False
                        for r2 in range(r-1, -1, -1):
                            if g[r2][c] != bg:
                                has_target = True
                                break
                        if has_target:
                            g[r-1][c] = 6
                            changed = True

    return g


# ============================================================
# 332f06d7: Path of 1s on bg of 3s with 0 and 2 markers.
# Move the 0 block along the path by its size (area) steps.
# 2->0 if 0 reaches it, 0->1.
# ============================================================
def solve_332f06d7(grid):
    R, C = len(grid), len(grid[0])
    g = [row[:] for row in grid]

    # Find 0 cells and 2 cells
    zero_cells = set()
    two_cells = set()
    one_cells = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 0:
                zero_cells.add((r, c))
            elif grid[r][c] == 2:
                two_cells.add((r, c))
            elif grid[r][c] == 1:
                one_cells.add((r, c))

    if not zero_cells:
        return g

    # The 0 block has a specific size and shape.
    # It moves along the path (1s) by 'area' steps toward the 2.

    # The path consists of 0, 1, and 2 cells (all non-3).
    path_cells = zero_cells | one_cells | two_cells

    # Build adjacency for the path
    adj = defaultdict(list)
    for r, c in path_cells:
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if (nr, nc) in path_cells:
                adj[(r,c)].append((nr, nc))

    # The 0 block is a connected component
    # Find the 0-component
    zero_visited = set()
    zero_queue = [next(iter(zero_cells))]
    zero_visited.add(zero_queue[0])
    while zero_queue:
        r, c = zero_queue.pop(0)
        for nr, nc in adj[(r,c)]:
            if (nr, nc) in zero_cells and (nr, nc) not in zero_visited:
                zero_visited.add((nr, nc))
                zero_queue.append((nr, nc))

    zero_block = zero_visited
    zero_area = len(zero_block)

    # Find the boundary of the 0-block that touches 1-cells
    # and determine the direction of movement
    zero_boundary_toward_2 = set()
    for r, c in zero_block:
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if (nr, nc) in one_cells:
                zero_boundary_toward_2.add((nr, nc))

    # Find direction from 0-block toward 2-block
    # The 0-block has two ends on the path. One leads to 2, one leads to edge.
    # We need to move the 0-block along the path toward 2 by 'area' steps.

    # Actually: find the shortest path from 0-block to 2-block along 1-cells.
    # Then move 0-block 'area' steps along that path.

    # Simpler: convert the path to a 1D sequence.
    # The path is tree-like (L-shaped in some cases).
    # Actually, for the 0 block, it occupies a rectangular region.

    # Let me try a simpler approach:
    # Find the 0-block bounding box. Find the direction it can move (has 1s adjacent).
    # Move it 'area' cells in that direction. Fill old 0 with 1, new 1 with 0.
    # If new 0 overlaps 2, convert 2 to 0.

    zero_rmin = min(r for r, c in zero_block)
    zero_rmax = max(r for r, c in zero_block)
    zero_cmin = min(c for r, c in zero_block)
    zero_cmax = max(c for r, c in zero_block)
    bh = zero_rmax - zero_rmin + 1
    bw = zero_cmax - zero_cmin + 1

    # Find which direction has 1s (or 2s) adjacent to the block
    # Check all 4 directions
    possible_dirs = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        # Check if all cells on the leading edge have 1s/2s adjacent
        if dr == -1:
            edge = [(zero_rmin, c) for c in range(zero_cmin, zero_cmax+1)]
            nexts = [(zero_rmin + dr, c) for c in range(zero_cmin, zero_cmax+1)]
        elif dr == 1:
            edge = [(zero_rmax, c) for c in range(zero_cmin, zero_cmax+1)]
            nexts = [(zero_rmax + dr, c) for c in range(zero_cmin, zero_cmax+1)]
        elif dc == -1:
            edge = [(r, zero_cmin) for r in range(zero_rmin, zero_rmax+1)]
            nexts = [(r, zero_cmin + dc) for r in range(zero_rmin, zero_rmax+1)]
        elif dc == 1:
            edge = [(r, zero_cmax) for r in range(zero_rmin, zero_rmax+1)]
            nexts = [(r, zero_cmax + dc) for r in range(zero_rmin, zero_rmax+1)]

        all_valid = all(0 <= nr < R and 0 <= nc < C and grid[nr][nc] in [1, 2]
                       for nr, nc in nexts)
        if all_valid:
            possible_dirs.append((dr, dc))

    if not possible_dirs:
        return g

    # If multiple directions, pick the one toward 2
    best_dir = possible_dirs[0]
    if two_cells and len(possible_dirs) > 1:
        two_center = (sum(r for r,c in two_cells)/len(two_cells),
                      sum(c for r,c in two_cells)/len(two_cells))
        zero_center = ((zero_rmin+zero_rmax)/2, (zero_cmin+zero_cmax)/2)
        best_dist = float('inf')
        for dr, dc in possible_dirs:
            new_center = (zero_center[0] + dr, zero_center[1] + dc)
            dist = abs(new_center[0] - two_center[0]) + abs(new_center[1] - two_center[1])
            if dist < best_dist:
                best_dist = dist
                best_dir = (dr, dc)

    dr, dc = best_dir

    # Move the block 'area' steps (or until it overlaps 2 or hits edge)
    steps = zero_area
    new_rmin = zero_rmin + dr * steps
    new_rmax = zero_rmax + dr * steps
    new_cmin = zero_cmin + dc * steps
    new_cmax = zero_cmax + dc * steps

    # Check bounds
    if new_rmin < 0 or new_rmax >= R or new_cmin < 0 or new_cmax >= C:
        # Reduce steps
        while steps > 0:
            new_rmin = zero_rmin + dr * steps
            new_rmax = zero_rmax + dr * steps
            new_cmin = zero_cmin + dc * steps
            new_cmax = zero_cmax + dc * steps
            if 0 <= new_rmin and new_rmax < R and 0 <= new_cmin and new_cmax < C:
                break
            steps -= 1

    # Fill old 0-block with 1
    for r, c in zero_block:
        g[r][c] = 1

    # Fill new position with 0
    for r in range(new_rmin, new_rmax+1):
        for c in range(new_cmin, new_cmax+1):
            g[r][c] = 0

    # If 2 cells become 0, that's fine (they're now covered by 0)
    # If 2 cells are not covered, keep them as 2
    # (2 cells not in the new 0-block area stay as they were)
    for r, c in two_cells:
        if not (new_rmin <= r <= new_rmax and new_cmin <= c <= new_cmax):
            g[r][c] = 2

    return g


# ============================================================
# 13e47133: Cross of 2s divides grid into regions.
# Each region gets concentric rectangles based on corner markers.
# ============================================================
def solve_13e47133(grid):
    R, C = len(grid), len(grid[0])
    g = [row[:] for row in grid]

    # Find the separator value (forms cross/L lines)
    # The separator value has full columns or rows spanning the grid
    bg_vals = Counter(grid[r][c] for r in range(R) for c in range(C))

    # Find value that forms the most full columns + has connecting horizontal segments
    sep = None
    best_score = 0
    for v in range(10):
        full_cols = sum(1 for c in range(C) if all(grid[r][c] == v for r in range(R)))
        full_rows = sum(1 for r in range(R) if all(grid[r][c] == v for c in range(C)))
        # Also count partial rows/cols that connect
        partial = sum(1 for r in range(R) for c in range(C) if grid[r][c] == v)
        score = full_cols * 100 + full_rows * 100 + partial
        # Must have at least one full col or row
        if (full_cols > 0 or full_rows > 0) and score > best_score:
            # Don't pick the most common value (background)
            if bg_vals[v] < R * C * 0.5:
                best_score = score
                sep = v

    if sep is None:
        return g

    bg = max((v for v in bg_vals if v != sep), key=lambda v: bg_vals[v])

    # Find connected regions of non-separator cells
    visited = [[False]*C for _ in range(R)]
    components = []

    def flood_fill(sr, sc):
        stack = [(sr, sc)]
        cells = []
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= R or c < 0 or c >= C: continue
            if visited[r][c]: continue
            if grid[r][c] == sep: continue
            visited[r][c] = True
            cells.append((r, c))
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                stack.append((r+dr, c+dc))
        return cells

    for r in range(R):
        for c in range(C):
            if not visited[r][c] and grid[r][c] != sep:
                cells = flood_fill(r, c)
                if cells:
                    components.append(cells)

    # Sort regions by size (largest first) so larger regions are filled first
    # and smaller regions can overwrite
    components.sort(key=lambda x: -len(x))

    for comp in components:
        rmin = min(r for r, c in comp)
        rmax = max(r for r, c in comp)
        cmin = min(c for r, c in comp)
        cmax = max(c for r, c in comp)

        # Find markers (non-bg, non-sep)
        comp_markers = [(r, c, grid[r][c]) for r, c in comp if grid[r][c] != bg]

        marker_colors = list(set(v for _, _, v in comp_markers))

        if len(marker_colors) == 0:
            # No markers - fill with bg
            for r in range(rmin, rmax+1):
                for c in range(cmin, cmax+1):
                    if grid[r][c] != sep:
                        g[r][c] = bg
        elif len(marker_colors) == 1:
            # Single marker - use marker and bg as two concentric colors
            marker_color = marker_colors[0]
            outer, inner = bg, marker_color

            comp_set2 = set((r, c) for r, c in comp)
            for r, c in comp:
                d_up = 0
                rr = r - 1
                while rr >= 0 and (rr, c) in comp_set2: d_up += 1; rr -= 1
                d_down = 0
                rr = r + 1
                while rr < R and (rr, c) in comp_set2: d_down += 1; rr += 1
                d_left = 0
                cc = c - 1
                while cc >= 0 and (r, cc) in comp_set2: d_left += 1; cc -= 1
                d_right = 0
                cc = c + 1
                while cc < C and (r, cc) in comp_set2: d_right += 1; cc += 1
                dd = min(d_up, d_down, d_left, d_right)
                if dd % 2 == 0:
                    g[r][c] = outer
                else:
                    g[r][c] = inner
        else:
            # Two markers - concentric rectangles
            # Distance is computed from nearest separator/grid edge, not bbox
            c1, c2 = marker_colors[0], marker_colors[1]
            pos1 = [(r, c) for r, c, v in comp_markers if v == c1]
            pos2 = [(r, c) for r, c, v in comp_markers if v == c2]
            corners = [(rmin, cmin), (rmin, cmax), (rmax, cmin), (rmax, cmax)]

            def min_corner_dist(positions):
                return min(abs(r-cr)+abs(c-cc) for r, c in positions for cr, cc in corners)

            d1 = min_corner_dist(pos1)
            d2 = min_corner_dist(pos2)
            outer = c1 if d1 <= d2 else c2
            inner = c2 if d1 <= d2 else c1

            comp_set = set((r, c) for r, c in comp)

            # Compute chebyshev distance (min of 4 directional distances to edge)
            for r, c in comp:
                # Distance to nearest non-region cell in each direction
                d_up = 0
                rr = r - 1
                while rr >= 0 and (rr, c) in comp_set:
                    d_up += 1
                    rr -= 1

                d_down = 0
                rr = r + 1
                while rr < R and (rr, c) in comp_set:
                    d_down += 1
                    rr += 1

                d_left = 0
                cc = c - 1
                while cc >= 0 and (r, cc) in comp_set:
                    d_left += 1
                    cc -= 1

                d_right = 0
                cc = c + 1
                while cc < C and (r, cc) in comp_set:
                    d_right += 1
                    cc += 1

                d = min(d_up, d_down, d_left, d_right)
                if d % 2 == 0:
                    g[r][c] = outer
                else:
                    g[r][c] = inner

    # Redraw separator lines
    for r in range(R):
        for c in range(C):
            if grid[r][c] == sep:
                g[r][c] = sep

    return g


# ============================================================
# 271d71e2: Rectangles on bg=6 with 9-markers indicating shift.
# Shift content one step toward the 9-side.
# ============================================================
def solve_271d71e2(grid):
    R, C = len(grid), len(grid[0])
    g = [row[:] for row in grid]
    bg_val = 6

    visited = [[False]*C for _ in range(R)]
    rects = []

    def flood(sr, sc):
        stack = [(sr, sc)]
        cells = []
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= R or c < 0 or c >= C: continue
            if visited[r][c]: continue
            if grid[r][c] == bg_val: continue
            visited[r][c] = True
            cells.append((r, c))
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                stack.append((r+dr, c+dc))
        return cells

    for r in range(R):
        for c in range(C):
            if not visited[r][c] and grid[r][c] != bg_val:
                cells = flood(r, c)
                if cells:
                    rects.append(cells)

    for cells in rects:
        rmin = min(r for r, c in cells)
        rmax = max(r for r, c in cells)
        cmin = min(c for r, c in cells)
        cmax = max(c for r, c in cells)
        w = cmax - cmin + 1
        h = rmax - rmin + 1

        nine_top = sum(1 for c2 in range(cmin, cmax+1) if grid[rmin][c2] == 9) / max(w, 1)
        nine_bot = sum(1 for c2 in range(cmin, cmax+1) if grid[rmax][c2] == 9) / max(w, 1)
        nine_left = sum(1 for r2 in range(rmin, rmax+1) if grid[r2][cmin] == 9) / max(h, 1)
        nine_right = sum(1 for r2 in range(rmin, rmax+1) if grid[r2][cmax] == 9) / max(h, 1)

        nine_counts = {'top': nine_top, 'bot': nine_bot, 'left': nine_left, 'right': nine_right}
        direction = max(nine_counts, key=nine_counts.get)

        if nine_counts[direction] < 0.3:
            continue

        if direction == 'top':
            for r2 in range(rmin, rmax):
                for c2 in range(cmin, cmax+1):
                    g[r2][c2] = grid[r2+1][c2]
            for c2 in range(cmin, cmax+1):
                g[rmax][c2] = bg_val
        elif direction == 'bot':
            for r2 in range(rmax, rmin, -1):
                for c2 in range(cmin, cmax+1):
                    g[r2][c2] = grid[r2-1][c2]
            for c2 in range(cmin, cmax+1):
                g[rmin][c2] = bg_val
        elif direction == 'left':
            for c2 in range(cmin, cmax):
                for r2 in range(rmin, rmax+1):
                    g[r2][c2] = grid[r2][c2+1]
            for r2 in range(rmin, rmax+1):
                g[r2][cmax] = bg_val
        elif direction == 'right':
            for c2 in range(cmax, cmin, -1):
                for r2 in range(rmin, rmax+1):
                    g[r2][c2] = grid[r2][c2-1]
            for r2 in range(rmin, rmax+1):
                g[r2][cmin] = bg_val

    return g


# ============================================================
# Solver registry
# ============================================================
SOLVERS = {
    '2ba387bc': solve_2ba387bc,
    '38007db0': solve_38007db0,
    '3e6067c3': solve_3e6067c3,
}

def main():
    solutions = {}

    for tid in TASK_IDS:
        if tid not in SOLVERS:
            print(f"  {tid}: no solver")
            continue

        solve_fn = SOLVERS[tid]
        try:
            passed = test_solve(tid, solve_fn)
        except Exception as e:
            import traceback
            print(f"  {tid}: ERROR - {e}")
            traceback.print_exc()
            continue

        if passed:
            print(f"  {tid}: PASS")
            results = apply_solve(tid, solve_fn)
            solutions[tid] = results
        else:
            print(f"  {tid}: FAIL")

    print(f"\nTotal solutions: {len(solutions)}/{len(TASK_IDS)}")

    os.makedirs(os.path.dirname(OUTPUT_FILE) if os.path.dirname(OUTPUT_FILE) else '.', exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(solutions, f)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
