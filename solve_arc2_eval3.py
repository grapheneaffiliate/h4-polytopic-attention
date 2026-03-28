import json
import copy
from collections import Counter, defaultdict

def load_task(tid):
    with open(f'data/arc2/{tid}.json') as f:
        return json.load(f)

def test_solve(solve_fn, task):
    for i, p in enumerate(task['train']):
        result = solve_fn(p['input'])
        if result != p['output']:
            return False
    return True

def apply_solve(solve_fn, task):
    results = []
    for p in task['test']:
        results.append(solve_fn(p['input']))
    return results

# ============================================================
# dbff022c: Key-fill mapping for rectangle interiors
# Grid has hollow rectangles and a separate key (stacked pairs).
# Key pair (a,b) means: rect with border a gets interior 0s filled with b.
# ============================================================
def solve_dbff022c(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # Step 1: Find connected components
    visited = [[False]*W for _ in range(H)]
    components = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0 and not visited[r][c]:
                q = [(r,c)]
                visited[r][c] = True
                cells = [(r,c)]
                idx = 0
                while idx < len(q):
                    cr, cc = q[idx]; idx += 1
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<H and 0<=nc<W and not visited[nr][nc] and grid[nr][nc] != 0:
                            visited[nr][nc] = True
                            q.append((nr,nc))
                            cells.append((nr,nc))
                components.append(cells)

    # Step 2: Identify the KEY component
    # The key is a block of colored cells forming (border, fill) pairs
    # It can be 2xN (column-wise pairs) or Nx2 (row-wise pairs)
    key_mapping = {}

    for ci, comp in enumerate(components):
        minr = min(r for r,c in comp)
        maxr = max(r for r,c in comp)
        minc = min(c for r,c in comp)
        maxc = max(c for r,c in comp)
        h = maxr - minr + 1
        w = maxc - minc + 1

        # Try 2-row (column-wise pairs): each column is (border, fill)
        if h == 2 and w >= 2:
            pairs = []
            valid = True
            for c in range(minc, maxc + 1):
                top_val = grid[minr][c] if grid[minr][c] != 0 else None
                bot_val = grid[maxr][c] if grid[maxr][c] != 0 else None
                if top_val is not None and bot_val is not None:
                    pairs.append((top_val, bot_val))
                else:
                    valid = False
                    break
            if valid and len(pairs) >= 2:
                for t, b in pairs:
                    key_mapping[t] = b
                break

        # Try 2-col (row-wise pairs): each row is (border, fill)
        if w == 2 and h >= 2:
            pairs = []
            valid = True
            for r in range(minr, maxr + 1):
                left_val = grid[r][minc] if grid[r][minc] != 0 else None
                right_val = grid[r][maxc] if grid[r][maxc] != 0 else None
                if left_val is not None and right_val is not None:
                    pairs.append((left_val, right_val))
                else:
                    valid = False
                    break
            if valid and len(pairs) >= 2:
                for t, b in pairs:
                    key_mapping[t] = b
                break

    # Step 3: Flood fill from boundary to find external 0s
    external = [[False]*W for _ in range(H)]
    bfs = []
    for r in range(H):
        for c in range(W):
            if (r == 0 or r == H-1 or c == 0 or c == W-1) and grid[r][c] == 0:
                if not external[r][c]:
                    external[r][c] = True
                    bfs.append((r, c))

    idx = 0
    while idx < len(bfs):
        cr, cc = bfs[idx]; idx += 1
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr+dr, cc+dc
            if 0<=nr<H and 0<=nc<W and not external[nr][nc] and grid[nr][nc] == 0:
                external[nr][nc] = True
                bfs.append((nr, nc))

    # Step 4: Fill interior 0s based on enclosing shape's key mapping
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 0 and not external[r][c]:
                # Find enclosing shape color by searching nearest non-0 cell
                enclosing_color = None
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0<=nr<H and 0<=nc<W and grid[nr][nc] != 0:
                        enclosing_color = grid[nr][nc]
                        break
                if enclosing_color is None:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        for dist in range(1, max(H, W)):
                            nr, nc = r + dr*dist, c + dc*dist
                            if 0<=nr<H and 0<=nc<W and grid[nr][nc] != 0:
                                enclosing_color = grid[nr][nc]
                                break
                            if not (0<=nr<H and 0<=nc<W):
                                break
                        if enclosing_color:
                            break

                if enclosing_color and enclosing_color in key_mapping:
                    out[r][c] = key_mapping[enclosing_color]

    return out


# ============================================================
# c4d067a0: Template dots + stamp blocks
# Grid bg=8. Has scattered single pixels (template) and blocks of same color (stamps).
# Each template pixel maps to a stamp position.
# Template pixels not matching existing stamps generate new stamps.
# ============================================================
def solve_c4d067a0(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    # Detect bg as most common value
    all_vals = [grid[r][c] for r in range(H) for c in range(W)]
    bg = Counter(all_vals).most_common(1)[0][0]

    # Find non-bg cells
    non_bg = [(r, c, grid[r][c]) for r in range(H) for c in range(W) if grid[r][c] != bg]

    if not non_bg:
        return out

    # Find connected components
    visited = set()
    components = []
    non_bg_set = set((r,c) for r,c,v in non_bg)

    for r, c, v in non_bg:
        if (r,c) not in visited:
            q = [(r,c)]
            visited.add((r,c))
            cells = [(r,c)]
            idx = 0
            while idx < len(q):
                cr, cc = q[idx]; idx += 1
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = cr+dr, cc+dc
                    if (nr,nc) in non_bg_set and (nr,nc) not in visited:
                        visited.add((nr,nc))
                        q.append((nr,nc))
                        cells.append((nr,nc))
            components.append(cells)

    # Separate single pixels (template dots) from multi-cell blocks (stamps)
    dots = []   # (r, c, color)
    stamps = []  # (minr, minc, maxr, maxc, color, cells)

    for comp in components:
        if len(comp) == 1:
            r, c = comp[0]
            dots.append((r, c, grid[r][c]))
        else:
            minr = min(r for r,c in comp)
            maxr = max(r for r,c in comp)
            minc = min(c for r,c in comp)
            maxc = max(c for r,c in comp)
            colors = set(grid[r][c] for r,c in comp)
            if len(colors) == 1:
                stamps.append((minr, minc, maxr, maxc, list(colors)[0], comp))

    if not stamps or not dots:
        return out

    # Find the stamp shape (all stamps should have the same size)
    stamp_h = stamps[0][2] - stamps[0][0] + 1
    stamp_w = stamps[0][3] - stamps[0][1] + 1

    # All stamp positions (regardless of color)
    all_stamp_positions = sorted([(s[0], s[1]) for s in stamps])

    # Find template dot rows/cols
    all_dot_rows = sorted(set(r for r,c,v in dots))
    all_dot_cols = sorted(set(c for r,c,v in dots))

    if len(all_dot_rows) >= 2:
        template_row_step = all_dot_rows[1] - all_dot_rows[0]
    else:
        template_row_step = 2

    if len(all_dot_cols) >= 2:
        template_col_step = all_dot_cols[1] - all_dot_cols[0]
    else:
        template_col_step = 2

    # Find stamp spacing
    all_stamp_rows = sorted(set(r for r,c in all_stamp_positions))
    all_stamp_cols = sorted(set(c for r,c in all_stamp_positions))

    if len(all_stamp_cols) >= 2:
        stamp_col_step = all_stamp_cols[1] - all_stamp_cols[0]
    else:
        stamp_col_step = stamp_w + 2

    if len(all_stamp_rows) >= 2:
        stamp_row_step = all_stamp_rows[1] - all_stamp_rows[0]
    else:
        stamp_row_step = stamp_col_step  # assume square spacing

    # Find anchor: a template dot whose color matches a stamp color
    # and whose position maps to a stamp position
    anchor_dot = None
    anchor_stamp = None

    # Build a map of stamp positions to stamp info
    stamp_pos_map = {}
    for s in stamps:
        stamp_pos_map[(s[0], s[1])] = s

    # Find the correct anchor: the dot closest to the stamps with matching color
    # Try all dot-stamp pairs with matching color and pick the one that
    # produces the most valid stamp positions (within grid bounds)

    best_anchor = None
    best_score = -1

    for s in stamps:
        scolor = s[4]
        matching_dots = [(r,c) for r,c,v in dots if v == scolor]
        for dr, dc in matching_dots:
            # Test this anchor pairing
            score = 0
            for d_r, d_c, d_color in dots:
                rel_r = (d_r - dr) / template_row_step if template_row_step != 0 else 0
                rel_c = (d_c - dc) / template_col_step if template_col_step != 0 else 0
                sr = int(round(s[0] + rel_r * stamp_row_step))
                sc = int(round(s[1] + rel_c * stamp_col_step))
                if 0 <= sr < H and 0 <= sc < W and 0 <= sr + stamp_h - 1 < H and 0 <= sc + stamp_w - 1 < W:
                    score += 1
            if score > best_score:
                best_score = score
                best_anchor = ((dr, dc), (s[0], s[1]))

    if best_anchor:
        anchor_dot, anchor_stamp = best_anchor
    else:
        return out

    if not anchor_dot:
        return out

    # Now map every dot to a stamp position
    for dr, dc, color in dots:
        # Relative position in template grid
        rel_r = (dr - anchor_dot[0]) / template_row_step if template_row_step != 0 else 0
        rel_c = (dc - anchor_dot[1]) / template_col_step if template_col_step != 0 else 0

        sr = int(round(anchor_stamp[0] + rel_r * stamp_row_step))
        sc = int(round(anchor_stamp[1] + rel_c * stamp_col_step))

        # Check if stamp already exists
        if (sr, sc) in stamp_pos_map:
            continue

        # Draw new stamp
        for r in range(stamp_h):
            for c in range(stamp_w):
                nr, nc = sr + r, sc + c
                if 0 <= nr < H and 0 <= nc < W:
                    out[nr][nc] = color

    return out


# ============================================================
# db695cfb: Diagonal lines through same-color pixel pairs
# Pixels of same color on same diagonal get connected by that diagonal line.
# Single pixels on intersection with other color's line generate perpendicular lines.
# ============================================================
def solve_db695cfb(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    bg = grid[0][0]

    # Find non-bg pixels
    pixels = [(r, c, grid[r][c]) for r in range(H) for c in range(W) if grid[r][c] != bg]

    by_color = defaultdict(list)
    for r, c, v in pixels:
        by_color[v].append((r, c))

    # For each color, check if any pair of pixels shares a diagonal
    # (r+c same for '/' or r-c same for '\')
    # But only if those pixels are NOT on another color's diagonal line
    lines_to_draw = []  # (diagonal_type, diagonal_value, color)

    # First pass: find all potential lines
    potential_lines = []
    for color, plist in by_color.items():
        if len(plist) < 2:
            continue

        slash = defaultdict(list)
        for r, c in plist:
            slash[r+c].append((r, c))
        for key, group in slash.items():
            if len(group) >= 2:
                potential_lines.append(('/', key, color, group))

        backslash = defaultdict(list)
        for r, c in plist:
            backslash[r-c].append((r, c))
        for key, group in backslash.items():
            if len(group) >= 2:
                potential_lines.append(('\\', key, color, group))

    # Check if a potential line's pixels are all on another color's line
    # of the SAME type and value. If so, suppress this line.
    # The line with MORE pixels on the diagonal wins. If equal, the one whose
    # pixels form the outermost pair wins (i.e., the one that "owns" the diagonal).
    for diag_type, diag_val, color, group in potential_lines:
        suppressed = False
        for dt2, dv2, col2, g2 in potential_lines:
            if col2 != color and dt2 == diag_type and dv2 == diag_val:
                # Same diagonal, different color
                # Suppress if ALL our pixels lie strictly BETWEEN the other's endpoints
                other_rows = [r for r, c in g2]
                our_rows = [r for r, c in group]
                if min(our_rows) >= min(other_rows) and max(our_rows) <= max(other_rows):
                    # Our pixels are contained within the other's range
                    suppressed = True
                    break

        if not suppressed:
            lines_to_draw.append((diag_type, diag_val, color))

    # Draw the diagonal lines - only between the paired pixels
    drawn = set()
    for diag_type, diag_val, color in lines_to_draw:
        # Find the pixel endpoints on this diagonal
        if diag_type == '/':
            pts = [(r, c) for r, c in by_color[color] if r + c == diag_val]
        else:
            pts = [(r, c) for r, c in by_color[color] if r - c == diag_val]

        if len(pts) >= 2:
            min_r = min(r for r, c in pts)
            max_r = max(r for r, c in pts)
        else:
            min_r = 0
            max_r = H - 1

        for r in range(min_r, max_r + 1):
            if diag_type == '/':
                c = diag_val - r
            else:
                c = r - diag_val
            if 0 <= c < W:
                if out[r][c] == bg:
                    out[r][c] = color
                drawn.add((r, c))

    # Now check for ALL pixels that lie on a DRAWN line of a different color
    # These intersection pixels generate perpendicular lines
    for color, plist in by_color.items():
        for r, c in plist:
            for diag_type, diag_val, line_color in lines_to_draw:
                if line_color != color:
                    if diag_type == '/' and r + c == diag_val:
                        # On '/' line, draw '\' through this pixel (full extent)
                        perp_val = r - c
                        for r2 in range(H):
                            c2 = r2 - perp_val
                            if 0 <= c2 < W and out[r2][c2] == bg:
                                out[r2][c2] = color
                    elif diag_type == '\\' and r - c == diag_val:
                        # On '\' line, draw '/' through this pixel (full extent)
                        perp_val = r + c
                        for r2 in range(H):
                            c2 = perp_val - r2
                            if 0 <= c2 < W and out[r2][c2] == bg:
                                out[r2][c2] = color

    return out


# ============================================================
# e376de54: Parallel lines equalized to median length
# Background 7. Non-7 pixels form parallel line segments.
# All segments get equalized to the median segment length.
# ============================================================
def solve_e376de54(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    bg = 7

    non_bg = [(r, c, grid[r][c]) for r in range(H) for c in range(W) if grid[r][c] != bg]
    if not non_bg:
        return out

    # Detect line direction and group lines
    by_color = defaultdict(list)
    for r, c, v in non_bg:
        by_color[v].append((r, c))

    # Group by possible line types: horizontal, vertical, '/' diagonal, '\' diagonal
    # Try '/' diagonal first (r+c = const)
    all_slash = defaultdict(list)
    for r, c, v in non_bg:
        all_slash[r+c].append((r, c, v))

    all_backslash = defaultdict(list)
    for r, c, v in non_bg:
        all_backslash[r-c].append((r, c, v))

    by_row = defaultdict(list)
    for r, c, v in non_bg:
        by_row[r].append((c, v))

    by_col = defaultdict(list)
    for r, c, v in non_bg:
        by_col[c].append((r, v))

    # Determine which grouping produces the best line structure
    # (most cells on lines of length >= 2)
    def score_grouping(groups):
        return sum(len(g) for g in groups.values() if len(g) >= 2)

    scores = [
        ('h', score_grouping(by_row), by_row),
        ('v', score_grouping(by_col), by_col),
        ('/', score_grouping(all_slash), all_slash),
        ('\\', score_grouping(all_backslash), all_backslash),
    ]
    scores.sort(key=lambda x: -x[1])
    best_type = scores[0][0]

    # Extract lines based on best grouping
    lines = []  # (key, start_r, start_c, length, color)

    if best_type == 'h':
        for r, cells in by_row.items():
            cells.sort()
            lines.append((r, r, cells[0][0], len(cells), cells[0][1]))
    elif best_type == 'v':
        for c, cells in by_col.items():
            cells.sort()
            lines.append((c, cells[0][0], c, len(cells), cells[0][1]))
    elif best_type == '/':
        for key, cells in all_slash.items():
            cells.sort()
            lines.append((key, cells[0][0], cells[0][1], len(cells), cells[0][2]))
    else:  # '\'
        for key, cells in all_backslash.items():
            cells.sort()
            lines.append((key, cells[0][0], cells[0][1], len(cells), cells[0][2]))

    if not lines:
        return out

    # Compute target length: median of per-color max line lengths
    color_max = defaultdict(int)
    for _, _, _, length, color in lines:
        color_max[color] = max(color_max[color], length)

    if len(color_max) >= 2:
        target_len = sorted(color_max.values())[len(color_max) // 2]
    else:
        # Single color: median of all line lengths
        lengths = sorted(l[3] for l in lines)
        target_len = lengths[len(lengths) // 2]

    # Erase all non-bg
    for r, c, v in non_bg:
        out[r][c] = bg

    # Compute common start reference
    if best_type == 'h':
        start_ref = min(l[2] for l in lines)  # min start col
        for _, r, sc, length, color in lines:
            for c in range(start_ref, start_ref + target_len):
                if 0 <= c < W:
                    out[r][c] = color
    elif best_type == 'v':
        start_ref = min(l[1] for l in lines)  # min start row
        for _, sr, c, length, color in lines:
            for r in range(start_ref, start_ref + target_len):
                if 0 <= r < H:
                    out[r][c] = color
    elif best_type == '/':
        # '/' lines: r increases, c decreases. Lines sorted by key (r+c).
        # Need to find common reference for start positions.
        # The start position (top-right) of each line should be evenly spaced.
        # Find the bottom-left end point of each line and align from there.

        # Sort lines by key (r+c value)
        lines.sort(key=lambda l: l[0])

        # The bottom-left point of each line stays fixed? Or the center?
        # From train 0: the bottom-end (max r) of each line is preserved.
        # Line r+c=7: input ends at (7,0), output also ends at (7,0). ✓
        # Line r+c=11: input ends at (9,2), output also ends at (9,2). ✓
        # Line r+c=15: input ends at (11,4), output ends at (11,4). ✓
        # Line r+c=19: input ends at (13,6), output ends at (13,6). ✓
        # Line r+c=23: input ends at (15,8), output ends at (15,8). ✓

        # So: the bottom-left endpoint is preserved, extend UPWARD to target length.
        for key, sr, sc, length, color in lines:
            # Find the bottom-left end (max r)
            max_r = sr + length - 1
            min_c = sc - (length - 1)
            # New start: target_len cells up from bottom
            new_sr = max_r - target_len + 1
            new_sc = key - new_sr
            for i in range(target_len):
                nr = new_sr + i
                nc = new_sc - i
                if 0 <= nr < H and 0 <= nc < W:
                    out[nr][nc] = color

    else:  # '\'
        lines.sort(key=lambda l: l[0])
        for key, sr, sc, length, color in lines:
            max_r = sr + length - 1
            new_sr = max_r - target_len + 1
            new_sc = new_sr - key
            for i in range(target_len):
                nr = new_sr + i
                nc = new_sc + i
                if 0 <= nr < H and 0 <= nc < W:
                    out[nr][nc] = color

    return out


# ============================================================
# c7f57c3e: Two patterns - swap their inner decorations
# Grid has two composite shapes. Each has a base and decorations.
# The decorations get swapped/reflected between the two shapes.
# ============================================================
def solve_c7f57c3e(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    bg = grid[0][0]

    # Find the two main patterns
    # Each pattern has a rectangular core and scattered decoration pixels
    # The decorations swap positions

    # Find connected components
    visited = set()
    components = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and (r,c) not in visited:
                q = [(r,c)]
                visited.add((r,c))
                cells = [(r,c)]
                idx = 0
                while idx < len(q):
                    cr, cc = q[idx]; idx += 1
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<H and 0<=nc<W and (nr,nc) not in visited and grid[nr][nc] != bg:
                            visited.add((nr,nc))
                            q.append((nr,nc))
                            cells.append((nr,nc))
                components.append(cells)

    if len(components) != 2:
        return out

    # For each component, separate core and decoration
    # The core is the main structured pattern (like a cross, frame, etc.)
    # Decorations are small isolated groups of a different color

    # Actually, looking at the examples more carefully:
    # Each pattern has sub-components of different colors
    # One sub-component is the "decoration" that swaps

    # From train 0:
    # Pattern A (right): has 1/2/4/8/3 colors, with 8 and 3 being the structure
    # Pattern B (left): has 1/2/4/3 colors, same structure
    # In output: the small decorations (1-cell) swap positions

    # The patterns have identical STRUCTURE but different small single-pixel decorations
    # The decorations (single pixels of unique colors) swap between the two patterns

    # For each component, find the "base" colors (forming the main shape)
    # and the "decoration" colors (sparse pixels)

    for comp in components:
        colors = Counter(grid[r][c] for r,c in comp)
        # Most common color is the base

    # This is complex. Let me try a different approach based on what I see in the examples.
    # The two patterns have a reflective relationship.
    # Specific cells swap between input and output.

    # From train 0 and 1, it looks like each pattern has a specific small element
    # that should be on one side but is on the other in the input.
    # The transformation moves it to the correct side.

    # This needs more careful analysis. Skip for now.
    return out


# ============================================================
# dd6b8c4b: 3x3 ring of 3s with center 2, scattered 9s
# The 9s get removed. 3s in direction toward majority of 9s become 9.
# ============================================================
def solve_dd6b8c4b(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    bg = 7

    # Find center (the 2)
    center = None
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 2:
                center = (r, c)
                break
        if center:
            break

    if not center:
        return out

    cr, cc = center

    # Find 9 positions
    nines = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == 9]

    # Find 3 positions (ring around center)
    threes = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == 3]

    # Also find 6 positions (part of outer frame, not to be modified)
    sixes = set((r, c) for r in range(H) for c in range(W) if grid[r][c] == 6)

    # Remove all 9s from output
    for r, c in nines:
        out[r][c] = bg

    # For each 3 in the ring, check if it should become 9
    # The rule: count 9s in each half relative to center
    above = sum(1 for r, c in nines if r < cr)
    below = sum(1 for r, c in nines if r > cr)
    left = sum(1 for r, c in nines if c < cc)
    right = sum(1 for r, c in nines if c > cc)

    for r, c in threes:
        dr = r - cr
        dc = c - cc

        # Only modify cells in the 3x3 ring around center (distance 1)
        if abs(dr) > 1 or abs(dc) > 1:
            continue

        flip = False
        if dr == -1:  # top row of ring
            if above > below:
                flip = True
        elif dr == 1:  # bottom row of ring
            if below > above:
                flip = True
        # middle row (dr == 0): check horizontal
        if dc == -1:  # left of ring
            if left > right and dr == 0:
                flip = True
        elif dc == 1:  # right of ring
            if right > left and dr == 0:
                flip = True

        # For corner cells: flip if both directions agree
        if abs(dr) == 1 and abs(dc) == 1:
            vert_flip = (dr == -1 and above > below) or (dr == 1 and below > above)
            horiz_flip = (dc == -1 and left > right) or (dc == 1 and right > left)
            flip = vert_flip or horiz_flip

        if flip:
            out[r][c] = 9

    return out


# ============================================================
# d8e07eb2: Band-based color replacement
# Grid has horizontal bands separated by 6-lines.
# Empty bands (all 8) become 3. Stamp patterns are preserved.
# ============================================================
def solve_d8e07eb2(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # Find rows that are all 6s (dividers)
    divider_rows = [r for r in range(H) if all(grid[r][c] == 6 for c in range(W))]

    if len(divider_rows) < 2:
        return out

    # Bands between dividers and at edges
    bands = []
    if divider_rows[0] > 0:
        bands.append((0, divider_rows[0] - 1))
    for i in range(len(divider_rows) - 1):
        bands.append((divider_rows[i] + 1, divider_rows[i+1] - 1))
    if divider_rows[-1] < H - 1:
        bands.append((divider_rows[-1] + 1, H - 1))

    # For each band, check if it has stamps (non-8 content)
    for start, end in bands:
        has_content = False
        for r in range(start, end + 1):
            for c in range(W):
                if grid[r][c] != 8:
                    has_content = True
                    break
            if has_content:
                break

        if not has_content:
            # Empty band -> fill with 3
            for r in range(start, end + 1):
                for c in range(W):
                    out[r][c] = 3

    # Also check: bands adjacent to dividers that have stamps
    # The first and last rows of stamp bands may need special treatment
    # Looking at the examples: bands with stamps that are adjacent to a divider
    # have their border row (facing the divider) also transformed

    # Check if the band immediately above/below a divider has content
    # If the band on the OTHER side of the divider is empty (now 3),
    # then add a 3-border to the stamp band

    # Actually let me look at this differently.
    # In the output, the stamp bands that are between two dividers get
    # a 3-border IF the adjacent band is empty.

    # Let me check: for each stamp band, if the band on either side is empty,
    # add a row of 3s as border

    # Actually from the examples: the empty bands plus any rows of 8s between
    # a divider and a stamp band all become 3.

    # Let me try: find which rows have stamps (non-8 content excluding dividers)
    stamp_rows = set()
    for r in range(H):
        if any(grid[r][c] not in (8, 6) for c in range(W)):
            stamp_rows.add(r)

    # For rows that are all 8 (not dividers, not stamps), check if they should become 3
    # Rule: rows of all 8s that are in bands that contain NO stamps become 3
    # This is what we already do above. Let me also check for column-based 3-borders.

    # Looking at train 1 output more carefully: there's a column of 3s
    # next to the stamps. Let me check if that's from a column-based analysis.

    # For train 0: the stamp patterns in rows 1-3 (band 0) and rows 13-15 (band 2)
    # get a 3-border. Specifically:
    # Band 0 (rows 0-4): rows 0, 4 are all 8 -> become 3
    #   Rows 1-3 have stamps -> stay
    # But in output, band 0 rows 0-4 ALL become 3, not just the 8-only rows
    # Wait: output rows 0-4:
    # 3333333333333333333333  <- all 3
    # 3300033111337333336633
    # 3300033313337773333633
    # 3300033111337333336633
    # 3333333333333333333333  <- all 3
    # So rows 0 and 4 are all 3 (they were all 8)
    # Rows 1-3: stamps preserved but 8s around stamps became 3

    # So the rule is: in bands that have stamps, ALL 8s become 3 as well!
    # Not just empty bands, but the 8 background within stamp bands too.

    # Wait but that contradicts the middle bands where stamps exist and 8s stay as 8.
    # Let me recheck: band 0 = rows 0-4 (above first divider at row 5)
    # Band 0 has stamps in rows 1-3. Output: ALL 8s in band 0 become 3.
    # Band 1 = rows 6-26 (between dividers at 5 and 27)
    # Band 1 has stamps. Output: stamps preserved, 8s STAY as 8.
    # Band 2 = rows 28-29 (below last divider at 27)
    # Band 2 has no stamps. Output: all 8s become 3.

    # So: bands OUTSIDE the outermost dividers get 8->3.
    # Bands INSIDE the dividers keep 8s.

    # Let me also check bands adjacent to dividers within:
    # In train 0 output, band immediately inside first divider (rows 6-7):
    # rows 6-7 are all 8 in input, stay all 8 in output. ✓
    # Band 2 in train 0: rows 12-16 have stamps at 13-15
    # Row 12 has all 8 in input. In output: row 12 starts with 8 but has 3s:
    # 8333333333333333333338
    # So row 12 is NOT all 8 -> has 3s on the edges!
    # This is a different pattern - the 3s form a border AROUND the stamps.

    # OK I think I misread the output. Let me go back to the raw data.
    # The 3-border appears to surround each stamp group within a band.

    # This is getting complex. Let me just implement the simple rule:
    # Outer bands (outside dividers) get 8->3
    # And recheck.

    # Revert to simple approach
    out = [row[:] for row in grid]

    first_div = divider_rows[0]
    last_div = divider_rows[-1]

    for r in range(H):
        if r < first_div or r > last_div:
            for c in range(W):
                if out[r][c] == 8:
                    out[r][c] = 3

    # Now check for inner bands that need 3-borders around stamps
    # For each inner band (between dividers), find stamp groups
    # and add 3 borders around them

    for bi in range(len(divider_rows) - 1):
        band_start = divider_rows[bi] + 1
        band_end = divider_rows[bi + 1] - 1
        if band_start > band_end:
            continue

        # Find stamp groups (connected non-8 regions within band)
        stamp_cells = set()
        for r in range(band_start, band_end + 1):
            for c in range(W):
                if grid[r][c] != 8:
                    stamp_cells.add((r, c))

        if not stamp_cells:
            continue

        # Find bounding boxes of stamp groups
        visited = set()
        groups = []
        for r, c in stamp_cells:
            if (r,c) not in visited:
                q = [(r,c)]
                visited.add((r,c))
                cells = [(r,c)]
                idx = 0
                while idx < len(q):
                    cr, cc = q[idx]; idx += 1
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if (nr,nc) in stamp_cells and (nr,nc) not in visited:
                            visited.add((nr,nc))
                            q.append((nr,nc))
                            cells.append((nr,nc))
                groups.append(cells)

        # For each stamp group, add a 3-border (1-pixel border of 3 around it)
        for group in groups:
            minr = min(r for r,c in group)
            maxr = max(r for r,c in group)
            minc = min(c for r,c in group)
            maxc = max(c for r,c in group)

            # Check if the band boundaries need 3-rows
            # If the stamp group doesn't touch the band edges, add 3-border rows
            if minr > band_start:
                for c in range(minc-1, maxc+2):
                    if 0 <= c < W and band_start <= minr-1 <= band_end:
                        if out[minr-1][c] == 8:
                            out[minr-1][c] = 3
            if maxr < band_end:
                for c in range(minc-1, maxc+2):
                    if 0 <= c < W and band_start <= maxr+1 <= band_end:
                        if out[maxr+1][c] == 8:
                            out[maxr+1][c] = 3

            # Add side borders
            for r in range(minr, maxr+1):
                if minc > 0 and out[r][minc-1] == 8:
                    out[r][minc-1] = 3
                if maxc < W-1 and out[r][maxc+1] == 8:
                    out[r][maxc+1] = 3

    return out


# ============================================================
# e3721c99: Recolor blobs based on pattern strip
# Grid divided by horizontal line of 1s.
# Above: pattern strip (colored rectangles) + blob shapes
# Below: same blob shapes to be recolored
# Rule: each blob below gets colored based on matching from pattern
# ============================================================
def solve_e3721c99(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # Find the horizontal divider line of 1s
    divider_row = -1
    for r in range(H):
        if all(grid[r][c] == 1 for c in range(W)):
            divider_row = r
            break

    if divider_row == -1:
        return out

    # Find the pattern row (colored rectangles above divider)
    # Pattern rows are typically rows 1-3 (between row 0 of 0s and the divider)
    # They have colored segments showing the color mapping

    # Find non-zero rows above divider
    pattern_rows = []
    for r in range(divider_row):
        if any(grid[r][c] != 0 for c in range(W)):
            pattern_rows.append(r)

    if not pattern_rows:
        return out

    # Extract color segments from the pattern rows
    # Find the first row with multiple colored segments
    # The pattern is: groups of same-colored cells in a row

    # Use all pattern rows to find the color mapping
    # Each column in the pattern maps to a color for blobs at that column position

    # Build column-to-color mapping from the pattern
    # For each column, find the non-0 colors in the pattern rows
    col_colors = {}
    for c in range(W):
        for r in pattern_rows:
            if grid[r][c] != 0:
                col_colors[c] = grid[r][c]
                break

    # Find colored segments (contiguous columns with same color)
    segments = []
    c = 0
    while c < W:
        if c in col_colors:
            color = col_colors[c]
            start = c
            while c < W and col_colors.get(c) == color:
                c += 1
            segments.append((start, c-1, color))
        else:
            c += 1

    # Below the divider, find blobs and recolor them
    below_start = divider_row + 1

    # Each blob below should be recolored based on its horizontal position
    # matching the pattern segment above

    # Build a column-to-segment-color mapping
    col_to_color = {}
    for start, end, color in segments:
        for c in range(start, end+1):
            col_to_color[c] = color

    # Find connected components below divider (non-0 cells)
    visited = set()
    blobs = []
    for r in range(below_start, H):
        for c in range(W):
            if grid[r][c] != 0 and (r, c) not in visited:
                q = [(r, c)]
                visited.add((r, c))
                cells = [(r, c)]
                idx = 0
                while idx < len(q):
                    cr, cc = q[idx]; idx += 1
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if below_start <= nr < H and 0 <= nc < W and (nr, nc) not in visited and grid[nr][nc] != 0:
                            visited.add((nr, nc))
                            q.append((nr, nc))
                            cells.append((nr, nc))
                blobs.append(cells)

    # For each blob, find its center column and map to segment color
    for blob in blobs:
        blob_cols = [c for r, c in blob]
        center_c = (min(blob_cols) + max(blob_cols)) // 2

        # Find matching segment color
        if center_c in col_to_color:
            new_color = col_to_color[center_c]
        else:
            # Find nearest column with a color
            best_c = min(col_to_color.keys(), key=lambda k: abs(k - center_c))
            new_color = col_to_color[best_c]

        original_color = grid[blob[0][0]][blob[0][1]]
        for r, c in blob:
            out[r][c] = new_color

    return out


# ============================================================
# MAIN SOLVER DISPATCH
# ============================================================
solvers = {
    'dbff022c': solve_dbff022c,
    'c4d067a0': solve_c4d067a0,
    'db695cfb': solve_db695cfb,
    'e376de54': solve_e376de54,
    # These don't pass yet but are kept for future work:
    #'dd6b8c4b': solve_dd6b8c4b,
    #'d8e07eb2': solve_d8e07eb2,
    #'e3721c99': solve_e3721c99,
}

task_ids = [
    'b5ca7ac4','b6f77b65','b99e7126','b9e38dc0','bf45cf4b',
    'c4d067a0','c7f57c3e','cb2d8a2c','cbebaa4b','d35bdbdc',
    'd59b0160','d8e07eb2','da515329','db0c5428','db695cfb',
    'dbff022c','dd6b8c4b','de809cff','dfadab01','e12f9a14',
    'e3721c99','e376de54','e8686506','e87109e9','edb79dae',
    'eee78d87','f560132c','f931b4a8','faa9f03d','fc7cae8d',
]

solutions = {}
passed = []
failed = []

for tid in task_ids:
    if tid not in solvers:
        continue

    task = load_task(tid)
    solver = solvers[tid]

    try:
        if test_solve(solver, task):
            results = apply_solve(solver, task)
            solutions[tid] = results
            passed.append(tid)
            print(f'{tid}: PASSED - {len(results)} test outputs')
        else:
            for i, p in enumerate(task['train']):
                result = solver(p['input'])
                if result != p['output']:
                    H = len(p['output'])
                    W = len(p['output'][0])
                    rH = len(result)
                    rW = len(result[0]) if result else 0
                    if rH != H or rW != W:
                        print(f'{tid}: FAILED train {i} - size mismatch {rH}x{rW} vs {H}x{W}')
                    else:
                        ndiff = sum(1 for r in range(H) for c in range(W) if result[r][c] != p['output'][r][c])
                        print(f'{tid}: FAILED train {i} - {ndiff} diffs')
                    break
            failed.append(tid)
    except Exception as e:
        import traceback
        print(f'{tid}: ERROR - {e}')
        traceback.print_exc()
        failed.append(tid)

print(f'\nPassed: {len(passed)}/{len(task_ids)}')
print(f'Passed tasks: {passed}')

with open('data/arc2_solutions_eval3.json', 'w') as f:
    json.dump(solutions, f)
print(f'Saved {len(solutions)} solutions to data/arc2_solutions_eval3.json')
