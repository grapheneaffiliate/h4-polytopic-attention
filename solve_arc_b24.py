import json
import copy
from collections import Counter

def load_task(tid):
    with open(f'data/arc1/{tid}.json') as f:
        return json.load(f)

def solve_e3497940(grid):
    """Fold: col 4 divider of 5s. Reverse right side and overlay with left."""
    R, C = len(grid), len(grid[0])
    div_col = 4
    right_w = C - div_col - 1
    out = [[0]*right_w for _ in range(R)]
    for r in range(R):
        right = grid[r][div_col+1:][::-1]
        left = grid[r][:div_col]
        for c in range(right_w):
            if c < len(right) and right[c] != 0:
                out[r][c] = right[c]
            elif c < len(left) and left[c] != 0:
                out[r][c] = left[c]
    return out

def solve_e40b9e2f(grid):
    """Shape + single marker pixel. Create 4-fold symmetry around shape centroid."""
    R, C = len(grid), len(grid[0])
    # Find connected components
    visited = set()
    comps = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] != 0 and (r,c) not in visited:
                comp = []
                stack = [(r,c)]
                while stack:
                    cr,cc = stack.pop()
                    if (cr,cc) in visited or grid[cr][cc]==0: continue
                    visited.add((cr,cc))
                    comp.append((cr,cc,grid[cr][cc]))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=cr+dr,cc+dc
                        if 0<=nr<R and 0<=nc<C and (nr,nc) not in visited and grid[nr][nc]!=0:
                            stack.append((nr,nc))
                comps.append(comp)

    single = main = None
    for comp in comps:
        if len(comp)==1: single=comp[0]
        else: main=comp

    if single is None or main is None: return grid

    sr,sc,sv = single
    # Shape centroid (rounded)
    cr = round(sum(r for r,c,v in main)/len(main))
    cc = round(sum(c for r,c,v in main)/len(main))

    out = [[0]*C for _ in range(R)]
    # 4-fold reflect all cells (shape + single)
    all_cells = main + [single]
    for r,c,v in all_cells:
        for dr_sign in [-1,1]:
            for dc_sign in [-1,1]:
                nr = cr + dr_sign*(r-cr)
                nc = cc + dc_sign*(c-cc)
                if 0<=nr<R and 0<=nc<C:
                    out[int(nr)][int(nc)] = v

    # Extend center row and center col to match single pixel offset
    row_offset = abs(sr - cr)
    col_offset = abs(sc - cc)

    # Extend center row (r=cr) outward
    for dc_sign in [-1,1]:
        for d in range(1, col_offset+1):
            nc = cc + dc_sign*d
            if 0<=nc<C:
                # Find the edge value on center row in that direction
                src_c = cc + dc_sign*min(d, col_offset-1)
                # Use the value from the shape at center row, closest column
                edge_val = 0
                for sc2 in range(cc, cc+dc_sign*(col_offset+1), dc_sign):
                    if 0<=sc2<C and out[cr][sc2] != 0:
                        edge_val = out[cr][sc2]
                if out[cr][int(nc)] == 0 and edge_val != 0:
                    out[cr][int(nc)] = edge_val

    # Extend center col (c=cc) outward
    for dr_sign in [-1,1]:
        for d in range(1, row_offset+1):
            nr = cr + dr_sign*d
            if 0<=nr<R:
                edge_val = 0
                for sr2 in range(cr, cr+dr_sign*(row_offset+1), dr_sign):
                    if 0<=sr2<R and out[sr2][cc] != 0:
                        edge_val = out[sr2][cc]
                if out[int(nr)][cc] == 0 and edge_val != 0:
                    out[int(nr)][cc] = edge_val

    return out

def solve_e48d4e1a(grid):
    """Cross with 5-markers. Move cross by number of 5s in opposite direction."""
    R, C = len(grid), len(grid[0])
    color = None
    for r in range(R):
        for c in range(C):
            if grid[r][c] not in (0,5):
                color = grid[r][c]; break
        if color: break

    cross_row = cross_col = None
    for r in range(R):
        if all(grid[r][c]==color for c in range(C)): cross_row=r
    for c in range(C):
        if all(grid[r][c]==color for r in range(R)): cross_col=c

    fives = [(r,c) for r in range(R) for c in range(C) if grid[r][c]==5]
    n = len(fives)

    above = sum(1 for r,c in fives if r < cross_row)
    right = sum(1 for r,c in fives if c > cross_col)

    new_row = cross_row + (n if above >= n-above else -n)
    new_col = cross_col + (-n if right >= n-right else n)

    out = [[0]*C for _ in range(R)]
    for r in range(R): out[r][new_col] = color
    for c in range(C): out[new_row][c] = color
    return out

def solve_e5062a87(grid):
    """Mark 0-cells adjacent to 5-cells with 2."""
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 0:
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc = r+dr, c+dc
                    if 0<=nr<R and 0<=nc<C and grid[nr][nc]==5:
                        out[r][c] = 2; break
    return out

def solve_e50d258f(grid):
    """Find rectangular block with most 2s among separated non-zero blocks."""
    R, C = len(grid), len(grid[0])
    visited = set()
    blocks = []
    for r in range(R):
        for c in range(C):
            if grid[r][c]!=0 and (r,c) not in visited:
                comp = []
                stack = [(r,c)]
                while stack:
                    cr,cc = stack.pop()
                    if (cr,cc) in visited or grid[cr][cc]==0: continue
                    visited.add((cr,cc))
                    comp.append((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=cr+dr,cc+dc
                        if 0<=nr<R and 0<=nc<C and (nr,nc) not in visited and grid[nr][nc]!=0:
                            stack.append((nr,nc))
                blocks.append(comp)

    best = None
    best_score = -1
    for block in blocks:
        count_2 = sum(1 for r,c in block if grid[r][c]==2)
        if count_2 > best_score:
            best_score = count_2
            min_r = min(r for r,c in block)
            max_r = max(r for r,c in block)
            min_c = min(c for r,c in block)
            max_c = max(c for r,c in block)
            best = [[grid[r][c] for c in range(min_c,max_c+1)] for r in range(min_r,max_r+1)]
    return best

def solve_e76a88a6(grid):
    """Template pattern replaces 5-blocks. Copy template to each 5-region."""
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    visited = set()
    five_blocks = []
    non_five = []
    for r in range(R):
        for c in range(C):
            if grid[r][c]==5 and (r,c) not in visited:
                comp = []
                stack = [(r,c)]
                while stack:
                    cr,cc = stack.pop()
                    if (cr,cc) in visited or grid[cr][cc]!=5: continue
                    visited.add((cr,cc))
                    comp.append((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=cr+dr,cc+dc
                        if 0<=nr<R and 0<=nc<C and (nr,nc) not in visited:
                            stack.append((nr,nc))
                five_blocks.append(comp)
            elif grid[r][c]!=0 and grid[r][c]!=5:
                non_five.append((r,c))

    if not non_five: return out
    t_min_r = min(r for r,c in non_five)
    t_max_r = max(r for r,c in non_five)
    t_min_c = min(c for r,c in non_five)
    t_max_c = max(c for r,c in non_five)
    t_h = t_max_r - t_min_r + 1
    t_w = t_max_c - t_min_c + 1

    template = [[0]*t_w for _ in range(t_h)]
    for r,c in non_five:
        template[r-t_min_r][c-t_min_c] = grid[r][c]

    for block in five_blocks:
        b_min_r = min(r for r,c in block)
        b_min_c = min(c for r,c in block)
        b_max_r = max(r for r,c in block)
        b_max_c = max(c for r,c in block)
        for r in range(b_min_r, b_max_r+1):
            for c in range(b_min_c, b_max_c+1):
                out[r][c] = template[(r-b_min_r)%t_h][(c-b_min_c)%t_w]
    return out

def solve_e8593010(grid):
    """Replace 0s: size 1->3, size 2->2, size 3+->1."""
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    visited = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c]==0 and (r,c) not in visited:
                comp = []
                stack = [(r,c)]
                while stack:
                    cr,cc = stack.pop()
                    if (cr,cc) in visited or grid[cr][cc]!=0: continue
                    visited.add((cr,cc))
                    comp.append((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=cr+dr,cc+dc
                        if 0<=nr<R and 0<=nc<C and (nr,nc) not in visited and grid[nr][nc]==0:
                            stack.append((nr,nc))
                color = 3 if len(comp)==1 else (2 if len(comp)==2 else 1)
                for cr,cc in comp:
                    out[cr][cc] = color
    return out

def solve_e8dc4411(grid):
    """Shape + colored marker. Propagate shape in marker direction repeatedly (diagonal delta)."""
    R, C = len(grid), len(grid[0])
    bg = grid[0][0]
    out = [row[:] for row in grid]

    shape_cells = []
    marker = None
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg and grid[r][c] == 0:
                shape_cells.append((r,c))
            elif grid[r][c] != bg and grid[r][c] != 0:
                marker = (r,c,grid[r][c])

    if not marker or not shape_cells: return out
    mr, mc, mv = marker
    shape_set = set(shape_cells)

    # Find delta: marker = some_shape_cell + delta, |delta_r| == |delta_c|, no overlap
    best_delta = None
    for sr, sc in shape_cells:
        dr = mr - sr
        dc = mc - sc
        if abs(dr) != abs(dc) or dr == 0: continue
        shifted = set((r+dr, c+dc) for r,c in shape_cells)
        if not shifted.intersection(shape_set):
            if best_delta is None or abs(dr) < abs(best_delta[0]):
                best_delta = (dr, dc)

    if best_delta is None:
        # Fallback: try any delta where marker = shape_cell + delta, no overlap
        for sr, sc in shape_cells:
            dr = mr - sr
            dc = mc - sc
            shifted = set((r+dr, c+dc) for r,c in shape_cells)
            if not shifted.intersection(shape_set):
                best_delta = (dr, dc)
                break

    if best_delta is None: return out
    delta_r, delta_c = best_delta

    # Propagate
    for n in range(1, max(R,C)*2):
        any_in_bounds = False
        for sr,sc in shape_cells:
            nr = sr + n*delta_r
            nc = sc + n*delta_c
            if 0<=nr<R and 0<=nc<C:
                any_in_bounds = True
                if out[nr][nc] == bg:
                    out[nr][nc] = mv
        if not any_in_bounds:
            break
    return out

def solve_e9614598(grid):
    """Two 1s. Place cross of 3s at midpoint."""
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    ones = [(r,c) for r in range(R) for c in range(C) if grid[r][c]==1]
    if len(ones)==2:
        r1,c1 = ones[0]; r2,c2 = ones[1]
        mr,mc = (r1+r2)//2, (c1+c2)//2
        out[mr][mc] = 3
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc = mr+dr, mc+dc
            if 0<=nr<R and 0<=nc<C: out[nr][nc] = 3
    return out

def solve_e98196ab(grid):
    """Split by row of 5s. Overlay top and bottom."""
    R, C = len(grid), len(grid[0])
    split_row = None
    for r in range(R):
        if all(grid[r][c]==5 for c in range(C)):
            split_row=r; break
    top = grid[:split_row]
    bottom = grid[split_row+1:]
    out = [[0]*C for _ in range(len(top))]
    for r in range(len(top)):
        for c in range(C):
            if top[r][c]!=0: out[r][c]=top[r][c]
            elif bottom[r][c]!=0: out[r][c]=bottom[r][c]
    return out

def solve_e9afcf9a(grid):
    """Checkerboard from 2 uniform rows."""
    c1, c2 = grid[0][0], grid[1][0]
    C = len(grid[0])
    return [[c1 if c%2==0 else c2 for c in range(C)],
            [c2 if c%2==0 else c1 for c in range(C)]]

def solve_ea32f347(grid):
    """Lines of 5s colored by length: longest=1, middle=4, shortest=2."""
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    visited = set()
    lines = []
    for r in range(R):
        for c in range(C):
            if grid[r][c]==5 and (r,c) not in visited:
                line = [(r,c)]
                visited.add((r,c))
                # Try vertical
                for d in [1,-1]:
                    nr = r+d
                    while 0<=nr<R and grid[nr][c]==5:
                        line.append((nr,c)); visited.add((nr,c)); nr+=d
                if len(line)==1:
                    # Try horizontal
                    for d in [1,-1]:
                        nc = c+d
                        while 0<=nc<C and grid[r][nc]==5:
                            line.append((r,nc)); visited.add((r,nc)); nc+=d
                lines.append(line)
    lines.sort(key=len, reverse=True)
    colors = [1,4,2]
    for i,line in enumerate(lines):
        if i<len(colors):
            for r,c in line: out[r][c]=colors[i]
    return out

def solve_ea786f4a(grid):
    """X pattern of 0s from center 0 to corners."""
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(R):
        for c in range(C):
            if grid[r][c]==0:
                cr,cc = r,c
                break
    for d in range(max(R,C)):
        for dr,dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr,nc = cr+d*dr, cc+d*dc
            if 0<=nr<R and 0<=nc<C: out[nr][nc]=0
    return out

def solve_ec883f72(grid):
    """L-shaped border with fill. The fill color escapes through the opening diagonally."""
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    non_zero = {}
    for r in range(R):
        for c in range(C):
            if grid[r][c]!=0: non_zero[(r,c)]=grid[r][c]

    if not non_zero: return out
    colors = Counter(non_zero.values())
    sorted_c = colors.most_common()
    border_color = sorted_c[0][0]
    fill_color = sorted_c[1][0] if len(sorted_c)>1 else None
    if fill_color is None: return out

    border_cells = [(r,c) for (r,c),v in non_zero.items() if v==border_color]
    fill_cells = [(r,c) for (r,c),v in non_zero.items() if v==fill_color]

    # Find bounding box of border
    b_min_r = min(r for r,c in border_cells)
    b_max_r = max(r for r,c in border_cells)
    b_min_c = min(c for r,c in border_cells)
    b_max_c = max(c for r,c in border_cells)

    # Find the full row/col of border (the base of the L/U)
    full_rows = []
    for r in range(b_min_r, b_max_r+1):
        cnt = sum(1 for (rr,cc) in border_cells if rr==r)
        if cnt == b_max_c - b_min_c + 1:
            full_rows.append(r)
    full_cols = []
    for c in range(b_min_c, b_max_c+1):
        cnt = sum(1 for (rr,cc) in border_cells if cc==c)
        if cnt == b_max_r - b_min_r + 1:
            full_cols.append(c)

    # Determine open direction and corner
    # The opening is opposite to the full row/col
    # Fill cells escape through the opening

    # For each fill cell, find its offset from the corner of the L
    # The corner is at the intersection of the full row and full col
    if full_rows and full_cols:
        corner_r = full_rows[0]
        corner_c = full_cols[0]
    elif full_rows:
        corner_r = full_rows[0]
        # corner_c is the end of the full row closest to fill cells
        fc_avg = sum(c for r,c in fill_cells)/len(fill_cells)
        corner_c = b_min_c if fc_avg < (b_min_c+b_max_c)/2 else b_max_c
    elif full_cols:
        corner_c = full_cols[0]
        fr_avg = sum(r for r,c in fill_cells)/len(fill_cells)
        corner_r = b_min_r if fr_avg < (b_min_r+b_max_r)/2 else b_max_r
    else:
        return out

    # Determine escape direction (away from corner through the opening)
    # Fill cells are inside the L. The opening faces away from the full row/col.
    open_dr = -1 if corner_r == b_max_r else 1
    open_dc = -1 if corner_c == b_max_c else 1

    # For each fill cell, compute distance from the border edges and project outward
    for fr, fc in fill_cells:
        # Distance from corner in the open direction
        dr = fr - corner_r  # how far from the full row
        dc = fc - corner_c  # how far from the full col

        # The fill cell escapes diagonally through the opening
        # Project: continue stepping away from the L
        step = 1
        while True:
            nr = corner_r + open_dr * step + (fr - corner_r) * (0 if full_rows else 1)
            nc = corner_c + open_dc * step + (fc - corner_c) * (0 if full_cols else 1)
            # Actually simpler: the fill at offset (dr, dc) from corner reflects to (-dr, -dc) but going outward
            # Let me think about this differently
            break

    # Alternative approach: the fill cells have specific positions relative to the inside of the L
    # When projected outward, each fill cell traces a diagonal line

    # For each fill cell, compute its perpendicular distances from the two arms of the L
    # Then project outward from the opening

    # Simpler: the fill color appears at positions that are reflections through the opening edge
    # The opening edge runs perpendicular to each arm

    # Let me try: for each fill cell, project it through the nearest opening edge
    # to the other side, continuing diagonally away

    # The arms of the L: one horizontal (the full row) and one vertical (the full col)
    # The fill is enclosed by both arms. When it escapes, it moves diagonally away
    # from both arms simultaneously.

    # For each fill cell at (fr, fc):
    # dist_from_row = abs(fr - corner_r)  (distance from the base row)
    # dist_from_col = abs(fc - corner_c)  (distance from the base col)
    # The cell escapes to: row = corner_r + open_dr * (dist_from_col), col = corner_c + open_dc * (dist_from_row)
    # And continues diagonally

    # Looking at train[0]: corner at (3,3) (full row=3, full col=3), open direction = (-1,-1) (up-left)
    # Fill at (0,0): dist from row 3 = 3, dist from col 3 = 3
    # Escaped to: row = 3 + (-1)*3 = 0, col = 3 + (-1)*3 = 0... that's back at (0,0), not useful

    # Let me try for train[0]: border col 3 (full col), border row 3 (full row)
    # Fill at (0,0),(0,1),(1,0),(1,1)
    # Opening is upward (row < b_min_r) and leftward (col < b_min_c)?
    # Actually the L opens at the opposite of the corner
    # Corner = (3,3). Opening faces up-left. Wait, the opening is where border doesn't exist.

    # Border occupies: row 3 (full), col 3 (full), plus (0,3),(1,3),(2,3) for the col
    # and (3,0),(3,1),(3,2),(3,3) for the row.
    # So the border forms an L in the bottom-right area.
    # Fill (3 color) is at top-left: (0,0),(0,1),(1,0),(1,1)
    # The opening faces down-right from the L

    # In output, fill escapes as: (4,4)=3, (5,5)=3
    # These are at positions (corner_r + step, corner_c + step) for step=1,2

    # Fill cell (0,0): max distance from corner (3,3) = max(3,3) = 3
    # Diag steps = 2: from corner going away
    # Fill cell (1,1): dist from corner = max(2,2) = 2

    # Actually the number of diag steps seems related to the min distance from each arm
    # (0,0): dist from row 3 = 3, dist from col 3 = 3. Both 3.
    # (0,1): dist from row 3 = 3, dist from col 3 = 2. Min = 2.
    # (1,0): dist from row 3 = 2, dist from col 3 = 3. Min = 2.
    # (1,1): dist from row 3 = 2, dist from col 3 = 2. Both 2.

    # Output has fill at (4,4) and (5,5)
    # (4,4) = corner + (1,1). (5,5) = corner + (2,2).
    # There are 2 fill cells with min_dist=2: (0,1) and (1,0)... but we need 2 output positions
    # Actually we have 4 fill cells and 2 output positions

    # The fill cells per diagonal (where diag = dr+dc from corner):
    # (0,0): diag=6, (0,1): diag=5, (1,0): diag=5, (1,1): diag=4
    # Output: (4,4): diag=2 (from corner (3,3), so actual diag step=1)
    # Hmm, this doesn't obviously map

    # Let me try: count fill cells per anti-diagonal line
    # In the fill block, how many cells per each row?
    # Row 0: 2 cells (cols 0,1). Row 1: 2 cells.
    # In output: (4,4) and (5,5). That's 1 cell per row for 2 rows.
    # Maybe each row of fill maps to one diagonal step

    # I think the rule is: count the fill cells per column (or row depending on orientation)
    # and place that many fill color cells in the diagonal escape direction

    # For this L, escape direction is (+1,+1) from corner (3,3)
    # Count fill cells per perpendicular slice:
    # Along the escape direction, fill cells at distance d from the opening:
    # d=3: cells (0,0) and (1,0) -> 2 cells? No...

    # I'll skip this complex task for now
    return None

def solve_ecdecbb3(grid):
    """8-lines with 2-dots. Draw line from dot to 8-line, add 3x3 8-box at intersection."""
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    eight_rows = set(r for r in range(R) if all(grid[r][c]==8 for c in range(C)))
    eight_cols = set(c for c in range(C) if all(grid[r][c]==8 for r in range(R)))
    twos = [(r,c) for r in range(R) for c in range(C) if grid[r][c]==2]

    for tr, tc in twos:
        # Connect to ALL 8-lines (rows and cols)
        for er in eight_rows:
            step = 1 if er > tr else -1
            for r in range(tr, er-step, step):
                out[r][tc] = 2
            out[er][tc] = 2
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    nr, nc = er+dr, tc+dc
                    if 0<=nr<R and 0<=nc<C:
                        if nr == er and nc == tc: continue
                        if nr in eight_rows: continue
                        if out[nr][nc] == 0: out[nr][nc] = 8

        for ec in eight_cols:
            step = 1 if ec > tc else -1
            for c in range(tc, ec-step, step):
                out[tr][c] = 2
            out[tr][ec] = 2
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    nr, nc = tr+dr, ec+dc
                    if 0<=nr<R and 0<=nc<C:
                        if nr == tr and nc == ec: continue
                        if nc in eight_cols: continue
                        if out[nr][nc] == 0: out[nr][nc] = 8

    return out

def solve_ef135b50(grid):
    """Fill gaps between facing rectangle pairs with 9s."""
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    visited = set()
    rects = []
    for r in range(R):
        for c in range(C):
            if grid[r][c]==2 and (r,c) not in visited:
                comp = []
                stack = [(r,c)]
                while stack:
                    cr,cc = stack.pop()
                    if (cr,cc) in visited or grid[cr][cc]!=2: continue
                    visited.add((cr,cc))
                    comp.append((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=cr+dr,cc+dc
                        if 0<=nr<R and 0<=nc<C and (nr,nc) not in visited:
                            stack.append((nr,nc))
                rects.append(comp)

    for i in range(len(rects)):
        for j in range(i+1, len(rects)):
            ri_rows = set(r for r,c in rects[i])
            rj_rows = set(r for r,c in rects[j])
            ri_cols = set(c for r,c in rects[i])
            rj_cols = set(c for r,c in rects[j])

            shared_rows = ri_rows & rj_rows
            if shared_rows:
                ri_cs = [c for r,c in rects[i] if r in shared_rows]
                rj_cs = [c for r,c in rects[j] if r in shared_rows]
                if ri_cs and rj_cs:
                    gap_start_c = gap_end_c = None
                    if max(ri_cs) < min(rj_cs):
                        gap_start_c = max(ri_cs)+1
                        gap_end_c = min(rj_cs)
                    elif max(rj_cs) < min(ri_cs):
                        gap_start_c = max(rj_cs)+1
                        gap_end_c = min(ri_cs)
                    if gap_start_c is not None:
                        # Check gap is clear of rect cells
                        gap_clear = True
                        all_rect_cells = set()
                        for rect in rects:
                            for r2,c2 in rect:
                                all_rect_cells.add((r2,c2))
                        for r in shared_rows:
                            for c in range(gap_start_c, gap_end_c):
                                if (r,c) in all_rect_cells:
                                    gap_clear = False
                                    break
                            if not gap_clear: break
                        if gap_clear:
                            for r in shared_rows:
                                for c in range(gap_start_c, gap_end_c):
                                    if out[r][c]==0: out[r][c]=9

            shared_cols = ri_cols & rj_cols
            if shared_cols:
                ri_rs = [r for r,c in rects[i] if c in shared_cols]
                rj_rs = [r for r,c in rects[j] if c in shared_cols]
                if ri_rs and rj_rs:
                    gap_start = gap_end = None
                    if max(ri_rs) < min(rj_rs):
                        gap_start = max(ri_rs)+1
                        gap_end = min(rj_rs)
                    elif max(rj_rs) < min(ri_rs):
                        gap_start = max(rj_rs)+1
                        gap_end = min(ri_rs)
                    if gap_start is not None:
                        # Check NO other rect has rows in gap range
                        gap_clear = True
                        for k, rect in enumerate(rects):
                            if k == i or k == j: continue
                            rect_rows = set(r for r,c in rect)
                            for r in range(gap_start, gap_end):
                                if r in rect_rows:
                                    gap_clear = False
                                    break
                            if not gap_clear: break
                        if gap_clear:
                            for c in shared_cols:
                                for r in range(gap_start, gap_end):
                                    if out[r][c]==0: out[r][c]=9
    return out

def solve_f25fbde4(grid):
    """Shape scaled 2x."""
    R, C = len(grid), len(grid[0])
    cells = [(r,c) for r in range(R) for c in range(C) if grid[r][c]!=0]
    if not cells: return grid
    min_r = min(r for r,c in cells)
    max_r = max(r for r,c in cells)
    min_c = min(c for r,c in cells)
    max_c = max(c for r,c in cells)
    h = max_r-min_r+1; w = max_c-min_c+1
    out = [[0]*(w*2) for _ in range(h*2)]
    for r,c in cells:
        v = grid[r][c]
        rr = (r-min_r)*2; cc = (c-min_c)*2
        out[rr][cc]=out[rr][cc+1]=out[rr+1][cc]=out[rr+1][cc+1]=v
    return out

def solve_f2829549(grid):
    """4x7 split by col of 1s. Output 3 where both sides are 0."""
    R, C = len(grid), len(grid[0])
    div_col = None
    for c in range(C):
        if all(grid[r][c]==1 for r in range(R)):
            div_col=c; break
    left_w = div_col
    right_w = C-div_col-1
    out_w = max(left_w, right_w)
    out = [[0]*out_w for _ in range(R)]
    for r in range(R):
        for c in range(out_w):
            left_val = grid[r][c] if c<left_w else 0
            right_val = grid[r][div_col+1+c] if c<right_w else 0
            if left_val==0 and right_val==0:
                out[r][c] = 3
    return out

def solve_f35d900a(grid):
    """Two colored dots. Draw 4 boxes (at corners of rectangle) and connect with 5s."""
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    dots = [(r,c,grid[r][c]) for r in range(R) for c in range(C) if grid[r][c]!=0]

    if len(dots) == 4:
        # 4 dots forming a rectangle. Each dot gets a box with border = other diagonal's color.
        # Find the two colors
        dot_map = {(r,c): v for r,c,v in dots}
        rows = sorted(set(r for r,c,v in dots))
        cols = sorted(set(c for r,c,v in dots))
        r1, r2 = rows[0], rows[1]
        c1, c2 = cols[0], cols[1]
    elif len(dots) == 2:
        r1,c1,color1 = dots[0]; r2,c2,color2 = dots[1]
        dot_map = {(r1,c1): color1, (r2,c2): color2,
                   (r1,c2): color2, (r2,c1): color1}
    else:
        return out

    # Build boxes: each dot's border = adjacent corner's color
    boxes = []
    for br, bc in [(r1,c1),(r1,c2),(r2,c1),(r2,c2)]:
        center = dot_map.get((br,bc), 0)
        # Adjacent corner shares one coordinate
        if (br,bc) == (r1,c1): border = dot_map.get((r1,c2), 0)
        elif (br,bc) == (r1,c2): border = dot_map.get((r1,c1), 0)
        elif (br,bc) == (r2,c1): border = dot_map.get((r2,c2), 0)
        else: border = dot_map.get((r2,c1), 0)
        boxes.append((br, bc, center, border))

    for br, bc, center, border in boxes:
        for dr in range(-1,2):
            for dc in range(-1,2):
                nr, nc = br+dr, bc+dc
                if 0<=nr<R and 0<=nc<C:
                    out[nr][nc] = center if (dr==0 and dc==0) else border

    # Connect with 5s: every 2 steps from each edge, stopping at center
    if r1 != r2 and c1 != c2:
        r_lo, r_hi = min(r1,r2), max(r1,r2)
        c_lo, c_hi = min(c1,c2), max(c1,c2)
        r_center = (r_lo + r_hi) / 2
        c_center = (c_lo + c_hi) / 2

        # Vertical 5s at c1 and c2
        k = 1
        while r_lo + 2*k <= r_center:
            out[r_lo + 2*k][c_lo] = 5
            out[r_lo + 2*k][c_hi] = 5
            k += 1
        k = 1
        while r_hi - 2*k > r_center:
            out[r_hi - 2*k][c_lo] = 5
            out[r_hi - 2*k][c_hi] = 5
            k += 1

        # Horizontal 5s at r1 and r2
        k = 1
        while c_lo + 2*k <= c_center:
            out[r_lo][c_lo + 2*k] = 5
            out[r_hi][c_lo + 2*k] = 5
            k += 1
        k = 1
        while c_hi - 2*k > c_center:
            out[r_lo][c_hi - 2*k] = 5
            out[r_hi][c_hi - 2*k] = 5
            k += 1

    return out

def solve_f5b8619d(grid):
    """Tile 2x2, replace 0s in non-zero columns with 8."""
    R, C = len(grid), len(grid[0])
    # Find which columns have non-zero
    nz_cols = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] != 0:
                nz_cols.add(c)

    out = [[0]*(C*2) for _ in range(R*2)]
    for r in range(R*2):
        for c in range(C*2):
            src_r = r % R
            src_c = c % C
            val = grid[src_r][src_c]
            if val != 0:
                out[r][c] = val
            elif src_c in nz_cols:
                out[r][c] = 8
    return out

def solve_f76d97a5(grid):
    """Swap: non-5 color becomes 0, 5 becomes non-5 color."""
    R, C = len(grid), len(grid[0])
    other = None
    for r in range(R):
        for c in range(C):
            if grid[r][c] != 5:
                other = grid[r][c]; break
        if other is not None: break

    out = [[0]*C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 5:
                out[r][c] = other
            else:
                out[r][c] = 0
    return out

def solve_f8a8fe49(grid):
    """Box of 2s with 5-shape inside. Reflect 5s through box edge to outside, remove from inside."""
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    five_cells = [(r,c) for r in range(R) for c in range(C) if grid[r][c]==5]
    two_cells = [(r,c) for r in range(R) for c in range(C) if grid[r][c]==2]

    if not five_cells or not two_cells: return out

    t_min_r = min(r for r,c in two_cells)
    t_max_r = max(r for r,c in two_cells)
    t_min_c = min(c for r,c in two_cells)
    t_max_c = max(c for r,c in two_cells)

    box_h = t_max_r - t_min_r + 1
    box_w = t_max_c - t_min_c + 1

    # Remove 5s from inside
    for r,c in five_cells:
        out[r][c] = 0

    # Reflect 5s through nearest edge
    if box_h > box_w:
        # Taller box: reflect through horizontal edges (top/bottom)
        for r,c in five_cells:
            dist_top = r - t_min_r
            dist_bot = t_max_r - r
            if dist_top <= dist_bot:
                nr = t_min_r - dist_top
            else:
                nr = t_max_r + dist_bot
            if 0<=nr<R:
                out[nr][c] = 5
    else:
        # Wider/square box: reflect through vertical edges (left/right)
        for r,c in five_cells:
            dist_left = c - t_min_c
            dist_right = t_max_c - c
            if dist_left <= dist_right:
                nc = t_min_c - dist_left
            else:
                nc = t_max_c + dist_right
            if 0<=nc<C:
                out[r][nc] = 5

    return out

def solve_f8b3ba0a(grid):
    """Grid of 2x2 colored blocks. Output minority colors sorted by count descending."""
    R, C = len(grid), len(grid[0])
    # Blocks at rows 1,3,5,... (step 2) and cols 1,4,7,10,... (step 3)
    blocks = []
    for r in range(1, R, 2):
        for c in range(1, C, 3):
            if r < R and c+1 < C:
                v = grid[r][c]
                blocks.append(v)

    counts = Counter(blocks)
    majority = max(counts, key=counts.get)
    minority = [(c, cnt) for c, cnt in counts.items() if c != majority]
    minority.sort(key=lambda x: -x[1])
    return [[c] for c, _ in minority]

def solve_f8c80d96(grid):
    """Zigzag staircase extended to fill grid."""
    R, C = len(grid), len(grid[0])
    # Find the staircase color and fill color
    non_zero = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] != 0:
                non_zero.add(grid[r][c])

    if not non_zero: return None

    stair_color = list(non_zero)[0]
    # The other color in output
    # Find from training: staircase color and fill color
    # The input has staircase of one color, output fills with staircase_color and another color (5)

    # The pattern: extend the staircase zigzag to fill the entire grid
    # Each horizontal line of 8 starts from left edge, and its length decreases by 2 each time
    # going from top to bottom (or vice versa)
    # Alternating with single-cell-high rows of fill_color

    # Detect the staircase from input and extrapolate
    # Find the step positions from the input
    steps = []
    for r in range(R):
        row_vals = [c for c in range(C) if grid[r][c] != 0]
        if row_vals:
            steps.append((r, min(row_vals), max(row_vals)))

    if not steps: return None

    # Determine fill color (5 based on training data patterns)
    fill_color = 5

    # Build the full zigzag
    # From the output pattern: alternating full-width 8-lines and 5-lines
    # But with a zigzag boundary
    # Column C-1 is always stair_color from some row
    # The pattern creates a zigzag from bottom-right expanding left and up

    # The staircase defines column boundaries at each row
    # Extrapolate: each pair of rows, the boundary moves 2 columns to the left
    # until reaching column 0, then starts from the right again

    # Looking at output row by row from bottom:
    # row 9: S,8,S,8,S,8,S,8,S,8 -> alternating from col 0
    # The rightmost column changes by some step

    # This is a complex pattern. Let me try to just extend the zigzag from input.
    return None

def solve_f8ff0b80(grid):
    """Multiple colored shapes -> 3x1 output sorted by size (descending)."""
    R, C = len(grid), len(grid[0])

    # Find connected components of non-zero cells
    visited = set()
    shapes = []
    for r in range(R):
        for c in range(C):
            if grid[r][c]!=0 and (r,c) not in visited:
                comp = []
                color = grid[r][c]
                stack = [(r,c)]
                while stack:
                    cr,cc = stack.pop()
                    if (cr,cc) in visited or grid[cr][cc]!=color: continue
                    visited.add((cr,cc))
                    comp.append((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=cr+dr,cc+dc
                        if 0<=nr<R and 0<=nc<C and (nr,nc) not in visited and grid[nr][nc]==color:
                            stack.append((nr,nc))
                shapes.append((color, len(comp)))

    # Sort by size descending
    shapes.sort(key=lambda x: -x[1])
    return [[color] for color, _ in shapes[:3]]

def solve_f9012d9b(grid):
    """Extract the 'missing' corner pattern from a tiled grid."""
    R, C = len(grid), len(grid[0])
    # The grid has a repeating 2x2 or similar tile pattern with a section of 0s
    # The 0 section indicates the missing part, output is the missing part's pattern

    # Find 0 cells
    zeros = [(r,c) for r in range(R) for c in range(C) if grid[r][c]==0]
    if not zeros: return grid

    min_r = min(r for r,c in zeros)
    max_r = max(r for r,c in zeros)
    min_c = min(c for r,c in zeros)
    max_c = max(c for r,c in zeros)

    h = max_r - min_r + 1
    w = max_c - min_c + 1

    # Find the tile pattern from non-zero area
    # The tile repeats. Find the period.
    # Look at row 0 to find column period

    # Extract what should be in the 0 region by finding the tile period
    # and filling in from the existing pattern

    # Find tile size by looking at the pattern
    # Try different periods
    for period_r in range(1, R):
        for period_c in range(1, C):
            valid = True
            for r in range(R):
                for c in range(C):
                    if grid[r][c] != 0:
                        # Check if this matches the periodic pattern
                        ref_r = r % period_r
                        ref_c = c % period_c
                        # Find the reference cell
                        found = False
                        for rr in range(R):
                            for cc in range(C):
                                if rr % period_r == ref_r and cc % period_c == ref_c and grid[rr][cc] != 0:
                                    if grid[rr][cc] != grid[r][c]:
                                        valid = False
                                    found = True
                                    break
                            if found: break
                    if not valid: break
                if not valid: break

            if valid:
                # Found the period. Now fill in the missing region.
                # Build the tile
                tile = [[0]*period_c for _ in range(period_r)]
                for r in range(R):
                    for c in range(C):
                        if grid[r][c] != 0:
                            tile[r % period_r][c % period_c] = grid[r][c]

                out = [[0]*w for _ in range(h)]
                for r in range(h):
                    for c in range(w):
                        out[r][c] = tile[(min_r+r) % period_r][(min_c+c) % period_c]
                return out

    return [[0]*w for _ in range(h)]

def solve_fafffa47(grid):
    """6x3 -> 3x3. Output 2 where both halves are 0."""
    R, C = len(grid), len(grid[0])
    half = R // 2
    out = [[0]*C for _ in range(half)]
    for r in range(half):
        for c in range(C):
            if grid[r][c]==0 and grid[r+half][c]==0:
                out[r][c] = 2
    return out

def solve_fcb5c309(grid):
    """Bordered rectangle. Extract largest border-color rect, replace border color with other color."""
    R, C = len(grid), len(grid[0])

    # Find all non-zero colors
    colors = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] != 0:
                colors.add(grid[r][c])

    if len(colors) < 2: return None

    # Try each color as the border color
    # The border forms a connected rectangle. Find the largest connected component of each color.
    best_rect = None
    best_size = 0
    border_color = None

    for color in colors:
        cells = set((r,c) for r in range(R) for c in range(C) if grid[r][c]==color)
        visited = set()
        for r,c in cells:
            if (r,c) not in visited:
                comp = []
                stack = [(r,c)]
                while stack:
                    cr,cc = stack.pop()
                    if (cr,cc) in visited or (cr,cc) not in cells: continue
                    visited.add((cr,cc))
                    comp.append((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=cr+dr,cc+dc
                        if (nr,nc) in cells and (nr,nc) not in visited:
                            stack.append((nr,nc))

                if len(comp) > best_size:
                    best_size = len(comp)
                    best_rect = comp
                    border_color = color

    if not best_rect: return None

    min_r = min(r for r,c in best_rect)
    max_r = max(r for r,c in best_rect)
    min_c = min(c for r,c in best_rect)
    max_c = max(c for r,c in best_rect)

    other_color = [c for c in colors if c != border_color][0]

    # Extract sub-grid, replace border_color with other_color
    out = []
    for r in range(min_r, max_r+1):
        row = []
        for c in range(min_c, max_c+1):
            v = grid[r][c]
            if v == border_color:
                row.append(other_color)
            else:
                row.append(v)
        out.append(row)

    return out

def solve_fcc82909(grid):
    """2x2 colored blocks. Extend with 3s to fill remaining space below/around."""
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # Find 2x2 colored blocks
    visited = set()
    blocks = []
    for r in range(R):
        for c in range(C):
            if grid[r][c]!=0 and (r,c) not in visited:
                comp = []
                stack = [(r,c)]
                while stack:
                    cr,cc = stack.pop()
                    if (cr,cc) in visited or grid[cr][cc]==0: continue
                    visited.add((cr,cc))
                    comp.append((cr,cc,grid[cr][cc]))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=cr+dr,cc+dc
                        if 0<=nr<R and 0<=nc<C and (nr,nc) not in visited and grid[nr][nc]!=0:
                            stack.append((nr,nc))
                blocks.append(comp)

    # Each block is a 2x2 colored pattern. Extend downward with 3s until hitting grid edge or another block.
    # Looking at the output: below each 2x2 block, a column of 3x2 (same width) extends downward
    # Actually: fill 3s in the columns of each block going downward (and upward?) to fill empty space

    for block in blocks:
        min_r = min(r for r,c,v in block)
        max_r = max(r for r,c,v in block)
        min_c = min(c for r,c,v in block)
        max_c = max(c for r,c,v in block)

        # Extend downward
        for r in range(max_r+1, R):
            all_zero = all(grid[r][c]==0 for c in range(min_c, max_c+1))
            if not all_zero: break
            for c in range(min_c, max_c+1):
                out[r][c] = 3

        # Extend upward
        for r in range(min_r-1, -1, -1):
            all_zero = all(grid[r][c]==0 for c in range(min_c, max_c+1))
            if not all_zero: break
            for c in range(min_c, max_c+1):
                out[r][c] = 3

    return out

def solve_feca6190(grid):
    """1x5 -> NxN diagonal pattern. N = 5 * count(non-zero)."""
    vals = grid[0]
    non_zero = [(i,v) for i,v in enumerate(vals) if v != 0]
    N = 5 * len(non_zero)

    out = [[0]*N for _ in range(N)]
    for pos, val in non_zero:
        # Place on diagonal: at row r, col = N-1-r+pos
        for r in range(N):
            c = N - 1 - r + pos
            if 0 <= c < N:
                out[r][c] = val
    return out

def solve_ff28f65a(grid):
    """Count 2x2 blocks and fill 3x3 output in diagonal order."""
    R, C = len(grid), len(grid[0])
    blocks = []
    for r in range(R-1):
        for c in range(C-1):
            if grid[r][c]==2 and grid[r][c+1]==2 and grid[r+1][c]==2 and grid[r+1][c+1]==2:
                blocks.append((r,c))

    order = [(0,0),(0,2),(1,1),(2,0),(2,2),(0,1),(1,0),(1,2),(2,1)]
    out = [[0]*3 for _ in range(3)]
    for i in range(min(len(blocks), len(order))):
        r,c = order[i]
        out[r][c] = 1
    return out

# Test framework
def test_solver(tid, solver):
    data = load_task(tid)
    all_pairs = data['train'] + data['test']
    correct = 0
    for pair in all_pairs:
        try:
            result = solver(pair['input'])
            if result is not None and result == pair['output']:
                correct += 1
        except: pass
    return correct, len(all_pairs)

solvers = {
    'e3497940': solve_e3497940,
    'e40b9e2f': solve_e40b9e2f,
    'e48d4e1a': solve_e48d4e1a,
    'e5062a87': solve_e5062a87,
    'e50d258f': solve_e50d258f,
    'e76a88a6': solve_e76a88a6,
    'e8593010': solve_e8593010,
    'e8dc4411': solve_e8dc4411,
    'e9614598': solve_e9614598,
    'e98196ab': solve_e98196ab,
    'e9afcf9a': solve_e9afcf9a,
    'ea32f347': solve_ea32f347,
    'ea786f4a': solve_ea786f4a,
    'ec883f72': solve_ec883f72,
    'ecdecbb3': solve_ecdecbb3,
    'ef135b50': solve_ef135b50,
    'f25fbde4': solve_f25fbde4,
    'f2829549': solve_f2829549,
    'f35d900a': solve_f35d900a,
    'f5b8619d': solve_f5b8619d,
    'f76d97a5': solve_f76d97a5,
    'f8a8fe49': solve_f8a8fe49,
    'f8b3ba0a': solve_f8b3ba0a,
    'f8c80d96': solve_f8c80d96,
    'f8ff0b80': solve_f8ff0b80,
    'f9012d9b': solve_f9012d9b,
    'fafffa47': solve_fafffa47,
    'fcb5c309': solve_fcb5c309,
    'fcc82909': solve_fcc82909,
    'feca6190': solve_feca6190,
    'ff28f65a': solve_ff28f65a,
}

if __name__ == '__main__':
    results = {}
    for tid, solver in solvers.items():
        try:
            correct, total = test_solver(tid, solver)
            results[tid] = (correct, total)
            status = "PASS" if correct==total else f"{correct}/{total}"
            print(f"{tid}: {status}")
        except Exception as e:
            results[tid] = (0, 0)
            print(f"{tid}: ERROR - {e}")

    passed = sum(1 for c,t in results.values() if c==t and t>0)
    print(f"\nPassed: {passed}/{len(results)}")

    # Save solutions
    solutions = {}
    for tid, solver in solvers.items():
        data = load_task(tid)
        c, t = results.get(tid, (0,0))
        if c == t and t > 0:
            test_outputs = []
            for pair in data['test']:
                try:
                    result = solver(pair['input'])
                    test_outputs.append(result)
                except:
                    test_outputs.append(None)
            solutions[tid] = test_outputs

    with open('data/arc_python_solutions_b24.json', 'w') as f:
        json.dump(solutions, f)
    print(f"\nSaved {len(solutions)} solutions to data/arc_python_solutions_b24.json")
