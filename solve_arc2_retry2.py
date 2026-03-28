import json, os, sys
from collections import Counter, defaultdict, deque

sys.setrecursionlimit(50000)

DATA_DIR = "data/arc2"
OUT_FILE = "data/arc2_solutions_retry2.json"

def load_task(tid):
    with open(f"{DATA_DIR}/{tid}.json") as f:
        return json.load(f)

def test_solver(tid, solver):
    task = load_task(tid)
    for i, pair in enumerate(task['train']):
        try:
            result = solver(pair['input'])
        except Exception as e:
            return False
        if result != pair['output']:
            return False
    return True

def get_test_solutions(tid, solver):
    task = load_task(tid)
    return [solver(tp['input']) for tp in task['test']]

###############################################################################
# SOLVERS
###############################################################################

def solve_8b7bacbf(grid):
    """Fill interior of 2-rectangle frames with marker color.
    Key: each 2-frame is a hollow diamond/rectangle. Interior 0s get filled.
    Use per-cell check: for each 0-cell, if in every cardinal direction the
    NEAREST non-0 cell is a 2, then it's interior."""
    g = [row[:] for row in grid]
    R, C = len(g), len(g[0])

    marker_val = None
    for r in range(R):
        for c in range(C):
            if g[r][c] not in (0, 1, 2):
                marker_val = g[r][c]
    if marker_val is None:
        return g

    out = [row[:] for row in g]

    # For each 0-cell, check if the nearest non-0 cell in each of 4 cardinal
    # directions is a 2-cell.
    for r in range(R):
        for c in range(C):
            if g[r][c] != 0:
                continue

            all_dirs_2 = True
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                found_2 = False
                while 0 <= nr < R and 0 <= nc < C:
                    if g[nr][nc] != 0:
                        if g[nr][nc] == 2:
                            found_2 = True
                        break
                    nr += dr
                    nc += dc
                if not found_2:
                    all_dirs_2 = False
                    break

            if all_dirs_2:
                out[r][c] = marker_val

    return out

def solve_800d221b(grid):
    """Replace a 'border' color with the local region's dominant color.
    The border color separates regions of different colors."""
    g = [row[:] for row in grid]
    R, C = len(g), len(g[0])
    all_colors = set(g[r][c] for r in range(R) for c in range(C))

    best_border = None
    best_score = -1

    for border_color in all_colors:
        visit = set()
        regions = []
        for r in range(R):
            for c in range(C):
                if g[r][c] != border_color and (r,c) not in visit:
                    queue = deque([(r,c)])
                    visit.add((r,c))
                    region_cells = [(r,c)]
                    while queue:
                        cr, cc = queue.popleft()
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0 <= nr < R and 0 <= nc < C and (nr,nc) not in visit and g[nr][nc] != border_color:
                                visit.add((nr,nc))
                                queue.append((nr,nc))
                                region_cells.append((nr,nc))
                    colors_in_region = set(g[rr][cc] for rr, cc in region_cells)
                    regions.append((colors_in_region, region_cells))

        mono_count = sum(1 for colors, cells in regions if len(colors) == 1)
        total_mono_cells = sum(len(cells) for colors, cells in regions if len(colors) == 1)
        score = total_mono_cells
        if len(regions) >= 2 and score > best_score:
            best_score = score
            best_border = border_color

    if best_border is None:
        return g

    border_color = best_border
    out = [row[:] for row in g]

    region_map = [[None]*C for _ in range(R)]
    visit = set()
    region_colors = {}
    region_id = 0

    for r in range(R):
        for c in range(C):
            if g[r][c] != border_color and (r,c) not in visit:
                queue = deque([(r,c)])
                visit.add((r,c))
                region_cells = [(r,c)]
                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < R and 0 <= nc < C and (nr,nc) not in visit and g[nr][nc] != border_color:
                            visit.add((nr,nc))
                            queue.append((nr,nc))
                            region_cells.append((nr,nc))

                color_counts = Counter(g[rr][cc] for rr, cc in region_cells)
                dominant = color_counts.most_common(1)[0][0]
                region_colors[region_id] = dominant
                for rr, cc in region_cells:
                    region_map[rr][cc] = region_id
                region_id += 1

    dist = [[float('inf')]*C for _ in range(R)]
    q = deque()
    for r in range(R):
        for c in range(C):
            if region_map[r][c] is not None:
                dist[r][c] = 0
                q.append((r,c))

    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < R and 0 <= nc < C and dist[nr][nc] > dist[r][c] + 1:
                dist[nr][nc] = dist[r][c] + 1
                region_map[nr][nc] = region_map[r][c]
                q.append((nr,nc))

    for r in range(R):
        for c in range(C):
            if g[r][c] == border_color:
                rid = region_map[r][c]
                if rid is not None and rid in region_colors:
                    out[r][c] = region_colors[rid]

    return out

def solve_88bcf3b4(grid):
    """A small shape slides along a straight line toward a longer line,
    stopping adjacent to it. The shape's cells on the line's side merge."""
    g = [row[:] for row in grid]
    R, C = len(g), len(g[0])
    bg = Counter(g[r][c] for r in range(R) for c in range(C)).most_common(1)[0][0]

    visited = [[False]*C for _ in range(R)]
    components = []
    for sr in range(R):
        for sc in range(C):
            if g[sr][sc] != bg and not visited[sr][sc]:
                comp = []
                color = g[sr][sc]
                q = deque([(sr, sc)])
                visited[sr][sc] = True
                while q:
                    r, c = q.popleft()
                    comp.append((r, c))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and g[nr][nc] != bg:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append((g[comp[0][0]][comp[0][1]], comp))

    if len(components) < 2:
        return g

    # Find the line (longest straight component)
    line_comp = None
    line_idx = -1
    for idx, (color, cells) in enumerate(components):
        rows = set(r for r,c in cells)
        cols = set(c for r,c in cells)
        if len(rows) == 1 or len(cols) == 1:
            if line_comp is None or len(cells) > len(line_comp[1]):
                line_comp = (color, cells)
                line_idx = idx

    if line_comp is None:
        return g

    line_color, line_cells = line_comp
    line_set = set(line_cells)
    line_rows = [r for r,c in line_cells]
    line_cols = [c for r,c in line_cells]

    # Determine line orientation
    if len(set(line_cols)) == 1:
        line_orient = 'v'
        line_c = line_cols[0]
    else:
        line_orient = 'h'
        line_r = line_rows[0]

    # Other components slide toward the line
    out = [row[:] for row in g]

    for idx, (color, cells) in enumerate(components):
        if idx == line_idx:
            continue

        cell_colors = [(r, c, g[r][c]) for r, c in cells]

        if line_orient == 'v':
            shape_center_c = sum(c for r,c in cells) / len(cells)
            dc = 1 if shape_center_c < line_c else -1
            # Find min distance to slide
            min_dist = R + C
            for r, c in cells:
                dist = 0
                nc = c + dc
                while 0 <= nc < C and (r, nc) not in line_set:
                    dist += 1
                    nc += dc
                if 0 <= nc < C and (r, nc) in line_set:
                    min_dist = min(min_dist, dist)

            # Clear and redraw
            for r, c in cells:
                out[r][c] = bg
            for r, c, col in cell_colors:
                nr, nc = r, c + dc * min_dist
                if 0 <= nr < R and 0 <= nc < C:
                    out[nr][nc] = col
        else:
            shape_center_r = sum(r for r,c in cells) / len(cells)
            dr = 1 if shape_center_r < line_r else -1
            min_dist = R + C
            for r, c in cells:
                dist = 0
                nr = r + dr
                while 0 <= nr < R and (nr, c) not in line_set:
                    dist += 1
                    nr += dr
                if 0 <= nr < R and (nr, c) in line_set:
                    min_dist = min(min_dist, dist)

            for r, c in cells:
                out[r][c] = bg
            for r, c, col in cell_colors:
                nr, nc = r + dr * min_dist, c
                if 0 <= nr < R and 0 <= nc < C:
                    out[nr][nc] = col

    return out

###############################################################################
# MAIN
###############################################################################

SOLVERS = {
    '8b7bacbf': solve_8b7bacbf,
    '800d221b': solve_800d221b,
    '88bcf3b4': solve_88bcf3b4,
}

def main():
    solutions = {}

    for tid, solver in SOLVERS.items():
        print(f"Testing {tid}...", end=" ")
        if test_solver(tid, solver):
            print("ALL PASS")
            solutions[tid] = get_test_solutions(tid, solver)
        else:
            print("FAIL")
            # Show details
            task = load_task(tid)
            for i, pair in enumerate(task['train']):
                try:
                    result = solver(pair['input'])
                    if result == pair['output']:
                        print(f"  train[{i}]: PASS")
                    else:
                        diffs = sum(1 for r in range(len(pair['output'])) for c in range(len(pair['output'][0]))
                                   if r<len(result) and c<len(result[0]) and result[r][c]!=pair['output'][r][c])
                        sz_match = len(result)==len(pair['output']) and (not result or len(result[0])==len(pair['output'][0]))
                        print(f"  train[{i}]: FAIL ({diffs} diffs, size_match={sz_match})")
                except Exception as e:
                    print(f"  train[{i}]: ERROR {e}")

    os.makedirs(os.path.dirname(OUT_FILE) if os.path.dirname(OUT_FILE) else '.', exist_ok=True)
    with open(OUT_FILE, 'w') as f:
        json.dump(solutions, f)

    print(f"\nSaved {len(solutions)} task solutions to {OUT_FILE}")

if __name__ == '__main__':
    main()
