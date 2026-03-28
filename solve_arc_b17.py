import json

def solve_aba27056(inp):
    rows, cols = len(inp), len(inp[0])
    shape_cells = set()
    for r in range(rows):
        for c in range(cols):
            if inp[r][c] != 0:
                shape_cells.add((r,c))
    if not shape_cells:
        return inp

    min_r = min(r for r,c in shape_cells)
    max_r = max(r for r,c in shape_cells)
    min_c = min(c for r,c in shape_cells)
    max_c = max(c for r,c in shape_cells)

    bbox_interior = set()
    for r in range(min_r, max_r+1):
        for c in range(min_c, max_c+1):
            if (r,c) not in shape_cells:
                bbox_interior.add((r,c))

    grid = [row[:] for row in inp]
    for r,c in bbox_interior:
        grid[r][c] = 4

    # Find opening side of bounding box
    top_gap = [(min_r, c) for c in range(min_c, max_c+1) if (min_r, c) not in shape_cells]
    bottom_gap = [(max_r, c) for c in range(min_c, max_c+1) if (max_r, c) not in shape_cells]
    left_gap = [(r, min_c) for r in range(min_r, max_r+1) if (r, min_c) not in shape_cells]
    right_gap = [(r, max_c) for r in range(min_r, max_r+1) if (r, max_c) not in shape_cells]

    gaps = [
        (len(top_gap), top_gap, (-1, 0)),
        (len(bottom_gap), bottom_gap, (1, 0)),
        (len(left_gap), left_gap, (0, -1)),
        (len(right_gap), right_gap, (0, 1)),
    ]
    gaps.sort(key=lambda x: -x[0])
    gap_cells = gaps[0][1]
    dr, dc = gaps[0][2]
    if not gap_cells:
        return grid

    # Straight rays from all gap cells
    for r, c in gap_cells:
        nr, nc = r + dr, c + dc
        while 0 <= nr < rows and 0 <= nc < cols:
            if (nr, nc) in shape_cells:
                break
            grid[nr][nc] = 4
            nr += dr
            nc += dc

    # Diagonal rays from corner gap cells
    if dr == 0:
        gap_cells_sorted = sorted(gap_cells, key=lambda x: x[0])
        perp_dirs = [(-1, 0), (1, 0)]
    else:
        gap_cells_sorted = sorted(gap_cells, key=lambda x: x[1])
        perp_dirs = [(0, -1), (0, 1)]

    corner1 = gap_cells_sorted[0]
    diag1 = (dr + perp_dirs[0][0], dc + perp_dirs[0][1])
    nr, nc = corner1[0] + diag1[0], corner1[1] + diag1[1]
    while 0 <= nr < rows and 0 <= nc < cols:
        if (nr, nc) in shape_cells:
            break
        grid[nr][nc] = 4
        nr += diag1[0]
        nc += diag1[1]

    corner2 = gap_cells_sorted[-1]
    diag2 = (dr + perp_dirs[1][0], dc + perp_dirs[1][1])
    nr, nc = corner2[0] + diag2[0], corner2[1] + diag2[1]
    while 0 <= nr < rows and 0 <= nc < cols:
        if (nr, nc) in shape_cells:
            break
        grid[nr][nc] = 4
        nr += diag2[0]
        nc += diag2[1]

    return grid


def solve_ac0a08a4(grid):
    rows, cols = len(grid), len(grid[0])
    non_zero = sum(1 for r in range(rows) for c in range(cols) if grid[r][c] != 0)
    n = non_zero
    result = [[0]*(cols*n) for _ in range(rows*n)]
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            for dr in range(n):
                for dc in range(n):
                    result[r*n+dr][c*n+dc] = val
    return result


def solve_ae3edfdc(grid):
    rows, cols = len(grid), len(grid[0])
    cells = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color = grid[r][c]
                if color not in cells:
                    cells[color] = []
                cells[color].append((r, c))

    centers = {}
    satellite_colors = {}
    for color, positions in cells.items():
        if len(positions) == 1:
            centers[color] = positions[0]
        else:
            satellite_colors[color] = positions

    pairings = {}
    for sat_color, sat_positions in satellite_colors.items():
        best_center = None
        best_count = 0
        for cen_color, cen_pos in centers.items():
            cr, cc = cen_pos
            count = sum(1 for sr, sc in sat_positions if sr == cr or sc == cc)
            if count > best_count:
                best_count = count
                best_center = cen_color
        pairings[sat_color] = best_center

    result = [[0]*cols for _ in range(rows)]
    for sat_color, cen_color in pairings.items():
        cr, cc = centers[cen_color]
        result[cr][cc] = cen_color
        for sr, sc in satellite_colors[sat_color]:
            dr = 0 if sr == cr else (1 if sr > cr else -1)
            dc = 0 if sc == cc else (1 if sc > cc else -1)
            result[cr + dr][cc + dc] = sat_color
    return result


def solve_ae4f1146(grid):
    rows, cols = len(grid), len(grid[0])
    visited = [[False]*cols for _ in range(rows)]
    blocks = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                block_cells = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    block_cells.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != 0:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                blocks.append(block_cells)

    best_block = None
    best_count = -1
    for block_cells in blocks:
        count_1s = sum(1 for r, c in block_cells if grid[r][c] == 1)
        if count_1s > best_count:
            best_count = count_1s
            min_r = min(r for r, c in block_cells)
            min_c = min(c for r, c in block_cells)
            best_block = (min_r, min_c)

    r0, c0 = best_block
    return [[grid[r][c] for c in range(c0, c0+3)] for r in range(r0, r0+3)]


def solve_aedd82e4(grid):
    grid = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                has_neighbor = False
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 2:
                        has_neighbor = True
                        break
                if not has_neighbor:
                    grid[r][c] = 1
    return grid


def solve_af902bf9(grid):
    grid = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])
    fours = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 4]

    fours_set = set(fours)
    used = set()
    for i in range(len(fours)):
        for j in range(i+1, len(fours)):
            r1, c1 = fours[i]
            r2, c2 = fours[j]
            if r1 != r2 and c1 != c2:
                if (r1, c2) in fours_set and (r2, c1) in fours_set:
                    rect = tuple(sorted([(r1,c1),(r1,c2),(r2,c1),(r2,c2)]))
                    if rect not in used:
                        used.add(rect)
                        mr, Mr = min(r1,r2), max(r1,r2)
                        mc, Mc = min(c1,c2), max(c1,c2)
                        for r in range(mr+1, Mr):
                            for c in range(mc+1, Mc):
                                grid[r][c] = 2
    return grid


def solve_b0c4d837(grid):
    rows, cols = len(grid), len(grid[0])
    fives = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                fives.add((r, c))

    min_r = min(r for r, c in fives)
    max_r = max(r for r, c in fives)
    min_c = min(c for r, c in fives)
    max_c = max(c for r, c in fives)

    interior_cols = list(range(min_c+1, max_c))
    wall_top = None
    for r in range(min_r, max_r+1):
        if (r, min_c) in fives and (r, max_c) in fives:
            if wall_top is None:
                wall_top = r

    interior_rows = list(range(wall_top, max_r))
    empty_rows = 0
    for r in interior_rows:
        has_eight = any(grid[r][c] == 8 for c in interior_cols)
        if not has_eight:
            empty_rows += 1

    spiral_order = [(0,0),(0,1),(0,2),(1,2),(2,2),(2,1),(2,0),(1,0),(1,1)]
    result = [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(min(empty_rows, 9)):
        r, c = spiral_order[i]
        result[r][c] = 8
    return result


def solve_b190f7f5(grid):
    rows, cols = len(grid), len(grid[0])

    if cols > rows:
        half_c = cols // 2
        left_has_8 = any(grid[r][c] == 8 for r in range(rows) for c in range(half_c))
        if not left_has_8:
            color_map = [row[:half_c] for row in grid]
            shape_grid = [row[half_c:] for row in grid]
        else:
            color_map = [row[half_c:] for row in grid]
            shape_grid = [row[:half_c] for row in grid]
        map_rows, map_cols = rows, half_c
    else:
        half_r = rows // 2
        top_has_8 = any(grid[r][c] == 8 for r in range(half_r) for c in range(cols))
        if top_has_8:
            shape_grid = [grid[r][:] for r in range(half_r)]
            color_map = [grid[r][:] for r in range(half_r, rows)]
        else:
            color_map = [grid[r][:] for r in range(half_r)]
            shape_grid = [grid[r][:] for r in range(half_r, rows)]
        map_rows, map_cols = half_r, cols

    sh = len(shape_grid)
    sw = len(shape_grid[0])
    shape_mask = [[1 if shape_grid[r][c] == 8 else 0 for c in range(sw)] for r in range(sh)]

    result = [[0]*(map_cols*sw) for _ in range(map_rows*sh)]
    for r in range(map_rows):
        for c in range(map_cols):
            color = color_map[r][c]
            if color != 0 and color != 8:
                for sr in range(sh):
                    for sc in range(sw):
                        if shape_mask[sr][sc]:
                            result[r*sh + sr][c*sw + sc] = color
    return result


def solve_b230c067(grid):
    grid = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    def normalize_shape(comp):
        min_r = min(r for r,c in comp)
        min_c = min(c for r,c in comp)
        return frozenset((r-min_r, c-min_c) for r,c in comp)

    visited = [[False]*cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8 and not visited[r][c]:
                component = set()
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    component.add((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 8:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                components.append(component)

    # Normalize shapes and count duplicates
    norms = [normalize_shape(c) for c in components]
    norm_counts = {}
    for n in norms:
        norm_counts[n] = norm_counts.get(n, 0) + 1

    for i, component in enumerate(components):
        is_unique = (norm_counts[norms[i]] == 1)
        color = 2 if is_unique else 1
        for r, c in component:
            grid[r][c] = color

    return grid


def solve_b2862040(grid):
    grid = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    visited = [[False]*cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and not visited[r][c]:
                component = set()
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    component.add((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 1:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                components.append(component)

    # Flood fill 9-cells from boundary
    reachable = set()
    bfs = []
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows-1 or c == 0 or c == cols-1) and grid[r][c] == 9:
                if (r, c) not in reachable:
                    reachable.add((r, c))
                    bfs.append((r, c))
    while bfs:
        r, c = bfs.pop()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in reachable and grid[nr][nc] == 9:
                reachable.add((nr, nc))
                bfs.append((nr, nc))

    for component in components:
        min_r = min(r for r, c in component)
        max_r = max(r for r, c in component)
        min_c = min(c for r, c in component)
        max_c = max(c for r, c in component)

        has_interior = False
        for r in range(min_r, max_r+1):
            for c in range(min_c, max_c+1):
                if grid[r][c] == 9 and (r, c) not in reachable:
                    has_interior = True
                    break
            if has_interior:
                break

        if has_interior:
            for r, c in component:
                grid[r][c] = 8

    return grid


# ===== TEST ALL SOLUTIONS =====
task_ids = [
    'aba27056', 'ac0a08a4', 'ae3edfdc', 'ae4f1146', 'aedd82e4',
    'af902bf9', 'b0c4d837', 'b190f7f5', 'b230c067', 'b2862040'
]
solvers = {
    'aba27056': solve_aba27056,
    'ac0a08a4': solve_ac0a08a4,
    'ae3edfdc': solve_ae3edfdc,
    'ae4f1146': solve_ae4f1146,
    'aedd82e4': solve_aedd82e4,
    'af902bf9': solve_af902bf9,
    'b0c4d837': solve_b0c4d837,
    'b190f7f5': solve_b190f7f5,
    'b230c067': solve_b230c067,
    'b2862040': solve_b2862040,
}

all_pass = True
solutions = {}

for task_id in task_ids:
    with open(f'data/arc1/{task_id}.json') as f:
        task = json.load(f)
    solver = solvers[task_id]

    task_pass = True
    for i, pair in enumerate(task['train'] + task['test']):
        result = solver(pair['input'])
        expected = pair['output']
        match = result == expected
        if not match:
            task_pass = False
            all_pass = False
        print(f'{task_id} pair {i}: {"PASS" if match else "FAIL"}')
        if not match:
            for r in range(min(len(result), len(expected))):
                for c in range(min(len(result[0]), len(expected[0]))):
                    if result[r][c] != expected[r][c]:
                        print(f'  ({r},{c}): got {result[r][c]}, exp {expected[r][c]}')

    # Generate test output
    test_outputs = []
    for test_pair in task['test']:
        test_outputs.append(solver(test_pair['input']))
    solutions[task_id] = test_outputs

    print(f'{task_id}: {"ALL PASS" if task_pass else "SOME FAIL"}')
    print()

print(f'\nOverall: {"ALL PASS" if all_pass else "SOME FAIL"}')

# Save solutions
with open('data/arc_python_solutions_b17.json', 'w') as f:
    json.dump(solutions, f)
print('Saved to data/arc_python_solutions_b17.json')
