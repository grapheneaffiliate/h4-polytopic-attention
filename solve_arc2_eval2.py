#!/usr/bin/env python3
"""ARC-AGI-2 Evaluation Solver - Batch 2"""
import json
import os
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "arc2")
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "arc2_solutions_eval2.json")

TASK_IDS = "7b80bb43,7c66cb00,7ed72f31,800d221b,80a900e0,8698868d,88bcf3b4,88e364bc,89565ca0,898e7135,8b7bacbf,8b9c3697,8e5c0c38,8f215267,8f3a5a89,9385bd28,97d7923e,981571dc,9aaea919,9bbf930d,a251c730,a25697e4,a32d8b75,a395ee82,a47bf94d,a6f40cea,aa4ec2a5,abc82100,b0039139,b10624e5".split(",")


def load_task(tid):
    with open(os.path.join(DATA_DIR, f"{tid}.json")) as f:
        return json.load(f)


def test_solver(task, solver):
    import copy
    for pair in task["train"]:
        try:
            result = solver(copy.deepcopy(pair["input"]))
            if result != pair["output"]:
                return False
        except Exception:
            return False
    return True


def apply_solver(task, solver):
    import copy
    results = []
    for pair in task["test"]:
        results.append(solver(copy.deepcopy(pair["input"])))
    return results


# ============================================================
# TASK: aa4ec2a5
# Rectangular outlines of color 1 on background 4:
# - Outlines with holes: 1->8, interior 4->6, adjacent 4->2
# - Solid shapes: keep 1, add border of 2
# ============================================================
def solve_aa4ec2a5(grid):
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    bg = 4
    shape_color = 1

    shape_cells = {(r, c) for r in range(R) for c in range(C) if grid[r][c] == shape_color}

    visited = set()
    comps = []
    for start in shape_cells:
        if start in visited:
            continue
        comp = set()
        queue = [start]
        while queue:
            cur = queue.pop(0)
            if cur in comp:
                continue
            comp.add(cur)
            r, c = cur
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in shape_cells and (nr, nc) not in comp:
                    queue.append((nr, nc))
        visited |= comp
        comps.append(comp)

    for comp in comps:
        rs = [r for r, c in comp]
        cs = [c for r, c in comp]
        r1, r2 = min(rs), max(rs)
        c1, c2 = min(cs), max(cs)

        pr1, pr2 = max(0, r1 - 1), min(R - 1, r2 + 1)
        pc1, pc2 = max(0, c1 - 1), min(C - 1, c2 + 1)

        exterior = set()
        fill_queue = []
        for r in range(pr1, pr2 + 1):
            for c in range(pc1, pc2 + 1):
                if r in (pr1, pr2) or c in (pc1, pc2):
                    if grid[r][c] == bg:
                        fill_queue.append((r, c))

        while fill_queue:
            r, c = fill_queue.pop(0)
            if (r, c) in exterior:
                continue
            if r < pr1 or r > pr2 or c < pc1 or c > pc2:
                continue
            if (r, c) in comp:
                continue
            if grid[r][c] != bg:
                continue
            exterior.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if pr1 <= nr <= pr2 and pc1 <= nc <= pc2 and (nr, nc) not in exterior:
                    fill_queue.append((nr, nc))

        interior = set()
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if grid[r][c] == bg and (r, c) not in exterior and (r, c) not in comp:
                    interior.add((r, c))

        has_holes = len(interior) > 0

        if has_holes:
            for r, c in comp:
                out[r][c] = 8
            for r, c in interior:
                out[r][c] = 6

        for r, c in comp:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in comp and (nr, nc) not in interior:
                    if grid[nr][nc] == bg:
                        out[nr][nc] = 2

    return out


# ============================================================
# TASK: 8f3a5a89
# Grid of 8s with 1-shapes and a 6-marker.
# Draw a border of 7 around the reachable region.
# Remove 1-clusters on opposite side of complete walls.
# Only border around 1-clusters touching the grid edge.
# ============================================================
def solve_8f3a5a89(grid):
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    marker = None
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 6:
                marker = (r, c)
    if not marker:
        return grid

    bg = 8
    wall = 1
    border_color = 7

    reachable = set()
    queue = [marker]
    reachable.add(marker)
    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in reachable:
                if grid[nr][nc] != wall:
                    reachable.add((nr, nc))
                    queue.append((nr, nc))

    all_walls = {(r, c) for r in range(R) for c in range(C) if grid[r][c] == wall}
    visited = set()
    wall_comps = []
    for start in all_walls:
        if start in visited:
            continue
        comp = set()
        q = [start]
        while q:
            cur = q.pop(0)
            if cur in comp:
                continue
            comp.add(cur)
            rr, cc = cur
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = rr + dr, cc + dc
                if (nr, nc) in all_walls and (nr, nc) not in comp:
                    q.append((nr, nc))
        visited |= comp
        wall_comps.append(comp)

    edge_walls = set()
    for comp in wall_comps:
        touches_edge = any(r == 0 or r == R - 1 or c == 0 or c == C - 1 for r, c in comp)
        if touches_edge:
            edge_walls |= comp

    has_complete_wall = False
    for c in range(C):
        if all(grid[r][c] == wall for r in range(R)):
            has_complete_wall = True
            break
    for r in range(R):
        if all(grid[r][c] == wall for c in range(C)):
            has_complete_wall = True
            break

    if has_complete_wall:
        adjacent_walls = set()
        for r, c in reachable:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == wall:
                    adjacent_walls.add((nr, nc))
        for r, c in all_walls - adjacent_walls:
            out[r][c] = bg

    for r, c in reachable:
        if grid[r][c] == 6:
            continue
        is_border = False
        if r == 0 or r == R - 1 or c == 0 or c == C - 1:
            is_border = True
        if not is_border:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= R or nc < 0 or nc >= C:
                        is_border = True
                        break
                    if (nr, nc) in edge_walls:
                        is_border = True
                        break
                if is_border:
                    break
        if is_border:
            out[r][c] = border_color

    return out


# ============================================================
# TASK: 7ed72f31
# Colored clusters on background of 1s with 2-colored axis cells.
# Each cluster reflects 180 degrees around the 2-cells (axis of symmetry).
# ============================================================
def solve_7ed72f31(grid):
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    from collections import Counter as Ctr
    flat = [grid[r][c] for r in range(R) for c in range(C)]
    bg = Ctr(flat).most_common(1)[0][0]
    non_bg = {}
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg:
                non_bg[(r, c)] = grid[r][c]
    axis_color = 2
    all_non_bg = set(non_bg.keys())
    visited = set()
    clusters = []
    for start in all_non_bg:
        if start in visited:
            continue
        comp = set()
        q = [start]
        while q:
            cur = q.pop(0)
            if cur in comp:
                continue
            comp.add(cur)
            r, c = cur
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in all_non_bg and (nr, nc) not in comp:
                        q.append((nr, nc))
        visited |= comp
        clusters.append(comp)
    for cluster in clusters:
        axis_cells = [(r, c) for r, c in cluster if non_bg[(r, c)] == axis_color]
        colored_cells = [(r, c) for r, c in cluster if non_bg[(r, c)] != axis_color]
        if not axis_cells or not colored_cells:
            continue
        if len(axis_cells) == 1:
            ar, ac = axis_cells[0]
            for cr, cc in colored_cells:
                color = non_bg[(cr, cc)]
                nr, nc = 2 * ar - cr, 2 * ac - cc
                if 0 <= nr < R and 0 <= nc < C and out[nr][nc] == bg:
                    out[nr][nc] = color
        else:
            axis_rs = [r for r, c in axis_cells]
            axis_cs = [c for r, c in axis_cells]
            if len(set(axis_rs)) == 1:
                axis_r = axis_rs[0]
                for cr, cc in colored_cells:
                    color = non_bg[(cr, cc)]
                    nr = 2 * axis_r - cr
                    if 0 <= nr < R and out[nr][cc] == bg:
                        out[nr][cc] = color
            elif len(set(axis_cs)) == 1:
                axis_c = axis_cs[0]
                for cr, cc in colored_cells:
                    color = non_bg[(cr, cc)]
                    nc = 2 * axis_c - cc
                    if 0 <= nc < C and out[cr][nc] == bg:
                        out[cr][nc] = color
            else:
                ar = sum(axis_rs) / len(axis_rs)
                ac = sum(axis_cs) / len(axis_cs)
                for cr, cc in colored_cells:
                    color = non_bg[(cr, cc)]
                    nr, nc = round(2 * ar - cr), round(2 * ac - cc)
                    if 0 <= nr < R and 0 <= nc < C and out[nr][nc] == bg:
                        out[nr][nc] = color
    return out


# ============================================================
# TASK: 80a900e0
# Checkerboard with diamond of 3s and special color lines.
# Each line of special colors extends perpendicular from endpoints.
# ============================================================
def solve_80a900e0(grid):
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    bg_colors = {0, 1}
    diamond_color = 3

    threes = {(r, c) for r in range(R) for c in range(C) if grid[r][c] == diamond_color}
    specials = {}
    for r in range(R):
        for c in range(C):
            v = grid[r][c]
            if v not in bg_colors and v != diamond_color:
                specials.setdefault(v, []).append((r, c))

    if not threes:
        return out

    rs3 = [r for r, c in threes]
    cs3 = [c for r, c in threes]
    cr = (min(rs3) + max(rs3)) / 2
    cc = (min(cs3) + max(cs3)) / 2

    for color, cells in specials.items():
        if len(cells) < 1:
            continue

        cell_set = set(cells)
        visited = set()
        comps = []
        for start in cells:
            if start in visited:
                continue
            comp = []
            q = [start]
            while q:
                cur = q.pop(0)
                if cur in visited:
                    continue
                visited.add(cur)
                comp.append(cur)
                r, c = cur
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        if (r + dr, c + dc) in cell_set and (r + dr, c + dc) not in visited:
                            q.append((r + dr, c + dc))
            comps.append(sorted(comp))

        for comp in comps:
            if len(comp) < 2:
                sr, sc = comp[0]
                dr = 1 if sr > cr else -1
                dc = 1 if sc > cc else -1
                r, c = sr + dr, sc + dc
                while 0 <= r < R and 0 <= c < C:
                    if grid[r][c] in bg_colors:
                        out[r][c] = color
                    r += dr
                    c += dc
                continue

            dr_line = comp[1][0] - comp[0][0]
            dc_line = comp[1][1] - comp[0][1]
            g = max(abs(dr_line), abs(dc_line))
            if g > 0:
                dr_line //= g
                dc_line //= g

            perp1 = (-dc_line, dr_line)
            perp2 = (dc_line, -dr_line)

            first = comp[0]
            chosen_perp = perp1
            for perp in [perp1, perp2]:
                pr, pc = perp
                nr, nc = first[0] + pr, first[1] + pc
                dist_before = abs(first[0] - cr) + abs(first[1] - cc)
                dist_after = abs(nr - cr) + abs(nc - cc)
                if dist_after > dist_before:
                    chosen_perp = perp
                    break

            endpoints = [comp[0], comp[-1]]
            for ep in endpoints:
                er, ec = ep
                pr, pc = chosen_perp
                r, c = er + pr, ec + pc
                while 0 <= r < R and 0 <= c < C:
                    if grid[r][c] in bg_colors:
                        out[r][c] = color
                    r += pr
                    c += pc

    return out


# ============================================================
# MAIN
# ============================================================
def main():
    solutions = {}

    solvers = {
        "aa4ec2a5": solve_aa4ec2a5,
        "8f3a5a89": solve_8f3a5a89,
        "80a900e0": solve_80a900e0,
        "7ed72f31": solve_7ed72f31,
    }

    for tid in TASK_IDS:
        task = load_task(tid)

        if tid in solvers:
            solver = solvers[tid]
            if test_solver(task, solver):
                print(f"[PASS] {tid}")
                results = apply_solver(task, solver)
                for i, result in enumerate(results):
                    solutions[f"{tid}_{i}"] = result
            else:
                print(f"[FAIL] {tid}")
        else:
            print(f"[SKIP] {tid}")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(solutions, f)

    print(f"\nSaved {len(solutions)} solutions to {OUT_FILE}")


if __name__ == "__main__":
    main()
