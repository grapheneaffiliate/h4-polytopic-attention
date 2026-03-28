#!/usr/bin/env python3
"""ARC-AGI-2 retry3 solver for hard evaluation tasks."""

import json
import os
from collections import Counter, defaultdict

DATA_DIR = "data/arc2"
OUTPUT_FILE = "data/arc2_solutions_retry3.json"

TASK_IDS = [
    "b0039139","b10624e5","b5ca7ac4","b6f77b65","b99e7126","b9e38dc0",
    "bf45cf4b","c7f57c3e","cb2d8a2c","cbebaa4b","d35bdbdc","d59b0160",
    "d8e07eb2","da515329","db0c5428","dd6b8c4b","de809cff","dfadab01",
    "e12f9a14","e3721c99","e8686506","e87109e9","edb79dae","eee78d87",
    "f560132c","f931b4a8"
]

def load_task(task_id):
    with open(os.path.join(DATA_DIR, f"{task_id}.json")) as f:
        return json.load(f)

def grids_equal(a, b):
    if len(a) != len(b): return False
    for i in range(len(a)):
        if len(a[i]) != len(b[i]): return False
        for j in range(len(a[i])):
            if a[i][j] != b[i][j]: return False
    return True

def test_solver(task, solver):
    for pair in task["train"]:
        try:
            result = solver(pair["input"])
            if not grids_equal(result, pair["output"]): return False
        except Exception: return False
    return True

def apply_solver(task, solver):
    results = []
    for test in task["test"]:
        try: results.append(solver(test["input"]))
        except Exception: return None
    return results


# ============================================================
# bf45cf4b: Stamp tiling
# ============================================================
def solve_bf45cf4b(grid):
    H, W = len(grid), len(grid[0])
    vals = Counter()
    for r in grid: vals.update(r)
    bg = vals.most_common(1)[0][0]

    non_bg = {}
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg: non_bg[(r,c)] = grid[r][c]

    visited = set()
    components = []
    for (r,c) in non_bg:
        if (r,c) in visited: continue
        queue = [(r,c)]
        comp = []
        while queue:
            cr,cc = queue.pop(0)
            if (cr,cc) in visited or (cr,cc) not in non_bg: continue
            visited.add((cr,cc))
            comp.append((cr,cc))
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    if dr==0 and dc==0: continue
                    if (cr+dr,cc+dc) in non_bg and (cr+dr,cc+dc) not in visited:
                        queue.append((cr+dr,cc+dc))
        components.append(comp)

    stamp_comp = shape_comp = None
    for comp in components:
        colors = set(non_bg[p] for p in comp)
        if len(colors) >= 2:
            if stamp_comp is None: stamp_comp = comp
        else:
            if shape_comp is None: shape_comp = comp

    sr0 = min(r for r,c in stamp_comp)
    sc0 = min(c for r,c in stamp_comp)
    stamp_h = max(r for r,c in stamp_comp) - sr0 + 1
    stamp_w = max(c for r,c in stamp_comp) - sc0 + 1
    stamp = [[bg]*stamp_w for _ in range(stamp_h)]
    for r,c in stamp_comp: stamp[r-sr0][c-sc0] = non_bg[(r,c)]

    shr0 = min(r for r,c in shape_comp)
    shc0 = min(c for r,c in shape_comp)
    shape_h = max(r for r,c in shape_comp) - shr0 + 1
    shape_w = max(c for r,c in shape_comp) - shc0 + 1
    shape_color = non_bg[shape_comp[0]]
    shape = [[bg]*shape_w for _ in range(shape_h)]
    for r,c in shape_comp: shape[r-shr0][c-shc0] = non_bg[(r,c)]

    out = [[bg]*(stamp_w*shape_w) for _ in range(stamp_h*shape_h)]
    for sr in range(shape_h):
        for sc in range(shape_w):
            if shape[sr][sc] == shape_color:
                for dr in range(stamp_h):
                    for dc in range(stamp_w):
                        out[sr*stamp_h+dr][sc*stamp_w+dc] = stamp[dr][dc]
    return out


# ============================================================
# d59b0160: Remove patches containing ALL L-corner colors
# ============================================================
def solve_d59b0160(grid):
    H, W = len(grid), len(grid[0])
    bg = 7

    cross_row = cross_col = None
    for r in range(H):
        tc = [c for c in range(W) if grid[r][c] == 3]
        if len(tc) >= 3 and tc == list(range(tc[0], tc[-1]+1)):
            cross_row = r; break
    for c in range(W):
        tr = [r for r in range(H) if grid[r][c] == 3]
        if len(tr) >= 3 and tr == list(range(tr[0], tr[-1]+1)):
            cross_col = c; break

    corner_colors = set()
    corner_cells = set()
    for r in range(cross_row + 1):
        for c in range(cross_col + 1):
            if grid[r][c] != bg:
                corner_cells.add((r,c))
                if grid[r][c] != 3: corner_colors.add(grid[r][c])

    non_bg = set()
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and (r,c) not in corner_cells:
                non_bg.add((r,c))

    visited = set()
    components = []
    for (r,c) in non_bg:
        if (r,c) in visited: continue
        queue = [(r,c)]
        comp = []
        while queue:
            cr,cc = queue.pop(0)
            if (cr,cc) in visited or (cr,cc) not in non_bg: continue
            visited.add((cr,cc))
            comp.append((cr,cc))
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                if (cr+dr,cc+dc) in non_bg and (cr+dr,cc+dc) not in visited:
                    queue.append((cr+dr,cc+dc))
        components.append(comp)

    out = [row[:] for row in grid]
    for comp in components:
        cc = set(grid[r][c] for r,c in comp if grid[r][c] != 0)
        if corner_colors.issubset(cc):
            for r,c in comp: out[r][c] = bg
    return out


# ============================================================
# d35bdbdc: 3x3 blocks with frame+center, recursive chain pairing
# ============================================================
def solve_d35bdbdc(grid):
    H, W = len(grid), len(grid[0])

    blocks = []
    for r in range(H - 2):
        for c in range(W - 2):
            border = []
            for dr in range(3):
                for dc in range(3):
                    if dr == 1 and dc == 1: continue
                    border.append(grid[r+dr][c+dc])
            center = grid[r+1][c+1]
            if center == 0 or center == 5: continue
            if len(set(border)) == 1 and border[0] != 0 and border[0] != 5 and border[0] != center:
                blocks.append({'r':r,'c':c,'frame':border[0],'center':center,'cr':r+1,'cc':c+1})

    fives = set((r,c) for r in range(H) for c in range(W) if grid[r][c] == 5)

    def touches_five(b):
        for dr in range(3):
            for dc in range(3):
                rr, cc = b['r']+dr, b['c']+dc
                for nr, nc in [(rr-1,cc),(rr+1,cc),(rr,cc-1),(rr,cc+1)]:
                    if (nr, nc) in fives: return True
        return False

    for b in blocks: b['t5'] = touches_five(b)

    frame_map = {b['frame']: b for b in blocks}
    status = {}

    def get_status(idx, vis=None):
        if idx in status: return status[idx]
        if vis is None: vis = set()
        if idx in vis:
            status[idx] = 'removed'; return 'removed'
        vis = vis | {idx}
        b = blocks[idx]
        if not b['t5']:
            status[idx] = 'removed'; return 'removed'
        donor = frame_map.get(b['center'])
        if donor is None:
            status[idx] = 'removed'; return 'removed'
        didx = next(j for j,bb in enumerate(blocks) if bb is donor)
        ds = get_status(didx, vis)
        status[idx] = 'kept' if ds == 'removed' else 'removed'
        return status[idx]

    for i in range(len(blocks)): get_status(i)

    out = [row[:] for row in grid]
    for i, b in enumerate(blocks):
        if status[i] == 'kept':
            donor = frame_map[b['center']]
            out[b['cr']][b['cc']] = donor['center']
        else:
            for dr in range(3):
                for dc in range(3):
                    out[b['r']+dr][b['c']+dc] = 0
    return out


# Also try d35bdbdc WITHOUT touches_5 constraint on donor
def solve_d35bdbdc_v2(grid):
    H, W = len(grid), len(grid[0])
    blocks = []
    for r in range(H - 2):
        for c in range(W - 2):
            border = []
            for dr in range(3):
                for dc in range(3):
                    if dr == 1 and dc == 1: continue
                    border.append(grid[r+dr][c+dc])
            center = grid[r+1][c+1]
            if center == 0 or center == 5: continue
            if len(set(border)) == 1 and border[0] != 0 and border[0] != 5 and border[0] != center:
                blocks.append({'r':r,'c':c,'frame':border[0],'center':center,'cr':r+1,'cc':c+1})

    fives = set((r,c) for r in range(H) for c in range(W) if grid[r][c] == 5)

    def touches_five(b):
        for dr in range(3):
            for dc in range(3):
                rr, cc = b['r']+dr, b['c']+dc
                for nr, nc in [(rr-1,cc),(rr+1,cc),(rr,cc-1),(rr,cc+1)]:
                    if (nr, nc) in fives: return True
        return False

    for b in blocks: b['t5'] = touches_five(b)
    frame_map = {b['frame']: b for b in blocks}
    status = {}

    def get_status(idx, vis=None):
        if idx in status: return status[idx]
        if vis is None: vis = set()
        if idx in vis:
            status[idx] = 'removed'; return 'removed'
        vis = vis | {idx}
        b = blocks[idx]
        if not b['t5']:
            status[idx] = 'removed'; return 'removed'
        donor = frame_map.get(b['center'])
        if donor is None:
            status[idx] = 'removed'; return 'removed'
        # No constraint on donor touching 5
        didx = next(j for j,bb in enumerate(blocks) if bb is donor)
        ds = get_status(didx, vis)
        status[idx] = 'kept' if ds == 'removed' else 'removed'
        return status[idx]

    for i in range(len(blocks)): get_status(i)

    out = [row[:] for row in grid]
    for i, b in enumerate(blocks):
        if status[i] == 'kept':
            donor = frame_map[b['center']]
            out[b['cr']][b['cc']] = donor['center']
        else:
            for dr in range(3):
                for dc in range(3):
                    out[b['r']+dr][b['c']+dc] = 0
    return out


# Placeholder solvers
def _nope(grid): raise NotImplementedError
solve_b0039139 = _nope
solve_b10624e5 = _nope
solve_b5ca7ac4 = _nope
solve_b6f77b65 = _nope
solve_b99e7126 = _nope
solve_b9e38dc0 = _nope
solve_c7f57c3e = _nope
solve_cb2d8a2c = _nope
solve_cbebaa4b = _nope
solve_d8e07eb2 = _nope
solve_da515329 = _nope
solve_db0c5428 = _nope
solve_dd6b8c4b = _nope
solve_de809cff = _nope
solve_dfadab01 = _nope
solve_e12f9a14 = _nope
solve_e3721c99 = _nope
solve_e8686506 = _nope
solve_e87109e9 = _nope
solve_edb79dae = _nope
solve_eee78d87 = _nope
solve_f560132c = _nope
solve_f931b4a8 = _nope


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    solvers = {}
    for tid in TASK_IDS:
        fn = globals().get(f"solve_{tid}")
        if fn and fn is not _nope:
            solvers[tid] = [fn]

    # d35bdbdc has two variants to try
    solvers["d35bdbdc"] = [solve_d35bdbdc, solve_d35bdbdc_v2]

    # dfadab01 might use same pattern as d35bdbdc
    solvers["dfadab01"] = [solve_d35bdbdc, solve_d35bdbdc_v2]

    solutions = {}
    for task_id in TASK_IDS:
        solver_list = solvers.get(task_id, [])
        if not solver_list: continue

        try:
            task = load_task(task_id)
        except Exception as e:
            print(f"{task_id}: Failed to load: {e}")
            continue

        for solver in solver_list:
            try:
                if test_solver(task, solver):
                    test_outputs = apply_solver(task, solver)
                    if test_outputs:
                        solutions[task_id] = test_outputs
                        print(f"{task_id}: PASSED ({solver.__name__})")
                        break
            except Exception:
                pass
        else:
            print(f"{task_id}: no solver passed")

    os.makedirs(os.path.dirname(OUTPUT_FILE) if os.path.dirname(OUTPUT_FILE) else ".", exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(solutions, f)

    print(f"\nSaved {len(solutions)} solutions to {OUTPUT_FILE}")
