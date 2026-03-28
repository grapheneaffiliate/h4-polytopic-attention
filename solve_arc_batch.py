import json

tasks = {}
task_ids = ["42a50994","4347f46a","444801d8","445eab21","447fd412","44d8ac46","44f52bb0","4522001f","4612dd53","46442a0e"]
for tid in task_ids:
    with open(f"data/arc1/{tid}.json") as f:
        tasks[tid] = json.load(f)

# 42a50994: Remove isolated pixels (8-connectivity), keep components with 2+ cells
def solve_42a50994(grid):
    h, w = len(grid), len(grid[0])
    visited = [[False]*w for _ in range(h)]
    components = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 and not visited[r][c]:
                comp = [(r,c)]
                visited[r][c] = True
                queue = [(r,c)]
                while queue:
                    cr, cc = queue.pop(0)
                    for dr in [-1,0,1]:
                        for dc in [-1,0,1]:
                            if dr==0 and dc==0: continue
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and grid[nr][nc] != 0:
                                visited[nr][nc] = True
                                comp.append((nr,nc))
                                queue.append((nr,nc))
                components.append(comp)
    out = [[0]*w for _ in range(h)]
    for comp in components:
        if len(comp) >= 2:
            for r,c in comp:
                out[r][c] = grid[r][c]
    return out

# 4347f46a: Solid rectangles become hollow (keep border, clear interior)
def solve_4347f46a(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    visited = [[False]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                comp = [(r,c)]
                visited[r][c] = True
                queue = [(r,c)]
                while queue:
                    cr,cc = queue.pop(0)
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc = cr+dr, cc+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and grid[nr][nc]==color:
                            visited[nr][nc] = True
                            comp.append((nr,nc))
                            queue.append((nr,nc))
                minr = min(r for r,c in comp)
                maxr = max(r for r,c in comp)
                minc = min(c for r,c in comp)
                maxc = max(c for r,c in comp)
                for cr,cc in comp:
                    if cr > minr and cr < maxr and cc > minc and cc < maxc:
                        out[cr][cc] = 0
    return out

# 444801d8: Box with gap, colored dot inside -> fill interior + gap + extend through gap
def solve_444801d8(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    visited = [[False]*w for _ in range(h)]
    boxes = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 1 and not visited[r][c]:
                comp = set()
                queue = [(r,c)]
                visited[r][c] = True
                while queue:
                    cr,cc = queue.pop(0)
                    comp.add((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc = cr+dr,cc+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and grid[nr][nc]==1:
                            visited[nr][nc] = True
                            queue.append((nr,nc))
                boxes.append(comp)
    for box in boxes:
        minr = min(r for r,c in box)
        maxr = max(r for r,c in box)
        minc = min(c for r,c in box)
        maxc = max(c for r,c in box)
        color = None
        for r in range(minr, maxr+1):
            for c in range(minc, maxc+1):
                if grid[r][c] not in (0, 1):
                    color = grid[r][c]
                    break
            if color: break
        if not color: continue
        top_gap = any((minr, c) not in box for c in range(minc, maxc+1))
        bot_gap = any((maxr, c) not in box for c in range(minc, maxc+1))
        left_gap = any((r, minc) not in box for r in range(minr, maxr+1))
        right_gap = any((r, maxc) not in box for r in range(minr, maxr+1))
        # Fill interior
        for r in range(minr+1, maxr):
            for c in range(minc+1, maxc):
                if grid[r][c] != 1:
                    out[r][c] = color
        # Fill gap cells on border AND extend one row/col outside
        if top_gap:
            for c in range(minc, maxc+1):
                if (minr, c) not in box:
                    out[minr][c] = color
            if minr > 0:
                for c in range(minc, maxc+1):
                    out[minr-1][c] = color
        if bot_gap:
            for c in range(minc, maxc+1):
                if (maxr, c) not in box:
                    out[maxr][c] = color
            if maxr < h-1:
                for c in range(minc, maxc+1):
                    out[maxr+1][c] = color
        if left_gap:
            for r in range(minr, maxr+1):
                if (r, minc) not in box:
                    out[r][minc] = color
            if minc > 0:
                for r in range(minr, maxr+1):
                    out[r][minc-1] = color
        if right_gap:
            for r in range(minr, maxr+1):
                if (r, maxc) not in box:
                    out[r][maxc] = color
            if maxc < w-1:
                for r in range(minr, maxr+1):
                    out[r][maxc+1] = color
    return out

# 445eab21: Two hollow rectangles -> output 2x2 filled with color of larger interior
def solve_445eab21(grid):
    h, w = len(grid), len(grid[0])
    visited = [[False]*w for _ in range(h)]
    rects = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                comp = set()
                queue = [(r,c)]
                visited[r][c] = True
                while queue:
                    cr,cc = queue.pop(0)
                    comp.add((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc = cr+dr,cc+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and grid[nr][nc]==color:
                            visited[nr][nc] = True
                            queue.append((nr,nc))
                minr = min(r for r,c in comp)
                maxr = max(r for r,c in comp)
                minc = min(c for r,c in comp)
                maxc = max(c for r,c in comp)
                interior_area = (maxr - minr - 1) * (maxc - minc - 1)
                rects.append((color, interior_area))
    rects.sort(key=lambda x: -x[1])
    c = rects[0][0]
    return [[c, c], [c, c]]

# 447fd412: Template pattern with 1s and 2s, scale and place at 2-marker blocks
def solve_447fd412(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    visited = [[False]*w for _ in range(h)]
    components = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 and not visited[r][c]:
                comp = []
                queue = [(r,c)]
                visited[r][c] = True
                while queue:
                    cr,cc = queue.pop(0)
                    comp.append((cr,cc, grid[cr][cc]))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc = cr+dr,cc+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and grid[nr][nc]!=0:
                            visited[nr][nc] = True
                            queue.append((nr,nc))
                components.append(comp)
    template_comp = None
    target_comps = []
    for comp in components:
        vals = set(v for _,_,v in comp)
        if 1 in vals and 2 in vals:
            template_comp = comp
        elif vals == {2}:
            target_comps.append(comp)
    if template_comp is None:
        return out
    t1_cells = [(r,c) for r,c,v in template_comp if v == 1]
    t2_cells = [(r,c) for r,c,v in template_comp if v == 2]
    target_blocks = []
    for comp in target_comps:
        rs = [r for r,c,v in comp]
        cs = [c for r,c,v in comp]
        minr,maxr,minc,maxc = min(rs),max(rs),min(cs),max(cs)
        bh = maxr - minr + 1
        bw = maxc - minc + 1
        target_blocks.append((minr,maxr,minc,maxc,bh,bw))
    if len(t2_cells) == 1:
        t2r, t2c = t2_cells[0]
        for blk in target_blocks:
            minr,maxr,minc,maxc,bh,bw = blk
            scale = max(bh, bw)
            for r1, c1 in t1_cells:
                dr = r1 - t2r
                dc = c1 - t2c
                for rr in range(scale):
                    for cc in range(scale):
                        nr = minr + dr * scale + rr
                        nc = minc + dc * scale + cc
                        if 0 <= nr < h and 0 <= nc < w and out[nr][nc] == 0:
                            out[nr][nc] = 1
    elif len(t2_cells) == 2:
        t2a, t2b = t2_cells[0], t2_cells[1]
        t_dr = t2b[0] - t2a[0]
        t_dc = t2b[1] - t2a[1]
        used = [False] * len(target_blocks)
        for i in range(len(target_blocks)):
            if used[i]: continue
            for j in range(i+1, len(target_blocks)):
                if used[j]: continue
                bi = target_blocks[i]
                bj = target_blocks[j]
                for ba, bb in [(bi, bj), (bj, bi)]:
                    scale = max(ba[4], ba[5])
                    exp_r = ba[0] + t_dr * scale
                    exp_c = ba[2] + t_dc * scale
                    if exp_r == bb[0] and exp_c == bb[2]:
                        used[i] = True
                        used[j] = True
                        for r1, c1 in t1_cells:
                            dr = r1 - t2a[0]
                            dc = c1 - t2a[1]
                            for rr in range(scale):
                                for cc in range(scale):
                                    nr = ba[0] + dr * scale + rr
                                    nc = ba[2] + dc * scale + cc
                                    if 0 <= nr < h and 0 <= nc < w and out[nr][nc] == 0:
                                        out[nr][nc] = 1
                        break
                if used[i]:
                    break
    return out

# 44d8ac46: Fill interior of 5-rectangles with 2 only if interior is square
def solve_44d8ac46(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    visited = [[False]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 5 and not visited[r][c]:
                comp = set()
                queue = [(r,c)]
                visited[r][c] = True
                while queue:
                    cr,cc = queue.pop(0)
                    comp.add((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc = cr+dr,cc+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and grid[nr][nc]==5:
                            visited[nr][nc] = True
                            queue.append((nr,nc))
                minr = min(r for r,c in comp)
                maxr = max(r for r,c in comp)
                minc = min(c for r,c in comp)
                maxc = max(c for r,c in comp)
                interior_zeros = []
                for r2 in range(minr+1, maxr):
                    for c2 in range(minc+1, maxc):
                        if grid[r2][c2] == 0:
                            interior_zeros.append((r2,c2))
                if not interior_zeros:
                    continue
                zr = [r for r,c in interior_zeros]
                zc = [c for r,c in interior_zeros]
                zminr, zmaxr = min(zr), max(zr)
                zminc, zmaxc = min(zc), max(zc)
                zh = zmaxr - zminr + 1
                zw = zmaxc - zminc + 1
                if len(interior_zeros) != zh * zw:
                    continue
                if zh != zw:
                    continue
                for r2,c2 in interior_zeros:
                    out[r2][c2] = 2
    return out

# 44f52bb0: 3x3 grid -> 1x1: left-right symmetric -> 1, else -> 7
def solve_44f52bb0(grid):
    h, w = len(grid), len(grid[0])
    symmetric = all(grid[r][c] == grid[r][w-1-c] for r in range(h) for c in range(w))
    return [[1]] if symmetric else [[7]]

# 4522001f: 3x3 with 2x2 non-zero block -> 9x9 with two 4x4 blocks of 3
def solve_4522001f(grid):
    nz = set()
    for r in range(3):
        for c in range(3):
            if grid[r][c] != 0:
                nz.add((r,c))
    block_r, block_c = 0, 0
    for r0 in range(2):
        for c0 in range(2):
            if all((r0+dr, c0+dc) in nz for dr in range(2) for dc in range(2)):
                block_r, block_c = r0, c0
    out = [[0]*9 for _ in range(9)]
    if block_r == 0 and block_c == 0:
        blocks = [(0,0), (4,4)]
    elif block_r == 0 and block_c == 1:
        blocks = [(0,5), (4,1)]
    elif block_r == 1 and block_c == 0:
        blocks = [(5,0), (1,4)]
    else:
        blocks = [(1,1), (5,5)]
    for br, bc in blocks:
        for dr in range(4):
            for dc in range(4):
                if 0 <= br+dr < 9 and 0 <= bc+dc < 9:
                    out[br+dr][bc+dc] = 3
    return out

# 4612dd53: Rectangle frame of 1s with gaps -> fill gaps with 2
def solve_4612dd53(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    ones = [(r,c) for r in range(h) for c in range(w) if grid[r][c] == 1]
    if not ones:
        return out
    minr = min(r for r,c in ones)
    maxr = max(r for r,c in ones)
    minc = min(c for r,c in ones)
    maxc = max(c for r,c in ones)

    def has_consecutive(seq):
        for i in range(len(seq)-1):
            if seq[i]==1 and seq[i+1]==1:
                return True
        return False

    wall_rows = set()
    for r in range(minr, maxr+1):
        row = [grid[r][c] for c in range(minc, maxc+1)]
        if has_consecutive(row):
            wall_rows.add(r)
    wall_cols = set()
    for c in range(minc, maxc+1):
        col = [grid[r][c] for r in range(minr, maxr+1)]
        if has_consecutive(col):
            wall_cols.add(c)

    wc_min = min(wall_cols)
    wc_max = max(wall_cols)
    wr_min = min(wall_rows)
    wr_max = max(wall_rows)

    for r in wall_rows:
        for c in range(wc_min, wc_max+1):
            if out[r][c] == 0:
                out[r][c] = 2
    for c in wall_cols:
        for r in range(wr_min, wr_max+1):
            if out[r][c] == 0:
                out[r][c] = 2
    return out

# 46442a0e: TL=original, TR=rot90CW, BL=rot90CCW, BR=rot180
def solve_46442a0e(grid):
    n = len(grid)
    m = len(grid[0])
    def rot90cw(g):
        return [[g[n-1-j][i] for j in range(n)] for i in range(m)]
    def rot90ccw(g):
        return [[g[j][m-1-i] for j in range(n)] for i in range(m)]
    def rot180(g):
        return [[g[n-1-r][m-1-c] for c in range(m)] for r in range(n)]
    tr = rot90cw(grid)
    bl = rot90ccw(grid)
    br = rot180(grid)
    out = [[0]*(2*m) for _ in range(2*n)]
    for r in range(n):
        for c in range(m):
            out[r][c] = grid[r][c]
            out[r][m+c] = tr[r][c]
            out[n+r][c] = bl[r][c]
            out[n+r][m+c] = br[r][c]
    return out

# Verify all
solve_map = {
    "42a50994": solve_42a50994,
    "4347f46a": solve_4347f46a,
    "444801d8": solve_444801d8,
    "445eab21": solve_445eab21,
    "447fd412": solve_447fd412,
    "44d8ac46": solve_44d8ac46,
    "44f52bb0": solve_44f52bb0,
    "4522001f": solve_4522001f,
    "4612dd53": solve_4612dd53,
    "46442a0e": solve_46442a0e,
}

all_results = {}
for tid in task_ids:
    d = tasks[tid]
    fn = solve_map[tid]
    print(f"=== {tid} ===")
    ok = True
    for i, p in enumerate(d['train']):
        result = fn(p['input'])
        if result == p['output']:
            print(f"  Train {i}: OK")
        else:
            print(f"  Train {i}: FAIL")
            for r in range(min(len(result), len(p['output']))):
                if result[r] != p['output'][r]:
                    print(f"    Row {r}: got  {result[r]}")
                    print(f"    Row {r}: want {p['output'][r]}")
            ok = False
    for i, p in enumerate(d['test']):
        if 'output' in p:
            result = fn(p['input'])
            if result == p['output']:
                print(f"  Test {i}: OK")
            else:
                print(f"  Test {i}: FAIL")
                for r in range(min(len(result), len(p['output']))):
                    if result[r] != p['output'][r]:
                        print(f"    Row {r}: got  {result[r]}")
                        print(f"    Row {r}: want {p['output'][r]}")
                ok = False
    all_results[tid] = ok

print("\n=== SUMMARY ===")
for tid in task_ids:
    print(f"{tid}: {'PASS' if all_results[tid] else 'FAIL'}")

# Save solutions to JSON
import inspect
output = {}
for tid, fn in solve_map.items():
    output[tid] = inspect.getsource(fn)

with open("data/arc_python_solutions_b6.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nSaved to data/arc_python_solutions_b6.json")
