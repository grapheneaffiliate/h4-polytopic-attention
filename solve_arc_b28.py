import json

solutions = {}

# 56dc2b01: Move shape next to 2-line, add 8-line on other side
solutions["56dc2b01"] = """def solve(grid):
    import copy
    R = len(grid)
    C = len(grid[0])
    two_cells = [(r,c) for r in range(R) for c in range(C) if grid[r][c] == 2]
    three_cells = [(r,c) for r in range(R) for c in range(C) if grid[r][c] == 3]
    two_rows = set(r for r,c in two_cells)
    two_cols = set(c for r,c in two_cells)
    out = [[0]*C for _ in range(R)]
    for r,c in two_cells:
        out[r][c] = 2
    if len(two_rows) == 1:
        line_row = list(two_rows)[0]
        shape_rows = set(r for r,c in three_cells)
        min_r = min(r for r,c in three_cells)
        max_r = max(r for r,c in three_cells)
        shape_h = max_r - min_r + 1
        if all(r < line_row for r in shape_rows):
            new_start_r = line_row - shape_h
            eight_row = new_start_r - 1
        else:
            new_start_r = line_row + 1
            eight_row = new_start_r + shape_h
        for r,c in three_cells:
            new_r = r - min_r + new_start_r
            out[new_r][c] = 3
        for c in range(C):
            if 0 <= eight_row < R:
                out[eight_row][c] = 8
    else:
        line_col = list(two_cols)[0]
        shape_cols = set(c for r,c in three_cells)
        min_c = min(c for r,c in three_cells)
        max_c = max(c for r,c in three_cells)
        shape_w = max_c - min_c + 1
        if all(c < line_col for c in shape_cols):
            new_start_c = line_col - shape_w
            eight_col = new_start_c - 1
        else:
            new_start_c = line_col + 1
            eight_col = new_start_c + shape_w
        for r,c in three_cells:
            new_c = c - min_c + new_start_c
            out[r][new_c] = 3
        for r in range(R):
            if 0 <= eight_col < C:
                out[r][eight_col] = 8
    return out"""

# 57aa92db: Template-based cross pattern application
solutions["57aa92db"] = """def solve(grid):
    import copy
    R=len(grid); C=len(grid[0]); out=copy.deepcopy(grid)
    visited=set(); objects=[]
    for r in range(R):
        for c in range(C):
            if grid[r][c]!=0 and (r,c) not in visited:
                obj=[]
                stk=[(r,c)]; visited.add((r,c))
                while stk:
                    cr,cc=stk.pop(); obj.append((cr,cc,grid[cr][cc]))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=cr+dr,cc+dc
                        if 0<=nr<R and 0<=nc<C and grid[nr][nc]!=0 and (nr,nc) not in visited:
                            visited.add((nr,nc)); stk.append((nr,nc))
                objects.append(obj)
    h2=[o for o in objects if any(v==2 for _,_,v in o)]
    n2=[o for o in objects if not any(v==2 for _,_,v in o)]
    if len(n2)==2:
        n2.sort(key=len); tobj=n2[0]; lobj=n2[1]
        tc={}; lc={}
        for r,c,v in tobj: tc.setdefault(v,[]).append((r,c))
        for r,c,v in lobj: lc.setdefault(v,[]).append((r,c))
        kc=(set(tc)&set(lc)).pop(); lo=(set(lc)-{kc}).pop()
        ta=[(r,c) for r,c,v in tobj]; tmr=min(r for r,c in ta); tmc=min(c for r,c in ta)
        txr=max(r for r,c in ta); txc=max(c for r,c in ta); th=txr-tmr+1; tw=txc-tmc+1
        tg=[[0]*tw for _ in range(th)]
        for r,c,v in tobj: tg[r-tmr][c-tmc]=v
        kp=None
        for tr in range(th):
            for tcc in range(tw):
                if tg[tr][tcc]==kc: kp=(tr,tcc); break
            if kp: break
        lkc=lc[kc]; lkmr=min(r for r,c in lkc); lkmc=min(c for r,c in lkc)
        lkxr=max(r for r,c in lkc); lkxc=max(c for r,c in lkc)
        bh=lkxr-lkmr+1; bw=lkxc-lkmc+1
        for tr in range(th):
            for tcc in range(tw):
                if tg[tr][tcc]==0: continue
                dr=tr-kp[0]; dc=tcc-kp[1]
                cl=kc if tg[tr][tcc]==kc else lo
                for rr in range(bh):
                    for cc in range(bw):
                        tr2=lkmr+dr*bh+rr; tc2=lkmc+dc*bw+cc
                        if 0<=tr2<R and 0<=tc2<C: out[tr2][tc2]=cl
    elif h2:
        best_ratio=-1; tobj=None
        for o in h2:
            ic={}
            for r,c,v in o: ic.setdefault(v,[]).append((r,c))
            n2c=sum(len(v) for k,v in ic.items() if k!=2)
            n2cnt=len(ic.get(2,[]))
            ratio=n2c/max(n2cnt,1)
            if ratio>best_ratio: best_ratio=ratio; tobj=o
        iobjs=[o for o in h2 if o is not tobj]
        tc={}
        for r,c,v in tobj: tc.setdefault(v,[]).append((r,c))
        t2=tc[2]; t2r,t2c=t2[0]; toc=(set(tc)-{2}).pop()
        rp=[(r-t2r,c-t2c) for r,c in tc[toc]]
        for io in iobjs:
            ic={}
            for r,c,v in io: ic.setdefault(v,[]).append((r,c))
            if 2 not in ic: continue
            i2=ic[2]; ioc=(set(ic)-{2}).pop()
            i2mr=min(r for r,c in i2); i2mc=min(c for r,c in i2)
            i2xr=max(r for r,c in i2); i2xc=max(c for r,c in i2)
            bh=i2xr-i2mr+1; bw=i2xc-i2mc+1
            if bh==1 and bw==1:
                for dr,dc in rp:
                    tr2=i2mr+dr; tc2=i2mc+dc
                    if 0<=tr2<R and 0<=tc2<C: out[tr2][tc2]=ioc
            else:
                for dr,dc in rp:
                    for rr in range(bh):
                        for cc in range(bw):
                            tr2=i2mr+dr*bh+rr; tc2=i2mc+dc*bw+cc
                            if 0<=tr2<R and 0<=tc2<C: out[tr2][tc2]=ioc
    return out"""

# 5ad4f10b: Find structured blocks, output which blocks are filled using noise color
solutions["5ad4f10b"] = """def solve(grid):
    R = len(grid)
    C = len(grid[0])
    colors = {}
    for r in range(R):
        for c in range(C):
            v = grid[r][c]
            if v != 0:
                colors.setdefault(v, []).append((r,c))
    for struct_color in colors:
        cells = set(colors[struct_color])
        other_colors = [c for c in colors if c != struct_color]
        if not other_colors: continue
        noise_color = other_colors[0]
        min_r = min(r for r,c in cells)
        max_r = max(r for r,c in cells)
        min_c = min(c for r,c in cells)
        max_c = max(c for r,c in cells)
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        oR, oC = 3, 3
        if h % oR != 0 or w % oC != 0: continue
        bh = h // oR
        bw = w // oC
        valid = True
        out = [[0]*oC for _ in range(oR)]
        for br in range(oR):
            for bc in range(oC):
                rs = min_r + br*bh
                cs = min_c + bc*bw
                block_cells = set()
                for r in range(rs, rs+bh):
                    for c in range(cs, cs+bw):
                        if (r,c) in cells:
                            block_cells.add((r,c))
                if len(block_cells) == bh * bw:
                    out[br][bc] = noise_color
                elif len(block_cells) == 0:
                    out[br][bc] = 0
                else:
                    valid = False; break
            if not valid: break
        if valid: return out
    return grid"""

# 6455b5f5: Largest 0-region -> 1, smallest 0-region -> 8
solutions["6455b5f5"] = """def solve(grid):
    import copy
    R = len(grid)
    C = len(grid[0])
    out = copy.deepcopy(grid)
    visited = [[False]*C for _ in range(R)]
    components = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 0 and not visited[r][c]:
                comp = []
                stack = [(r,c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc = cr+dr,cc+dc
                        if 0<=nr<R and 0<=nc<C and not visited[nr][nc] and grid[nr][nc] == 0:
                            visited[nr][nc] = True
                            stack.append((nr,nc))
                components.append(comp)
    if not components: return out
    sizes = [len(c) for c in components]
    max_size = max(sizes)
    min_size = min(sizes)
    for comp in components:
        if len(comp) == max_size:
            for r,c in comp: out[r][c] = 1
        elif len(comp) == min_size:
            for r,c in comp: out[r][c] = 8
    return out"""

# 673ef223: Two 2-segments on edges, 8-markers define cross pattern
solutions["673ef223"] = """def solve(grid):
    import copy
    R = len(grid); C = len(grid[0]); out = copy.deepcopy(grid)
    twos = [(r,c) for r in range(R) for c in range(C) if grid[r][c] == 2]
    eights = [(r,c) for r in range(R) for c in range(C) if grid[r][c] == 8]
    two_set = set(twos); visited = set(); groups = []
    for r,c in twos:
        if (r,c) not in visited:
            group = []
            stack = [(r,c)]; visited.add((r,c))
            while stack:
                cr,cc = stack.pop(); group.append((cr,cc))
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc = cr+dr,cc+dc
                    if (nr,nc) in two_set and (nr,nc) not in visited:
                        visited.add((nr,nc)); stack.append((nr,nc))
            groups.append(group)
    gwm = None; gwom = None
    for gi, group in enumerate(groups):
        group_rows = set(r for r,c in group)
        markers = [(r,c) for r,c in eights if r in group_rows]
        if markers: gwm = (gi, group, group[0][1])
        else: gwom = (gi, group, group[0][1])
    if gwm is None or gwom is None: return out
    gi1, g1, g1_col = gwm
    gi2, g2, g2_col = gwom
    g1_rows = sorted(set(r for r,c in g1))
    g2_rows = sorted(set(r for r,c in g2))
    g1_min_row = min(g1_rows); g2_min_row = min(g2_rows)
    for er, ec in eights:
        if er in set(g1_rows):
            out[er][ec] = 4
            if g1_col == 0:
                for c in range(1, ec): out[er][c] = 8
            else:
                for c in range(ec+1, g1_col): out[er][c] = 8
    marker_offsets = sorted([er - g1_min_row for er, ec in eights if er in set(g1_rows)])
    for offset in marker_offsets:
        target_row = g2_min_row + offset
        if g2_col == 0:
            for c in range(1, C): out[target_row][c] = 8
        else:
            for c in range(0, g2_col): out[target_row][c] = 8
    return out"""

# 6b9890af: Fill rectangle with scaled shape pattern
solutions["6b9890af"] = """def solve(grid):
    R = len(grid); C = len(grid[0])
    twos = [(r,c) for r in range(R) for c in range(C) if grid[r][c] == 2]
    rect_min_r = min(r for r,c in twos); rect_max_r = max(r for r,c in twos)
    rect_min_c = min(c for r,c in twos); rect_max_c = max(c for r,c in twos)
    int_h = rect_max_r - rect_min_r - 1; int_w = rect_max_c - rect_min_c - 1
    shape_cells = [(r,c,grid[r][c]) for r in range(R) for c in range(C) if grid[r][c] not in (0, 2)]
    if not shape_cells:
        return [grid[r][rect_min_c:rect_max_c+1] for r in range(rect_min_r, rect_max_r+1)]
    sh_min_r = min(r for r,c,v in shape_cells); sh_max_r = max(r for r,c,v in shape_cells)
    sh_min_c = min(c for r,c,v in shape_cells); sh_max_c = max(c for r,c,v in shape_cells)
    sh_h = sh_max_r - sh_min_r + 1; sh_w = sh_max_c - sh_min_c + 1
    shape = [[0]*sh_w for _ in range(sh_h)]
    for r,c,v in shape_cells: shape[r-sh_min_r][c-sh_min_c] = v
    scale_r = int_h // sh_h; scale_c = int_w // sh_w
    out_h = rect_max_r - rect_min_r + 1; out_w = rect_max_c - rect_min_c + 1
    out = [[0]*out_w for _ in range(out_h)]
    for r in range(out_h):
        for c in range(out_w): out[r][c] = grid[rect_min_r+r][rect_min_c+c]
    for sr in range(sh_h):
        for sc in range(sh_w):
            val = shape[sr][sc]
            for dr in range(scale_r):
                for dc in range(scale_c):
                    out[1 + sr*scale_r + dr][1 + sc*scale_c + dc] = val
    return out"""

# Verify all solutions
passing = {}
for task_id, code in solutions.items():
    with open(f"data/arc1/{task_id}.json") as f:
        data = json.load(f)
    exec(code)
    all_pass = True
    for i, pair in enumerate(data["train"]):
        result = solve(pair["input"])
        if result != pair["output"]:
            print(f"FAIL: {task_id} train {i}")
            all_pass = False
    if all_pass:
        print(f"PASS: {task_id}")
        passing[task_id] = code

# Save passing solutions
with open("data/arc_python_solutions_b28.json", "w") as f:
    json.dump(passing, f, indent=2)
print(f"\nSaved {len(passing)} solutions")
