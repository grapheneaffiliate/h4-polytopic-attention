#!/usr/bin/env python3
"""ARC-AGI-2 solver for batch ae tasks."""
import json, os
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "arc2")
OUT_FILE = os.path.join(BASE, "data", "arc2_solutions_train_ae.json")

TASK_IDS = "ac0c5833,ac2e8ecf,ac3e2b04,ac605cbb,ac6f9922,ad173014,ad38a9d0,ad3b40cf,ad7e01d0,ae58858e,aee291af,af24b4cc,af726779,afe3afe9,b0722778,b0f4d537,b15fca0b,b1948b0a,b1986d4b,b1fc8b8e,b20f7c8b,b25e450b,b2bc3ffd,b457fec5,b4a43f3b,b5bb5719,b71a7747,b7256dcd,b745798f,b74ca5d1,b7955b3c,b7999b51,b7cb93ac,b7f8a4d8,b7fb29bc,b942fd60,b9630600,ba1aa698,ba9d41b8,bae5c565,baf41dbf,bb52a14b,bbb1b8b6,bc4146bd,bc93ec48,bcb3040b,bd14c3bf,bd283c4a,bd5af378,be03b35f,bf32578f,bf699163,bf89d739,c074846d".split(",")

def load_task(tid):
    with open(os.path.join(DATA, f"{tid}.json")) as f:
        return json.load(f)

def grid_eq(a, b):
    if len(a) != len(b): return False
    return all(r1 == r2 for r1, r2 in zip(a, b))

def test_solver(task, solver):
    for p in task["train"]:
        try:
            if not grid_eq(solver(p["input"]), p["output"]): return False
        except: return False
    return True

def apply_solver(task, solver):
    return [solver(p["input"]) for p in task["test"]]

def G(R, C, v=0): return [[v]*C for _ in range(R)]
def CG(g): return [r[:] for r in g]
def CC(g):
    c = Counter()
    for r in g:
        for v in r: c[v] += 1
    return c

def bbox(cells):
    rs = [r for r,c in cells]; cs = [c for r,c in cells]
    return min(rs), min(cs), max(rs), max(cs)

def extract(g, r1, c1, r2, c2):
    return [row[c1:c2+1] for row in g[r1:r2+1]]

def flood(g, r, c, vis=None):
    if vis is None: vis = set()
    color = g[r][c]; R, C = len(g), len(g[0])
    stack = [(r,c)]; cells = []
    while stack:
        cr,cc = stack.pop()
        if (cr,cc) in vis or cr<0 or cr>=R or cc<0 or cc>=C: continue
        if g[cr][cc] != color: continue
        vis.add((cr,cc)); cells.append((cr,cc))
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]: stack.append((cr+dr,cc+dc))
    return cells

def get_objs(g, bg=0):
    vis = set(); objs = []
    for r in range(len(g)):
        for c in range(len(g[0])):
            if g[r][c] != bg and (r,c) not in vis:
                objs.append(flood(g, r, c, vis))
    return objs

def find_pos(g, color):
    return [(r,c) for r in range(len(g)) for c in range(len(g[0])) if g[r][c] == color]

def rotate90(g):
    R, C = len(g), len(g[0])
    return [[g[R-1-c][r] for c in range(R)] for r in range(C)]
def rotate180(g): return rotate90(rotate90(g))
def rotate270(g): return rotate90(rotate90(rotate90(g)))
def flip_h(g): return [r[::-1] for r in g]
def flip_v(g): return g[::-1]
def transpose(g):
    R, C = len(g), len(g[0])
    return [[g[r][c] for r in range(R)] for c in range(C)]

def _sep_sections(grid, sc, ori):
    R, C = len(grid), len(grid[0])
    if ori in ('h','both'):
        sep_r = [r for r in range(R) if all(grid[r][c2]==sc for c2 in range(C))]
    else: sep_r = []
    if ori in ('v','both'):
        sep_c = [c for c in range(C) if all(grid[r2][c]==sc for r2 in range(R))]
    else: sep_c = []
    if not sep_r and not sep_c: return None
    rb = []; prev = 0
    for sr in sep_r:
        if sr > prev: rb.append((prev, sr))
        prev = sr + 1
    if prev < R: rb.append((prev, R))
    cb = []; prev = 0
    for sc2 in sep_c:
        if sc2 > prev: cb.append((prev, sc2))
        prev = sc2 + 1
    if prev < C: cb.append((prev, C))
    if not rb: rb = [(0, R)]
    if not cb: cb = [(0, C)]
    secs = []
    for r1,r2 in rb:
        for c1,c2 in cb:
            secs.append([row[c1:c2] for row in grid[r1:r2]])
    return secs, len(rb), len(cb)

# ============================================================
# STRATEGIES
# ============================================================

def S_basics(task):
    fns = [lambda g: CG(g), rotate90, rotate180, rotate270, flip_h, flip_v, transpose, lambda g: flip_h(rotate90(g))]
    for fn in fns:
        if test_solver(task, fn): return fn
    return None

def S_color_map(task):
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None
    cmap = {}
    for p in task["train"]:
        for r in range(len(p["input"])):
            for c in range(len(p["input"][0])):
                ic, oc = p["input"][r][c], p["output"][r][c]
                if ic in cmap and cmap[ic] != oc: return None
                cmap[ic] = oc
    def s(g): return [[cmap.get(c,c) for c in row] for row in g]
    if test_solver(task, s): return s
    return None

def S_swap_colors(task):
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None
    for a in range(10):
        for b in range(a+1, 10):
            def mk(a,b):
                def s(g): return [[b if v==a else (a if v==b else v) for v in row] for row in g]
                return s
            if test_solver(task, mk(a,b)): return mk(a,b)
    return None

def S_scale(task):
    t0 = task["train"][0]
    iR, iC = len(t0["input"]), len(t0["input"][0])
    oR, oC = len(t0["output"]), len(t0["output"][0])
    if oR % iR != 0 or oC % iC != 0: return None
    sR, sC = oR // iR, oC // iC
    for p in task["train"]:
        if len(p["output"]) != len(p["input"])*sR or len(p["output"][0]) != len(p["input"][0])*sC: return None
    def s(g):
        R, C = len(g), len(g[0])
        o = G(R*sR, C*sC)
        for r in range(R):
            for c in range(C):
                for dr in range(sR):
                    for dc in range(sC):
                        o[r*sR+dr][c*sC+dc] = g[r][c]
        return o
    if test_solver(task, s): return s
    return None

def S_ad7e01d0(task):
    """Most common non-zero color cells get a copy of full input."""
    t0 = task["train"][0]
    iR, iC = len(t0["input"]), len(t0["input"][0])
    oR, oC = len(t0["output"]), len(t0["output"][0])
    if oR != iR*iR or oC != iC*iC: return None
    def s(g):
        R, C = len(g), len(g[0])
        cnt = Counter(v for row in g for v in row if v != 0)
        mc = cnt.most_common(1)[0][0]
        o = G(R*R, C*C)
        for r in range(R):
            for c in range(C):
                if g[r][c] == mc:
                    for dr in range(R):
                        for dc in range(C):
                            o[r*R+dr][c*C+dc] = g[dr][dc]
        return o
    if test_solver(task, s): return s
    return None

def S_tile(task):
    t0 = task["train"][0]
    iR, iC = len(t0["input"]), len(t0["input"][0])
    oR, oC = len(t0["output"]), len(t0["output"][0])
    if oR < iR or oC < iC: return None
    if oR % iR != 0 or oC % iC != 0: return None
    tR, tC = oR // iR, oC // iC
    def s(g):
        R, C = len(g), len(g[0])
        o = G(R*tR, C*tC)
        for tr in range(tR):
            for tc in range(tC):
                for r in range(R):
                    for c in range(C):
                        o[tr*R+r][tc*C+c] = g[r][c]
        return o
    if test_solver(task, s): return s
    return None

def S_htile(task):
    t0 = task["train"][0]
    iR, iC = len(t0["input"]), len(t0["input"][0])
    oR, oC = len(t0["output"]), len(t0["output"][0])
    if oR != iR or oC % iC != 0: return None
    n = oC // iC
    def s1(g): return [row * n for row in g]
    if test_solver(task, s1): return s1
    # Alt with flip
    def s2(g):
        R, C = len(g), len(g[0])
        o = G(R, C*n)
        for i in range(n):
            cur = g if i%2==0 else flip_h(g)
            for r in range(R):
                for c in range(C):
                    o[r][i*C+c] = cur[r][c]
        return o
    if test_solver(task, s2): return s2
    # With row shift
    for sr in range(-3, 4):
        if sr == 0: continue
        def mk(sr2, n2):
            def s(g):
                R, C = len(g), len(g[0])
                o = G(R, C*n2)
                for i in range(n2):
                    for r in range(R):
                        for c in range(C):
                            nr = (r + sr2*i) % R
                            o[r][i*C+c] = g[nr][c]
                return o
            return s
        if test_solver(task, mk(sr, n)): return mk(sr, n)
    return None

def S_crop(task):
    def s(g):
        R, C = len(g), len(g[0])
        pos = [(r,c) for r in range(R) for c in range(C) if g[r][c] != 0]
        if not pos: return [[0]]
        r1,c1,r2,c2 = bbox(pos)
        return extract(g, r1, c1, r2, c2)
    if test_solver(task, s): return s
    return None

def S_crop_excl(task):
    for ec in range(1, 10):
        def mk(ec):
            def s(g):
                R, C = len(g), len(g[0])
                pos = [(r,c) for r in range(R) for c in range(C) if g[r][c] != 0 and g[r][c] != ec]
                if not pos: return [[0]]
                r1,c1,r2,c2 = bbox(pos)
                return extract(g, r1, c1, r2, c2)
            return s
        if test_solver(task, mk(ec)): return mk(ec)
    return None

def S_crop_frame(task):
    for fc in range(1, 10):
        def mk(fc):
            def s(g):
                pos = find_pos(g, fc)
                if not pos: return [[0]]
                r1,c1,r2,c2 = bbox(pos)
                if r2-r1>=2 and c2-c1>=2:
                    return extract(g, r1+1, c1+1, r2-1, c2-1)
                return extract(g, r1, c1, r2, c2)
            return s
        if test_solver(task, mk(fc)): return mk(fc)
    return None

def S_halves(task):
    t0 = task["train"][0]
    iR, iC = len(t0["input"]), len(t0["input"][0])
    oR, oC = len(t0["output"]), len(t0["output"][0])
    tests = []
    if iR % 2 == 0 and oR == iR//2 and oC == iC:
        tests += [lambda g: [r[:] for r in g[:len(g)//2]], lambda g: [r[:] for r in g[len(g)//2:]]]
    if iC % 2 == 0 and oR == iR and oC == iC//2:
        tests += [lambda g: [r[:len(r)//2] for r in g], lambda g: [r[len(r)//2:] for r in g]]
    for fn in tests:
        if test_solver(task, fn): return fn
    return None

def S_select_cols(task):
    t0 = task["train"][0]
    iR, iC = len(t0["input"]), len(t0["input"][0])
    oR, oC = len(t0["output"]), len(t0["output"][0])
    if oR != iR or oC >= iC: return None
    for sc in range(iC - oC + 1):
        def mk(sc):
            def s(g): return [row[sc:sc+oC] for row in g]
            return s
        if test_solver(task, mk(sc)): return mk(sc)
    return None

def S_gravity(task):
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None
    def down(g):
        R, C = len(g), len(g[0]); o = G(R, C)
        for c in range(C):
            vs = [g[r][c] for r in range(R) if g[r][c] != 0]
            for i,v in enumerate(reversed(vs)): o[R-1-i][c] = v
        return o
    def up(g):
        R, C = len(g), len(g[0]); o = G(R, C)
        for c in range(C):
            vs = [g[r][c] for r in range(R) if g[r][c] != 0]
            for i,v in enumerate(vs): o[i][c] = v
        return o
    for fn in [down, up]:
        if test_solver(task, fn): return fn
    return None

def S_fill_enclosed(task):
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None
    def interior(g, bg=0):
        R, C = len(g), len(g[0])
        vis = set(); q = []
        for r in range(R):
            for c in range(C):
                if (r==0 or r==R-1 or c==0 or c==C-1) and g[r][c] == bg:
                    q.append((r,c)); vis.add((r,c))
        while q:
            cr,cc = q.pop(0)
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc = cr+dr, cc+dc
                if 0<=nr<R and 0<=nc<C and (nr,nc) not in vis and g[nr][nc] == bg:
                    vis.add((nr,nc)); q.append((nr,nc))
        return [(r,c) for r in range(R) for c in range(C) if g[r][c] == bg and (r,c) not in vis]
    for fc in range(1, 10):
        def mk(fc):
            def s(g):
                o = CG(g)
                for r,c in interior(g): o[r][c] = fc
                return o
            return s
        if test_solver(task, mk(fc)): return mk(fc)
    return None

def S_fill_rect(task):
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None
    def s(g):
        R, C = len(g), len(g[0]); o = CG(g)
        for obj in get_objs(g):
            color = g[obj[0][0]][obj[0][1]]
            r1,c1,r2,c2 = bbox(obj)
            for r in range(r1, r2+1):
                for c in range(c1, c2+1): o[r][c] = color
        return o
    if test_solver(task, s): return s
    return None

def S_remove_color(task):
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None
    for rc in range(1, 10):
        def mk(rc):
            def s(g): return [[0 if v==rc else v for v in row] for row in g]
            return s
        if test_solver(task, mk(rc)): return mk(rc)
    return None

def S_denoise(task):
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None
    def s(g):
        R, C = len(g), len(g[0]); o = CG(g)
        for r in range(R):
            for c in range(C):
                if g[r][c] != 0:
                    n = sum(1 for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)] if 0<=r+dr<R and 0<=c+dc<C and g[r+dr][c+dc]==g[r][c])
                    if n == 0: o[r][c] = 0
        return o
    if test_solver(task, s): return s
    return None

def S_connect_dots(task):
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None
    def s(g):
        R, C = len(g), len(g[0]); o = CG(g)
        colors = defaultdict(list)
        for r in range(R):
            for c in range(C):
                if g[r][c] != 0: colors[g[r][c]].append((r,c))
        for color, ps in colors.items():
            if len(ps) == 2:
                (r1,c1),(r2,c2) = ps
                if r1==r2:
                    for c in range(min(c1,c2)+1, max(c1,c2)):
                        if o[r1][c] == 0: o[r1][c] = color
                elif c1==c2:
                    for r in range(min(r1,r2)+1, max(r1,r2)):
                        if o[r][c1] == 0: o[r][c1] = color
        return o
    if test_solver(task, s): return s
    return None

def S_fill_between(task):
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None
    def s(g):
        R, C = len(g), len(g[0]); o = CG(g)
        colors = defaultdict(list)
        for r in range(R):
            for c in range(C):
                if g[r][c] != 0: colors[g[r][c]].append((r,c))
        for color, ps in colors.items():
            rg = defaultdict(list); cg = defaultdict(list)
            for r,c in ps: rg[r].append(c); cg[c].append(r)
            for r, cs in rg.items():
                cs.sort()
                for i in range(len(cs)-1):
                    for c in range(cs[i], cs[i+1]+1): o[r][c] = color
            for c, rs in cg.items():
                rs.sort()
                for i in range(len(rs)-1):
                    for r in range(rs[i], rs[i+1]+1): o[r][c] = color
        return o
    if test_solver(task, s): return s
    return None

def S_mirror_half(task):
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None
    fns = [
        lambda g: [[g[r][c] if c < len(g[0])//2 else g[r][len(g[0])-1-c] for c in range(len(g[0]))] for r in range(len(g))],
        lambda g: [[g[r][c] if c >= len(g[0])//2 else g[r][len(g[0])-1-c] for c in range(len(g[0]))] for r in range(len(g))],
        lambda g: [[g[r][c] if r < len(g)//2 else g[len(g)-1-r][c] for c in range(len(g[0]))] for r in range(len(g))],
        lambda g: [[g[r][c] if r >= len(g)//2 else g[len(g)-1-r][c] for c in range(len(g[0]))] for r in range(len(g))],
    ]
    for fn in fns:
        if test_solver(task, fn): return fn
    return None

def S_complete_sym(task):
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None
    def s(g):
        R, C = len(g), len(g[0]); o = CG(g)
        for r in range(R):
            for c in range(C):
                if o[r][c] == 0:
                    for mr,mc in [(R-1-r,c),(r,C-1-c),(R-1-r,C-1-c)]:
                        if 0<=mr<R and 0<=mc<C and o[mr][mc] != 0:
                            o[r][c] = o[mr][mc]; break
        return o
    if test_solver(task, s): return s
    return None

def S_surround(task):
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None
    new_cs = set()
    R, C = len(t0["input"]), len(t0["input"][0])
    for r in range(R):
        for c in range(C):
            if t0["input"][r][c] == 0 and t0["output"][r][c] != 0:
                new_cs.add(t0["output"][r][c])
    for bc in new_cs:
        def mk(bc):
            def s(g):
                R2, C2 = len(g), len(g[0]); o = CG(g)
                for r in range(R2):
                    for c in range(C2):
                        if g[r][c] != 0:
                            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                                nr,nc = r+dr, c+dc
                                if 0<=nr<R2 and 0<=nc<C2 and g[nr][nc] == 0: o[nr][nc] = bc
                return o
            return s
        if test_solver(task, mk(bc)): return mk(bc)
    return None

def S_sep_combine(task):
    t0 = task["train"][0]
    iR, iC = len(t0["input"]), len(t0["input"][0])
    oR, oC = len(t0["output"]), len(t0["output"][0])
    if oR >= iR and oC >= iC: return None
    g0 = t0["input"]
    for sc in range(1, 10):
        for ori in ['h','v','both']:
            res = _sep_sections(g0, sc, ori)
            if res is None: continue
            secs, nR, nC = res
            if len(secs) < 2: continue
            sizes = [(len(s), len(s[0])) for s in secs]
            if len(set(sizes)) != 1: continue
            sR, sC = sizes[0]
            if sR != oR or sC != oC: continue
            for comb in ['or','xor','unique','majority']:
                def mk(sc2, ori2, comb2):
                    def s(g):
                        R, C = len(g), len(g[0])
                        res2 = _sep_sections(g, sc2, ori2)
                        if not res2: return [[0]]
                        secs2, _, _ = res2
                        sR2, sC2 = len(secs2[0]), len(secs2[0][0])
                        o = G(sR2, sC2)
                        if comb2 == 'or':
                            for sec in secs2:
                                for r in range(sR2):
                                    for c in range(sC2):
                                        if sec[r][c] != 0 and sec[r][c] != sc2: o[r][c] = sec[r][c]
                        elif comb2 == 'xor':
                            for r in range(sR2):
                                for c in range(sC2):
                                    nz = [sec[r][c] for sec in secs2 if sec[r][c] != 0 and sec[r][c] != sc2]
                                    if len(nz) == 1: o[r][c] = nz[0]
                        elif comb2 == 'unique':
                            ts = [tuple(tuple(row) for row in sec) for sec in secs2]
                            for i,t in enumerate(ts):
                                if ts.count(t) == 1: return [list(row) for row in secs2[i]]
                            return [list(row) for row in secs2[0]]
                        elif comb2 == 'majority':
                            for r in range(sR2):
                                for c in range(sC2):
                                    o[r][c] = Counter(sec[r][c] for sec in secs2).most_common(1)[0][0]
                        return o
                    return s
                if test_solver(task, mk(sc, ori, comb)): return mk(sc, ori, comb)
    return None

def S_sep_pick(task):
    t0 = task["train"][0]
    iR, iC = len(t0["input"]), len(t0["input"][0])
    oR, oC = len(t0["output"]), len(t0["output"][0])
    if oR >= iR and oC >= iC: return None
    g0 = t0["input"]
    for sc in range(1, 10):
        for ori in ['h','v','both']:
            res = _sep_sections(g0, sc, ori)
            if res is None: continue
            secs, _, _ = res
            if len(secs) < 2: continue
            for idx in range(len(secs)):
                if len(secs[idx]) == oR and len(secs[idx][0]) == oC:
                    def mk(sc2, ori2, idx2):
                        def s(g):
                            R, C = len(g), len(g[0])
                            res2 = _sep_sections(g, sc2, ori2)
                            if res2 and idx2 < len(res2[0]): return res2[0][idx2]
                            return [[0]]
                        return s
                    if test_solver(task, mk(sc, ori, idx)): return mk(sc, ori, idx)
    return None

def S_sep_to_small(task):
    t0 = task["train"][0]
    iR, iC = len(t0["input"]), len(t0["input"][0])
    oR, oC = len(t0["output"]), len(t0["output"][0])
    if oR >= iR and oC >= iC: return None
    g0 = t0["input"]
    for sc in range(10):
        for ori in ['both','h','v']:
            res = _sep_sections(g0, sc, ori)
            if res is None: continue
            secs, nR, nC = res
            if len(secs) < 2: continue
            sizes = [(len(s), len(s[0])) for s in secs]
            if len(set(sizes)) != 1: continue
            if nR != oR or nC != oC: continue
            # Color of content
            def mk(sc2, ori2):
                def s(g):
                    R, C = len(g), len(g[0])
                    res2 = _sep_sections(g, sc2, ori2)
                    if not res2: return [[0]]
                    secs2, nR2, nC2 = res2
                    o = G(nR2, nC2)
                    for i, sec in enumerate(secs2):
                        ri, ci = i // nC2, i % nC2
                        cnt = Counter()
                        for row in sec:
                            for v in row:
                                if v != 0 and v != sc2: cnt[v] += 1
                        if cnt: o[ri][ci] = cnt.most_common(1)[0][0]
                    return o
                return s
            if test_solver(task, mk(sc, ori)): return mk(sc, ori)
            # Binary
            for fill in range(1, 10):
                def mk2(sc2, ori2, fc):
                    def s(g):
                        R, C = len(g), len(g[0])
                        res2 = _sep_sections(g, sc2, ori2)
                        if not res2: return [[0]]
                        secs2, nR2, nC2 = res2
                        o = G(nR2, nC2)
                        for i, sec in enumerate(secs2):
                            ri, ci = i // nC2, i % nC2
                            has = any(v != 0 and v != sc2 for row in sec for v in row)
                            if has: o[ri][ci] = fc
                        return o
                    return s
                if test_solver(task, mk2(sc, ori, fill)): return mk2(sc, ori, fill)
    return None

def S_compress_block(task):
    t0 = task["train"][0]
    iR, iC = len(t0["input"]), len(t0["input"][0])
    oR, oC = len(t0["output"]), len(t0["output"][0])
    if oR >= iR or oC >= iC: return None
    if iR % oR != 0 or iC % oC != 0: return None
    bR, bC = iR // oR, iC // oC
    def s1(g):
        R, C = len(g), len(g[0])
        o = G(R//bR, C//bC)
        for r in range(R//bR):
            for c in range(C//bC):
                cnt = Counter()
                for dr in range(bR):
                    for dc in range(bC):
                        v = g[r*bR+dr][c*bC+dc]
                        if v != 0: cnt[v] += 1
                if cnt: o[r][c] = cnt.most_common(1)[0][0]
        return o
    if test_solver(task, s1): return s1
    return None

def S_pattern_in_pattern(task):
    t0 = task["train"][0]
    iR, iC = len(t0["input"]), len(t0["input"][0])
    oR, oC = len(t0["output"]), len(t0["output"][0])
    if oR % iR != 0 or oC % iC != 0: return None
    bR, bC = oR // iR, oC // iC
    bmap = {}
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        R, C = len(inp), len(inp[0])
        for r in range(R):
            for c in range(C):
                v = inp[r][c]
                block = tuple(tuple(out[r*bR+dr][c*bC:c*bC+bC]) for dr in range(bR))
                if v in bmap and bmap[v] != block: return None
                bmap[v] = block
    def s(g):
        R, C = len(g), len(g[0])
        o = G(R*bR, C*bC)
        for r in range(R):
            for c in range(C):
                b = bmap.get(g[r][c])
                if b:
                    for dr in range(bR):
                        for dc in range(bC):
                            o[r*bR+dr][c*bC+dc] = b[dr][dc]
        return o
    if test_solver(task, s): return s
    return None

def S_bbb1b8b6(task):
    """Sep col splits grid; if right pattern fits left interior, overlay."""
    t0 = task["train"][0]
    iR, iC = len(t0["input"]), len(t0["input"][0])
    oR, oC = len(t0["output"]), len(t0["output"][0])
    if iR != oR: return None
    g0 = t0["input"]
    for sep_c in range(1, iC-1):
        vals = set(g0[r][sep_c] for r in range(iR))
        if len(vals) != 1 or 0 in vals: continue
        sv = list(vals)[0]
        lw = sep_c; rw = iC - sep_c - 1
        if lw != oC: continue
        def mk(sc, sv2):
            def s(g):
                R, C = len(g), len(g[0])
                left = [row[:sc] for row in g]
                right = [row[sc+1:] for row in g]
                interior = set()
                for r in range(R):
                    for c in range(sc):
                        if left[r][c] == 0: interior.add((r,c))
                right_nz = set()
                rw2 = min(sc, len(right[0]) if right else 0)
                for r in range(R):
                    for c in range(rw2):
                        if right[r][c] != 0: right_nz.add((r,c))
                o = CG(left)
                if right_nz.issubset(interior):
                    for r,c in right_nz:
                        o[r][c] = right[r][c]
                return o
            return s
        if test_solver(task, mk(sep_c, sv)): return mk(sep_c, sv)
    return None

def S_ae58858e(task):
    """Objects of color X: recolor to Y if size >= threshold."""
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None
    # Find recoloring: which objects changed
    for src in range(1, 10):
        dsts = set()
        for p in task["train"]:
            objs = get_objs(p["input"])
            for obj in objs:
                color = p["input"][obj[0][0]][obj[0][1]]
                if color != src: continue
                oc = p["output"][obj[0][0]][obj[0][1]]
                if oc != src: dsts.add(oc)
        if len(dsts) != 1: continue
        dst = list(dsts)[0]
        # Find threshold: what separates recolored from not
        for thresh in range(1, 20):
            def mk(src2, dst2, th):
                def s(g):
                    R, C = len(g), len(g[0]); o = CG(g)
                    for obj in get_objs(g):
                        color = g[obj[0][0]][obj[0][1]]
                        if color == src2 and len(obj) >= th:
                            for r,c in obj: o[r][c] = dst2
                    return o
                return s
            if test_solver(task, mk(src, dst, thresh)): return mk(src, dst, thresh)
    return None

def S_ad38a9d0(task):
    """Each connected group of target color gets unique color by shape."""
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None
    bg = CC(t0["input"]).most_common(1)[0][0]
    ic = set(v for row in t0["input"] for v in row)
    oc = set(v for row in t0["output"] for v in row)
    removed = ic - oc
    if len(removed) != 1: return None
    target = list(removed)[0]
    smap = {}
    for p in task["train"]:
        vis = set()
        for r in range(len(p["input"])):
            for c in range(len(p["input"][0])):
                if p["input"][r][c] == target and (r,c) not in vis:
                    cells = flood(p["input"], r, c, vis)
                    r1,c1,_,_ = bbox(cells)
                    shape = tuple(sorted((r2-r1, c2-c1) for r2,c2 in cells))
                    nc = p["output"][cells[0][0]][cells[0][1]]
                    if shape in smap and smap[shape] != nc: return None
                    smap[shape] = nc
    def s(g):
        R, C = len(g), len(g[0]); o = CG(g)
        vis = set()
        for r in range(R):
            for c in range(C):
                if g[r][c] == target and (r,c) not in vis:
                    cells = flood(g, r, c, vis)
                    r1,c1,_,_ = bbox(cells)
                    shape = tuple(sorted((r2-r1, c2-c1) for r2,c2 in cells))
                    nc = smap.get(shape, target)
                    for r2,c2 in cells: o[r2][c2] = nc
        return o
    if test_solver(task, s): return s
    return None

def S_be03b35f(task):
    """5x5 -> 2x2. Output = TR rotated 180 degrees."""
    t0 = task["train"][0]
    if len(t0["input"]) != 5 or len(t0["input"][0]) != 5: return None
    if len(t0["output"]) != 2 or len(t0["output"][0]) != 2: return None

    # Output = TR (rows 0:2, cols 3:5) rotated 180
    def s1(g):
        tr = [g[r][3:5] for r in range(2)]
        return [r[::-1] for r in tr[::-1]]
    if test_solver(task, s1): return s1

    # Output = BL rotated 90 CW
    def s2(g):
        bl = [g[r][:2] for r in range(3,5)]
        return [[bl[1][0], bl[0][0]], [bl[1][1], bl[0][1]]]
    if test_solver(task, s2): return s2

    # Try majority/unique etc
    def s3(g):
        tl = [g[r][:2] for r in range(2)]
        tr = [g[r][3:5] for r in range(2)]
        bl = [g[r][:2] for r in range(3,5)]
        qs = [tl, tr, bl]
        ts = [tuple(tuple(r) for r in q) for q in qs]
        cnt = Counter(ts)
        mc = cnt.most_common(1)[0][0]
        return [list(r) for r in mc]
    if test_solver(task, s3): return s3
    return None

def S_bf32578f(task):
    """Outline shape -> filled at center-of-mass with interior."""
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None

    def s(g):
        R, C = len(g), len(g[0])
        pos = [(r,c) for r in range(R) for c in range(C) if g[r][c] != 0]
        if not pos: return CG(g)
        color = g[pos[0][0]][pos[0][1]]
        # Get outline shape
        r1,c1,r2,c2 = bbox(pos)
        h, w = r2-r1+1, c2-c1+1
        # For each row of the outline, find leftmost and rightmost columns
        # Then "close" the outline by reflecting and filling
        # Actually, the output fills the interior of the closed curve
        # and centers it vertically/horizontally

        # Get per-row extent
        row_extents = {}
        for r,c in pos:
            if r not in row_extents:
                row_extents[r] = [c, c]
            else:
                row_extents[r] = [min(row_extents[r][0], c), max(row_extents[r][1], c)]

        # Mirror: for each row, mirror the extent
        # The output is the filled region between left and right mirror
        # Actually let me think differently: the output is the "shadow" or "convex hull"
        # projected inward

        # Look at the pattern: outline (boundary) gets reflected to create closed shape, then filled
        # Train 0: L-shape of 8s -> filled rectangle of 8s
        # The outline cols: row0: [0,2], row1: [0,0], row2: [0,0], row3: [0,0], row4: [0,2]
        # Mirror across center: creates rectangle

        # For each row, find min and max columns of outline
        # Also for each col, find min and max rows
        col_extents = {}
        for r,c in pos:
            if c not in col_extents:
                col_extents[c] = [r, r]
            else:
                col_extents[c] = [min(col_extents[c][0], r), max(col_extents[c][1], r)]

        # The output seems to be: the area enclosed by reflecting the outline
        # across its symmetry axis, filled with color

        # Simpler: The L-shape has 2 arms. Mirror one arm to create a closed rectangle.
        # Then fill the rectangle.

        # Let me try: find the enclosed area by mirroring the shape across both axes
        # through its center of mass

        # Center of bounding box
        cr = (r1 + r2) / 2.0
        cc = (c1 + c2) / 2.0

        # Mirror all points and union
        all_pos = set(pos)
        for r, c in pos:
            mr = int(2*cr - r)
            mc = int(2*cc - c)
            if 0<=mr<R and 0<=mc<C:
                all_pos.add((mr, mc))

        # Fill bounding box of mirrored set
        if not all_pos: return CG(g)
        r1n,c1n,r2n,c2n = bbox(list(all_pos))

        o = G(R, C)
        for r in range(r1n, r2n+1):
            for c in range(c1n, c2n+1):
                o[r][c] = color
        return o
    if test_solver(task, s): return s
    return None

def S_c074846d(task):
    """Line of color-2 ending with 5. Rotate 90 degrees around 5, change to 3."""
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None

    # For each rotation direction
    for rot_fn in [(lambda dr,dc: (-dc,dr)), (lambda dr,dc: (dc,-dr)), (lambda dr,dc: (-dr,-dc))]:
        def mk(rf):
            def s(g):
                R, C = len(g), len(g[0])
                fives = find_pos(g, 5)
                if not fives: return CG(g)
                o = G(R, C)
                for pr, pc in fives:
                    o[pr][pc] = 5
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = pr+dr, pc+dc
                        if 0<=nr<R and 0<=nc<C and g[nr][nc] != 0 and g[nr][nc] != 5:
                            lc = g[nr][nc]
                            length = 0
                            cr, cc = nr, nc
                            while 0<=cr<R and 0<=cc<C and g[cr][cc] == lc:
                                length += 1; cr += dr; cc += dc
                            ndr, ndc = rf(dr, dc)
                            for i in range(1, length+1):
                                rr, rc = pr + ndr*i, pc + ndc*i
                                if 0<=rr<R and 0<=rc<C: o[rr][rc] = 3
                            # Also keep original as 3
                            for i in range(1, length+1):
                                rr, rc = pr + dr*i, pc + dc*i
                                if 0<=rr<R and 0<=rc<C: o[rr][rc] = 3
                return o
            return s
        if test_solver(task, mk(rot_fn)): return mk(rot_fn)

    # Try: rotate and keep as 2 (original stays as 2, new direction as 2)
    for rot_fn in [(lambda dr,dc: (-dc,dr)), (lambda dr,dc: (dc,-dr))]:
        def mk(rf):
            def s(g):
                R, C = len(g), len(g[0])
                fives = find_pos(g, 5)
                if not fives: return CG(g)
                o = G(R, C)
                for pr, pc in fives:
                    o[pr][pc] = 5
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = pr+dr, pc+dc
                        if 0<=nr<R and 0<=nc<C and g[nr][nc] != 0 and g[nr][nc] != 5:
                            lc = g[nr][nc]
                            length = 0
                            cr, cc = nr, nc
                            while 0<=cr<R and 0<=cc<C and g[cr][cc] == lc:
                                length += 1; cr += dr; cc += dc
                            ndr, ndc = rf(dr, dc)
                            # Rotated direction gets color 2
                            for i in range(1, length+1):
                                rr, rc = pr + ndr*i, pc + ndc*i
                                if 0<=rr<R and 0<=rc<C: o[rr][rc] = 2
                            # Original direction gets color 3
                            for i in range(1, length+1):
                                rr, rc = pr + dr*i, pc + dc*i
                                if 0<=rr<R and 0<=rc<C: o[rr][rc] = 3
                return o
            return s
        if test_solver(task, mk(rot_fn)): return mk(rot_fn)

    # Also try original=3, rotated=2
    for rot_fn in [(lambda dr,dc: (-dc,dr)), (lambda dr,dc: (dc,-dr))]:
        def mk(rf):
            def s(g):
                R, C = len(g), len(g[0])
                fives = find_pos(g, 5)
                if not fives: return CG(g)
                o = G(R, C)
                for pr, pc in fives:
                    o[pr][pc] = 5
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = pr+dr, pc+dc
                        if 0<=nr<R and 0<=nc<C and g[nr][nc] != 0 and g[nr][nc] != 5:
                            lc = g[nr][nc]
                            length = 0
                            cr, cc = nr, nc
                            while 0<=cr<R and 0<=cc<C and g[cr][cc] == lc:
                                length += 1; cr += dr; cc += dc
                            ndr, ndc = rf(dr, dc)
                            for i in range(1, length+1):
                                rr, rc = pr + ndr*i, pc + ndc*i
                                if 0<=rr<R and 0<=rc<C: o[rr][rc] = lc
                            for i in range(1, length+1):
                                rr, rc = pr + dr*i, pc + dc*i
                                if 0<=rr<R and 0<=rc<C: o[rr][rc] = 3
                return o
            return s
        if test_solver(task, mk(rot_fn)): return mk(rot_fn)
    return None

def S_b15fca0b(task):
    """Two 2-endpoints connected by 1-walls; fill enclosed region between them with 4."""
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None

    # Strategy: find 0-cells that are NOT reachable from grid edges without crossing 1s or 2s
    # Those enclosed cells get filled with 4
    def s1(g):
        R, C = len(g), len(g[0])
        o = CG(g)
        # Find exterior 0-cells (reachable from edge without crossing non-0)
        vis = set()
        q = []
        for r in range(R):
            for c in range(C):
                if (r == 0 or r == R-1 or c == 0 or c == C-1) and g[r][c] == 0:
                    q.append((r,c)); vis.add((r,c))
        while q:
            cr, cc = q.pop(0)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr+dr, cc+dc
                if 0<=nr<R and 0<=nc<C and (nr,nc) not in vis and g[nr][nc] == 0:
                    vis.add((nr,nc)); q.append((nr,nc))
        # Interior 0-cells = not visited
        for r in range(R):
            for c in range(C):
                if g[r][c] == 0 and (r,c) not in vis:
                    o[r][c] = 4
        return o
    if test_solver(task, s1): return s1

    # Strategy 2: 0-cells adjacent to 1-walls or 2-endpoints get filled with 4
    # BFS from 2-endpoints, spreading through 0-cells, but only those adjacent to 1 or 2
    def s2(g):
        R, C = len(g), len(g[0])
        o = CG(g)
        # 0-cells that are adjacent to at least one 1 or 2 cell
        adj_wall = set()
        for r in range(R):
            for c in range(C):
                if g[r][c] == 0:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<R and 0<=nc<C and g[nr][nc] in (1, 2):
                            adj_wall.add((r,c))
                            break
        # BFS from 2-endpoints through adj_wall cells
        twos = find_pos(g, 2)
        vis = set()
        q = []
        for r, c in twos:
            # Start from 0-cells adjacent to this 2
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if (nr, nc) in adj_wall and (nr,nc) not in vis:
                    vis.add((nr,nc)); q.append((nr,nc))
        while q:
            cr, cc = q.pop(0)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr+dr, cc+dc
                if (nr, nc) in adj_wall and (nr,nc) not in vis:
                    vis.add((nr,nc)); q.append((nr,nc))
        for r, c in vis:
            o[r][c] = 4
        return o
    if test_solver(task, s2): return s2

    # Strategy 3: iterative BFS from 2s through 0s, spreading to cells adjacent to 1/2/4
    def s3(g):
        R, C = len(g), len(g[0])
        o = CG(g)
        twos = find_pos(g, 2)
        if len(twos) < 2: return o

        # Iterative: mark 0-cells as 4 if adjacent to 1, 2, or already-marked 4
        # BFS from 2 positions
        changed = True
        filled = set()
        while changed:
            changed = False
            for r in range(R):
                for c in range(C):
                    if o[r][c] == 0 and (r,c) not in filled:
                        adj = any(0<=r+dr<R and 0<=c+dc<C and o[r+dr][c+dc] in (1, 2, 4)
                                  for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)])
                        if adj:
                            # Check reachability from both 2s
                            o[r][c] = 4
                            filled.add((r,c))
                            changed = True

        # Verify: only keep 4s that are reachable from both 2s through 4/2 cells
        # Actually this might fill too much. Let me try a different approach.
        return o
    if test_solver(task, s3): return s3

    # Strategy 4: flood from 2s, the path between them through walls defines enclosed region
    def s4(g):
        R, C = len(g), len(g[0])
        o = CG(g)
        twos = find_pos(g, 2)
        if len(twos) < 2: return o

        # For each 2, BFS outward through 0-cells
        # A 0-cell gets filled if it can reach both 2s via a path that stays
        # adjacent to 1-walls (each cell on the path is adjacent to at least one 1 or 2)
        def bfs_adj(sr, sc):
            vis = set(); q = [(sr,sc)]; vis.add((sr,sc))
            while q:
                cr, cc = q.pop(0)
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = cr+dr, cc+dc
                    if 0<=nr<R and 0<=nc<C and (nr,nc) not in vis and g[nr][nc] == 0:
                        # Must be adjacent to a wall (1 or 2) or to an already-visited cell that's adj to wall
                        adj_wall = any(0<=nr+dr2<R and 0<=nc+dc2<C and g[nr+dr2][nc+dc2] in (1,2)
                                      for dr2,dc2 in [(-1,0),(1,0),(0,-1),(0,1)])
                        adj_vis = any((nr+dr2,nc+dc2) in vis
                                     for dr2,dc2 in [(-1,0),(1,0),(0,-1),(0,1)])
                        if adj_wall:
                            vis.add((nr,nc)); q.append((nr,nc))
            return vis

        r1 = bfs_adj(twos[0][0], twos[0][1])
        r2 = bfs_adj(twos[1][0], twos[1][1])
        common = r1 & r2
        for r, c in common:
            if g[r][c] == 0: o[r][c] = 4
        return o
    if test_solver(task, s4): return s4
    return None

def S_b7256dcd(task):
    """Connected components with multiple colors: minority marker recolors majority shape."""
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None
    bg = CC(t0["input"]).most_common(1)[0][0]

    # Find which color is the "shape" color (most common non-bg)
    cnt = CC(t0["input"])
    non_bg = [(c, n) for c, n in cnt.items() if c != bg]
    if not non_bg: return None
    shape_color = max(non_bg, key=lambda x: x[1])[0]

    def s(g):
        R, C2 = len(g), len(g[0])
        bg2 = CC(g).most_common(1)[0][0]
        cnt2 = CC(g)
        non_bg2 = [(c, n) for c, n in cnt2.items() if c != bg2]
        sc = max(non_bg2, key=lambda x: x[1])[0] if non_bg2 else 0

        o = [[bg2]*C2 for _ in range(R)]
        objs = get_objs(g, bg=bg2)
        for obj in objs:
            cbc = defaultdict(list)
            for r, c in obj: cbc[g[r][c]].append((r,c))
            colors = list(cbc.keys())
            if len(colors) == 1:
                color = colors[0]
                if color != sc:
                    # Single-color marker that's not shape color - keep as is
                    for r,c in obj: o[r][c] = color
                else:
                    for r,c in obj: o[r][c] = color
            else:
                # Multi-color: find marker (non-shape-color, minority)
                markers = [c for c in colors if c != sc]
                if markers:
                    marker = markers[0]
                    # Shape cells get marker color
                    for r,c in cbc[sc]: o[r][c] = marker
                    # Marker cells become bg (already set)
                else:
                    for r,c in obj: o[r][c] = g[r][c]
        return o
    if test_solver(task, s): return s

    # Fallback: each shape-color cell gets nearest marker color via BFS
    def s2(g):
        R, C2 = len(g), len(g[0])
        bg2 = CC(g).most_common(1)[0][0]
        cnt2 = CC(g)
        non_bg2 = [(c, n) for c, n in cnt2.items() if c != bg2]
        sc2 = max(non_bg2, key=lambda x: x[1])[0] if non_bg2 else 0

        o = [[bg2]*C2 for _ in range(R)]
        # BFS from marker cells to assign colors to shape cells
        # Markers = non-bg, non-shape-color cells
        from collections import deque
        color_map = [[0]*C2 for _ in range(R)]
        q = deque()
        visited = [[False]*C2 for _ in range(R)]

        for r in range(R):
            for c in range(C2):
                if g[r][c] != bg2 and g[r][c] != sc2:
                    # Marker cell - start BFS from here
                    q.append((r, c, g[r][c]))
                    visited[r][c] = True
                    # Marker itself becomes bg (don't set in output)

        while q:
            r, c, mc = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<R and 0<=nc<C2 and not visited[nr][nc] and g[nr][nc] == sc2:
                    visited[nr][nc] = True
                    o[nr][nc] = mc
                    q.append((nr, nc, mc))

        # Shape cells not reached by any marker stay as shape color
        for r in range(R):
            for c in range(C2):
                if g[r][c] == sc2 and not visited[r][c]:
                    o[r][c] = sc2

        return o
    if test_solver(task, s2): return s2
    return None

def S_grow_extend(task):
    """Extend non-zero cells in cardinal directions to edges/boundaries."""
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None
    def s(g):
        R, C = len(g), len(g[0]); o = CG(g)
        for r in range(R):
            for c in range(C):
                if g[r][c] != 0:
                    v = g[r][c]
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        while 0<=nr<R and 0<=nc<C and g[nr][nc] == 0:
                            o[nr][nc] = v; nr += dr; nc += dc
        return o
    if test_solver(task, s): return s
    return None

def S_keep_largest(task):
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None
    def s(g):
        R, C = len(g), len(g[0])
        objs = get_objs(g)
        if not objs: return CG(g)
        largest = max(objs, key=len)
        o = G(R, C)
        for r,c in largest: o[r][c] = g[r][c]
        return o
    if test_solver(task, s): return s
    return None

def S_spiral_fill(task):
    """Fill interior of rectangular frame with spiral colors."""
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None
    # Check for rectangular frame in output with spiral fill
    # Learn colors from first training pair
    inp, out = t0["input"], t0["output"]
    R, C = len(inp), len(inp[0])

    # Find frame object
    objs = get_objs(inp)
    if not objs: return None

    # For each pair of alternating colors
    for c1 in range(1, 10):
        for c2 in range(1, 10):
            if c1 == c2: continue
            def mk(cc1, cc2):
                def s(g):
                    R2, C2 = len(g), len(g[0]); o = CG(g)
                    for obj in get_objs(g):
                        color = g[obj[0][0]][obj[0][1]]
                        r1,c1b,r2,c2b = bbox(obj)
                        h, w = r2-r1+1, c2b-c1b+1
                        area = len(obj)
                        int_area = (h-2)*(w-2) if h>2 and w>2 else 0
                        if area == h*w - int_area and int_area > 0:
                            cs = [cc1, cc2]
                            ci = 0
                            ir1,ic1,ir2,ic2 = r1+1,c1b+1,r2-1,c2b-1
                            while ir1 <= ir2 and ic1 <= ic2:
                                fc = cs[ci % len(cs)]
                                for c in range(ic1, ic2+1): o[ir1][c] = fc
                                for r in range(ir1+1, ir2+1): o[r][ic2] = fc
                                if ir1 < ir2:
                                    for c in range(ic2-1, ic1-1, -1): o[ir2][c] = fc
                                if ic1 < ic2:
                                    for r in range(ir2-1, ir1, -1): o[r][ic1] = fc
                                ir1 += 1; ic1 += 1; ir2 -= 1; ic2 -= 1; ci += 1
                    return o
                return s
            if test_solver(task, mk(c1, c2)): return mk(c1, c2)
    return None

def S_remove_bg_rows_cols(task):
    def s(g):
        R, C = len(g), len(g[0])
        kr = [r for r in range(R) if any(g[r][c] != 0 for c in range(C))]
        kc = [c for c in range(C) if any(g[r][c] != 0 for r in range(R))]
        if not kr or not kc: return [[0]]
        return [[g[r][c] for c in kc] for r in kr]
    if test_solver(task, s): return s
    return None

def S_bd14c3bf(task):
    """Objects with same shape as a 'source' object get recolored to source color."""
    t0 = task["train"][0]
    if len(t0["input"]) != len(t0["output"]) or len(t0["input"][0]) != len(t0["output"][0]): return None

    # Find source (unchanged) and target (changed) objects
    objs = get_objs(t0["input"])
    changed = {}
    for obj in objs:
        ic = t0["input"][obj[0][0]][obj[0][1]]
        oc = t0["output"][obj[0][0]][obj[0][1]]
        if ic != oc:
            changed[ic] = oc

    if len(changed) != 1: return None
    fc, tc = list(changed.items())[0]

    # Check: objects of color fc with same shape as an object of color tc get recolored
    def s(g):
        R, C = len(g), len(g[0]); o = CG(g)
        objs2 = get_objs(g)
        # Find source shape(s)
        source_shapes = set()
        target_objs = []
        for obj in objs2:
            color = g[obj[0][0]][obj[0][1]]
            r1,c1,_,_ = bbox(obj)
            shape = frozenset((r-r1, c-c1) for r,c in obj)
            if color == tc:
                source_shapes.add(shape)
            elif color == fc:
                target_objs.append((obj, shape))

        for obj, shape in target_objs:
            if shape in source_shapes:
                for r,c in obj: o[r][c] = tc
        return o
    if test_solver(task, s): return s
    return None

def S_xor_halves(task):
    t0 = task["train"][0]
    iR, iC = len(t0["input"]), len(t0["input"][0])
    oR, oC = len(t0["output"]), len(t0["output"][0])
    combos = []
    if iR % 2 == 0 and oR == iR//2 and oC == iC:
        def mk_h(mode):
            def s(g):
                R, C = len(g), len(g[0]); h = R//2
                o = G(h, C)
                for r in range(h):
                    for c in range(C):
                        a, b = g[r][c], g[r+h][c]
                        if mode == 'or':
                            o[r][c] = a if a != 0 else b
                        elif mode == 'and':
                            o[r][c] = a if a != 0 and b != 0 else 0
                        elif mode == 'xor':
                            o[r][c] = a if a != 0 and b == 0 else (b if b != 0 and a == 0 else 0)
                return o
            return s
        combos += [mk_h('or'), mk_h('and'), mk_h('xor')]
    if iC % 2 == 0 and oR == iR and oC == iC//2:
        def mk_v(mode):
            def s(g):
                R, C = len(g), len(g[0]); h = C//2
                o = G(R, h)
                for r in range(R):
                    for c in range(h):
                        a, b = g[r][c], g[r][c+h]
                        if mode == 'or':
                            o[r][c] = a if a != 0 else b
                        elif mode == 'and':
                            o[r][c] = a if a != 0 and b != 0 else 0
                        elif mode == 'xor':
                            o[r][c] = a if a != 0 and b == 0 else (b if b != 0 and a == 0 else 0)
                return o
            return s
        combos += [mk_v('or'), mk_v('and'), mk_v('xor')]
    for fn in combos:
        if test_solver(task, fn): return fn
    return None

def S_af24b4cc(task):
    """Grid with 0-separator rows and cols -> smaller grid via block summarization."""
    t0 = task["train"][0]
    # Use 0-valued separators
    res = _sep_sections(t0["input"], 0, 'both')
    if res is None: return None
    secs, nR, nC = res
    oR, oC = len(t0["output"]), len(t0["output"][0])

    # Output has same sep structure: nR block rows + sep rows, nC block cols + sep cols
    # Actually the output IS nR+seps x nC+seps... let me just try the standard approach
    def s(g):
        R, C = len(g), len(g[0])
        # Find 0-separator rows and cols
        sep_r = sorted([r for r in range(R) if all(g[r][c2] == 0 for c2 in range(C))])
        sep_c = sorted([c for c in range(C) if all(g[r2][c] == 0 for r2 in range(R))])
        rb = []; prev = 0
        for sr in sep_r:
            if sr > prev: rb.append((prev, sr))
            prev = sr + 1
        if prev < R: rb.append((prev, R))
        cb = []; prev = 0
        for sc in sep_c:
            if sc > prev: cb.append((prev, sc))
            prev = sc + 1
        if prev < C: cb.append((prev, C))

        # Output dimensions: sep rows + block rows, sep cols + block cols
        oR2 = len(sep_r) + len(rb)
        oC2 = len(sep_c) + len(cb)
        o = G(oR2, oC2)

        # Build row/col mapping
        out_rows = []; ri = 0
        for r in range(R):
            if r in sep_r:
                out_rows.append(('sep', r))
            elif ri < len(rb) and r == rb[ri][0]:
                out_rows.append(('block', ri))
                ri += 1
            # skip intermediate rows of same block

        # Simpler: for each block, get dominant non-zero color
        out_r = 0
        for i, sr in enumerate(sep_r):
            # Block rows before this separator
            pass

        # Even simpler: output has interleaved sep-rows(all 0) and block-rows(1 cell per block)
        # Let's build it directly
        all_rows = []
        bi = 0
        prev = 0
        for sr in sep_r:
            if sr > prev:
                all_rows.append(('block', bi))
                bi += 1
            all_rows.append(('sep',))
            prev = sr + 1
        if prev < R:
            all_rows.append(('block', bi))

        all_cols = []
        bi = 0
        prev = 0
        for sc in sep_c:
            if sc > prev:
                all_cols.append(('block', bi))
                bi += 1
            all_cols.append(('sep',))
            prev = sc + 1
        if prev < C:
            all_cols.append(('block', bi))

        o = G(len(all_rows), len(all_cols))
        for ri2, rm in enumerate(all_rows):
            if rm[0] == 'sep': continue
            rbi = rm[1]
            if rbi >= len(rb): continue
            for ci2, cm in enumerate(all_cols):
                if cm[0] == 'sep': continue
                cbi = cm[1]
                if cbi >= len(cb): continue
                r1, r2 = rb[rbi]
                c1, c2 = cb[cbi]
                cnt = Counter()
                for r in range(r1, r2):
                    for c in range(c1, c2):
                        v = g[r][c]
                        if v != 0: cnt[v] += 1
                if cnt:
                    # Use the unique/minority color
                    if len(cnt) == 1:
                        o[ri2][ci2] = cnt.most_common(1)[0][0]
                    else:
                        # Use majority
                        o[ri2][ci2] = cnt.most_common(1)[0][0]
        return o
    if test_solver(task, s): return s
    return None

def S_b1fc8b8e(task):
    """6x6 -> 5x5. Remove one row and one column."""
    t0 = task["train"][0]
    iR, iC = len(t0["input"]), len(t0["input"][0])
    oR, oC = len(t0["output"]), len(t0["output"][0])
    if iR - oR != 1 or iC - oC != 1: return None

    # Try removing each row/col combination
    for dr in range(iR):
        for dc in range(iC):
            def mk(dr2, dc2):
                def s(g):
                    R, C = len(g), len(g[0])
                    return [[g[r][c] for c in range(C) if c != dc2] for r in range(R) if r != dr2]
                return s
            if test_solver(task, mk(dr, dc)): return mk(dr, dc)

    # Try: remove the row/col with most zeros
    def s(g):
        R, C = len(g), len(g[0])
        row_zeros = [(sum(1 for c in range(C) if g[r][c] == 0), r) for r in range(R)]
        col_zeros = [(sum(1 for r in range(R) if g[r][c] == 0), c) for c in range(C)]
        _, dr = max(row_zeros)
        _, dc = max(col_zeros)
        return [[g[r][c] for c in range(C) if c != dc] for r in range(R) if r != dr]
    if test_solver(task, s): return s
    return None

# ============================================================
# MAIN
# ============================================================

STRATEGIES = [
    S_basics, S_color_map, S_swap_colors,
    S_scale, S_ad7e01d0, S_pattern_in_pattern, S_tile, S_htile,
    S_crop, S_crop_excl, S_crop_frame, S_remove_bg_rows_cols,
    S_halves, S_select_cols, S_xor_halves,
    S_sep_combine, S_sep_pick, S_sep_to_small,
    S_bbb1b8b6, S_compress_block,
    S_gravity, S_fill_enclosed, S_fill_rect,
    S_remove_color, S_denoise, S_connect_dots, S_fill_between,
    S_grow_extend, S_complete_sym, S_mirror_half, S_surround,
    S_keep_largest,
    S_be03b35f, S_ae58858e, S_b15fca0b, S_b7256dcd, S_bf32578f,
    S_ad38a9d0, S_bd14c3bf, S_c074846d, S_spiral_fill,
    S_af24b4cc, S_b1fc8b8e,
]

def main():
    solutions = {}
    solved = 0
    for tid in TASK_IDS:
        task = load_task(tid)
        found = False
        for strat in STRATEGIES:
            try:
                result = strat(task)
                if result and test_solver(task, result):
                    solutions[tid] = apply_solver(task, result)
                    solved += 1
                    print(f"  SOLVED {tid}")
                    found = True
                    break
            except Exception as e:
                pass
        if not found:
            print(f"  FAILED {tid}")

    print(f"\nSolved: {solved}/{len(TASK_IDS)}")
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, 'w') as f:
        json.dump(solutions, f)
    print(f"Saved {len(solutions)} solutions to {OUT_FILE}")

if __name__ == "__main__":
    main()
