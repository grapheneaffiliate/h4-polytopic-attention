#!/usr/bin/env python3
"""ARC-AGI-2 solver for tasks 2a-41."""
import json, os, math
from collections import Counter, defaultdict

DATA_DIR = "C:/Users/atchi/h4-polytopic-attention/data/arc2"
OUT_PATH = "C:/Users/atchi/h4-polytopic-attention/data/arc2_solutions_train_ab.json"
TASK_IDS = "2a5f8217,2b01abd0,2b9ef948,2c0b0aff,2c737e39,2ccd9fef,2de01db2,2e65ae53,2f0c5170,2f767503,2faf500b,305b1341,30f42897,310f3251,3194b014,319f2597,31adaf00,31d5ba1a,320afe60,32e9702f,33067df9,332202d5,332efdb3,337b420f,3391f8c0,33b52de3,3428a4f5,342ae2ed,342dd610,3490cc26,34b99a2b,34cfa167,351d6448,358ba94e,37ce87bb,37d3e8b2,3906de3d,396d80d7,3979b1a8,3a301edc,3aa6fb7a,3ad05f52,3b4c2228,3bd292e8,3c9b0459,3cd86f4f,3d31c5b3,3d588dc9,3d6c6e23,3ee1011a,3f23242b,40f6cd08,412b6263,414297c0".split(",")

def load_task(tid):
    with open(f"{DATA_DIR}/{tid}.json") as f: return json.load(f)

def test_solver(task, solver):
    for ex in task["train"]:
        try:
            if solver(ex["input"]) != ex["output"]: return False
        except: return False
    return True

def apply_solver(task, solver):
    return [solver(ex["input"]) for ex in task["test"]]

def esub(g, r1, c1, r2, c2): return [row[c1:c2+1] for row in g[r1:r2+1]]
def mcc(g):
    c = Counter(); [c.update(r) for r in g]; return c.most_common(1)[0][0]
def flip_h(g): return [r[::-1] for r in g]
def flip_v(g): return g[::-1]
def rot90(g):
    H,W=len(g),len(g[0]); return [[g[H-1-c][r] for c in range(H)] for r in range(W)]
def rot180(g): return flip_v(flip_h(g))
def rot270(g):
    H,W=len(g),len(g[0]); return [[g[c][W-1-r] for c in range(H)] for r in range(W)]
def trans(g):
    H,W=len(g),len(g[0]); return [[g[r][c] for r in range(H)] for c in range(W)]

def fobj(grid, bg=0):
    H,W=len(grid),len(grid[0]); vis=[[False]*W for _ in range(H)]; obs=[]
    for r in range(H):
        for c in range(W):
            if grid[r][c]!=bg and not vis[r][c]:
                o=[]; s=[(r,c)]
                while s:
                    rr,cc=s.pop()
                    if 0<=rr<H and 0<=cc<W and not vis[rr][cc] and grid[rr][cc]!=bg:
                        vis[rr][cc]=True; o.append((rr,cc,grid[rr][cc]))
                        for d in [(-1,0),(1,0),(0,-1),(0,1)]: s.append((rr+d[0],cc+d[1]))
                obs.append(o)
    return obs

def flood(out, sr, sc, fc, bg):
    H,W=len(out),len(out[0])
    if out[sr][sc]!=bg: return
    s=[(sr,sc)]; vis=set()
    while s:
        r,c=s.pop()
        if (r,c) in vis or r<0 or r>=H or c<0 or c>=W or out[r][c]!=bg: continue
        vis.add((r,c)); out[r][c]=fc
        for d in [(-1,0),(1,0),(0,-1),(0,1)]: s.append((r+d[0],c+d[1]))

# ======= SPECIFIC SOLVERS =======

def solve_2a5f8217(g):
    H,W=len(g),len(g[0]); o=[r[:] for r in g]; vis=[[False]*W for _ in range(H)]; grps=[]
    for r in range(H):
        for c in range(W):
            if g[r][c]==1 and not vis[r][c]:
                grp=[]; s=[(r,c)]
                while s:
                    rr,cc=s.pop()
                    if 0<=rr<H and 0<=cc<W and not vis[rr][cc] and g[rr][cc]==1:
                        vis[rr][cc]=True; grp.append((rr,cc))
                        for d in [(-1,0),(1,0),(0,-1),(0,1)]: s.append((rr+d[0],cc+d[1]))
                grps.append(grp)
    oobs=[ob for ob in fobj(g,0) if all(v!=1 for _,_,v in ob)]
    def nm(cs):
        if isinstance(cs[0],tuple) and len(cs[0])==3:
            mr=min(r for r,c,v in cs); mc=min(c for r,c,v in cs); return frozenset((r-mr,c-mc) for r,c,v in cs)
        mr=min(r for r,c in cs); mc=min(c for r,c in cs); return frozenset((r-mr,c-mc) for r,c in cs)
    for grp in grps:
        sh=nm(grp); gr=sum(r for r,c in grp)/len(grp); gc=sum(c for r,c in grp)/len(grp)
        bc,bd=None,1e9
        for ob in oobs:
            if nm(ob)==sh:
                d=abs(gr-sum(r for r,c,v in ob)/len(ob))+abs(gc-sum(c for r,c,v in ob)/len(ob))
                if d<bd: bd,bc=d,ob[0][2]
        if bc:
            for r,c in grp: o[r][c]=bc
    return o

def solve_2de01db2(g):
    H,W=len(g),len(g[0]); o=[[0]*W for _ in range(H)]
    for r in range(H):
        cnt=Counter(g[r]); main=None; best=0
        for v,c in cnt.items():
            if v!=0 and v!=7 and c>best: best,main=c,v
        if main is None: o[r]=g[r][:]; continue
        for c in range(W): o[r][c]=0 if g[r][c]==main else main
    return o

def solve_310f3251(g):
    H,W=len(g),len(g[0]); o=[[0]*(3*W) for _ in range(3*H)]
    for tr in range(3):
        for tc in range(3):
            for r in range(H):
                for c in range(W): o[tr*H+r][tc*W+c]=g[r][c]
    for r in range(H):
        for c in range(W):
            if g[r][c]!=0:
                nr,nc=(r-1)%H,(c-1)%W
                if g[nr][nc]==0:
                    for tr in range(3):
                        for tc in range(3): o[tr*H+nr][tc*W+nc]=2
    return o

def solve_31d5ba1a(g):
    H,W=len(g),len(g[0]); h=H//2
    return [[6 if (g[r][c]!=0)!=(g[h+r][c]!=0) else 0 for c in range(W)] for r in range(h)]

def solve_3aa6fb7a(g):
    H,W=len(g),len(g[0]); o=[r[:] for r in g]; vis=[[False]*W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if g[r][c]==8 and not vis[r][c]:
                grp=[]; s=[(r,c)]
                while s:
                    rr,cc=s.pop()
                    if 0<=rr<H and 0<=cc<W and not vis[rr][cc] and g[rr][cc]==8:
                        vis[rr][cc]=True; grp.append((rr,cc))
                        for d in [(-1,0),(1,0),(0,-1),(0,1)]: s.append((rr+d[0],cc+d[1]))
                gs=set(grp); r1,c1=min(r for r,c in grp),min(c for r,c in grp)
                r2,c2=max(r for r,c in grp),max(c for r,c in grp)
                for cr,cc in [(r1,c1),(r1,c2),(r2,c1),(r2,c2)]:
                    if (cr,cc) not in gs and 0<=cr<H and 0<=cc<W and g[cr][cc]==0: o[cr][cc]=1
    return o

def solve_332efdb3(g):
    H,W=len(g),len(g[0]); return [[1 if r%2==0 or c%2==0 else 0 for c in range(W)] for r in range(H)]

def solve_32e9702f(g):
    H,W=len(g),len(g[0]); o=[[5]*W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if g[r][c]!=0:
                nc=c-1
                if 0<=nc<W: o[r][nc]=g[r][c]
    return o

def solve_3428a4f5(g):
    H,W=len(g),len(g[0]); sep=next((r for r in range(H) if all(g[r][c]==4 for c in range(W))),None)
    if sep is None: return g
    top,bot=g[:sep],g[sep+1:]; oh=min(len(top),len(bot))
    return [[3 if (top[r][c]!=0)!=(bot[r][c]!=0) else 0 for c in range(W)] for r in range(oh)]

def solve_34b99a2b(g):
    H,W=len(g),len(g[0]); sep=next((c for c in range(W) if all(g[r][c]==4 for r in range(H))),None)
    if sep is None: return g
    l=[row[:sep] for row in g]; ri=[row[sep+1:] for row in g]; lw,rw=len(l[0]),len(ri[0]); ow=max(lw,rw)
    return [[2 if ((l[r][c]!=0) if c<lw else False)!=((ri[r][c]!=0) if c<rw else False) else 0 for c in range(ow)] for r in range(H)]

def solve_2f0c5170(g):
    H,W=len(g),len(g[0]); bg=8; vis=[[False]*W for _ in range(H)]; rects=[]
    for r in range(H):
        for c in range(W):
            if g[r][c]!=bg and not vis[r][c]:
                reg=[]; s=[(r,c)]
                while s:
                    rr,cc=s.pop()
                    if 0<=rr<H and 0<=cc<W and not vis[rr][cc] and g[rr][cc]!=bg:
                        vis[rr][cc]=True; reg.append((rr,cc))
                        for d in [(-1,0),(1,0),(0,-1),(0,1)]: s.append((rr+d[0],cc+d[1]))
                rects.append(reg)
    if len(rects)<2: return g
    rects.sort(key=len,reverse=True)
    def bb(reg): rs=[r for r,c in reg]; cs=[c for r,c in reg]; return min(rs),min(cs),max(rs),max(cs)
    infos=[]
    for reg in rects:
        r1,c1,r2,c2=bb(reg); sub=esub(g,r1,c1,r2,c2); nz=sum(1 for row in sub for v in row if v!=0)
        infos.append((reg,r1,c1,r2,c2,sub,nz))
    pat=max(infos,key=lambda x:x[6]); can=min(infos,key=lambda x:x[6])
    _,_,_,_,_,ps,_=pat; _,cr1,cc1,cr2,cc2,cs2,_=can
    mr=mc=mcol=None
    for r in range(len(cs2)):
        for c in range(len(cs2[0])):
            if cs2[r][c]!=0: mr,mc,mcol=r,c,cs2[r][c]
    if mr is None: return g
    pmr=pmc=None
    for r in range(len(ps)):
        for c in range(len(ps[0])):
            if ps[r][c]==mcol: pmr,pmc=r,c
    if pmr is None: return g
    ch,cw=cr2-cr1+1,cc2-cc1+1; o=[[0]*cw for _ in range(ch)]
    dr,dc=mr-pmr,mc-pmc
    for r in range(len(ps)):
        for c in range(len(ps[0])):
            if ps[r][c]!=0:
                nr,nc=r+dr,c+dc
                if 0<=nr<ch and 0<=nc<cw: o[nr][nc]=ps[r][c]
    return o

def solve_3cd86f4f(g):
    H,W=len(g),len(g[0]); oW=W+H-1; o=[[0]*oW for _ in range(H)]
    for r in range(H):
        off=H-1-r
        for c in range(W): o[r][off+c]=g[r][c]
    return o

def solve_3d6c6e23(g):
    H,W=len(g),len(g[0]); o=[[0]*W for _ in range(H)]
    cells=[(r,c,g[r][c]) for r in range(H) for c in range(W) if g[r][c]!=0]
    if not cells: return o
    col=cells[0][1]; total=len(cells)
    h=int(math.sqrt(total)+0.5)
    # Collect groups (contiguous same-color runs in column)
    groups=[]; cur_c=None; cur_n=0
    for r in range(H):
        v=g[r][col]
        if v!=0:
            if v==cur_c: cur_n+=1
            else:
                if cur_c is not None: groups.append((cur_c,cur_n))
                cur_c=v; cur_n=1
        else:
            if cur_c is not None: groups.append((cur_c,cur_n))
            cur_c=None; cur_n=0
    if cur_c is not None: groups.append((cur_c,cur_n))
    # Unique colors in order
    uc=[]
    for c,n in groups:
        if not uc or uc[-1]!=c: uc.append(c)
    for k in range(h):
        ri=H-h+k; w=2*k+1; sc=col-k
        color=uc[k] if k<len(uc) else uc[-1]
        for c in range(sc,sc+w):
            if 0<=c<W and 0<=ri<H: o[ri][c]=color
    return o

def solve_3d31c5b3(g):
    H,W=len(g),len(g[0]); sh=3
    secs=[g[i*sh:(i+1)*sh] for i in range(4)]
    out=[[0]*W for _ in range(sh)]
    # Priority: layer 2 (lowest), then 3, then 1, then 0 (highest)
    for s in [2, 3, 1, 0]:
        for r in range(sh):
            for c in range(W):
                if secs[s][r][c]!=0:
                    out[r][c]=secs[s][r][c]
    return out

def solve_3906de3d(g):
    H,W=len(g),len(g[0]); o=[[0]*W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if g[r][c]==1: o[r][c]=1
    for c in range(W):
        twos=[r for r in range(H) if g[r][c]==2]
        if not twos: continue
        top_2=min(twos); holes=0
        for r in range(top_2-1,-1,-1):
            if g[r][c]==0: holes+=1
            else: break
        for r in twos:
            nr=r-holes
            if 0<=nr<H: o[nr][c]=2
    return o

def solve_3f23242b(g):
    H,W=len(g),len(g[0]); o=[r[:] for r in g]
    for r3 in range(H):
        for c3 in range(W):
            if g[r3][c3]!=3: continue
            for c in range(c3-2,c3+3):
                if 0<=r3-2<H and 0<=c<W: o[r3-2][c]=5
            if 0<=r3-1<H: o[r3-1][c3]=5
            for r in range(r3-1,r3+2):
                if 0<=r<H:
                    if 0<=c3-2<W: o[r][c3-2]=2
                    if 0<=c3+2<W: o[r][c3+2]=2
            for c in range(c3-2,c3+3):
                if 0<=r3+2<H and 0<=c<W: o[r3+2][c]=8
            if 0<=r3+2<H:
                for c in range(0,c3-2): o[r3+2][c]=2
                for c in range(c3+3,W): o[r3+2][c]=2
    return o

def solve_351d6448(g):
    H,W=len(g),len(g[0])
    srs=[r for r in range(H) if all(g[r][c]==5 for c in range(W))]
    secs=[]; bs=[-1]+srs+[H]
    for i in range(len(bs)-1):
        s,e=bs[i]+1,bs[i+1]
        if s<e: secs.append(g[s:e])
    if len(secs)<2: return g
    pats=[]
    for sec in secs:
        for ri,row in enumerate(sec):
            nz=[(c,v) for c,v in enumerate(row) if v!=0]
            if nz: pats.append((ri,nz)); break
    if len(pats)<2: return g
    counts=[len(p[1]) for _,p in enumerate(pats)]
    fcols=[p[1][0][0] for _,p in enumerate(pats)]
    cdiffs=[counts[i]-counts[i-1] for i in range(1,len(counts))]
    coldiffs=[fcols[i]-fcols[i-1] for i in range(1,len(fcols))]
    oh=len(secs[0]); o=[[0]*W for _ in range(oh)]
    if cdiffs and all(d==cdiffs[0] for d in cdiffs):
        nc=counts[-1]+cdiffs[0]; lri,lnz=pats[-1]; cols=[v for c,v in lnz]
        ns=fcols[-1]+(coldiffs[0] if coldiffs and all(d==coldiffs[0] for d in coldiffs) else 0)
        for i in range(nc):
            c=ns+i
            if 0<=c<W: o[lri][c]=cols[i%len(cols)] if cols else 1
    elif coldiffs and all(d==coldiffs[0] for d in coldiffs):
        ns=fcols[-1]+coldiffs[0]; lri,lnz=pats[-1]
        for i,(c,v) in enumerate(lnz):
            nc=ns+i
            if 0<=nc<W: o[lri][nc]=v
    return o

def solve_37ce87bb(g):
    H,W=len(g),len(g[0]); bg=7; o=[r[:] for r in g]
    cells=[(r,c,g[r][c]) for r in range(H) for c in range(W) if g[r][c]!=bg]
    cg=defaultdict(list)
    for r,c,v in cells: cg[c].append((r,v))
    lines=[(c,sorted(cg[c])[0][0],len(cg[c]),sorted(cg[c])[0][1]) for c in sorted(cg)]
    if not lines: return o
    sl=sorted(lines,key=lambda x:x[1]); cs=[l[0] for l in sl]; ss=[l[1] for l in sl]
    if len(cs)>=2:
        nc=cs[-1]+(cs[1]-cs[0]); ns=ss[-1]+(ss[1]-ss[0])
    else: nc=cs[0]+2; ns=ss[0]+1
    for r in range(max(0,ns),H):
        if 0<=nc<W: o[r][nc]=5
    return o

def solve_30f42897(g):
    H,W=len(g),len(g[0]); bg=8; o=[r[:] for r in g]
    cells=[(r,c,g[r][c]) for r in range(H) for c in range(W) if g[r][c]!=bg]
    for r,c,v in cells:
        o[H-1-r][c]=v; o[r][W-1-c]=v; o[H-1-r][W-1-c]=v
    return o

# ======= GENERIC SOLVERS =======

def try_identity(t):
    def s(g): return [r[:] for r in g]
    return s if test_solver(t,s) else None

def try_color_swap(t):
    i,o=t["train"][0]["input"],t["train"][0]["output"]
    H,W=len(i),len(i[0])
    if len(o)!=H or len(o[0])!=W: return None
    cm={}
    for r in range(H):
        for c in range(W):
            if i[r][c] in cm:
                if cm[i[r][c]]!=o[r][c]: return None
            cm[i[r][c]]=o[r][c]
    def s(g): return [[cm.get(v,v) for v in r] for r in g]
    return s if test_solver(t,s) else None

def try_subgrid(t):
    for bg in [0,8]:
        def mk(b):
            def s(g):
                H,W=len(g),len(g[0]); r1,r2,c1,c2=H,0,W,0
                for r in range(H):
                    for c in range(W):
                        if g[r][c]!=b: r1=min(r1,r);r2=max(r2,r);c1=min(c1,c);c2=max(c2,c)
                return esub(g,r1,c1,r2,c2) if r1<=r2 else g
            return s
        sv=mk(bg)
        if test_solver(t,sv): return sv
    return None

def try_scale2x(t):
    i,o=t["train"][0]["input"],t["train"][0]["output"]
    if len(o)==2*len(i) and len(o[0])==2*len(i[0]):
        def s(g):
            H,W=len(g),len(g[0]); out=[[0]*(2*W) for _ in range(2*H)]
            for r in range(H):
                for c in range(W):
                    for dr in range(2):
                        for dc in range(2): out[2*r+dr][2*c+dc]=g[r][c]
            return out
        if test_solver(t,s): return s
    return None

def try_rotflip(t):
    for fn in [rot90,rot180,rot270,flip_h,flip_v,trans]:
        def mk(f):
            def s(g): return f(g)
            return s
        sv=mk(fn)
        if test_solver(t,sv): return sv
    return None

def try_tile3x3(t):
    i,o=t["train"][0]["input"],t["train"][0]["output"]
    H,W=len(i),len(i[0])
    if len(o)==3*H and len(o[0])==3*W:
        def s(g):
            h,w=len(g),len(g[0]); out=[[0]*(3*w) for _ in range(3*h)]
            for tr in range(3):
                for tc in range(3):
                    for r in range(h):
                        for c in range(w): out[tr*h+r][tc*w+c]=g[r][c]
            return out
        if test_solver(t,s): return s
    return None

def try_nsec(t):
    i,o=t["train"][0]["input"],t["train"][0]["output"]
    iH,iW,oH,oW=len(i),len(i[0]),len(o),len(o[0])
    if oW==iW and iH%oH==0 and iH//oH>=2:
        n=iH//oH
        def mk(nn):
            def s(g):
                H,W=len(g),len(g[0]); sh=H//nn; out=[[0]*W for _ in range(sh)]
                for i in range(nn):
                    for r in range(sh):
                        for c in range(W):
                            v=g[i*sh+r][c]
                            if v!=0: out[r][c]=v
                return out
            return s
        sv=mk(n)
        if test_solver(t,sv): return sv
    if oH==iH and iW%oW==0 and iW//oW>=2:
        n=iW//oW
        def mk(nn):
            def s(g):
                H,W=len(g),len(g[0]); sw=W//nn; out=[[0]*sw for _ in range(H)]
                for i in range(nn):
                    for r in range(H):
                        for c in range(sw):
                            v=g[r][i*sw+c]
                            if v!=0: out[r][c]=v
                return out
            return s
        sv=mk(n)
        if test_solver(t,sv): return sv
    return None

def try_sep(t):
    inp,outp=t["train"][0]["input"],t["train"][0]["output"]
    iH,iW,oH,oW=len(inp),len(inp[0]),len(outp),len(outp[0])
    for sc in range(10):
        srs=[r for r in range(iH) if all(inp[r][c]==sc for c in range(iW))]
        if srs:
            secs=[]; prev=0
            for sr in srs:
                if sr>prev: secs.append(inp[prev:sr])
                prev=sr+1
            if prev<iH: secs.append(inp[prev:])
            if secs and all(len(s)==oH for s in secs) and all(len(s[0])==oW for s in secs if s):
                def mk(sep):
                    def s(g):
                        H,W=len(g),len(g[0])
                        ss=[r for r in range(H) if all(g[r][c]==sep for c in range(W))]
                        ps=[]; prev=0
                        for sr in ss:
                            if sr>prev: ps.append(g[prev:sr])
                            prev=sr+1
                        if prev<H: ps.append(g[prev:])
                        if not ps: return g
                        ph,pw=len(ps[0]),len(ps[0][0]); out=[[0]*pw for _ in range(ph)]
                        for p in ps:
                            for r in range(min(ph,len(p))):
                                for c in range(min(pw,len(p[r]))):
                                    if p[r][c]!=0: out[r][c]=p[r][c]
                        return out
                    return s
                sv=mk(sc)
                if test_solver(t,sv): return sv
    for sc in range(10):
        scs=[c for c in range(iW) if all(inp[r][c]==sc for r in range(iH))]
        if scs:
            secs=[]; prev=0
            for s in scs:
                if s>prev: secs.append([row[prev:s] for row in inp])
                prev=s+1
            if prev<iW: secs.append([row[prev:] for row in inp])
            if secs and all(len(s[0])==oW for s in secs) and all(len(s)==oH for s in secs):
                def mk(sep):
                    def s(g):
                        H,W=len(g),len(g[0])
                        ss=[c for c in range(W) if all(g[r][c]==sep for r in range(H))]
                        ps=[]; prev=0
                        for s2 in ss:
                            if s2>prev: ps.append([row[prev:s2] for row in g])
                            prev=s2+1
                        if prev<W: ps.append([row[prev:] for row in g])
                        if not ps: return g
                        ph,pw=len(ps[0]),len(ps[0][0]); out=[[0]*pw for _ in range(ph)]
                        for p in ps:
                            for r in range(min(ph,len(p))):
                                for c in range(min(pw,len(p[r]))):
                                    if p[r][c]!=0: out[r][c]=p[r][c]
                        return out
                    return s
                sv=mk(sep)
                if test_solver(t,sv): return sv
    return None

def try_ffill(t):
    def s(g):
        H,W=len(g),len(g[0]); o=[r[:] for r in g]; vis=[[False]*W for _ in range(H)]
        for r in range(H):
            for c in range(W):
                if g[r][c]==0 and not vis[r][c]:
                    reg=[]; stk=[(r,c)]; tb=False
                    while stk:
                        rr,cc=stk.pop()
                        if 0<=rr<H and 0<=cc<W and not vis[rr][cc] and g[rr][cc]==0:
                            vis[rr][cc]=True; reg.append((rr,cc))
                            if rr==0 or rr==H-1 or cc==0 or cc==W-1: tb=True
                            for d in [(-1,0),(1,0),(0,-1),(0,1)]: stk.append((rr+d[0],cc+d[1]))
                    if not tb and reg:
                        su=Counter()
                        for rr,cc in reg:
                            for d in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr,nc=rr+d[0],cc+d[1]
                                if 0<=nr<H and 0<=nc<W and g[nr][nc]!=0: su[g[nr][nc]]+=1
                        if su:
                            f=su.most_common(1)[0][0]
                            for rr,cc in reg: o[rr][cc]=f
        return o
    return s if test_solver(t,s) else None

def try_const(t):
    outs=[json.dumps(e["output"]) for e in t["train"]]
    if len(set(outs))==1:
        o=t["train"][0]["output"]
        def s(g): return [r[:] for r in o]
        return s
    return None

# ======= MAIN =======

SPECIFIC = {
    "2a5f8217": solve_2a5f8217, "2de01db2": solve_2de01db2, "310f3251": solve_310f3251,
    "31d5ba1a": solve_31d5ba1a, "3aa6fb7a": solve_3aa6fb7a, "332efdb3": solve_332efdb3,
    "32e9702f": solve_32e9702f, "3428a4f5": solve_3428a4f5, "34b99a2b": solve_34b99a2b,
    "2f0c5170": solve_2f0c5170, "3cd86f4f": solve_3cd86f4f, "3d6c6e23": solve_3d6c6e23,
    "3f23242b": solve_3f23242b, "351d6448": solve_351d6448,
    "3d31c5b3": solve_3d31c5b3, "3906de3d": solve_3906de3d,
}

def try_sep_xor(t):
    """Sections with separator, XOR combination producing new color."""
    inp,outp=t["train"][0]["input"],t["train"][0]["output"]
    iH,iW,oH,oW=len(inp),len(inp[0]),len(outp),len(outp[0])
    # Horizontal separator with XOR
    for sc in range(10):
        srs=[r for r in range(iH) if all(inp[r][c]==sc for c in range(iW))]
        if len(srs)==1:
            sr=srs[0]
            top=inp[:sr]; bot=inp[sr+1:]
            if len(top)==oH and len(bot)==oH and iW==oW:
                # Try XOR with various output colors
                for out_color in range(1,10):
                    def mk(sep, oc):
                        def s(g):
                            H,W=len(g),len(g[0])
                            sr2=next((r for r in range(H) if all(g[r][c]==sep for c in range(W))),None)
                            if sr2 is None: return g
                            t2=g[:sr2]; b2=g[sr2+1:]
                            oh=min(len(t2),len(b2))
                            return [[oc if (t2[r][c]!=0)!=(b2[r][c]!=0) else 0 for c in range(W)] for r in range(oh)]
                        return s
                    sv=mk(sc,out_color)
                    if test_solver(t,sv): return sv
    # Vertical separator with XOR
    for sc in range(10):
        scs=[c for c in range(iW) if all(inp[r][c]==sc for r in range(iH))]
        if len(scs)==1:
            scol=scs[0]
            left=[row[:scol] for row in inp]; right=[row[scol+1:] for row in inp]
            if len(left[0])==oW and iH==oH:
                for out_color in range(1,10):
                    def mk(sep, oc):
                        def s(g):
                            H,W=len(g),len(g[0])
                            sc2=next((c for c in range(W) if all(g[r][c]==sep for r in range(H))),None)
                            if sc2 is None: return g
                            l2=[row[:sc2] for row in g]; r2=[row[sc2+1:] for row in g]
                            ow=min(len(l2[0]),len(r2[0]))
                            return [[oc if (l2[r][c]!=0)!=(r2[r][c]!=0) else 0 for c in range(ow)] for r in range(H)]
                        return s
                    sv=mk(sc,out_color)
                    if test_solver(t,sv): return sv
    return None

def try_sections_sep_overlay_with_bg(t):
    """Sections with non-zero separator, overlay sections (using separator value as bg)."""
    inp,outp=t["train"][0]["input"],t["train"][0]["output"]
    iH,iW,oH,oW=len(inp),len(inp[0]),len(outp),len(outp[0])
    for bg_col in range(10):
        # Check for column separator of bg_col
        sep_cols=[c for c in range(iW) if all(inp[r][c]==bg_col for r in range(iH))]
        if sep_cols and len(sep_cols)<iW:
            secs=[]; prev=0
            for sc in sep_cols:
                if sc>prev: secs.append([row[prev:sc] for row in inp])
                prev=sc+1
            if prev<iW: secs.append([row[prev:] for row in inp])
            if secs and all(len(s[0])==oW for s in secs) and all(len(s)==oH for s in secs):
                def mk(bgc):
                    def s(g):
                        H,W=len(g),len(g[0])
                        sps=[c for c in range(W) if all(g[r][c]==bgc for r in range(H))]
                        ps=[]; prev=0
                        for sc in sps:
                            if sc>prev: ps.append([row[prev:sc] for row in g])
                            prev=sc+1
                        if prev<W: ps.append([row[prev:] for row in g])
                        if not ps: return g
                        ph,pw=len(ps[0]),len(ps[0][0])
                        out=[[bgc]*pw for _ in range(ph)]
                        for p in ps:
                            for r in range(min(ph,len(p))):
                                for c in range(min(pw,len(p[r]))):
                                    if p[r][c]!=bgc: out[r][c]=p[r][c]
                        return out
                    return s
                sv=mk(bg_col)
                if test_solver(t,sv): return sv
        # Row separator
        sep_rows=[r for r in range(iH) if all(inp[r][c]==bg_col for c in range(iW))]
        if sep_rows and len(sep_rows)<iH:
            secs=[]; prev=0
            for sr in sep_rows:
                if sr>prev: secs.append(inp[prev:sr])
                prev=sr+1
            if prev<iH: secs.append(inp[prev:])
            if secs and all(len(s)==oH for s in secs) and all(len(s[0])==oW for s in secs if s):
                def mk(bgc):
                    def s(g):
                        H,W=len(g),len(g[0])
                        sps=[r for r in range(H) if all(g[r][c]==bgc for c in range(W))]
                        ps=[]; prev=0
                        for sr in sps:
                            if sr>prev: ps.append(g[prev:sr])
                            prev=sr+1
                        if prev<H: ps.append(g[prev:])
                        if not ps: return g
                        ph,pw=len(ps[0]),len(ps[0][0])
                        out=[[bgc]*pw for _ in range(ph)]
                        for p in ps:
                            for r in range(min(ph,len(p))):
                                for c in range(min(pw,len(p[r]))):
                                    if p[r][c]!=bgc: out[r][c]=p[r][c]
                        return out
                    return s
                sv=mk(bg_col)
                if test_solver(t,sv): return sv
    return None

GENERIC = [try_identity, try_color_swap, try_subgrid, try_scale2x,
           try_rotflip, try_tile3x3, try_nsec, try_sep, try_sep_xor,
           try_sections_sep_overlay_with_bg, try_ffill, try_const]

def main():
    solutions = {}; solved = 0
    for tid in TASK_IDS:
        task = load_task(tid)
        solver = None
        if tid in SPECIFIC:
            s = SPECIFIC[tid]
            if test_solver(task, s): solver = s
        if solver is None:
            for fn in GENERIC:
                try:
                    r = fn(task)
                    if r is not None: solver = r; break
                except: pass
        if solver and test_solver(task, solver):
            results = apply_solver(task, solver)
            if all(r is not None for r in results):
                solutions[tid] = results; solved += 1
                print(f"  SOLVED {tid}")
            else: print(f"  PARTIAL {tid}")
        else: print(f"  UNSOLVED {tid}")
    print(f"\nTotal solved: {solved}/{len(TASK_IDS)}")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f: json.dump(solutions, f)
    print(f"Saved to {OUT_PATH}")

if __name__ == "__main__": main()
