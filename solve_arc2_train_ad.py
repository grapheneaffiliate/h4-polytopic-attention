#!/usr/bin/env python3
"""Solve ARC-AGI-2 tasks batch ad (84-98 prefix)."""

import json
import copy
import math
from collections import Counter, defaultdict

DATA_DIR = "C:/Users/atchi/h4-polytopic-attention/data/arc2"
OUT_FILE = "C:/Users/atchi/h4-polytopic-attention/data/arc2_solutions_train_ad.json"

TASK_IDS = """84f2aca1 8597cfd7 85b81ff1 85fa5666 8618d23e 8719f442 878187ab 87ab05b8
880c1354 88207623 8886d717 891232d6 896d5239 8a371977 8a6d367c 8abad3cf
8b28cd80 8ba14f53 8cb8642d 8dab14c2 8dae5dfc 8e2edd66 8e301a54 8ee62060
8fbca751 8fff9e47 902510d5 90347967 9110e3c5 917bccba 92e50de0 9344f635
9356391f 93b4f4b3 93c31fbe 94133066 94414823 9473c6fb 94be5b80 95755ff2
95a58926 963c33f8 963f59bc 96a8c0cd 9720b24f 97239e3d 973e499e 9772c176
97c75046 981add89 9841fdad 984d8a3e 985ae207 98c475bf""".split()


def load_task(tid):
    with open(f"{DATA_DIR}/{tid}.json") as f:
        return json.load(f)

def grid_eq(a, b):
    if len(a) != len(b): return False
    return all(r1 == r2 for r1, r2 in zip(a, b))

def test_solve(task, solve_fn):
    for pair in task["train"]:
        inp = [row[:] for row in pair["input"]]
        try:
            result = solve_fn(inp)
        except Exception:
            return False
        if not grid_eq(result, pair["output"]):
            return False
    return True

def get_test_outputs(task, solve_fn):
    return [solve_fn([row[:] for row in p["input"]]) for p in task["test"]]


# ====================== SOLVERS ======================

def solve_84f2aca1(grid):
    """Fill interior of colored rectangles: 1 cell->5, >1 cells->7."""
    R, C = len(grid), len(grid[0])
    out = [r[:] for r in grid]
    vis = [[False]*C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if grid[r][c] != 0 and not vis[r][c]:
                col = grid[r][c]
                stk = [(r,c)]; cells = []
                while stk:
                    cr,cc = stk.pop()
                    if 0<=cr<R and 0<=cc<C and not vis[cr][cc] and grid[cr][cc]==col:
                        vis[cr][cc]=True; cells.append((cr,cc))
                        stk.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
                mr,Mr = min(x[0] for x in cells), max(x[0] for x in cells)
                mc,Mc = min(x[1] for x in cells), max(x[1] for x in cells)
                interior = [(ir,ic) for ir in range(mr,Mr+1) for ic in range(mc,Mc+1) if grid[ir][ic]==0]
                if interior:
                    f = 7 if len(interior)>1 else 5
                    for ir,ic in interior: out[ir][ic]=f
    return out

def solve_8597cfd7(grid):
    """Color with most cells below vs above separator wins. Output 2x2."""
    rows = len(grid)
    sep = next(r for r in range(rows) if all(v==5 for v in grid[r]))
    colors = {}
    for r in range(rows):
        if r==sep: continue
        for c in range(len(grid[0])):
            v = grid[r][c]
            if v and v!=5:
                colors.setdefault(v,{'a':0,'b':0})
                colors[v]['b' if r>sep else 'a'] += 1
    best = max(colors, key=lambda c: colors[c]['b']-colors[c]['a'])
    return [[best,best],[best,best]]

def solve_8618d23e(grid):
    """Two halves stacked. Add 9-border between, shift diagonally."""
    R,C = len(grid),len(grid[0])
    h = R//2
    out = [[9]*(C+1) for _ in range(R+1)]
    for r in range(h):
        for c in range(C): out[r][c] = grid[r][c]
    for r in range(h):
        for c in range(C): out[h+1+r][1+c] = grid[h+r][c]
    return out

def solve_8719f442(grid):
    """3x3->15x15. Meta 5x5: center 3x3=input pattern(filled), border=pattern at exposed faces."""
    out = [[0]*15 for _ in range(15)]
    meta = [[0]*5 for _ in range(5)]
    for r in range(3):
        for c in range(3):
            if grid[r][c]==5: meta[r+1][c+1]=2
    for r in range(1,4):
        for c in range(1,4):
            if meta[r][c]==2:
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if 0<=nr<5 and 0<=nc<5 and not(1<=nr<=3 and 1<=nc<=3) and meta[nr][nc]==0:
                        meta[nr][nc]=1
    for mr in range(5):
        for mc in range(5):
            for r in range(3):
                for c in range(3):
                    if meta[mr][mc]==2: out[mr*3+r][mc*3+c]=5
                    elif meta[mr][mc]==1: out[mr*3+r][mc*3+c]=grid[r][c]
    return out

def solve_87ab05b8(grid):
    """4x4, bg=6. Place color 2 as 2x2 in its quadrant."""
    R,C=len(grid),len(grid[0])
    pos={}
    for r in range(R):
        for c in range(C):
            if grid[r][c]!=6: pos[grid[r][c]]=(r,c)
    tr,tc=pos[2]
    out=[[6]*C for _ in range(R)]
    qr,qc = 2 if tr>=2 else 0, 2 if tc>=2 else 0
    for r in range(qr,qr+2):
        for c in range(qc,qc+2): out[r][c]=2
    return out

def solve_880c1354(grid):
    """Rotate colored corner regions clockwise around 4/7 center."""
    R,C=len(grid),len(grid[0])
    out=[r[:] for r in grid]
    fours=[(r,c) for r in range(R) for c in range(C) if grid[r][c]==4]
    if not fours: return out
    cr=sum(r for r,c in fours)/len(fours)
    cc=sum(c for r,c in fours)/len(fours)
    rc=defaultdict(list)
    for r in range(R):
        for c in range(C):
            v=grid[r][c]
            if v!=4 and v!=7: rc[v].append((r,c))
    ca={}
    for col in rc:
        cells=rc[col]
        ar=sum(r for r,c in cells)/len(cells)
        ac=sum(c for r,c in cells)/len(cells)
        ca[col]=math.atan2(ar-cr,ac-cc)
    sc=sorted(rc.keys(),key=lambda c:ca[c])
    n=len(sc)
    cm={sc[i]:sc[(i-1)%n] for i in range(n)}
    for r in range(R):
        for c in range(C):
            if grid[r][c] in cm: out[r][c]=cm[grid[r][c]]
    return out

def solve_88207623(grid):
    """Mirror 4-shape across 2-line, fill with marker color."""
    R,C=len(grid),len(grid[0])
    out=[r[:] for r in grid]
    # Find vertical 2-lines
    two_cols=defaultdict(list)
    for r in range(R):
        for c in range(C):
            if grid[r][c]==2: two_cols[c].append(r)
    for col,rlist in two_cols.items():
        if len(rlist)<2: continue
        mr,Mr=min(rlist),max(rlist)
        left4=any(col>0 and grid[r][col-1]==4 for r in range(mr,Mr+1))
        right4=any(col<C-1 and grid[r][col+1]==4 for r in range(mr,Mr+1))
        if left4:
            marker=0
            for r in range(R):
                for c in range(col+1,C):
                    if grid[r][c] not in (0,2,4): marker=grid[r][c]; break
                if marker: break
            if not marker: continue
            for r in range(mr,Mr+1):
                for dc in range(1,col+1):
                    if col-dc>=0 and grid[r][col-dc]==4 and col+dc<C:
                        out[r][col+dc]=marker
        elif right4:
            marker=0
            for r in range(R):
                for c in range(col):
                    if grid[r][c] not in (0,2,4): marker=grid[r][c]; break
                if marker: break
            if not marker: continue
            for r in range(mr,Mr+1):
                for dc in range(1,C-col):
                    if col+dc<C and grid[r][col+dc]==4 and col-dc>=0:
                        out[r][col-dc]=marker
    return out

def solve_8886d717(grid):
    """8s in 2-region disappear. 8s in 7-region duplicate toward 9-border."""
    R,C=len(grid),len(grid[0])
    out=[r[:] for r in grid]
    nines=[(r,c) for r in range(R) for c in range(C) if grid[r][c]==9]
    if not nines: return out
    # Determine 9-border direction
    dr,dc=0,0
    if all(c==nines[0][1] for r,c in nines):
        dc=-1 if nines[0][1]==0 else 1
    elif all(r==nines[0][0] for r,c in nines):
        dr=-1 if nines[0][0]==0 else 1
    for r in range(R):
        for c in range(C):
            if grid[r][c]==8:
                # Find region: BFS to nearest 2 or 7
                in_2=False
                vis=set(); q=[(r,c)]; vis.add((r,c))
                found=None
                while q and found is None:
                    cr,cc=q.pop(0)
                    for nr,nc in [(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)]:
                        if 0<=nr<R and 0<=nc<C and (nr,nc) not in vis:
                            if grid[nr][nc]==2: found=2; break
                            elif grid[nr][nc]==7: found=7; break
                            elif grid[nr][nc]==8: vis.add((nr,nc)); q.append((nr,nc))
                if found==2:
                    out[r][c]=2
                elif found==7:
                    nr2,nc2=r+dr,c+dc
                    if 0<=nr2<R and 0<=nc2<C and grid[nr2][nc2]==7:
                        out[nr2][nc2]=8
    return out

def solve_8e2edd66(grid):
    """3x3->9x9. Place inverted pattern at zero positions."""
    n=3
    color=next(grid[r][c] for r in range(n) for c in range(n) if grid[r][c]!=0)
    inv=[[color if grid[r][c]==0 else 0 for c in range(n)] for r in range(n)]
    out=[[0]*9 for _ in range(9)]
    for mr in range(n):
        for mc in range(n):
            if grid[mr][mc]==0:
                for r in range(n):
                    for c in range(n):
                        out[mr*3+r][mc*3+c]=inv[r][c]
    return out

def solve_973e499e(grid):
    """Each cell becomes NxN block showing only that color's positions."""
    n=len(grid)
    out=[[0]*(n*n) for _ in range(n*n)]
    for mr in range(n):
        for mc in range(n):
            cc=grid[mr][mc]
            if cc!=0:
                for r in range(n):
                    for c in range(n):
                        if grid[r][c]==cc:
                            out[mr*n+r][mc*n+c]=cc
    return out

def solve_8dae5dfc(grid):
    """Concentric rectangles: reverse unique color order (outside<->inside)."""
    R,C=len(grid),len(grid[0])
    out=[r[:] for r in grid]
    vis=[[False]*C for _ in range(R)]
    regions=[]
    for r in range(R):
        for c in range(C):
            if grid[r][c]!=0 and not vis[r][c]:
                q=[(r,c)]; vis[r][c]=True; cells=[(r,c)]
                while q:
                    cr,cc=q.pop(0)
                    for nr,nc in [(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)]:
                        if 0<=nr<R and 0<=nc<C and not vis[nr][nc] and grid[nr][nc]!=0:
                            vis[nr][nc]=True; cells.append((nr,nc)); q.append((nr,nc))
                regions.append(cells)
    for cells in regions:
        mr=min(r for r,c in cells); Mr=max(r for r,c in cells)
        mc=min(c for r,c in cells); Mc=max(c for r,c in cells)
        # Extract raw layers from diagonal
        raw_layers=[]
        r1,r2,c1,c2=mr,Mr,mc,Mc
        while r1<=r2 and c1<=c2:
            raw_layers.append(grid[r1][c1])
            r1+=1; r2-=1; c1+=1; c2-=1
        # Extract unique color sequence (deduplicate consecutive)
        unique=[]
        for col in raw_layers:
            if not unique or unique[-1]!=col:
                unique.append(col)
        # Take first half (outside to center) - map each to its mirror
        n=len(unique)
        color_map={unique[i]:unique[n-1-i] for i in range(n)}
        # Apply mapping to all cells in region
        cell_set=set(cells)
        for r,c in cells:
            old_col=grid[r][c]
            out[r][c]=color_map.get(old_col,old_col)
    return out

def solve_8ee62060(grid):
    """Diagonal blocks - reverse their diagonal positions, keeping same pattern."""
    R,C=len(grid),len(grid[0])
    out=[[0]*C for _ in range(R)]
    vis=[[False]*C for _ in range(R)]
    blocks=[]
    for r in range(R):
        for c in range(C):
            if grid[r][c]!=0 and not vis[r][c]:
                q=[(r,c)]; vis[r][c]=True; cells=[(r,c)]
                while q:
                    cr,cc=q.pop(0)
                    for nr,nc in [(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)]:
                        if 0<=nr<R and 0<=nc<C and not vis[nr][nc] and grid[nr][nc]!=0:
                            vis[nr][nc]=True; cells.append((nr,nc)); q.append((nr,nc))
                mr=min(r for r,c in cells); mc=min(c for r,c in cells)
                pattern={(r-mr,c-mc):grid[r][c] for r,c in cells}
                blocks.append((mr,mc,pattern))
    if blocks:
        positions=[(b[0],b[1]) for b in blocks]
        patterns=[b[2] for b in blocks]
        n=len(blocks)
        # Place pattern i at position n-1-i
        for i in range(n):
            pr,pc=positions[n-1-i]
            for (dr,dc),val in patterns[i].items():
                if 0<=pr+dr<R and 0<=pc+dc<C:
                    out[pr+dr][pc+dc]=val
    return out

def solve_8fbca751(grid):
    """Fill 0s between 8-cells on each row/col with 2, within 8-shape groups."""
    R,C=len(grid),len(grid[0])
    out=[r[:] for r in grid]
    # Group 8-cells into clusters (4-connected)
    vis=[[False]*C for _ in range(R)]
    shapes=[]
    for r in range(R):
        for c in range(C):
            if grid[r][c]==8 and not vis[r][c]:
                q=[(r,c)]; vis[r][c]=True; cells=set(); cells.add((r,c))
                while q:
                    cr,cc=q.pop(0)
                    for nr,nc in [(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)]:
                        if 0<=nr<R and 0<=nc<C and not vis[nr][nc] and grid[nr][nc]==8:
                            vis[nr][nc]=True; cells.add((nr,nc)); q.append((nr,nc))
                shapes.append(cells)
    # Merge shapes that share bounding box overlap
    merged=True
    while merged:
        merged=False
        for i in range(len(shapes)):
            for j in range(i+1,len(shapes)):
                si,sj=shapes[i],shapes[j]
                if not si or not sj: continue
                # Check if any cell in si is adjacent (including diag) to any cell in sj
                for r,c in si:
                    for dr in [-1,0,1]:
                        for dc in [-1,0,1]:
                            if (r+dr,c+dc) in sj:
                                shapes[i]=si|sj; shapes[j]=set(); merged=True; break
                        if merged: break
                    if merged: break
                if merged: break
            if merged: break
    shapes=[s for s in shapes if s]
    for shape in shapes:
        mr=min(r for r,c in shape); Mr=max(r for r,c in shape)
        mc=min(c for r,c in shape); Mc=max(c for r,c in shape)
        for r in range(mr,Mr+1):
            for c in range(mc,Mc+1):
                if grid[r][c]==0:
                    out[r][c]=2
    return out

def solve_92e50de0(grid):
    """Grid of cells separated by grid lines of one color. One cell has pattern. Tile to matching parity cells."""
    R,C=len(grid),len(grid[0])
    out=[r[:] for r in grid]
    # Find the grid line color: the color that forms full rows and full columns
    line_color=None
    for c_val in range(10):
        h=[r for r in range(R) if all(grid[r][c]==c_val for c in range(C))]
        v=[c for c in range(C) if all(grid[r][c]==c_val for r in range(R))]
        if len(h)>=2 and len(v)>=2:
            line_color=c_val; break
    if line_color is None: return out
    h_lines=[r for r in range(R) if all(grid[r][c]==line_color for c in range(C))]
    v_lines=[c for c in range(C) if all(grid[r][c]==line_color for r in range(R))]
    rb=[-1]+h_lines+[R]; cb=[-1]+v_lines+[C]
    pattern=None; pi=pj=0
    for i in range(len(rb)-1):
        for j in range(len(cb)-1):
            r1,r2=rb[i]+1,rb[i+1]; c1,c2=cb[j]+1,cb[j+1]
            if any(grid[r][c] not in (0,line_color) for r in range(r1,r2) for c in range(c1,c2)):
                pattern=[[grid[r][c] for c in range(c1,c2)] for r in range(r1,r2)]
                pi,pj=i,j; break
        if pattern: break
    if not pattern: return out
    for i in range(len(rb)-1):
        for j in range(len(cb)-1):
            if i%2==pi%2 and j%2==pj%2:
                r1,r2=rb[i]+1,rb[i+1]; c1,c2=cb[j]+1,cb[j+1]
                for dr,r in enumerate(range(r1,r2)):
                    for dc,c in enumerate(range(c1,c2)):
                        if dr<len(pattern) and dc<len(pattern[0]):
                            out[r][c]=pattern[dr][dc]
    return out

def solve_90347967(grid):
    """Rotate non-zero shape 180 degrees around the cell with color 5."""
    R,C=len(grid),len(grid[0])
    out=[[0]*C for _ in range(R)]
    # Find cell with color 5
    cr,cc=0,0
    for r in range(R):
        for c in range(C):
            if grid[r][c]==5: cr,cc=r,c
    for r in range(R):
        for c in range(C):
            if grid[r][c]!=0:
                nr,nc=2*cr-r,2*cc-c
                if 0<=nr<R and 0<=nc<C:
                    out[nr][nc]=grid[r][c]
    return out

def solve_9720b24f(grid):
    """Remove foreign-color components inside shape outlines.
    Use total color count to determine which is outline vs marker."""
    R,C=len(grid),len(grid[0])
    out=[r[:] for r in grid]
    # Count total cells per color
    color_count=Counter()
    for r in range(R):
        for c in range(C):
            if grid[r][c]!=0: color_count[grid[r][c]]+=1
    # Find connected components
    vis=[[False]*C for _ in range(R)]
    components=[]
    cell_to_comp={}
    for r in range(R):
        for c in range(C):
            if grid[r][c]!=0 and not vis[r][c]:
                col=grid[r][c]
                q=[(r,c)]; vis[r][c]=True; cells=set(); cells.add((r,c))
                while q:
                    cr,cc=q.pop(0)
                    for nr,nc in [(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)]:
                        if 0<=nr<R and 0<=nc<C and not vis[nr][nc] and grid[nr][nc]==col:
                            vis[nr][nc]=True; cells.add((nr,nc)); q.append((nr,nc))
                idx=len(components)
                components.append((col,cells))
                for cell in cells: cell_to_comp[cell]=idx
    # For each component, check adjacency
    for idx,(col,cells) in enumerate(components):
        adj_comp_idxs=set()
        for r,c in cells:
            for nr,nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                if (nr,nc) in cell_to_comp and cell_to_comp[(nr,nc)]!=idx:
                    adj_comp_idxs.add(cell_to_comp[(nr,nc)])
        if adj_comp_idxs:
            adj_colors=set(components[i][0] for i in adj_comp_idxs)
            if len(adj_colors)==1:
                adj_col=list(adj_colors)[0]
                if adj_col!=col and color_count[adj_col]>color_count[col]:
                    for r,c in cells: out[r][c]=0
        else:
            # Not adjacent to any other component. Check if inside a shape's bbox.
            # Use color with most cells as potential encloser
            for oidx,(ocol,ocells) in enumerate(components):
                if ocol==col: continue
                if color_count[ocol]<=color_count[col]: continue
                # Use all cells of that color combined
                all_ocol=set()
                for oi,(oc,ocs) in enumerate(components):
                    if oc==ocol: all_ocol.update(ocs)
                mr=min(r for r,c in all_ocol); Mr=max(r for r,c in all_ocol)
                mc=min(c for r,c in all_ocol); Mc=max(c for r,c in all_ocol)
                if all(mr<=r<=Mr and mc<=c<=Mc for r,c in cells):
                    for r,c in cells: out[r][c]=0
                    break
    return out

def solve_97c75046(grid):
    """Move color 5 to be adjacent to the 0-region (next to boundary)."""
    R,C=len(grid),len(grid[0])
    out=[r[:] for r in grid]
    bg=7
    # Find 5 position
    five_r=five_c=None
    for r in range(R):
        for c in range(C):
            if grid[r][c]==5:
                five_r,five_c=r,c; break
        if five_r is not None: break
    # Find 0-region boundary - the 0-cell closest to the 5
    zeros=[(r,c) for r in range(R) for c in range(C) if grid[r][c]==0]
    if not zeros or five_r is None: return out
    # Find the 0-boundary cell adjacent to bg (7)
    # The 5 moves to be adjacent to the 0-region boundary
    # From examples: 5 moves along the boundary of the 0-region
    # T0: 5 at (2,5), 0s start at (5,5). Output: 5 at (6,3) which is adjacent to 0-boundary
    # T1: 5 at (5,1), 0s at right side. Output: 5 at (2,6) adjacent to 0-boundary
    # T2: 5 at (11,0), 0s on left side. Output: 5 at (6,0) adjacent to 0-boundary

    # The 5 moves to the boundary of the 0-shape, specifically to the cell
    # that is bg-colored and adjacent to the "start" of the 0-shape (closest corner).

    # Find the bounding of the 0 region and figure out which bg cell the 5 should go to.
    # The 5 appears to move to be right at the edge where the 0-region begins,
    # in the same row/column alignment.

    # Let me figure out the direction: 5's position relative to 0-region.
    # For each 0-cell, find the closest bg-cell that's between 5 and the 0-region.

    # Simpler: find the shortest path from 5 to any 0-cell through bg cells.
    # The 5 moves to the bg cell just before entering the 0-region.

    # BFS from 5 through bg cells to find nearest 0-adjacent bg cell
    from collections import deque
    q=deque()
    q.append((five_r,five_c,[(five_r,five_c)]))
    visited=set()
    visited.add((five_r,five_c))
    target=None
    zero_set=set(zeros)
    while q:
        r,c,path=q.popleft()
        # Check if adjacent to a 0 cell
        for nr,nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
            if (nr,nc) in zero_set:
                target=(r,c)
                break
        if target: break
        for nr,nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
            if 0<=nr<R and 0<=nc<C and (nr,nc) not in visited and grid[nr][nc]==bg:
                visited.add((nr,nc))
                q.append((nr,nc,path+[(nr,nc)]))
    if target:
        out[five_r][five_c]=bg
        out[target[0]][target[1]]=5
    return out

def solve_981add89(grid):
    """Non-background colors in row 0 become vertical columns through entire grid."""
    R,C=len(grid),len(grid[0])
    bg=Counter(grid[r][c] for r in range(R) for c in range(C)).most_common(1)[0][0]
    out=[r[:] for r in grid]
    for c in range(C):
        if grid[0][c]!=bg:
            color=grid[0][c]
            for r in range(R):
                out[r][c]=color
    return out

def solve_902510d5(grid):
    """Remove certain marker colors. Shape remains. A corner marker creates a triangle fill."""
    R,C=len(grid),len(grid[0])
    out=[r[:] for r in grid]
    # Find the main shape (largest non-0, non-marker connected component)
    # Markers are isolated single cells of specific colors.
    # From T0: colors 4(marker), 2(shape), 3(marker). Output: 4 creates a triangle at corner.
    # Actually: 4s disappear, 3 disappears. A triangle of 4 appears at the bottom-right corner.
    # The shape is the 8 or 2 colored cells forming a diagonal/shape.
    # The triangle fill is in the "empty" corner opposite to where the shape opens.

    # This is complex. Let me look at it differently.
    # In T0: 4 appears at (0,1) and (4,3). 3 at (5,8). 2s form the shape.
    # Output: 4s removed, 3 removed. 2s unchanged. New 4s at (4,8) and (5,7),(5,8).
    # So the 4 fills a triangle at the bottom-right corner (where 3 was).

    # The rule: find the non-shape singleton markers. One marker defines the fill color,
    # another defines where to fill (creates triangle growing from that position).

    # T3: 9 at (0,0), 7 at (6,8) and (7,1), 4 at (9,1), 3 at (9,8).
    # 8s form the shape. Output: 7s at (0,0) triangle area growing.
    # 9->removed, 7 fills triangle from (0,0) corner.

    # The fill creates a right triangle from one corner, with the fill color being
    # determined by the marker closest to that corner, and the opposite marker determines direction.

    # This is too complex without more analysis. Skip.
    return out

def solve_94414823(grid):
    """Rectangle with 5-border. Two markers outside -> fill interior with quadrant pattern."""
    R,C=len(grid),len(grid[0])
    out=[r[:] for r in grid]
    fives=[(r,c) for r in range(R) for c in range(C) if grid[r][c]==5]
    if not fives: return out
    mr,Mr=min(r for r,c in fives),max(r for r,c in fives)
    mc,Mc=min(c for r,c in fives),max(c for r,c in fives)
    markers=[(r,c,grid[r][c]) for r in range(R) for c in range(C) if grid[r][c] not in (0,5)]
    ir,Ir=mr+1,Mr-1; ic,Ic=mc+1,Mc-1
    ih=Ir-ir+1; iw=Ic-ic+1
    hh,hw=ih//2,iw//2
    mid_r=(ir+Ir)/2; mid_c=(ic+Ic)/2
    qc={}
    for r,c,col in markers:
        if r<mid_r:
            if c>mid_c: qc['TR']=col
            else: qc['TL']=col
        else:
            if c>mid_c: qc['BR']=col
            else: qc['BL']=col
    tr=qc.get('TR',0); br=qc.get('BR',0)
    for r in range(ir,Ir+1):
        for c in range(ic,Ic+1):
            rr,cc=r-ir,c-ic
            top=rr<hh; left=cc<hw
            if (top and left) or (not top and not left): out[r][c]=br
            else: out[r][c]=tr
    return out

def solve_93b4f4b3(grid):
    """Left half has 5-bordered sections with 0-holes. Right half has color templates.
    Fill 0-holes with color from REVERSED right panel sections."""
    R,C=len(grid),len(grid[0])
    sep=None
    for c in range(C):
        if all(grid[r][c]==0 for r in range(R)):
            sep=c; break
    if sep is None: return [r[:] for r in grid]

    left=[r[:sep] for r in grid]
    right=[r[sep+1:] for r in grid]
    LW=len(left[0]); RW=len(right[0])

    # Find all-5 rows in left
    five_rows=[r for r in range(R) if all(left[r][c]==5 for c in range(LW))]
    sections=[]
    for i in range(len(five_rows)-1):
        r1,r2=five_rows[i]+1,five_rows[i+1]
        if r1<r2: sections.append((r1,r2))

    # Find all-0 rows in right (section separators)
    zero_rows=[r for r in range(R) if all(right[r][c]==0 for c in range(RW))]
    right_sections=[]
    for i in range(len(zero_rows)-1):
        r1,r2=zero_rows[i]+1,zero_rows[i+1]
        if r1<r2: right_sections.append((r1,r2))

    # Get colors from right sections
    right_colors=[]
    for rs,re in right_sections:
        color=0
        for r in range(rs,re):
            for c in range(RW):
                if right[r][c]!=0: color=right[r][c]; break
            if color: break
        right_colors.append(color)

    # REVERSE the right colors to match left sections
    right_colors_rev=right_colors[::-1]

    out=[r[:sep] for r in grid]
    for si,(s_start,s_end) in enumerate(sections):
        if si<len(right_colors_rev):
            color=right_colors_rev[si]
            for r in range(s_start,s_end):
                for c in range(LW):
                    if out[r][c]==0:
                        out[r][c]=color
    return out

def solve_917bccba(grid):
    """Cross pattern with 8 and 1 border. The 8-cross shifts to align with 1-border."""
    R,C=len(grid),len(grid[0])
    out=[r[:] for r in grid]

    # Find the 1-rectangle
    ones=[(r,c) for r in range(R) for c in range(C) if grid[r][c]==1]
    if not ones: return out
    mr=min(r for r,c in ones); Mr=max(r for r,c in ones)
    mc=min(c for r,c in ones); Mc=max(c for r,c in ones)

    # Find 8-lines (horizontal and vertical)
    # The 8-line that's horizontal
    h_row=None; v_col=None
    for r in range(R):
        eight_count=sum(1 for c in range(C) if grid[r][c]==8)
        if eight_count>C//2: h_row=r; break
    for c in range(C):
        eight_count=sum(1 for r in range(R) if grid[r][c]==8)
        if eight_count>R//2: v_col=c; break

    if h_row is None or v_col is None: return out

    # The 8-cross should move so its intersection aligns with the 1-rectangle corner
    # From T0: 8-cross at (6,5), 1-rect at rows 3-9, cols 2-8.
    # Output: 8-cross moves to (3,8) - top-right corner of 1-rect.
    # Wait: output has horizontal 8 at row 3, vertical 8 at col 8.
    # 1-rect stays at rows 3-9, cols 2-8.
    # So: the 8 lines move to the edges of the 1-rectangle.

    # Find where the 8-lines should go:
    # The 8 horizontal line is at a certain row. In output it shifts to the 1-rect border row.
    # The 8 vertical column shifts to the 1-rect border column.

    # In T0: h_row=6 (middle of 1-rect rows 3-9, range midpoint=6). Output: h_row=3 (top).
    # v_col=5 (middle of cols 2-8, midpoint=5). Output: v_col=8 (right).
    # The shift: from center toward the border farthest from current position?
    # h_row=6, rect rows 3-9. Distance to top=3, to bottom=3. Same distance.
    # v_col=5, rect cols 2-8. Distance to left=3, to right=3. Same distance.
    # Hmm, that's symmetric. But the output goes to (3,8). Why top-right?

    # Maybe the 8-lines move to the border of the 1-rect on the side away from center?
    # The 1-rect has internal 0-cells (where 8-lines used to be) plus actual 0-cells inside.

    # Actually, looking at T0 output: the interior of the 1-rect is all 0 (no 8s inside).
    # The 8-horizontal goes to row 3 (top of 1-rect), 8-vertical to col 8 (right of 1-rect).
    # Inside the 1-rect (rows 4-8, cols 3-7): all 0.

    # So the 8-lines move to overlap with the 1-rect border, and the interior becomes all 0.

    # Clear all 8s from current positions
    for r in range(R):
        for c in range(C):
            if out[r][c]==8: out[r][c]=0

    # Place 8-horizontal at the 1-rect border rows and 8-vertical at border cols
    # But which border? From T0 it seems specific.
    # Let me check T1 and T2 to see the pattern.

    # I need more data. Let me try: the 8-horizontal moves to the row of the 1-rect
    # that is farthest from the original 8-row. Similarly for column.
    # T0: h_row=6, rect rows 3,9. |6-3|=3, |6-9|=3. Same. Pick top=3.
    # Hmm. Let me check if there's a pattern with the inside of the rectangle.

    # Actually in input, inside the rect between 1-border:
    # row 4 col 5 = 8 (vertical line)
    # row 6 cols 3-8 = 8 (horizontal line)
    # The 8s inside the rect form a cross. In output, the cross moves to border.

    # I'll skip this one for now as it's complex.
    return [r[:] for r in grid]

def solve_9356391f(grid):
    """Row 0 has colors. Colored dot below -> draw concentric rectangles centered on dot."""
    R,C=len(grid),len(grid[0])
    out=[r[:] for r in grid]
    # Find dot (non-0 cell below row 1)
    dot_r=dot_c=dot_color=None
    for r in range(2,R):
        for c in range(C):
            if grid[r][c]!=0:
                dot_r,dot_c,dot_color=r,c,grid[r][c]; break
        if dot_r is not None: break
    if dot_r is None: return out
    # Row 0 colors
    row0=grid[0][:]
    # Find dot_color position in row0
    dot_pos=None
    for c in range(C):
        if row0[c]==dot_color: dot_pos=c; break
    if dot_pos is None: return out
    # Replace the marker in row 0 at dot_c with 5
    for c in range(C):
        if row0[c]!=0 and row0[c]!=5:
            pass  # keep
    # The color at dot_c in row 0 needs to be replaced by 5
    out[0][dot_c]=5
    # Ring colors: distance d from center -> row0[dot_pos + d]
    ring_colors=[]
    for d in range(max(R,C)):
        c=dot_pos+d
        if c<C: ring_colors.append(row0[c])
        else: ring_colors.append(0)
    # Draw concentric rectangles (Chebyshev distance)
    for r in range(2,R):
        for c in range(C):
            d=max(abs(r-dot_r),abs(c-dot_c))
            if d<len(ring_colors) and ring_colors[d]!=0:
                out[r][c]=ring_colors[d]
    return out

def solve_85b81ff1(grid):
    """Grid with notch pattern on column pairs at odd rows. Transpose the notch grid."""
    R,C=len(grid),len(grid[0])
    out=[r[:] for r in grid]
    color=next(grid[r][c] for r in range(R) for c in range(C) if grid[r][c]!=0)
    sep_set=set(c for c in range(C) if all(grid[r][c]==0 for r in range(R)))
    cgs=[]; cg=[]
    for c in range(C):
        if c in sep_set:
            if cg: cgs.append(cg); cg=[]
        else: cg.append(c)
    if cg: cgs.append(cg)
    odd_rows=[r for r in range(R) if r%2==1]
    # Build notch grid: for each (row, col_group), which sub-col has 0?
    ng=[]
    for r in odd_rows:
        row=[]
        for cg in cgs:
            if len(cg)==2:
                if grid[r][cg[0]]==0: row.append(0)
                elif grid[r][cg[1]]==0: row.append(1)
                else: row.append(-1)
            elif len(cg)==1:
                row.append(0 if grid[r][cg[0]]==0 else -1)
            else: row.append(-1)
        ng.append(row)
    nr,nc=len(ng),len(ng[0]) if ng else 0
    # Try transpose
    if nr>0 and nc>0 and nr==nc:
        tng=[[ng[j][i] for j in range(nr)] for i in range(nc)]
    else:
        tng=[row[::-1] for row in ng[::-1]]  # 180 rotation as fallback
    for ri,r in enumerate(odd_rows):
        for gi,cg in enumerate(cgs):
            if gi<len(tng[ri]) if ri<len(tng) else False:
                if len(cg)==2:
                    out[r][cg[0]]=color; out[r][cg[1]]=color
                    if tng[ri][gi]==0: out[r][cg[0]]=0
                    elif tng[ri][gi]==1: out[r][cg[1]]=0
                elif len(cg)==1:
                    out[r][cg[0]]=color
                    if tng[ri][gi]==0: out[r][cg[0]]=0
    return out


def solve_9344f635(grid):
    """Non-bg cells that are horizontally adjacent fill entire row; others fill entire column."""
    R,C=len(grid),len(grid[0])
    bg=7
    out=[[bg]*C for _ in range(R)]
    col_fills={}
    row_fills={}
    # Find horizontally adjacent pairs/groups of same non-bg color
    for r in range(R):
        for c in range(C-1):
            if grid[r][c]!=bg and grid[r][c+1]!=bg and grid[r][c]==grid[r][c+1]:
                row_fills[r]=grid[r][c]
    # Find column singletons: non-bg cells that are NOT part of a horizontal pair
    for r in range(R):
        for c in range(C):
            if grid[r][c]!=bg:
                # Check if this cell is part of a horizontal pair
                is_horiz = False
                if c>0 and grid[r][c-1]==grid[r][c]: is_horiz=True
                if c<C-1 and grid[r][c+1]==grid[r][c]: is_horiz=True
                if not is_horiz:
                    col_fills[c]=grid[r][c]
    for c,col in col_fills.items():
        for r in range(R): out[r][c]=col
    for r,col in row_fills.items():
        for c in range(C): out[r][c]=col
    return out


def solve_8e301a54(grid):
    """Non-bg cells fall down due to gravity. The grid rotates or reflects."""
    R,C=len(grid),len(grid[0])
    bg=7
    out=[[bg]*C for _ in range(R)]
    # Collect all non-bg cells
    cells=[]
    for r in range(R):
        for c in range(C):
            if grid[r][c]!=bg:
                cells.append((r,c,grid[r][c]))
    # From T0: cells at r1-4 (top half) move to r5-8 (bottom half) in a reflected pattern
    # The non-bg pattern gets reflected vertically (top->bottom)
    # Let me check: 9 at (1,8),(2,8),(3,8),(4,8) -> output: 9 at (5,8),(6,8),(7,8),(8,8)?
    # No, output has 9 at (5,8),(6,8),(7,8). And 2,2,2 at (2,1)-(2,3) -> (5,1)-(5,3).
    # And 5 at (4,4) -> (5,4).
    # All non-bg cells shift down to align with the bottom of the grid.
    # The 9-column stays at same col but shifts down to start from row 5.

    # Actually it looks like: the non-bg cells get "gravity-dropped" to the bottom of the grid.
    # For each column, the non-bg cells in that column fall to the bottom.
    for c in range(C):
        col_cells=[]
        for r in range(R):
            if grid[r][c]!=bg:
                col_cells.append(grid[r][c])
        # Place at bottom
        for i,v in enumerate(reversed(col_cells)):
            out[R-1-i][c]=v
    return out


# ====================== MAIN ======================

def analyze_and_solve():
    solutions = {}
    solvers = {
        '84f2aca1': solve_84f2aca1,
        '8597cfd7': solve_8597cfd7,
        #'85b81ff1': solve_85b81ff1,
        '8618d23e': solve_8618d23e,
        '8719f442': solve_8719f442,
        '87ab05b8': solve_87ab05b8,
        '880c1354': solve_880c1354,
        '88207623': solve_88207623,
        '8886d717': solve_8886d717,
        '8dae5dfc': solve_8dae5dfc,
        '8e2edd66': solve_8e2edd66,
        #'8ee62060': solve_8ee62060,
        #'8fbca751': solve_8fbca751,
        '90347967': solve_90347967,
        '92e50de0': solve_92e50de0,
        #'93b4f4b3': solve_93b4f4b3,
        #'9356391f': solve_9356391f,
        #'94414823': solve_94414823,
        '973e499e': solve_973e499e,
        #'97c75046': solve_97c75046,
        #'981add89': solve_981add89,
        '9344f635': solve_9344f635,
        #'8e301a54': solve_8e301a54,
        '9720b24f': solve_9720b24f,
    }
    for tid in TASK_IDS:
        task = load_task(tid)
        if tid in solvers:
            if test_solve(task, solvers[tid]):
                solutions[tid] = get_test_outputs(task, solvers[tid])
                print(f"  PASS {tid}")
            else:
                for i, pair in enumerate(task["train"]):
                    inp = [row[:] for row in pair["input"]]
                    try:
                        result = solvers[tid](inp)
                        if not grid_eq(result, pair["output"]):
                            print(f"  FAIL {tid} (pair {i})")
                            break
                    except Exception as e:
                        print(f"  FAIL {tid} (pair {i}, error: {e})")
                        import traceback; traceback.print_exc()
                        break
        else:
            print(f"  SKIP {tid}")
    return solutions

if __name__ == "__main__":
    solutions = analyze_and_solve()
    with open(OUT_FILE, 'w') as f:
        json.dump(solutions, f)
    print(f"\nSaved {len(solutions)} solutions to {OUT_FILE}")
