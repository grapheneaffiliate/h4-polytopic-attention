"""ARC-AGI-2 custom solvers batch A."""

import math
from collections import Counter
from itertools import permutations


def solve_4c7dc4dd(grid):
    """Find 4 bordered rectangles, extract interiors, combine defects."""
    H = len(grid)
    W = len(grid[0])

    def find_rects(border_color):
        colored = set(
            (r, c) for r in range(H) for c in range(W) if grid[r][c] == border_color
        )
        rects = []
        found = set()
        for r1 in range(H):
            for c1 in range(W):
                if (r1, c1) not in colored:
                    continue
                for r2 in range(r1 + 2, H):
                    if (r2, c1) not in colored:
                        break
                    for c2 in range(c1 + 2, W):
                        if (r1, c2) not in colored:
                            break
                        if (r2, c2) not in colored:
                            continue
                        h, w = r2 - r1 + 1, c2 - c1 + 1
                        key = (r1, c1, h, w)
                        if key in found:
                            continue
                        if not all((r1, c1 + j) in colored for j in range(w)):
                            continue
                        if not all((r2, c1 + j) in colored for j in range(w)):
                            continue
                        if not all((r1 + i, c1) in colored for i in range(h)):
                            continue
                        if not all((r1 + i, c2) in colored for i in range(h)):
                            continue
                        has_inside = any(
                            grid[r1 + i][c1 + j] == border_color
                            for i in range(1, h - 1)
                            for j in range(1, w - 1)
                        )
                        if not has_inside:
                            rects.append((r1, c1, h, w))
                            found.add(key)
        return rects

    all_colors = set(grid[r][c] for r in range(H) for c in range(W))
    target_rects = None

    for bc in all_colors:
        rects = find_rects(bc)
        sizes = Counter((h - 2, w - 2) for _, _, h, w in rects)
        for (ih, iw), cnt in sizes.most_common():
            if cnt >= 4 and ih >= 2 and iw >= 2:
                matching = [
                    (r, c, h, w)
                    for r, c, h, w in rects
                    if h - 2 == ih and w - 2 == iw
                ][:4]
                target_rects = matching
                break
        if target_rects:
            break

    if not target_rects or len(target_rects) < 4:
        return None

    int_h = target_rects[0][2] - 2
    int_w = target_rects[0][3] - 2

    interiors = []
    for r0, c0, h, w in target_rects:
        g = [
            [grid[r0 + 1 + i][c0 + 1 + j] for j in range(int_w)]
            for i in range(int_h)
        ]
        interiors.append({"grid": g, "r": r0, "c": c0})

    def get_noise(g):
        return frozenset(v for row in g for v in row if v != 0 and v != 6)

    noise_groups = {}
    for d in interiors:
        nc = get_noise(d["grid"])
        noise_groups.setdefault(nc, []).append(d)

    result = [[0] * int_w for _ in range(int_h)]
    has_comp = False
    paired_noise = set()

    for nc_key, group in noise_groups.items():
        if not nc_key or len(group) != 2:
            continue
        nc = list(nc_key)[0]
        paired_noise.add(nc)
        g1, g2 = group[0]["grid"], group[1]["grid"]
        r1, c1 = group[0]["r"], group[0]["c"]
        r2, c2 = group[1]["r"], group[1]["c"]

        np1 = set(
            (r, c)
            for r in range(int_h)
            for c in range(int_w)
            if g1[r][c] == nc
        )
        np2 = set(
            (r, c)
            for r in range(int_h)
            for c in range(int_w)
            if g2[r][c] == nc
        )
        all_pos = set((r, c) for r in range(int_h) for c in range(int_w))

        if np1 | np2 == all_pos and not (np1 & np2):
            has_comp = True
            continue

        diff = np1 - np2 if len(np1) > len(np2) else np2 - np1
        if abs(c1 - c2) > abs(r1 - r2):
            for dr, dc in diff:
                result[dr][int_w - 1 - dc] = nc
        else:
            for dr, dc in diff:
                result[int_h - 1 - dr][dc] = nc

    for nc_key, group in noise_groups.items():
        if not nc_key or len(group) != 1:
            continue
        nc = list(nc_key)[0]
        if nc in paired_noise:
            continue
        g = group[0]["grid"]
        if has_comp:
            for r in range(int_h):
                for c in range(int_w):
                    v = g[r][c]
                    if v == 6:
                        result[r][c] = 6
                    elif v == 0:
                        result[r][c] = nc
        else:
            for r in range(int_h):
                for c in range(int_w):
                    if g[r][c] != 0 and result[r][c] == 0:
                        result[r][c] = g[r][c]

    for d in interiors:
        for r in range(int_h):
            for c in range(int_w):
                if d["grid"][r][c] == 6 and result[r][c] == 0:
                    result[r][c] = 6

    if not has_comp:
        standalone_color = None
        for nc_key, group in noise_groups.items():
            if nc_key and len(group) == 1:
                standalone_color = list(nc_key)[0]
                break
        if standalone_color:
            for r in range(int_h):
                for c in range(int_w):
                    if result[r][c] not in (0, 6):
                        result[r][c] = standalone_color

    return result


def solve_7b3084d4(grid):
    """4 shapes + marker 5 -> composite tiled output."""
    H, W = len(grid), len(grid[0])
    five_pos = None
    color_cells = {}
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v == 5:
                five_pos = (r, c)
            elif v != 0:
                color_cells.setdefault(v, []).append((r, c))
    if not five_pos:
        return None

    fr, fc = five_pos
    five_adj = None
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = fr + dr, fc + dc
        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] not in (0, 5):
            five_adj = grid[nr][nc]
            break
    if not five_adj:
        return None

    total = sum(len(c) for c in color_cells.values()) + 1
    N = int(math.isqrt(total))
    if N * N != total:
        return None

    shapes = {}
    for color, cells in color_cells.items():
        rmin = min(r for r, c in cells)
        cmin = min(c for r, c in cells)
        rmax = max(r for r, c in cells)
        cmax = max(c for r, c in cells)
        p = [[0] * (cmax - cmin + 1) for _ in range(rmax - rmin + 1)]
        for r, c in cells:
            p[r - rmin][c - cmin] = color
        shapes[color] = p

    all_m = color_cells[five_adj] + [five_pos]
    mr = min(r for r, c in all_m)
    mc = min(c for r, c in all_m)
    mp = [
        [0] * (max(c for r, c in all_m) - mc + 1)
        for _ in range(max(r for r, c in all_m) - mr + 1)
    ]
    for r, c in all_m:
        mp[r - mr][c - mc] = grid[r][c]
    shapes[five_adj] = mp

    def transforms(p):
        res = []
        for _ in range(4):
            res.append(p)
            res.append([row[::-1] for row in p])
            h, w = len(p), len(p[0])
            p = [[p[h - 1 - c][r] for c in range(h)] for r in range(w)]
        return res

    others = [c for c in color_cells if c != five_adj]

    for mt in transforms(mp):
        if mt[0][0] != 5:
            continue
        mh, mw = len(mt), len(mt[0])
        for perm in permutations(others):
            base = [[0] * N for _ in range(N)]
            for i in range(mh):
                for j in range(mw):
                    if mt[i][j] and i < N and j < N:
                        base[i][j] = mt[i][j]
            for t1 in transforms(shapes[perm[0]]):
                h1, w1 = len(t1), len(t1[0])
                r1 = [row[:] for row in base]
                for i in range(h1):
                    for j in range(w1):
                        rp, cp = i, N - w1 + j
                        if (
                            t1[i][j]
                            and 0 <= rp < N
                            and 0 <= cp < N
                            and r1[rp][cp] == 0
                        ):
                            r1[rp][cp] = t1[i][j]
                for t2 in transforms(shapes[perm[1]]):
                    h2, w2 = len(t2), len(t2[0])
                    r2 = [row[:] for row in r1]
                    for i in range(h2):
                        for j in range(w2):
                            rp, cp = N - h2 + i, j
                            if (
                                t2[i][j]
                                and 0 <= rp < N
                                and 0 <= cp < N
                                and r2[rp][cp] == 0
                            ):
                                r2[rp][cp] = t2[i][j]
                    for t3 in transforms(shapes[perm[2]]):
                        h3, w3 = len(t3), len(t3[0])
                        r3 = [row[:] for row in r2]
                        for i in range(h3):
                            for j in range(w3):
                                rp, cp = N - h3 + i, N - w3 + j
                                if (
                                    t3[i][j]
                                    and 0 <= rp < N
                                    and 0 <= cp < N
                                    and r3[rp][cp] == 0
                                ):
                                    r3[rp][cp] = t3[i][j]
                        if all(
                            r3[r][c] != 0 for r in range(N) for c in range(N)
                        ):
                            return r3
    return None


ARC2_SOLVERS_A = {
    "4c7dc4dd": solve_4c7dc4dd,
    "7b3084d4": solve_7b3084d4,
}
