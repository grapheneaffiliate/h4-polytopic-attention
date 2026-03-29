"""ARC-AGI-2 custom solvers batch C."""


def solve_4c4377d9(grid):
    """Reverse rows and concatenate with original."""
    return list(reversed(grid)) + [row[:] for row in grid]


def solve_446ef5d2(grid):
    """Jigsaw puzzle assembly - tile fragments into rectangle with edge matching.

    Fragments of a pattern are scattered across an 8-background grid.
    One fragment has 4-markers indicating which corner it occupies.
    The output assembles all fragments into a single rectangle,
    positioned so the marked fragment stays at its original location.
    """
    R, C = len(grid), len(grid[0])
    bg = 8

    visited = [[False] * C for _ in range(R)]

    def flood(r, c):
        stack = [(r, c)]
        cells = []
        while stack:
            r2, c2 = stack.pop()
            if r2 < 0 or r2 >= R or c2 < 0 or c2 >= C:
                continue
            if visited[r2][c2] or grid[r2][c2] == bg:
                continue
            visited[r2][c2] = True
            cells.append((r2, c2))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((r2 + dr, c2 + dc))
        return cells

    raw_fragments = []
    for r in range(R):
        for c in range(C):
            if not visited[r][c] and grid[r][c] != bg:
                cells = flood(r, c)
                if cells:
                    raw_fragments.append(cells)

    fragments = []
    marked_idx = -1
    marked_corner = None
    marked_pos = None

    for idx, frag in enumerate(raw_fragments):
        rmin = min(r for r, c in frag)
        rmax = max(r for r, c in frag)
        cmin = min(c for r, c in frag)
        cmax = max(c for r, c in frag)
        cell_set = set(frag)

        frag_grid = []
        for r in range(rmin, rmax + 1):
            row = []
            for c in range(cmin, cmax + 1):
                row.append(grid[r][c] if (r, c) in cell_set else bg)
            frag_grid.append(row)

        fours = [(r - rmin, c - cmin) for r, c in frag if grid[r][c] == 4]

        if fours:
            marked_idx = len(fragments)
            fr, fc = len(frag_grid), len(frag_grid[0])
            avg_r = sum(r for r, c in fours) / len(fours)
            avg_c = sum(c for r, c in fours) / len(fours)
            marked_corner = ("bottom" if avg_r > fr / 2 else "top") + "_" + (
                "right" if avg_c > fc / 2 else "left"
            )
            for r_rel, c_rel in fours:
                frag_grid[r_rel][c_rel] = bg

        non_bg = [
            (r, c)
            for r in range(len(frag_grid))
            for c in range(len(frag_grid[0]))
            if frag_grid[r][c] != bg
        ]
        if not non_bg:
            continue

        tr_min = min(r for r, c in non_bg)
        tr_max = max(r for r, c in non_bg)
        tc_min = min(c for r, c in non_bg)
        tc_max = max(c for r, c in non_bg)

        trimmed = [row[tc_min : tc_max + 1] for row in frag_grid[tr_min : tr_max + 1]]
        if fours:
            marked_pos = (rmin + tr_min, cmin + tc_min)
        fragments.append(trimmed)

    if marked_idx < 0:
        return grid

    total_cells = sum(len(f) * len(f[0]) for f in fragments)
    max_fh = max(len(f) for f in fragments)
    max_fw = max(len(f[0]) for f in fragments)

    # Sort dimensions by squareness (most square first)
    dims = []
    for out_h in range(max_fh, total_cells + 1):
        if total_cells % out_h != 0:
            continue
        out_w = total_cells // out_h
        if out_w < max_fw or out_h > R or out_w > C:
            continue
        dims.append((abs(out_h - out_w), out_h, out_w))
    dims.sort()

    for _, out_h, out_w in dims:
        marked_frag = fragments[marked_idx]
        mh, mw = len(marked_frag), len(marked_frag[0])
        if mh > out_h or mw > out_w:
            continue

        mr = 0 if "top" in marked_corner else out_h - mh
        mc = 0 if "left" in marked_corner else out_w - mw

        canvas = [[bg] * out_w for _ in range(out_h)]
        for r in range(mh):
            for c in range(mw):
                canvas[mr + r][mc + c] = marked_frag[r][c]

        others = [fragments[i] for i in range(len(fragments)) if i != marked_idx]

        for check_tb in [True, False]:
            result = _backtrack_446(canvas, out_h, out_w, others, bg, check_tb)
            if result is not None:
                sr = marked_pos[0] - mr
                sc = marked_pos[1] - mc
                out = [[bg] * C for _ in range(R)]
                for r in range(out_h):
                    for c in range(out_w):
                        out_r, out_c = sr + r, sc + c
                        if 0 <= out_r < R and 0 <= out_c < C:
                            out[out_r][out_c] = result[r][c]
                return out

    return grid


def _check_edges_446(canvas, H, W, pr, pc, frag, bg, check_tb):
    fh, fw = len(frag), len(frag[0])
    for r in range(fh):
        for c in range(fw):
            cr, cc = pr + r, pc + c
            if canvas[cr][cc] != bg:
                return False
            if c == 0 and cc > 0 and canvas[cr][cc - 1] != bg:
                if frag[r][c] != canvas[cr][cc - 1]:
                    return False
            if c == fw - 1 and cc < W - 1 and canvas[cr][cc + 1] != bg:
                if frag[r][c] != canvas[cr][cc + 1]:
                    return False
            if check_tb:
                if r == 0 and cr > 0 and canvas[cr - 1][cc] != bg:
                    if frag[r][c] != canvas[cr - 1][cc]:
                        return False
                if r == fh - 1 and cr < H - 1 and canvas[cr + 1][cc] != bg:
                    if frag[r][c] != canvas[cr + 1][cc]:
                        return False
    return True


def _backtrack_446(canvas, H, W, remaining, bg, check_tb):
    if not remaining:
        return (
            canvas
            if all(canvas[r][c] != bg for r in range(H) for c in range(W))
            else None
        )

    empty_r, empty_c = -1, -1
    for r in range(H):
        for c in range(W):
            if canvas[r][c] == bg:
                empty_r, empty_c = r, c
                break
        if empty_r >= 0:
            break

    if empty_r < 0:
        return canvas

    for i, frag in enumerate(remaining):
        fh, fw = len(frag), len(frag[0])
        for dr in range(fh):
            for dc in range(fw):
                pr, pc = empty_r - dr, empty_c - dc
                if pr < 0 or pc < 0 or pr + fh > H or pc + fw > W:
                    continue
                valid = True
                for fr in range(fh):
                    for fc in range(fw):
                        tr, tc = pr + fr, pc + fc
                        if tr < empty_r or (tr == empty_r and tc < empty_c):
                            if canvas[tr][tc] == bg:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    continue
                if _check_edges_446(canvas, H, W, pr, pc, frag, bg, check_tb):
                    new_canvas = [row[:] for row in canvas]
                    for fr in range(fh):
                        for fc in range(fw):
                            new_canvas[pr + fr][pc + fc] = frag[fr][fc]
                    result = _backtrack_446(
                        new_canvas, H, W, remaining[:i] + remaining[i + 1 :], bg, check_tb
                    )
                    if result is not None:
                        return result
    return None


ARC2_SOLVERS_C = {
    "4c4377d9": solve_4c4377d9,
    "446ef5d2": solve_446ef5d2,
}
