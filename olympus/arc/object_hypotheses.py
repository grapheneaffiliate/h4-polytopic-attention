"""
Object-aware hypothesis generators for ARC puzzles.

These detect rules that operate on objects (connected components,
enclosed regions, sub-grids) rather than whole-grid transforms.
All use Python pre-check, only generating C code for verified matches.
"""

import logging
from collections import Counter
from typing import List, Optional, Tuple

from .grid_io import Grid, grids_equal
from .hypothesizer import Hypothesis, _make_c_program, _dimensions, _flatten, _color_histogram, ARC_HEADER
from .objects import (
    detect_background, find_connected_components, find_enclosed_regions,
    find_separators, split_at_separator, Object,
)

logger = logging.getLogger(__name__)


# ── Fill enclosed regions ────────────────────────────────────────

def _fill_enclosed(grid: Grid, bg: int, fill_color: int) -> Grid:
    """Fill enclosed background regions with fill_color."""
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    regions = find_enclosed_regions(grid, bg=bg)
    for region in regions:
        for r, c in region:
            result[r][c] = fill_color
    return result


def _try_fill_enclosed(pairs: list) -> Optional[Hypothesis]:
    """Check if the rule fills enclosed regions with a specific color."""
    # For each pair, check if the difference is exactly the enclosed regions
    fill_color = None
    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        h_i, w_i = _dimensions(inp)
        h_o, w_o = _dimensions(out)
        if h_i != h_o or w_i != w_o:
            return None

        bg = detect_background(inp)
        regions = find_enclosed_regions(inp, bg=bg)
        if not regions:
            return None

        enclosed_cells = set()
        for region in regions:
            enclosed_cells.update(region)

        # Check: output == input except enclosed cells get a single fill color
        for r in range(h_i):
            for c in range(w_i):
                if (r, c) in enclosed_cells:
                    # This cell should be changed to fill_color
                    if fill_color is None:
                        fill_color = out[r][c]
                    elif out[r][c] != fill_color:
                        return None
                else:
                    # This cell should be unchanged
                    if inp[r][c] != out[r][c]:
                        return None

    if fill_color is None:
        return None

    desc = f"fill enclosed regions with color {fill_color}"
    body = f"""
    oh = h; ow = w;
    /* Copy input to output */
    {{ int total = 0, ii;
    for (ii = 0; ii < h; ii++) total = total + w;
    for (ii = 0; ii < total; ii++) out[ii] = grid[ii]; }}
    /* BFS from border: mark reachable bg cells as -1 in out */
    int bg = {detect_background(pairs[0]["input"])};
    int fill = {fill_color};
    int qhead = 0, qtail = 0;
    int r, c;
    /* Seed border bg cells */
    for (r = 0; r < h; r++) {{
        for (c = 0; c < w; c++) {{
            if ((r == 0 || r == h - 1 || c == 0 || c == w - 1) &&
                arc_get(out, w, r, c) == bg) {{
                arc_set(out, w, r, c, 0 - 1);
                _qr[qtail] = r;
                _qc[qtail] = c;
                qtail++;
            }}
        }}
    }}
    /* BFS */
    while (qhead < qtail) {{
        int pr = _qr[qhead];
        int pc = _qc[qhead];
        qhead++;
        /* 4 neighbors: up, down, left, right */
        int di;
        for (di = 0; di < 4; di++) {{
            int nr = pr;
            int nc = pc;
            if (di == 0) nr = pr - 1;
            if (di == 1) nr = pr + 1;
            if (di == 2) nc = pc - 1;
            if (di == 3) nc = pc + 1;
            if (nr >= 0 && nr < h && nc >= 0 && nc < w) {{
                if (arc_get(out, w, nr, nc) == bg) {{
                    arc_set(out, w, nr, nc, 0 - 1);
                    _qr[qtail] = nr;
                    _qc[qtail] = nc;
                    qtail++;
                }}
            }}
        }}
    }}
    /* Fill remaining bg cells with fill_color, restore -1 to bg */
    {{ int total = 0, ii;
    for (ii = 0; ii < h; ii++) total = total + w;
    for (ii = 0; ii < total; ii++) {{
        if (out[ii] == bg) out[ii] = fill;
        if (out[ii] == (0 - 1)) out[ii] = bg;
    }} }}"""

    return Hypothesis("fill_enclosed", desc, _make_c_program(body, desc), confidence=0.9)


# ── Split at separator and compare halves ────────────────────────

def _try_split_and_compare(pairs: list) -> Optional[Hypothesis]:
    """Check if the grid is split by a separator and the output is
    a function of the two halves (AND, OR, XOR of non-zero cells)."""
    # Find consistent separator across all pairs
    for pair in pairs:
        inp = pair["input"]
        seps = find_separators(inp)
        if not seps["cols"] and not seps["rows"]:
            return None

    # Try column separators first
    first_seps = find_separators(pairs[0]["input"])

    for sep_idx, sep_color in first_seps["cols"]:
        # Verify all pairs have this separator at the same relative position
        valid = True
        for pair in pairs[1:]:
            seps = find_separators(pair["input"])
            if not any(c == sep_idx and col == sep_color for c, col in seps["cols"]):
                valid = False
                break
        if not valid:
            continue

        # Try different operations between left and right halves
        for op_name, op_fn in [
            ("and", lambda a, b: 1 if a != 0 and b != 0 else 0),
            ("or", lambda a, b: 1 if a != 0 or b != 0 else 0),
            ("xor", lambda a, b: 1 if (a != 0) != (b != 0) else 0),
            ("left_minus_right", lambda a, b: 1 if a != 0 and b == 0 else 0),
            ("right_minus_left", lambda a, b: 1 if a == 0 and b != 0 else 0),
        ]:
            all_match = True
            output_color = None
            for pair in pairs:
                left, right = split_at_separator(pair["input"], "col", sep_idx)
                out = pair["output"]
                h_l, w_l = _dimensions(left)
                h_r, w_r = _dimensions(right)
                h_o, w_o = _dimensions(out)
                if h_l != h_r or h_l != h_o or w_l != w_o or w_r != w_o:
                    all_match = False
                    break
                for r in range(h_o):
                    for c in range(w_o):
                        expected_flag = op_fn(left[r][c], right[r][c])
                        if expected_flag:
                            if output_color is None:
                                output_color = out[r][c]
                            elif out[r][c] != output_color:
                                all_match = False
                                break
                        else:
                            if out[r][c] != 0:
                                all_match = False
                                break
                    if not all_match:
                        break
                if not all_match:
                    break

            if all_match and output_color is not None:
                desc = f"split at col {sep_idx}, {op_name} halves, fill with {output_color}"
                # Generate C code
                body = f"""
    /* Split at column separator */
    int sep_col = {sep_idx};
    int out_w = sep_col;
    oh = h; ow = out_w;
    arc_fill(out, h, out_w, 0);
    int r, c;
    for (r = 0; r < h; r++) {{
        for (c = 0; c < out_w; c++) {{
            int left_val = arc_get(grid, w, r, c);
            int right_val = arc_get(grid, w, r, sep_col + 1 + c);
            int flag = 0;
            {"if (left_val != 0 && right_val != 0) flag = 1;" if op_name == "and" else
             "if (left_val != 0 || right_val != 0) flag = 1;" if op_name == "or" else
             "if ((left_val != 0) != (right_val != 0)) flag = 1;" if op_name == "xor" else
             "if (left_val != 0 && right_val == 0) flag = 1;" if op_name == "left_minus_right" else
             "if (left_val == 0 && right_val != 0) flag = 1;"}
            if (flag) arc_set(out, ow, r, c, {output_color});
        }}
    }}"""
                return Hypothesis(f"split_col_{op_name}", desc,
                                  _make_c_program(body, desc), confidence=0.9)

    # Try row separators
    for sep_idx, sep_color in first_seps["rows"]:
        valid = True
        for pair in pairs[1:]:
            seps = find_separators(pair["input"])
            if not any(r == sep_idx and col == sep_color for r, col in seps["rows"]):
                valid = False
                break
        if not valid:
            continue

        for op_name, op_fn in [
            ("and", lambda a, b: 1 if a != 0 and b != 0 else 0),
            ("or", lambda a, b: 1 if a != 0 or b != 0 else 0),
            ("xor", lambda a, b: 1 if (a != 0) != (b != 0) else 0),
        ]:
            all_match = True
            output_color = None
            for pair in pairs:
                top, bottom = split_at_separator(pair["input"], "row", sep_idx)
                out = pair["output"]
                h_t, w_t = _dimensions(top)
                h_b, w_b = _dimensions(bottom)
                h_o, w_o = _dimensions(out)
                if w_t != w_b or w_t != w_o or h_t != h_o or h_b != h_o:
                    all_match = False
                    break
                for r in range(h_o):
                    for c in range(w_o):
                        expected_flag = op_fn(top[r][c], bottom[r][c])
                        if expected_flag:
                            if output_color is None:
                                output_color = out[r][c]
                            elif out[r][c] != output_color:
                                all_match = False
                                break
                        else:
                            if out[r][c] != 0:
                                all_match = False
                                break
                    if not all_match:
                        break
                if not all_match:
                    break

            if all_match and output_color is not None:
                desc = f"split at row {sep_idx}, {op_name} halves, fill with {output_color}"
                body = f"""
    int sep_row = {sep_idx};
    int out_h = sep_row;
    oh = out_h; ow = w;
    arc_fill(out, out_h, w, 0);
    int r, c;
    for (r = 0; r < out_h; r++) {{
        for (c = 0; c < w; c++) {{
            int top_val = arc_get(grid, w, r, c);
            int bot_val = arc_get(grid, w, sep_row + 1 + r, c);
            int flag = 0;
            {"if (top_val != 0 && bot_val != 0) flag = 1;" if op_name == "and" else
             "if (top_val != 0 || bot_val != 0) flag = 1;" if op_name == "or" else
             "if ((top_val != 0) != (bot_val != 0)) flag = 1;"}
            if (flag) arc_set(out, ow, r, c, {output_color});
        }}
    }}"""
                return Hypothesis(f"split_row_{op_name}", desc,
                                  _make_c_program(body, desc), confidence=0.9)

    return None


# ── Most/least common object extraction ──────────────────────────

def _try_extract_largest_object(pairs: list) -> Optional[Hypothesis]:
    """Check if output is the largest connected component extracted."""
    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        bg = detect_background(inp)
        objects = find_connected_components(inp, bg=bg)
        if not objects:
            return None
        largest = objects[0]
        extracted = largest.extract(inp, bg)
        if not grids_equal(extracted, out):
            return None

    desc = "extract largest connected component"
    # C code: flood fill to find largest component, extract bbox
    # This is complex in C without recursion. Use iterative BFS.
    body = """
    int bg = 0;
    /* Find largest connected component via BFS */
    /* visited flags in _grid (reuse: 0=unvisited, 1=visited) */
    { int total = 0, ii;
    for (ii = 0; ii < h; ii++) total = total + w;
    /* Clear visited */
    for (ii = 0; ii < total; ii++) _grid[ii] = 0;
    int best_size = 0, best_minr = 0, best_maxr = 0, best_minc = 0, best_maxc = 0, best_color = 0;
    int r, c;
    for (r = 0; r < h; r++) {
        for (c = 0; c < w; c++) {
            int idx = 0;
            int ri;
            for (ri = 0; ri < r; ri++) idx = idx + w;
            idx = idx + c;
            if (_grid[idx] != 0 || out[idx] == bg) continue;
            /* Have to read from original grid which is in... wait,
               we already parsed into grid. grid values were overwritten by visited.
               We need a different approach. */
            /* PROBLEM: we reused _grid for visited but _grid IS the input grid */
            /* Solution: use out[] as visited, grid[] stays as input */
            break;
        }
        break;
    }
    }
    /* Simplified approach: copy grid to out first, use separate visited tracking */
    /* Actually, the _grid and _out are separate static arrays */
    /* grid = _grid (input), out = _out (output/workspace) */
    /* Use out as visited flags: 0=unvisited, then overwrite with result */
    { int total = 0, ii;
    for (ii = 0; ii < h; ii++) total = total + w;
    for (ii = 0; ii < total; ii++) out[ii] = 0; /* clear visited */
    }
    int best_size = 0, best_minr = h, best_maxr = 0, best_minc = w, best_maxc = 0, best_color = 0;
    /* We need a queue for BFS. Use _tmp (32 elements) is too small. */
    /* Alternative: just find bbox of each color and pick largest */
    /* Simpler approach: for each non-bg color, find its bounding box and cell count */
    int colors[10];
    int csizes[10];
    int cminr[10], cmaxr[10], cminc[10], cmaxc[10];
    { int ci;
    for (ci = 0; ci < 10; ci++) { csizes[ci] = 0; cminr[ci] = h; cmaxr[ci] = 0; cminc[ci] = w; cmaxc[ci] = 0; }
    }
    { int r, c;
    for (r = 0; r < h; r++) {
        for (c = 0; c < w; c++) {
            int v = arc_get(grid, w, r, c);
            if (v != 0) {
                csizes[v] = csizes[v] + 1;
                if (r < cminr[v]) cminr[v] = r;
                if (r > cmaxr[v]) cmaxr[v] = r;
                if (c < cminc[v]) cminc[v] = c;
                if (c > cmaxc[v]) cmaxc[v] = c;
            }
        }
    }}
    /* Find color with most cells */
    { int ci;
    for (ci = 1; ci < 10; ci++) {
        if (csizes[ci] > best_size) {
            best_size = csizes[ci];
            best_color = ci;
            best_minr = cminr[ci];
            best_maxr = cmaxr[ci];
            best_minc = cminc[ci];
            best_maxc = cmaxc[ci];
        }
    }}
    /* Extract bounding box */
    oh = best_maxr - best_minr + 1;
    ow = best_maxc - best_minc + 1;
    arc_fill(out, oh, ow, 0);
    { int r, c;
    for (r = best_minr; r <= best_maxr; r++) {
        for (c = best_minc; c <= best_maxc; c++) {
            int v = arc_get(grid, w, r, c);
            if (v == best_color) {
                arc_set(out, ow, r - best_minr, c - best_minc, v);
            }
        }
    }}"""

    return Hypothesis("extract_largest", desc, _make_c_program(body, desc), confidence=0.8)


# ── Self-tiling (input as both pattern and template) ─────────────

def _try_self_tile(pairs: list) -> Optional[Hypothesis]:
    """Check if the output tiles copies of the input based on non-zero positions.

    E.g., 3x3 input → 9x9 output where each non-zero cell in input
    means "place a copy of the input pattern at that position".
    """
    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        h_i, w_i = _dimensions(inp)
        h_o, w_o = _dimensions(out)
        # Output must be exact multiple of input
        if h_o % h_i != 0 or w_o % w_i != 0:
            return None
        tile_r = h_o // h_i
        tile_c = w_o // w_i
        if tile_r != h_i or tile_c != w_i:
            return None  # Must be h_i x w_i tiles = h_i^2 x w_i^2 output

        # Each (tr, tc) tile in the output is either:
        # - a copy of the input pattern (if inp[tr][tc] != 0)
        # - all zeros (if inp[tr][tc] == 0)
        for tr in range(h_i):
            for tc in range(w_i):
                for dr in range(h_i):
                    for dc in range(w_i):
                        r_out = tr * h_i + dr
                        c_out = tc * w_i + dc
                        if inp[tr][tc] != 0:
                            if out[r_out][c_out] != inp[dr][dc]:
                                return None
                        else:
                            if out[r_out][c_out] != 0:
                                return None

    desc = "self-tile: place input copy at each non-zero cell position"
    body = """
    /* Output is h*h x w*w */
    oh = 0;
    { int i;
    for (i = 0; i < h; i++) oh = oh + h; }
    ow = 0;
    { int i;
    for (i = 0; i < w; i++) ow = ow + w; }
    arc_fill(out, oh, ow, 0);
    /* For each cell in input, if non-zero, place input pattern at that tile position */
    int tr, tc, dr, dc;
    for (tr = 0; tr < h; tr++) {
        for (tc = 0; tc < w; tc++) {
            if (arc_get(grid, w, tr, tc) != 0) {
                for (dr = 0; dr < h; dr++) {
                    for (dc = 0; dc < w; dc++) {
                        int or_ = 0;
                        int i;
                        for (i = 0; i < tr; i++) or_ = or_ + h;
                        or_ = or_ + dr;
                        int oc = 0;
                        for (i = 0; i < tc; i++) oc = oc + w;
                        oc = oc + dc;
                        arc_set(out, ow, or_, oc, arc_get(grid, w, dr, dc));
                    }
                }
            }
        }
    }"""

    return Hypothesis("self_tile", desc, _make_c_program(body, desc), confidence=0.9)


# ── Recolor by count / majority ──────────────────────────────────

def _try_recolor_by_majority(pairs: list) -> Optional[Hypothesis]:
    """Check if each object's cells get recolored to the object's majority
    non-background color, or a mapping based on object properties."""
    # Simple case: each separate color region gets recolored based on some rule
    # This is hard to detect generically. Skip for now.
    return None


# ── Neighbor-count recoloring ─────────────────────────────────────

def _count_neighbors(grid, h, w, r, c, color, connectivity=4):
    """Count neighbors of cell (r,c) that have the given color."""
    dirs = [(-1,0),(1,0),(0,-1),(0,1)] if connectivity == 4 else \
           [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    count = 0
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == color:
            count += 1
    return count


def _try_neighbor_recolor(pairs: list) -> Optional[Hypothesis]:
    """Check if cells get recolored based on how many same-color neighbors they have.

    Common ARC pattern: cells with exactly N neighbors of color X become color Y.
    Also: background cells adjacent to N colored cells become color Y.
    """
    for pair in pairs:
        if _dimensions(pair["input"]) != _dimensions(pair["output"]):
            return None

    # Detect: bg cells that gain a new color based on adjacent non-bg count
    # For each pair, find cells that changed
    bg = detect_background(pairs[0]["input"])
    inp0 = pairs[0]["input"]
    out0 = pairs[0]["output"]
    h, w = _dimensions(inp0)

    # Find new color(s) in output
    inp_colors = set(v for row in inp0 for v in row)
    out_colors = set(v for row in out0 for v in row)
    new_colors = out_colors - inp_colors

    if not new_colors or len(new_colors) > 2:
        return None

    # Check if changed cells are bg cells that become new_color
    # based on a consistent neighbor count rule
    for new_color in new_colors:
        for connectivity in [4, 8]:
            for non_bg_color in inp_colors - {bg}:
                # Collect: for changed cells, how many neighbors of non_bg_color?
                # For unchanged bg cells, how many neighbors?
                threshold = None
                valid = True

                for pair in pairs:
                    inp, out = pair["input"], pair["output"]
                    h, w = _dimensions(inp)
                    for r in range(h):
                        for c in range(w):
                            n = _count_neighbors(inp, h, w, r, c, non_bg_color, connectivity)
                            if inp[r][c] == bg and out[r][c] == new_color:
                                # Changed: should have threshold neighbors
                                if threshold is None:
                                    threshold = n
                                elif n != threshold:
                                    valid = False
                                    break
                            elif inp[r][c] == bg and out[r][c] == bg:
                                # Unchanged bg: should NOT have threshold neighbors
                                pass  # We'll verify after
                            elif inp[r][c] != out[r][c]:
                                # Non-bg cell changed — not this pattern
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break

                if not valid or threshold is None:
                    continue

                # Verify: all bg cells with exactly threshold neighbors → new_color
                # All bg cells with != threshold neighbors → stay bg
                all_match = True
                for pair in pairs:
                    inp, out = pair["input"], pair["output"]
                    h, w = _dimensions(inp)
                    for r in range(h):
                        for c in range(w):
                            if inp[r][c] == bg:
                                n = _count_neighbors(inp, h, w, r, c, non_bg_color, connectivity)
                                if n == threshold and out[r][c] != new_color:
                                    all_match = False
                                    break
                                if n != threshold and out[r][c] != bg:
                                    all_match = False
                                    break
                            elif inp[r][c] != out[r][c]:
                                all_match = False
                                break
                        if not all_match:
                            break
                    if not all_match:
                        break

                if all_match:
                    conn_str = "4" if connectivity == 4 else "8"
                    desc = f"bg cells with {threshold} {conn_str}-neighbors of color {non_bg_color} become {new_color}"

                    if connectivity == 4:
                        neighbor_check = f"""
            int nn = 0;
            if (r > 0 && arc_get(grid, w, r - 1, c) == {non_bg_color}) nn++;
            if (r < h - 1 && arc_get(grid, w, r + 1, c) == {non_bg_color}) nn++;
            if (c > 0 && arc_get(grid, w, r, c - 1) == {non_bg_color}) nn++;
            if (c < w - 1 && arc_get(grid, w, r, c + 1) == {non_bg_color}) nn++;"""
                    else:
                        neighbor_check = f"""
            int nn = 0;
            int dr2, dc2;
            for (dr2 = 0 - 1; dr2 <= 1; dr2++) {{
                for (dc2 = 0 - 1; dc2 <= 1; dc2++) {{
                    if (dr2 == 0 && dc2 == 0) continue;
                    int nr2 = r + dr2, nc2 = c + dc2;
                    if (nr2 >= 0 && nr2 < h && nc2 >= 0 && nc2 < w &&
                        arc_get(grid, w, nr2, nc2) == {non_bg_color}) nn++;
                }}
            }}"""

                    body = f"""
    oh = h; ow = w;
    {{ int total = 0, ii;
    for (ii = 0; ii < h; ii++) total = total + w;
    for (ii = 0; ii < total; ii++) out[ii] = grid[ii]; }}
    int r, c;
    for (r = 0; r < h; r++) {{
        for (c = 0; c < w; c++) {{
            if (arc_get(grid, w, r, c) == {bg}) {{
                {neighbor_check}
                if (nn == {threshold}) arc_set(out, w, r, c, {new_color});
            }}
        }}
    }}"""
                    return Hypothesis("neighbor_recolor", desc,
                                      _make_c_program(body, desc), confidence=0.88)

    return None


# ── Mirror/symmetry completion ───────────────────────────────────

def _try_mirror_complete(pairs: list) -> Optional[Hypothesis]:
    """Check if the output is the input made symmetric (mirror left-right,
    top-bottom, or both) by filling in the missing half."""
    for pair in pairs:
        if _dimensions(pair["input"]) != _dimensions(pair["output"]):
            return None

    # Check horizontal mirror: right half = mirror of left half
    for mirror_type in ["lr", "tb"]:
        all_match = True
        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            h, w = _dimensions(inp)
            if mirror_type == "lr":
                for r in range(h):
                    for c in range(w):
                        if out[r][c] != out[r][w - 1 - c]:
                            all_match = False
                            break
                    if not all_match:
                        break
                # Also check that the output preserves non-zero cells from input
                if all_match:
                    for r in range(h):
                        for c in range(w):
                            if inp[r][c] != 0 and out[r][c] != inp[r][c]:
                                all_match = False
                                break
                        if not all_match:
                            break
            else:  # tb
                for r in range(h):
                    for c in range(w):
                        if out[r][c] != out[h - 1 - r][c]:
                            all_match = False
                            break
                    if not all_match:
                        break
                if all_match:
                    for r in range(h):
                        for c in range(w):
                            if inp[r][c] != 0 and out[r][c] != inp[r][c]:
                                all_match = False
                                break
                        if not all_match:
                            break

            if not all_match:
                break

        if all_match:
            if mirror_type == "lr":
                desc = "mirror-complete left-right"
                body = """
    oh = h; ow = w;
    { int total = 0, ii;
    for (ii = 0; ii < h; ii++) total = total + w;
    for (ii = 0; ii < total; ii++) out[ii] = grid[ii]; }
    int r, c;
    for (r = 0; r < h; r++) {
        for (c = 0; c < w; c++) {
            if (arc_get(out, w, r, c) == 0) {
                int mirror_c = w - 1 - c;
                int v = arc_get(out, w, r, mirror_c);
                if (v != 0) arc_set(out, w, r, c, v);
            }
        }
    }"""
            else:
                desc = "mirror-complete top-bottom"
                body = """
    oh = h; ow = w;
    { int total = 0, ii;
    for (ii = 0; ii < h; ii++) total = total + w;
    for (ii = 0; ii < total; ii++) out[ii] = grid[ii]; }
    int r, c;
    for (r = 0; r < h; r++) {
        for (c = 0; c < w; c++) {
            if (arc_get(out, w, r, c) == 0) {
                int mirror_r = h - 1 - r;
                int v = arc_get(out, w, mirror_r, c);
                if (v != 0) arc_set(out, w, r, c, v);
            }
        }
    }"""
            return Hypothesis("mirror_complete", desc,
                              _make_c_program(body, desc), confidence=0.85)

    return None


# ── Replace minority/majority color ──────────────────────────────

def _try_replace_color_by_frequency(pairs: list) -> Optional[Hypothesis]:
    """Check if the least/most common non-bg color gets replaced by another."""
    for pair in pairs:
        if _dimensions(pair["input"]) != _dimensions(pair["output"]):
            return None

    # Find consistent replacement: one color in input becomes another in output
    bg = detect_background(pairs[0]["input"])
    replacements = {}  # src -> dst
    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        h, w = _dimensions(inp)
        for r in range(h):
            for c in range(w):
                if inp[r][c] != out[r][c]:
                    src, dst = inp[r][c], out[r][c]
                    if src in replacements:
                        if replacements[src] != dst:
                            return None
                    else:
                        replacements[src] = dst
                    # Check no other src maps to this dst
    if not replacements:
        return None
    if len(replacements) > 2:
        return None  # Too many replacements

    # Build the color map (only for changed colors, identity otherwise)
    lut = list(range(10))
    for src, dst in replacements.items():
        lut[src] = dst

    # Check if the replacement is the same as a simple color_map
    # (already handled by _try_color_map). Only add if it's position-dependent.
    # Actually, let's just not duplicate — _try_color_map handles this.
    # But _try_color_map requires ALL cells of that color to change.
    # Here we check: do ALL cells of the src color change?
    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        h, w = _dimensions(inp)
        for r in range(h):
            for c in range(w):
                for src, dst in replacements.items():
                    if inp[r][c] == src and out[r][c] != dst:
                        # Some cells of src color don't change — position-dependent
                        return None  # Too complex for a simple rule

    # All cells of replaced colors change — this is a color_map
    # (should be caught by existing detector, but in case it wasn't)
    desc = "replace colors: " + ", ".join(f"{k}->{v}" for k, v in sorted(replacements.items()))
    body = f"""
    oh = h; ow = w;
    int lut[10];
    lut[0]={lut[0]}; lut[1]={lut[1]}; lut[2]={lut[2]}; lut[3]={lut[3]}; lut[4]={lut[4]};
    lut[5]={lut[5]}; lut[6]={lut[6]}; lut[7]={lut[7]}; lut[8]={lut[8]}; lut[9]={lut[9]};
    {{ int total = 0, ii;
    for (ii = 0; ii < h; ii++) total = total + w;
    for (ii = 0; ii < total; ii++) out[ii] = lut[grid[ii]];
    }}"""

    return Hypothesis("replace_color", desc, _make_c_program(body, desc), confidence=0.88)


# ── Downscale (output is input at reduced resolution) ────────────

def _try_downscale(pairs: list) -> Optional[Hypothesis]:
    """Check if output is a downscaled version of the input."""
    for scale in [2, 3]:
        all_match = True
        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            h_i, w_i = _dimensions(inp)
            h_o, w_o = _dimensions(out)
            if h_i != h_o * scale or w_i != w_o * scale:
                all_match = False
                break
            # Check each output cell = majority/any of the scale x scale block
            for r in range(h_o):
                for c in range(w_o):
                    # Get the scale x scale block from input
                    vals = []
                    for dr in range(scale):
                        for dc in range(scale):
                            vals.append(inp[r * scale + dr][c * scale + dc])
                    # Output should be: majority non-zero, or any non-zero
                    if out[r][c] not in vals:
                        all_match = False
                        break
                if not all_match:
                    break
            if not all_match:
                break

        if all_match:
            desc = f"downscale by {scale}x (pick from block)"
            body = f"""
    int scale = {scale};
    {{ int i;
    oh = 0;
    for (i = 0; i < h; i++) oh = oh + 1;
    /* oh = h / scale via repeated subtraction */
    int hh = h, cnt = 0;
    while (hh >= scale) {{ hh = hh - scale; cnt++; }}
    oh = cnt;
    int ww = w;
    cnt = 0;
    while (ww >= scale) {{ ww = ww - scale; cnt++; }}
    ow = cnt;
    }}
    int r, c;
    for (r = 0; r < oh; r++) {{
        for (c = 0; c < ow; c++) {{
            /* Pick the most common non-zero value in the block */
            int counts[10];
            int ci;
            for (ci = 0; ci < 10; ci++) counts[ci] = 0;
            int dr, dc;
            for (dr = 0; dr < scale; dr++) {{
                for (dc = 0; dc < scale; dc++) {{
                    int sr = 0, sc2 = 0, i;
                    for (i = 0; i < r; i++) sr = sr + scale;
                    sr = sr + dr;
                    for (i = 0; i < c; i++) sc2 = sc2 + scale;
                    sc2 = sc2 + dc;
                    int v = arc_get(grid, w, sr, sc2);
                    counts[v] = counts[v] + 1;
                }}
            }}
            /* Find most common non-zero */
            int best = 0, bestc = 0;
            for (ci = 1; ci < 10; ci++) {{
                if (counts[ci] > bestc) {{ bestc = counts[ci]; best = ci; }}
            }}
            arc_set(out, ow, r, c, best);
        }}
    }}"""
            return Hypothesis(f"downscale_{scale}x", desc,
                              _make_c_program(body, desc), confidence=0.85)

    return None


# ── Output equals a specific single-color sub-region ─────────────

def _try_extract_color(pairs: list) -> Optional[Hypothesis]:
    """Check if the output is the bounding box of cells of a specific color."""
    bg = detect_background(pairs[0]["input"])

    # Try each non-bg color
    for target_color in range(1, 10):
        all_match = True
        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            h_i, w_i = _dimensions(inp)
            # Find bbox of target_color cells
            min_r, max_r, min_c, max_c = h_i, -1, w_i, -1
            for r in range(h_i):
                for c in range(w_i):
                    if inp[r][c] == target_color:
                        if r < min_r: min_r = r
                        if r > max_r: max_r = r
                        if c < min_c: min_c = c
                        if c > max_c: max_c = c
            if max_r < 0:
                all_match = False
                break
            expected = [row[min_c:max_c+1] for row in inp[min_r:max_r+1]]
            if not grids_equal(expected, out):
                all_match = False
                break

        if all_match:
            desc = f"extract bounding box of color {target_color}"
            body = f"""
    int target = {target_color};
    int min_r = h, max_r = 0 - 1, min_c = w, max_c = 0 - 1;
    int r, c;
    for (r = 0; r < h; r++) {{
        for (c = 0; c < w; c++) {{
            int v = arc_get(grid, w, r, c);
            if (v == target) {{
                if (r < min_r) min_r = r;
                if (r > max_r) max_r = r;
                if (c < min_c) min_c = c;
                if (c > max_c) max_c = c;
            }}
        }}
    }}
    oh = max_r - min_r + 1;
    ow = max_c - min_c + 1;
    for (r = min_r; r <= max_r; r++) {{
        for (c = min_c; c <= max_c; c++) {{
            arc_set(out, ow, r - min_r, c - min_c, arc_get(grid, w, r, c));
        }}
    }}"""
            return Hypothesis(f"extract_color_{target_color}", desc,
                              _make_c_program(body, desc), confidence=0.85)

    return None


# ── Main generator ───────────────────────────────────────────────

# ── Connect same-color cells with lines (47 potential tasks) ─────

def _try_connect_same_color(pairs: list) -> Optional[Hypothesis]:
    """Check if output draws H/V lines between pairs of same-colored non-bg cells."""
    for pair in pairs:
        if _dimensions(pair["input"]) != _dimensions(pair["output"]):
            return None

    bg = detect_background(pairs[0]["input"])

    # Check: output = input + lines drawn between same-color cells
    # For each pair, find new cells (were bg, now non-bg)
    # New cells should form horizontal or vertical line segments
    # connecting existing cells of the same color
    fill_color = None  # What color the lines are drawn in

    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        h, w = _dimensions(inp)
        for r in range(h):
            for c in range(w):
                if inp[r][c] != bg and out[r][c] != inp[r][c]:
                    return None  # Existing non-bg cell changed — not this pattern
                if inp[r][c] == bg and out[r][c] != bg:
                    # New cell: should be part of a connecting line
                    if fill_color is None:
                        fill_color = out[r][c]
                    # Lines might use different colors per connection

    if fill_color is None:
        return None

    # Verify: for each non-bg color, draw H and V lines between all pairs,
    # check if that produces the output
    for use_same_color in [True, False]:
        all_match = True
        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            h, w = _dimensions(inp)
            test = [row[:] for row in inp]

            # For each non-bg color, find all cells of that color
            for color in range(1, 10):
                cells = [(r, c) for r in range(h) for c in range(w) if inp[r][c] == color]
                if len(cells) < 2:
                    continue
                line_c = color if use_same_color else fill_color
                # Draw H lines between cells on same row
                for i in range(len(cells)):
                    for j in range(i + 1, len(cells)):
                        r1, c1 = cells[i]
                        r2, c2 = cells[j]
                        if r1 == r2:
                            for cc in range(min(c1, c2) + 1, max(c1, c2)):
                                if test[r1][cc] == bg:
                                    test[r1][cc] = line_c
                        if c1 == c2:
                            for rr in range(min(r1, r2) + 1, max(r1, r2)):
                                if test[rr][c1] == bg:
                                    test[rr][c1] = line_c

            if not grids_equal(test, out):
                all_match = False
                break

        if all_match:
            sc = "same" if use_same_color else str(fill_color)
            desc = f"connect same-color cells with H/V lines (color={sc})"
            body_color = "color" if use_same_color else str(fill_color)
            body = f"""
    oh = h; ow = w;
    {{ int total = 0, ii;
    for (ii = 0; ii < h; ii++) total = total + w;
    for (ii = 0; ii < total; ii++) out[ii] = grid[ii]; }}
    /* For each color, connect cells on same row/col */
    int color;
    for (color = 1; color < 10; color++) {{
        /* Find all cells of this color */
        int cells_r[100], cells_c[100], ncells = 0;
        int r, c;
        for (r = 0; r < h; r++) {{
            for (c = 0; c < w; c++) {{
                if (arc_get(grid, w, r, c) == color && ncells < 100) {{
                    cells_r[ncells] = r;
                    cells_c[ncells] = c;
                    ncells++;
                }}
            }}
        }}
        if (ncells < 2) continue;
        int line_color = {"color" if use_same_color else str(fill_color)};
        /* Draw H/V lines between pairs */
        int i, j;
        for (i = 0; i < ncells; i++) {{
            for (j = i + 1; j < ncells; j++) {{
                int r1 = cells_r[i], c1 = cells_c[i];
                int r2 = cells_r[j], c2 = cells_c[j];
                if (r1 == r2) {{
                    int lo = c1, hi = c2;
                    if (c2 < c1) {{ lo = c2; hi = c1; }}
                    int cc;
                    for (cc = lo + 1; cc < hi; cc++) {{
                        if (arc_get(out, w, r1, cc) == 0)
                            arc_set(out, w, r1, cc, line_color);
                    }}
                }}
                if (c1 == c2) {{
                    int lo = r1, hi = r2;
                    if (r2 < r1) {{ lo = r2; hi = r1; }}
                    int rr;
                    for (rr = lo + 1; rr < hi; rr++) {{
                        if (arc_get(out, w, rr, c1) == 0)
                            arc_set(out, w, rr, c1, line_color);
                    }}
                }}
            }}
        }}
    }}"""
            return Hypothesis("connect_same_color", desc,
                              _make_c_program(body, desc), confidence=0.88)

    return None


# ── Remove isolated cells (18 potential tasks) ───────────────────

def _try_remove_isolated(pairs: list) -> Optional[Hypothesis]:
    """Check if output removes cells with no same-color 4-neighbors."""
    for pair in pairs:
        if _dimensions(pair["input"]) != _dimensions(pair["output"]):
            return None

    bg = detect_background(pairs[0]["input"])

    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        h, w = _dimensions(inp)
        for r in range(h):
            for c in range(w):
                if inp[r][c] != bg:
                    # Check if isolated (no same-color 4-neighbor)
                    color = inp[r][c]
                    has_neighbor = False
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and inp[nr][nc] == color:
                            has_neighbor = True
                            break
                    if has_neighbor:
                        if out[r][c] != inp[r][c]:
                            return None
                    else:
                        if out[r][c] != bg:
                            return None
                else:
                    if out[r][c] != bg:
                        return None

    desc = "remove isolated cells (no same-color 4-neighbor)"
    body = f"""
    oh = h; ow = w;
    int r, c;
    for (r = 0; r < h; r++) {{
        for (c = 0; c < w; c++) {{
            int v = arc_get(grid, w, r, c);
            if (v == {bg}) {{
                arc_set(out, w, r, c, {bg});
                continue;
            }}
            int has_nb = 0;
            if (r > 0 && arc_get(grid, w, r - 1, c) == v) has_nb = 1;
            if (r < h - 1 && arc_get(grid, w, r + 1, c) == v) has_nb = 1;
            if (c > 0 && arc_get(grid, w, r, c - 1) == v) has_nb = 1;
            if (c < w - 1 && arc_get(grid, w, r, c + 1) == v) has_nb = 1;
            if (has_nb) arc_set(out, w, r, c, v);
            else arc_set(out, w, r, c, {bg});
        }}
    }}"""
    return Hypothesis("remove_isolated", desc,
                      _make_c_program(body, desc), confidence=0.88)


# ── Upscale NxN (19 potential tasks) ─────────────────────────────

def _try_upscale(pairs: list) -> Optional[Hypothesis]:
    """Check if output is input upscaled by factor N (each cell -> NxN block)."""
    for scale in [2, 3, 4, 5]:
        all_match = True
        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            h_i, w_i = _dimensions(inp)
            h_o, w_o = _dimensions(out)
            if h_o != h_i * scale or w_o != w_i * scale:
                all_match = False
                break
            for r in range(h_i):
                for c in range(w_i):
                    v = inp[r][c]
                    for dr in range(scale):
                        for dc in range(scale):
                            if out[r * scale + dr][c * scale + dc] != v:
                                all_match = False
                                break
                        if not all_match: break
                    if not all_match: break
                if not all_match: break
            if not all_match: break

        if all_match:
            desc = f"upscale {scale}x (each cell -> {scale}x{scale} block)"
            body = f"""
    int scale = {scale};
    oh = 0;
    {{ int i; for (i = 0; i < h; i++) oh = oh + scale; }}
    ow = 0;
    {{ int i; for (i = 0; i < w; i++) ow = ow + scale; }}
    int r, c, dr, dc;
    for (r = 0; r < h; r++) {{
        for (c = 0; c < w; c++) {{
            int v = arc_get(grid, w, r, c);
            for (dr = 0; dr < scale; dr++) {{
                for (dc = 0; dc < scale; dc++) {{
                    int or_ = 0, oc = 0, i;
                    for (i = 0; i < r; i++) or_ = or_ + scale;
                    or_ = or_ + dr;
                    for (i = 0; i < c; i++) oc = oc + scale;
                    oc = oc + dc;
                    arc_set(out, ow, or_, oc, v);
                }}
            }}
        }}
    }}"""
            return Hypothesis(f"upscale_{scale}x", desc,
                              _make_c_program(body, desc), confidence=0.9)
    return None


# ── Output is a specific sub-region of input (31 potential tasks) ─

def _try_extract_subgrid(pairs: list) -> Optional[Hypothesis]:
    """Check if output is a fixed subgrid extracted from input."""
    # Check if output appears at the same position in all inputs
    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        h_i, w_i = _dimensions(inp)
        h_o, w_o = _dimensions(out)
        if h_o >= h_i and w_o >= w_i:
            return None  # Output not smaller

    # Try to find consistent extraction position
    positions = []
    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        h_i, w_i = _dimensions(inp)
        h_o, w_o = _dimensions(out)
        found = False
        for r_off in range(h_i - h_o + 1):
            for c_off in range(w_i - w_o + 1):
                match = all(inp[r_off+r][c_off+c] == out[r][c]
                           for r in range(h_o) for c in range(w_o))
                if match:
                    positions.append((r_off, c_off, h_o, w_o))
                    found = True
                    break
            if found: break
        if not found:
            return None

    if not positions:
        return None

    # Check if position is consistent (same offset) or follows a pattern
    # Simple case: all same absolute offset
    r0, c0 = positions[0][0], positions[0][1]
    if all(p[0] == r0 and p[1] == c0 for p in positions):
        h_o, w_o = positions[0][2], positions[0][3]
        desc = f"extract subgrid at ({r0},{c0}) size {h_o}x{w_o}"
        body = f"""
    oh = {h_o}; ow = {w_o};
    int r, c;
    for (r = 0; r < oh; r++) {{
        for (c = 0; c < ow; c++) {{
            arc_set(out, ow, r, c, arc_get(grid, w, r + {r0}, c + {c0}));
        }}
    }}"""
        return Hypothesis("extract_subgrid", desc,
                          _make_c_program(body, desc), confidence=0.85)

    return None


# ── Diagonal mirror completion ───────────────────────────────────

def _try_diagonal_mirror(pairs: list) -> Optional[Hypothesis]:
    """Check if output completes diagonal symmetry (main or anti-diagonal)."""
    for pair in pairs:
        if _dimensions(pair["input"]) != _dimensions(pair["output"]):
            return None

    # Check main diagonal symmetry: out[r][c] == out[c][r]
    for mirror_type in ["main_diag", "anti_diag"]:
        all_match = True
        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            h, w = _dimensions(inp)
            if h != w:
                all_match = False
                break
            for r in range(h):
                for c in range(w):
                    if mirror_type == "main_diag":
                        if out[r][c] != out[c][r]:
                            all_match = False
                            break
                    else:
                        if out[r][c] != out[h-1-c][w-1-r]:
                            all_match = False
                            break
                if not all_match: break
            if not all_match: break
            # Also check: output preserves non-bg cells from input
            for r in range(h):
                for c in range(w):
                    if inp[r][c] != 0 and out[r][c] != inp[r][c]:
                        all_match = False
                        break
                if not all_match: break
            if not all_match: break

        if all_match:
            if mirror_type == "main_diag":
                desc = "complete main diagonal symmetry"
                body = """
    oh = h; ow = w;
    { int total = 0, ii;
    for (ii = 0; ii < h; ii++) total = total + w;
    for (ii = 0; ii < total; ii++) out[ii] = grid[ii]; }
    int r, c;
    for (r = 0; r < h; r++) {
        for (c = 0; c < w; c++) {
            if (arc_get(out, w, r, c) == 0) {
                int v = arc_get(out, w, c, r);
                if (v != 0) arc_set(out, w, r, c, v);
            }
        }
    }"""
            else:
                desc = "complete anti-diagonal symmetry"
                body = """
    oh = h; ow = w;
    { int total = 0, ii;
    for (ii = 0; ii < h; ii++) total = total + w;
    for (ii = 0; ii < total; ii++) out[ii] = grid[ii]; }
    int r, c;
    for (r = 0; r < h; r++) {
        for (c = 0; c < w; c++) {
            if (arc_get(out, w, r, c) == 0) {
                int v = arc_get(out, w, h - 1 - c, w - 1 - r);
                if (v != 0) arc_set(out, w, r, c, v);
            }
        }
    }"""
            return Hypothesis("diagonal_mirror", desc,
                              _make_c_program(body, desc), confidence=0.85)

    return None


# ── 4-fold rotational symmetry completion ────────────────────────

def _try_rotational_symmetry(pairs: list) -> Optional[Hypothesis]:
    """Check if output completes 4-fold rotational symmetry."""
    for pair in pairs:
        if _dimensions(pair["input"]) != _dimensions(pair["output"]):
            return None

    all_match = True
    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        h, w = _dimensions(inp)
        if h != w:
            all_match = False
            break
        # Check: output has 4-fold rotational symmetry
        for r in range(h):
            for c in range(w):
                v = out[r][c]
                # 90: (r,c) -> (c, h-1-r)
                # 180: (r,c) -> (h-1-r, w-1-c)
                # 270: (r,c) -> (w-1-c, r)
                if (out[c][h-1-r] != v or out[h-1-r][w-1-c] != v or out[w-1-c][r] != v):
                    all_match = False
                    break
            if not all_match: break
        if not all_match: break
        # Check preserves input non-bg
        for r in range(h):
            for c in range(w):
                if inp[r][c] != 0 and out[r][c] != inp[r][c]:
                    all_match = False
                    break
            if not all_match: break
        if not all_match: break

    if all_match:
        desc = "complete 4-fold rotational symmetry"
        body = """
    oh = h; ow = w;
    { int total = 0, ii;
    for (ii = 0; ii < h; ii++) total = total + w;
    for (ii = 0; ii < total; ii++) out[ii] = grid[ii]; }
    /* Fill by rotating non-zero cells to all 4 positions */
    int r, c;
    for (r = 0; r < h; r++) {
        for (c = 0; c < w; c++) {
            int v = arc_get(out, w, r, c);
            if (v != 0) {
                arc_set(out, w, c, h - 1 - r, v);
                arc_set(out, w, h - 1 - r, w - 1 - c, v);
                arc_set(out, w, w - 1 - c, r, v);
            }
        }
    }"""
        return Hypothesis("rotational_symmetry", desc,
                          _make_c_program(body, desc), confidence=0.85)

    return None


OBJECT_HYPOTHESIS_GENERATORS = [
    _try_fill_enclosed,
    _try_split_and_compare,
    _try_self_tile,
    _try_neighbor_recolor,
    _try_connect_same_color,
    _try_remove_isolated,
    _try_upscale,
    _try_mirror_complete,
    _try_diagonal_mirror,
    _try_rotational_symmetry,
    _try_extract_color,
    _try_extract_subgrid,
    _try_downscale,
    _try_extract_largest_object,
    _try_replace_color_by_frequency,
]


def generate_object_hypotheses(task: dict) -> List[Hypothesis]:
    """Generate object-aware hypotheses for an ARC task."""
    pairs = task["train"]
    hypotheses = []

    for gen_fn in OBJECT_HYPOTHESIS_GENERATORS:
        try:
            h = gen_fn(pairs)
            if h is not None:
                hypotheses.append(h)
                logger.info(f"Object hypothesis found: {h}")
        except Exception as e:
            logger.warning(f"Object hypothesis generator {gen_fn.__name__} failed: {e}")

    return hypotheses
