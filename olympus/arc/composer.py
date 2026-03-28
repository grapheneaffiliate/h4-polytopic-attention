"""
Composition Engine — chain primitive transforms to solve harder ARC puzzles.

For each pair (A, B) of primitives, applies A then B in Python to check
if the composition matches training outputs. Only verified compositions
get compiled to C via transformer-vm.

This is pure mechanical search: no LLM, no heuristics. The verifier is
the oracle. Each new composition that works becomes a permanent tool.
"""

import logging
from collections import Counter
from itertools import product
from typing import List, Optional, Tuple, Callable

from .grid_io import Grid, grids_equal
from .hypothesizer import (
    Hypothesis, _make_c_program, _dimensions, _flatten, _color_histogram,
    _rotate_90, _rotate_180, _rotate_270, _flip_h, _flip_v, _transpose,
    ARC_HEADER,
)

logger = logging.getLogger(__name__)


# ── Primitive transforms (Python-side) ───────────────────────────

def _scale_2x(grid: Grid) -> Grid:
    h, w = _dimensions(grid)
    out = [[0] * (w * 2) for _ in range(h * 2)]
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            out[r*2][c*2] = v
            out[r*2][c*2+1] = v
            out[r*2+1][c*2] = v
            out[r*2+1][c*2+1] = v
    return out


def _scale_3x(grid: Grid) -> Grid:
    h, w = _dimensions(grid)
    out = [[0] * (w * 3) for _ in range(h * 3)]
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            for dr in range(3):
                for dc in range(3):
                    out[r*3+dr][c*3+dc] = v
    return out


def _crop_nonzero(grid: Grid) -> Optional[Grid]:
    h, w = _dimensions(grid)
    if h == 0 or w == 0:
        return None
    min_r, max_r, min_c, max_c = h, -1, w, -1
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                if r < min_r: min_r = r
                if r > max_r: max_r = r
                if c < min_c: min_c = c
                if c > max_c: max_c = c
    if max_r < 0:
        return None
    return [grid[r][min_c:max_c+1] for r in range(min_r, max_r+1)]


def _gravity_down(grid: Grid) -> Grid:
    h, w = _dimensions(grid)
    out = [[0] * w for _ in range(h)]
    for c in range(w):
        vals = [grid[r][c] for r in range(h) if grid[r][c] != 0]
        start = h - len(vals)
        for i, v in enumerate(vals):
            out[start + i][c] = v
    return out


def _gravity_up(grid: Grid) -> Grid:
    h, w = _dimensions(grid)
    out = [[0] * w for _ in range(h)]
    for c in range(w):
        vals = [grid[r][c] for r in range(h) if grid[r][c] != 0]
        for i, v in enumerate(vals):
            out[i][c] = v
    return out


def _gravity_left(grid: Grid) -> Grid:
    h, w = _dimensions(grid)
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        vals = [grid[r][c] for c in range(w) if grid[r][c] != 0]
        for i, v in enumerate(vals):
            out[r][i] = v
    return out


def _gravity_right(grid: Grid) -> Grid:
    h, w = _dimensions(grid)
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        vals = [grid[r][c] for c in range(w) if grid[r][c] != 0]
        start = w - len(vals)
        for i, v in enumerate(vals):
            out[r][start + i] = v
    return out


def _invert_colors(grid: Grid) -> Grid:
    """Map each color c to 9-c (swap 0<->9, 1<->8, etc.)."""
    return [[9 - v for v in row] for row in grid]


def _replace_zero_with_most_common(grid: Grid) -> Grid:
    """Replace 0 with the most common non-zero color."""
    flat = _flatten(grid)
    nonzero = [v for v in flat if v != 0]
    if not nonzero:
        return grid
    most_common = Counter(nonzero).most_common(1)[0][0]
    return [[most_common if v == 0 else v for v in row] for row in grid]


def _keep_only_nonzero(grid: Grid) -> Grid:
    """Same as identity for grids — but useful conceptually."""
    return grid


# ── Dynamic color map detection ──────────────────────────────────

def _detect_color_map(input_grid: Grid, output_grid: Grid) -> Optional[dict]:
    """Detect a per-cell color mapping between two same-sized grids."""
    h_i, w_i = _dimensions(input_grid)
    h_o, w_o = _dimensions(output_grid)
    if h_i != h_o or w_i != w_o:
        return None
    mapping = {}
    for r in range(h_i):
        for c in range(w_i):
            src, dst = input_grid[r][c], output_grid[r][c]
            if src in mapping:
                if mapping[src] != dst:
                    return None
            else:
                mapping[src] = dst
    return mapping


def _apply_color_map(grid: Grid, mapping: dict) -> Grid:
    """Apply a color mapping to a grid."""
    return [[mapping.get(v, v) for v in row] for row in grid]


# ── All primitives with names and Python functions ───────────────

# These are transforms that can be composed.
# Each entry: (name, python_fn, can_change_size)
# can_change_size: if True, output dims may differ from input dims

PRIMITIVES = [
    ("rotate_90", _rotate_90, True),
    ("rotate_180", _rotate_180, False),
    ("rotate_270", _rotate_270, True),
    ("flip_h", _flip_h, False),
    ("flip_v", _flip_v, False),
    ("transpose", _transpose, True),
    ("gravity_down", _gravity_down, False),
    ("gravity_up", _gravity_up, False),
    ("gravity_left", _gravity_left, False),
    ("gravity_right", _gravity_right, False),
    ("invert_colors", _invert_colors, False),
]


# ── C code bodies for each primitive ─────────────────────────────
# These operate on (grid, h, w) -> (out, oh, ow)
# For composition, step 1 writes to out, step 2 reads from out and writes to grid (ping-pong)

C_BODIES = {
    "rotate_90": """
    {oh} = {w}; {ow} = {h};
    {{ int r, c;
    for (c = 0; c < {w}; c++)
        for (r = 0; r < {h}; r++)
            arc_set({dst}, {oh}, c, {h} - 1 - r, arc_get({src}, {w}, r, c));
    }}""",

    "rotate_180": """
    {oh} = {h}; {ow} = {w};
    {{ int r, c;
    for (r = 0; r < {h}; r++)
        for (c = 0; c < {w}; c++)
            arc_set({dst}, {ow}, {h} - 1 - r, {w} - 1 - c, arc_get({src}, {w}, r, c));
    }}""",

    "rotate_270": """
    {oh} = {w}; {ow} = {h};
    {{ int r, c;
    for (c = 0; c < {w}; c++)
        for (r = 0; r < {h}; r++)
            arc_set({dst}, {oh}, {w} - 1 - c, r, arc_get({src}, {w}, r, c));
    }}""",

    "flip_h": """
    {oh} = {h}; {ow} = {w};
    {{ int r, c;
    for (r = 0; r < {h}; r++)
        for (c = 0; c < {w}; c++)
            arc_set({dst}, {ow}, r, {w} - 1 - c, arc_get({src}, {w}, r, c));
    }}""",

    "flip_v": """
    {oh} = {h}; {ow} = {w};
    {{ int r, c;
    for (r = 0; r < {h}; r++)
        for (c = 0; c < {w}; c++)
            arc_set({dst}, {ow}, {h} - 1 - r, c, arc_get({src}, {w}, r, c));
    }}""",

    "transpose": """
    {oh} = {w}; {ow} = {h};
    {{ int r, c;
    for (r = 0; r < {h}; r++)
        for (c = 0; c < {w}; c++)
            arc_set({dst}, {ow}, c, r, arc_get({src}, {w}, r, c));
    }}""",

    "gravity_down": """
    {oh} = {h}; {ow} = {w};
    arc_fill({dst}, {h}, {w}, 0);
    {{ int c, r, count, wr, ii;
    for (c = 0; c < {w}; c++) {{
        count = 0;
        for (r = 0; r < {h}; r++) {{
            int v = arc_get({src}, {w}, r, c);
            if (v != 0) {{ _tmp[count] = v; count++; }}
        }}
        wr = {h} - count;
        for (ii = 0; ii < count; ii++)
            arc_set({dst}, {ow}, wr + ii, c, _tmp[ii]);
    }} }}""",

    "gravity_up": """
    {oh} = {h}; {ow} = {w};
    arc_fill({dst}, {h}, {w}, 0);
    {{ int c, r, count, ii;
    for (c = 0; c < {w}; c++) {{
        count = 0;
        for (r = 0; r < {h}; r++) {{
            int v = arc_get({src}, {w}, r, c);
            if (v != 0) {{ _tmp[count] = v; count++; }}
        }}
        for (ii = 0; ii < count; ii++)
            arc_set({dst}, {ow}, ii, c, _tmp[ii]);
    }} }}""",

    "gravity_left": """
    {oh} = {h}; {ow} = {w};
    arc_fill({dst}, {h}, {w}, 0);
    {{ int r, c, count, ii;
    for (r = 0; r < {h}; r++) {{
        count = 0;
        for (c = 0; c < {w}; c++) {{
            int v = arc_get({src}, {w}, r, c);
            if (v != 0) {{ _tmp[count] = v; count++; }}
        }}
        for (ii = 0; ii < count; ii++)
            arc_set({dst}, {ow}, r, ii, _tmp[ii]);
    }} }}""",

    "gravity_right": """
    {oh} = {h}; {ow} = {w};
    arc_fill({dst}, {h}, {w}, 0);
    {{ int r, c, count, wc, ii;
    for (r = 0; r < {h}; r++) {{
        count = 0;
        for (c = 0; c < {w}; c++) {{
            int v = arc_get({src}, {w}, r, c);
            if (v != 0) {{ _tmp[count] = v; count++; }}
        }}
        wc = {w} - count;
        for (ii = 0; ii < count; ii++)
            arc_set({dst}, {ow}, r, wc + ii, _tmp[ii]);
    }} }}""",

    "invert_colors": """
    {oh} = {h}; {ow} = {w};
    {{ int total = 0, ii;
    for (ii = 0; ii < {h}; ii++) total = total + {w};
    for (ii = 0; ii < total; ii++)
        {dst}[ii] = 9 - {src}[ii];
    }}""",
}


def _make_color_map_body(mapping: dict, src: str, dst: str, h: str, w: str, oh: str, ow: str) -> str:
    """Generate C body for a specific color map."""
    lut = list(range(10))
    for s, d in mapping.items():
        lut[s] = d
    return f"""
    {oh} = {h}; {ow} = {w};
    {{ int lut[10];
    lut[0]={lut[0]}; lut[1]={lut[1]}; lut[2]={lut[2]}; lut[3]={lut[3]}; lut[4]={lut[4]};
    lut[5]={lut[5]}; lut[6]={lut[6]}; lut[7]={lut[7]}; lut[8]={lut[8]}; lut[9]={lut[9]};
    {{ int total = 0, ii;
    for (ii = 0; ii < {h}; ii++) total = total + {w};
    for (ii = 0; ii < total; ii++)
        {dst}[ii] = lut[{src}[ii]];
    }} }}"""


# ── Composition: A then B ───────────────────────────────────────

def _compose_py(fn_a, fn_b, grid: Grid) -> Optional[Grid]:
    """Apply fn_a then fn_b. Returns None if either fails."""
    try:
        mid = fn_a(grid)
        if mid is None:
            return None
        result = fn_b(mid)
        return result
    except Exception:
        return None


def _check_composition(fn_a, fn_b, pairs: list) -> bool:
    """Check if A->B matches all training pairs."""
    for pair in pairs:
        result = _compose_py(fn_a, fn_b, pair["input"])
        if result is None:
            return False
        if not grids_equal(result, pair["output"]):
            return False
    return True


def _check_single(fn, pairs: list) -> bool:
    """Check if a single transform matches all training pairs."""
    for pair in pairs:
        try:
            result = fn(pair["input"])
            if result is None or not grids_equal(result, pair["output"]):
                return False
        except Exception:
            return False
    return True


def _make_composition_c(name_a: str, name_b: str, color_map: dict = None) -> str:
    """Generate C code for a two-step composition A->B.

    Uses ping-pong between grid and out buffers:
    Step 1: grid -> out (applying A)
    Step 2: out -> grid (applying B)
    Then copy grid -> out for final emit.
    """
    # Step 1: A reads from grid, writes to out
    if name_a == "color_map" and color_map is not None:
        step1 = _make_color_map_body(color_map, "grid", "out", "h", "w", "oh", "ow")
    else:
        template_a = C_BODIES.get(name_a)
        if template_a is None:
            return None
        step1 = template_a.format(src="grid", dst="out", h="h", w="w", oh="oh", ow="ow")

    # Step 2: B reads from out, writes to grid
    # Dimensions after step 1: oh x ow
    if name_b == "color_map" and color_map is not None:
        step2 = _make_color_map_body(color_map, "out", "grid", "oh", "ow", "h", "w")
    else:
        template_b = C_BODIES.get(name_b)
        if template_b is None:
            return None
        step2 = template_b.format(src="out", dst="grid", h="oh", w="ow", oh="h", ow="w")

    # Final: copy grid (with new dims h, w) -> out for emit
    body = f"""
    /* Step 1: {name_a} */
    {step1}
    /* Step 2: {name_b} */
    {step2}
    /* Final: result is in grid with dims (h, w), copy to out for emit */
    oh = h; ow = w;
    {{ int total = 0, ii;
    for (ii = 0; ii < h; ii++) total = total + w;
    for (ii = 0; ii < total; ii++) out[ii] = grid[ii];
    }}"""

    desc = f"compose: {name_a} then {name_b}"
    return _make_c_program(body, desc)


# ── Check for transform + color_map compositions ────────────────

def _check_transform_then_colormap(fn, pairs: list) -> Optional[Tuple[str, dict]]:
    """Check if applying transform fn, then some color map, matches output.

    Returns the color mapping dict if found, None otherwise.
    """
    # Apply transform to all inputs, then check if a consistent color map
    # maps the transformed result to the expected output
    mapping = {}
    for pair in pairs:
        try:
            transformed = fn(pair["input"])
        except Exception:
            return None
        if transformed is None:
            return None
        h_t, w_t = _dimensions(transformed)
        h_o, w_o = _dimensions(pair["output"])
        if h_t != h_o or w_t != w_o:
            return None
        for r in range(h_t):
            for c in range(w_t):
                src = transformed[r][c]
                dst = pair["output"][r][c]
                if src in mapping:
                    if mapping[src] != dst:
                        return None
                else:
                    mapping[src] = dst
    if not mapping or all(k == v for k, v in mapping.items()):
        return None
    return mapping


def _check_colormap_then_transform(fn, pairs: list) -> Optional[dict]:
    """Check if applying some color map, then transform fn, matches output.

    Returns the color mapping dict if found, None otherwise.
    """
    # For each pair, we need: color_map(input) then fn = output
    # So fn(color_map(input)) = output
    # Equivalently: color_map(input) = fn_inverse(output)
    # But we don't have inverses for all transforms. Instead:
    # Try all consistent color maps that make fn(color_map(input)) = output
    # This is expensive, so we use a different approach:
    # For pair 0, try to find a color map M such that fn(M(input0)) = output0
    # Then verify M works for all other pairs.

    # First, detect the mapping from pair 0
    # Apply fn to output to get what the pre-transform grid should look like
    # ... this requires inverse transforms. Skip for now and just brute-force check.

    # Simple approach: try all 10! color maps? No, too many.
    # Better: for pair 0, compute what M must be cell by cell.
    # M(input[r][c]) must make fn(M_grid) == output
    # This is complex for arbitrary fn. Skip this direction for now.
    return None


# ── Main composition generator ───────────────────────────────────

def generate_compositions(task: dict) -> List[Hypothesis]:
    """Generate composition hypotheses for an ARC task.

    Strategy:
    1. Try all pairwise compositions of geometric primitives (Python pre-check)
    2. Try transform + color_map compositions
    3. Only generate C code for verified matches

    Returns list of Hypothesis objects.
    """
    pairs = task["train"]
    hypotheses = []

    # 1. New single primitives not in the original hypothesizer
    # (gravity_up, gravity_left, gravity_right, invert_colors)
    extra_singles = [
        ("gravity_up", _gravity_up),
        ("gravity_left", _gravity_left),
        ("gravity_right", _gravity_right),
        ("invert_colors", _invert_colors),
    ]
    for name, fn in extra_singles:
        if _check_single(fn, pairs):
            c_body_template = C_BODIES.get(name)
            if c_body_template:
                body = c_body_template.format(
                    src="grid", dst="out", h="h", w="w", oh="oh", ow="ow"
                )
                desc = f"single: {name}"
                c_code = _make_c_program(body, desc)
                hypotheses.append(Hypothesis(name, desc, c_code, confidence=0.9))

    # 2. All pairwise compositions of primitives
    for (name_a, fn_a, _), (name_b, fn_b, _) in product(PRIMITIVES, PRIMITIVES):
        if name_a == name_b:
            continue  # Skip A->A (often identity or handled by singles)
        if _check_composition(fn_a, fn_b, pairs):
            c_code = _make_composition_c(name_a, name_b)
            if c_code:
                comp_name = f"{name_a}_then_{name_b}"
                desc = f"compose: {name_a} then {name_b}"
                hypotheses.append(Hypothesis(comp_name, desc, c_code, confidence=0.85))

    # 3. Transform + color_map: apply a geometric transform, then a color map
    for name, fn, _ in PRIMITIVES:
        mapping = _check_transform_then_colormap(fn, pairs)
        if mapping is not None:
            c_code = _make_composition_c(name, "color_map", color_map=mapping)
            if c_code:
                map_desc = ", ".join(f"{k}->{v}" for k, v in sorted(mapping.items()))
                comp_name = f"{name}_then_colormap"
                desc = f"compose: {name} then color_map({map_desc})"
                hypotheses.append(Hypothesis(comp_name, desc, c_code, confidence=0.88))

    # 4. Color_map + transform: apply color map first, then geometric transform
    # For each transform, try to find a color map that makes it work
    for name, fn, _ in PRIMITIVES:
        # Brute-force: for each pair, compute what the intermediate grid should be
        # intermediate = fn_inverse(output), then color_map(input) = intermediate
        # We can compute fn_inverse for geometric transforms:
        inverses = {
            "rotate_90": _rotate_270,
            "rotate_180": _rotate_180,
            "rotate_270": _rotate_90,
            "flip_h": _flip_h,
            "flip_v": _flip_v,
            "transpose": _transpose,
        }
        inv_fn = inverses.get(name)
        if inv_fn is None:
            continue

        # Find color map: for each pair, apply inverse to output to get intermediate
        mapping = {}
        valid = True
        for pair in pairs:
            try:
                intermediate = inv_fn(pair["output"])
            except Exception:
                valid = False
                break
            h_i, w_i = _dimensions(pair["input"])
            h_m, w_m = _dimensions(intermediate)
            if h_i != h_m or w_i != w_m:
                valid = False
                break
            for r in range(h_i):
                for c in range(w_i):
                    src = pair["input"][r][c]
                    dst = intermediate[r][c]
                    if src in mapping:
                        if mapping[src] != dst:
                            valid = False
                            break
                    else:
                        mapping[src] = dst
                if not valid:
                    break
            if not valid:
                break

        if valid and mapping and not all(k == v for k, v in mapping.items()):
            c_code = _make_composition_c("color_map", name, color_map=mapping)
            if c_code:
                map_desc = ", ".join(f"{k}->{v}" for k, v in sorted(mapping.items()))
                comp_name = f"colormap_then_{name}"
                desc = f"compose: color_map({map_desc}) then {name}"
                hypotheses.append(Hypothesis(comp_name, desc, c_code, confidence=0.88))

    return hypotheses
