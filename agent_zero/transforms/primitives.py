"""Pure numpy transform primitives for ARC grid manipulation.

Every function takes a numpy 2D int array (values 0-9, sizes 1x1 to 64x64)
and returns a new array. No side effects. No external dependencies beyond numpy.

Categories:
    - Geometric (12): rotations, reflections, shifts
    - Scaling (4): up/down scaling, tiling
    - Crop and Pad (4): cropping, padding, resizing
    - Color (8): remapping, swapping, counting
    - Region (10): connected components, masks, flood fill
    - Gravity (4): drop cells in a direction
    - Pattern (6): mirroring, borders, line drawing
    - Conditional (4): masking, component filtering
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Geometric transforms (12)
# ---------------------------------------------------------------------------

def rotate_90(grid: np.ndarray) -> np.ndarray:
    """Rotate grid 90 degrees clockwise."""
    return np.rot90(grid, k=-1).copy()


def rotate_180(grid: np.ndarray) -> np.ndarray:
    """Rotate grid 180 degrees."""
    return np.rot90(grid, k=2).copy()


def rotate_270(grid: np.ndarray) -> np.ndarray:
    """Rotate grid 270 degrees clockwise (90 degrees counter-clockwise)."""
    return np.rot90(grid, k=-3).copy()


def reflect_horizontal(grid: np.ndarray) -> np.ndarray:
    """Reflect grid left-to-right (flip columns)."""
    return np.fliplr(grid).copy()


def reflect_vertical(grid: np.ndarray) -> np.ndarray:
    """Reflect grid top-to-bottom (flip rows)."""
    return np.flipud(grid).copy()


def reflect_diagonal_main(grid: np.ndarray) -> np.ndarray:
    """Reflect across the main diagonal (top-left to bottom-right). Same as transpose."""
    return grid.T.copy()


def reflect_diagonal_anti(grid: np.ndarray) -> np.ndarray:
    """Reflect across the anti-diagonal (top-right to bottom-left)."""
    return np.rot90(grid, k=1).T.copy()


def transpose(grid: np.ndarray) -> np.ndarray:
    """Transpose the grid (swap rows and columns)."""
    return grid.T.copy()


def shift_up(grid: np.ndarray, wrap: bool = False) -> np.ndarray:
    """Shift all rows up by one. Bottom row filled with 0 or wrapped."""
    out = np.zeros_like(grid)
    if grid.shape[0] <= 1:
        return grid.copy() if wrap else out
    out[:-1] = grid[1:]
    if wrap:
        out[-1] = grid[0]
    return out


def shift_down(grid: np.ndarray, wrap: bool = False) -> np.ndarray:
    """Shift all rows down by one. Top row filled with 0 or wrapped."""
    out = np.zeros_like(grid)
    if grid.shape[0] <= 1:
        return grid.copy() if wrap else out
    out[1:] = grid[:-1]
    if wrap:
        out[0] = grid[-1]
    return out


def shift_left(grid: np.ndarray, wrap: bool = False) -> np.ndarray:
    """Shift all columns left by one. Right column filled with 0 or wrapped."""
    out = np.zeros_like(grid)
    if grid.shape[1] <= 1:
        return grid.copy() if wrap else out
    out[:, :-1] = grid[:, 1:]
    if wrap:
        out[:, -1] = grid[:, 0]
    return out


def shift_right(grid: np.ndarray, wrap: bool = False) -> np.ndarray:
    """Shift all columns right by one. Left column filled with 0 or wrapped."""
    out = np.zeros_like(grid)
    if grid.shape[1] <= 1:
        return grid.copy() if wrap else out
    out[:, 1:] = grid[:, :-1]
    if wrap:
        out[:, 0] = grid[:, -1]
    return out


# ---------------------------------------------------------------------------
# Scaling transforms (4)
# ---------------------------------------------------------------------------

def scale_up_2x(grid: np.ndarray) -> np.ndarray:
    """Scale grid up by 2x using nearest-neighbor (each cell becomes 2x2)."""
    return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)


def scale_up_3x(grid: np.ndarray) -> np.ndarray:
    """Scale grid up by 3x using nearest-neighbor (each cell becomes 3x3)."""
    return np.repeat(np.repeat(grid, 3, axis=0), 3, axis=1)


def scale_down_2x(grid: np.ndarray) -> np.ndarray:
    """Scale grid down by 2x using majority vote per 2x2 block.

    If the grid has an odd dimension, the last row/column is dropped before
    downscaling.
    """
    h, w = grid.shape
    # Trim to even dimensions
    h_even = h - (h % 2)
    w_even = w - (w % 2)
    if h_even == 0 or w_even == 0:
        return grid.copy()
    trimmed = grid[:h_even, :w_even]
    # Reshape into 2x2 blocks
    blocks = trimmed.reshape(h_even // 2, 2, w_even // 2, 2)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(h_even // 2, w_even // 2, 4)
    # Majority vote per block
    out = np.zeros((h_even // 2, w_even // 2), dtype=grid.dtype)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            vals, counts = np.unique(blocks[i, j], return_counts=True)
            out[i, j] = vals[np.argmax(counts)]
    return out


def tile(grid: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Tile the grid into a *rows* x *cols* repeated pattern."""
    return np.tile(grid, (rows, cols))


# ---------------------------------------------------------------------------
# Crop and Pad (4)
# ---------------------------------------------------------------------------

def crop(grid: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> np.ndarray:
    """Crop grid to the rectangle [r1:r2, c1:c2] (exclusive end)."""
    return grid[r1:r2, c1:c2].copy()


def crop_to_content(grid: np.ndarray, background: int = 0) -> np.ndarray:
    """Crop to the smallest bounding box around non-background cells.

    Returns the full grid unchanged if every cell equals *background*.
    """
    mask = grid != background
    if not mask.any():
        return grid.copy()
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r1, r2 = np.argmax(rows), grid.shape[0] - np.argmax(rows[::-1])
    c1, c2 = np.argmax(cols), grid.shape[1] - np.argmax(cols[::-1])
    return grid[r1:r2, c1:c2].copy()


def pad(grid: np.ndarray, top: int, bottom: int, left: int, right: int,
        fill: int = 0) -> np.ndarray:
    """Pad the grid on each side with a constant *fill* value."""
    return np.pad(grid, ((top, bottom), (left, right)),
                  mode='constant', constant_values=fill)


def resize(grid: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Resize grid to (*new_h*, *new_w*) using nearest-neighbor interpolation."""
    h, w = grid.shape
    row_idx = (np.arange(new_h) * h / new_h).astype(int)
    col_idx = (np.arange(new_w) * w / new_w).astype(int)
    # Clamp indices to valid range
    row_idx = np.clip(row_idx, 0, h - 1)
    col_idx = np.clip(col_idx, 0, w - 1)
    return grid[np.ix_(row_idx, col_idx)].copy()


# ---------------------------------------------------------------------------
# Color transforms (8)
# ---------------------------------------------------------------------------

def color_swap(grid: np.ndarray, a: int, b: int) -> np.ndarray:
    """Swap colors *a* and *b* throughout the grid."""
    out = grid.copy()
    mask_a = grid == a
    mask_b = grid == b
    out[mask_a] = b
    out[mask_b] = a
    return out


def color_map(grid: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    """Remap colors according to *mapping* dict. Unmapped colors stay the same."""
    out = grid.copy()
    for old_val, new_val in mapping.items():
        out[grid == old_val] = new_val
    return out


def replace_color(grid: np.ndarray, old: int, new: int) -> np.ndarray:
    """Replace all occurrences of color *old* with *new*."""
    out = grid.copy()
    out[out == old] = new
    return out


def most_common_color(grid: np.ndarray) -> int:
    """Return the most common color in the grid."""
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def least_common_color(grid: np.ndarray) -> int:
    """Return the least common color in the grid."""
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmin(counts)])


def count_colors(grid: np.ndarray) -> Dict[int, int]:
    """Return a dict mapping each color to its count in the grid."""
    vals, counts = np.unique(grid, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def recolor_by_rank(grid: np.ndarray) -> np.ndarray:
    """Recolor so the most common color becomes 0, next most common 1, etc."""
    vals, counts = np.unique(grid, return_counts=True)
    # Sort by descending count, then ascending value for ties
    order = np.lexsort((vals, -counts))
    mapping = {int(vals[idx]): rank for rank, idx in enumerate(order)}
    out = grid.copy()
    for old_val, new_val in mapping.items():
        out[grid == old_val] = new_val
    return out


def invert_colors(grid: np.ndarray, max_color: int = 9) -> np.ndarray:
    """Invert colors: each value *v* becomes *max_color - v*."""
    return (max_color - grid).astype(grid.dtype)


# ---------------------------------------------------------------------------
# Region transforms (10)
# ---------------------------------------------------------------------------

def _bfs_component(grid: np.ndarray, visited: np.ndarray,
                   start_r: int, start_c: int) -> List[Tuple[int, int]]:
    """BFS helper returning the list of (row, col) for one connected component."""
    h, w = grid.shape
    color = grid[start_r, start_c]
    queue: deque[Tuple[int, int]] = deque()
    queue.append((start_r, start_c))
    visited[start_r, start_c] = True
    pixels: List[Tuple[int, int]] = []
    while queue:
        r, c = queue.popleft()
        pixels.append((r, c))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == color:
                visited[nr, nc] = True
                queue.append((nr, nc))
    return pixels


def connected_components(grid: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
    """Find all 4-connected components in the grid.

    Returns:
        labeled_grid: 2D array where each cell holds its component label (1-indexed).
        components: List of dicts, each with keys:
            - 'label': int, 1-indexed component label
            - 'color': int, the grid color of this component
            - 'size': int, number of cells
            - 'bbox': (r1, c1, r2, c2), bounding box (exclusive end)
            - 'centroid': (float, float), mean row and column
            - 'pixels': list of (row, col) tuples
    """
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    labeled = np.zeros((h, w), dtype=int)
    components: List[dict] = []
    label = 0

    for r in range(h):
        for c in range(w):
            if not visited[r, c]:
                label += 1
                pixels = _bfs_component(grid, visited, r, c)
                rows = [p[0] for p in pixels]
                cols = [p[1] for p in pixels]
                for pr, pc in pixels:
                    labeled[pr, pc] = label
                components.append({
                    'label': label,
                    'color': int(grid[r, c]),
                    'size': len(pixels),
                    'bbox': (min(rows), min(cols), max(rows) + 1, max(cols) + 1),
                    'centroid': (float(np.mean(rows)), float(np.mean(cols))),
                    'pixels': pixels,
                })

    return labeled, components


def mask_by_color(grid: np.ndarray, color: int) -> np.ndarray:
    """Return a binary mask (0/1 int array) where cells match *color*."""
    return (grid == color).astype(int)


def extract_by_color(grid: np.ndarray, color: int) -> np.ndarray:
    """Extract the bounding-box sub-grid containing all cells of *color*.

    Non-matching cells within the bbox are set to 0.
    Returns the full grid if *color* is not present.
    """
    mask = grid == color
    if not mask.any():
        return grid.copy()
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r1, r2 = np.argmax(rows), grid.shape[0] - np.argmax(rows[::-1])
    c1, c2 = np.argmax(cols), grid.shape[1] - np.argmax(cols[::-1])
    sub = grid[r1:r2, c1:c2].copy()
    sub[~mask[r1:r2, c1:c2]] = 0
    return sub


def fill_region(grid: np.ndarray, mask: np.ndarray, color: int) -> np.ndarray:
    """Fill all cells where *mask* is truthy with *color*."""
    out = grid.copy()
    out[mask.astype(bool)] = color
    return out


def flood_fill(grid: np.ndarray, r: int, c: int, new_color: int) -> np.ndarray:
    """Flood-fill from (r, c) with *new_color* (4-connected)."""
    out = grid.copy()
    h, w = out.shape
    if r < 0 or r >= h or c < 0 or c >= w:
        return out
    old_color = out[r, c]
    if old_color == new_color:
        return out
    queue: deque[Tuple[int, int]] = deque()
    queue.append((r, c))
    out[r, c] = new_color
    while queue:
        cr, cc = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == old_color:
                out[nr, nc] = new_color
                queue.append((nr, nc))
    return out


def bounding_box(grid: np.ndarray, color: int) -> Tuple[int, int, int, int]:
    """Return (r1, c1, r2, c2) bounding box of *color* (exclusive end).

    Returns (0, 0, 0, 0) if *color* is not present.
    """
    mask = grid == color
    if not mask.any():
        return (0, 0, 0, 0)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r1 = int(np.argmax(rows))
    r2 = int(grid.shape[0] - np.argmax(rows[::-1]))
    c1 = int(np.argmax(cols))
    c2 = int(grid.shape[1] - np.argmax(cols[::-1]))
    return (r1, c1, r2, c2)


def largest_component(grid: np.ndarray) -> np.ndarray:
    """Return a binary mask of the largest connected component."""
    _, comps = connected_components(grid)
    if not comps:
        return np.zeros_like(grid, dtype=int)
    best = max(comps, key=lambda c: c['size'])
    mask = np.zeros_like(grid, dtype=int)
    for r, c in best['pixels']:
        mask[r, c] = 1
    return mask


def smallest_component(grid: np.ndarray) -> np.ndarray:
    """Return a binary mask of the smallest connected component."""
    _, comps = connected_components(grid)
    if not comps:
        return np.zeros_like(grid, dtype=int)
    best = min(comps, key=lambda c: c['size'])
    mask = np.zeros_like(grid, dtype=int)
    for r, c in best['pixels']:
        mask[r, c] = 1
    return mask


def border_cells(grid: np.ndarray) -> np.ndarray:
    """Return a binary mask of cells on the grid border."""
    mask = np.zeros_like(grid, dtype=int)
    if grid.size == 0:
        return mask
    mask[0, :] = 1
    mask[-1, :] = 1
    mask[:, 0] = 1
    mask[:, -1] = 1
    return mask


def interior_cells(grid: np.ndarray) -> np.ndarray:
    """Return a binary mask of cells NOT on the grid border."""
    return 1 - border_cells(grid)


# ---------------------------------------------------------------------------
# Gravity transforms (4)
# ---------------------------------------------------------------------------

def gravity_down(grid: np.ndarray, background: int = 0) -> np.ndarray:
    """Drop all non-background cells downward within each column."""
    out = np.full_like(grid, background)
    h = grid.shape[0]
    for c in range(grid.shape[1]):
        col = grid[:, c]
        non_bg = col[col != background]
        if non_bg.size > 0:
            out[h - non_bg.size:, c] = non_bg
    return out


def gravity_up(grid: np.ndarray, background: int = 0) -> np.ndarray:
    """Drop all non-background cells upward within each column."""
    out = np.full_like(grid, background)
    for c in range(grid.shape[1]):
        col = grid[:, c]
        non_bg = col[col != background]
        if non_bg.size > 0:
            out[:non_bg.size, c] = non_bg
    return out


def gravity_left(grid: np.ndarray, background: int = 0) -> np.ndarray:
    """Drop all non-background cells leftward within each row."""
    out = np.full_like(grid, background)
    for r in range(grid.shape[0]):
        row = grid[r, :]
        non_bg = row[row != background]
        if non_bg.size > 0:
            out[r, :non_bg.size] = non_bg
    return out


def gravity_right(grid: np.ndarray, background: int = 0) -> np.ndarray:
    """Drop all non-background cells rightward within each row."""
    out = np.full_like(grid, background)
    w = grid.shape[1]
    for r in range(grid.shape[0]):
        row = grid[r, :]
        non_bg = row[row != background]
        if non_bg.size > 0:
            out[r, w - non_bg.size:] = non_bg
    return out


# ---------------------------------------------------------------------------
# Pattern transforms (6)
# ---------------------------------------------------------------------------

def repeat_pattern(grid: np.ndarray, axis: int, count: int) -> np.ndarray:
    """Repeat the grid along *axis* (0 = vertical, 1 = horizontal) *count* times."""
    return np.concatenate([grid] * count, axis=axis)


def mirror_extend(grid: np.ndarray, direction: str) -> np.ndarray:
    """Extend the grid by appending a mirrored copy on the given side.

    Args:
        direction: One of 'up', 'down', 'left', 'right'.
    """
    if direction == 'down':
        return np.concatenate([grid, np.flipud(grid)], axis=0)
    elif direction == 'up':
        return np.concatenate([np.flipud(grid), grid], axis=0)
    elif direction == 'right':
        return np.concatenate([grid, np.fliplr(grid)], axis=1)
    elif direction == 'left':
        return np.concatenate([np.fliplr(grid), grid], axis=1)
    else:
        raise ValueError(f"Invalid direction: {direction!r}. Use 'up','down','left','right'.")


def symmetrize(grid: np.ndarray, axis: str) -> np.ndarray:
    """Force symmetry by copying the top/left half onto the bottom/right half.

    Args:
        axis: 'horizontal' copies top half to bottom, 'vertical' copies left
              half to right.
    """
    out = grid.copy()
    if axis == 'horizontal':
        h = out.shape[0]
        mid = h // 2
        out[h - mid:] = np.flipud(out[:mid])
    elif axis == 'vertical':
        w = out.shape[1]
        mid = w // 2
        out[:, w - mid:] = np.fliplr(out[:, :mid])
    else:
        raise ValueError(f"Invalid axis: {axis!r}. Use 'horizontal' or 'vertical'.")
    return out


def draw_border(grid: np.ndarray, color: int, thickness: int = 1) -> np.ndarray:
    """Draw a border of *color* with given *thickness* around the grid edges."""
    out = grid.copy()
    t = min(thickness, min(grid.shape) // 2 + 1)
    for i in range(t):
        out[i, :] = color
        out[-(i + 1), :] = color
        out[:, i] = color
        out[:, -(i + 1)] = color
    return out


def draw_line(grid: np.ndarray, r1: int, c1: int, r2: int, c2: int,
              color: int) -> np.ndarray:
    """Draw a line from (r1, c1) to (r2, c2) using Bresenham's algorithm."""
    out = grid.copy()
    h, w = out.shape
    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    sr = 1 if r1 < r2 else -1
    sc = 1 if c1 < c2 else -1
    err = dr - dc
    r, c = r1, c1
    while True:
        if 0 <= r < h and 0 <= c < w:
            out[r, c] = color
        if r == r2 and c == c2:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return out


def fill_enclosed(grid: np.ndarray, background: int = 0) -> np.ndarray:
    """Fill enclosed interior regions of *background* color.

    Flood-fills *background* from all border cells, then replaces any remaining
    *background* cells with the most common non-background color.
    """
    out = grid.copy()
    h, w = out.shape

    # Mark all background cells reachable from the border
    reachable = np.zeros((h, w), dtype=bool)
    queue: deque[Tuple[int, int]] = deque()
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and out[r, c] == background:
                reachable[r, c] = True
                queue.append((r, c))
    while queue:
        cr, cc = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr, nc] and out[nr, nc] == background:
                reachable[nr, nc] = True
                queue.append((nr, nc))

    # Interior background cells are those not reached from the border
    interior_bg = (out == background) & ~reachable
    if not interior_bg.any():
        return out

    # Fill with most common non-background color
    non_bg = out[out != background]
    if non_bg.size == 0:
        return out
    vals, counts = np.unique(non_bg, return_counts=True)
    fill_color = int(vals[np.argmax(counts)])
    out[interior_bg] = fill_color
    return out


# ---------------------------------------------------------------------------
# Conditional transforms (4)
# ---------------------------------------------------------------------------

def where(condition_mask: np.ndarray, grid_true: np.ndarray,
          grid_false: np.ndarray) -> np.ndarray:
    """Element-wise selection: pick from *grid_true* where mask is truthy, else *grid_false*."""
    return np.where(condition_mask.astype(bool), grid_true, grid_false).astype(grid_true.dtype)


def select_by_size(components: List[dict], min_size: int = 0,
                   max_size: int = 9999) -> List[dict]:
    """Filter a component list to those with *min_size* <= size <= *max_size*."""
    return [c for c in components if min_size <= c['size'] <= max_size]


def select_by_color(components: List[dict],
                    colors: Sequence[int]) -> List[dict]:
    """Filter a component list to those whose color is in *colors*."""
    color_set = set(colors)
    return [c for c in components if c['color'] in color_set]


def apply_to_each_component(grid: np.ndarray,
                            fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply *fn* to each connected component independently.

    For each component, extracts its bounding-box sub-grid (non-component cells
    set to 0), applies *fn*, and pastes the result back. If the transformed
    sub-grid changes shape, it is placed at the same top-left corner, clipped
    to the output bounds.
    """
    out = grid.copy()
    _, comps = connected_components(grid)
    h, w = grid.shape

    for comp in comps:
        r1, c1, r2, c2 = comp['bbox']
        sub = grid[r1:r2, c1:c2].copy()
        # Zero out cells not in this component
        comp_mask = np.zeros_like(sub, dtype=bool)
        for pr, pc in comp['pixels']:
            comp_mask[pr - r1, pc - c1] = True
        sub[~comp_mask] = 0

        transformed = fn(sub)
        th, tw = transformed.shape

        # Paste back, clipping to grid bounds
        paste_h = min(th, h - r1)
        paste_w = min(tw, w - c1)
        if paste_h <= 0 or paste_w <= 0:
            continue

        # Only overwrite cells that were part of the original component or
        # that the transform produced non-zero values for
        patch = transformed[:paste_h, :paste_w]
        # Clear original component cells in output
        for pr, pc in comp['pixels']:
            if pr < h and pc < w:
                out[pr, pc] = 0
        # Write transformed cells
        region = out[r1:r1 + paste_h, c1:c1 + paste_w]
        write_mask = patch != 0
        region[write_mask] = patch[write_mask]
        out[r1:r1 + paste_h, c1:c1 + paste_w] = region

    return out


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------
# Conditional / Spatial transforms (high-impact additions)
# ---------------------------------------------------------------------------

def fill_enclosed_with_color(grid: np.ndarray, fill_color: int = 4, background: int = 0) -> np.ndarray:
    """Fill enclosed regions with a specific color (not necessarily the border color)."""
    out = grid.copy()
    h, w = out.shape
    reachable = np.zeros((h, w), dtype=bool)
    queue: deque[Tuple[int, int]] = deque()
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and out[r, c] == background:
                reachable[r, c] = True
                queue.append((r, c))
    while queue:
        cr, cc = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr, nc] and out[nr, nc] == background:
                reachable[nr, nc] = True
                queue.append((nr, nc))
    interior = (out == background) & ~reachable
    out[interior] = fill_color
    return out


def fill_enclosed_per_region(grid: np.ndarray, background: int = 0) -> np.ndarray:
    """Fill each enclosed region with a unique NEW color (starting from max+1).
    Each separate enclosed cavity gets its own color."""
    out = grid.copy()
    h, w = out.shape
    reachable = np.zeros((h, w), dtype=bool)
    queue: deque[Tuple[int, int]] = deque()
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and out[r, c] == background:
                reachable[r, c] = True
                queue.append((r, c))
    while queue:
        cr, cc = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr, nc] and out[nr, nc] == background:
                reachable[nr, nc] = True
                queue.append((nr, nc))
    interior = (out == background) & ~reachable
    if not interior.any():
        return out
    # BFS label each interior region
    next_color = int(out.max()) + 1
    labeled = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if interior[r, c] and not labeled[r, c]:
                q2: deque[Tuple[int, int]] = deque([(r, c)])
                labeled[r, c] = True
                while q2:
                    cr2, cc2 = q2.popleft()
                    out[cr2, cc2] = next_color
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr2 + dr, cc2 + dc
                        if 0 <= nr < h and 0 <= nc < w and interior[nr, nc] and not labeled[nr, nc]:
                            labeled[nr, nc] = True
                            q2.append((nr, nc))
                next_color = min(next_color + 1, 9)
    return out


def fill_adjacent_to_color(grid: np.ndarray, target_color: int = 0,
                           near_color: int = 1, fill_color: int = 2) -> np.ndarray:
    """Fill cells of target_color that are adjacent (4-connected) to near_color with fill_color."""
    out = grid.copy()
    h, w = out.shape
    for r in range(h):
        for c in range(w):
            if out[r, c] == target_color:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == near_color:
                        out[r, c] = fill_color
                        break
    return out


def draw_line_between_same_color(grid: np.ndarray, color: int = 1,
                                  line_color: int = 1) -> np.ndarray:
    """Draw horizontal or vertical lines between pairs of cells of the same color.
    Only draws if the cells share a row or column with only background between them."""
    out = grid.copy()
    h, w = out.shape
    # Find all cells of the target color
    cells = [(r, c) for r in range(h) for c in range(w) if grid[r, c] == color]
    # For each pair sharing a row
    for i, (r1, c1) in enumerate(cells):
        for r2, c2 in cells[i+1:]:
            if r1 == r2:  # same row
                cmin, cmax = min(c1, c2), max(c1, c2)
                if all(grid[r1, c] == 0 or grid[r1, c] == color for c in range(cmin, cmax + 1)):
                    for c in range(cmin, cmax + 1):
                        out[r1, c] = line_color
            elif c1 == c2:  # same column
                rmin, rmax = min(r1, r2), max(r1, r2)
                if all(grid[r, c1] == 0 or grid[r, c1] == color for r in range(rmin, rmax + 1)):
                    for r in range(rmin, rmax + 1):
                        out[r, c1] = line_color
    return out


def move_object_until_wall(grid: np.ndarray, obj_color: int = 1,
                           wall_color: int = 2, direction: str = "down",
                           background: int = 0) -> np.ndarray:
    """Move all cells of obj_color in a direction until they hit wall_color or grid edge."""
    out = grid.copy()
    h, w = out.shape
    obj_cells = [(r, c) for r in range(h) for c in range(w) if grid[r, c] == obj_color]
    if not obj_cells:
        return out

    dr, dc = {"down": (1, 0), "up": (-1, 0), "right": (0, 1), "left": (0, -1)}[direction]

    # Clear original positions
    for r, c in obj_cells:
        out[r, c] = background

    # Find max shift
    max_shift = max(h, w)
    for shift in range(1, max_shift):
        blocked = False
        for r, c in obj_cells:
            nr, nc = r + dr * shift, c + dc * shift
            if not (0 <= nr < h and 0 <= nc < w):
                blocked = True
                break
            if grid[nr, nc] == wall_color:
                blocked = True
                break
        if blocked:
            shift -= 1
            break

    # Place at new position
    for r, c in obj_cells:
        nr, nc = r + dr * shift, c + dc * shift
        if 0 <= nr < h and 0 <= nc < w:
            out[nr, nc] = obj_color
    return out


def extract_subgrid_by_color(grid: np.ndarray, border_color: int = 1) -> np.ndarray:
    """Extract the rectangular region bounded by cells of border_color (exclusive)."""
    rows = np.any(grid == border_color, axis=1)
    cols = np.any(grid == border_color, axis=0)
    if not rows.any() or not cols.any():
        return grid
    r1, r2 = np.where(rows)[0][[0, -1]]
    c1, c2 = np.where(cols)[0][[0, -1]]
    return grid[r1+1:r2, c1+1:c2].copy()


def extract_subgrid_by_color_inclusive(grid: np.ndarray, border_color: int = 1) -> np.ndarray:
    """Extract the rectangular region bounded by cells of border_color (inclusive)."""
    rows = np.any(grid == border_color, axis=1)
    cols = np.any(grid == border_color, axis=0)
    if not rows.any() or not cols.any():
        return grid
    r1, r2 = np.where(rows)[0][[0, -1]]
    c1, c2 = np.where(cols)[0][[0, -1]]
    return grid[r1:r2+1, c1:c2+1].copy()


def recolor_by_size(grid: np.ndarray, small_color: int = 1,
                    large_color: int = 2, background: int = 0) -> np.ndarray:
    """Connected components: smaller ones get small_color, larger ones get large_color."""
    out = grid.copy()
    h, w = out.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != background and not visited[r, c]:
                color = grid[r, c]
                cells = []
                q: deque[Tuple[int, int]] = deque([(r, c)])
                visited[r, c] = True
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == color:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                components.append(cells)
    if len(components) < 2:
        return out
    sizes = [len(c) for c in components]
    median_size = sorted(sizes)[len(sizes)//2]
    for cells in components:
        c = small_color if len(cells) <= median_size else large_color
        for r, cc in cells:
            out[r, cc] = c
    return out


def copy_pattern_to_markers(grid: np.ndarray, pattern_color: int = 1,
                            marker_color: int = 2, background: int = 0) -> np.ndarray:
    """Find a pattern (connected component of pattern_color), then copy it to
    every cell of marker_color. The marker cell becomes the top-left of the copied pattern."""
    out = grid.copy()
    h, w = out.shape
    # Find pattern bounding box
    pat_cells = np.argwhere(grid == pattern_color)
    if pat_cells.size == 0:
        return out
    pr1, pc1 = pat_cells.min(axis=0)
    pr2, pc2 = pat_cells.max(axis=0)
    pattern = grid[pr1:pr2+1, pc1:pc2+1].copy()
    ph, pw = pattern.shape
    # Find markers
    markers = np.argwhere(grid == marker_color)
    for mr, mc in markers:
        out[mr, mc] = background  # clear marker
        for r in range(ph):
            for c in range(pw):
                if pattern[r, c] == pattern_color:
                    nr, nc = mr + r, mc + c
                    if 0 <= nr < h and 0 <= nc < w:
                        out[nr, nc] = pattern_color
    return out


# ---------------------------------------------------------------------------

# Auto-discover all primitives
import types as _types

ALL_PRIMITIVES = {name: fn for name, fn in globals().items()
                  if isinstance(fn, _types.FunctionType) and not name.startswith('_')}
TRANSFORM_PRIMITIVES = [name for name in ALL_PRIMITIVES
                        if name not in ('most_common_color', 'least_common_color',
                                        'count_colors', 'connected_components',
                                        'select_by_size', 'select_by_color')]
