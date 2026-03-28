"""
Object Detector for ARC grids.

Finds connected components, extracts bounding boxes, computes object metadata.
All operations are pure Python for fast hypothesis pre-checking.

Object types:
- Connected components (same-color 4-connected regions)
- Background detection (most common color, usually 0)
- Enclosed regions (areas fully surrounded by a border color)
- Sub-grid extraction at separators (rows/cols of single color)
"""

from collections import Counter, deque
from typing import List, Optional, Tuple, Set

from .grid_io import Grid, grids_equal


class Object:
    """A detected object in an ARC grid."""

    def __init__(self, color: int, cells: Set[Tuple[int, int]],
                 grid_h: int, grid_w: int):
        self.color = color
        self.cells = cells
        self.grid_h = grid_h
        self.grid_w = grid_w
        # Bounding box
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        self.min_r = min(rows)
        self.max_r = max(rows)
        self.min_c = min(cols)
        self.max_c = max(cols)
        self.bbox_h = self.max_r - self.min_r + 1
        self.bbox_w = self.max_c - self.min_c + 1
        self.size = len(cells)
        self.position = (self.min_r, self.min_c)

    def extract(self, grid: Grid, bg: int = 0) -> Grid:
        """Extract this object as a sub-grid (bbox with bg fill)."""
        sub = [[bg] * self.bbox_w for _ in range(self.bbox_h)]
        for r, c in self.cells:
            sub[r - self.min_r][c - self.min_c] = grid[r][c]
        return sub

    def mask(self) -> Grid:
        """Return a binary mask grid (1 where object, 0 elsewhere)."""
        mask = [[0] * self.bbox_w for _ in range(self.bbox_h)]
        for r, c in self.cells:
            mask[r - self.min_r][c - self.min_c] = 1
        return mask

    def __repr__(self):
        return (f"Object(color={self.color}, size={self.size}, "
                f"pos=({self.min_r},{self.min_c}), "
                f"bbox={self.bbox_h}x{self.bbox_w})")


# ── Background detection ─────────────────────────────────────────

def detect_background(grid: Grid) -> int:
    """Detect the background color (most common, usually 0)."""
    flat = [v for row in grid for v in row]
    return Counter(flat).most_common(1)[0][0]


# ── Connected components ─────────────────────────────────────────

def find_connected_components(grid: Grid, bg: int = None,
                               connectivity: int = 4) -> List[Object]:
    """Find all connected components of non-background colors.

    Args:
        grid: Input grid
        bg: Background color (auto-detected if None)
        connectivity: 4 or 8 (4-connected or 8-connected)

    Returns:
        List of Object instances, sorted by size (largest first)
    """
    if bg is None:
        bg = detect_background(grid)

    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    visited = [[False] * w for _ in range(h)]
    objects = []

    if connectivity == 4:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 1), (1, -1), (1, 0), (1, 1)]

    for r in range(h):
        for c in range(w):
            if visited[r][c] or grid[r][c] == bg:
                continue
            # BFS flood fill
            color = grid[r][c]
            cells = set()
            queue = deque([(r, c)])
            visited[r][c] = True
            while queue:
                cr, cc = queue.popleft()
                cells.add((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < h and 0 <= nc < w and
                            not visited[nr][nc] and grid[nr][nc] == color):
                        visited[nr][nc] = True
                        queue.append((nr, nc))
            objects.append(Object(color, cells, h, w))

    objects.sort(key=lambda o: o.size, reverse=True)
    return objects


def find_all_objects(grid: Grid, bg: int = None) -> List[Object]:
    """Find objects including multi-color connected components.

    Uses 4-connectivity for same-color, then merges touching objects
    of different colors that form a single visual unit.
    """
    # For now, just return same-color components
    return find_connected_components(grid, bg, connectivity=4)


# ── Enclosed region detection ────────────────────────────────────

def find_enclosed_regions(grid: Grid, border_color: int = None,
                          bg: int = 0) -> List[Set[Tuple[int, int]]]:
    """Find regions of background color that are fully enclosed by border_color.

    Uses flood fill from edges — any bg cell reachable from the grid
    border is NOT enclosed. Everything else is.

    Returns list of enclosed region cell sets.
    """
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0

    # Flood fill from all bg cells on the border
    reachable = [[False] * w for _ in range(h)]
    queue = deque()

    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1):
                if grid[r][c] == bg and not reachable[r][c]:
                    reachable[r][c] = True
                    queue.append((r, c))

    while queue:
        cr, cc = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if (0 <= nr < h and 0 <= nc < w and
                    not reachable[nr][nc] and grid[nr][nc] == bg):
                reachable[nr][nc] = True
                queue.append((nr, nc))

    # All bg cells not reachable from border are enclosed
    enclosed_cells = set()
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg and not reachable[r][c]:
                enclosed_cells.add((r, c))

    if not enclosed_cells:
        return []

    # Group into connected regions
    visited = set()
    regions = []
    for r, c in enclosed_cells:
        if (r, c) in visited:
            continue
        region = set()
        queue = deque([(r, c)])
        visited.add((r, c))
        while queue:
            cr, cc = queue.popleft()
            region.add((cr, cc))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if ((nr, nc) in enclosed_cells and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        regions.append(region)

    return regions


# ── Grid separator detection ────────────────────────────────────

def find_separators(grid: Grid) -> dict:
    """Find rows or columns that act as separators (single solid color).

    Returns dict with 'rows' and 'cols' lists of (index, color) tuples.
    """
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    result = {"rows": [], "cols": []}

    # Check rows
    for r in range(h):
        if len(set(grid[r])) == 1 and grid[r][0] != 0:
            result["rows"].append((r, grid[r][0]))

    # Check columns
    for c in range(w):
        col_vals = [grid[r][c] for r in range(h)]
        if len(set(col_vals)) == 1 and col_vals[0] != 0:
            result["cols"].append((c, col_vals[0]))

    return result


def split_at_separator(grid: Grid, axis: str, index: int) -> Tuple[Grid, Grid]:
    """Split a grid at a separator row or column.

    Returns (left/top part, right/bottom part).
    The separator itself is excluded.
    """
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0

    if axis == "row":
        top = [row[:] for row in grid[:index]]
        bottom = [row[:] for row in grid[index + 1:]]
        return top, bottom
    else:  # col
        left = [row[:index] for row in grid]
        right = [row[index + 1:] for row in grid]
        return left, right


# ── Pattern analysis helpers ─────────────────────────────────────

def objects_same_shape(objects: List[Object]) -> bool:
    """Check if all objects have the same shape (mask)."""
    if len(objects) < 2:
        return True
    masks = [tuple(tuple(r) for r in o.mask()) for o in objects]
    return len(set(masks)) == 1


def object_colors(grid: Grid, obj: Object) -> Set[int]:
    """Get all colors present in an object."""
    return {grid[r][c] for r, c in obj.cells}


def count_objects_by_color(objects: List[Object]) -> dict:
    """Count objects grouped by color."""
    counts = Counter(o.color for o in objects)
    return dict(counts)
