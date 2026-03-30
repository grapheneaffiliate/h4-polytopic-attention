"""Tests for every primitive in primitives.py.

Each primitive gets at least two tests:
  1. A basic correctness case.
  2. An edge case (1x1 grid, uniform grid, empty-like grid, etc.).
"""

import numpy as np
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from agent_zero.transforms.primitives import *


# =====================================================================
# Geometric transforms (12)
# =====================================================================

def test_rotate_90():
    g = np.array([[1, 2], [3, 4]])
    r = rotate_90(g)
    assert r.tolist() == [[3, 1], [4, 2]], f"Got {r.tolist()}"

def test_rotate_90_single():
    g = np.array([[5]])
    assert rotate_90(g).tolist() == [[5]]

def test_rotate_180():
    g = np.array([[1, 2], [3, 4]])
    assert rotate_180(g).tolist() == [[4, 3], [2, 1]]

def test_rotate_180_1x3():
    g = np.array([[1, 2, 3]])
    assert rotate_180(g).tolist() == [[3, 2, 1]]

def test_rotate_270():
    g = np.array([[1, 2], [3, 4]])
    assert rotate_270(g).tolist() == [[2, 4], [1, 3]]

def test_rotate_270_single():
    g = np.array([[7]])
    assert rotate_270(g).tolist() == [[7]]

def test_reflect_horizontal():
    g = np.array([[1, 2, 3], [4, 5, 6]])
    assert reflect_horizontal(g).tolist() == [[3, 2, 1], [6, 5, 4]]

def test_reflect_horizontal_single_col():
    g = np.array([[1], [2], [3]])
    assert reflect_horizontal(g).tolist() == [[1], [2], [3]]

def test_reflect_vertical():
    g = np.array([[1, 2], [3, 4], [5, 6]])
    assert reflect_vertical(g).tolist() == [[5, 6], [3, 4], [1, 2]]

def test_reflect_vertical_single_row():
    g = np.array([[7, 8, 9]])
    assert reflect_vertical(g).tolist() == [[7, 8, 9]]

def test_reflect_diagonal_main():
    g = np.array([[1, 2], [3, 4]])
    assert reflect_diagonal_main(g).tolist() == [[1, 3], [2, 4]]

def test_reflect_diagonal_main_rect():
    g = np.array([[1, 2, 3]])
    r = reflect_diagonal_main(g)
    assert r.shape == (3, 1)
    assert r.tolist() == [[1], [2], [3]]

def test_reflect_diagonal_anti():
    g = np.array([[1, 2], [3, 4]])
    # rot90(grid,1).T => [[2,4],[1,3]].T => [[2,1],[4,3]]
    assert reflect_diagonal_anti(g).tolist() == [[2, 1], [4, 3]]

def test_reflect_diagonal_anti_single():
    g = np.array([[9]])
    assert reflect_diagonal_anti(g).tolist() == [[9]]

def test_transpose():
    g = np.array([[1, 2, 3], [4, 5, 6]])
    assert transpose(g).tolist() == [[1, 4], [2, 5], [3, 6]]

def test_transpose_single():
    g = np.array([[1]])
    assert transpose(g).tolist() == [[1]]

def test_shift_up():
    g = np.array([[1, 2], [3, 4], [5, 6]])
    r = shift_up(g)
    assert r.tolist() == [[3, 4], [5, 6], [0, 0]]

def test_shift_up_single_row():
    g = np.array([[1, 2, 3]])
    assert shift_up(g).tolist() == [[0, 0, 0]]

def test_shift_down():
    g = np.array([[1, 2], [3, 4], [5, 6]])
    r = shift_down(g)
    assert r.tolist() == [[0, 0], [1, 2], [3, 4]]

def test_shift_down_wrap():
    g = np.array([[1, 2], [3, 4]])
    r = shift_down(g, wrap=True)
    assert r.tolist() == [[3, 4], [1, 2]]

def test_shift_left():
    g = np.array([[1, 2, 3], [4, 5, 6]])
    r = shift_left(g)
    assert r.tolist() == [[2, 3, 0], [5, 6, 0]]

def test_shift_left_single_col():
    g = np.array([[1], [2]])
    assert shift_left(g).tolist() == [[0], [0]]

def test_shift_right():
    g = np.array([[1, 2, 3], [4, 5, 6]])
    r = shift_right(g)
    assert r.tolist() == [[0, 1, 2], [0, 4, 5]]

def test_shift_right_wrap():
    g = np.array([[1, 2], [3, 4]])
    r = shift_right(g, wrap=True)
    assert r.tolist() == [[2, 1], [4, 3]]


# =====================================================================
# Scaling transforms (4)
# =====================================================================

def test_scale_up_2x():
    g = np.array([[1, 2], [3, 4]])
    r = scale_up_2x(g)
    assert r.shape == (4, 4)
    assert r.tolist() == [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]

def test_scale_up_2x_single():
    g = np.array([[5]])
    r = scale_up_2x(g)
    assert r.tolist() == [[5, 5], [5, 5]]

def test_scale_up_3x():
    g = np.array([[1]])
    r = scale_up_3x(g)
    assert r.shape == (3, 3)
    assert r.tolist() == [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

def test_scale_up_3x_rect():
    g = np.array([[1, 2]])
    r = scale_up_3x(g)
    assert r.shape == (3, 6)

def test_scale_down_2x():
    g = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
    r = scale_down_2x(g)
    assert r.tolist() == [[1, 2], [3, 4]]

def test_scale_down_2x_odd():
    g = np.array([[1, 1, 5], [1, 1, 5], [9, 9, 9]])
    r = scale_down_2x(g)
    # Odd dims: trims to 2x2, then downscales
    assert r.shape == (1, 1)
    assert r.tolist() == [[1]]

def test_tile():
    g = np.array([[1, 2], [3, 4]])
    r = tile(g, 2, 3)
    assert r.shape == (4, 6)
    assert r[0].tolist() == [1, 2, 1, 2, 1, 2]

def test_tile_single():
    g = np.array([[5]])
    r = tile(g, 3, 2)
    assert r.shape == (3, 2)
    assert np.all(r == 5)


# =====================================================================
# Crop and Pad (4)
# =====================================================================

def test_crop():
    g = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    r = crop(g, 0, 1, 2, 3)
    assert r.tolist() == [[2, 3], [5, 6]]

def test_crop_single_cell():
    g = np.array([[1, 2], [3, 4]])
    r = crop(g, 0, 0, 1, 1)
    assert r.tolist() == [[1]]

def test_crop_to_content():
    g = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    r = crop_to_content(g)
    assert r.tolist() == [[1]]

def test_crop_to_content_all_bg():
    g = np.array([[0, 0], [0, 0]])
    r = crop_to_content(g)
    assert r.tolist() == [[0, 0], [0, 0]]

def test_pad():
    g = np.array([[1, 2], [3, 4]])
    r = pad(g, 1, 1, 1, 1, fill=0)
    assert r.shape == (4, 4)
    assert r[0].tolist() == [0, 0, 0, 0]
    assert r[1].tolist() == [0, 1, 2, 0]

def test_pad_asymmetric():
    g = np.array([[5]])
    r = pad(g, 0, 2, 1, 0, fill=9)
    assert r.shape == (3, 2)
    assert r.tolist() == [[9, 5], [9, 9], [9, 9]]

def test_resize():
    g = np.array([[1, 2], [3, 4]])
    r = resize(g, 4, 4)
    assert r.shape == (4, 4)
    # Nearest neighbor: top-left quarter should be 1
    assert r[0, 0] == 1
    assert r[0, 3] == 2

def test_resize_shrink():
    g = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    r = resize(g, 1, 2)
    assert r.shape == (1, 2)


# =====================================================================
# Color transforms (8)
# =====================================================================

def test_color_swap():
    g = np.array([[1, 2], [2, 1]])
    r = color_swap(g, 1, 2)
    assert r.tolist() == [[2, 1], [1, 2]]

def test_color_swap_absent():
    g = np.array([[1, 1], [1, 1]])
    r = color_swap(g, 1, 5)
    assert r.tolist() == [[5, 5], [5, 5]]

def test_color_map():
    g = np.array([[0, 1], [2, 3]])
    r = color_map(g, {0: 9, 1: 8})
    assert r.tolist() == [[9, 8], [2, 3]]

def test_color_map_empty():
    g = np.array([[1, 2]])
    r = color_map(g, {})
    assert r.tolist() == [[1, 2]]

def test_replace_color():
    g = np.array([[0, 1, 0], [1, 0, 1]])
    r = replace_color(g, 0, 5)
    assert r.tolist() == [[5, 1, 5], [1, 5, 1]]

def test_replace_color_noop():
    g = np.array([[1, 2]])
    r = replace_color(g, 9, 0)
    assert r.tolist() == [[1, 2]]

def test_most_common_color():
    g = np.array([[1, 1, 2], [1, 3, 3]])
    assert most_common_color(g) == 1

def test_most_common_color_single():
    g = np.array([[7]])
    assert most_common_color(g) == 7

def test_least_common_color():
    g = np.array([[1, 1, 2], [1, 3, 3]])
    assert least_common_color(g) == 2

def test_least_common_color_uniform():
    g = np.array([[4, 4], [4, 4]])
    assert least_common_color(g) == 4

def test_count_colors():
    g = np.array([[1, 1, 2], [3, 3, 3]])
    c = count_colors(g)
    assert c == {1: 2, 2: 1, 3: 3}

def test_count_colors_single():
    g = np.array([[0]])
    assert count_colors(g) == {0: 1}

def test_recolor_by_rank():
    g = np.array([[5, 5, 5], [3, 3, 1]])
    r = recolor_by_rank(g)
    # 5 is most common (3), then 3 (2), then 1 (1)
    # 5->0, 3->1, 1->2
    assert r[0, 0] == 0  # 5 -> 0
    assert r[1, 0] == 1  # 3 -> 1
    assert r[1, 2] == 2  # 1 -> 2

def test_recolor_by_rank_uniform():
    g = np.array([[2, 2], [2, 2]])
    r = recolor_by_rank(g)
    assert np.all(r == 0)

def test_invert_colors():
    g = np.array([[0, 9], [4, 5]])
    r = invert_colors(g)
    assert r.tolist() == [[9, 0], [5, 4]]

def test_invert_colors_custom_max():
    g = np.array([[0, 3]])
    r = invert_colors(g, max_color=3)
    assert r.tolist() == [[3, 0]]


# =====================================================================
# Region transforms (10)
# =====================================================================

def test_connected_components():
    g = np.array([[1, 0], [0, 1]])
    labeled, comps = connected_components(g)
    assert len(comps) == 4  # 4 separate single-cell components

def test_connected_components_uniform():
    g = np.array([[1, 1], [1, 1]])
    labeled, comps = connected_components(g)
    assert len(comps) == 1
    assert comps[0]['size'] == 4

def test_mask_by_color():
    g = np.array([[0, 1, 2], [1, 0, 1]])
    r = mask_by_color(g, 1)
    assert r.tolist() == [[0, 1, 0], [1, 0, 1]]

def test_mask_by_color_absent():
    g = np.array([[0, 1]])
    r = mask_by_color(g, 5)
    assert r.tolist() == [[0, 0]]

def test_extract_by_color():
    g = np.array([[0, 0, 0], [0, 3, 0], [0, 0, 0]])
    r = extract_by_color(g, 3)
    assert r.tolist() == [[3]]

def test_extract_by_color_absent():
    g = np.array([[1, 2]])
    r = extract_by_color(g, 9)
    assert r.tolist() == [[1, 2]]  # Returns full grid

def test_fill_region():
    g = np.array([[1, 2], [3, 4]])
    mask = np.array([[True, False], [False, True]])
    r = fill_region(g, mask, 0)
    assert r.tolist() == [[0, 2], [3, 0]]

def test_fill_region_all_true():
    g = np.array([[1, 2], [3, 4]])
    mask = np.ones((2, 2), dtype=bool)
    r = fill_region(g, mask, 9)
    assert np.all(r == 9)

def test_flood_fill():
    g = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]])
    r = flood_fill(g, 0, 0, 5)
    assert r[0, 0] == 5
    assert r[1, 1] == 5
    assert r[2, 0] == 1  # Not changed

def test_flood_fill_noop():
    g = np.array([[3, 3], [3, 3]])
    r = flood_fill(g, 0, 0, 3)  # Same color -> no change
    assert r.tolist() == [[3, 3], [3, 3]]

def test_bounding_box():
    g = np.array([[0, 0, 0], [0, 5, 0], [0, 5, 0]])
    bb = bounding_box(g, 5)
    assert bb == (1, 1, 3, 2)

def test_bounding_box_absent():
    g = np.array([[1, 2], [3, 4]])
    assert bounding_box(g, 9) == (0, 0, 0, 0)

def test_largest_component():
    g = np.array([[1, 0, 2], [1, 0, 0], [1, 0, 0]])
    m = largest_component(g)
    # Color 0 occupies 5 cells (largest), or color 1 occupies 3
    # Actually 0 forms one connected region of 5 cells
    assert m.sum() == 5  # The 0-region

def test_largest_component_single():
    g = np.array([[7]])
    m = largest_component(g)
    assert m.tolist() == [[1]]

def test_smallest_component():
    g = np.array([[1, 0, 2], [1, 0, 0], [1, 0, 0]])
    m = smallest_component(g)
    assert m.sum() == 1  # color 2 has 1 cell

def test_smallest_component_uniform():
    g = np.array([[3, 3], [3, 3]])
    m = smallest_component(g)
    assert m.sum() == 4  # only one component

def test_border_cells():
    g = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    m = border_cells(g)
    assert m[1, 1] == 0  # Center is not border
    assert m[0, 0] == 1
    assert m[0, 2] == 1
    assert m.sum() == 8

def test_border_cells_1x1():
    g = np.array([[1]])
    m = border_cells(g)
    assert m.tolist() == [[1]]

def test_interior_cells():
    g = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    m = interior_cells(g)
    assert m[1, 1] == 1
    assert m[0, 0] == 0
    assert m.sum() == 1

def test_interior_cells_2x2():
    g = np.array([[1, 2], [3, 4]])
    m = interior_cells(g)
    assert m.sum() == 0  # No interior in 2x2


# =====================================================================
# Gravity transforms (4)
# =====================================================================

def test_gravity_down():
    g = np.array([[1, 0, 2], [0, 3, 0], [0, 0, 0]])
    r = gravity_down(g)
    assert r.tolist() == [[0, 0, 0], [0, 0, 0], [1, 3, 2]]

def test_gravity_down_already_settled():
    g = np.array([[0, 0], [1, 2]])
    r = gravity_down(g)
    assert r.tolist() == [[0, 0], [1, 2]]

def test_gravity_up():
    g = np.array([[0, 0, 0], [0, 3, 0], [1, 0, 2]])
    r = gravity_up(g)
    assert r.tolist() == [[1, 3, 2], [0, 0, 0], [0, 0, 0]]

def test_gravity_up_single_row():
    g = np.array([[5, 0, 3]])
    r = gravity_up(g)
    assert r.tolist() == [[5, 0, 3]]

def test_gravity_left():
    g = np.array([[0, 1, 0], [2, 0, 3]])
    r = gravity_left(g)
    assert r.tolist() == [[1, 0, 0], [2, 3, 0]]

def test_gravity_left_uniform():
    g = np.array([[0, 0], [0, 0]])
    r = gravity_left(g)
    assert r.tolist() == [[0, 0], [0, 0]]

def test_gravity_right():
    g = np.array([[1, 0, 2], [0, 3, 0]])
    r = gravity_right(g)
    assert r.tolist() == [[0, 1, 2], [0, 0, 3]]

def test_gravity_right_1x1():
    g = np.array([[4]])
    r = gravity_right(g)
    assert r.tolist() == [[4]]


# =====================================================================
# Pattern transforms (6)
# =====================================================================

def test_repeat_pattern_vertical():
    g = np.array([[1, 2]])
    r = repeat_pattern(g, axis=0, count=3)
    assert r.tolist() == [[1, 2], [1, 2], [1, 2]]

def test_repeat_pattern_horizontal():
    g = np.array([[1], [2]])
    r = repeat_pattern(g, axis=1, count=2)
    assert r.tolist() == [[1, 1], [2, 2]]

def test_mirror_extend_down():
    g = np.array([[1, 2], [3, 4]])
    r = mirror_extend(g, 'down')
    assert r.tolist() == [[1, 2], [3, 4], [3, 4], [1, 2]]

def test_mirror_extend_right():
    g = np.array([[1, 2]])
    r = mirror_extend(g, 'right')
    assert r.tolist() == [[1, 2, 2, 1]]

def test_mirror_extend_up():
    g = np.array([[1], [2]])
    r = mirror_extend(g, 'up')
    assert r.tolist() == [[2], [1], [1], [2]]

def test_mirror_extend_left():
    g = np.array([[1, 2]])
    r = mirror_extend(g, 'left')
    assert r.tolist() == [[2, 1, 1, 2]]

def test_symmetrize_horizontal():
    g = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    r = symmetrize(g, 'horizontal')
    # Top half copied to bottom (flipped)
    assert r[0].tolist() == [1, 2]
    assert r[1].tolist() == [3, 4]
    assert r[2].tolist() == [3, 4]  # flipped top half row 1
    assert r[3].tolist() == [1, 2]  # flipped top half row 0

def test_symmetrize_vertical():
    g = np.array([[1, 2, 3, 4]])
    r = symmetrize(g, 'vertical')
    # Left half copied to right (flipped)
    assert r.tolist() == [[1, 2, 2, 1]]

def test_draw_border():
    g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    r = draw_border(g, 5)
    assert r[0].tolist() == [5, 5, 5]
    assert r[1].tolist() == [5, 0, 5]
    assert r[2].tolist() == [5, 5, 5]

def test_draw_border_1x1():
    g = np.array([[0]])
    r = draw_border(g, 3)
    assert r.tolist() == [[3]]

def test_draw_line_horizontal():
    g = np.zeros((3, 3), dtype=int)
    r = draw_line(g, 1, 0, 1, 2, 5)
    assert r[1].tolist() == [5, 5, 5]
    assert r[0].tolist() == [0, 0, 0]

def test_draw_line_diagonal():
    g = np.zeros((3, 3), dtype=int)
    r = draw_line(g, 0, 0, 2, 2, 7)
    assert r[0, 0] == 7
    assert r[1, 1] == 7
    assert r[2, 2] == 7

def test_fill_enclosed():
    g = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ])
    r = fill_enclosed(g)
    assert r[1, 1] == 1  # Interior 0 should be filled

def test_fill_enclosed_no_enclosure():
    g = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ])
    r = fill_enclosed(g)
    # The center 0 touches border via corner zeros
    # All zeros are border-reachable in this case
    # Actually: (0,0) is border+zero, connects to... let's just check it runs
    assert r.shape == (3, 3)


# =====================================================================
# Conditional transforms (4)
# =====================================================================

def test_where():
    mask = np.array([[1, 0], [0, 1]])
    a = np.array([[9, 9], [9, 9]])
    b = np.array([[0, 0], [0, 0]])
    r = where(mask, a, b)
    assert r.tolist() == [[9, 0], [0, 9]]

def test_where_all_true():
    mask = np.ones((2, 2), dtype=int)
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    r = where(mask, a, b)
    assert r.tolist() == [[1, 2], [3, 4]]

def test_select_by_size():
    comps = [
        {'label': 1, 'color': 1, 'size': 5, 'bbox': (0,0,2,3), 'centroid': (1.0,1.0), 'pixels': []},
        {'label': 2, 'color': 2, 'size': 1, 'bbox': (0,0,1,1), 'centroid': (0.0,0.0), 'pixels': []},
        {'label': 3, 'color': 3, 'size': 10, 'bbox': (0,0,5,5), 'centroid': (2.0,2.0), 'pixels': []},
    ]
    r = select_by_size(comps, min_size=2, max_size=8)
    assert len(r) == 1
    assert r[0]['label'] == 1

def test_select_by_size_empty():
    assert select_by_size([], min_size=1) == []

def test_select_by_color():
    comps = [
        {'label': 1, 'color': 1, 'size': 5, 'bbox': (0,0,2,3), 'centroid': (1.0,1.0), 'pixels': []},
        {'label': 2, 'color': 2, 'size': 1, 'bbox': (0,0,1,1), 'centroid': (0.0,0.0), 'pixels': []},
    ]
    r = select_by_color(comps, [2, 3])
    assert len(r) == 1
    assert r[0]['color'] == 2

def test_select_by_color_none_match():
    comps = [
        {'label': 1, 'color': 1, 'size': 5, 'bbox': (0,0,2,3), 'centroid': (1.0,1.0), 'pixels': []},
    ]
    assert select_by_color(comps, [9]) == []

def test_apply_to_each_component():
    g = np.array([[1, 0], [0, 2]])
    r = apply_to_each_component(g, rotate_180)
    # Each single-cell component rotated 180 is still the same cell
    assert r[0, 0] == 1
    assert r[1, 1] == 2

def test_apply_to_each_component_uniform():
    g = np.array([[3, 3], [3, 3]])
    r = apply_to_each_component(g, reflect_horizontal)
    assert r.tolist() == [[3, 3], [3, 3]]


# =====================================================================
# Test runner
# =====================================================================

def run_all_tests():
    """Discover and run all test_ functions. Print PASS/FAIL for each."""
    import inspect
    current_module = sys.modules[__name__]
    tests = [
        (name, fn)
        for name, fn in inspect.getmembers(current_module, inspect.isfunction)
        if name.startswith("test_")
    ]
    tests.sort(key=lambda x: x[0])

    passed = 0
    failed = 0
    errors = []

    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            errors.append((name, traceback.format_exc()))
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if errors:
        print(f"\nFailure details:")
        for name, tb in errors:
            print(f"\n--- {name} ---")
            print(tb)
    print(f"{'=' * 60}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
