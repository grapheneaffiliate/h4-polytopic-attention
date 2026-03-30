"""
TransformComposer — UCB1-guided search over primitive compositions.

Given training examples (input→output), finds a chain of primitives
that transforms every input into its corresponding output.

Search: depth 1 → 2 → 3, scored by cell accuracy, pruned by timeout.
"""

import time
import math
import numpy as np
from typing import Optional, Callable
from collections import defaultdict


class TransformChain:
    """Ordered sequence of (fn_name, fn, params) that transforms any grid."""

    def __init__(self, steps=None):
        self.steps = steps or []

    def apply(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        for name, fn, params in self.steps:
            try:
                result = fn(result, **params) if params else fn(result)
                if not isinstance(result, np.ndarray):
                    result = np.array(result, dtype=int)
            except Exception:
                return grid  # abort on error
        return result

    def describe(self) -> str:
        parts = []
        for name, fn, params in self.steps:
            if params:
                p = ", ".join(f"{k}={v}" for k, v in params.items())
                parts.append(f"{name}({p})")
            else:
                parts.append(name)
        return " → ".join(parts) if parts else "(identity)"

    def __len__(self):
        return len(self.steps)

    def __repr__(self):
        return f"TransformChain({self.describe()})"


def cell_accuracy(predicted: np.ndarray, target: np.ndarray) -> float:
    """Cell-level accuracy between two grids."""
    if predicted.shape != target.shape:
        return 0.0
    return float(np.sum(predicted == target)) / max(target.size, 1)


def score_chain(chain: TransformChain,
                examples: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Score a chain by mean cell accuracy across all examples."""
    if not examples:
        return 0.0
    total = 0.0
    for inp, out in examples:
        try:
            predicted = chain.apply(inp)
            total += cell_accuracy(predicted, out)
        except Exception:
            pass
    return total / len(examples)


class TransformComposer:
    """
    UCB1-guided composition search over primitives.

    Strategy:
    1. Depth 1: try every parameterized primitive. Score by accuracy.
    2. Depth 2: top-K depth-1, compose with every primitive.
    3. Depth 3: only if needed.

    Uses analysis hints to order primitives (most likely first).
    """

    def __init__(self, max_depth: int = 3, timeout_seconds: float = 30.0,
                 top_k: int = 10):
        self.max_depth = max_depth
        self.timeout = timeout_seconds
        self.top_k = top_k
        self._primitives = None  # lazy load

    def _load_primitives(self):
        """Load primitives from primitives.py."""
        if self._primitives is not None:
            return
        try:
            from . import primitives as P
            self._primitives = {}
            # Get all callable functions that aren't private
            for name in dir(P):
                if name.startswith('_'):
                    continue
                obj = getattr(P, name)
                if callable(obj) and name not in (
                    'most_common_color', 'least_common_color', 'count_colors',
                    'connected_components', 'bounding_box',
                    'select_by_size', 'select_by_color',
                ):
                    self._primitives[name] = obj
        except ImportError:
            self._primitives = {}

    def solve(self, examples: list[tuple[np.ndarray, np.ndarray]],
              priority_primitives: list[str] = None) -> Optional[TransformChain]:
        """
        Find a chain of primitives that maps every input to its output.

        Args:
            examples: list of (input_grid, output_grid) pairs
            priority_primitives: ordered list of primitive names to try first

        Returns:
            TransformChain if found (1.0 accuracy on all examples), else None
        """
        self._load_primitives()
        if not self._primitives:
            return None

        t0 = time.time()

        # Build parameterized candidates
        candidates = self._parameterize(examples, priority_primitives)

        # Depth 1
        scored_d1 = []
        for name, fn, params in candidates:
            if time.time() - t0 > self.timeout:
                break
            chain = TransformChain([(name, fn, params)])
            sc = score_chain(chain, examples)
            if sc >= 1.0:
                return chain
            if sc > 0.0:
                scored_d1.append((sc, chain))

        if self.max_depth < 2:
            return None

        # Depth 2: compose top-K from depth 1 with all candidates
        scored_d1.sort(key=lambda x: x[0], reverse=True)
        top = scored_d1[:self.top_k]

        scored_d2 = []
        for base_sc, base_chain in top:
            for name, fn, params in candidates:
                if time.time() - t0 > self.timeout:
                    return self._best_partial(scored_d1 + scored_d2)
                new_steps = base_chain.steps + [(name, fn, params)]
                chain = TransformChain(new_steps)
                sc = score_chain(chain, examples)
                if sc >= 1.0:
                    return chain
                if sc > base_sc:  # only keep if improved
                    scored_d2.append((sc, chain))

        if self.max_depth < 3:
            return self._best_partial(scored_d1 + scored_d2)

        # Depth 3: compose top-K from depth 2
        scored_d2.sort(key=lambda x: x[0], reverse=True)
        top2 = scored_d2[:self.top_k]

        for base_sc, base_chain in top2:
            for name, fn, params in candidates:
                if time.time() - t0 > self.timeout:
                    return self._best_partial(scored_d1 + scored_d2)
                new_steps = base_chain.steps + [(name, fn, params)]
                chain = TransformChain(new_steps)
                sc = score_chain(chain, examples)
                if sc >= 1.0:
                    return chain

        return self._best_partial(scored_d1 + scored_d2)

    def _best_partial(self, scored):
        """Return best partial match, or None if no progress."""
        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        best_sc, best_chain = scored[0]
        return best_chain if best_sc > 0.3 else None

    def _parameterize(self, examples, priority_primitives=None):
        """
        Generate (name, fn, params) tuples for all primitives with
        appropriate parameter values inferred from examples.
        """
        self._load_primitives()
        candidates = []

        # Collect colors present in examples
        all_input_colors = set()
        all_output_colors = set()
        for inp, out in examples:
            all_input_colors.update(np.unique(inp).tolist())
            all_output_colors.update(np.unique(out).tolist())
        all_colors = all_input_colors | all_output_colors

        # Order primitives: priority first, then alphabetical
        ordered_names = []
        if priority_primitives:
            for name in priority_primitives:
                if name in self._primitives:
                    ordered_names.append(name)
        for name in sorted(self._primitives.keys()):
            if name not in ordered_names:
                ordered_names.append(name)

        for name in ordered_names:
            fn = self._primitives[name]

            # No-param primitives
            if name in (
                'rotate_90', 'rotate_180', 'rotate_270',
                'reflect_horizontal', 'reflect_vertical',
                'reflect_diagonal_main', 'reflect_diagonal_anti',
                'transpose', 'recolor_by_rank',
                'border_cells', 'interior_cells',
                'largest_component', 'smallest_component',
            ):
                candidates.append((name, fn, {}))

            # Direction-param primitives
            elif name in ('shift_up', 'shift_down', 'shift_left', 'shift_right'):
                candidates.append((name, fn, {'wrap': False}))
                candidates.append((name, fn, {'wrap': True}))

            elif name in ('gravity_down', 'gravity_up', 'gravity_left', 'gravity_right'):
                for bg in [0] + list(all_colors - {0}):
                    candidates.append((name, fn, {'background': bg}))

            elif name == 'fill_enclosed':
                for bg in [0] + list(all_colors - {0}):
                    candidates.append((name, fn, {'background': bg}))

            # Color-param primitives
            elif name == 'color_swap':
                colors = sorted(all_colors)
                for i, a in enumerate(colors):
                    for b in colors[i+1:]:
                        candidates.append((name, fn, {'a': a, 'b': b}))

            elif name == 'replace_color':
                for old in all_input_colors:
                    for new in all_output_colors:
                        if old != new:
                            candidates.append((name, fn, {'old': old, 'new': new}))

            elif name == 'color_map':
                # Try simple 1:1 remapping inferred from first example
                if examples:
                    inp0, out0 = examples[0]
                    if inp0.shape == out0.shape:
                        mapping = {}
                        valid = True
                        for r in range(inp0.shape[0]):
                            for c in range(inp0.shape[1]):
                                ic, oc = int(inp0[r, c]), int(out0[r, c])
                                if ic in mapping and mapping[ic] != oc:
                                    valid = False
                                    break
                                mapping[ic] = oc
                            if not valid:
                                break
                        if valid and mapping:
                            candidates.append((name, fn, {'mapping': mapping}))

            elif name == 'invert_colors':
                candidates.append((name, fn, {'max_color': 9}))

            # Scaling
            elif name in ('scale_up_2x', 'scale_up_3x', 'scale_down_2x'):
                candidates.append((name, fn, {}))

            elif name == 'tile':
                for rows in [1, 2, 3]:
                    for cols in [1, 2, 3]:
                        if rows == 1 and cols == 1:
                            continue
                        candidates.append((name, fn, {'rows': rows, 'cols': cols}))

            # Crop/pad
            elif name == 'crop_to_content':
                for bg in [0] + list(all_colors - {0}):
                    candidates.append((name, fn, {'background': bg}))

            elif name == 'pad':
                for sz in [1, 2]:
                    for fill in [0]:
                        candidates.append((name, fn, {
                            'top': sz, 'bottom': sz,
                            'left': sz, 'right': sz, 'fill': fill
                        }))

            elif name == 'resize':
                if examples:
                    oh, ow = examples[0][1].shape
                    candidates.append((name, fn, {'new_h': oh, 'new_w': ow}))

            elif name == 'crop':
                # Try cropping to output size from various positions
                if examples:
                    oh, ow = examples[0][1].shape
                    ih, iw = examples[0][0].shape
                    for r1 in range(max(1, ih - oh + 1)):
                        for c1 in range(max(1, iw - ow + 1)):
                            candidates.append((name, fn, {
                                'r1': r1, 'c1': c1,
                                'r2': r1 + oh, 'c2': c1 + ow
                            }))

            # Pattern
            elif name == 'mirror_extend':
                for d in ['up', 'down', 'left', 'right']:
                    candidates.append((name, fn, {'direction': d}))

            elif name == 'symmetrize':
                for ax in ['horizontal', 'vertical']:
                    candidates.append((name, fn, {'axis': ax}))

            elif name == 'repeat_pattern':
                for ax in [0, 1]:
                    for cnt in [2, 3]:
                        candidates.append((name, fn, {'axis': ax, 'count': cnt}))

            elif name == 'draw_border':
                for c in all_output_colors:
                    candidates.append((name, fn, {'color': c, 'thickness': 1}))

            # mask_by_color, extract_by_color, fill_region
            elif name == 'mask_by_color':
                for c in all_colors:
                    candidates.append((name, fn, {'color': c}))

            elif name == 'extract_by_color':
                for c in all_colors:
                    if c != 0:
                        candidates.append((name, fn, {'color': c}))

            elif name == 'fill_region':
                pass  # needs mask, skip in auto-param

            elif name == 'flood_fill':
                pass  # needs coordinates, skip

            elif name == 'draw_line':
                pass  # needs coordinates, skip

            elif name == 'where':
                pass  # needs mask, skip

            elif name == 'apply_to_each_component':
                pass  # needs fn, skip in auto-param

            # NEW conditional/spatial primitives
            elif name == 'fill_enclosed_with_color':
                for fc in all_output_colors:
                    candidates.append((name, fn, {'fill_color': fc, 'background': 0}))

            elif name == 'fill_enclosed_per_region':
                candidates.append((name, fn, {'background': 0}))

            elif name == 'fill_adjacent_to_color':
                for near in all_input_colors - {0}:
                    for fc in all_output_colors - all_input_colors:
                        candidates.append((name, fn, {'target_color': 0, 'near_color': near, 'fill_color': fc}))
                    for fc in all_output_colors:
                        if fc != near:
                            candidates.append((name, fn, {'target_color': 0, 'near_color': near, 'fill_color': fc}))

            elif name == 'draw_line_between_same_color':
                for c in all_input_colors - {0}:
                    candidates.append((name, fn, {'color': c, 'line_color': c}))
                    for lc in all_output_colors - {0, c}:
                        candidates.append((name, fn, {'color': c, 'line_color': lc}))

            elif name == 'move_object_until_wall':
                for obj_c in all_input_colors - {0}:
                    for wall_c in all_input_colors - {0, obj_c}:
                        for d in ['down', 'up', 'left', 'right']:
                            candidates.append((name, fn, {'obj_color': obj_c, 'wall_color': wall_c, 'direction': d}))

            elif name in ('extract_subgrid_by_color', 'extract_subgrid_by_color_inclusive'):
                for c in all_input_colors - {0}:
                    candidates.append((name, fn, {'border_color': c}))

            elif name == 'recolor_by_size':
                for sc in all_output_colors - {0}:
                    for lc in all_output_colors - {0, sc}:
                        candidates.append((name, fn, {'small_color': sc, 'large_color': lc}))

            elif name == 'copy_pattern_to_markers':
                for pc in all_input_colors - {0}:
                    for mc in all_input_colors - {0, pc}:
                        candidates.append((name, fn, {'pattern_color': pc, 'marker_color': mc}))

            else:
                # Try with no params
                try:
                    candidates.append((name, fn, {}))
                except Exception:
                    pass

        return candidates
