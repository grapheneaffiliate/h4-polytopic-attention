"""
Pre-analysis that compares input/output grid pairs and extracts structural
hints to narrow the transformation search space.
"""

from __future__ import annotations

import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Tuple


class GridAnalyzer:
    """Compare ARC input/output pairs and extract structural hints."""

    # ── Single-pair analysis ────────────────────────────────

    def analyze(self, input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
        """Compare input and output. Return hints about what changed."""
        hints: dict = {}

        ih, iw = input_grid.shape
        oh, ow = output_grid.shape

        # --- size_change ---
        hints["size_change"] = self._classify_size_change(ih, iw, oh, ow)

        # --- color_changes ---
        hints["color_changes"] = self._detect_color_remap(input_grid, output_grid)

        # --- symmetry ---
        hints["symmetry_input"] = self._check_symmetry(input_grid)
        hints["symmetry_output"] = self._check_symmetry(output_grid)

        # --- content_overlap ---
        if ih == oh and iw == ow:
            total = ih * iw
            hints["content_overlap"] = float(np.sum(input_grid == output_grid)) / total if total > 0 else 1.0
        else:
            hints["content_overlap"] = None

        # --- geometric_match ---
        hints["geometric_match"] = self._detect_geometric(input_grid, output_grid)

        # --- border_added ---
        hints["border_added"] = self._detect_border_added(input_grid, output_grid)

        # --- is_tiled ---
        hints["is_tiled"] = self._detect_tiling(input_grid, output_grid)

        # --- gravity_match ---
        hints["gravity_match"] = self._detect_gravity(input_grid, output_grid)

        return hints

    # ── Multi-example consensus ─────────────────────────────

    def analyze_examples(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> dict:
        """Analyze ALL training examples and return consensus hints.

        Only hints that are CONSISTENT across every example are kept.
        """
        if not examples:
            return {}

        all_hints = [self.analyze(inp, out) for inp, out in examples]

        consensus: dict = {}

        # size_change: must agree
        vals = [h["size_change"] for h in all_hints]
        consensus["size_change"] = vals[0] if len(set(vals)) == 1 else "mixed"

        # color_changes: must be the same mapping for every pair
        cmaps = [h["color_changes"] for h in all_hints]
        if all(cm is not None for cm in cmaps):
            ref = cmaps[0]
            if all(cm == ref for cm in cmaps):
                consensus["color_changes"] = ref
            else:
                consensus["color_changes"] = None
        else:
            consensus["color_changes"] = None

        # symmetry
        sym_in = [h["symmetry_input"] for h in all_hints]
        sym_out = [h["symmetry_output"] for h in all_hints]
        consensus["symmetry_input"] = sym_in[0] if len(set(sym_in)) == 1 else "mixed"
        consensus["symmetry_output"] = sym_out[0] if len(set(sym_out)) == 1 else "mixed"

        # content_overlap: report min across examples (conservative)
        overlaps = [h["content_overlap"] for h in all_hints if h["content_overlap"] is not None]
        consensus["content_overlap"] = min(overlaps) if overlaps else None

        # geometric_match: must agree
        geos = [h["geometric_match"] for h in all_hints]
        consensus["geometric_match"] = geos[0] if len(set(geos)) == 1 else None

        # border_added: must all be True
        consensus["border_added"] = all(h["border_added"] for h in all_hints)

        # is_tiled: must all be True
        consensus["is_tiled"] = all(h["is_tiled"] for h in all_hints)

        # gravity_match: must agree
        gravs = [h["gravity_match"] for h in all_hints]
        consensus["gravity_match"] = gravs[0] if len(set(gravs)) == 1 else None

        return consensus

    # ── Suggest primitives ──────────────────────────────────

    def suggest_primitives(self, analysis: dict) -> List[str]:
        """Given analysis, return ORDERED list of primitive names to try first."""
        suggestions: list[str] = []

        geo = analysis.get("geometric_match")
        if geo is not None:
            suggestions.append(geo)

        grav = analysis.get("gravity_match")
        if grav is not None:
            suggestions.append(grav)

        if analysis.get("color_changes") is not None:
            suggestions.extend(["color_map", "color_swap", "replace_color"])

        size = analysis.get("size_change", "same")
        if size == "same":
            overlap = analysis.get("content_overlap")
            if overlap is not None and overlap > 0.8:
                # High overlap + same size -> mostly color operations
                if "color_map" not in suggestions:
                    suggestions.extend(["replace_color", "color_swap", "color_map"])
                suggestions.extend(["draw_border", "fill_enclosed", "flood_fill"])
            elif overlap is not None and overlap > 0.5:
                suggestions.extend(["fill_enclosed", "draw_border", "flood_fill",
                                    "complete_symmetry_h", "complete_symmetry_v"])
            else:
                suggestions.extend(["gravity_down", "gravity_up", "gravity_left",
                                    "gravity_right", "rotate_90", "rotate_180",
                                    "rotate_270", "reflect_horizontal", "reflect_vertical",
                                    "transpose"])

        elif size == "doubled_h":
            suggestions.extend(["mirror_extend", "repeat_pattern", "tile"])
        elif size == "doubled_w":
            suggestions.extend(["mirror_extend", "repeat_pattern", "tile"])
        elif size == "doubled_both":
            suggestions.extend(["mirror_extend", "tile", "scale_up_2x"])
        elif size == "halved_h" or size == "halved_w":
            suggestions.extend(["crop_to_content", "crop", "scale_down_2x"])
        elif size == "tripled_both":
            suggestions.extend(["scale_up_3x", "tile"])
        else:
            # arbitrary size change
            suggestions.extend(["crop_to_content", "crop", "resize",
                                "extract_by_color", "tile"])

        if analysis.get("border_added"):
            if "draw_border" not in suggestions:
                suggestions.insert(0, "draw_border")

        if analysis.get("is_tiled"):
            if "tile" not in suggestions:
                suggestions.insert(0, "tile")

        sym_out = analysis.get("symmetry_output", "none")
        if sym_out in ("horizontal", "both"):
            if "symmetrize" not in suggestions:
                suggestions.append("symmetrize")
        if sym_out in ("vertical", "both"):
            if "symmetrize" not in suggestions:
                suggestions.append("symmetrize")

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        return deduped

    # ── Private helpers ─────────────────────────────────────

    @staticmethod
    def _classify_size_change(ih: int, iw: int, oh: int, ow: int) -> str:
        if ih == oh and iw == ow:
            return "same"
        if oh == 2 * ih and ow == 2 * iw:
            return "doubled_both"
        if oh == 2 * ih and ow == iw:
            return "doubled_h"
        if oh == ih and ow == 2 * iw:
            return "doubled_w"
        if oh == 3 * ih and ow == 3 * iw:
            return "tripled_both"
        if oh * 2 == ih and ow == iw:
            return "halved_h"
        if oh == ih and ow * 2 == iw:
            return "halved_w"
        return "arbitrary"

    @staticmethod
    def _detect_color_remap(inp: np.ndarray, out: np.ndarray) -> Optional[Dict[int, int]]:
        """Detect a simple 1:1 color remapping. Returns None if not a clean remap."""
        if inp.shape != out.shape:
            return None
        mapping: dict[int, int] = {}
        ih, iw = inp.shape
        for r in range(ih):
            for c in range(iw):
                ic = int(inp[r, c])
                oc = int(out[r, c])
                if ic in mapping:
                    if mapping[ic] != oc:
                        return None
                else:
                    mapping[ic] = oc
        # Check injectivity (1:1)
        if len(set(mapping.values())) != len(mapping):
            # Not injective -- still a valid remap, just not invertible
            pass
        # If it is the identity, return None (not interesting)
        if all(k == v for k, v in mapping.items()):
            return None
        return mapping

    @staticmethod
    def _check_symmetry(grid: np.ndarray) -> str:
        h, w = grid.shape
        h_sym = np.array_equal(grid, np.fliplr(grid))
        v_sym = np.array_equal(grid, np.flipud(grid))
        rot_sym = np.array_equal(grid, np.rot90(grid, 2)) if h == w else False

        if h_sym and v_sym:
            return "both"
        if h_sym:
            return "horizontal"
        if v_sym:
            return "vertical"
        # Check rotational (180) for non-square grids too
        if h == w:
            if rot_sym:
                return "rotational"
            # Check 90-degree rotational symmetry
            if np.array_equal(grid, np.rot90(grid, -1)):
                return "rotational"
        return "none"

    @staticmethod
    def _detect_geometric(inp: np.ndarray, out: np.ndarray) -> Optional[str]:
        """Check if output matches a simple geometric transform of input."""
        if np.array_equal(np.rot90(inp, -1), out):
            return "rotate_90"
        if np.array_equal(np.rot90(inp, 2), out):
            return "rotate_180"
        if np.array_equal(np.rot90(inp, -3), out):
            return "rotate_270"
        if np.array_equal(np.fliplr(inp), out):
            return "reflect_horizontal"
        if np.array_equal(np.flipud(inp), out):
            return "reflect_vertical"
        if inp.shape[0] == inp.shape[1] or (out.shape == (inp.shape[1], inp.shape[0])):
            if np.array_equal(inp.T, out):
                return "transpose"
        return None

    @staticmethod
    def _detect_border_added(inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if output is input wrapped in a uniform-color border."""
        ih, iw = inp.shape
        oh, ow = out.shape
        # Try border thicknesses 1, 2
        for t in (1, 2):
            if oh == ih + 2 * t and ow == iw + 2 * t:
                inner = out[t:oh - t, t:ow - t]
                if np.array_equal(inner, inp):
                    # Check border is uniform
                    border_vals = set()
                    border_vals.update(out[0, :].tolist())
                    border_vals.update(out[-1, :].tolist())
                    border_vals.update(out[:, 0].tolist())
                    border_vals.update(out[:, -1].tolist())
                    if len(border_vals) == 1:
                        return True
        return False

    @staticmethod
    def _detect_tiling(inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if output is a tiling (repeat) of input."""
        ih, iw = inp.shape
        oh, ow = out.shape
        if oh < ih or ow < iw:
            return False
        if oh % ih != 0 or ow % iw != 0:
            return False
        if oh == ih and ow == iw:
            return False  # Same size -- not a tiling
        reps_h = oh // ih
        reps_w = ow // iw
        tiled = np.tile(inp, (reps_h, reps_w))
        return np.array_equal(tiled, out)

    @staticmethod
    def _detect_gravity(inp: np.ndarray, out: np.ndarray) -> Optional[str]:
        """Check if output matches gravity applied to input in any direction."""
        if inp.shape != out.shape:
            return None
        h, w = inp.shape

        # gravity_down
        gd = np.zeros_like(inp)
        for c in range(w):
            col = inp[:, c]
            nz = col[col != 0]
            if nz.size > 0:
                gd[h - nz.size:, c] = nz
        if np.array_equal(gd, out):
            return "gravity_down"

        # gravity_up
        gu = np.zeros_like(inp)
        for c in range(w):
            col = inp[:, c]
            nz = col[col != 0]
            if nz.size > 0:
                gu[:nz.size, c] = nz
        if np.array_equal(gu, out):
            return "gravity_up"

        # gravity_left
        gl = np.zeros_like(inp)
        for r in range(h):
            row = inp[r, :]
            nz = row[row != 0]
            if nz.size > 0:
                gl[r, :nz.size] = nz
        if np.array_equal(gl, out):
            return "gravity_left"

        # gravity_right
        gr = np.zeros_like(inp)
        for r in range(h):
            row = inp[r, :]
            nz = row[row != 0]
            if nz.size > 0:
                gr[r, w - nz.size:] = nz
        if np.array_equal(gr, out):
            return "gravity_right"

        return None
