"""
TransformEngine — top-level orchestrator.

Ties together analyzer (narrow search), composer (find chain), and caching.
Integrates with Agent Zero as the first layer tried on each game level.
"""

import time
import numpy as np
from typing import Optional

from .composer import TransformComposer, TransformChain, score_chain


class TransformEngine:
    """
    Main entry point for the transformation search layer.

    Usage:
        engine = TransformEngine()
        examples = [(input1, output1), (input2, output2)]
        chain = engine.try_solve(examples, game_id="lp85_level_3")
        if chain:
            result = chain.apply(test_input)
    """

    def __init__(self, max_depth=3, timeout=30.0):
        self.composer = TransformComposer(max_depth=max_depth,
                                          timeout_seconds=timeout)
        self._analyzer = None  # lazy load
        self.solved_cache = {}  # game_id -> TransformChain
        self.analysis_cache = {}  # game_id -> analysis dict
        self.log = []

    def _get_analyzer(self):
        if self._analyzer is None:
            try:
                from .analyzer import GridAnalyzer
                self._analyzer = GridAnalyzer()
            except ImportError:
                self._analyzer = None
        return self._analyzer

    def try_solve(self, examples: list[tuple[np.ndarray, np.ndarray]],
                  game_id: str = None) -> Optional[TransformChain]:
        """
        Try to find a transform chain for the given training examples.

        1. Check cache
        2. Run analyzer to get priority primitives
        3. Run composer with priorities
        4. Cache and log result
        """
        # Check cache
        if game_id and game_id in self.solved_cache:
            cached = self.solved_cache[game_id]
            if score_chain(cached, examples) >= 1.0:
                return cached

        t0 = time.time()

        # Analyze
        priority = []
        analyzer = self._get_analyzer()
        if analyzer:
            try:
                analysis = analyzer.analyze_examples(examples)
                priority = analyzer.suggest_primitives(analysis)
                if game_id:
                    self.analysis_cache[game_id] = analysis
            except Exception:
                pass

        # Compose
        chain = self.composer.solve(examples, priority_primitives=priority)
        elapsed = time.time() - t0

        if chain:
            sc = score_chain(chain, examples)
            if sc >= 1.0:
                if game_id:
                    self.solved_cache[game_id] = chain
                self._log(game_id, chain, sc, elapsed, "SOLVED")
                return chain
            else:
                self._log(game_id, chain, sc, elapsed, "PARTIAL")
                return None  # only return exact solutions
        else:
            self._log(game_id, None, 0.0, elapsed, "FAILED")
            return None

    def suggest_actions(self, examples: list[tuple[np.ndarray, np.ndarray]]) -> dict:
        """
        Even without a full solution, analyze examples and suggest
        which action types the explorer should prioritize.

        Returns dict of suggestions: {"bias_clicks": True, "bias_arrows": True, ...}
        """
        analyzer = self._get_analyzer()
        if not analyzer:
            return {}

        try:
            analysis = analyzer.analyze_examples(examples)
        except Exception:
            return {}

        suggestions = {}

        # If color changes detected, bias toward clicking colored segments
        if analysis.get("color_changes"):
            suggestions["bias_clicks"] = True

        # If geometric transform, bias toward spatial actions (arrows)
        if analysis.get("geometric_match"):
            suggestions["bias_arrows"] = True

        # If gravity detected, bias toward directional actions
        if analysis.get("gravity_match"):
            suggestions["bias_gravity"] = True

        # If high overlap, only small changes needed
        overlap = analysis.get("content_overlap", 0)
        if overlap > 0.8:
            suggestions["small_changes"] = True

        return suggestions

    def _log(self, game_id, chain, score, elapsed, status):
        entry = {
            "game_id": game_id or "unknown",
            "status": status,
            "score": score,
            "elapsed": elapsed,
            "chain": chain.describe() if chain else "none",
        }
        self.log.append(entry)
        # Print log line
        chain_str = chain.describe() if chain else "no solution"
        print(f"TransformEngine: {status} {game_id or '?'} → "
              f"{chain_str} ({score:.0%} match, {elapsed:.1f}s)")

    def get_stats(self) -> dict:
        solved = sum(1 for e in self.log if e["status"] == "SOLVED")
        partial = sum(1 for e in self.log if e["status"] == "PARTIAL")
        failed = sum(1 for e in self.log if e["status"] == "FAILED")
        return {
            "solved": solved,
            "partial": partial,
            "failed": failed,
            "total": len(self.log),
            "cached": len(self.solved_cache),
        }
