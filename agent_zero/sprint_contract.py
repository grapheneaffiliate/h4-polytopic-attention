"""
Sprint Contract — negotiated plan before each level.

Before exploring a new level, generate a contract from previous level data:
- Goal hypothesis (from frame diff of solved levels)
- Action model (from action-effect memory)
- Budget allocation
- Success/failure criteria
- Fallback strategy

The explorer executes against the contract.
The evaluator grades against the contract.

This is the structure that the 3B model reads/writes.
Heuristic generation now, model-driven later.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SprintContract:
    """Contract for one level of exploration."""
    level: int
    game_id: str = ""

    # Goal
    goal_hypothesis: str = "unknown"          # "make grid symmetric", "sort colors", etc.
    goal_features: Optional[dict] = None      # measurable target features

    # Action model
    action_model: dict = field(default_factory=dict)  # {action: effect_type}
    preferred_actions: list = field(default_factory=list)  # ordered by expected utility
    dead_actions: list = field(default_factory=list)

    # Budget
    budget: int = 200000
    checkpoint_at: int = 50000                # evaluate progress here
    abort_at: int = 100000                    # give up if no progress by here

    # Success criteria
    success_metric: str = "level_complete"    # what counts as success
    progress_indicators: list = field(default_factory=list)  # intermediate signals

    # Fallback
    fallback_mode: Optional[str] = None       # mode to try if primary fails
    fallback_at: int = 50000                  # when to trigger fallback

    # Metadata
    confidence: float = 0.0                   # 0-1, how confident are we in this plan
    based_on_levels: list = field(default_factory=list)  # which solved levels informed this

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "game_id": self.game_id,
            "goal_hypothesis": self.goal_hypothesis,
            "action_model": self.action_model,
            "preferred_actions": self.preferred_actions,
            "dead_actions": self.dead_actions,
            "budget": self.budget,
            "checkpoint_at": self.checkpoint_at,
            "abort_at": self.abort_at,
            "fallback_mode": self.fallback_mode,
            "confidence": round(self.confidence, 3),
            "based_on_levels": self.based_on_levels,
        }

    def to_prompt(self) -> str:
        """Generate a natural-language prompt for the 3B model."""
        lines = [
            f"=== Sprint Contract: Level {self.level} ===",
            f"Game: {self.game_id}",
            f"Goal hypothesis: {self.goal_hypothesis}",
        ]
        if self.action_model:
            lines.append("Action model:")
            for action, effect in sorted(self.action_model.items()):
                lines.append(f"  Action {action} = {effect}")
        if self.preferred_actions:
            lines.append(f"Preferred actions (in order): {self.preferred_actions}")
        if self.dead_actions:
            lines.append(f"Dead actions (skip): {self.dead_actions}")
        lines.extend([
            f"Budget: {self.budget} actions",
            f"Checkpoint at: {self.checkpoint_at} (evaluate progress)",
            f"Abort at: {self.abort_at} (if no progress)",
        ])
        if self.fallback_mode:
            lines.append(f"Fallback: switch to {self.fallback_mode} at {self.fallback_at}")
        lines.append(f"Confidence: {self.confidence:.0%}")
        if self.based_on_levels:
            lines.append(f"Based on solved levels: {self.based_on_levels}")
        return "\n".join(lines)


class SprintPlanner:
    """
    Generates sprint contracts from memory and solved-level data.

    Usage:
        planner = SprintPlanner()
        contract = planner.plan(
            level=2,
            game_id="dc22",
            memory=action_effect_memory,
            solved_levels={0: path0, 1: path1},
            prev_frames={0: (start, end), 1: (start, end)},
            total_budget=300000,
        )
    """

    def plan(self,
             level: int,
             game_id: str = "",
             memory=None,
             solved_levels: Optional[dict] = None,
             prev_frames: Optional[dict] = None,
             total_budget: int = 200000,
             current_mode: str = "segment") -> SprintContract:
        """Generate a sprint contract for the next level."""

        contract = SprintContract(level=level, game_id=game_id)

        # Budget: diminishing per level (earlier levels more important)
        remaining_budget = total_budget
        contract.budget = remaining_budget
        contract.checkpoint_at = min(50000, remaining_budget // 4)
        contract.abort_at = min(100000, remaining_budget // 2)

        # Action model from memory
        if memory is not None:
            self._fill_action_model(contract, memory, level)

        # Goal hypothesis from solved levels
        if solved_levels and prev_frames:
            self._infer_goal(contract, solved_levels, prev_frames)

        # Fallback strategy
        if current_mode == "segment":
            contract.fallback_mode = "grid"
            contract.fallback_at = contract.checkpoint_at
        elif current_mode == "grid":
            contract.fallback_mode = "grid_fine"
            contract.fallback_at = contract.checkpoint_at

        # Confidence based on how much we know
        contract.confidence = self._estimate_confidence(contract, memory, solved_levels)

        return contract

    def _fill_action_model(self, contract: SprintContract, memory, level: int):
        """Fill action model from memory module."""
        # Get models from previous levels (transfer)
        all_actions = set()
        for lvl in range(level):
            summary = memory.get_action_summary(level=lvl)
            all_actions.update(summary.keys())

        # Also check global models
        global_summary = memory.get_action_summary()
        all_actions.update(global_summary.keys())

        for action in sorted(all_actions):
            model = memory.get_action_model(action, level=max(0, level - 1))
            if model is None:
                model = memory.get_action_model(action)
            if model is None:
                continue

            info = model.to_dict()

            if info["change_rate"] == 0 and info["observations"] >= 10:
                contract.dead_actions.append(action)
            else:
                contract.action_model[action] = info["dominant_effect"]
                if info["change_rate"] > 0.3:
                    contract.preferred_actions.append(action)

        # Sort preferred by change rate (best first)
        # Re-sort using actual model data
        def _sort_key(a):
            m = memory.get_action_model(a, level=max(0, level - 1))
            if m is None:
                m = memory.get_action_model(a)
            return m.change_rate if m else 0
        contract.preferred_actions.sort(key=_sort_key, reverse=True)

    def _infer_goal(self, contract: SprintContract,
                    solved_levels: dict,
                    prev_frames: dict):
        """Infer goal from solved level frame diffs."""
        contract.based_on_levels = list(solved_levels.keys())

        # Analyze start→end frame diffs from solved levels
        patterns = []
        for lvl in sorted(solved_levels.keys()):
            if lvl not in prev_frames:
                continue
            start_frame, end_frame = prev_frames[lvl]
            if start_frame is None or end_frame is None:
                continue

            pattern = self._analyze_transformation(start_frame, end_frame)
            if pattern:
                patterns.append(pattern)

        if not patterns:
            contract.goal_hypothesis = "unknown — no solved levels to learn from"
            return

        # Find common patterns
        all_types = [p["type"] for p in patterns]
        from collections import Counter
        type_counts = Counter(all_types)
        dominant = type_counts.most_common(1)[0]

        contract.goal_hypothesis = (
            f"{dominant[0]} (seen in {dominant[1]}/{len(patterns)} solved levels)"
        )

        # Extract measurable features
        contract.goal_features = {
            "dominant_transform": dominant[0],
            "consistency": dominant[1] / len(patterns),
            "patterns": patterns,
        }

    def _analyze_transformation(self, start: np.ndarray, end: np.ndarray) -> Optional[dict]:
        """Analyze what transformation was applied from start to end frame."""
        if start.shape != end.shape:
            return {"type": "resize", "from": start.shape, "to": end.shape}

        diff = start != end
        magnitude = np.mean(diff)

        if magnitude == 0:
            return None

        # Check symmetry change
        start_h_sym = np.mean(start == np.fliplr(start))
        end_h_sym = np.mean(end == np.fliplr(end))
        start_v_sym = np.mean(start == np.flipud(start))
        end_v_sym = np.mean(end == np.flipud(end))

        if end_h_sym > start_h_sym + 0.2 or end_v_sym > start_v_sym + 0.2:
            return {
                "type": "symmetrize",
                "h_sym_gain": float(end_h_sym - start_h_sym),
                "v_sym_gain": float(end_v_sym - start_v_sym),
            }

        # Check rotation
        for k in [1, 2, 3]:
            rotated = np.rot90(start, k)
            if rotated.shape == end.shape and np.mean(rotated == end) > 0.9:
                return {"type": f"rotate_{k*90}"}

        # Check color sorting
        start_colors = len(np.unique(start))
        end_colors = len(np.unique(end))
        if end_colors < start_colors:
            return {"type": "simplify_colors", "from": start_colors, "to": end_colors}

        # Check gravity (colors moved to one edge)
        # Compare row-wise color distribution
        end_nonzero_rows = np.any(end > 0, axis=1)
        if end_nonzero_rows.any():
            first_row = np.argmax(end_nonzero_rows)
            last_row = len(end_nonzero_rows) - np.argmax(end_nonzero_rows[::-1]) - 1
            compactness = np.sum(end_nonzero_rows) / (last_row - first_row + 1)
            if compactness > 0.9 and magnitude > 0.1:
                return {"type": "gravity", "direction": "compact"}

        if magnitude > 0.5:
            return {"type": "major_change", "magnitude": float(magnitude)}

        return {"type": "minor_change", "magnitude": float(magnitude)}

    def _estimate_confidence(self, contract: SprintContract,
                            memory, solved_levels: Optional[dict]) -> float:
        """Estimate how confident we are in this contract."""
        score = 0.0

        # Have action model?
        if contract.action_model:
            score += 0.3

        # Have goal hypothesis beyond "unknown"?
        if "unknown" not in contract.goal_hypothesis:
            score += 0.3

        # Have preferred actions?
        if contract.preferred_actions:
            score += 0.2

        # Based on multiple solved levels?
        if len(contract.based_on_levels) >= 2:
            score += 0.2
        elif len(contract.based_on_levels) == 1:
            score += 0.1

        return min(1.0, score)
