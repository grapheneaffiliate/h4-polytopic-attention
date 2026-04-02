"""
Handoff — structured state compression between episodes.

From Anthropic's harness design: context resets with structured handoff beat
summarizing and continuing. Models lose coherence on long tasks.

After each episode, compress state into a handoff document.
Next episode starts fresh but reads the handoff.

This is the "SESSION_HANDOFF.md" pattern applied within a single game run.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import json


@dataclass
class LevelHandoff:
    """Compressed state for one level, carried across episodes."""
    level: int
    episodes_attempted: int = 0

    # Exploration state
    states_explored: int = 0
    max_depth: int = 0
    actions_taken: int = 0

    # Action model (compiled from memory)
    action_model: dict = field(default_factory=dict)   # {action: effect_type}
    dead_actions: list = field(default_factory=list)

    # Best partial progress
    best_path: list = field(default_factory=list)       # best action sequence so far
    best_path_score: float = 0.0                        # how close to solution
    best_state_hash: Optional[str] = None               # deepest useful state

    # Goal hypothesis
    goal_hypothesis: str = "unknown"
    goal_confidence: float = 0.0

    # What failed
    failed_approaches: list = field(default_factory=list)  # ["random grid clicks", ...]
    failed_modes: list = field(default_factory=list)        # ["segment", ...]

    # Mode decision
    current_mode: str = "segment"
    mode_efficacies: dict = field(default_factory=dict)    # {"segment": 0.02, "grid": 0.15}

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "episodes": self.episodes_attempted,
            "states": self.states_explored,
            "depth": self.max_depth,
            "actions": self.actions_taken,
            "action_model": self.action_model,
            "dead_actions": self.dead_actions,
            "best_path_len": len(self.best_path),
            "best_score": round(self.best_path_score, 4),
            "goal": self.goal_hypothesis,
            "goal_confidence": round(self.goal_confidence, 3),
            "failed_approaches": self.failed_approaches,
            "mode": self.current_mode,
            "mode_efficacies": {k: round(v, 4) for k, v in self.mode_efficacies.items()},
        }

    def to_prompt(self) -> str:
        """Generate handoff text for 3B model or next episode."""
        lines = [
            f"=== Level {self.level} Handoff (episode {self.episodes_attempted}) ===",
            f"States explored: {self.states_explored}",
            f"Max depth: {self.max_depth}",
            f"Actions taken: {self.actions_taken}",
        ]
        if self.action_model:
            lines.append("Action model:")
            for action, effect in sorted(self.action_model.items()):
                lines.append(f"  {action}: {effect}")
        if self.dead_actions:
            lines.append(f"Dead actions: {self.dead_actions}")
        if self.best_path:
            lines.append(f"Best path ({len(self.best_path)} steps, score={self.best_path_score:.3f}):")
            # Show last 10 actions of best path
            show = self.best_path[-10:] if len(self.best_path) > 10 else self.best_path
            lines.append(f"  ...{show}")
        lines.append(f"Goal hypothesis: {self.goal_hypothesis} (confidence: {self.goal_confidence:.0%})")
        if self.failed_approaches:
            lines.append(f"What failed: {', '.join(self.failed_approaches)}")
        lines.append(f"Current mode: {self.current_mode}")
        if self.mode_efficacies:
            effs = [f"{m}: {e:.1%}" for m, e in self.mode_efficacies.items()]
            lines.append(f"Mode efficacies: {', '.join(effs)}")
        return "\n".join(lines)


@dataclass
class GameHandoff:
    """Full game handoff containing all level handoffs."""
    game_id: str
    total_levels: int = 0
    levels_completed: int = 0
    total_actions: int = 0
    total_episodes: int = 0
    level_handoffs: dict = field(default_factory=dict)  # {level: LevelHandoff}

    # Cross-level insights
    game_type: str = "unknown"          # "rotation", "gravity", "color_sort", etc.
    game_type_confidence: float = 0.0
    winning_patterns: list = field(default_factory=list)  # patterns from solved levels

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "total_levels": self.total_levels,
            "completed": self.levels_completed,
            "actions": self.total_actions,
            "episodes": self.total_episodes,
            "game_type": self.game_type,
            "game_type_confidence": round(self.game_type_confidence, 3),
            "levels": {k: v.to_dict() for k, v in self.level_handoffs.items()},
        }

    def to_prompt(self) -> str:
        """Full game handoff prompt."""
        lines = [
            f"=== Game Handoff: {self.game_id} ===",
            f"Type: {self.game_type} (confidence: {self.game_type_confidence:.0%})",
            f"Progress: {self.levels_completed}/{self.total_levels} levels",
            f"Total: {self.total_actions} actions, {self.total_episodes} episodes",
        ]
        if self.winning_patterns:
            lines.append(f"Winning patterns: {self.winning_patterns}")
        lines.append("")
        for level in sorted(self.level_handoffs.keys()):
            lines.append(self.level_handoffs[level].to_prompt())
            lines.append("")
        return "\n".join(lines)


class HandoffManager:
    """
    Manages handoff state across episodes and levels.

    Usage:
        mgr = HandoffManager(game_id="dc22", total_levels=6)

        # After each episode:
        mgr.update_from_episode(
            level=0,
            states_explored=1234,
            max_depth=45,
            actions_taken=5000,
            mode="segment",
            efficacy=0.03,
            effective_history=[(state, action), ...],
        )

        # After level complete:
        mgr.on_level_complete(level=0, winning_path=[...])

        # Before next episode:
        handoff = mgr.get_handoff(level=1)
        print(handoff.to_prompt())  # read this before exploring

        # With memory integration:
        mgr.update_from_memory(memory, level=1)
    """

    def __init__(self, game_id: str = "", total_levels: int = 0):
        self.game = GameHandoff(game_id=game_id, total_levels=total_levels)

    def get_or_create_level(self, level: int) -> LevelHandoff:
        if level not in self.game.level_handoffs:
            self.game.level_handoffs[level] = LevelHandoff(level=level)
        return self.game.level_handoffs[level]

    def update_from_episode(self,
                           level: int,
                           states_explored: int = 0,
                           max_depth: int = 0,
                           actions_taken: int = 0,
                           mode: str = "segment",
                           efficacy: float = 0.0,
                           effective_history: Optional[list] = None):
        """Update handoff after an episode ends."""
        lh = self.get_or_create_level(level)
        lh.episodes_attempted += 1
        lh.states_explored = max(lh.states_explored, states_explored)
        lh.max_depth = max(lh.max_depth, max_depth)
        lh.actions_taken += actions_taken
        lh.current_mode = mode
        lh.mode_efficacies[mode] = efficacy

        # Track best path
        if effective_history:
            path_len = len(effective_history)
            # Simple score: longer effective paths are better
            score = path_len / max(actions_taken, 1)
            if score > lh.best_path_score or not lh.best_path:
                lh.best_path = list(effective_history)
                lh.best_path_score = score

        # Record failed modes
        if efficacy < 0.01 and mode not in lh.failed_modes:
            if actions_taken > 5000:  # enough data to judge
                lh.failed_modes.append(mode)
                lh.failed_approaches.append(f"{mode} mode ({efficacy:.1%} efficacy)")

        self.game.total_episodes += 1
        self.game.total_actions += actions_taken

    def on_level_complete(self, level: int, winning_path: Optional[list] = None):
        """Record that a level was solved."""
        lh = self.get_or_create_level(level)
        if winning_path:
            lh.best_path = list(winning_path)
            lh.best_path_score = 1.0

        self.game.levels_completed = max(self.game.levels_completed, level + 1)

        # Extract winning pattern
        if winning_path and len(winning_path) > 0:
            # Count action frequencies in winning path
            from collections import Counter
            if isinstance(winning_path[0], tuple):
                # winning_path entries can be 2-tuples (state, action) or 3-tuples (action_id, x, y)
                action_counts = Counter(t[0] for t in winning_path)
            else:
                action_counts = Counter(winning_path)
            top_actions = action_counts.most_common(3)
            pattern = {
                "level": level,
                "path_length": len(winning_path),
                "top_actions": top_actions,
                "mode": lh.current_mode,
            }
            self.game.winning_patterns.append(pattern)

    def update_from_memory(self, memory, level: int):
        """Pull action model from memory module into handoff."""
        lh = self.get_or_create_level(level)

        # Get hypotheses
        hypotheses = memory.compile_hypotheses()
        if hypotheses:
            lh.goal_hypothesis = "; ".join(hypotheses[:3])

        # Get action model
        summary = memory.get_action_summary(level=max(0, level - 1))
        if not summary:
            summary = memory.get_action_summary()

        lh.action_model.clear()
        lh.dead_actions.clear()
        for action, info in summary.items():
            if isinstance(info, dict):
                if info.get("change_rate", 1) == 0 and info.get("observations", 0) >= 10:
                    lh.dead_actions.append(action)
                else:
                    lh.action_model[action] = info.get("dominant_effect", "unknown")

    def update_goal_from_contract(self, level: int, contract):
        """Pull goal from sprint contract into handoff."""
        lh = self.get_or_create_level(level)
        lh.goal_hypothesis = contract.goal_hypothesis
        lh.goal_confidence = contract.confidence

    def get_handoff(self, level: int) -> LevelHandoff:
        """Get the handoff for a level (creates if needed)."""
        return self.get_or_create_level(level)

    def get_game_handoff(self) -> GameHandoff:
        """Get the full game handoff."""
        return self.game

    def infer_game_type(self):
        """Infer game type from winning patterns."""
        if not self.game.winning_patterns:
            return

        # Look at action models across levels
        all_effects = []
        for lh in self.game.level_handoffs.values():
            all_effects.extend(lh.action_model.values())

        if not all_effects:
            return

        from collections import Counter
        effect_counts = Counter(all_effects)
        dominant = effect_counts.most_common(1)[0]

        type_map = {
            "rotation": "rotation",
            "gravity": "gravity",
            "shift": "sliding",
            "color_change": "color_puzzle",
            "fill": "fill_puzzle",
            "swap": "swap_puzzle",
        }
        self.game.game_type = type_map.get(dominant[0], "unknown")
        self.game.game_type_confidence = dominant[1] / len(all_effects)
