"""
CGE Agent — the unified Explore-Compress-Synthesize loop.

This is the main entry point. It wraps GraphExplorer + CompressionLayer
into a single agent that:
1. Explores an environment (building a state graph)
2. Periodically compresses (learns patterns from the graph)
3. Uses learned patterns to guide future exploration
4. Records wins and transfers knowledge across levels
"""

from .core import GraphExplorer
from .compression import CompressionLayer


class CGEAgent:
    """
    Compression-Guided Explorer.

    Drop-in replacement for a BFS explorer, but learns from its own exploration.
    """

    # How often to run compression analysis (every N actions)
    ANALYZE_INTERVAL = 500
    # Minimum actions before using learned priorities
    LEARN_THRESHOLD = 200

    def __init__(self, analyze_interval=500, learn_threshold=200):
        self.explorer = GraphExplorer()
        self.compression = CompressionLayer()
        self.ANALYZE_INTERVAL = analyze_interval
        self.LEARN_THRESHOLD = learn_threshold

        # Level tracking
        self.current_level = 0
        self.total_actions = 0
        self.level_actions = 0

        # Winning path tracking (for replay + learning)
        self.winning_paths: dict[int, list] = {}   # level -> [(state, action)]
        self.effective_history: list[tuple] = []     # (state, action) for current level

        # Replay
        self.replaying = False
        self.replay_queue: list[tuple] = []  # (action,) to replay

        # Stats
        self.levels_solved = 0
        self.compression_guided_actions = 0
        self.random_actions = 0

    def on_new_state(self, state: str, available_actions: set):
        """Notify agent of a new/current state and its available actions."""
        if state not in self.explorer.nodes:
            self.explorer.add_node(state, available_actions)

    def choose_action(self, state: str, available_actions: set):
        """
        Choose the next action. The core loop:
        1. If replaying a winning path, follow it
        2. If current state has untested actions, pick one (compression-guided)
        3. Otherwise, navigate toward best frontier node
        """
        self.total_actions += 1
        self.level_actions += 1

        # Periodic compression
        if self.level_actions % self.ANALYZE_INTERVAL == 0 and self.level_actions > 0:
            self.compression.analyze(self.explorer)

        # Ensure state is registered
        self.on_new_state(state, available_actions)

        # Replay mode: follow known winning path
        if self.replaying and self.replay_queue:
            action = self.replay_queue.pop(0)
            if action in available_actions:
                return action
            # Replay broken — fall through to normal exploration
            self.replaying = False
            self.replay_queue = []

        # Get action ordering from compression (if we have enough data)
        action_order = None
        if self.level_actions >= self.LEARN_THRESHOLD and self.compression.action_stats:
            action_order = self.compression.rank_actions(state, available_actions)
            if action_order:
                self.compression_guided_actions += 1
            else:
                self.random_actions += 1
        else:
            self.random_actions += 1

        # Ask explorer for next action (with compression-guided ordering)
        action = self.explorer.choose_action(state, action_order=action_order)
        return action

    def observe_result(self, prev_state: str, action, new_state: str,
                      new_actions: set, changed: bool):
        """Record the result of taking an action."""
        target = new_state if changed else None
        target_actions = new_actions if changed else None
        self.explorer.record_transition(prev_state, action, changed,
                                       target=target, target_actions=target_actions)
        if changed:
            self.effective_history.append((prev_state, action))

    def on_level_complete(self, level: int):
        """
        Called when a level is solved.
        Saves winning path, transfers knowledge, resets explorer for next level.
        """
        # Save winning path
        self.winning_paths[level] = list(self.effective_history)

        # Tell compression about the win
        win_states = [s for s, a in self.effective_history]
        win_actions = [a for s, a in self.effective_history]
        self.compression.record_win(win_states, win_actions)

        # Re-analyze with winning data
        self.compression.analyze(self.explorer)

        self.levels_solved += 1
        self.current_level = level + 1

        # Reset explorer for new level (but compression knowledge persists!)
        self.explorer.reset()
        self.effective_history = []
        self.level_actions = 0

    def on_episode_reset(self):
        """
        Called on GAME_OVER / episode reset.
        Sets up replay of solved levels.
        """
        self.replay_queue = []
        for level in sorted(self.winning_paths.keys()):
            for state, action in self.winning_paths[level]:
                self.replay_queue.append(action)
        self.replaying = bool(self.replay_queue)
        self.effective_history = []

    def get_stats(self) -> dict:
        """Return current agent statistics."""
        return {
            "total_actions": self.total_actions,
            "level_actions": self.level_actions,
            "levels_solved": self.levels_solved,
            "states_explored": self.explorer.num_states,
            "max_depth": self.explorer.max_depth,
            "compression_guided": self.compression_guided_actions,
            "random": self.random_actions,
            "guided_ratio": (self.compression_guided_actions /
                           max(self.compression_guided_actions + self.random_actions, 1)),
            "bottlenecks": len(self.compression.bottlenecks),
        }

    def get_summary(self) -> str:
        """Human-readable summary."""
        stats = self.get_stats()
        lines = [
            f"CGEAgent: {stats['levels_solved']} levels solved in {stats['total_actions']} actions",
            f"  States: {stats['states_explored']}, Depth: {stats['max_depth']}",
            f"  Guided: {stats['guided_ratio']:.0%} ({stats['compression_guided']}/{stats['compression_guided']+stats['random']})",
            self.compression.get_summary(),
        ]
        return "\n".join(lines)


class BFSAgent:
    """
    Baseline BFS agent (no compression). Same interface as CGEAgent.
    Used for comparison to show compression's impact.
    """

    def __init__(self):
        self.explorer = GraphExplorer()
        self.total_actions = 0
        self.level_actions = 0
        self.current_level = 0
        self.levels_solved = 0
        self.winning_paths: dict[int, list] = {}
        self.effective_history: list[tuple] = []
        self.replaying = False
        self.replay_queue: list[tuple] = []

    def on_new_state(self, state: str, available_actions: set):
        if state not in self.explorer.nodes:
            self.explorer.add_node(state, available_actions)

    def choose_action(self, state: str, available_actions: set):
        self.total_actions += 1
        self.level_actions += 1
        self.on_new_state(state, available_actions)

        if self.replaying and self.replay_queue:
            action = self.replay_queue.pop(0)
            if action in available_actions:
                return action
            self.replaying = False
            self.replay_queue = []

        return self.explorer.choose_action(state)

    def observe_result(self, prev_state, action, new_state, new_actions, changed):
        target = new_state if changed else None
        target_actions = new_actions if changed else None
        self.explorer.record_transition(prev_state, action, changed,
                                       target=target, target_actions=target_actions)
        if changed:
            self.effective_history.append((prev_state, action))

    def on_level_complete(self, level):
        self.winning_paths[level] = list(self.effective_history)
        self.levels_solved += 1
        self.current_level = level + 1
        self.explorer.reset()
        self.effective_history = []
        self.level_actions = 0

    def on_episode_reset(self):
        self.replay_queue = []
        for level in sorted(self.winning_paths.keys()):
            for state, action in self.winning_paths[level]:
                self.replay_queue.append(action)
        self.replaying = bool(self.replay_queue)
        self.effective_history = []

    def get_stats(self):
        return {
            "total_actions": self.total_actions,
            "level_actions": self.level_actions,
            "levels_solved": self.levels_solved,
            "states_explored": self.explorer.num_states,
            "max_depth": self.explorer.max_depth,
        }
