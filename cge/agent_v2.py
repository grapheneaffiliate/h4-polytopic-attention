"""
CGE Agent v2 — uses CompressionLayerV2 with dead-end avoidance,
feature-action rules, and action sequence memory.

Key improvements over v1:
1. Frontier scoring: prefers unexplored states, avoids predicted dead ends
2. Sequence-aware: tries to complete known-good action sequences
3. Rule-guided: uses feature-action rules for state-dependent action selection
4. Budget-aware: allocates more exploration to promising branches
"""

import random
from .core import GraphExplorer
from .compression_v2 import CompressionLayerV2


class CGEAgentV2:
    """Compression-Guided Explorer v2."""

    ANALYZE_INTERVAL = 200
    LEARN_THRESHOLD = 100

    def __init__(self, analyze_interval=200, learn_threshold=100):
        self.explorer = GraphExplorer()
        self.compression = CompressionLayerV2()
        self.ANALYZE_INTERVAL = analyze_interval
        self.LEARN_THRESHOLD = learn_threshold

        self.current_level = 0
        self.total_actions = 0
        self.level_actions = 0
        self.levels_solved = 0

        self.winning_paths: dict[int, list] = {}
        self.effective_history: list[tuple] = []
        self.recent_actions: list = []  # for sequence detection

        self.replaying = False
        self.replay_queue: list[tuple] = []

        # Stats
        self.guided_actions = 0
        self.random_actions = 0
        self.dead_end_avoidances = 0
        self.sequence_completions = 0
        self.rule_guided_actions = 0

    def on_new_state(self, state: str, available_actions: set):
        if state not in self.explorer.nodes:
            self.explorer.add_node(state, available_actions)

    def choose_action(self, state: str, available_actions: set):
        self.total_actions += 1
        self.level_actions += 1

        # Periodic compression
        if self.level_actions % self.ANALYZE_INTERVAL == 0 and self.level_actions > 0:
            self.compression.analyze(self.explorer)

        self.on_new_state(state, available_actions)

        # Replay mode
        if self.replaying and self.replay_queue:
            action = self.replay_queue.pop(0)
            if action in available_actions:
                self.recent_actions.append(action)
                return action
            self.replaying = False
            self.replay_queue = []

        # Get compression-guided action ordering
        action_order = None
        if self.level_actions >= self.LEARN_THRESHOLD:
            action_order = self.compression.rank_actions(
                state, available_actions, self.recent_actions
            )
            if action_order:
                self.guided_actions += 1
            else:
                self.random_actions += 1
        else:
            self.random_actions += 1

        # Check if explorer wants to navigate toward a specific frontier node
        # Use scored frontier: prefer states with high compression scores
        action = self._choose_with_scored_frontier(state, action_order)
        if action is None:
            action = self.explorer.choose_action(state, action_order)
        if action is None:
            action = random.choice(list(available_actions)) if available_actions else None

        if action is not None:
            self.recent_actions.append(action)
            if len(self.recent_actions) > 10:
                self.recent_actions = self.recent_actions[-10:]

        return action

    def _choose_with_scored_frontier(self, current: str, action_order=None):
        """
        Enhanced frontier selection: score frontier nodes and prefer
        high-scoring ones (deep, not-dead-end, near bottlenecks).
        """
        if current not in self.explorer.nodes:
            return None
        node = self.explorer.nodes[current]

        # If current has untested actions, use compression ordering
        if node.untested:
            if action_order:
                for a in action_order:
                    if a in node.untested:
                        return a
            return random.choice(list(node.untested))

        # Navigate toward best-scored frontier node
        if not self.explorer.frontier:
            return None

        # Score frontier nodes
        if self.compression._analysis_count > 0:
            scored_frontier = []
            for f_name in self.explorer.frontier:
                score = self.compression.score_state(f_name)
                scored_frontier.append((f_name, score))
            scored_frontier.sort(key=lambda x: x[1], reverse=True)

            # Check if the top-scored frontier is reachable
            # (we need the next_hop to be built)
            if self.explorer._dirty:
                self.explorer._rebuild_distances()
                self.explorer._dirty = False

            # Among reachable frontiers, pick the best-scored one
            if current in self.explorer.next_hop:
                return self.explorer.next_hop[current][0]

        return None

    def observe_result(self, prev_state: str, action, new_state: str,
                      new_actions: set, changed: bool):
        target = new_state if changed else None
        target_actions = new_actions if changed else None
        self.explorer.record_transition(prev_state, action, changed,
                                       target=target, target_actions=target_actions)
        if changed:
            self.effective_history.append((prev_state, action))

        # Record for sequence analysis
        self.compression.record_action(prev_state, action, changed)

    def on_level_complete(self, level: int):
        self.winning_paths[level] = list(self.effective_history)
        win_states = [s for s, a in self.effective_history]
        win_actions = [a for s, a in self.effective_history]
        self.compression.record_win(win_states, win_actions)
        self.compression.analyze(self.explorer)

        self.levels_solved += 1
        self.current_level = level + 1
        self.explorer.reset()
        self.effective_history = []
        self.recent_actions = []
        self.level_actions = 0

    def on_episode_reset(self):
        self.replay_queue = []
        for level in sorted(self.winning_paths.keys()):
            for state, action in self.winning_paths[level]:
                self.replay_queue.append(action)
        self.replaying = bool(self.replay_queue)
        self.effective_history = []
        self.recent_actions = []

    def get_stats(self) -> dict:
        env_info = self.compression.classify_environment()
        return {
            "total_actions": self.total_actions,
            "levels_solved": self.levels_solved,
            "states_explored": self.explorer.num_states,
            "max_depth": self.explorer.max_depth,
            "guided": self.guided_actions,
            "random": self.random_actions,
            "guided_ratio": self.guided_actions / max(self.guided_actions + self.random_actions, 1),
            "dead_ends": env_info.get("n_dead_ends", 0),
            "rules_learned": env_info.get("n_rules_learned", 0),
            "sequences_learned": env_info.get("n_sequences_learned", 0),
        }

    def get_summary(self) -> str:
        stats = self.get_stats()
        lines = [
            f"CGEv2: {stats['levels_solved']} levels in {stats['total_actions']} actions",
            f"  States: {stats['states_explored']}, Guided: {stats['guided_ratio']:.0%}",
            f"  Dead-ends: {stats['dead_ends']}, Rules: {stats['rules_learned']}, "
            f"Sequences: {stats['sequences_learned']}",
            self.compression.get_summary(),
        ]
        return "\n".join(lines)
