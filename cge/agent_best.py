"""
CGE Best — merges proven strengths from v1 and v3.

v1 core (proven on CausalChain, RuleLearning):
- CompressionLayer with periodic graph-based analysis
- State-dependent action ranking

v3 additions (proven on NeedleInHaystack, LargeStateSpace, MazeNavigation):
- Cross-level transfer: bias toward action types from winning paths
- Adaptive branch pruning: escape low-change-rate branches
- Incremental global efficacy tracking (faster convergence)

Key principle: v1's ranking is the DEFAULT. v3 features are ADDITIVE bonuses
that nudge the ranking, not override it.
"""

import random
from collections import defaultdict
from .core import GraphExplorer
from .compression import CompressionLayer


class CGEBest:
    """The best of v1 + v3."""

    ANALYZE_INTERVAL = 150
    LEARN_THRESHOLD = 80
    PRUNE_WINDOW = 20
    PRUNE_RATIO = 0.25  # prune if branch rate < 25% of global average

    def __init__(self, analyze_interval=150, learn_threshold=80):
        self.explorer = GraphExplorer()
        self.compression = CompressionLayer()  # v1's proven compression
        self.ANALYZE_INTERVAL = analyze_interval
        self.LEARN_THRESHOLD = learn_threshold

        self.current_level = 0
        self.total_actions = 0
        self.level_actions = 0
        self.levels_solved = 0

        self.winning_paths: dict[int, list] = {}
        self.effective_history: list[tuple] = []
        self.recent_actions: list = []

        self.replaying = False
        self.replay_queue: list[tuple] = []

        # --- v3 additions ---
        # Cross-level transfer
        self._winning_action_counts: dict[object, int] = defaultdict(int)
        self._transfer_bias: list = []

        # Adaptive branch pruning
        self._branch_window: list[bool] = []
        self._global_changes = 0
        self._global_actions = 0

        # Incremental global efficacy (faster than waiting for analyze)
        self._fast_efficacy: dict[object, list] = defaultdict(lambda: [0, 0])

        # Stats
        self.guided_actions = 0
        self.random_actions = 0
        self.pruned_branches = 0

    def on_new_state(self, state: str, available_actions: set):
        if state not in self.explorer.nodes:
            self.explorer.add_node(state, available_actions)

    def choose_action(self, state: str, available_actions: set):
        self.total_actions += 1
        self.level_actions += 1

        # Periodic v1 compression analysis
        if self.level_actions % self.ANALYZE_INTERVAL == 0 and self.level_actions > 0:
            self.compression.analyze(self.explorer)

        self.on_new_state(state, available_actions)

        # Replay mode
        if self.replaying and self.replay_queue:
            action = self.replay_queue.pop(0)
            if action in available_actions:
                return action
            self.replaying = False
            self.replay_queue = []

        # Adaptive branch pruning (v3)
        if self._should_prune():
            self._branch_window.clear()
            self.pruned_branches += 1
            untested = self.explorer.nodes[state].untested if state in self.explorer.nodes else set()
            if untested:
                # Pick least-tried action type
                return self._pick_least_tried(untested)

        # Build action ordering
        action_order = None
        if self.level_actions >= self.LEARN_THRESHOLD:
            action_order = self._rank_actions(state, available_actions)
            self.guided_actions += 1
        else:
            self.random_actions += 1

        action = self.explorer.choose_action(state, action_order=action_order)
        if action is None:
            action = random.choice(list(available_actions)) if available_actions else None
        return action

    def _rank_actions(self, state: str, available: set) -> list:
        """
        Hybrid ranking: v1 compression as base, v3 transfer as bonus.
        """
        # Get v1's ranking (proven, state-dependent)
        v1_ranked = self.compression.rank_actions(state, available)

        # Add v3 transfer bonus
        if not self._transfer_bias:
            return v1_ranked

        # Score: v1 position gives base score, transfer bias adds bonus
        scored = {}
        for i, a in enumerate(v1_ranked):
            scored[a] = len(v1_ranked) - i  # higher rank = higher score

        for i, a in enumerate(self._transfer_bias[:5]):
            if a in available:
                scored[a] = scored.get(a, 0) + 3.0 / (i + 1)

        # Also boost actions with high fast-efficacy
        for a in available:
            if a in self._fast_efficacy:
                attempts, successes = self._fast_efficacy[a]
                if attempts >= 5:
                    scored[a] = scored.get(a, 0) + 2.0 * (successes / attempts)

        return sorted(available, key=lambda a: scored.get(a, 0), reverse=True)

    def _pick_least_tried(self, actions: set) -> object:
        """Pick the action we've tried the least (for escaping dead branches)."""
        min_attempts = float('inf')
        best = None
        for a in actions:
            attempts = self._fast_efficacy.get(a, [0, 0])[0]
            if attempts < min_attempts:
                min_attempts = attempts
                best = a
        return best if best is not None else random.choice(list(actions))

    def _should_prune(self) -> bool:
        if len(self._branch_window) < self.PRUNE_WINDOW:
            return False
        if self._global_actions < 50:
            return False
        global_rate = self._global_changes / max(self._global_actions, 1)
        if global_rate < 0.05:
            return False
        recent = self._branch_window[-self.PRUNE_WINDOW:]
        branch_rate = sum(recent) / len(recent)
        return branch_rate < global_rate * self.PRUNE_RATIO

    def observe_result(self, prev_state: str, action, new_state: str,
                      new_actions: set, changed: bool):
        target = new_state if changed else None
        target_actions = new_actions if changed else None
        self.explorer.record_transition(prev_state, action, changed,
                                       target=target, target_actions=target_actions)
        if changed:
            self.effective_history.append((prev_state, action))

        # Fast efficacy tracking
        self._fast_efficacy[action][0] += 1
        if changed:
            self._fast_efficacy[action][1] += 1

        # Branch/global tracking
        self._branch_window.append(changed)
        if len(self._branch_window) > 50:
            self._branch_window = self._branch_window[-50:]
        self._global_actions += 1
        if changed:
            self._global_changes += 1

    def on_level_complete(self, level: int):
        self.winning_paths[level] = list(self.effective_history)

        # Cross-level transfer
        for state, action in self.effective_history:
            self._winning_action_counts[action] += 1
        self._transfer_bias = sorted(
            self._winning_action_counts.keys(),
            key=lambda a: self._winning_action_counts[a],
            reverse=True,
        )

        # Tell v1 compression about the win
        win_states = [s for s, a in self.effective_history]
        win_actions = [a for s, a in self.effective_history]
        self.compression.record_win(win_states, win_actions)
        self.compression.analyze(self.explorer)

        self.levels_solved += 1
        self.current_level = level + 1
        self.explorer.reset()
        self.effective_history = []
        self._branch_window.clear()
        self.level_actions = 0
        # Keep fast_efficacy and transfer_bias — they transfer!

    def on_episode_reset(self):
        self.replay_queue = []
        for level in sorted(self.winning_paths.keys()):
            for state, action in self.winning_paths[level]:
                self.replay_queue.append(action)
        self.replaying = bool(self.replay_queue)
        self.effective_history = []
        self._branch_window.clear()

    def get_stats(self) -> dict:
        return {
            "total_actions": self.total_actions,
            "levels_solved": self.levels_solved,
            "states_explored": self.explorer.num_states,
            "max_depth": self.explorer.max_depth,
            "guided": self.guided_actions,
            "random": self.random_actions,
            "pruned": self.pruned_branches,
            "transfer_actions": len(self._transfer_bias),
        }

    def get_summary(self) -> str:
        s = self.get_stats()
        lines = [
            f"CGEBest: {s['levels_solved']} levels in {s['total_actions']} actions",
            f"  States: {s['states_explored']}, Guided: {s['guided']}/{s['guided']+s['random']}, Pruned: {s['pruned']}",
        ]
        if self._transfer_bias:
            lines.append(f"  Transfer: {self._transfer_bias[:5]}")
        eff = sorted(self._fast_efficacy.items(),
                    key=lambda x: x[1][1]/max(x[1][0],1), reverse=True)[:3]
        if eff:
            lines.append(f"  Top: {[(a, f'{s}/{t}') for a, (t, s) in eff]}")
        lines.append(f"  {self.compression.get_summary()}")
        return "\n".join(lines)
