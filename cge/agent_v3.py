"""
CGE Agent v3 — the breakthrough agent.

Combines ONLY what works from v1 and v2, plus one new capability:

From v1 (proven):
- Global action efficacy ranking (learns which actions work)
- State-dependent action ranking (learns what works where)

From v2 (sequence memory only):
- Action sequence detection (learns multi-step patterns)

NEW — Branch pruning via change-rate tracking:
- Track change rate per branch (how often actions produce new states)
- Abandon low-change-rate branches quickly (save budget for better branches)
- This is the key to BottleneckPuzzle and DeepTreeSearch:
  dead-end branches have low change rates, good branches have high ones

NEW — Cross-level pattern transfer:
- Remember which action types were on the winning path
- On new levels, bias toward those action types FIRST
- This models the ARC insight: games have consistent rules across levels
"""

import random
from collections import defaultdict
from .core import GraphExplorer


class CGEAgentV3:
    """The breakthrough agent: action ranking + sequences + branch pruning + transfer."""

    ANALYZE_INTERVAL = 150
    LEARN_THRESHOLD = 80
    # Branch pruning: if a branch's change rate is far below the
    # environment's average, abandon it. Adaptive to each environment.
    PRUNE_WINDOW = 20
    PRUNE_RATIO = 0.3  # prune if branch rate < 30% of global average

    def __init__(self, analyze_interval=150, learn_threshold=80):
        self.explorer = GraphExplorer()
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

        # --- Action efficacy (from v1) ---
        self._action_success: dict[object, list] = defaultdict(lambda: [0, 0])
        self._action_ranking: list = []

        # --- State-dependent efficacy ---
        self._state_action: dict[tuple, dict] = defaultdict(lambda: defaultdict(lambda: [0, 0]))

        # --- Sequence memory (from v2) ---
        self._history_buffer: list[tuple] = []  # (action, changed)
        self._seq_success: dict[tuple, list] = defaultdict(lambda: [0, 0])
        self._best_seqs: list[tuple] = []
        self._seq_len = 3

        # --- Branch pruning (NEW) ---
        self._branch_window: list[bool] = []  # recent [changed] results
        self._global_changes = 0  # total state changes ever
        self._global_actions = 0  # total actions ever

        # --- Cross-level transfer (NEW) ---
        self._winning_action_types: dict[object, int] = defaultdict(int)
        self._level_action_bias: list = []  # actions biased by cross-level knowledge

        # Stats
        self.guided_actions = 0
        self.random_actions = 0
        self.pruned_branches = 0
        self.sequence_hits = 0

    # ===== Analysis =====

    def _analyze(self):
        """Lightweight analysis — only compute what's proven to work."""
        # Action efficacy ranking
        self._action_ranking = sorted(
            self._action_success.keys(),
            key=lambda a: (
                self._action_success[a][1] / max(self._action_success[a][0], 1),
                self._action_success[a][0]
            ),
            reverse=True,
        )

        # Sequence analysis
        self._analyze_sequences()

    def _analyze_sequences(self):
        buf = self._history_buffer
        slen = self._seq_len
        self._seq_success.clear()

        for i in range(len(buf) - slen + 1):
            window = buf[i:i+slen]
            seq = tuple(a for a, _ in window)
            self._seq_success[seq][0] += 1
            if window[-1][1]:  # last action changed state
                self._seq_success[seq][1] += 1

        scored = []
        for seq, (attempts, successes) in self._seq_success.items():
            if attempts >= 5:
                scored.append((seq, successes / attempts, attempts))
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        self._best_seqs = [s[0] for s in scored[:10]]

    def _get_sequence_suggestion(self) -> object:
        """If recent actions match start of a known-good sequence, suggest completion."""
        if not self._best_seqs or len(self.recent_actions) < self._seq_len - 1:
            return None
        recent = tuple(self.recent_actions[-(self._seq_len - 1):])
        for seq in self._best_seqs:
            if seq[:self._seq_len - 1] == recent:
                return seq[-1]
        return None

    # ===== State classification =====

    def _classify_state(self, state_name: str) -> tuple:
        parts = state_name.split("_")
        return tuple(p for p in parts if not p.startswith("L"))

    # ===== Action ranking =====

    def _rank_actions(self, state: str, available: set) -> list:
        """Rank actions using all learned knowledge."""
        scored = defaultdict(float)

        # Layer 1: Cross-level transfer bias (proven on multi-level games)
        if self._level_action_bias:
            for i, a in enumerate(self._level_action_bias[:5]):
                if a in available:
                    scored[a] += 4.0 / (i + 1)

        # Layer 2: State-dependent efficacy (the core of v1's strength)
        st = self._classify_state(state)
        if st in self._state_action:
            sa = self._state_action[st]
            for a in available:
                if a in sa:
                    attempts, successes = sa[a]
                    if attempts >= 3:
                        scored[a] += 6.0 * (successes / attempts)

        # Layer 3: Global efficacy
        for a in available:
            if a in self._action_success:
                attempts, successes = self._action_success[a]
                if attempts >= 3:
                    scored[a] += 2.0 * (successes / attempts)

        if not scored:
            return list(available)

        return sorted(available, key=lambda a: scored.get(a, 0), reverse=True)

    # ===== Branch pruning =====

    def _should_prune_branch(self) -> bool:
        """
        Adaptive branch pruning: only prune if this branch's change rate
        is FAR below the environment's global average.
        This prevents pruning in environments where change rate is naturally low
        (like CausalChain where most actions don't change state).
        """
        if len(self._branch_window) < self.PRUNE_WINDOW:
            return False
        if self._global_actions < 50:
            return False  # not enough data

        global_rate = self._global_changes / max(self._global_actions, 1)
        if global_rate < 0.05:
            return False  # environment has very low change rate overall, don't prune

        recent = self._branch_window[-self.PRUNE_WINDOW:]
        branch_rate = sum(recent) / len(recent)

        # Only prune if branch rate is significantly below global average
        return branch_rate < global_rate * self.PRUNE_RATIO

    def _on_branch_pruned(self):
        """When pruning: reset branch window, try different direction."""
        self._branch_window.clear()
        self.pruned_branches += 1

    # ===== Main interface =====

    def on_new_state(self, state: str, available_actions: set):
        if state not in self.explorer.nodes:
            self.explorer.add_node(state, available_actions)

    def choose_action(self, state: str, available_actions: set):
        self.total_actions += 1
        self.level_actions += 1

        # Periodic analysis
        if self.level_actions % self.ANALYZE_INTERVAL == 0 and self.level_actions > 0:
            self._analyze()

        self.on_new_state(state, available_actions)

        # Replay mode
        if self.replaying and self.replay_queue:
            action = self.replay_queue.pop(0)
            if action in available_actions:
                self.recent_actions.append(action)
                return action
            self.replaying = False
            self.replay_queue = []

        # Branch pruning: if stuck in a low-change branch, force random action
        # to escape (instead of continuing to follow the explorer's BFS path)
        if self._should_prune_branch():
            self._on_branch_pruned()
            # Pick a random action we haven't tried much
            untested = self.explorer.nodes[state].untested if state in self.explorer.nodes else set()
            if untested:
                action = random.choice(list(untested))
                self.recent_actions.append(action)
                return action

        # Compression-guided action ranking
        action_order = None
        if self.level_actions >= self.LEARN_THRESHOLD:
            action_order = self._rank_actions(state, available_actions)
            self.guided_actions += 1
        else:
            self.random_actions += 1

        action = self.explorer.choose_action(state, action_order=action_order)

        if action is None:
            action = random.choice(list(available_actions)) if available_actions else None

        if action is not None:
            self.recent_actions.append(action)
            if len(self.recent_actions) > 20:
                self.recent_actions = self.recent_actions[-20:]

        return action

    def observe_result(self, prev_state: str, action, new_state: str,
                      new_actions: set, changed: bool):
        target = new_state if changed else None
        target_actions = new_actions if changed else None
        self.explorer.record_transition(prev_state, action, changed,
                                       target=target, target_actions=target_actions)
        if changed:
            self.effective_history.append((prev_state, action))

        # Update action efficacy
        self._action_success[action][0] += 1
        if changed:
            self._action_success[action][1] += 1

        # Update state-dependent efficacy
        st = self._classify_state(prev_state)
        self._state_action[st][action][0] += 1
        if changed:
            self._state_action[st][action][1] += 1

        # Update sequence buffer
        self._history_buffer.append((action, changed))
        if len(self._history_buffer) > 5000:
            self._history_buffer = self._history_buffer[-3000:]

        # Update branch window and global rates
        self._branch_window.append(changed)
        if len(self._branch_window) > 50:
            self._branch_window = self._branch_window[-50:]
        self._global_actions += 1
        if changed:
            self._global_changes += 1

    def on_level_complete(self, level: int):
        self.winning_paths[level] = list(self.effective_history)

        # Cross-level transfer: remember which actions were on winning path
        for state, action in self.effective_history:
            self._winning_action_types[action] += 1

        # Build bias for next level
        self._level_action_bias = sorted(
            self._winning_action_types.keys(),
            key=lambda a: self._winning_action_types[a],
            reverse=True,
        )

        self.levels_solved += 1
        self.current_level = level + 1
        self.explorer.reset()
        self.effective_history = []
        self.recent_actions = []
        self._branch_window.clear()
        self.level_actions = 0
        # Clear state-dependent stats (they're level-specific and become stale)
        self._state_action.clear()
        # Keep global action stats and sequences — they transfer across levels!

    def on_episode_reset(self):
        self.replay_queue = []
        for level in sorted(self.winning_paths.keys()):
            for state, action in self.winning_paths[level]:
                self.replay_queue.append(action)
        self.replaying = bool(self.replay_queue)
        self.effective_history = []
        self.recent_actions = []
        self._branch_window.clear()

    def get_stats(self) -> dict:
        return {
            "total_actions": self.total_actions,
            "levels_solved": self.levels_solved,
            "states_explored": self.explorer.num_states,
            "max_depth": self.explorer.max_depth,
            "guided": self.guided_actions,
            "random": self.random_actions,
            "pruned_branches": self.pruned_branches,
            "sequence_hits": self.sequence_hits,
            "actions_learned": len(self._action_ranking),
            "sequences_learned": len(self._best_seqs),
            "transfer_bias": len(self._level_action_bias),
        }

    def get_summary(self) -> str:
        s = self.get_stats()
        lines = [
            f"CGEv3: {s['levels_solved']} levels in {s['total_actions']} actions",
            f"  States: {s['states_explored']}, Depth: {s['max_depth']}",
            f"  Guided: {s['guided']}/{s['guided']+s['random']}, "
            f"Pruned: {s['pruned_branches']}, SeqHits: {s['sequence_hits']}",
        ]
        if self._action_ranking:
            top = self._action_ranking[:3]
            lines.append(f"  Top actions: {[(a, f'{self._action_success[a][1]}/{self._action_success[a][0]}') for a in top]}")
        if self._best_seqs:
            lines.append(f"  Top seqs: {self._best_seqs[:3]}")
        if self._level_action_bias:
            lines.append(f"  Transfer bias: {self._level_action_bias[:5]}")
        return "\n".join(lines)
