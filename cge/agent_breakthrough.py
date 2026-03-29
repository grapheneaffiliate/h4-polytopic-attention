"""
CGE Breakthrough Agent — UCB1 branch value estimation.

The key insight: BFS wastes equal budget on every branch. In a tree with
8 branches and 1 good one, BFS wastes 7/8 of its budget. With 4 levels
of branching, (7/8)^4 = 59% wasted.

The fix: treat action selection as a MULTI-ARMED BANDIT problem.
Each action is an "arm." The reward is how many new states we discover.
UCB1 naturally balances exploitation (try what worked) vs exploration
(try what we don't know).

This is NOT incremental improvement. This is a fundamentally different
exploration strategy that allocates budget proportionally to branch quality.

Additionally:
- Branch value memory persists across episodes within a level
- Cross-level transfer: action types that were productive carry bias
- Adaptive: learns the environment's branching structure from data
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from .compression import CompressionLayer


@dataclass
class StateNode:
    """Enhanced state node with branch value tracking."""
    name: str
    actions: set = field(default_factory=set)
    tested: dict = field(default_factory=dict)  # action -> (changed, target)
    depth: int = 0
    # Branch value: how many NEW states were discovered by taking this action
    action_descendants: dict = field(default_factory=lambda: defaultdict(int))
    visit_count: dict = field(default_factory=lambda: defaultdict(int))

    @property
    def untested(self) -> set:
        return self.actions - set(self.tested.keys())

    @property
    def change_rate(self) -> float:
        if not self.tested:
            return 0.0
        return sum(1 for c, _ in self.tested.values() if c) / len(self.tested)

    @property
    def fanout(self) -> int:
        return len(set(t for _, t in self.tested.values() if t is not None))

    @property
    def closed(self) -> bool:
        return not self.untested

    @closed.setter
    def closed(self, value):
        pass  # compatibility with CompressionLayer


class SmartExplorer:
    """
    Graph explorer with UCB1-guided action selection.

    Instead of random choice among untested actions, uses:
    1. UCB1 score = mean_reward + C * sqrt(ln(N) / n_i)
    2. mean_reward = average new states discovered by this action type
    3. Cross-state generalization: action types that work in one state
       tend to work in similar states
    """

    def __init__(self, exploration_constant=1.5):
        self.nodes: dict[str, StateNode] = {}
        self.edges: dict[str, set] = defaultdict(set)
        self.rev_edges: dict[str, set] = defaultdict(set)
        self.frontier: set[str] = set()
        self.next_hop: dict[str, tuple] = {}
        self.root: Optional[str] = None
        self._dirty = False
        self.C = exploration_constant

        # Per-state UCB statistics: (state, action) -> rewards
        self._sa_rewards: dict = defaultdict(list)  # (state, action) -> [reward1, ...]
        self._sa_visits: dict = defaultdict(int)  # (state, action) -> count
        self._state_total_visits: dict = defaultdict(int)  # state -> total visits

        # Global action stats (fallback when state hasn't been visited enough)
        self._action_type_rewards: dict = defaultdict(list)
        self._action_type_visits: dict = defaultdict(int)
        self._total_visits = 0

        # Feature-based learning: extract features from actions (parity, range, etc.)
        # and learn which features predict high reward across ALL states
        self._feature_rewards: dict = defaultdict(list)  # feature -> [rewards]
        self._feature_visits: dict = defaultdict(int)

        # Track descendants per action
        self._pending_source: Optional[str] = None
        self._pending_action = None
        self._states_at_action_start = 0

    def reset(self):
        self.nodes.clear()
        self.edges.clear()
        self.rev_edges.clear()
        self.frontier.clear()
        self.next_hop.clear()
        self.root = None
        self._dirty = False
        self._pending_source = None
        self._pending_action = None
        # Keep action_type stats across resets (cross-episode learning)

    @property
    def num_states(self):
        return len(self.nodes)

    @property
    def max_depth(self):
        return max((n.depth for n in self.nodes.values()), default=0)

    def add_node(self, name: str, actions: set, depth: int = 0):
        if name in self.nodes:
            return
        node = StateNode(name=name, actions=set(actions), depth=depth)
        self.nodes[name] = node
        if self.root is None:
            self.root = name
        if node.untested:
            self.frontier.add(name)
            self._dirty = True
        else:
            node.closed = True

    def record_transition(self, source, action, changed, target=None, target_actions=None):
        if source not in self.nodes:
            return
        node = self.nodes[source]
        node.tested[action] = (changed, target)

        if changed and target is not None:
            self.edges[source].add((action, target))
            self.rev_edges[target].add((action, source))
            if target not in self.nodes and target_actions is not None:
                self.add_node(target, target_actions, depth=node.depth + 1)

        if not node.untested:
            self._close_node(source)

    def begin_action(self, state: str, action):
        """Call before executing an exploration action — starts tracking descendants."""
        # If we had a pending exploration, finalize it first
        self._finalize_pending()
        self._pending_source = state
        self._pending_action = action
        self._states_at_action_start = len(self.nodes)

    def end_action_step(self):
        """Called after every action step. Doesn't finalize — lets reward accumulate."""
        pass  # reward accumulates until next begin_action or finalize

    def _finalize_pending(self):
        """Finalize the pending exploration action's reward."""
        if self._pending_source is None:
            return

        new_states = len(self.nodes) - self._states_at_action_start
        reward = new_states

        state = self._pending_source
        action = self._pending_action

        # Per-state UCB (primary signal)
        sa_key = (state, action)
        self._sa_rewards[sa_key].append(reward)
        self._sa_visits[sa_key] += 1
        self._state_total_visits[state] = self._state_total_visits.get(state, 0) + 1

        # Global fallback
        self._action_type_rewards[action].append(reward)
        self._action_type_visits[action] += 1
        self._total_visits += 1

        # Feature-based learning (structural transfer)
        for feat in self._action_features(action):
            self._feature_rewards[feat].append(reward)
            self._feature_visits[feat] += 1

        if state in self.nodes:
            node = self.nodes[state]
            node.action_descendants[action] += new_states
            node.visit_count[action] += 1

        self._pending_source = None
        self._pending_action = None

    @staticmethod
    def _action_features(action) -> list:
        """Extract generalizable features from an action.
        This is where structural learning happens: the agent learns that
        'even-indexed actions are productive' across all states."""
        features = []
        if isinstance(action, int):
            features.append(("parity", action % 2))
            features.append(("mod3", action % 3))
            features.append(("range4", action // 4))
            features.append(("low", int(action < 4)))
        return features

    def finalize(self):
        """Call at end of episode to record any pending exploration."""
        self._finalize_pending()

    def choose_action(self, current: str, action_order: Optional[list] = None) -> Optional[object]:
        """
        Adaptive action selection:
        - If branch values differ significantly: UCB1 (exploit productive branches)
        - If branch values are similar: use action_order from compression (learn action types)
        - If all tested: navigate toward frontier
        """
        if current not in self.nodes:
            return None
        node = self.nodes[current]

        untested = node.untested
        if untested:
            if self._has_discriminative_signal(current):
                return self._ucb1_select(untested, action_order, current_state=current)
            else:
                if action_order:
                    for a in action_order:
                        if a in untested:
                            return a
                return random.choice(list(untested))

        # Navigate toward frontier
        if self._dirty:
            self._rebuild_distances()
            self._dirty = False

        if current in self.next_hop:
            return self.next_hop[current][0]

        return None

    def _has_discriminative_signal(self, state: str) -> bool:
        """
        Check if UCB has enough data at this state to make meaningful distinctions.
        Returns True if branch values show significant variance (some actions
        are clearly better than others).
        """
        state_visits = self._state_total_visits.get(state, 0)
        if state_visits < 3:
            return False  # not enough data

        # Collect mean rewards for actions tried at this state
        rewards = []
        for action in self.nodes[state].actions:
            sa_key = (state, action)
            if self._sa_visits.get(sa_key, 0) >= 2:
                r = self._sa_rewards.get(sa_key, [0])
                rewards.append(sum(r) / max(len(r), 1))

        if len(rewards) < 2:
            return False

        # Check if there's meaningful variance
        max_r = max(rewards)
        min_r = min(rewards)
        mean_r = sum(rewards) / len(rewards)

        # Signal is discriminative if best action is >2x the average
        if mean_r > 0 and max_r > mean_r * 2:
            return True
        # Or if there's a clear gap between best and rest
        if max_r - min_r > 0.5:
            return True
        return False

    def _ucb1_select(self, actions: set, action_order: Optional[list] = None,
                     current_state: str = None) -> object:
        """
        Per-state UCB1 selection.

        Uses (state, action) rewards when available (strong signal).
        Falls back to global action rewards for unvisited state-action pairs.
        """
        # Check if we have per-state data for current state
        state_visits = self._state_total_visits.get(current_state, 0) if current_state else 0

        if state_visits < 2 and self._total_visits < 5:
            if action_order:
                for a in action_order:
                    if a in actions:
                        return a
            return random.choice(list(actions))

        best_score = -1
        best_action = None

        for action in actions:
            if current_state and state_visits >= 2:
                sa_key = (current_state, action)
                n_i = self._sa_visits.get(sa_key, 0)
                N = state_visits
                if n_i == 0:
                    score = float('inf')
                else:
                    rewards = self._sa_rewards.get(sa_key, [0])
                    mean_reward = sum(rewards) / max(len(rewards), 1)
                    effective_C = self.C / (1 + n_i / 10.0)
                    exploration = effective_C * math.sqrt(math.log(N + 1) / n_i)
                    score = mean_reward + exploration
                    if n_i >= 10 and mean_reward == 0:
                        score = -1.0
            else:
                n_i = self._action_type_visits.get(action, 0)
                N = self._total_visits
                if n_i == 0:
                    score = float('inf')
                else:
                    rewards = self._action_type_rewards.get(action, [0])
                    mean_reward = sum(rewards) / max(len(rewards), 1)
                    effective_C = self.C / (1 + n_i / 10.0)
                    exploration = effective_C * math.sqrt(math.log(N + 1) / n_i)
                    score = mean_reward + exploration
                    if n_i >= 10 and mean_reward == 0:
                        score = -1.0

            if score > best_score:
                best_score = score
                best_action = action
            elif score == best_score and action_order:
                # Tie-break
                a_in = action in (action_order or [])
                b_in = best_action in (action_order or [])
                if a_in and (not b_in or action_order.index(action) < action_order.index(best_action)):
                    best_action = action

        return best_action if best_action is not None else random.choice(list(actions))

    def _feature_prior_bonus(self, action) -> float:
        """
        Small bonus for ordering UNVISITED actions based on structural features.
        Returns 0-1 range — just a tiebreaker, doesn't block exploration.
        Key insight: this orders which unvisited actions to try FIRST,
        but all unvisited actions still get tried eventually.
        """
        features = self._action_features(action)
        if not features:
            return 0.0

        total_weight = 0.0
        weighted_reward = 0.0
        for feat in features:
            visits = self._feature_visits.get(feat, 0)
            if visits >= 10:
                rewards = self._feature_rewards.get(feat, [0])
                mean_r = sum(rewards) / max(len(rewards), 1)
                weight = math.log(visits + 1)
                weighted_reward += mean_r * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_reward / total_weight  # 0-1 range

    def _close_node(self, name):
        if name not in self.nodes:
            return
        node = self.nodes[name]
        self.frontier.discard(name)
        self._dirty = True

    def _score_frontier_node(self, name: str) -> float:
        """
        Score a frontier node by the quality of its ancestral branch.
        Nodes whose ancestors discovered many new states get higher scores.
        This makes the explorer PREFER exploring productive branches.
        """
        score = self.nodes[name].depth * 0.5  # depth bias (prefer deeper)

        # Walk up the ancestor chain and accumulate branch value
        current = name
        ancestor_value = 0
        for _ in range(10):  # max 10 ancestors
            # Find parent
            parent_found = False
            for action, src in self.rev_edges.get(current, set()):
                if src in self.nodes:
                    # Get branch value of the action that led here
                    src_node = self.nodes[src]
                    desc = src_node.action_descendants.get(action, 0)
                    visits = src_node.visit_count.get(action, 0)
                    if visits > 0:
                        ancestor_value += desc / visits
                    current = src
                    parent_found = True
                    break
            if not parent_found:
                break

        score += ancestor_value * 2.0  # strong bias toward high-value branches
        return score

    def _rebuild_distances(self):
        self.next_hop.clear()
        dist = {}
        for name in self.nodes:
            dist[name] = 2**30

        # Sort frontier by branch value score (highest first) instead of just depth
        sorted_frontier = sorted(self.frontier,
                                key=lambda f: self._score_frontier_node(f),
                                reverse=True)

        from collections import deque
        queue = deque()
        for f in sorted_frontier:
            dist[f] = 0
            queue.append(f)

        while queue:
            v = queue.popleft()
            v_dist = dist[v]
            v_score = self._score_frontier_node(v) if v in self.frontier else 0
            for action, u in self.rev_edges.get(v, set()):
                new_dist = v_dist + 1
                if new_dist < dist.get(u, 2**30):
                    dist[u] = new_dist
                    self.next_hop[u] = (action, v)
                    queue.append(u)
                elif new_dist == dist.get(u, 2**30) and u in self.next_hop:
                    # Tie-break by frontier score (prefer high-value branches)
                    _, old_next = self.next_hop[u]
                    old_score = self._score_frontier_node(old_next) if old_next in self.frontier else 0
                    if v_score > old_score:
                        self.next_hop[u] = (action, v)


class BreakthroughAgent:
    """
    The breakthrough: UCB1 branch value estimation + compression + transfer.

    Key differences from all previous agents:
    1. Actions are BANDITS, not random choices — UCB1 allocates budget to productive branches
    2. Branch value (new states discovered) is the reward signal
    3. Cross-episode learning: branch values persist, so episode 2 knows branch 0 is best
    4. Cross-level transfer: action types that were productive carry bias
    5. CompressionLayer provides state-dependent ranking as tie-breaker
    """

    ANALYZE_INTERVAL = 150
    LEARN_THRESHOLD = 50

    def __init__(self, analyze_interval=150, learn_threshold=50,
                 exploration_constant=1.5):
        self.explorer = SmartExplorer(exploration_constant=exploration_constant)
        self.compression = CompressionLayer()  # v1 compression for CausalChain-type envs
        self.ANALYZE_INTERVAL = analyze_interval
        self.LEARN_THRESHOLD = learn_threshold

        self.current_level = 0
        self.total_actions = 0
        self.level_actions = 0
        self.levels_solved = 0

        self.winning_paths: dict[int, list] = {}
        self.effective_history: list[tuple] = []

        self.replaying = False
        self.replay_queue: list[tuple] = []

        # Cross-level transfer
        self._winning_action_counts: dict = defaultdict(int)
        self._transfer_bias: list = []

        # Adaptive branch pruning
        self._branch_window: list[bool] = []
        self._global_changes = 0
        self._global_actions = 0

        # Action efficacy (for tie-breaking)
        self._action_efficacy: dict = defaultdict(lambda: [0, 0])

        # MCTS fallback: when explorer stalls, bypass it entirely
        self._sa_descendants: dict = defaultdict(int)
        self._sa_episode_tries: dict = defaultdict(int)
        self._all_states_ever: set = set()
        self._episode_path: list = []
        self._stall_count = 0
        self._states_at_episode_start = 0
        self._STALL_THRESHOLD = 5
        self._mcts_mode = False

        # Stats
        self.guided = 0
        self.random = 0

    def on_new_state(self, state: str, available_actions: set):
        if state not in self.explorer.nodes:
            self.explorer.add_node(state, available_actions)
            # New state discovered — credit episode path ancestors
            if state not in self._all_states_ever:
                self._all_states_ever.add(state)
                for s, a in self._episode_path:
                    self._sa_descendants[(s, a)] += 1

    def _choose_episode_branch(self, state: str, available_actions: set) -> object:
        """
        Forced rotation + exploit + dead-action avoidance.
        """
        al = sorted(available_actions)
        if not al:
            return None

        # First: filter out actions known to be dead at this state
        # (tested by the explorer, produced no state change)
        node = self.explorer.nodes.get(state)
        live = set(al)
        if node:
            for a, (changed, target) in node.tested.items():
                if not changed and a in live:
                    live.discard(a)
        if not live:
            live = set(al)  # all dead? just try anything

        al = sorted(live)
        min_tries = min(self._sa_episode_tries.get((state, a), 0) for a in al)

        # Candidates: least-tried LIVE actions
        candidates = [a for a in al
                     if self._sa_episode_tries.get((state, a), 0) <= min_tries]

        # If all tried at least once, exploit best branch
        if min_tries > 0:
            descs = {a: self._sa_descendants.get((state, a), 0) for a in al}
            best_desc = max(descs.values()) if descs else 0
            if best_desc > 0:
                candidates = [a for a in al if descs[a] >= best_desc * 0.3]

        return random.choice(candidates) if candidates else random.choice(sorted(available_actions))

    def choose_action(self, state: str, available_actions: set):
        self.total_actions += 1
        self.level_actions += 1

        self.on_new_state(state, available_actions)

        # Replay mode
        if self.replaying and self.replay_queue:
            action = self.replay_queue.pop(0)
            if action in available_actions:
                return action
            self.replaying = False
            self.replay_queue = []

        # MCTS MODE: explorer is stalled, use standalone MCTS (no explorer)
        if self._mcts_mode:
            # Pure MCTS: forced rotation + descendant exploit, ignoring explorer
            al = sorted(available_actions)
            # Filter dead actions (ones we know don't change state at this state)
            live = []
            node = self.explorer.nodes.get(state)
            if node:
                dead = set(a for a, (c, _) in node.tested.items() if not c)
                live = [a for a in al if a not in dead]
            if not live:
                live = al

            # Deterministic rotation: always pick lowest-index untried action
            min_t = min(self._sa_episode_tries.get((state, a), 0) for a in live)
            cands = [a for a in live if self._sa_episode_tries.get((state, a), 0) <= min_t]

            # Exploit if clear winner AND all tried at least once
            if min_t > 0:
                descs = {a: self._sa_descendants.get((state, a), 0) for a in live}
                best = max(descs.values()) if descs else 0
                if best > 0:
                    cands = [a for a in live if descs.get(a, 0) >= best * 0.3]

            action = random.choice(cands)
            self._episode_path.append((state, action))
            self._sa_episode_tries[(state, action)] = \
                self._sa_episode_tries.get((state, action), 0) + 1
            # ALSO register with explorer so it knows about transitions
            self.on_new_state(state, available_actions)
            return action

        # NORMAL MODE: explorer with compression
        node = self.explorer.nodes.get(state)
        is_exploration = node and bool(node.untested) if node else False

        if self.level_actions % self.ANALYZE_INTERVAL == 0 and self.level_actions > 0:
            self.compression.analyze(self.explorer)

        action_order = self._build_action_order(state, available_actions)
        action = self.explorer.choose_action(state, action_order=action_order)

        if action is None:
            action = random.choice(list(available_actions)) if available_actions else None

        if action is not None and is_exploration:
            self.explorer.begin_action(state, action)
        if action is not None:
            self._episode_path.append((state, action))

        return action

    def _build_action_order(self, state: str, available: set) -> list:
        """Build action ordering from compression + transfer + efficacy."""
        # Get v1 compression ranking (state-dependent, proven on CausalChain)
        compression_ranked = self.compression.rank_actions(state, available)

        scored = defaultdict(float)

        # Compression ranking (strongest signal)
        for i, a in enumerate(compression_ranked):
            scored[a] += 5.0 / (i + 1)

        # Transfer bias
        for i, a in enumerate(self._transfer_bias[:5]):
            if a in available:
                scored[a] += 3.0 / (i + 1)

        # Global efficacy
        for a in available:
            if a in self._action_efficacy:
                att, suc = self._action_efficacy[a]
                if att >= 3:
                    scored[a] += 2.0 * (suc / att)

        if not scored:
            return []
        return sorted(available, key=lambda a: scored.get(a, 0), reverse=True)

    def observe_result(self, prev_state: str, action, new_state: str,
                      new_actions: set, changed: bool):
        target = new_state if changed else None
        target_actions = new_actions if changed else None
        self.explorer.record_transition(prev_state, action, changed,
                                       target=target, target_actions=target_actions)

        self.explorer.end_action_step()

        if changed:
            self.effective_history.append((prev_state, action))
            # MCTS: credit ancestors when NEW state discovered
            if new_state not in self._all_states_ever:
                self._all_states_ever.add(new_state)
                for s, a in self._episode_path:
                    self._sa_descendants[(s, a)] += 1

        # Track efficacy
        self._action_efficacy[action][0] += 1
        if changed:
            self._action_efficacy[action][1] += 1

        # Global tracking
        self._global_actions += 1
        if changed:
            self._global_changes += 1
        self._branch_window.append(changed)
        if len(self._branch_window) > 50:
            self._branch_window = self._branch_window[-50:]

    def on_level_complete(self, level: int):
        self.winning_paths[level] = list(self.effective_history)

        for state, action in self.effective_history:
            self._winning_action_counts[action] += 1
        self._transfer_bias = sorted(
            self._winning_action_counts.keys(),
            key=lambda a: self._winning_action_counts[a],
            reverse=True,
        )

        # Tell compression about the win
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
        self._sa_descendants.clear()
        self._sa_episode_tries.clear()
        self._all_states_ever.clear()
        self._episode_path = []
        self._stall_count = 0
        self._states_at_episode_start = 0
        self._mcts_mode = False

    def on_episode_reset(self):
        # Stall detection: did this episode discover any new states?
        cur_states = len(self._all_states_ever)
        if cur_states <= self._states_at_episode_start:
            self._stall_count += 1
        else:
            self._stall_count = 0
        self._states_at_episode_start = cur_states

        # Switch to MCTS when stalled
        if not self._mcts_mode and self._stall_count >= self._STALL_THRESHOLD:
            self._mcts_mode = True

        self._episode_path = []
        self.explorer.finalize()
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
            "transfer": len(self._transfer_bias),
            "ucb_visits": self.explorer._total_visits,
        }

    def get_summary(self) -> str:
        s = self.get_stats()
        lines = [
            f"Breakthrough: {s['levels_solved']} levels in {s['total_actions']} actions",
            f"  States: {s['states_explored']}, Depth: {s['max_depth']}, UCB visits: {s['ucb_visits']}",
        ]
        # Show UCB1 learned values
        if self.explorer._action_type_visits:
            top = sorted(self.explorer._action_type_visits.keys(),
                        key=lambda a: (sum(self.explorer._action_type_rewards.get(a, [0])) /
                                      max(len(self.explorer._action_type_rewards.get(a, [0])), 1)),
                        reverse=True)[:5]
            for a in top:
                rewards = self.explorer._action_type_rewards.get(a, [0])
                visits = self.explorer._action_type_visits.get(a, 0)
                mean = sum(rewards) / max(len(rewards), 1)
                lines.append(f"  Action {a}: mean_reward={mean:.2f}, visits={visits}")
        if self._transfer_bias:
            lines.append(f"  Transfer: {self._transfer_bias[:5]}")
        return "\n".join(lines)
