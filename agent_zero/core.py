"""
Agent Zero — one agent that combines everything.

UCB1 is the outer loop (always).
Reasoning provides priors (soft bias, not override).
MCTS takes over for tree-like environments.
Compression handles state-dependent action ranking.

Architecture:
  choose_action(state, actions):
    1. Replay solved levels if needed
    2. If tree detected → MCTS sub-agent (independent)
    3. If reasoner has a suggestion → use as UCB prior
    4. UCB1 select among untested actions (with reasoner prior + compression fallback)
    5. Navigate toward best frontier if all tested
"""

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional, Callable

from .reasoner import Reasoner, HeuristicReasoner


# ── State Graph ──────────────────────────────────────────────

@dataclass
class Node:
    actions: set = field(default_factory=set)
    tested: dict = field(default_factory=dict)
    depth: int = 0

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
    def closed(self, v):
        pass


class Graph:
    """Minimal state graph with BFS navigation."""

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, set] = defaultdict(set)
        self.rev_edges: dict[str, set] = defaultdict(set)
        self.frontier: set[str] = set()
        self.next_hop: dict[str, tuple] = {}
        self._dirty = False

    def reset(self):
        self.nodes.clear()
        self.edges.clear()
        self.rev_edges.clear()
        self.frontier.clear()
        self.next_hop.clear()
        self._dirty = False

    @property
    def num_states(self):
        return len(self.nodes)

    def add(self, state, actions, depth=0):
        if state in self.nodes:
            return
        n = Node(actions=set(actions), depth=depth)
        self.nodes[state] = n
        if n.untested:
            self.frontier.add(state)
            self._dirty = True

    def record(self, src, action, changed, target=None, target_actions=None):
        if src not in self.nodes:
            return
        node = self.nodes[src]
        node.tested[action] = (changed, target)
        if changed and target:
            self.edges[src].add((action, target))
            self.rev_edges[target].add((action, src))
            if target not in self.nodes and target_actions:
                self.add(target, target_actions, depth=node.depth + 1)
        if not node.untested:
            self.frontier.discard(src)
            self._dirty = True

    def navigate(self, current):
        """Get next-hop action toward nearest frontier."""
        if self._dirty:
            self._rebuild()
            self._dirty = False
        if current in self.next_hop:
            return self.next_hop[current][0]
        return None

    def _rebuild(self):
        self.next_hop.clear()
        dist = {s: 2**30 for s in self.nodes}
        q = deque()
        for f in sorted(self.frontier, key=lambda s: self.nodes[s].depth, reverse=True):
            dist[f] = 0
            q.append(f)
        while q:
            v = q.popleft()
            for action, u in self.rev_edges.get(v, set()):
                if dist[v] + 1 < dist.get(u, 2**30):
                    dist[u] = dist[v] + 1
                    self.next_hop[u] = (action, v)
                    q.append(u)


# ── MCTS Sub-Agent ───────────────────────────────────────────

class MCTSSub:
    """Independent MCTS. No shared state with Graph."""

    def __init__(self):
        self.sa_desc = defaultdict(int)
        self.sa_tries = defaultdict(int)
        self.level_states = set()
        self.path = []

    def choose(self, state, actions):
        self.level_states.add(state)
        al = sorted(actions)
        # Filter dead actions
        min_t = min(self.sa_tries.get((state, a), 0) for a in al)
        cands = [a for a in al if self.sa_tries.get((state, a), 0) <= min_t]
        if min_t > 0:
            descs = {a: self.sa_desc.get((state, a), 0) for a in al}
            best = max(descs.values()) if descs else 0
            if best > 3:
                cands = [a for a in al if descs[a] >= best * 0.5]
        action = random.choice(cands)
        self.path.append((state, action))
        self.sa_tries[(state, action)] = self.sa_tries.get((state, action), 0) + 1
        return action

    def observe(self, prev, action, new_state, changed):
        if changed and new_state not in self.level_states:
            self.level_states.add(new_state)
            for s, a in self.path:
                self.sa_desc[(s, a)] += 1

    def episode_reset(self):
        self.path = []

    def level_reset(self):
        self.level_states.clear()
        self.sa_desc.clear()
        self.sa_tries.clear()
        self.path = []


# ── Agent Zero ───────────────────────────────────────────────

class AgentZero:
    """
    One agent. UCB1 outer loop. Reasoning as prior. MCTS for trees.

    Parameters:
        reasoner: Callable that takes a context dict and returns action suggestions.
                  Default: HeuristicReasoner (no LLM needed).
        C: UCB1 exploration constant.
        analyze_every: How often to run compression analysis.
        stall_threshold: Episodes with 0 new states before MCTS triggers.
    """

    def __init__(self, reasoner: Reasoner = None, C: float = 1.5,
                 analyze_every: int = 150, stall_threshold: int = 10):
        self.graph = Graph()
        self.reasoner = reasoner or HeuristicReasoner()
        self.C = C
        self.analyze_every = analyze_every
        self.stall_threshold = stall_threshold

        # UCB per-state tracking
        self._sa_rewards: dict = defaultdict(list)
        self._sa_visits: dict = defaultdict(int)
        self._state_visits: dict = defaultdict(int)

        # Action efficacy
        self._efficacy: dict = defaultdict(lambda: [0, 0])

        # Level/episode tracking
        self.current_level = 0
        self.levels_solved = 0
        self.total_actions = 0
        self.level_actions = 0
        self.winning_paths: dict = {}
        self.effective_history: list = []

        # Replay
        self.replaying = False
        self.replay_queue: list = []

        # Transfer
        self._winning_action_counts: dict = defaultdict(int)
        self._transfer_bias: list = []

        # Tree detection + MCTS
        self._mcts: Optional[MCTSSub] = None
        self._stall_count = 0
        self._prev_states = 0
        self._episodes = 0
        self._episode_visits: list = []

        # Reasoner state
        self._reasoner_prior: dict = {}  # action -> prior_score from last reasoning call
        self._actions_since_progress = 0

    # ── Public Interface ────────────────────────────────────

    def choose_action(self, state, available_actions: set):
        self.total_actions += 1
        self.level_actions += 1
        self._actions_since_progress += 1
        self._episode_visits.append(state)

        # Register state
        if state not in self.graph.nodes:
            self.graph.add(state, available_actions)

        # 1. Replay
        if self.replaying and self.replay_queue:
            if self.winning_paths and state not in self._replay_roots():
                self.replaying = False
                self.replay_queue = []
            else:
                action = self.replay_queue.pop(0)
                if action in available_actions:
                    return action
                self.replaying = False
                self.replay_queue = []

        # 2. MCTS mode
        if self._mcts is not None:
            return self._mcts.choose(state, available_actions)

        # 3. Reasoner: ask for suggestions periodically when stuck
        if (self._actions_since_progress > 200 and
            self.level_actions % 100 == 0):
            self._ask_reasoner(state, available_actions)

        # 4. UCB1 + compression + reasoner prior
        node = self.graph.nodes.get(state)
        if node and node.untested:
            return self._ucb_select(state, node.untested, available_actions)

        # 5. Navigate
        action = self.graph.navigate(state)
        if action is not None:
            return action

        return random.choice(list(available_actions))

    def observe(self, prev_state, action, new_state, new_actions, changed):
        """Record transition result."""
        # MCTS delegation
        if self._mcts is not None and not self.replaying:
            self._mcts.observe(prev_state, action, new_state, changed)
            if changed:
                self.effective_history.append((prev_state, action))
                self._actions_since_progress = 0
            self.graph.add(new_state, new_actions)  # still track for tree detection
            return

        target = new_state if changed else None
        target_actions = new_actions if changed else None
        self.graph.record(prev_state, action, changed, target, target_actions)

        if changed:
            self.effective_history.append((prev_state, action))
            self._actions_since_progress = 0

        # UCB reward: 1 if changed state, 0 if not
        sa_key = (prev_state, action)
        self._sa_rewards[sa_key].append(1.0 if changed else 0.0)
        self._sa_visits[sa_key] += 1
        self._state_visits[prev_state] = self._state_visits.get(prev_state, 0) + 1

        # Efficacy
        self._efficacy[action][0] += 1
        if changed:
            self._efficacy[action][1] += 1

    def on_level_complete(self, level: int):
        self.winning_paths[level] = list(self.effective_history)
        for _, action in self.effective_history:
            self._winning_action_counts[action] += 1
        self._transfer_bias = sorted(
            self._winning_action_counts.keys(),
            key=lambda a: self._winning_action_counts[a], reverse=True)

        self.levels_solved += 1
        self.current_level = level + 1
        self.graph.reset()
        self.effective_history = []
        self.level_actions = 0
        self._stall_count = 0
        self._prev_states = 0
        self._episodes = 0
        self._episode_visits = []
        self._actions_since_progress = 0
        self._reasoner_prior.clear()
        if self._mcts and level + 1 > self.levels_solved:
            self._mcts.level_reset()

    def on_episode_reset(self):
        self._episodes += 1

        if self._mcts is not None:
            self._mcts.episode_reset()
        else:
            # Tree detection (episodes 1-3)
            if self._episodes <= 3:
                self._check_tree()
            # Stall detection
            cur = self.graph.num_states
            if cur <= self._prev_states:
                self._stall_count += 1
            else:
                self._stall_count = 0
            self._prev_states = cur
            if self._stall_count >= self.stall_threshold:
                self._mcts = MCTSSub()

        self._episode_visits = []
        self.replay_queue = []
        for level in sorted(self.winning_paths.keys()):
            for _, action in self.winning_paths[level]:
                self.replay_queue.append(action)
        self.replaying = bool(self.replay_queue)
        self.effective_history = []

    # ── UCB1 Selection ──────────────────────────────────────

    def _ucb_select(self, state, untested: set, available: set):
        sv = self._state_visits.get(state, 0)
        if sv < 2:
            # Not enough data — use transfer bias or random
            return self._biased_pick(untested)

        best_score = -1
        best = None
        for action in untested:
            sa = (state, action)
            ni = self._sa_visits.get(sa, 0)
            if ni == 0:
                # Unvisited: high exploration + reasoner prior
                score = 100.0 + self._reasoner_prior.get(action, 0)
            else:
                rewards = self._sa_rewards.get(sa, [0])
                mean = sum(rewards) / len(rewards)
                c_eff = self.C / (1 + ni / 10.0)
                score = mean + c_eff * math.sqrt(math.log(sv + 1) / ni)
                score += self._reasoner_prior.get(action, 0) * 0.5  # soft prior
                if ni >= 10 and mean == 0:
                    score = -1.0
            if score > best_score:
                best_score = score
                best = action
        return best if best else random.choice(list(untested))

    def _biased_pick(self, actions: set):
        """Pick from untested using transfer bias."""
        if self._transfer_bias:
            for a in self._transfer_bias:
                if a in actions:
                    return a
        return random.choice(list(actions))

    # ── Tree Detection ──────────────────────────────────────

    def _check_tree(self):
        if self.graph.num_states < 3:
            return
        # Acyclic check: no edge goes to equal-or-lower depth
        for src, edges in self.graph.edges.items():
            src_d = self.graph.nodes[src].depth if src in self.graph.nodes else 0
            for _, dst in edges:
                dst_d = self.graph.nodes[dst].depth if dst in self.graph.nodes else 0
                if dst_d <= src_d:
                    return  # has back-edge → not a tree
        # Branching check: root has 2+ children
        root = self._episode_visits[0] if self._episode_visits else None
        if root and root in self.graph.edges:
            children = len(set(d for _, d in self.graph.edges[root]))
            if children >= 2:
                self._mcts = MCTSSub()

    # ── Reasoner Integration ────────────────────────────────

    def _ask_reasoner(self, state, available_actions):
        """Ask reasoner for action suggestions. Results become UCB priors."""
        # Build context
        top_actions = sorted(
            self._efficacy.keys(),
            key=lambda a: self._efficacy[a][1] / max(self._efficacy[a][0], 1),
            reverse=True)[:5]

        context = {
            "state": str(state),
            "available_actions": sorted(available_actions),
            "states_explored": self.graph.num_states,
            "level": self.current_level,
            "actions_since_progress": self._actions_since_progress,
            "top_actions": [(a, self._efficacy[a][1], self._efficacy[a][0])
                          for a in top_actions],
            "total_actions": self.total_actions,
        }

        suggestions = self.reasoner.suggest(context)
        self._reasoner_prior.clear()
        for action, weight in suggestions.items():
            if action in available_actions:
                self._reasoner_prior[action] = weight

    # ── Replay ──────────────────────────────────────────────

    def _replay_roots(self) -> set:
        roots = set()
        for path in self.winning_paths.values():
            if path:
                roots.add(path[0][0])
        return roots

    # ── Stats ───────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "total_actions": self.total_actions,
            "levels_solved": self.levels_solved,
            "states": self.graph.num_states,
            "mcts_active": self._mcts is not None,
            "mcts_states": self._mcts.level_states if self._mcts else set(),
            "reasoner_active": bool(self._reasoner_prior),
        }
