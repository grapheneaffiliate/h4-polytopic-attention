"""BFS baseline agent for comparison."""
import random
from collections import defaultdict, deque


class BFSAgent:
    """Plain BFS. No learning. The control group."""

    def __init__(self):
        self.nodes = {}  # state -> {actions, tested, untested}
        self.edges = defaultdict(set)
        self.rev_edges = defaultdict(set)
        self.frontier = set()
        self.next_hop = {}
        self._dirty = False
        self.winning_paths = {}
        self.effective_history = []
        self.replaying = False
        self.replay_queue = []
        self.current_level = 0
        self.levels_solved = 0
        self.total_actions = 0

    def choose_action(self, state, available_actions):
        self.total_actions += 1
        if state not in self.nodes:
            self.nodes[state] = {"actions": set(available_actions), "tested": {}}
            untested = set(available_actions)
            if untested:
                self.frontier.add(state)
                self._dirty = True

        if self.replaying and self.replay_queue:
            a = self.replay_queue.pop(0)
            if a in available_actions:
                return a
            self.replaying = False
            self.replay_queue = []

        untested = self.nodes[state]["actions"] - set(self.nodes[state]["tested"].keys())
        if untested:
            return random.choice(list(untested))

        if self._dirty:
            self._rebuild()
            self._dirty = False
        if state in self.next_hop:
            return self.next_hop[state][0]
        return random.choice(list(available_actions))

    def observe(self, prev, action, new_state, new_actions, changed):
        if prev not in self.nodes:
            return
        self.nodes[prev]["tested"][action] = (changed, new_state if changed else None)
        if changed and new_state:
            self.edges[prev].add((action, new_state))
            self.rev_edges[new_state].add((action, prev))
            if new_state not in self.nodes:
                self.nodes[new_state] = {"actions": set(new_actions), "tested": {}}
                if new_actions:
                    self.frontier.add(new_state)
                    self._dirty = True
        untested = self.nodes[prev]["actions"] - set(self.nodes[prev]["tested"].keys())
        if not untested:
            self.frontier.discard(prev)
            self._dirty = True
        if changed:
            self.effective_history.append((prev, action))

    def on_level_complete(self, level):
        self.winning_paths[level] = list(self.effective_history)
        self.levels_solved += 1
        self.current_level = level + 1
        self.nodes.clear(); self.edges.clear(); self.rev_edges.clear()
        self.frontier.clear(); self.next_hop.clear(); self._dirty = False
        self.effective_history = []

    def on_episode_reset(self):
        self.replay_queue = []
        for level in sorted(self.winning_paths.keys()):
            for _, action in self.winning_paths[level]:
                self.replay_queue.append(action)
        self.replaying = bool(self.replay_queue)
        self.effective_history = []

    def _rebuild(self):
        self.next_hop.clear()
        dist = {s: 2**30 for s in self.nodes}
        q = deque()
        for f in self.frontier:
            dist[f] = 0; q.append(f)
        while q:
            v = q.popleft()
            for a, u in self.rev_edges.get(v, set()):
                if dist[v] + 1 < dist.get(u, 2**30):
                    dist[u] = dist[v] + 1
                    self.next_hop[u] = (a, v)
                    q.append(u)
