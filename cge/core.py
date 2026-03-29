"""
Core graph explorer — tracks state transitions and frontier.

This is a clean reimplementation of the priority-group BFS from the ARC explorers,
but with hooks for the compression layer to influence action selection.
"""

import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

INFINITY = 2**30


@dataclass
class NodeInfo:
    """A node in the state transition graph."""
    name: str
    actions: set = field(default_factory=set)       # all available action ids
    tested: dict = field(default_factory=dict)       # action -> (changed_state, target)
    depth: int = 0
    closed: bool = False
    distance: int = INFINITY  # BFS distance to nearest frontier node

    @property
    def untested(self) -> set:
        return self.actions - set(self.tested.keys())

    @property
    def num_successes(self) -> int:
        return sum(1 for changed, _ in self.tested.values() if changed)

    @property
    def change_rate(self) -> float:
        if not self.tested:
            return 0.0
        return self.num_successes / len(self.tested)

    @property
    def fanout(self) -> int:
        """Number of distinct states reachable from this node."""
        return len(set(t for _, t in self.tested.values() if t is not None))


class GraphExplorer:
    """
    State graph with BFS navigation to frontier.

    Frontier = nodes with untested actions.
    Navigation uses reverse BFS from frontier to find shortest path from any node.
    Supports external action ordering (from CompressionLayer).
    """

    def __init__(self):
        self.nodes: dict[str, NodeInfo] = {}
        self.edges: dict[str, set] = defaultdict(set)      # src -> {(action, dst)}
        self.rev_edges: dict[str, set] = defaultdict(set)   # dst -> {(action, src)}
        self.frontier: set[str] = set()
        self.next_hop: dict[str, tuple] = {}  # node -> (action, next_node)
        self.root: Optional[str] = None
        self._dirty = False

    def reset(self):
        self.nodes.clear()
        self.edges.clear()
        self.rev_edges.clear()
        self.frontier.clear()
        self.next_hop.clear()
        self.root = None
        self._dirty = False

    @property
    def num_states(self) -> int:
        return len(self.nodes)

    @property
    def max_depth(self) -> int:
        return max((n.depth for n in self.nodes.values()), default=0)

    def add_node(self, name: str, actions: set, depth: int = 0):
        if name in self.nodes:
            return
        node = NodeInfo(name=name, actions=set(actions), depth=depth)
        self.nodes[name] = node
        if self.root is None:
            self.root = name
        if node.untested:
            self.frontier.add(name)
            self._dirty = True
        else:
            node.closed = True

    def record_transition(self, source: str, action, changed: bool,
                         target: Optional[str] = None, target_actions: set = None):
        """Record the result of taking action in source state."""
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

    def choose_action(self, current: str,
                     action_order: Optional[list] = None) -> Optional[object]:
        """
        Choose next action from current state.

        If current has untested actions, pick one (respecting action_order if given).
        Otherwise, navigate toward nearest frontier node.
        """
        if current not in self.nodes:
            return None
        node = self.nodes[current]

        # Has untested actions — pick one
        untested = node.untested
        if untested:
            if action_order:
                # Use compression-guided ordering
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

    def _close_node(self, name: str):
        node = self.nodes[name]
        if node.closed:
            return
        node.closed = True
        self.frontier.discard(name)
        self._dirty = True

    def _rebuild_distances(self):
        """Reverse BFS from frontier. Depth-biased: prefer deeper frontier nodes."""
        self.next_hop.clear()
        dist = {}

        for name in self.nodes:
            dist[name] = INFINITY
            self.nodes[name].distance = INFINITY

        # Seed from frontier, deepest first (tie-breaking prefers deeper)
        sorted_frontier = sorted(self.frontier,
                                key=lambda f: self.nodes[f].depth, reverse=True)
        queue = deque()
        for f in sorted_frontier:
            dist[f] = 0
            self.nodes[f].distance = 0
            queue.append(f)

        while queue:
            v = queue.popleft()
            v_dist = dist[v]
            v_depth = self.nodes[v].depth
            for action, u in self.rev_edges.get(v, set()):
                new_dist = v_dist + 1
                if new_dist < dist.get(u, INFINITY):
                    dist[u] = new_dist
                    self.nodes[u].distance = new_dist
                    self.next_hop[u] = (action, v)
                    queue.append(u)
                elif new_dist == dist.get(u, INFINITY) and u in self.next_hop:
                    # Tie-break: prefer route toward deeper frontier
                    _, old_next = self.next_hop[u]
                    old_depth = self.nodes[old_next].depth
                    if v_depth > old_depth:
                        self.next_hop[u] = (action, v)
