"""
Hard environments modeled on real ARC-AGI-3 game properties.

Based on diagnostic data from SESSION_HANDOFF.md:
- re86: 5807 states, 99% change rate, 100-action episodes, arrows only
- cn04: 5614 states, 67% change rate, 150-action episodes, clicks+arrows
- sb26: 2191 states, 82% change rate, click+space+undo
- sk48: 5 states total (stuck, needs grid-click breakthrough)
- lp85: 344-action episodes, 5/8 levels solved (the best performer)

These environments test the failure modes we need to solve:
1. DeepTreeSearch: many dead-end branches (models BottleneckPuzzle but harder)
2. NeedleInHaystack: huge state space, tiny solution corridor
3. StuckGame: almost no state transitions unless you find the right action type
4. MultiLevelMaze: long episodes, multiple levels, needs replay + transfer
5. RuleLearning: state-dependent rules that transfer across levels
6. CausalChain: must discover multi-step action sequences
"""

import random
from abc import ABC, abstractmethod
from collections import defaultdict


class Environment(ABC):
    def __init__(self):
        self.current_level = 0
        self.total_levels = 1
        self.steps_this_episode = 0
        self.episode_budget = 100
        self.name = self.__class__.__name__

    @abstractmethod
    def reset(self) -> tuple:
        """Returns (state, available_actions)."""
    @abstractmethod
    def step(self, action) -> tuple:
        """Returns (state, available_actions, changed, level_up, game_over)."""


class DeepTreeSearch(Environment):
    """
    Deep tree with many dead-end branches. Models the BottleneckPuzzle failure.

    Key test: can the agent learn to RECOGNIZE dead-end branch patterns
    and avoid them, instead of exhaustively exploring each one?

    Structure per level:
    - Root has K branches (K=8)
    - Each branch has D levels of sub-branching (D=4)
    - Only 1 path through the tree leads to the goal
    - Dead ends have recognizable signatures (odd-numbered branches tend to be dead)
    - The PATTERN of which branches are good transfers across levels
    """

    def __init__(self, n_branches=8, depth=4, n_levels=4, episode_budget=60):
        super().__init__()
        self.n_branches = n_branches
        self.depth = depth
        self.total_levels = n_levels
        self.episode_budget = episode_budget
        # Actions: 0..n_branches-1 = choose branch, n_branches = "interact"
        self.n_actions = n_branches + 1
        self.state = "root"

        # The key pattern: even-indexed branches at each depth tend to lead forward
        # This is learnable! Dead-end branches share a signature.
        self.graphs = {}
        for level in range(n_levels):
            self.graphs[level] = self._build_graph(level)

    def _build_graph(self, level):
        """Build tree with a pattern: "good" branches have even index at each depth."""
        graph = {}
        # Use a consistent pattern with slight variation per level
        rng = random.Random(level * 137 + 42)

        # Which branch index is "good" at each depth — stays mostly even with drift
        good_path = []
        for d in range(self.depth):
            # Tend toward even indices, but not always the same one
            candidates = [i for i in range(self.n_branches) if i % 2 == 0]
            good_path.append(rng.choice(candidates))

        # Build the tree
        def build(prefix, d):
            if d >= self.depth:
                return
            transitions = {}
            for b in range(self.n_branches):
                child = f"{prefix}_B{b}"
                transitions[b] = child
                if b == good_path[d]:
                    build(child, d + 1)
                else:
                    # Dead end: 1-2 more nodes then nothing
                    dead_depth = rng.randint(1, 2)
                    cur = child
                    for dd in range(dead_depth):
                        nxt = f"{cur}_D{dd}"
                        graph[cur] = {self.n_branches: nxt}  # only "forward" works
                        cur = nxt
                    graph[cur] = {}  # terminal dead end

            graph[prefix] = transitions

            # At the deepest good node, "interact" leads to goal
            if d == self.depth - 1:
                deepest = f"{prefix}_B{good_path[d]}"
                # Walk to the end of the good path
                cur = deepest
                for dd in range(2):
                    nxt = f"{cur}_F{dd}"
                    graph[cur] = {self.n_branches: nxt}
                    cur = nxt
                graph[cur] = {self.n_branches: "GOAL"}

        build(f"L{level}", 0)
        graph["GOAL"] = {}
        return graph

    def reset(self):
        self.state = f"L{self.current_level}"
        self.steps_this_episode = 0
        return self.state, set(range(self.n_actions))

    def step(self, action):
        self.steps_this_episode += 1
        if self.steps_this_episode >= self.episode_budget:
            return self.state, set(range(self.n_actions)), False, False, True

        graph = self.graphs[self.current_level]
        transitions = graph.get(self.state, {})

        if action in transitions:
            new_state = transitions[action]
            if new_state == "GOAL":
                self.current_level += 1
                self.state = f"L{self.current_level}" if self.current_level < self.total_levels else "WIN"
                won = self.current_level >= self.total_levels
                return self.state, set(range(self.n_actions)), True, True, won
            self.state = new_state
            return self.state, set(range(self.n_actions)), True, False, False

        return self.state, set(range(self.n_actions)), False, False, False


class NeedleInHaystack(Environment):
    """
    Huge state space (grid navigation) with a specific solution corridor.
    Models re86 (5807 states, 99% change rate) and cn04 (5614 states).

    20x20 grid = 400 states. Most actions change state (high change rate).
    But the goal requires visiting specific waypoints in order.
    Random exploration finds many states but never the solution.

    Key test: can the agent learn to follow the progress gradient
    (waypoint sequence) instead of exploring aimlessly?
    """

    def __init__(self, grid_size=15, n_waypoints=4, n_levels=2,
                 n_actions=8, episode_budget=150):
        super().__init__()
        self.grid_size = grid_size
        self.total_levels = n_levels
        self.n_actions = n_actions  # 4 arrows + 4 useless
        self.episode_budget = episode_budget
        self.pos = (0, 0)
        self.waypoint_idx = 0

        # Generate waypoints per level
        self.waypoints = {}
        for level in range(n_levels):
            rng = random.Random(level * 97 + 13)
            wps = []
            for _ in range(n_waypoints):
                wps.append((rng.randint(0, grid_size-1), rng.randint(0, grid_size-1)))
            # Final waypoint is the goal
            wps.append((grid_size-1, grid_size-1))
            self.waypoints[level] = wps

    def reset(self):
        self.pos = (0, 0)
        self.waypoint_idx = 0
        self.steps_this_episode = 0
        return self._make_state(), set(range(self.n_actions))

    def step(self, action):
        self.steps_this_episode += 1
        if self.steps_this_episode >= self.episode_budget:
            return self._make_state(), set(range(self.n_actions)), False, False, True

        x, y = self.pos
        changed = False

        if action == 0 and y > 0: self.pos = (x, y-1); changed = True
        elif action == 1 and y < self.grid_size-1: self.pos = (x, y+1); changed = True
        elif action == 2 and x > 0: self.pos = (x-1, y); changed = True
        elif action == 3 and x < self.grid_size-1: self.pos = (x+1, y); changed = True
        # Actions 4-7: useless

        # Check waypoint
        wps = self.waypoints[self.current_level]
        if self.waypoint_idx < len(wps) and self.pos == wps[self.waypoint_idx]:
            self.waypoint_idx += 1
            changed = True  # waypoint hit changes state representation
            if self.waypoint_idx >= len(wps):
                self.current_level += 1
                self.pos = (0, 0)
                self.waypoint_idx = 0
                won = self.current_level >= self.total_levels
                return self._make_state(), set(range(self.n_actions)), True, True, won

        return self._make_state(), set(range(self.n_actions)), changed, False, False

    def _make_state(self):
        return f"L{self.current_level}_{self.pos[0]}_{self.pos[1]}_W{self.waypoint_idx}"


class StuckGame(Environment):
    """
    Models sk48/lf52: almost no state transitions with normal actions.
    Only a specific rare action type (like grid-click) discovers new states.

    12 actions total. Actions 0-9 almost never change state.
    Actions 10-11 are "special" and change state, but only at specific positions.

    Key test: can the agent learn that most actions are useless and
    focus budget on the rare effective ones? This is the grid-click insight.
    """

    def __init__(self, n_positions=6, n_levels=2, episode_budget=100):
        super().__init__()
        self.n_positions = n_positions
        self.total_levels = n_levels
        self.episode_budget = episode_budget
        self.n_actions = 12  # 0-9 useless, 10-11 special
        self.pos = 0
        self.activated = set()

        # Solution: activate all positions using special actions
        self.activate_map = {}
        for level in range(n_levels):
            rng = random.Random(level * 53 + 7)
            self.activate_map[level] = {}
            for p in range(n_positions):
                # Which special action works at this position
                self.activate_map[level][p] = rng.choice([10, 11])

    def reset(self):
        self.pos = 0
        self.activated = set()
        self.steps_this_episode = 0
        return self._make_state(), set(range(self.n_actions))

    def step(self, action):
        self.steps_this_episode += 1
        if self.steps_this_episode >= self.episode_budget:
            return self._make_state(), set(range(self.n_actions)), False, False, True

        changed = False
        amap = self.activate_map[self.current_level]

        if action in (10, 11):
            # Special actions: might activate current position
            if self.pos not in self.activated and amap.get(self.pos) == action:
                self.activated.add(self.pos)
                changed = True
                if len(self.activated) >= self.n_positions:
                    self.current_level += 1
                    self.pos = 0
                    self.activated = set()
                    won = self.current_level >= self.total_levels
                    return self._make_state(), set(range(self.n_actions)), True, True, won
            # Move to next position after special action
            self.pos = (self.pos + 1) % self.n_positions
            changed = True
        elif action < 4:
            # Arrow actions: move between positions (sometimes)
            if action == 0 and self.pos > 0:
                self.pos -= 1; changed = True
            elif action == 1 and self.pos < self.n_positions - 1:
                self.pos += 1; changed = True
        # Actions 2-9: completely useless

        return self._make_state(), set(range(self.n_actions)), changed, False, False

    def _make_state(self):
        activated = "".join(str(int(i in self.activated)) for i in range(self.n_positions))
        return f"L{self.current_level}_P{self.pos}_A{activated}"


class CausalChain(Environment):
    """
    Must discover and execute multi-step action sequences.
    No single action advances the puzzle — only specific SEQUENCES work.

    Models games where you must: click A, then click B, then press arrow,
    and ONLY that exact sequence advances the level.

    Key test: can the agent learn that progress requires sequences,
    not individual actions? This is the hardest challenge for BFS.
    """

    def __init__(self, chain_length=3, n_actions=8, n_levels=3, episode_budget=80):
        super().__init__()
        self.chain_length = chain_length
        self.n_actions = n_actions
        self.total_levels = n_levels
        self.episode_budget = episode_budget
        self.progress = 0  # progress through current chain
        self.recent_actions = []  # last N actions taken

        # Generate required chains per level
        # Key: same action TYPES appear in chains across levels (transferable)
        self.chains = {}
        effective = random.sample(range(n_actions), min(chain_length + 1, n_actions))
        for level in range(n_levels):
            rng = random.Random(level * 71 + 29)
            chain = [rng.choice(effective) for _ in range(chain_length)]
            self.chains[level] = chain

    def reset(self):
        self.progress = 0
        self.recent_actions = []
        self.steps_this_episode = 0
        return self._make_state(), set(range(self.n_actions))

    def step(self, action):
        self.steps_this_episode += 1
        if self.steps_this_episode >= self.episode_budget:
            return self._make_state(), set(range(self.n_actions)), False, False, True

        chain = self.chains[self.current_level]
        self.recent_actions.append(action)
        if len(self.recent_actions) > self.chain_length:
            self.recent_actions = self.recent_actions[-self.chain_length:]

        changed = False

        # Check if recent actions match the chain
        if len(self.recent_actions) >= self.chain_length:
            if self.recent_actions[-self.chain_length:] == chain:
                self.progress += 1
                self.recent_actions = []
                changed = True
                # Need to complete chain 3 times to solve level
                if self.progress >= 3:
                    self.current_level += 1
                    self.progress = 0
                    won = self.current_level >= self.total_levels
                    return self._make_state(), set(range(self.n_actions)), True, True, won
        else:
            # Partial chain match gives a "hint" (state changes)
            if action == chain[min(self.progress, len(chain)-1) % len(chain)]:
                changed = True

        return self._make_state(), set(range(self.n_actions)), changed, False, False

    def _make_state(self):
        return f"L{self.current_level}_P{self.progress}_R{''.join(map(str, self.recent_actions[-3:]))}"


class RuleLearning(Environment):
    """
    Each state has observable features, and the correct action is determined
    by a RULE over those features. The rule transfers across levels.

    Features: (color, shape, size) encoded in state name.
    Rule: e.g., "if color==red, action 2; if shape==circle, action 5"

    This models ARC games where visual features determine the correct interaction.
    Key test: can the agent learn the rule from exploration and apply it
    to unseen states? This is the core of abstract reasoning.
    """

    def __init__(self, n_features=3, n_values=4, n_actions=8,
                 n_steps=10, n_levels=5, episode_budget=60):
        super().__init__()
        self.n_features = n_features
        self.n_values = n_values
        self.n_actions = n_actions
        self.n_steps = n_steps
        self.total_levels = n_levels
        self.episode_budget = episode_budget
        self.progress = 0

        # The RULE: one feature determines the action
        # This stays the same across ALL levels (transfer learning test)
        self.rule_feature = random.randint(0, n_features - 1)
        self.rule_map = {}
        actions = list(range(n_actions))
        random.shuffle(actions)
        for v in range(n_values):
            self.rule_map[v] = actions[v % n_actions]

        # Generate state sequences per level
        self.sequences = {}
        for level in range(n_levels):
            rng = random.Random(level * 83 + 17)
            seq = []
            for _ in range(n_steps):
                features = tuple(rng.randint(0, n_values-1) for _ in range(n_features))
                seq.append(features)
            self.sequences[level] = seq

    def reset(self):
        self.progress = 0
        self.steps_this_episode = 0
        return self._make_state(), set(range(self.n_actions))

    def step(self, action):
        self.steps_this_episode += 1
        if self.steps_this_episode >= self.episode_budget:
            return self._make_state(), set(range(self.n_actions)), False, False, True

        seq = self.sequences[self.current_level]
        if self.progress < len(seq):
            features = seq[self.progress]
            correct = self.rule_map[features[self.rule_feature]]
            if action == correct:
                self.progress += 1
                if self.progress >= len(seq):
                    self.current_level += 1
                    self.progress = 0
                    won = self.current_level >= self.total_levels
                    return self._make_state(), set(range(self.n_actions)), True, True, won
                return self._make_state(), set(range(self.n_actions)), True, False, False

        return self._make_state(), set(range(self.n_actions)), False, False, False

    def _make_state(self):
        seq = self.sequences.get(self.current_level, [])
        if self.progress < len(seq):
            features = seq[self.progress]
            feat_str = "_".join(f"F{i}V{v}" for i, v in enumerate(features))
        else:
            feat_str = "done"
        return f"L{self.current_level}_P{self.progress}_{feat_str}"


def get_hard_environments(seed=42) -> list[Environment]:
    """Create the full hard benchmark suite."""
    random.seed(seed)
    return [
        DeepTreeSearch(n_branches=8, depth=4, n_levels=4, episode_budget=60),
        NeedleInHaystack(grid_size=12, n_waypoints=3, n_levels=2, episode_budget=120),
        StuckGame(n_positions=5, n_levels=2, episode_budget=80),
        CausalChain(chain_length=3, n_actions=8, n_levels=3, episode_budget=60),
        RuleLearning(n_features=3, n_values=4, n_actions=8, n_steps=8, n_levels=5, episode_budget=50),
    ]
