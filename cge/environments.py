"""
Simulated test environments for CGE.

These model the key properties of ARC-AGI-3 games without needing the SDK:
- State transitions from actions
- Multiple levels (solve one to advance)
- Episode resets (GAME_OVER after N actions)
- Different "game types" testing different aspects of the algorithm

Each environment implements:
  reset() -> (state, available_actions)
  step(action) -> (state, available_actions, changed, level_up, game_over)
"""

import random
from abc import ABC, abstractmethod


class Environment(ABC):
    """Base class for test environments."""

    def __init__(self):
        self.current_level = 0
        self.total_levels = 1
        self.steps_this_episode = 0
        self.episode_budget = 100
        self.name = self.__class__.__name__

    @abstractmethod
    def reset(self) -> tuple:
        """Reset environment. Returns (state, available_actions)."""

    @abstractmethod
    def step(self, action) -> tuple:
        """Take action. Returns (state, available_actions, changed, level_up, game_over)."""


class LinearPuzzle(Environment):
    """
    Simple linear puzzle: must press actions in sequence A→B→C→D→E.
    10 possible actions but only 5 matter. Tests action efficacy learning.

    With 10 actions and only 5 correct, BFS wastes time on the 5 useless ones.
    CGE should learn which 5 work and prioritize them.
    """

    def __init__(self, n_levels=3, solution_length=5, n_actions=10, episode_budget=30):
        super().__init__()
        self.total_levels = n_levels
        self.solution_length = solution_length
        self.n_actions = n_actions
        self.episode_budget = episode_budget
        self.solutions = {}
        # Generate a random solution for each level
        # Key: same action TYPES work across levels but in different order
        self.effective_actions = set(random.sample(range(n_actions), solution_length))
        for level in range(n_levels):
            self.solutions[level] = random.sample(list(self.effective_actions), solution_length)
        self.state = 0  # progress through current level's sequence

    def reset(self):
        self.state = 0
        self.steps_this_episode = 0
        return self._make_state(), set(range(self.n_actions))

    def step(self, action):
        self.steps_this_episode += 1
        if self.steps_this_episode >= self.episode_budget:
            return self._make_state(), set(range(self.n_actions)), False, False, True

        sol = self.solutions[self.current_level]
        if self.state < len(sol) and action == sol[self.state]:
            self.state += 1
            if self.state >= len(sol):
                # Level complete!
                self.current_level += 1
                self.state = 0
                won = self.current_level >= self.total_levels
                return self._make_state(), set(range(self.n_actions)), True, True, won
            return self._make_state(), set(range(self.n_actions)), True, False, False
        else:
            return self._make_state(), set(range(self.n_actions)), False, False, False

    def _make_state(self):
        return f"L{self.current_level}_S{self.state}"


class MazeNavigation(Environment):
    """
    Grid maze: navigate from start to goal using arrow actions.
    Tests: spatial exploration, bottleneck detection, progress gradients.

    The maze has walls that block movement. Only a subset of actions (arrows)
    work, and the effective direction depends on position.
    """

    def __init__(self, width=6, height=6, n_levels=2, episode_budget=80):
        super().__init__()
        self.width = width
        self.height = height
        self.total_levels = n_levels
        self.episode_budget = episode_budget
        # Actions: 0=up, 1=down, 2=left, 3=right, 4-7=useless clicks
        self.n_actions = 8
        self.mazes = {}
        self.goals = {}
        for level in range(n_levels):
            self.mazes[level] = self._generate_maze()
            self.goals[level] = (width - 1, height - 1)
        self.pos = (0, 0)

    def _generate_maze(self):
        """Generate a simple maze (set of blocked cells)."""
        blocked = set()
        for _ in range(self.width * self.height // 4):
            r, c = random.randint(0, self.height-1), random.randint(0, self.width-1)
            if (r, c) != (0, 0) and (r, c) != (self.width-1, self.height-1):
                blocked.add((c, r))
        # Ensure path exists by removing blocks along one route
        for i in range(self.width):
            blocked.discard((i, 0))
        for j in range(self.height):
            blocked.discard((self.width-1, j))
        return blocked

    def reset(self):
        self.pos = (0, 0)
        self.steps_this_episode = 0
        return self._make_state(), set(range(self.n_actions))

    def step(self, action):
        self.steps_this_episode += 1
        if self.steps_this_episode >= self.episode_budget:
            return self._make_state(), set(range(self.n_actions)), False, False, True

        dx, dy = 0, 0
        if action == 0: dy = -1   # up
        elif action == 1: dy = 1  # down
        elif action == 2: dx = -1 # left
        elif action == 3: dx = 1  # right
        else:
            # Useless actions (clicks)
            return self._make_state(), set(range(self.n_actions)), False, False, False

        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        blocked = self.mazes[self.current_level]

        if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in blocked:
            self.pos = (nx, ny)
            changed = True
        else:
            changed = False

        # Check goal
        if self.pos == self.goals[self.current_level]:
            self.current_level += 1
            self.pos = (0, 0)
            won = self.current_level >= self.total_levels
            return self._make_state(), set(range(self.n_actions)), True, True, won

        return self._make_state(), set(range(self.n_actions)), changed, False, False

    def _make_state(self):
        return f"L{self.current_level}_{self.pos[0]}_{self.pos[1]}"


class BottleneckPuzzle(Environment):
    """
    Environment with a tree structure and a single bottleneck.
    Many branches are dead ends; only one path leads through the bottleneck to the goal.
    Tests: bottleneck detection, avoiding dead-end exploration.

    Structure:
      root -> [A, B, C, D, E]  (5 branches)
      Only branch C leads to bottleneck -> goal
      Other branches have 3-5 dead-end states each
    """

    def __init__(self, n_branches=5, dead_end_depth=4, n_levels=2, episode_budget=60):
        super().__init__()
        self.n_branches = n_branches
        self.dead_end_depth = dead_end_depth
        self.total_levels = n_levels
        self.episode_budget = episode_budget
        self.n_actions = n_branches + 2  # branch choices + 2 within-branch actions
        self.graphs = {}
        for level in range(n_levels):
            self.graphs[level] = self._build_graph(level)
        self.state = "root"

    def _build_graph(self, level):
        """Build the state graph for this level."""
        winning_branch = random.randint(0, self.n_branches - 1)
        graph = {}  # state -> {action -> next_state or None}

        # Root state: each branch action leads somewhere
        root_transitions = {}
        for b in range(self.n_branches):
            root_transitions[b] = f"L{level}_B{b}_0"
        graph["root"] = root_transitions

        # Dead-end branches
        for b in range(self.n_branches):
            if b == winning_branch:
                continue
            for d in range(self.dead_end_depth):
                state = f"L{level}_B{b}_{d}"
                transitions = {}
                # Action n_branches = "forward" in branch
                if d < self.dead_end_depth - 1:
                    transitions[self.n_branches] = f"L{level}_B{b}_{d+1}"
                # Action n_branches+1 = "interact" (does nothing in dead ends)
                graph[state] = transitions

        # Winning branch: leads through bottleneck to goal
        for d in range(self.dead_end_depth):
            state = f"L{level}_B{winning_branch}_{d}"
            transitions = {}
            transitions[self.n_branches] = f"L{level}_B{winning_branch}_{d+1}"
            graph[state] = transitions

        # Bottleneck state
        bottleneck = f"L{level}_B{winning_branch}_{self.dead_end_depth}"
        graph[bottleneck] = {self.n_branches + 1: "GOAL"}

        # Goal state
        graph["GOAL"] = {}

        self._winning_branch = winning_branch
        return graph

    def reset(self):
        self.state = "root"
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
                self.state = "root"
                won = self.current_level >= self.total_levels
                return self.state, set(range(self.n_actions)), True, True, won
            self.state = new_state
            return self.state, set(range(self.n_actions)), True, False, False
        else:
            return self.state, set(range(self.n_actions)), False, False, False

    def _make_state(self):
        return self.state


class HiddenPatternPuzzle(Environment):
    """
    Environment where the correct action depends on a hidden pattern in the state.
    States have a "color" (0-3) and the correct action matches the color.
    Tests: learning state→action mapping from exploration data.

    This models ARC games where you need to figure out WHAT to click based on
    visual features — the core perception challenge.
    """

    def __init__(self, n_steps=8, n_levels=3, n_actions=6, episode_budget=50):
        super().__init__()
        self.n_steps = n_steps
        self.total_levels = n_levels
        self.n_actions = n_actions
        self.episode_budget = episode_budget
        # The "rule": state color determines correct action
        # This rule persists across levels (transfer learning opportunity)
        self.color_to_action = {}
        actions = list(range(n_actions))
        random.shuffle(actions)
        for color in range(4):
            self.color_to_action[color] = actions[color]
        self.sequences = {}
        for level in range(n_levels):
            self.sequences[level] = [random.randint(0, 3) for _ in range(n_steps)]
        self.progress = 0

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
            correct = self.color_to_action[seq[self.progress]]
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
        color = seq[self.progress] if self.progress < len(seq) else -1
        return f"L{self.current_level}_P{self.progress}_C{color}"


class LargeStateSpace(Environment):
    """
    Environment with many states but only a narrow solution corridor.
    Models ARC games like re86 (5807 states) where most exploration is wasted.
    Tests: focusing budget on promising regions, not exhaustive BFS.

    Grid of states. Most actions cycle between nearby states.
    Only one specific sequence of actions advances toward the goal.
    """

    def __init__(self, grid_size=10, n_levels=1, episode_budget=200):
        super().__init__()
        self.grid_size = grid_size
        self.total_levels = n_levels
        self.episode_budget = episode_budget
        self.n_actions = 6  # 4 arrows + 2 interact
        self.pos = (0, 0)
        self.goal = (grid_size - 1, grid_size - 1)
        # "Interact" action only works at specific positions
        self.interact_positions = set()
        for i in range(grid_size):
            self.interact_positions.add((i, i))  # diagonal

    def reset(self):
        self.pos = (0, 0)
        self.steps_this_episode = 0
        return self._make_state(), set(range(self.n_actions))

    def step(self, action):
        self.steps_this_episode += 1
        if self.steps_this_episode >= self.episode_budget:
            return self._make_state(), set(range(self.n_actions)), False, False, True

        x, y = self.pos
        changed = False

        if action == 0 and y > 0:
            self.pos = (x, y - 1); changed = True
        elif action == 1 and y < self.grid_size - 1:
            self.pos = (x, y + 1); changed = True
        elif action == 2 and x > 0:
            self.pos = (x - 1, y); changed = True
        elif action == 3 and x < self.grid_size - 1:
            self.pos = (x + 1, y); changed = True
        elif action == 4 and self.pos in self.interact_positions:
            # "Interact" at diagonal positions advances a hidden counter
            changed = True  # state changes (visible via counter in state)
        elif action == 5:
            pass  # useless action

        if self.pos == self.goal:
            self.current_level += 1
            self.pos = (0, 0)
            won = self.current_level >= self.total_levels
            return self._make_state(), set(range(self.n_actions)), True, True, won

        return self._make_state(), set(range(self.n_actions)), changed, False, False

    def _make_state(self):
        return f"L{self.current_level}_{self.pos[0]}_{self.pos[1]}"


def get_all_environments(seed=42) -> list[Environment]:
    """Create one of each environment type with fixed seed for reproducibility."""
    random.seed(seed)
    return [
        LinearPuzzle(n_levels=3, solution_length=5, n_actions=10, episode_budget=30),
        MazeNavigation(width=6, height=6, n_levels=2, episode_budget=80),
        BottleneckPuzzle(n_branches=5, dead_end_depth=4, n_levels=2, episode_budget=60),
        HiddenPatternPuzzle(n_steps=6, n_levels=3, n_actions=6, episode_budget=40),
        LargeStateSpace(grid_size=8, n_levels=1, episode_budget=150),
    ]
