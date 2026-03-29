"""
ARC Bridge — wraps real ARC-AGI-1/2 tasks as Agent Zero environments.

OFFLINE: no API needed. Uses local task JSON files + Python solutions as ground truth.

How it works:
  - Loads a task (train examples + test input)
  - Executes the Python solution to get the correct test output
  - Agent's state = current grid (as tuple-of-tuples hash)
  - Agent's actions = cell edits: set grid[r][c] = color
  - Reward = improvement in cell-level accuracy vs solution
  - Solve = 100% accuracy

This is a SEARCH problem: the agent explores the space of grid edits
to find the transformation that matches the ground truth.
"""

import json
import os
import glob
from typing import Optional

from .env_interface import Env


def load_task(task_id: str, data_dir: str = "data/arc1") -> Optional[dict]:
    """Load an ARC task by ID."""
    path = os.path.join(data_dir, f"{task_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_solutions(data_dir: str = "data") -> dict:
    """Load all Python solutions. Returns {task_id: code_string}."""
    solutions = {}
    for path in glob.glob(os.path.join(data_dir, "arc_python_solutions*.json")):
        with open(path) as f:
            solutions.update(json.load(f))
    return solutions


def execute_solution(code: str, input_grid: list) -> Optional[list]:
    """Execute a Python solution on an input grid. Returns output grid or None."""
    try:
        namespace = {}
        exec(code, namespace)
        solve_fn = namespace.get("solve")
        if solve_fn is None:
            return None
        return solve_fn(input_grid)
    except Exception:
        return None


def grid_accuracy(current: list, target: list) -> float:
    """Cell-level accuracy between two grids."""
    if not current or not target:
        return 0.0
    total = 0
    correct = 0
    for r in range(min(len(current), len(target))):
        for c in range(min(len(current[r]) if r < len(current) else 0,
                          len(target[r]) if r < len(target) else 0)):
            total += 1
            if current[r][c] == target[r][c]:
                correct += 1
    # Penalty for size mismatch
    target_cells = sum(len(row) for row in target)
    if total < target_cells:
        total = target_cells
    return correct / max(total, 1)


class ARCGridEnv(Env):
    """
    Single ARC task as an environment.

    State: the current grid (hashed as string)
    Actions: (row, col, color) encoded as integer = row * W * 10 + col * 10 + color
    Reward: improvement in cell accuracy vs ground truth
    Done: 100% accuracy or budget exhausted

    The grid starts as the test input. The agent edits cells to match the solution.
    """

    def __init__(self, task_id: str, task_data: dict, solution_grid: list,
                 budget: int = 200):
        self._task_id = task_id
        self._task = task_data
        self._target = solution_grid
        self._budget = budget
        self._level = 0
        self._steps = 0

        # Determine output grid dimensions from target
        self._H = len(solution_grid)
        self._W = len(solution_grid[0]) if solution_grid else 0
        self._colors = 10  # ARC uses colors 0-9

        # Start with blank grid or test input resized to target dims
        test_input = task_data["test"][0]["input"]
        self._grid = []
        for r in range(self._H):
            row = []
            for c in range(self._W):
                if r < len(test_input) and c < len(test_input[r]):
                    row.append(test_input[r][c])
                else:
                    row.append(0)
            self._grid = self._grid + [row]

        self._initial_grid = [row[:] for row in self._grid]
        self._n_actions = self._H * self._W * self._colors

    @property
    def name(self) -> str:
        return f"ARC_{self._task_id}"

    @property
    def total_levels(self) -> int:
        return 1

    @property
    def current_level(self) -> int:
        return self._level

    def reset(self):
        self._grid = [row[:] for row in self._initial_grid]
        self._steps = 0
        return self._state(), set(range(self._n_actions))

    def step(self, action):
        self._steps += 1
        if self._steps >= self._budget:
            return self._state(), set(range(self._n_actions)), 0, False, True

        # Decode action: row * W * 10 + col * 10 + color
        color = action % self._colors
        remainder = action // self._colors
        col = remainder % self._W
        row = remainder // self._W

        if row < self._H and col < self._W:
            old_acc = grid_accuracy(self._grid, self._target)
            self._grid[row][col] = color
            new_acc = grid_accuracy(self._grid, self._target)

            improvement = new_acc - old_acc
            changed = improvement != 0

            if new_acc >= 1.0:
                self._level = 1
                return self._state(), set(range(self._n_actions)), 100, True, True

            reward = max(0, improvement * 10)  # scale up small improvements
            return self._state(), set(range(self._n_actions)), reward, False, False

        return self._state(), set(range(self._n_actions)), 0, False, False

    def _state(self):
        return str(tuple(tuple(row) for row in self._grid))


def get_arc_environments(n_tasks: int = 10, data_dir: str = "data",
                        budget: int = 200) -> list[Env]:
    """
    Load N ARC tasks as environments.
    Only loads tasks where we have both the task JSON and a Python solution.
    """
    solutions = load_solutions(data_dir)
    task_dir = os.path.join(data_dir, "arc1")

    envs = []
    for task_id in sorted(solutions.keys()):
        if len(envs) >= n_tasks:
            break

        task_data = load_task(task_id, task_dir)
        if task_data is None:
            continue

        # Execute solution to get ground truth
        code = solutions[task_id]
        test_input = task_data["test"][0]["input"]
        target = execute_solution(code, test_input)
        if target is None:
            continue

        # Skip tasks with very large grids (too many actions for search)
        h, w = len(target), len(target[0]) if target else 0
        if h * w * 10 > 1000:  # more than 1000 actions = too large
            continue

        envs.append(ARCGridEnv(task_id, task_data, target, budget=budget))

    return envs
