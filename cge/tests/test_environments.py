"""Tests for simulated environments."""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from cge.environments import (
    LinearPuzzle, MazeNavigation, BottleneckPuzzle,
    HiddenPatternPuzzle, LargeStateSpace, get_all_environments
)


def test_linear_solvable():
    """LinearPuzzle can be solved by following the solution."""
    random.seed(42)
    env = LinearPuzzle(n_levels=1, solution_length=3, n_actions=6, episode_budget=20)
    state, actions = env.reset()
    sol = env.solutions[0]
    for a in sol:
        state, actions, changed, level_up, game_over = env.step(a)
        assert changed, f"Correct action {a} should change state"
    assert level_up, "Should complete level after full solution"


def test_linear_wrong_action():
    random.seed(42)
    env = LinearPuzzle(n_levels=1, solution_length=3, n_actions=6, episode_budget=20)
    state, actions = env.reset()
    # Find a wrong action
    sol = env.solutions[0]
    wrong = [a for a in range(6) if a != sol[0]][0]
    state, actions, changed, level_up, game_over = env.step(wrong)
    assert not changed
    assert not level_up


def test_linear_episode_budget():
    random.seed(42)
    env = LinearPuzzle(n_levels=1, solution_length=3, n_actions=6, episode_budget=5)
    state, actions = env.reset()
    for _ in range(5):
        state, actions, changed, level_up, game_over = env.step(0)
    assert game_over


def test_maze_movement():
    random.seed(42)
    env = MazeNavigation(width=5, height=5, n_levels=1, episode_budget=100)
    state, actions = env.reset()
    # Move right (action 3)
    new_state, actions, changed, _, _ = env.step(3)
    if changed:
        assert new_state != state


def test_maze_useless_actions():
    random.seed(42)
    env = MazeNavigation(width=5, height=5, n_levels=1, episode_budget=100)
    state, actions = env.reset()
    # Actions 4-7 should never change state
    for a in [4, 5, 6, 7]:
        new_state, _, changed, _, _ = env.step(a)
        assert not changed, f"Action {a} should be useless in maze"


def test_bottleneck_structure():
    random.seed(42)
    env = BottleneckPuzzle(n_branches=3, dead_end_depth=3, n_levels=1, episode_budget=40)
    state, actions = env.reset()
    assert state == "root"
    assert env.n_actions == 5  # 3 branches + 2 within-branch


def test_bottleneck_solvable():
    random.seed(42)
    env = BottleneckPuzzle(n_branches=3, dead_end_depth=3, n_levels=1, episode_budget=40)
    state, actions = env.reset()
    # Navigate to winning branch
    wb = env._winning_branch
    state, actions, changed, level_up, game_over = env.step(wb)
    assert changed
    # Walk through branch
    for _ in range(env.dead_end_depth):
        state, actions, changed, level_up, game_over = env.step(env.n_branches)
        if level_up or game_over:
            break
    # Interact at bottleneck
    if not level_up:
        state, actions, changed, level_up, game_over = env.step(env.n_branches + 1)
    assert level_up, "Should solve after reaching bottleneck and interacting"


def test_hidden_pattern_solvable():
    random.seed(42)
    env = HiddenPatternPuzzle(n_steps=3, n_levels=1, n_actions=6, episode_budget=30)
    state, actions = env.reset()
    seq = env.sequences[0]
    for color in seq:
        correct_action = env.color_to_action[color]
        state, actions, changed, level_up, game_over = env.step(correct_action)
        assert changed
    assert level_up


def test_large_state_reachable():
    random.seed(42)
    env = LargeStateSpace(grid_size=4, n_levels=1, episode_budget=50)
    state, actions = env.reset()
    assert state == "L0_0_0"
    # Navigate to (3,3) via right then down
    for _ in range(3):
        env.step(3)  # right
    for _ in range(3):
        env.step(1)  # down
    assert env.current_level == 1  # should have reached goal


def test_get_all_environments():
    envs = get_all_environments(seed=42)
    assert len(envs) == 5
    names = [e.name for e in envs]
    assert "LinearPuzzle" in names
    assert "MazeNavigation" in names


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"  PASS {name}")
            except AssertionError as e:
                print(f"  FAIL {name}: {e}")
            except Exception as e:
                import traceback
                print(f"  ERROR {name}: {e}")
                traceback.print_exc()
    print("Done.")
