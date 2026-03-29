"""Tests for CGEAgent and BFSAgent."""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from cge.agent import CGEAgent, BFSAgent
from cge.environments import LinearPuzzle, MazeNavigation, BottleneckPuzzle


def _run_agent_on_env(agent, env, max_actions=3000):
    """Run an agent on an environment and return levels solved."""
    state, actions = env.reset()
    agent.on_new_state(state, actions)
    for _ in range(max_actions):
        if env.current_level >= env.total_levels:
            break
        action = agent.choose_action(state, actions)
        if action is None:
            action = random.choice(list(actions))
        prev = state
        state, actions, changed, level_up, game_over = env.step(action)
        agent.observe_result(prev, action, state, actions, changed)
        if level_up:
            agent.on_level_complete(env.current_level - 1)
        if game_over:
            if env.current_level >= env.total_levels:
                break
            agent.on_episode_reset()
            state, actions = env.reset()
            agent.on_new_state(state, actions)
    return env.current_level


def test_cge_solves_linear():
    random.seed(42)
    env = LinearPuzzle(n_levels=1, solution_length=3, n_actions=6, episode_budget=20)
    agent = CGEAgent(analyze_interval=100, learn_threshold=50)
    solved = _run_agent_on_env(agent, env, max_actions=2000)
    assert solved >= 1, f"CGE should solve linear puzzle, got {solved}"


def test_bfs_solves_linear():
    random.seed(42)
    env = LinearPuzzle(n_levels=1, solution_length=3, n_actions=6, episode_budget=20)
    agent = BFSAgent()
    solved = _run_agent_on_env(agent, env, max_actions=2000)
    assert solved >= 1, f"BFS should solve linear puzzle, got {solved}"


def test_cge_solves_maze():
    random.seed(42)
    env = MazeNavigation(width=5, height=5, n_levels=1, episode_budget=60)
    agent = CGEAgent(analyze_interval=100, learn_threshold=50)
    solved = _run_agent_on_env(agent, env, max_actions=3000)
    assert solved >= 1, f"CGE should solve maze, got {solved}"


def test_cge_solves_bottleneck():
    random.seed(42)
    env = BottleneckPuzzle(n_branches=3, dead_end_depth=3, n_levels=1, episode_budget=40)
    agent = CGEAgent(analyze_interval=100, learn_threshold=50)
    solved = _run_agent_on_env(agent, env, max_actions=3000)
    assert solved >= 1, f"CGE should solve bottleneck, got {solved}"


def test_cge_uses_compression():
    random.seed(42)
    # Hard enough that it takes many actions, so compression kicks in
    env = LinearPuzzle(n_levels=3, solution_length=5, n_actions=10, episode_budget=30)
    agent = CGEAgent(analyze_interval=50, learn_threshold=30)
    _run_agent_on_env(agent, env, max_actions=5000)
    stats = agent.get_stats()
    # After enough actions, compression should be guiding some decisions
    assert stats["compression_guided"] > 0, f"CGE should use compression guidance, stats={stats}"


def test_cge_stats():
    random.seed(42)
    env = LinearPuzzle(n_levels=1, solution_length=3, n_actions=6, episode_budget=20)
    agent = CGEAgent(analyze_interval=50, learn_threshold=30)
    _run_agent_on_env(agent, env, max_actions=1000)
    stats = agent.get_stats()
    assert "total_actions" in stats
    assert "states_explored" in stats
    assert stats["total_actions"] > 0


def test_cge_summary():
    random.seed(42)
    env = MazeNavigation(width=4, height=4, n_levels=1, episode_budget=40)
    agent = CGEAgent(analyze_interval=50, learn_threshold=30)
    _run_agent_on_env(agent, env, max_actions=1000)
    summary = agent.get_summary()
    assert "CGEAgent" in summary


def test_episode_reset_and_replay():
    random.seed(42)
    env = LinearPuzzle(n_levels=2, solution_length=2, n_actions=4, episode_budget=10)
    agent = CGEAgent(analyze_interval=50, learn_threshold=20)
    solved = _run_agent_on_env(agent, env, max_actions=2000)
    # Should solve at least level 0 and replay it after reset
    assert solved >= 1


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
