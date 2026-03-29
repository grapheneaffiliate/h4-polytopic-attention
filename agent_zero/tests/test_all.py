"""Tests for Agent Zero."""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from agent_zero.core import AgentZero
from agent_zero.baseline import BFSAgent
from agent_zero.reasoner import HeuristicReasoner, LLMReasoner
from agent_zero.environments import (
    get_all_environments, LinearPuzzle, MazeNavigation, DeepTreeSearch,
    CausalChain, TextReasoning, CodeEnv, FeatureTransfer, CompositeEnv,
)


def _run(agent, env, max_actions=5000):
    state, actions = env.reset()
    for _ in range(max_actions):
        if env.current_level >= env.total_levels: break
        action = agent.choose_action(state, actions)
        if action is None: action = random.choice(list(actions))
        prev = state
        state, actions, reward, level_up, done = env.step(action)
        agent.observe(prev, action, state, actions, reward > 0)
        if level_up: agent.on_level_complete(env.current_level - 1)
        if done:
            if env.current_level >= env.total_levels: break
            agent.on_episode_reset()
            state, actions = env.reset()
    return env.current_level


# ── Environment tests ─────────────────────────────────────

def test_all_envs_created():
    envs = get_all_environments(42)
    assert len(envs) == 14
    names = [e.name for e in envs]
    assert "LinearPuzzle" in names
    assert "DeepTreeSearch" in names
    assert "TextReasoning" in names
    assert "CodeEnv" in names
    assert "CompositeEnv" in names

def test_linear_solvable():
    random.seed(42)
    env = LinearPuzzle(n_levels=1, sol_len=3, n_actions=6, budget=20)
    state, actions = env.reset()
    for a in env._solutions[0]:
        state, actions, reward, lu, done = env.step(a)
    assert lu

def test_code_env_solvable():
    random.seed(42)
    env = CodeEnv(length=3, n_values=3, n_levels=1, budget=20)
    state, actions = env.reset()
    target = env._targets[0]
    for i, v in enumerate(target):
        state, actions, reward, lu, done = env.step(i * 3 + v)
        if lu or done: break
    assert env.current_level >= 1

def test_text_reasoning_solvable():
    random.seed(42)
    env = TextReasoning(n_actions=10, n_steps=3, n_levels=1, budget=20)
    state, actions = env.reset()
    for pos in range(3):
        correct = (pos * env._mult + env._off + 0) % 10
        env.step(correct)
    assert env.current_level == 1

def test_composite_solvable():
    random.seed(42)
    env = CompositeEnv()
    # First sub-env is LargeStateSpace(5), reach (4,4)
    state, actions = env.reset()
    for _ in range(4): env.step(3)  # right
    for _ in range(4): env.step(1)  # down
    assert env.current_level >= 1

# ── Agent tests ───────────────────────────────────────────

def test_agent_zero_solves_linear():
    random.seed(42)
    env = LinearPuzzle(n_levels=1, sol_len=3, n_actions=6, budget=20)
    agent = AgentZero()
    assert _run(agent, env, 3000) >= 1

def test_agent_zero_solves_maze():
    random.seed(42)
    env = MazeNavigation(w=5, h=5, n_levels=1, budget=60)
    agent = AgentZero()
    assert _run(agent, env, 5000) >= 1

def test_agent_zero_solves_code_env():
    random.seed(42)
    env = CodeEnv(length=3, n_values=3, n_levels=1, budget=30)
    agent = AgentZero()
    assert _run(agent, env, 3000) >= 1

def test_bfs_solves_linear():
    random.seed(42)
    env = LinearPuzzle(n_levels=1, sol_len=3, n_actions=6, budget=20)
    agent = BFSAgent()
    assert _run(agent, env, 3000) >= 1

# ── Reasoner tests ────────────────────────────────────────

def test_heuristic_reasoner():
    r = HeuristicReasoner()
    ctx = {
        "state": "test", "available_actions": [0,1,2,3],
        "states_explored": 10, "level": 0, "actions_since_progress": 100,
        "top_actions": [(0, 5, 10), (1, 2, 10)], "total_actions": 100,
    }
    s = r.suggest(ctx)
    assert isinstance(s, dict)
    assert 0 in s  # action 0 has high efficacy, should be suggested

def test_llm_reasoner():
    # Mock LLM
    def mock_llm(prompt):
        return "0: 3.0\n1: 1.0\n"
    r = LLMReasoner(mock_llm)
    ctx = {"state": "test", "available_actions": [0,1,2],
           "states_explored": 5, "level": 0, "actions_since_progress": 50,
           "top_actions": [], "total_actions": 50}
    s = r.suggest(ctx)
    assert s.get(0) == 3.0
    assert s.get(1) == 1.0

def test_llm_reasoner_error_handling():
    def bad_llm(prompt):
        raise RuntimeError("LLM down")
    r = LLMReasoner(bad_llm)
    s = r.suggest({"state":"x","available_actions":[0],"states_explored":0,
                   "level":0,"actions_since_progress":0,"top_actions":[],"total_actions":0})
    assert s == {}

# ── ARC Bridge tests ──────────────────────────────────────

def test_arc_bridge_loads():
    import os
    if not os.path.exists("data/arc1"):
        return  # skip if no data
    from agent_zero.arc_bridge import get_arc_environments
    envs = get_arc_environments(n_tasks=3, data_dir="data")
    assert len(envs) > 0

def test_arc_env_step():
    import os
    if not os.path.exists("data/arc1"):
        return
    from agent_zero.arc_bridge import get_arc_environments
    envs = get_arc_environments(n_tasks=1, data_dir="data")
    if not envs: return
    env = envs[0]
    state, actions = env.reset()
    assert len(actions) > 0
    state2, actions2, reward, lu, done = env.step(list(actions)[0])
    assert isinstance(state2, str)

def test_arc_grid_accuracy():
    from agent_zero.arc_bridge import grid_accuracy
    assert grid_accuracy([[1,2],[3,4]], [[1,2],[3,4]]) == 1.0
    assert grid_accuracy([[0,0],[0,0]], [[1,1],[1,1]]) == 0.0
    assert grid_accuracy([[1,0],[0,1]], [[1,1],[1,1]]) == 0.5

def test_heuristic_arc_reasoner():
    from agent_zero.llm_reasoner import HeuristicARCReasoner
    task = {"train": [{"input": [[0,1],[1,0]], "output": [[1,0],[0,1]]}],
            "test": [{"input": [[0,1],[1,0]]}]}
    r = HeuristicARCReasoner(task)
    ctx = {"state": "test", "available_actions": list(range(40)),
           "states_explored": 5, "level": 0, "actions_since_progress": 50,
           "top_actions": [], "total_actions": 50}
    s = r.suggest(ctx)
    assert isinstance(s, dict)

def test_dry_run_reasoner():
    from agent_zero.llm_reasoner import DryRunReasoner
    r = DryRunReasoner(verbose=False)
    s = r.suggest({"state": "x", "available_actions": [0,1],
                   "states_explored": 0, "level": 0,
                   "actions_since_progress": 0, "top_actions": [],
                   "total_actions": 0})
    assert s == {}
    assert r.call_count == 1


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"  PASS {name}")
            except AssertionError as e:
                print(f"  FAIL {name}: {e}")
            except Exception as e:
                print(f"  ERROR {name}: {e}")
                import traceback; traceback.print_exc()
    print("Done.")
