"""
Agent Zero benchmark: BFS vs CGE Breakthrough vs Agent Zero on all 14 environments.
"""
import random
import sys
import os

from .environments import get_all_environments
from .core import AgentZero
from .baseline import BFSAgent


def run_agent(agent, env, max_actions=10000):
    state, actions = env.reset()
    total = 0
    while total < max_actions:
        if env.current_level >= env.total_levels:
            break
        action = agent.choose_action(state, actions)
        if action is None:
            action = random.choice(list(actions))
        prev = state
        state, actions, reward, level_up, done = env.step(action)
        total += 1
        agent.observe(prev, action, state, actions, reward > 0)
        if level_up:
            agent.on_level_complete(env.current_level - 1)
        if done:
            if env.current_level >= env.total_levels:
                break
            agent.on_episode_reset()
            state, actions = env.reset()
    return {"levels": env.current_level, "total": env.total_levels, "actions": total}


def run_benchmark(n_seeds=20, max_actions=10000):
    agents = {
        "BFS": lambda: BFSAgent(),
        "Zero": lambda: AgentZero(),
    }

    # Try importing CGE breakthrough for comparison
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from cge.agent_breakthrough import BreakthroughAgent
        from cge.agent import BFSAgent as CGE_BFS

        class CGEAdapter:
            """Adapter: CGE BreakthroughAgent → agent_zero interface."""
            def __init__(self):
                self._inner = BreakthroughAgent()
            def choose_action(self, state, actions):
                self._inner.on_new_state(state, actions)
                return self._inner.choose_action(state, actions)
            def observe(self, prev, action, new_state, new_actions, changed):
                self._inner.observe_result(prev, action, new_state, new_actions, changed)
            def on_level_complete(self, level):
                self._inner.on_level_complete(level)
            def on_episode_reset(self):
                self._inner.on_episode_reset()

        agents["CGE"] = lambda: CGEAdapter()
    except ImportError:
        pass

    sample = get_all_environments(0)
    env_names = [e.name for e in sample]
    results = {a: {e: {"levels": 0, "actions": 0} for e in env_names} for a in agents}

    print("=" * 80)
    print(f"AGENT ZERO BENCHMARK ({n_seeds} seeds, {max_actions} budget, {len(env_names)} environments)")
    print("=" * 80)

    for seed in range(n_seeds):
        for ei in range(len(env_names)):
            for aname, afn in agents.items():
                envs = get_all_environments(seed)
                env = envs[ei]
                random.seed(seed * 10000 + ei * 100 + 1)
                agent = afn()
                r = run_agent(agent, env, max_actions)
                results[aname][env.name]["levels"] += r["levels"]
                results[aname][env.name]["actions"] += r["actions"]

    # Print table
    agent_names = list(agents.keys())
    header = f"{'Environment':<22}"
    for a in agent_names:
        header += f" {a+' lvl':>10} {a+' act':>10}"
    header += f" {'Winner':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    totals = {a: {"levels": 0, "actions": 0} for a in agent_names}
    for ename in env_names:
        row = f"{ename:<22}"
        max_lvl = 0
        for a in agent_names:
            r = results[a][ename]
            lvl = r["levels"]
            act = r["actions"] // n_seeds
            totals[a]["levels"] += lvl
            totals[a]["actions"] += r["actions"]
            max_lvl = max(max_lvl, lvl)
            row += f" {lvl:>7}/{n_seeds*4:<3} {act:>7}/run"
        winners = [a for a in agent_names if results[a][ename]["levels"] == max_lvl]
        winner = winners[0] if len(winners) == 1 else "tie"
        row += f" {winner:>8}"
        print(row)

    print("-" * len(header))
    row = f"{'TOTAL':<22}"
    for a in agent_names:
        row += f" {totals[a]['levels']:>10} {totals[a]['actions']//n_seeds:>10}"
    max_t = max(totals[a]["levels"] for a in agent_names)
    tw = [a for a in agent_names if totals[a]["levels"] == max_t]
    row += f" {tw[0] if len(tw)==1 else 'tie':>8}"
    print(row)


if __name__ == "__main__":
    run_benchmark()
