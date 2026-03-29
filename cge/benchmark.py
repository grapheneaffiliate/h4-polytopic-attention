"""
Comprehensive benchmark: CGE v2 vs CGE v1 vs BFS on all environments.
"""

import random
from .environments import get_all_environments
from .environments_hard import get_hard_environments
from .agent import CGEAgent, BFSAgent
from .agent_best import CGEBest
from .agent_breakthrough import BreakthroughAgent


def run_agent(agent, env, max_actions=10000):
    """Run any agent on any environment."""
    state, actions = env.reset()
    agent.on_new_state(state, actions)
    actions_used = 0

    while actions_used < max_actions:
        if env.current_level >= env.total_levels:
            break

        action = agent.choose_action(state, actions)
        if action is None:
            action = random.choice(list(actions))

        prev_state = state
        state, actions, changed, level_up, game_over = env.step(action)
        actions_used += 1

        agent.observe_result(prev_state, action, state, actions, changed)

        if level_up:
            agent.on_level_complete(env.current_level - 1)

        if game_over:
            if env.current_level >= env.total_levels:
                break
            agent.on_episode_reset()
            state, actions = env.reset()
            agent.on_new_state(state, actions)

    return {
        "levels_solved": env.current_level,
        "total_levels": env.total_levels,
        "actions_used": actions_used,
        "states": agent.explorer.num_states,
        "solved_all": env.current_level >= env.total_levels,
    }


def run_benchmark(n_seeds=10, max_actions=10000):
    """Run full benchmark across all environments and agents."""
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARK: CGE v2 vs CGE v1 vs BFS")
    print(f"({n_seeds} seeds, {max_actions} action budget per environment)")
    print("=" * 80)

    agent_types = {
        "BFS": lambda: BFSAgent(),
        "Best": lambda: CGEBest(analyze_interval=100, learn_threshold=50),
        "UCB": lambda: BreakthroughAgent(analyze_interval=100, learn_threshold=50),
    }

    # Combine original + hard environments
    def make_envs(seed):
        e1 = get_all_environments(seed)
        e2 = get_hard_environments(seed)
        return e1 + e2

    # Collect env names from one run
    sample_envs = make_envs(0)
    env_names = [e.name for e in sample_envs]

    # Results: agent_name -> env_name -> {levels, actions, solved}
    results = {a: {e: {"levels": 0, "actions": 0, "solved": 0}
                   for e in env_names}
               for a in agent_types}

    for seed in range(n_seeds):
        for agent_name, agent_fn in agent_types.items():
            envs = make_envs(seed)
            random.seed(seed * 1000 + hash(agent_name) % 1000)

            for env in envs:
                agent = agent_fn()
                r = run_agent(agent, env, max_actions)
                results[agent_name][env.name]["levels"] += r["levels_solved"]
                results[agent_name][env.name]["actions"] += r["actions_used"]
                results[agent_name][env.name]["solved"] += int(r["solved_all"])

    # Print results table
    print(f"\n{'Environment':<22}", end="")
    for a in agent_types:
        print(f" {a+' lvl':>10} {a+' act':>10}", end="")
    print(f" {'Best':>8}")
    print("-" * (22 + len(agent_types) * 22 + 10))

    total = {a: {"levels": 0, "actions": 0} for a in agent_types}

    for env_name in env_names:
        print(f"{env_name:<22}", end="")
        env_results = {}
        for a in agent_types:
            r = results[a][env_name]
            lvl = r["levels"]
            act = r["actions"] // n_seeds
            total[a]["levels"] += lvl
            total[a]["actions"] += r["actions"]
            env_results[a] = (lvl, act)
            print(f" {lvl:>7}/{n_seeds*4:<3} {act:>7}/run", end="")

        # Determine winner
        max_lvl = max(v[0] for v in env_results.values())
        winners = [a for a, v in env_results.items() if v[0] == max_lvl]
        if len(winners) == 1:
            print(f" {winners[0]:>8}", end="")
        elif len(winners) < len(agent_types):
            # Tie among some — pick lowest actions
            min_act = min(env_results[a][1] for a in winners)
            winner = [a for a in winners if env_results[a][1] == min_act][0]
            print(f" {winner:>8}", end="")
        else:
            print(f" {'tie':>8}", end="")
        print()

    print("-" * (22 + len(agent_types) * 22 + 10))
    print(f"{'TOTAL':<22}", end="")
    for a in agent_types:
        lvl = total[a]["levels"]
        act = total[a]["actions"] // n_seeds
        print(f" {lvl:>10} {act:>10}", end="")

    # Overall winner
    max_total = max(total[a]["levels"] for a in agent_types)
    top_agents = [a for a in agent_types if total[a]["levels"] == max_total]
    if len(top_agents) == 1:
        print(f" {top_agents[0]:>8}")
    else:
        min_act = min(total[a]["actions"] for a in top_agents)
        winner = [a for a in top_agents if total[a]["actions"] == min_act][0]
        print(f" {winner:>8}")

    print()

    # Print detailed stats for UCB run
    print("\nBreakthrough (UCB) detailed analysis (last seed):")
    envs = make_envs(n_seeds - 1)
    random.seed((n_seeds - 1) * 1000 + hash("UCB") % 1000)
    for env in envs:
        agent = BreakthroughAgent(analyze_interval=100, learn_threshold=50)
        r = run_agent(agent, env, max_actions)
        print(f"\n  {env.name}: {r['levels_solved']}/{r['total_levels']} levels")
        print(f"  {agent.get_summary()}")


if __name__ == "__main__":
    run_benchmark(n_seeds=10, max_actions=10000)
