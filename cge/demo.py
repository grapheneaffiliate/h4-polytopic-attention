"""
Demo: CGE vs BFS on all simulated environments.

Runs both agents on identical environments across multiple seeds and compares:
- Actions to solve (lower = better)
- Levels solved within budget
- Compression insights discovered

The key test: environments with many useless actions and tight budgets,
where learning which actions work gives a real advantage.
"""

import random
from .environments import (
    LinearPuzzle, MazeNavigation, BottleneckPuzzle,
    HiddenPatternPuzzle, LargeStateSpace,
)
from .agent import CGEAgent, BFSAgent


def make_hard_environments(seed=42):
    """Create harder environments where compression matters.

    Key: many useless actions + tight episode budgets.
    BFS wastes actions on useless ones; CGE learns to skip them.
    """
    random.seed(seed)
    return [
        # 5 correct actions out of 15, tight budget — must learn which 5 work
        LinearPuzzle(n_levels=4, solution_length=5, n_actions=15, episode_budget=25),
        # 4 useful arrows out of 10 actions, small maze but tight budget
        MazeNavigation(width=7, height=7, n_levels=3, episode_budget=50),
        # 7 branches but only 1 is right, deep dead ends
        BottleneckPuzzle(n_branches=7, dead_end_depth=5, n_levels=3, episode_budget=45),
        # Must learn color→action mapping, 8 actions but only 4 are ever correct
        HiddenPatternPuzzle(n_steps=8, n_levels=4, n_actions=8, episode_budget=35),
        # Large grid, 2 useless actions out of 6
        LargeStateSpace(grid_size=10, n_levels=1, episode_budget=120),
    ]


def run_agent(agent, env, max_actions=10000):
    """Run an agent on an environment. Returns results dict."""
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
        "states_explored": agent.explorer.num_states,
        "solved_all": env.current_level >= env.total_levels,
    }


def compare(n_seeds=10, max_actions=10000, verbose=True):
    """Run CGE vs BFS across multiple random seeds. Averages results."""

    if verbose:
        print("=" * 70)
        print("CGE vs BFS Comparison (averaged over multiple seeds)")
        print("=" * 70)

    env_names = ["LinearPuzzle", "MazeNavigation", "BottleneckPuzzle",
                 "HiddenPatternPuzzle", "LargeStateSpace"]

    # Accumulate per-environment results
    cge_totals = {n: {"levels": 0, "actions": 0, "solved": 0} for n in env_names}
    bfs_totals = {n: {"levels": 0, "actions": 0, "solved": 0} for n in env_names}

    for seed in range(n_seeds):
        envs_cge = make_hard_environments(seed)
        envs_bfs = make_hard_environments(seed)

        for env_cge, env_bfs in zip(envs_cge, envs_bfs):
            name = env_cge.name

            # Run CGE
            random.seed(seed * 1000 + 1)
            cge = CGEAgent(analyze_interval=100, learn_threshold=50)
            r_cge = run_agent(cge, env_cge, max_actions)

            # Run BFS with DIFFERENT random seed to avoid identical choices
            random.seed(seed * 1000 + 2)
            bfs = BFSAgent()
            r_bfs = run_agent(bfs, env_bfs, max_actions)

            cge_totals[name]["levels"] += r_cge["levels_solved"]
            cge_totals[name]["actions"] += r_cge["actions_used"]
            cge_totals[name]["solved"] += int(r_cge["solved_all"])

            bfs_totals[name]["levels"] += r_bfs["levels_solved"]
            bfs_totals[name]["actions"] += r_bfs["actions_used"]
            bfs_totals[name]["solved"] += int(r_bfs["solved_all"])

    if verbose:
        print(f"\nResults averaged over {n_seeds} seeds, {max_actions} action budget:\n")
        print(f"{'Environment':<22} {'CGE levels':>12} {'BFS levels':>12} {'CGE actions':>13} {'BFS actions':>13} {'Speedup':>8}")
        print("-" * 82)

        total_cge_levels = total_bfs_levels = 0
        total_cge_actions = total_bfs_actions = 0
        total_possible = 0

        for name in env_names:
            cl = cge_totals[name]["levels"]
            bl = bfs_totals[name]["levels"]
            ca = cge_totals[name]["actions"]
            ba = bfs_totals[name]["actions"]
            cs = cge_totals[name]["solved"]
            bs = bfs_totals[name]["solved"]

            total_cge_levels += cl
            total_bfs_levels += bl
            total_cge_actions += ca
            total_bfs_actions += ba

            speedup = ""
            if cs > 0 and bs > 0:
                # Compare avg actions for solved runs
                speedup = f"{ba/max(ca,1):.2f}x"
            elif cl > bl:
                speedup = "CGE wins"
            elif bl > cl:
                speedup = "BFS wins"
            else:
                speedup = "tie"

            print(f"{name:<22} {cl:>5}/{n_seeds*4:>3}     {bl:>5}/{n_seeds*4:>3}     {ca/n_seeds:>8.0f}/run   {ba/n_seeds:>8.0f}/run  {speedup:>8}")

        print("-" * 82)
        print(f"{'TOTAL':<22} {total_cge_levels:>5}         {total_bfs_levels:>5}         "
              f"{total_cge_actions/n_seeds:>8.0f}/run   {total_bfs_actions/n_seeds:>8.0f}/run")

        if total_cge_levels > total_bfs_levels:
            print(f"\n  CGE solved {total_cge_levels - total_bfs_levels} more levels than BFS!")
        elif total_bfs_levels > total_cge_levels:
            print(f"\n  BFS solved {total_bfs_levels - total_cge_levels} more levels than CGE.")
        else:
            print(f"\n  Same total levels. CGE used {total_cge_actions/max(total_bfs_actions,1):.1%} of BFS actions.")

    return cge_totals, bfs_totals


def single_run(seed=42, max_actions=10000, verbose=True):
    """Detailed single run showing compression learning in action."""
    if verbose:
        print("=" * 70)
        print(f"CGE Detailed Run (seed={seed})")
        print("=" * 70)

    envs = make_hard_environments(seed)

    for env in envs:
        random.seed(seed * 1000 + 1)
        cge = CGEAgent(analyze_interval=100, learn_threshold=50)
        r = run_agent(cge, env, max_actions)

        print(f"\n--- {env.name} ---")
        print(f"  Solved: {r['levels_solved']}/{r['total_levels']} in {r['actions_used']} actions")
        print(f"  {cge.get_summary()}")


if __name__ == "__main__":
    single_run(seed=42, max_actions=10000, verbose=True)
    print("\n")
    compare(n_seeds=10, max_actions=10000, verbose=True)
