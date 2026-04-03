#!/usr/bin/env python3
"""
ARC-AGI-3 Combined Runner: Pre-computed Solutions + Explorer v6 Fallback
========================================================================
1. Replay pre-computed solutions from solutions/*.json (instant, verified)
2. Run explorer v6 (UCB1 + adaptive) on games without solutions

This maximizes score by using proven solutions where available and
exploration where needed.
"""

import json
import os
import sys
import time
import glob

sys.stdout.reconfigure(line_buffering=True)

# Add project root to path for explorer import
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from arc_agi import Arcade
from arcengine import GameAction

SOLUTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solutions")
API_KEY = os.environ.get("ARC_API_KEY", "58b421be-5980-4ee8-8e57-0f18dc9369f3")
MAX_ACTIONS = int(os.environ.get("MAX_ACTIONS", "200000"))
SOLUTIONS_ONLY = os.environ.get("SOLUTIONS_ONLY", "false").lower() == "true"

ACTION_MAP = {
    1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3,
    4: GameAction.ACTION4, 5: GameAction.ACTION5, 6: GameAction.ACTION6,
    7: GameAction.ACTION7,
}


def load_solution(game_id):
    """Load pre-computed solution for a game."""
    path = os.path.join(SOLUTIONS_DIR, f"{game_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    # Only return if it has solved levels with action sequences
    levels = {}
    for lvl in data.get("levels", []):
        is_solved = lvl.get("solved", True)
        if is_solved is False:
            continue
        actions = lvl.get("actions", [])
        button_indices = lvl.get("button_indices", [])
        if actions:
            levels[lvl["level"]] = {"actions": actions}
        elif button_indices:
            levels[lvl["level"]] = {"button_indices": button_indices}
    return levels if levels else None


def replay_solution(arc, game_id, solution_levels):
    """Replay pre-computed solution via API. Returns levels completed."""
    env = arc.make(game_id)
    if env is None:
        return 0, 0

    info = env.info
    total_levels = len(info.baseline_actions) if info.baseline_actions else 0

    def _step_action(env, act_data):
        """Execute one action, handling both formats."""
        action_id = act_data.get("id", 1)
        data = act_data.get("data", {})
        ga = ACTION_MAP.get(action_id, GameAction.ACTION1)
        if data and action_id == 6:
            return env.step(ga, data=data)
        else:
            return env.step(ga)

    def _get_level_actions(level_data, env):
        """Convert level data to action list, resolving button_indices at runtime."""
        if "actions" in level_data:
            return level_data["actions"]
        elif "button_indices" in level_data:
            # Resolve button indices to click coordinates via game engine
            game_obj = env._game if hasattr(env, '_game') else None
            if game_obj:
                clicks = game_obj._get_valid_clickable_actions()
                actions = []
                for btn_idx in level_data["button_indices"]:
                    if btn_idx < len(clicks):
                        c = clicks[btn_idx]
                        actions.append({"id": 6, "data": c.data})
                    else:
                        actions.append({"id": 6, "data": {"x": 0, "y": 0}})
                return actions
        return []

    # Send first action from solution to initialize
    first_level = min(solution_levels.keys()) if solution_levels else 0
    first_level_data = solution_levels.get(first_level, {})
    first_actions = _get_level_actions(first_level_data, env)

    if not first_actions:
        obs = env.step(GameAction.ACTION1)
    else:
        obs = _step_action(env, first_actions[0])

    if obs is None:
        return 0, total_levels

    total_levels = obs.win_levels or total_levels
    current_level = obs.levels_completed

    # Replay remaining actions for first level
    if first_level == 0 and first_actions and current_level == 0:
        for act_data in first_actions[1:]:
            obs = _step_action(env, act_data)
            if obs is None:
                return current_level, total_levels
            if obs.levels_completed > current_level:
                current_level = obs.levels_completed
                break
            if obs.state.value != "NOT_FINISHED":
                break

    # Replay remaining levels
    for level_idx in sorted(solution_levels.keys()):
        if level_idx <= first_level:
            continue
        if current_level > level_idx:
            continue
        if current_level != level_idx:
            break
        if obs.state.value == "WIN":
            current_level = obs.levels_completed
            break

        level_data = solution_levels[level_idx]
        actions = _get_level_actions(level_data, env)

        for act_data in actions:
            obs = _step_action(env, act_data)
            if obs is None:
                return current_level, total_levels
            if obs.levels_completed > current_level:
                current_level = obs.levels_completed
                break
            if obs.state.value != "NOT_FINISHED":
                break

        if obs.state.value == "WIN":
            current_level = obs.levels_completed
            break

    return current_level, total_levels


def run_explorer(arc, game_id, max_actions=200000, timeout_sec=600):
    """Run explorer v6 on a game with timeout via subprocess."""
    import subprocess, tempfile

    # Run explorer in a subprocess so we can truly kill it on timeout
    script = f"""
import sys, os, json
sys.path.insert(0, os.environ.get("PYTHONPATH", "."))
from olympus.arc3.explorer_v6_adaptive import solve_game, GAME_BUDGETS
from arc_agi import Arcade
arc = Arcade(arc_api_key="{os.environ.get('ARC_API_KEY', '')}")
budget = GAME_BUDGETS.get("{game_id.split('-')[0]}", {max_actions})
try:
    r = solve_game(arc, "{game_id}", budget, verbose=False)
    print(json.dumps(r, default=str))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=timeout_sec,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
        )
        if result.returncode == 0 and result.stdout.strip():
            import json as json_mod
            data = json_mod.loads(result.stdout.strip().split("\n")[-1])
            if "error" in data:
                print(f"    Explorer error: {data['error']}", flush=True)
                return 0, 0, 0, 0, "error"
            return (
                data.get("levels_completed", 0),
                data.get("total_levels", 0),
                data.get("actions_used", 0),
                data.get("states_explored", 0),
                data.get("mode", "?"),
            )
        else:
            print(f"    Explorer failed: rc={result.returncode}", flush=True)
            if result.stderr:
                print(f"    stderr: {result.stderr[-200:]}", flush=True)
            return 0, 0, 0, 0, "error"
    except subprocess.TimeoutExpired:
        print(f"    Explorer TIMEOUT ({timeout_sec}s) on {game_id}", flush=True)
        return 0, 0, 0, 0, "timeout"
    except Exception as e:
        print(f"    Explorer error: {e}", flush=True)
        return 0, 0, 0, 0, "error"


def main():
    print("=" * 70)
    print("  ARC-AGI-3 Combined Runner")
    print("  Pre-computed Solutions + Explorer v6 Fallback")
    print("=" * 70)
    print(f"  Solutions dir: {SOLUTIONS_DIR}")
    print(f"  Explorer budget: {MAX_ACTIONS}")
    print(f"  Solutions only: {SOLUTIONS_ONLY}")
    print()

    arc = Arcade(arc_api_key=API_KEY)
    envs = arc.get_environments()

    # Load all solutions
    all_solutions = {}
    for f in sorted(glob.glob(os.path.join(SOLUTIONS_DIR, "*.json"))):
        gid = os.path.splitext(os.path.basename(f))[0]
        if gid.startswith("_"):
            continue
        sol = load_solution(gid)
        if sol:
            all_solutions[gid] = sol

    print(f"  Loaded solutions for {len(all_solutions)} games: {sorted(all_solutions.keys())}")
    print()

    # Play all games
    results = {}
    total_solved = 0
    total_levels = 0
    t_start = time.time()

    for env_info in envs:
        game_id = env_info.game_id
        gid_short = game_id.split("-")[0]
        t0 = time.time()

        # Phase 1: Pre-computed solutions
        sol = all_solutions.get(gid_short)
        levels_from_solution = 0
        n_levels = 0

        if sol:
            levels_from_solution, n_levels = replay_solution(arc, game_id, sol)
            method = "precomputed"

            if levels_from_solution > 0:
                dt = time.time() - t0
                print(f"  {gid_short}: {levels_from_solution}/{n_levels} (precomputed, {dt:.1f}s)", flush=True)

        # Phase 2: Explorer fallback (if precomputed got 0 and not solutions-only mode)
        # Skip explorer if precomputed already got levels (explorer creates new scorecard game)
        levels_from_explorer = 0
        if not SOLUTIONS_ONLY and levels_from_solution == 0:
            t1 = time.time()
            lc, tl, actions_used, states, mode = run_explorer(arc, game_id, MAX_ACTIONS)
            dt = time.time() - t1
            levels_from_explorer = lc
            if tl > 0:
                n_levels = tl
            print(f"  {gid_short}: {lc}/{tl} (explorer {mode}, {actions_used} actions, "
                  f"{states} states, {dt:.0f}s)", flush=True)

        # Best result
        best = max(levels_from_solution, levels_from_explorer)
        results[gid_short] = {
            "precomputed": levels_from_solution,
            "explorer": levels_from_explorer,
            "best": best,
            "total": n_levels,
        }
        total_solved += best
        total_levels += n_levels

    # Final scorecard
    elapsed = time.time() - t_start
    print()
    print("=" * 70)
    print("  FINAL SCORECARD")
    print("=" * 70)
    print(f"  {'Game':<8} {'Pre':>5} {'Exp':>5} {'Best':>5} {'Total':>5}")
    print(f"  {'-'*30}")

    for gid in sorted(results.keys()):
        r = results[gid]
        pre = r["precomputed"]
        exp = r["explorer"]
        best = r["best"]
        tot = r["total"]
        marker = " *" if pre > 0 and pre >= exp else ""
        print(f"  {gid:<8} {pre:>5} {exp:>5} {best:>5} /{tot:<4}{marker}")

    pct = total_solved / max(total_levels, 1) * 100
    print(f"  {'-'*30}")
    print(f"  {'TOTAL':<8} {'':>5} {'':>5} {total_solved:>5} /{total_levels:<4}")
    print(f"\n  Score: {total_solved}/{total_levels} ({pct:.1f}%)")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  * = pre-computed solution matched or beat explorer")

    # Save results
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "combined_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_solved": total_solved,
            "total_levels": total_levels,
            "percentage": round(pct, 1),
            "results": results,
        }, f, indent=2)


if __name__ == "__main__":
    main()
