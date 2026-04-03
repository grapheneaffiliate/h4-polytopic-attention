#!/usr/bin/env python3
"""
ARC-AGI-3 Unified Game Runner
==============================
Plays ALL 25 games via the ARC-AGI API, combining:
  1. Pre-computed solutions (instant replay from JSON files)
  2. Explorer fallback (random exploration for unsolved games)

Usage:
  python play_all.py                    # Play all 25 games
  python play_all.py ft09 lp85          # Play specific games
  python play_all.py --no-explorer      # Pre-computed only
  python play_all.py --budget 200       # Explorer step budget per level

Requires:
  pip install arc-agi
  Solution files in solutions/ directory (same folder as this script)
"""

import arc_agi
from arcengine import GameAction, ActionInput
import json
import os
import sys
import time
import random
from pathlib import Path
from collections import defaultdict

SOLUTIONS_DIR = Path(__file__).parent / "solutions"
EXPLORER_BUDGET = 300
USE_EXPLORER = True

ACTION_MAP = {
    1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3,
    4: GameAction.ACTION4, 5: GameAction.ACTION5, 6: GameAction.ACTION6,
    7: GameAction.ACTION7,
}


def load_solutions(game_id):
    """Load pre-computed solutions from JSON file."""
    path = SOLUTIONS_DIR / f"{game_id}.json"
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    levels = {}
    for lvl in data.get("levels", []):
        lvl_idx = lvl.get("level", 0)
        if "solved" in lvl and not lvl["solved"]:
            continue

        actions = lvl.get("actions", [])
        program = lvl.get("program", [])

        # Convert program format (TN36-style) to action format
        if not actions and program:
            # TN36: program is list of opcodes, need button toggles + play
            # For now skip program-only formats — need game-specific conversion
            continue

        if actions:
            levels[lvl_idx] = actions

    return levels if levels else None


def play_game(arc, game_id, precomputed=None, verbose=True):
    """Play a single game via API."""
    env = arc.make(game_id)
    if env is None:
        if verbose:
            print(f"  FAILED to load {game_id}")
        return 0, 0, 0, []

    info = env.info
    baseline = info.baseline_actions or []
    total_levels = len(baseline)

    # Initialize
    obs = env.step(GameAction.ACTION1)
    if obs is None:
        return 0, total_levels, 1, []

    total_levels = obs.win_levels or total_levels
    available = obs.available_actions or [1, 2, 3, 4]
    current_level = obs.levels_completed
    total_actions = 1
    method_log = []
    max_total = sum(baseline) * 5 if baseline else 2000

    while current_level < total_levels and total_actions < max_total:
        level_start = total_actions
        solved = False
        method = "none"

        # Method 1: Pre-computed solution
        if precomputed and current_level in precomputed:
            actions = precomputed[current_level]
            method = "precomputed"

            for act_data in actions:
                action_id = act_data.get("id", act_data.get("action", 1))
                data = act_data.get("data", {})
                ga = ACTION_MAP.get(action_id, GameAction.ACTION1)

                if data and action_id == 6:
                    # Click action with coordinates
                    obs = env.step(ga, data=data)
                else:
                    obs = env.step(ga)

                total_actions += 1

                if obs is None:
                    break
                if obs.levels_completed > current_level:
                    solved = True
                    break
                if obs.state.value != "NOT_FINISHED":
                    break

        # Method 2: Explorer fallback
        if not solved and USE_EXPLORER:
            method = "explorer"
            budget = min(EXPLORER_BUDGET, max_total - total_actions)

            for _ in range(budget):
                aid = random.choice(available)
                ga = ACTION_MAP.get(aid, GameAction.ACTION1)
                obs = env.step(ga)
                total_actions += 1

                if obs is None:
                    break
                if obs.levels_completed > current_level:
                    solved = True
                    break
                if obs.state.value != "NOT_FINISHED":
                    break

        if solved:
            lvl_actions = total_actions - level_start
            method_log.append((method, current_level, lvl_actions))
            current_level += 1
            if verbose:
                print(f"    Level {current_level-1}: solved ({method}, {lvl_actions} actions)")
        else:
            if verbose:
                print(f"    Level {current_level}: stuck")
            break

    return current_level, total_levels, total_actions, method_log


def main():
    args = sys.argv[1:]
    specific_games = []
    explorer_budget = EXPLORER_BUDGET

    i = 0
    while i < len(args):
        if args[i] == "--no-explorer":
            global USE_EXPLORER
            USE_EXPLORER = False
        elif args[i] == "--budget":
            explorer_budget = int(args[i + 1])
            i += 1
        elif not args[i].startswith("--"):
            specific_games.append(args[i].lower())
        i += 1

    print("=" * 65)
    print("  ARC-AGI-3 Unified Game Runner")
    print("=" * 65)

    arc = arc_agi.Arcade()
    envs = arc.get_environments()

    game_ids = specific_games if specific_games else [e.title.lower() for e in envs]

    print(f"  Games: {len(game_ids)}")
    print(f"  Solutions dir: {SOLUTIONS_DIR}")
    print()

    # Load pre-computed solutions
    all_solutions = {}
    for gid in game_ids:
        sol = load_solutions(gid)
        if sol:
            all_solutions[gid] = sol
            print(f"  Loaded {gid}: {len(sol)} levels")

    print()

    # Play all games
    results = {}
    total_solved = 0
    total_levels = 0

    for game_id in game_ids:
        print(f"--- {game_id.upper()} ---")

        precomputed = all_solutions.get(game_id)

        t0 = time.perf_counter()
        try:
            levels_done, n_levels, n_actions, log = play_game(
                arc, game_id, precomputed=precomputed, verbose=True,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            levels_done, n_levels, n_actions, log = 0, 0, 0, []

        dt = time.perf_counter() - t0

        results[game_id] = {
            "levels": levels_done,
            "total": n_levels,
            "actions": n_actions,
            "time": dt,
        }
        total_solved += levels_done
        total_levels += n_levels
        print(f"  Result: {levels_done}/{n_levels} in {dt:.1f}s")
        print()

    # Scorecard
    print("=" * 65)
    print("  SCORECARD")
    print("=" * 65)
    for gid in sorted(results.keys()):
        r = results[gid]
        print(f"  {gid:<10} {r['levels']:>3}/{r['total']:<3}")

    pct = total_solved / max(total_levels, 1) * 100
    print(f"\n  TOTAL: {total_solved}/{total_levels} ({pct:.1f}%)")

    # Save
    results_path = Path(__file__).parent / "run_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_solved": total_solved,
            "total_levels": total_levels,
            "results": results,
        }, f, indent=2)
    print(f"  Saved to {results_path}")


if __name__ == "__main__":
    main()
