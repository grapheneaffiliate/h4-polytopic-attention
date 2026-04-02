"""Execute pre-computed solutions via ARC-AGI-3 API.

Replays action sequences from solution JSON files.
Run on GitHub Actions (API blocked from local dev).
"""

import json
import os
import sys
import time
import glob

# This script runs on GitHub Actions where arc_agi is available
try:
    from arc_agi import Arcade, GameAction
except ImportError:
    print("arc_agi not installed - this script runs on GitHub Actions")
    sys.exit(1)


def load_solution(game_id, solutions_dir="solutions"):
    """Load a pre-computed solution."""
    path = os.path.join(solutions_dir, f"{game_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def execute_solution(arcade, solution, verbose=True):
    """Execute a solution via the API."""
    game_id_full = solution["game_id"]
    game_id = game_id_full.split("-")[0] if "-" in game_id_full else game_id_full

    if verbose:
        print(f"\nExecuting {game_id} ({solution.get('solved_levels', 0)}/{solution.get('total_levels', '?')} levels)")

    env = arcade.load(game_id)
    obs = env.reset()

    if verbose:
        print(f"  Initial: state={obs.state}, levels={obs.levels_completed}/{obs.win_levels}")

    total_actions = 0
    levels_completed = 0

    for level_data in solution.get("levels", []):
        if not level_data.get("solved"):
            break

        level_idx = level_data["level"]
        actions = level_data["actions"]

        if verbose:
            print(f"  Level {level_idx}: replaying {len(actions)} actions...")

        for action_dict in actions:
            action_id = action_dict["id"]
            data = action_dict.get("data", {})

            game_action = GameAction(action_id)
            obs = env.step(game_action, data=data if data else None)
            total_actions += 1

            if obs.state == "WIN":
                levels_completed = obs.levels_completed
                if verbose:
                    print(f"  WIN! {levels_completed} levels completed")
                break

            if obs.state == "GAME_OVER":
                if verbose:
                    print(f"  GAME OVER at action {total_actions}")
                # Reset and retry?
                obs = env.reset()
                break

        if obs.state == "WIN":
            break

        levels_completed = obs.levels_completed
        if verbose:
            print(f"  After level {level_idx}: {levels_completed} levels completed")

    if verbose:
        print(f"  Final: {levels_completed}/{solution.get('total_levels', '?')} levels, {total_actions} actions")

    return levels_completed


def main():
    api_key = os.environ.get("ARC_API_KEY", "58b421be-5980-4ee8-8e57-0f18dc9369f3")
    arcade = Arcade(api_key=api_key)

    solutions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solutions")

    # Find all solution files
    solution_files = sorted(glob.glob(os.path.join(solutions_dir, "*.json")))
    solution_files = [f for f in solution_files if not os.path.basename(f).startswith("_")]

    if not solution_files:
        print("No solutions found!")
        return

    grand_total = 0
    grand_possible = 0

    for sol_file in solution_files:
        game_id = os.path.splitext(os.path.basename(sol_file))[0]
        with open(sol_file) as f:
            solution = json.load(f)

        if not solution.get("levels"):
            continue

        try:
            levels = execute_solution(arcade, solution)
            grand_total += levels
            grand_possible += solution.get("total_levels", 0)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"GRAND TOTAL: {grand_total}/{grand_possible} levels")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
