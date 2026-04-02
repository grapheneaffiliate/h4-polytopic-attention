"""Fast BFS solver for ARC-AGI-3 games.

Uses deepcopy for correctness + fast tuple state_key for visited set.
Handles all game types generically.
"""

import copy
import json
import os
import sys
import time
from collections import deque
from typing import Optional

from local_runner import (
    action_to_dict,
    create_game,
    find_game_metadata,
    get_valid_actions,
    list_all_games,
    load_game_class,
    step_game,
)
from arcengine import ActionInput, GameAction, GameState


def state_key(game) -> tuple:
    """State key using rendered frame hash — guaranteed correct for all games."""
    import hashlib
    frame = game.camera.render(game.current_level.get_sprites())
    h = hashlib.blake2b(frame.tobytes(), digest_size=16).digest()
    return (game._score, game._current_level_index, h)


def enumerate_actions(game) -> list:
    """Get all meaningful actions for BFS.
    For keyboard actions: use _get_valid_actions.
    For click actions: use sys_click sprites + segment centroids."""
    from arcengine import ActionInput, GameAction
    import numpy as np

    actions = []

    # Add simple keyboard actions
    for act_id in game._available_actions:
        if act_id in (1, 2, 3, 4, 5, 7):
            actions.append(ActionInput(id=GameAction.from_id(act_id)))

    # For click (action 6): get sys_click targets + grid
    if 6 in game._available_actions:
        # First try sys_click sprites
        sys_actions = game._get_valid_clickable_actions()
        if len(sys_actions) > 2:
            # sys_click covers enough targets
            actions.extend(sys_actions)
        else:
            # Need grid clicks — use step=2 for fine coverage
            frame = game.camera.render(game.current_level.get_sprites())
            # Find non-background regions
            bg = int(np.median(frame[0, :]))  # top row median = background
            step = 4
            for y in range(0, 64, step):
                for x in range(0, 64, step):
                    if frame[y, x] != bg or True:  # click all grid points
                        actions.append(ActionInput(
                            id=GameAction.ACTION6,
                            data={"x": x, "y": y}
                        ))

    return actions


def solve_level(game, max_states=500_000, max_time=300, verbose=True):
    """BFS to solve current level using deepcopy + fast state_key."""
    initial_score = game._score
    initial_level = game._current_level_index
    t0 = time.time()

    visited = {state_key(game)}
    queue = deque([(copy.deepcopy(game), [])])
    explored = 0

    while queue:
        elapsed = time.time() - t0
        if explored >= max_states or elapsed > max_time:
            break

        g, history = queue.popleft()
        actions = enumerate_actions(g)

        for action in actions:
            clone = copy.deepcopy(g)
            step_game(clone, action)
            explored += 1

            if clone._score > initial_score or clone._state == GameState.WIN:
                elapsed = time.time() - t0
                if verbose:
                    print(f"    SOLVED level {initial_level}: {len(history)+1} actions, "
                          f"{explored} explored, {len(visited)} unique, {elapsed:.1f}s")
                return history + [action], clone

            if clone._state == GameState.GAME_OVER:
                continue

            key = state_key(clone)
            if key not in visited:
                visited.add(key)
                queue.append((clone, history + [action]))

        if verbose and explored % 5000 == 0 and explored > 0:
            elapsed = time.time() - t0
            rate = explored / max(elapsed, 0.001)
            print(f"    ... {explored} explored, {len(visited)} unique, "
                  f"depth~{len(history)}, queue={len(queue)}, "
                  f"{rate:.0f}/s, {elapsed:.1f}s")

    elapsed = time.time() - t0
    if verbose:
        rate = explored / max(elapsed, 0.001)
        print(f"    FAILED level {initial_level}: {explored} explored, "
              f"{len(visited)} unique, {rate:.0f}/s, {elapsed:.1f}s")
    return None


def solve_game(game_id, max_states_per_level=500_000, max_time_per_level=300, verbose=True):
    """Solve all levels of a game."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Solving {game_id}")
        print(f"{'='*60}")

    cls = load_game_class(game_id)
    game = create_game(cls)
    meta = find_game_metadata(game_id)
    baselines = meta.get("baseline_actions", [])
    total_levels = game.win_score

    if verbose:
        print(f"  Levels: {total_levels}, Baselines: {baselines}")
        print(f"  Actions: {game._available_actions}")
        actions = get_valid_actions(game)
        print(f"  Valid actions at start: {len(actions)}")

    solution = {
        "game_id": meta.get("game_id", game_id),
        "total_levels": total_levels,
        "levels": [],
    }

    total_start = time.time()

    for level_idx in range(total_levels):
        baseline = baselines[level_idx] if level_idx < len(baselines) else "?"
        if verbose:
            print(f"\n  Level {level_idx} (baseline: {baseline} actions)")

        result = solve_level(
            game,
            max_states=max_states_per_level,
            max_time=max_time_per_level,
            verbose=verbose,
        )

        if result is None:
            solution["levels"].append({"level": level_idx, "solved": False})
            break

        level_actions, game = result

        solution["levels"].append({
            "level": level_idx,
            "solved": True,
            "num_actions": len(level_actions),
            "baseline": baseline,
            "actions": [action_to_dict(a) for a in level_actions],
        })

        if game._state == GameState.WIN:
            if verbose:
                print(f"\n  GAME WON! All {total_levels} levels solved!")
            break

    total_time = time.time() - total_start
    solved_levels = sum(1 for l in solution["levels"] if l.get("solved"))
    solution["solved_levels"] = solved_levels
    solution["total_time"] = round(total_time, 1)

    if verbose:
        print(f"\n  Result: {solved_levels}/{total_levels} in {total_time:.1f}s")

    return solution


def solve_all(game_ids=None, max_states=500_000, max_time=300, output_dir=None):
    """Solve all games."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solutions")
    os.makedirs(output_dir, exist_ok=True)

    if game_ids is None:
        game_ids = list_all_games()

    grand_total = 0
    grand_possible = 0

    for game_id in game_ids:
        try:
            sol = solve_game(game_id, max_states_per_level=max_states,
                           max_time_per_level=max_time)

            with open(os.path.join(output_dir, f"{game_id}.json"), "w") as f:
                json.dump(sol, f, indent=2)

            grand_total += sol["solved_levels"]
            grand_possible += sol["total_levels"]
        except Exception as e:
            import traceback
            print(f"\n  ERROR on {game_id}: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"GRAND TOTAL: {grand_total}/{grand_possible} levels")
    print(f"{'='*60}")


if __name__ == "__main__":
    game_ids = sys.argv[1:] if len(sys.argv) > 1 else None
    solve_all(game_ids=game_ids)
