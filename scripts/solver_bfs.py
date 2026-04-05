"""BFS solver for ARC-AGI-3 games.

Solves games level-by-level using breadth-first search over the finite
action space provided by _get_valid_actions(). Runs entirely locally.

Uses replay-from-scratch instead of deepcopy to avoid 15ms/clone overhead.
"""

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
    hash_state,
    list_all_games,
    load_game_class,
    step_game,
)
from arcengine import ActionInput, GameAction, GameState


def replay_actions(game_class, all_actions: list):
    """Create a fresh game and replay a sequence of actions.
    Returns the game in the state after all actions."""
    game = create_game(game_class)
    for action in all_actions:
        step_game(game, action)
    return game


def solve_level_bfs(
    game_class,
    prefix_actions: list,
    max_states: int = 500_000,
    verbose: bool = True,
) -> Optional[tuple]:
    """BFS to solve the current level using replay-from-scratch.

    Args:
        game_class: The game class to instantiate
        prefix_actions: Actions already taken (to reach current level)
        max_states: Maximum states to explore
        verbose: Print progress

    Returns:
        (level_actions, full_actions) or None
    """
    # Build initial state by replaying prefix
    game = replay_actions(game_class, prefix_actions)
    initial_score = game._score
    initial_level = game._current_level_index
    initial_hash = hash_state(game)

    # BFS stores action histories, not game objects
    # frontier entries: list of actions FROM the level start (not including prefix)
    visited = {initial_hash}
    queue = deque([[]])  # start with empty action list

    explored = 0
    start_time = time.time()

    while queue:
        if explored >= max_states:
            if verbose:
                elapsed = time.time() - start_time
                print(f"    BFS limit: {explored} states, {len(visited)} unique, {elapsed:.1f}s")
            return None

        level_actions = queue.popleft()

        # Replay to reach this state
        game = replay_actions(game_class, prefix_actions + level_actions)

        # Get valid actions at this state
        actions = get_valid_actions(game)

        for action in actions:
            # Try this action on a fresh replay
            new_level_actions = level_actions + [action]
            game2 = replay_actions(game_class, prefix_actions + new_level_actions)
            explored += 1

            # Check for level completion
            if game2._score > initial_score or game2._state == GameState.WIN:
                elapsed = time.time() - start_time
                if verbose:
                    print(f"    SOLVED level {initial_level} in {len(new_level_actions)} actions, "
                          f"{explored} explored, {len(visited)} unique, {elapsed:.1f}s")
                return new_level_actions, prefix_actions + new_level_actions

            # Skip dead ends
            if game2._state == GameState.GAME_OVER:
                continue

            h = hash_state(game2)
            if h not in visited:
                visited.add(h)
                queue.append(new_level_actions)

        # Progress
        if verbose and explored % 5000 == 0:
            elapsed = time.time() - start_time
            depth = len(level_actions)
            print(f"    ... {explored} explored, {len(visited)} unique, "
                  f"depth={depth}, queue={len(queue)}, {elapsed:.1f}s")

    if verbose:
        elapsed = time.time() - start_time
        print(f"    BFS exhausted: {explored} explored, {len(visited)} unique, {elapsed:.1f}s")
    return None


def solve_level_bfs_checkpointed(
    game_class,
    prefix_actions: list,
    max_states: int = 500_000,
    checkpoint_interval: int = 10,
    verbose: bool = True,
) -> Optional[tuple]:
    """BFS with periodic checkpoints to reduce replay cost.

    Instead of replaying from scratch every time, stores game snapshots
    at regular depth intervals and replays from the nearest checkpoint.
    """
    import copy

    game = replay_actions(game_class, prefix_actions)
    initial_score = game._score
    initial_level = game._current_level_index
    initial_hash = hash_state(game)

    # Each entry: (level_actions, checkpoint_game_or_None)
    # checkpoint stored every checkpoint_interval depths
    visited = {initial_hash}
    queue = deque([([], game)])  # (actions_from_level_start, game_at_state)

    explored = 0
    start_time = time.time()

    while queue:
        if explored >= max_states:
            if verbose:
                elapsed = time.time() - start_time
                print(f"    BFS limit: {explored} states, {len(visited)} unique, {elapsed:.1f}s")
            return None

        level_actions, parent_game = queue.popleft()
        actions = get_valid_actions(parent_game)
        depth = len(level_actions)

        for action in actions:
            clone = copy.deepcopy(parent_game)
            result = step_game(clone, action)
            explored += 1
            new_level_actions = level_actions + [action]

            if clone._score > initial_score or clone._state == GameState.WIN:
                elapsed = time.time() - start_time
                if verbose:
                    print(f"    SOLVED level {initial_level} in {len(new_level_actions)} actions, "
                          f"{explored} explored, {len(visited)} unique, {elapsed:.1f}s")
                return new_level_actions, prefix_actions + new_level_actions

            if clone._state == GameState.GAME_OVER:
                continue

            h = hash_state(clone)
            if h not in visited:
                visited.add(h)
                queue.append((new_level_actions, clone))

        if verbose and explored % 5000 == 0:
            elapsed = time.time() - start_time
            print(f"    ... {explored} explored, {len(visited)} unique, "
                  f"depth={depth}, queue={len(queue)}, {elapsed:.1f}s")

    if verbose:
        elapsed = time.time() - start_time
        print(f"    BFS exhausted: {explored} explored, {len(visited)} unique, {elapsed:.1f}s")
    return None


def solve_game_full(
    game_id: str,
    max_states_per_level: int = 500_000,
    use_checkpoints: bool = True,
    verbose: bool = True,
) -> dict:
    """Solve all levels of a game. Returns solution dict."""
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

    solution = {
        "game_id": meta.get("game_id", game_id),
        "total_levels": total_levels,
        "levels": [],
    }

    prefix_actions = []  # All actions taken so far across all levels
    total_start = time.time()

    solver_fn = solve_level_bfs_checkpointed if use_checkpoints else solve_level_bfs

    for level_idx in range(total_levels):
        baseline = baselines[level_idx] if level_idx < len(baselines) else "?"
        if verbose:
            print(f"\n  Level {level_idx} (baseline: {baseline} actions)")

        result = solver_fn(
            cls,
            prefix_actions,
            max_states=max_states_per_level,
            verbose=verbose,
        )

        if result is None:
            if verbose:
                print(f"    FAILED to solve level {level_idx}")
            solution["levels"].append({
                "level": level_idx,
                "solved": False,
                "actions": [],
            })
            break

        level_actions, full_actions = result
        prefix_actions = full_actions

        solution["levels"].append({
            "level": level_idx,
            "solved": True,
            "num_actions": len(level_actions),
            "baseline": baseline,
            "actions": [action_to_dict(a) for a in level_actions],
        })

        # Check if game is fully won
        verify_game = replay_actions(cls, prefix_actions)
        if verify_game._state == GameState.WIN:
            if verbose:
                print(f"\n  GAME WON! All {total_levels} levels solved!")
            break

    total_time = time.time() - total_start
    solved_levels = sum(1 for l in solution["levels"] if l.get("solved"))
    solution["solved_levels"] = solved_levels
    solution["total_time"] = round(total_time, 1)

    if verbose:
        print(f"\n  Result: {solved_levels}/{total_levels} levels in {total_time:.1f}s")

    return solution


def solve_all_games(
    game_ids: Optional[list] = None,
    max_states_per_level: int = 500_000,
    output_dir: str = None,
) -> dict:
    """Solve all games and save solutions."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "solutions")
    os.makedirs(output_dir, exist_ok=True)

    if game_ids is None:
        game_ids = list_all_games()

    results = {}
    grand_total = 0
    grand_possible = 0

    for game_id in game_ids:
        try:
            solution = solve_game_full(
                game_id,
                max_states_per_level=max_states_per_level,
            )
            results[game_id] = solution

            out_path = os.path.join(output_dir, f"{game_id}.json")
            with open(out_path, "w") as f:
                json.dump(solution, f, indent=2)

            grand_total += solution["solved_levels"]
            grand_possible += solution["total_levels"]

        except Exception as e:
            print(f"\n  ERROR on {game_id}: {e}")
            import traceback
            traceback.print_exc()
            results[game_id] = {"error": str(e)}

    print(f"\n{'='*60}")
    print(f"GRAND TOTAL: {grand_total}/{grand_possible} levels")
    print(f"{'='*60}")

    summary_path = os.path.join(output_dir, "_summary.json")
    summary = {
        "total_solved": grand_total,
        "total_possible": grand_possible,
        "per_game": {
            gid: {
                "solved": r.get("solved_levels", 0),
                "total": r.get("total_levels", 0),
                "time": r.get("total_time", 0),
            }
            for gid, r in results.items()
            if "error" not in r
        },
        "errors": {gid: r["error"] for gid, r in results.items() if "error" in r},
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        game_ids = sys.argv[1:]
    else:
        game_ids = None
    solve_all_games(game_ids=game_ids)
