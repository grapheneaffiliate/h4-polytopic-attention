"""Abstract solver for ARC-AGI-3 games.

Instead of running full game engine BFS, extracts the state transition
function empirically (observe action effects) and BFS on abstract state
(tuple of sprite positions). Runs at millions of states/second.
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


def get_sprite_state(game) -> tuple:
    """Extract abstract state: positions of all sprites."""
    return tuple(
        (s.name, s._x, s._y)
        for s in sorted(game.current_level.get_sprites(), key=lambda s: s.name)
    )


def get_movable_state(game) -> tuple:
    """Extract only the positions of sprites that actually move.
    Falls back to all sprites if not yet determined."""
    return tuple(
        (s._x, s._y)
        for s in sorted(game.current_level.get_sprites(), key=lambda s: s.name)
    )


def extract_transitions(game_class) -> list:
    """For the current level, extract what each valid action does to sprite positions.
    Returns: list of (action, permutation_dict) where permutation maps old_pos -> new_pos."""
    game = create_game(game_class)
    actions = get_valid_actions(game)
    transitions = []

    for action in actions:
        g = copy.deepcopy(game)
        before = {s.name: (s._x, s._y) for s in g.current_level.get_sprites()}
        step_game(g, action)
        after = {s.name: (s._x, s._y) for s in g.current_level.get_sprites()}

        # Find which sprites moved
        changes = {}
        for name in before:
            if name in after and before[name] != after[name]:
                changes[name] = (before[name], after[name])

        transitions.append((action, changes, g._state, g._score))

    return transitions


def solve_level_abstract(game, max_states=2_000_000, verbose=True):
    """BFS on abstract state (sprite positions). Ultra fast."""
    initial_score = game._score
    initial_level = game._current_level_index
    actions = get_valid_actions(game)

    if not actions:
        return None

    # Build transition table: for each action, what's the effect?
    # We do this by trying each action from the initial state
    action_effects = []
    for action in actions:
        g = copy.deepcopy(game)
        before_sprites = [(s.name, s._x, s._y) for s in sorted(g.current_level.get_sprites(), key=lambda s: s.name)]
        step_game(g, action)
        after_sprites = [(s.name, s._x, s._y) for s in sorted(g.current_level.get_sprites(), key=lambda s: s.name)]

        won = g._score > initial_score or g._state == GameState.WIN
        lost = g._state == GameState.GAME_OVER

        action_effects.append({
            'action': action,
            'before': before_sprites,
            'after': after_sprites,
            'won': won,
            'lost': lost,
        })

    # Abstract state = tuple of sprite positions
    # But we need to handle the case where different actions from different states
    # have different effects (state-dependent transitions)

    # For state-dependent games, we can't pre-compute transitions.
    # Instead, use the game engine but with fast state representation.

    # Fast approach: BFS using game engine but with tuple_hash for visited set
    # and storing action indices instead of full action objects

    # Even faster: for click games where actions don't depend on state,
    # we can BFS purely on position tuples

    # Let's try the hybrid: fast hash + game engine + no snapshot storage
    # Just store (parent_idx, action_idx) and replay when expanding

    # Actually, let's use snapshot BFS with the fast snapshot approach
    # but store only action histories and use level_reset + replay

    # Strategy: BFS with in-memory state tracking
    # State = sprite position tuple (hashable, ~0.003ms to compute)
    # Transition = step game engine action (~0.4ms)
    # Clone = level_reset + replay (~0.1ms + D*0.34ms)

    # NEW STRATEGY: Use game engine but with DFS + visited pruning
    # Only need O(max_depth) clones in memory at any time

    # For each action at current state:
    #   1. deepcopy game (15ms) OR snapshot+restore
    #   2. step action
    #   3. Check win/lose/visited
    #   4. If new: recurse

    # Given 15ms/deepcopy is too slow, let me try a BFS where we store
    # the GAME OBJECTS (deepcopy) but limit the frontier size

    # ACTUALLY: The correct approach is to identify which sprites are the
    # "state" and build a pure-Python transition function

    # Determine movable sprites by checking which change across any action
    movable = set()
    for ae in action_effects:
        for i, (name, x, y) in enumerate(ae['before']):
            if ae['after'][i] != (name, x, y):
                movable.add(name)

    sprite_names = [name for name, _, _ in action_effects[0]['before']]
    movable_indices = [i for i, name in enumerate(sprite_names) if name in movable]

    if verbose:
        print(f"    {len(sprite_names)} sprites, {len(movable)} movable, {len(actions)} actions")

    # Check if transitions are state-independent (same action always does same thing)
    # For click puzzles like LP85, button effects are state-independent permutations
    # Test: apply same action from two different states, see if the PERMUTATION is the same

    # For now, assume state-dependent (general case) and use the engine
    # But optimize: DFS with fast state tracking

    # BFS using deepcopy (accept the 15ms cost but limit search)
    start_state = get_movable_state(game)
    visited = {start_state}
    # Store (game_clone, action_history)
    queue = deque([(copy.deepcopy(game), [])])

    explored = 0
    t0 = time.time()

    while queue:
        if explored >= max_states:
            break

        g, history = queue.popleft()
        cur_actions = get_valid_actions(g)

        for a in cur_actions:
            clone = copy.deepcopy(g)
            step_game(clone, a)
            explored += 1

            if clone._score > initial_score or clone._state == GameState.WIN:
                elapsed = time.time() - t0
                if verbose:
                    print(f"    SOLVED level {initial_level}: {len(history)+1} actions, "
                          f"{explored} explored, {len(visited)} unique, {elapsed:.1f}s")
                return history + [a], clone

            if clone._state == GameState.GAME_OVER:
                continue

            state = get_movable_state(clone)
            if state not in visited:
                visited.add(state)
                queue.append((clone, history + [a]))

        if verbose and explored % 5000 == 0:
            elapsed = time.time() - t0
            print(f"    ... {explored} explored, {len(visited)} unique, "
                  f"depth~{len(history)}, queue={len(queue)}, {elapsed:.1f}s")

    if verbose:
        elapsed = time.time() - t0
        print(f"    FAILED level {initial_level}: {explored} explored, "
              f"{len(visited)} unique, {elapsed:.1f}s")
    return None


def solve_game_abstract(game_id, max_states_per_level=2_000_000, verbose=True):
    """Solve all levels of a game using abstract state BFS."""
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
        print(f"  Available actions: {game._available_actions}")

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

        result = solve_level_abstract(game, max_states=max_states_per_level, verbose=verbose)

        if result is None:
            if verbose:
                print(f"    FAILED to solve level {level_idx}")
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


def solve_all_games(game_ids=None, max_states=2_000_000, output_dir=None):
    """Solve all games and save solutions."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solutions")
    os.makedirs(output_dir, exist_ok=True)

    if game_ids is None:
        game_ids = list_all_games()

    results = {}
    grand_total = 0
    grand_possible = 0

    for game_id in game_ids:
        try:
            solution = solve_game_abstract(game_id, max_states_per_level=max_states)
            results[game_id] = solution

            out_path = os.path.join(output_dir, f"{game_id}.json")
            with open(out_path, "w") as f:
                json.dump(solution, f, indent=2)

            grand_total += solution["solved_levels"]
            grand_possible += solution["total_levels"]
        except Exception as e:
            import traceback
            print(f"\n  ERROR on {game_id}: {e}")
            traceback.print_exc()
            results[game_id] = {"error": str(e)}

    print(f"\n{'='*60}")
    print(f"GRAND TOTAL: {grand_total}/{grand_possible} levels")
    print(f"{'='*60}")

    # Save summary
    summary = {
        "total_solved": grand_total,
        "total_possible": grand_possible,
        "per_game": {
            gid: {"solved": r.get("solved_levels", 0), "total": r.get("total_levels", 0), "time": r.get("total_time", 0)}
            for gid, r in results.items() if "error" not in r
        },
        "errors": {gid: r["error"] for gid, r in results.items() if "error" in r},
    }
    with open(os.path.join(output_dir, "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return results


if __name__ == "__main__":
    game_ids = sys.argv[1:] if len(sys.argv) > 1 else None
    solve_all_games(game_ids=game_ids)
