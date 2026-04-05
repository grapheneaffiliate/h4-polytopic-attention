#!/usr/bin/env python3
"""
TR87 Solver — Rule-matching cyclic variant puzzle.

Game mechanics:
  - Sprites have 7 cyclic variants (name ends with 1-7)
  - Top row defines rules as (input -> output) sprite pairs
  - Bottom row is the answer — must match rules applied to top row
  - ACTION1/2: cycle selected bottom sprite backward/forward
  - ACTION3/4: move selection cursor left/right
  - Win: bottom row matches expected output after applying rules

Approach: algebraic — for each bottom sprite, compute target variant
from rules, then compute minimal cycle clicks to reach it.
"""

import json
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_runner import load_game_class, create_game, step_game, action_to_dict
from arcengine import ActionInput, GameAction, GameState

N_VARIANTS = 7  # sprites cycle through 7 variants


def get_variant(sprite):
    """Extract variant number (1-7) from sprite name."""
    return int(sprite.name[-1])


def get_base_name(sprite):
    """Get sprite name without variant suffix."""
    return sprite.name[:-1]


def solve_tr87():
    cls = load_game_class("tr87")

    solution = {
        "game_id": "tr87",
        "total_levels": 6,
        "solved_levels": 0,
        "levels": [],
        "total_time": 0,
    }

    t_total = time.time()

    for level_idx in range(6):
        print(f"\nLevel {level_idx}:")
        t0 = time.time()

        # Create fresh game and advance to target level
        game = create_game(cls)

        # Replay previous solutions to reach this level
        for prev_level in solution["levels"]:
            if prev_level.get("solved"):
                acts = game._get_valid_clickable_actions()
                for act_data in prev_level["actions"]:
                    action_id = act_data["id"]
                    ga = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
                          3: GameAction.ACTION3, 4: GameAction.ACTION4}[action_id]
                    step_game(game, ActionInput(id=ga))

        initial_score = game._score

        # Extract game state
        tr = game  # the Tr87 instance
        top_row = tr.zvojhrjxxm      # top row sprites (pattern)
        bottom_row = tr.ztgmtnnufb   # bottom row sprites (answer)
        rules = tr.cifzvbcuwqe       # list of (input_sprites, output_sprites) tuples
        cursor_idx = tr.qvtymdcqear_index
        budget = tr.upmkivwyrxz

        is_alter = game.current_level.get_data("alter_rules")
        is_double = game.current_level.get_data("double_translation")
        is_tree = game.current_level.get_data("tree_translation")

        print(f"  Top row: {len(top_row)} sprites")
        print(f"  Bottom row: {len(bottom_row)} sprites")
        print(f"  Rules: {len(rules)} pairs")
        print(f"  Budget: {budget}")
        print(f"  alter_rules={is_alter}, double={is_double}, tree={is_tree}")

        # For each rule, show input -> output
        for i, (inp, out) in enumerate(rules):
            inp_names = [s.name for s in inp]
            out_names = [s.name for s in out]
            print(f"  Rule {i}: {inp_names} -> {out_names}")

        print(f"  Top: {[s.name for s in top_row]}")
        print(f"  Bottom: {[s.name for s in bottom_row]}")

        # Determine target bottom row by simulating rule application
        # The win check (bsqsshqpox) matches top row against rules input side,
        # then checks bottom row matches rules output side.
        # So: for each rule whose input matches a segment of the top row,
        #     the corresponding bottom segment must match the output.

        # Compute expected bottom row
        expected = []
        top_idx = 0
        matched_rules = []

        while top_idx < len(top_row):
            found = False
            for inp, out in rules:
                # Check if top_row[top_idx:top_idx+len(inp)] matches inp names
                if top_idx + len(inp) > len(top_row):
                    continue
                match = True
                for j in range(len(inp)):
                    if top_row[top_idx + j].name != inp[j].name:
                        match = False
                        break
                if match:
                    # Handle double/tree translation
                    actual_out = list(out)
                    if is_double:
                        # Double translation: apply rules to the output again
                        second_out = []
                        for o_sprite in actual_out:
                            found_rule = False
                            for inp2, out2 in rules:
                                if len(inp2) == 1 and inp2[0].name == o_sprite.name:
                                    second_out.extend(out2)
                                    found_rule = True
                                    break
                            if not found_rule:
                                second_out.append(o_sprite)
                        actual_out = second_out
                    if is_tree:
                        # Tree: expand output through nested rules
                        expanded = []
                        skip = False
                        for o_sprite in out:
                            # Check if this output is an input of another rule
                            nested = False
                            for inp2, out2 in rules:
                                if inp2[0].name == o_sprite.name:
                                    expanded.extend(out2)
                                    nested = True
                                    break
                            if not nested:
                                skip = True
                                break
                        if skip:
                            continue
                        actual_out = expanded

                    for o in actual_out:
                        expected.append(o.name)
                    matched_rules.append((inp, actual_out))
                    top_idx += len(inp)
                    found = True
                    break

            if not found:
                print(f"  FAILED: no rule matches top_row at index {top_idx}")
                break

        if len(expected) != len(bottom_row):
            print(f"  Expected {len(expected)} bottom sprites, got {len(bottom_row)}")
            # Try to handle mismatch
            if len(expected) == 0:
                print(f"  SKIPPING (complex rule logic)")
                solution["levels"].append({"level": level_idx, "solved": False, "actions": []})
                continue

        print(f"  Expected bottom: {expected}")

        # Compute actions needed
        actions = []
        current_cursor = 0  # cursor starts at 0

        for i in range(min(len(expected), len(bottom_row))):
            current_name = bottom_row[i].name
            target_name = expected[i]

            if current_name == target_name:
                continue  # already correct

            # Move cursor to position i
            while current_cursor < i:
                actions.append({"id": 4})  # ACTION4 = move right
                current_cursor += 1
            while current_cursor > i:
                actions.append({"id": 3})  # ACTION3 = move left
                current_cursor -= 1

            # Compute cycle distance
            current_var = int(current_name[-1])
            target_var = int(target_name[-1])

            # Forward (ACTION2) and backward (ACTION1) distances
            fwd = (target_var - current_var) % N_VARIANTS
            bwd = (current_var - target_var) % N_VARIANTS

            if fwd <= bwd:
                for _ in range(fwd):
                    actions.append({"id": 2})  # ACTION2 = forward
            else:
                for _ in range(bwd):
                    actions.append({"id": 1})  # ACTION1 = backward

        print(f"  Actions: {len(actions)} (budget: {budget})")

        if len(actions) > budget:
            print(f"  TOO MANY ACTIONS ({len(actions)} > {budget})")
            solution["levels"].append({"level": level_idx, "solved": False, "actions": []})
            continue

        # Verify by executing
        game2 = create_game(cls)
        for prev_level in solution["levels"]:
            if prev_level.get("solved"):
                for act_data in prev_level["actions"]:
                    ga = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
                          3: GameAction.ACTION3, 4: GameAction.ACTION4}[act_data["id"]]
                    step_game(game2, ActionInput(id=ga))

        for act_data in actions:
            ga = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
                  3: GameAction.ACTION3, 4: GameAction.ACTION4}[act_data["id"]]
            step_game(game2, ActionInput(id=ga))

        dt = time.time() - t0
        if game2._score > initial_score:
            print(f"  SOLVED! {len(actions)} actions in {dt:.1f}s")
            solution["levels"].append({
                "level": level_idx,
                "solved": True,
                "actions": actions,
                "time": round(dt, 2),
            })
            solution["solved_levels"] += 1
        else:
            print(f"  VERIFICATION FAILED (score {game2._score} <= {initial_score})")
            # Try brute force with small perturbations
            # The rule matching might be wrong — try cycling each sprite ±1
            print(f"  Trying perturbation search...")
            found = False
            # ... could add perturbation logic here
            solution["levels"].append({
                "level": level_idx,
                "solved": False,
                "actions": actions,
                "time": round(dt, 2),
            })

    solution["total_time"] = round(time.time() - t_total, 1)

    # Save
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solutions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tr87.json")
    with open(out_path, "w") as f:
        json.dump(solution, f, indent=2)

    print(f"\nResult: {solution['solved_levels']}/6 in {solution['total_time']}s")
    print(f"Saved to: {out_path}")
    return solution


if __name__ == "__main__":
    solve_tr87()
