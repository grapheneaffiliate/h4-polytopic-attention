"""TN36 solver — programming puzzle.

Approach: compute the required opcodes abstractly from start/target state,
then generate the button toggles needed. Uses reset+replay (not deepcopy)
because of the lambda closure bug.
"""

import json
import os
import sys
import time
import itertools

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_runner import load_game_class, create_game, step_game, action_to_dict
from arcengine import ActionInput, GameAction, GameState


# Opcode effects (from source analysis)
OPCODE_EFFECTS = {
    0:  (0, 0, 0, 0, None),    # noop
    1:  (-4, 0, 0, 0, None),   # move left
    2:  (4, 0, 0, 0, None),    # move right
    3:  (0, 4, 0, 0, None),    # move down
    5:  (0, 0, 90, 0, None),   # rotate CW
    6:  (0, 0, -90, 0, None),  # rotate CCW
    7:  (0, 0, 180, 0, None),  # rotate 180
    8:  (0, 0, 0, 1, None),    # scale up
    9:  (0, 0, 0, -1, None),   # scale down
    10: (8, 0, 0, 0, None),    # move right 8
    11: (8, 0, 0, 0, None),    # move right 8
    12: (-8, 0, 0, 0, None),   # move left 8
    13: (-8, 0, 0, 0, None),   # move left 8
    14: (0, 0, 0, 0, 9),       # color = 9
    15: (0, 0, 0, 0, 8),       # color = 8
    16: (0, 0, 270, 0, None),  # rotate 270
    33: (0, -4, 0, 0, None),   # move up
    34: (-4, 0, 0, 0, None),   # move left
    63: (0, 0, 0, 0, 15),      # color = 15
}


def compute_program_effect(program):
    """Compute cumulative effect of a program (list of opcodes)."""
    dx, dy, drot, dscale = 0, 0, 0, 0
    color = None
    for op in program:
        if op in OPCODE_EFFECTS:
            edx, edy, erot, escale, ecolor = OPCODE_EFFECTS[op]
            dx += edx
            dy += edy
            drot += erot
            dscale += escale
            if ecolor is not None:
                color = ecolor
    return dx, dy, drot % 360, dscale, color


def find_programs(n_rows, available_opcodes, target_dx, target_dy, target_drot, target_dscale, target_color, start_color):
    """Find programs that produce the target effect.

    Uses smart enumeration: first find which opcodes produce needed effects,
    then combine them.
    """
    # Categorize opcodes by their primary effect
    dx_ops = {}  # dx -> opcode
    dy_ops = {}  # dy -> opcode
    rot_ops = {}  # drot -> opcode
    noop = 0

    for op in available_opcodes:
        if op not in OPCODE_EFFECTS:
            continue
        edx, edy, erot, escale, ecolor = OPCODE_EFFECTS[op]
        if escale != 0:
            continue  # Skip scale changes unless needed
        if ecolor is not None:
            continue  # Skip color changes unless needed
        if edx != 0 and edy == 0 and erot == 0:
            dx_ops[edx] = op
        if edy != 0 and edx == 0 and erot == 0:
            dy_ops[edy] = op
        if erot != 0 and edx == 0 and edy == 0:
            rot_ops[erot] = op

    # Try to construct the program from individual effects
    needed = []

    # Handle rotation
    if target_drot != 0:
        if target_drot in rot_ops:
            needed.append(rot_ops[target_drot])
        elif target_drot == 180 and 90 in rot_ops:
            needed.extend([rot_ops[90], rot_ops[90]])
        elif target_drot == 270 and -90 in rot_ops:
            needed.extend([rot_ops[-90], rot_ops[-90], rot_ops[-90]])
        else:
            return None  # Can't achieve rotation

    # Handle color
    if target_color is not None and target_color != start_color:
        for op in available_opcodes:
            if op in OPCODE_EFFECTS and OPCODE_EFFECTS[op][4] == target_color:
                needed.append(op)
                break
        else:
            return None

    # Handle dy
    remaining_dy = target_dy
    dy_steps = []
    for step_size in sorted(dy_ops.keys(), key=lambda x: -abs(x)):
        if remaining_dy == 0:
            break
        if step_size != 0:
            while (step_size > 0 and remaining_dy >= step_size) or \
                  (step_size < 0 and remaining_dy <= step_size):
                dy_steps.append(dy_ops[step_size])
                remaining_dy -= step_size
    if remaining_dy != 0:
        return None
    needed.extend(dy_steps)

    # Handle dx
    remaining_dx = target_dx
    dx_steps = []
    for step_size in sorted(dx_ops.keys(), key=lambda x: -abs(x)):
        if remaining_dx == 0:
            break
        if step_size != 0:
            while (step_size > 0 and remaining_dx >= step_size) or \
                  (step_size < 0 and remaining_dx <= step_size):
                dx_steps.append(dx_ops[step_size])
                remaining_dx -= step_size
    if remaining_dx != 0:
        return None
    needed.extend(dx_steps)

    # Handle scale
    if target_dscale != 0:
        if target_dscale > 0 and 8 in available_opcodes:
            needed.extend([8] * target_dscale)
        elif target_dscale < 0 and 9 in available_opcodes:
            needed.extend([9] * (-target_dscale))
        else:
            return None

    # Pad with noops
    if len(needed) > n_rows:
        return None  # Too many operations needed
    while len(needed) < n_rows:
        needed.append(0)

    return needed


def solve_tn36():
    cls = load_game_class("tn36")

    solution = {
        "game_id": "tn36",
        "total_levels": 7,
        "levels": [],
    }

    prefix_actions = []
    total_solved = 0
    t_total = time.time()

    for level_idx in range(7):
        print(f"\nLevel {level_idx}:")
        t0 = time.time()

        # Reset and replay to current level
        game = create_game(cls)
        actions = game._get_valid_clickable_actions()
        for aidx in prefix_actions:
            step_game(game, actions[aidx])

        initial_score = game._score
        right = game.tsflfunycx.xsseeglmfh
        tablet = right.tlwkpfljid
        blk = right.ravxreuqho
        tgt = right.ddzsdagbti

        n_rows = len(tablet.thofkgziyd)
        n_btns = len(tablet.thofkgziyd[0].puakvdstpr)
        n_opcodes = 2**n_btns

        initial_btns = []
        for row in tablet.thofkgziyd:
            initial_btns.append([btn.hokejgzome for btn in row.puakvdstpr])

        # Get available opcodes
        available_ops = [k for k in right.dfguzecnsr.keys() if k < n_opcodes]

        print(f"  {n_rows} rows x {n_btns} btns, {n_opcodes} opcodes")
        print(f"  Block: ({blk.hvvoimjrdh},{blk.lqlzulricb}) rot={blk.blhnfftand} scale={blk.uaixbyfwch} color={blk.dtxpbtpcbh}")
        if tgt:
            print(f"  Target: ({tgt.x},{tgt.y}) rot={tgt.rotation} scale={tgt.scale} color={tgt.dtxpbtpcbh}")

            # Compute needed changes
            target_dx = tgt.x - blk.hvvoimjrdh
            target_dy = tgt.y - blk.lqlzulricb
            target_drot = (tgt.rotation - blk.blhnfftand) % 360
            target_dscale = tgt.scale - blk.uaixbyfwch
            target_color = tgt.dtxpbtpcbh if tgt.dtxpbtpcbh != blk.dtxpbtpcbh else None

            print(f"  Need: dx={target_dx} dy={target_dy} drot={target_drot} dscale={target_dscale} color={target_color}")

        # Check selection buttons (multiple programs)
        n_sel = len(game.tsflfunycx.mcxkhvobyv)
        print(f"  Selection buttons: {n_sel}")

        # Strategy 1: Abstract solution (fast)
        if tgt and n_sel == 0:
            program = find_programs(n_rows, available_ops, target_dx, target_dy,
                                   target_drot, target_dscale, target_color, blk.dtxpbtpcbh)
            if program:
                print(f"  Abstract solution: {program}")
                # Try all permutations of the program
                tried_perms = set()
                for perm in itertools.permutations(program):
                    if perm in tried_perms:
                        continue
                    tried_perms.add(perm)

                    # Verify via game
                    game2 = create_game(cls)
                    act2 = game2._get_valid_clickable_actions()
                    for aidx in prefix_actions:
                        step_game(game2, act2[aidx])

                    r2 = game2.tsflfunycx.xsseeglmfh
                    tab2 = r2.tlwkpfljid

                    toggles = []
                    for i, row in enumerate(tab2.thofkgziyd):
                        target_opcode = perm[i]
                        for j, btn in enumerate(row.puakvdstpr):
                            target_state = bool(target_opcode & (1 << j))
                            if btn.hokejgzome != target_state:
                                if j == 0:
                                    toggles.append(i)
                                else:
                                    toggles.append(n_btns + i + (j-1) * n_rows)
                                    # Need to map button index correctly

                    # Build toggle action list
                    # Actions: 0..(n_rows-1) = top buttons, then play button, then more buttons
                    # For 6 buttons: need to figure out the action layout

                    # Direct mapping discovered empirically:
                    # Action i (0..n_rows-1) = toggle button[i][0]
                    # Action n_rows+1+i = toggle button[i][1]
                    # Action n_rows = play button
                    play_action = n_rows  # action 5 for 5 rows

                    button_clicks = []
                    for i, row in enumerate(tab2.thofkgziyd):
                        target_opcode = perm[i]
                        for j, btn in enumerate(row.puakvdstpr):
                            target_state = bool(target_opcode & (1 << j))
                            if btn.hokejgzome != target_state:
                                if j == 0:
                                    button_clicks.append(i)        # action 0..4
                                else:
                                    button_clicks.append(n_rows + 1 + i)  # action 6..10

                    # Apply button clicks via game actions
                    for aidx in button_clicks:
                        step_game(game2, act2[aidx])

                    # Verify program
                    r2 = game2.tsflfunycx.xsseeglmfh
                    tab2 = r2.tlwkpfljid
                    actual_prog = list(tab2.ylczjoyapu)
                    if actual_prog == list(perm):
                        # Find the play button
                        play_idx = None
                        for idx, a in enumerate(act2):
                            x, y = a.data.get('x', 0), a.data.get('y', 0)
                            if r2.owdgwmdfzu and r2.owdgwmdfzu.jfctiffjzp(x, y):
                                play_idx = idx
                                break

                        if play_idx is not None:
                            step_game(game2, act2[play_idx])
                            if game2._score > initial_score:
                                elapsed = time.time() - t0
                                print(f"  SOLVED via abstract! perm={list(perm)} {elapsed:.1f}s")
                                # Record button_indices for API replay
                                all_indices = button_clicks + [play_idx]
                                solution["levels"].append({
                                    "level": level_idx,
                                    "solved": True,
                                    "program": list(perm),
                                    "button_indices": all_indices,
                                    "actions": [{"id": 6, "data": {"x": act2[i].data.get("x", 0), "y": act2[i].data.get("y", 0)}} for i in all_indices],
                                })
                                prefix_actions = all_indices
                                game = game2
                                total_solved += 1
                                break
                else:
                    print(f"  Abstract failed (no permutation worked)")

        # Strategy 2: Brute force enumeration (only for small search spaces)
        if total_solved <= level_idx:  # Not yet solved
            search_space = n_opcodes ** n_rows
            if search_space <= 10000:
                print(f"  Trying brute force ({search_space} combos)...")
                for prog_tuple in itertools.product(range(n_opcodes), repeat=n_rows):
                    game2 = create_game(cls)
                    act2 = game2._get_valid_clickable_actions()
                    for aidx in prefix_actions:
                        step_game(game2, act2[aidx])

                    # Set buttons
                    r2 = game2.tsflfunycx.xsseeglmfh
                    tab2 = r2.tlwkpfljid
                    for i, row in enumerate(tab2.thofkgziyd):
                        for j, btn in enumerate(row.puakvdstpr):
                            btn.igsdzjpapk(bool(prog_tuple[i] & (1 << j)))

                    # Find play button
                    play_idx = None
                    for idx, a in enumerate(act2):
                        x, y = a.data.get('x', 0), a.data.get('y', 0)
                        if r2.owdgwmdfzu and r2.owdgwmdfzu.jfctiffjzp(x, y):
                            play_idx = idx
                            break

                    if play_idx is not None:
                        step_game(game2, act2[play_idx])
                        if game2._score > initial_score:
                            print(f"  SOLVED! prog={list(prog_tuple)}")
                            solution["levels"].append({
                                "level": level_idx,
                                "solved": True,
                                "program": list(prog_tuple),
                            })
                            game = game2
                            total_solved += 1
                            break
                else:
                    print(f"  Brute force failed")
            else:
                print(f"  Search space too large ({search_space}), skipping brute force")

        if total_solved <= level_idx:
            print(f"  FAILED")
            solution["levels"].append({"level": level_idx, "solved": False})
            break

        if game._state == GameState.WIN:
            print("\n  GAME WON!")
            break

    solution["solved_levels"] = total_solved
    solution["total_time"] = round(time.time() - t_total, 1)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solutions")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "tn36.json"), "w") as f:
        json.dump(solution, f, indent=2)

    print(f"\nResult: {total_solved}/7 in {solution['total_time']}s")
    return solution


if __name__ == "__main__":
    solve_tn36()
