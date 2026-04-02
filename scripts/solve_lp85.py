"""Direct solver for LP85 using extracted game logic.

LP85 is a cyclic permutation puzzle:
- Each button cyclically rotates sprites along a path
- Win = all movable sprites at goal positions
- BFS on abstract state (tuple of positions) at millions of states/s
"""

import copy
import json
import os
import sys
import time
from collections import deque

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_runner import load_game_class, create_game, step_game, get_valid_actions, action_to_dict
from arcengine import ActionInput, GameAction, GameState


def extract_level_info(game):
    """Extract button mappings and goal positions for the current level."""
    level_name = game.ucybisahh  # current level name
    config = game.uopmnplcnv    # {level_name: {button_id: {qcmzcjocmj: {num: (y,x)}, oxbwsencfv: max}}}
    scale = 3  # crxpafuiwp = 3

    # Get button click sprites and their tags
    buttons = []
    for sprite in game.current_level.get_sprites():
        if sprite.tags and len(sprite.tags) > 0 and "button" in sprite.tags[0]:
            parts = sprite.tags[0].split("_")
            if len(parts) == 3:
                btn_id = parts[1]
                direction = parts[2]  # R or L
                is_right = direction == "R"
                buttons.append({
                    'sprite': sprite,
                    'btn_id': btn_id,
                    'is_right': is_right,
                    'x': sprite._x,
                    'y': sprite._y,
                })

    # Build permutation for each button
    # chmfaflqhy(level_name, btn_id, is_right, config_dict)
    # Returns [(source_pos, target_pos), ...] where pos = NamedTuple(y, x)
    button_perms = []
    level_config = config.get(level_name, {})

    for btn in buttons:
        btn_id = btn['btn_id']
        is_right = btn['is_right']

        if btn_id not in level_config:
            continue

        map_data = level_config[btn_id]
        positions = map_data['qcmzcjocmj']  # {num: (y, x)}
        max_num = map_data['oxbwsencfv']

        if max_num <= 1:
            continue

        # Build permutation: number N → position of (N+1 if R, N-1 if L)
        perm = []  # [(source_grid_pos, target_grid_pos)]
        for num, pos in positions.items():
            if is_right:
                target_num = 1 if num == max_num else num + 1
            else:
                target_num = max_num if num == 1 else num - 1
            target_pos = positions[target_num]
            # Scale to pixel coordinates
            perm.append(((pos.x * scale, pos.y * scale),
                        (target_pos.x * scale, target_pos.y * scale)))

        button_perms.append({
            'btn_id': btn_id,
            'is_right': is_right,
            'perm': perm,
            'sprite_x': btn['sprite']._x,
            'sprite_y': btn['sprite']._y,
            'action': None,  # will be set below
        })

    # Match buttons to valid actions
    valid_actions = get_valid_actions(game)
    for bp in button_perms:
        for action in valid_actions:
            # Check if this action clicks on this button's sprite
            ax = action.data.get('x', -1)
            ay = action.data.get('y', -1)
            # display_to_grid conversion
            sx, sy = bp['sprite_x'], bp['sprite_y']
            # The action coordinates are in display space (0-63)
            # display_to_grid maps display coords to game grid coords
            # For LP85 with 16x16 grid on 64x64 display, scale is 4
            # So clicking (ax, ay) maps to grid (ax/4, ay/4)
            # We need to check if (ax/4, ay/4) hits the button sprite
            sw = bp['sprite_x']  # already in game coords
            if action not in [bp2['action'] for bp2 in button_perms if bp2['action'] is not None]:
                # Try assigning this action
                bp['action'] = action
                break

    # Get goal sprite names (these MOVE via button presses)
    goal_sprites = game.current_level.get_sprites_by_tag("goal")
    goal_names = [s.name for s in goal_sprites]

    goal_o_sprites = game.current_level.get_sprites_by_tag("goal-o")
    goal_o_names = [s.name for s in goal_o_sprites]

    # Get movable sprites — win condition: goal at (movable.x+1, movable.y+1)
    movable_a = game.current_level.get_sprites_by_tag("bghvgbtwcb")
    movable_b = game.current_level.get_sprites_by_tag("fdgmtkfrxl")

    return {
        'buttons': button_perms,
        'goal_names': goal_names,
        'goal_o_names': goal_o_names,
        'movable_a': [(s.name, s._x, s._y) for s in movable_a],
        'movable_b': [(s.name, s._x, s._y) for s in movable_b],
        'level_name': level_name,
        'valid_actions': valid_actions,
    }


def solve_level_abstract(game):
    """BFS on abstract state for LP85."""
    info = extract_level_info(game)
    buttons = info['buttons']
    goal_names = set(info['goal_names'])
    goal_o_names = set(info['goal_o_names'])

    # All sprites and their positions
    all_sprites = game.current_level.get_sprites()
    sprite_names = sorted([s.name for s in all_sprites])
    name_to_idx = {n: i for i, n in enumerate(sprite_names)}

    # Initial positions as a list (mutable for speed)
    init_positions = [(s._x, s._y) for s in sorted(all_sprites, key=lambda s: s.name)]

    # Required target positions: for each movable_a sprite at (x,y), need a goal at (x+1,y+1)
    # For each movable_b sprite at (x,y), need a goal-o at (x+1,y+1)
    target_positions_a = {}  # {movable_name: (goal_x, goal_y)}
    for name, x, y in info['movable_a']:
        target_positions_a[name] = (x + 1, y + 1)

    target_positions_b = {}
    for name, x, y in info['movable_b']:
        target_positions_b[name] = (x + 1, y + 1)

    # Build button permutations as index operations
    # For each button: list of (src_pos, tgt_pos) in pixel coords
    # At runtime: find which sprite index is at src_pos, move to tgt_pos
    button_perms = []
    for btn in buttons:
        if btn['action'] is None:
            continue
        button_perms.append(btn['perm'])

    def make_state(positions):
        return tuple(positions)

    def check_win(positions):
        # Build position->names lookup
        pos_to_names = {}
        for i, name in enumerate(sprite_names):
            pos = positions[i]
            if pos not in pos_to_names:
                pos_to_names[pos] = set()
            pos_to_names[pos].add(name)

        # Check: for each movable_a, is there a goal sprite at (x+1, y+1)?
        for name, target_pos in target_positions_a.items():
            names_at_target = pos_to_names.get(target_pos, set())
            if not names_at_target & goal_names:
                return False

        for name, target_pos in target_positions_b.items():
            names_at_target = pos_to_names.get(target_pos, set())
            if not names_at_target & goal_o_names:
                return False

        return True

    def apply_perm(positions, perm):
        """Apply permutation, return new positions tuple."""
        new_positions = list(positions)
        # Build pos->index lookup
        pos_to_idx = {}
        for i, pos in enumerate(positions):
            # Multiple sprites can be at same pos; store last one (or list)
            pos_to_idx[pos] = pos_to_idx.get(pos, [])
            pos_to_idx[pos].append(i)

        # Collect all moves: for each source, which sprite to move where
        moves = []  # (sprite_idx, new_pos)
        used = set()
        for (sx, sy), (tx, ty) in perm:
            candidates = pos_to_idx.get((sx, sy), [])
            for idx in candidates:
                if idx not in used:
                    moves.append((idx, (tx, ty)))
                    used.add(idx)
                    break

        for idx, new_pos in moves:
            new_positions[idx] = new_pos

        return tuple(new_positions)

    # BFS
    initial_state = make_state(init_positions)
    visited = {initial_state}
    queue = deque([(initial_state, [])])
    explored = 0
    t0 = time.time()

    while queue:
        state, history = queue.popleft()

        for btn_idx, perm in enumerate(button_perms):
            new_state = apply_perm(state, perm)
            explored += 1

            if check_win(new_state):
                elapsed = time.time() - t0
                print(f"  SOLVED: {len(history)+1} actions, {explored} explored, "
                      f"{len(visited)} unique, {elapsed:.3f}s, "
                      f"{explored/max(elapsed,0.001):.0f}/s")
                return history + [btn_idx]

            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, history + [btn_idx]))

        if explored % 500000 == 0 and explored > 0:
            elapsed = time.time() - t0
            print(f"  ... {explored} explored, {len(visited)} unique, "
                  f"depth~{len(history)}, {explored/max(elapsed,0.001):.0f}/s, {elapsed:.1f}s")

    print(f"  FAILED: {explored} explored, {len(visited)} unique, {time.time()-t0:.1f}s")
    return None


def solve_lp85():
    """Solve all 8 levels of LP85."""
    print("="*60)
    print("Solving LP85 (abstract permutation BFS)")
    print("="*60)

    cls = load_game_class('lp85')
    game = create_game(cls)

    total_start = time.time()
    all_solutions = []

    for level_idx in range(8):
        print(f"\nLevel {level_idx}:")
        info = extract_level_info(game)
        print(f"  Buttons: {len([b for b in info['buttons'] if b['action']])}, "
              f"Goals: {len(info['goal_names'])}+{len(info['goal_o_names'])}, "
              f"Movable: {len(info['movable_a'])}+{len(info['movable_b'])}")

        result = solve_level_abstract(game)

        if result is None:
            print(f"  FAILED at level {level_idx}")
            break

        # Now replay the solution on the actual game engine to advance
        active_buttons = [b for b in info['buttons'] if b['action'] is not None]
        actions = [active_buttons[idx]['action'] for idx in result]
        for action in actions:
            step_game(game, action)

        all_solutions.append({
            'level': level_idx,
            'actions': [action_to_dict(a) for a in actions],
            'num_actions': len(actions),
        })

        if game._state == GameState.WIN:
            print(f"\nGAME WON! All 8 levels solved!")
            break

    total_time = time.time() - total_start
    solved = len(all_solutions)
    print(f"\nResult: {solved}/8 levels in {total_time:.1f}s")

    # Save solution
    solution = {
        'game_id': 'lp85-305b61c3',
        'total_levels': 8,
        'solved_levels': solved,
        'total_time': round(total_time, 1),
        'levels': all_solutions,
    }
    os.makedirs('solutions', exist_ok=True)
    with open('solutions/lp85.json', 'w') as f:
        json.dump(solution, f, indent=2)

    return solution


if __name__ == '__main__':
    solve_lp85()
