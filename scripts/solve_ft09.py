"""Direct solver for Ft09 — constraint satisfaction / Lights-Out puzzle.

Clicks commute and have binary effects → GF(2) linear algebra.
Solves purely abstractly (no engine calls in search loop).
"""

import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_runner import load_game_class, create_game, step_game, get_valid_actions, action_to_dict
from arcengine import ActionInput, GameAction, GameState
import numpy as np


def solve_ft09_level(game, verbose=True):
    """Solve current Ft09 level using abstract GF(2) search."""
    initial_score = game._score
    t0 = time.time()

    sprites = game.current_level.get_sprites()

    # Categorize sprites
    hkx_cells = []  # clickable cells (Hkx + NTi with click support)
    bst_constraints = []  # constraint checkers
    nti_cells = []  # NTi modifier cells
    for s in sprites:
        tags = s.tags or []
        tag = tags[0] if tags else ""
        if "Hkx" in tag:
            hkx_cells.append(s)
        elif "NTi" in tag:
            nti_cells.append(s)
            hkx_cells.append(s)  # NTi cells are also clickable
        elif "bsT" in tag:
            bst_constraints.append(s)

    n = len(hkx_cells)
    if verbose:
        print(f"    {n} cells, {len(bst_constraints)} constraints")

    if n == 0:
        return None

    palette = getattr(game, 'gqb', [9, 8])
    palette_map = {c: i for i, c in enumerate(palette)}

    # Build click actions (screen coordinates)
    scale, x_off, y_off = game.camera._calculate_scale_and_offset()
    click_actions = []
    for s in hkx_cells:
        sx = s._x * scale + x_off + 1
        sy = s._y * scale + y_off + 1
        click_actions.append(ActionInput(id=GameAction.ACTION6, data={"x": sx, "y": sy}))

    # Get initial colors
    initial_colors = []
    for s in hkx_cells:
        pixels = s.render()
        initial_colors.append(int(pixels[1, 1]))

    # Empirically determine click effects
    click_effects = []  # binary: which cells toggle
    for i, action in enumerate(click_actions):
        g = copy.deepcopy(game)
        before = [int(s.render()[1, 1]) for s in g.current_level.get_sprites() if (s.tags or [''])[0] and "Hkx" in (s.tags or [''])[0]]
        step_game(g, action)
        after = [int(s.render()[1, 1]) for s in g.current_level.get_sprites() if (s.tags or [''])[0] and "Hkx" in (s.tags or [''])[0]]
        effect = [1 if a != b else 0 for a, b in zip(after, before)]
        click_effects.append(effect)

    # Build abstract constraint check
    # Each bsT sprite is a 3x3 grid: center = target color, other pixels = per-neighbor modes
    # pixel == 0 means neighbor must EQUAL target, pixel != 0 means must NOT EQUAL target
    cell_positions = {(s._x, s._y): i for i, s in enumerate(hkx_cells)}
    # Also include NTi cells
    for s in sprites:
        tags = s.tags or []
        if any("NTi" in str(t) for t in tags):
            if (s._x, s._y) not in cell_positions:
                cell_positions[(s._x, s._y)] = len(hkx_cells) + len([k for k in cell_positions if k not in {(c._x, c._y) for c in hkx_cells}])

    # Offsets from bsT position to its 8 neighbors, matching pixel grid positions
    neighbor_offsets = [
        ((-4, -4), (0, 0)),  # top-left pixel
        ((0, -4), (0, 1)),   # top pixel
        ((4, -4), (0, 2)),   # top-right pixel
        ((-4, 0), (1, 0)),   # left pixel
        ((4, 0), (1, 2)),    # right pixel
        ((-4, 4), (2, 0)),   # bottom-left pixel
        ((0, 4), (2, 1)),    # bottom pixel
        ((4, 4), (2, 2)),    # bottom-right pixel
    ]

    constraint_rules = []  # list of (cell_index, must_equal_target, target_color)
    for bst in bst_constraints:
        pixels = bst.render()
        target_color = int(pixels[1, 1])
        bx, by = bst._x, bst._y

        for (dx, dy), (py, px) in neighbor_offsets:
            cell_pos = (bx + dx, by + dy)
            if cell_pos in cell_positions:
                cell_idx = cell_positions[cell_pos]
                if cell_idx < n:  # only check Hkx cells
                    mode_pixel = int(pixels[py, px])
                    must_equal = (mode_pixel == 0)
                    constraint_rules.append((cell_idx, must_equal, target_color))

    if verbose:
        eq_count = sum(1 for _, eq, _ in constraint_rules if eq)
        neq_count = sum(1 for _, eq, _ in constraint_rules if not eq)
        print(f"    Constraints: {eq_count} EQUAL + {neq_count} NOT_EQUAL = {len(constraint_rules)} rules")

    def check_win_abstract(colors):
        """Check if colors satisfy all constraints."""
        for cell_idx, must_equal, target in constraint_rules:
            if must_equal:
                if colors[cell_idx] != target:
                    return False
            else:
                if colors[cell_idx] == target:
                    return False
        return True

    def apply_clicks(initial, mask):
        """Apply a bitmask of clicks and return resulting colors."""
        colors = list(initial)
        for i in range(n):
            if mask & (1 << i):
                for j in range(n):
                    if click_effects[i][j]:
                        if colors[j] in palette_map:
                            cur = palette_map[colors[j]]
                            colors[j] = palette[(cur + 1) % len(palette)]
        return colors

    n_pal = len(palette)
    total_combos = n_pal ** n

    if verbose:
        print(f"    Search space: {n_pal}^{n} = {total_combos} combinations")

    # For 2-color: brute force bitmask (up to 2^25 = 33M)
    # For 3-color: BFS on abstract state (much faster)
    if n_pal == 2 and n <= 25:
        if verbose:
            print(f"    Binary brute force...")
        best_mask = None
        best_clicks = n + 1

        for mask in range(2**n):
            bits = bin(mask).count('1')
            if bits >= best_clicks:
                continue
            colors = apply_clicks(initial_colors, mask)
            if check_win_abstract(colors):
                best_mask = mask
                best_clicks = bits
    elif total_combos <= 50_000_000:
        # Small enough for brute force with multi-click
        if verbose:
            print(f"    Multi-color brute force...")
        best_mask = None
        best_clicks = n * n_pal

        def gen_combos(n, k):
            """Generate all n-tuples with values 0..k-1."""
            if n == 0:
                yield ()
                return
            for rest in gen_combos(n-1, k):
                for v in range(k):
                    yield rest + (v,)

        for combo in gen_combos(n, n_pal):
            total_c = sum(combo)
            if total_c >= best_clicks:
                continue
            colors = list(initial_colors)
            for i, clicks in enumerate(combo):
                for _ in range(clicks):
                    for j in range(len(colors)):
                        if click_effects[i][j]:
                            if colors[j] in palette_map:
                                cur = palette_map[colors[j]]
                                colors[j] = palette[(cur + 1) % n_pal]
            if check_win_abstract(colors):
                best_mask = combo
                best_clicks = total_c
    else:
        # Too large for brute force — use BFS on abstract state
        if verbose:
            print(f"    BFS on abstract state (too large for brute force)...")
        from collections import deque
        visited = {tuple(initial_colors)}
        queue = deque([(tuple(initial_colors), [])])
        best_mask = None
        t_bfs = time.time()

        while queue and time.time() - t_bfs < 300:
            state, history = queue.popleft()
            for i in range(n):
                new_colors = list(state)
                for j in range(len(new_colors)):
                    if click_effects[i][j]:
                        if new_colors[j] in palette_map:
                            cur = palette_map[new_colors[j]]
                            new_colors[j] = palette[(cur + 1) % n_pal]
                new_state = tuple(new_colors)
                if new_state not in visited:
                    visited.add(new_state)
                    new_hist = history + [i]
                    if check_win_abstract(new_colors):
                        best_mask = new_hist
                        if verbose:
                            print(f"    BFS found: {len(new_hist)} clicks, {len(visited)} states, {time.time()-t_bfs:.1f}s")
                        break
                    queue.append((new_state, new_hist))
            if best_mask:
                break
            if len(visited) % 100000 == 0:
                if verbose:
                    print(f"    ... {len(visited)} states, depth~{len(history)}, {time.time()-t_bfs:.1f}s")

        if not best_mask and verbose:
            print(f"    BFS exhausted: {len(visited)} states")
        best_clicks = len(best_mask) if best_mask else n * n_pal

    if best_mask is not None:
        # Build action list from mask
        if isinstance(best_mask, int):
            # Binary bitmask
            actions = [click_actions[i] for i in range(n) if best_mask & (1 << i)]
        elif isinstance(best_mask, (tuple, list)) and all(isinstance(x, int) for x in best_mask) and len(best_mask) == n:
            # Multi-click combo tuple
            actions = []
            for i, clicks in enumerate(best_mask):
                for _ in range(clicks):
                    actions.append(click_actions[i])
        else:
            # BFS result: list of click indices
            actions = [click_actions[idx] for idx in best_mask]
        g_verify = copy.deepcopy(game)
        for a in actions:
            step_game(g_verify, a)

        if g_verify._score > initial_score or g_verify._state == GameState.WIN:
            elapsed = time.time() - t0
            if verbose:
                print(f"    SOLVED: {best_clicks} clicks, verified with engine, {elapsed:.1f}s")
            return actions, g_verify
        else:
            # Abstract check passed but engine didn't advance
            # Try applying clicks individually (order might matter for step counter)
            if verbose:
                print(f"    Abstract win but engine didn't advance, trying sequential...")
            g2 = copy.deepcopy(game)
            ordered_actions = []
            for i in range(n):
                if best_mask & (1 << i):
                    step_game(g2, click_actions[i])
                    ordered_actions.append(click_actions[i])
                    if g2._score > initial_score:
                        elapsed = time.time() - t0
                        if verbose:
                            print(f"    SOLVED: {len(ordered_actions)} clicks (sequential), {elapsed:.1f}s")
                        return ordered_actions, g2

            # Still didn't win — constraint check might be wrong
            if verbose:
                print(f"    Abstract win didn't verify. Constraints may be incomplete.")

    elapsed = time.time() - t0
    if verbose:
        print(f"    FAILED after {elapsed:.1f}s")
    return None


def solve_ft09():
    """Solve all levels of Ft09."""
    print("="*60)
    print("Solving FT09")
    print("="*60)

    cls = load_game_class('ft09')
    game = create_game(cls)
    baselines = [20, 65, 6, 36, 30, 6]

    total_start = time.time()
    all_solutions = []

    for level_idx in range(game.win_score):
        baseline = baselines[level_idx] if level_idx < len(baselines) else "?"
        print(f"\nLevel {level_idx} (baseline: {baseline}):")

        result = solve_ft09_level(game)

        if result is None:
            print(f"  FAILED")
            all_solutions.append({"level": level_idx, "solved": False})
            break

        actions, game = result

        all_solutions.append({
            'level': level_idx,
            'solved': True,
            'num_actions': len(actions),
            'baseline': baseline,
            'actions': [action_to_dict(a) for a in actions],
        })

        if game._state == GameState.WIN:
            print(f"\nGAME WON!")
            break

    total_time = time.time() - total_start
    solved = sum(1 for l in all_solutions if l.get('solved'))
    print(f"\nResult: {solved}/{game.win_score} in {total_time:.1f}s")

    solution = {
        'game_id': 'ft09',
        'total_levels': game.win_score,
        'solved_levels': solved,
        'total_time': round(total_time, 1),
        'levels': all_solutions,
    }
    os.makedirs('solutions', exist_ok=True)
    with open('solutions/ft09.json', 'w') as f:
        json.dump(solution, f, indent=2)

    return solution


if __name__ == '__main__':
    solve_ft09()
