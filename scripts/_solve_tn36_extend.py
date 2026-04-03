"""Extend TN36 solution from level 3 onward with proper action sequences."""
import sys, json, itertools
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, ".")

for mod in list(sys.modules.keys()):
    if "arc_game" in mod:
        del sys.modules[mod]

from local_runner import load_game_class, create_game, step_game, action_to_dict
from arcengine import ActionInput, GameAction, GameState

cls = load_game_class("tn36")

with open("solutions/tn36.json") as f:
    sol = json.load(f)

prefix_actions_data = []
for l in sol["levels"]:
    if l.get("solved"):
        prefix_actions_data.extend(l["actions"])

def solve_level(level_idx, prefix_data, sol):
    game = create_game(cls)
    for ad in prefix_data:
        a = ActionInput(id=GameAction.from_id(ad["id"]), data=ad.get("data", {}))
        step_game(game, a)

    if game._state in (GameState.WIN, GameState.GAME_OVER):
        return None

    right = game.tsflfunycx.xsseeglmfh
    blk = right.ravxreuqho
    tgt = right.ddzsdagbti
    tab = right.tlwkpfljid
    n_rows = len(tab.thofkgziyd)
    n_btns = len(tab.thofkgziyd[0].puakvdstpr)

    dx = tgt.x - blk.hvvoimjrdh
    dy = tgt.y - blk.lqlzulricb
    dscale = tgt.scale - blk.uaixbyfwch
    drot = (tgt.rotation - blk.blhnfftand) % 360

    print(f"Level {level_idx}: dx={dx} dy={dy} dscale={dscale} drot={drot}, {n_rows}x{n_btns}")

    needed = []
    r = dy
    while r >= 4: needed.append(3); r -= 4
    while r <= -4: needed.append(33); r += 4
    r = dx
    while r >= 4: needed.append(2); r -= 4
    while r <= -4: needed.append(1); r += 4
    if dscale > 0:
        for _ in range(dscale): needed.append(8)
    elif dscale < 0:
        for _ in range(-dscale): needed.append(9)
    if drot == 180: needed.append(7)
    elif drot == 90: needed.append(5)
    elif drot == 270: needed.append(6)
    while len(needed) < n_rows: needed.append(0)

    if len(needed) > n_rows:
        print(f"  Too many opcodes ({len(needed)} > {n_rows})")
        return None

    act = game._get_valid_clickable_actions()

    # Map buttons
    btn_map = {}
    for ri, row in enumerate(tab.thofkgziyd):
        for bi, btn in enumerate(row.puakvdstpr):
            for ai, a in enumerate(act):
                if btn.jfctiffjzp(a.data.get("x", 0), a.data.get("y", 0)):
                    btn_map[(ri, bi)] = ai
                    break

    cs = right.owdgwmdfzu
    play_ai = None
    if cs:
        for ai, a in enumerate(act):
            if cs.jfctiffjzp(a.data.get("x", 0), a.data.get("y", 0)):
                play_ai = ai
                break

    if play_ai is None:
        print(f"  No play button")
        return None

    unique_perms = set(itertools.permutations(needed))
    print(f"  Trying {len(unique_perms)} permutations...")

    for perm in sorted(unique_perms):
        game2 = create_game(cls)
        for ad in prefix_data:
            a2 = ActionInput(id=GameAction.from_id(ad["id"]), data=ad.get("data", {}))
            step_game(game2, a2)

        initial_score = game2._score
        act2 = game2._get_valid_clickable_actions()
        tab2 = game2.tsflfunycx.xsseeglmfh.tlwkpfljid

        toggles = []
        valid = True
        for ri, row in enumerate(tab2.thofkgziyd):
            target_op = perm[ri]
            for bi, btn in enumerate(row.puakvdstpr):
                want = bool(target_op & (1 << bi))
                if btn.hokejgzome != want:
                    if (ri, bi) in btn_map:
                        toggles.append(btn_map[(ri, bi)])
                    else:
                        valid = False
                        break
            if not valid:
                break
        if not valid:
            continue

        for idx in toggles:
            step_game(game2, act2[idx])
        step_game(game2, act2[play_ai])

        if game2._score > initial_score:
            level_actions = [action_to_dict(act2[i]) for i in toggles + [play_ai]]
            print(f"  SOLVED! {len(level_actions)} actions, prog={list(perm)}")
            return level_actions

    print(f"  FAILED")
    return None


for level_idx in range(3, 7):
    result = solve_level(level_idx, prefix_actions_data, sol)
    if result is None:
        break
    sol["levels"].append({
        "level": level_idx,
        "solved": True,
        "num_actions": len(result),
        "actions": result,
    })
    prefix_actions_data.extend(result)

sol["solved_levels"] = sum(1 for l in sol["levels"] if l.get("solved"))
with open("solutions/tn36.json", "w") as f:
    json.dump(sol, f, indent=2)
print(f"\nTN36: {sol['solved_levels']}/7")
