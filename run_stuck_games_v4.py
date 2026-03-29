"""Run v4 on the 14 zero-level games that need grid-click breakthrough.

These games had 0 levels solved at 100K actions with v2.
Grid-click already unlocked cd82 and lf52 — this runs all 14 with
the v4 adaptive grid-click (auto-switches at 30 states / 3K actions).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from olympus.arc3.explorer_v4 import solve_game

API_KEY = os.environ.get("ARC_API_KEY", "58b421be-5980-4ee8-8e57-0f18dc9369f3")

# Games that had 0 levels at v2@100K (except cd82/lf52 which grid-click solved)
ZERO_LEVEL_GAMES = [
    "cd82-fb555c5d",  # 1/6 with grid-click, want more
    "cn04-65d47d14",  # solved 1/5 at 100K, inconsistent
    "dc22-4c9bff3e",
    "g50t-5849a774",
    "ka59-9f096b4a",
    "lf52-271a04aa",  # 1/10 with grid-click, want more
    "r11l-aa269680",
    "re86-4e57566e",  # 5807 states, deep exploration needed
    "sb26-7fbdac44",
    "sc25-f9b21a2f",
    "sk48-41055498",  # 316 states with grid-click, nearly cracking
    "tr87-cd924810",
    "tu93-2b534c15",
    "wa30-ee6fef47",
]

if __name__ == "__main__":
    from arc_agi import Arcade
    max_actions = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    arc = Arcade(arc_api_key=API_KEY)

    total_c = total_l = 0
    t0 = time.time()

    for gid in ZERO_LEVEL_GAMES:
        try:
            r = solve_game(arc, gid, max_actions, verbose=True)
            lc = r.get("levels_completed", 0)
            tl = r.get("total_levels", 0)
            total_c += lc
            total_l += tl
            t = time.time() - t0
            status = f"{lc}/{tl}"
            print(f"\n>>> {gid}: {status} ({r['actions_used']} actions, "
                  f"{r['states_explored']} states, mode={r['mode']}) [{t:.0f}s]\n",
                  flush=True)
        except Exception as ex:
            print(f"\n>>> {gid}: ERROR {ex}\n", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"STUCK GAMES TOTAL: {total_c}/{total_l} levels "
          f"({total_c/max(total_l,1)*100:.1f}%) in {elapsed:.0f}s")
