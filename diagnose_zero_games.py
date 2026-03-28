"""
Diagnose the 14 ARC-AGI-3 games with 0 levels solved.
For each game: run 10K actions, report states explored, action types,
state-change rate, group progression, and where the agent gets stuck.
"""
import os, sys, time, json
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from olympus.arc3.explorer import (
    ExplorationAgent, hash_frame, segment_frame,
    detect_status_bar_mask, classify_segments
)

API_KEY = os.environ.get("ARC_API_KEY", "58b421be-5980-4ee8-8e57-0f18dc9369f3")

# All 14 zero-level games (12 from handoff + r11l, tu93)
ZERO_GAMES = [
    "cd82-fb555c5d", "cn04-65d47d14", "dc22-4c9bff3e", "g50t-5849a774",
    "ka59-9f096b4a", "lf52-271a04aa", "re86-4e57566e", "sb26-7fbdac44",
    "sc25-f9b21a2f", "sk48-41055498", "su15-4c352900", "wa30-ee6fef47",
    "r11l-aa269680", "tu93-2b534c15",
]


def diagnose_game(arc, game_id, max_actions=10000):
    """Run diagnostic on a single game with detailed tracking."""
    from arcengine import GameAction

    env = arc.make(game_id, render_mode=None)
    obs = env.reset()

    if not obs.frame:
        return {"game_id": game_id, "error": "no_frame"}

    frame = np.array(obs.frame[-1])
    agent = ExplorationAgent()

    # Diagnostic counters
    stats = {
        "game_id": game_id,
        "grid_shape": list(frame.shape),
        "total_levels": obs.win_levels,
        "unique_colors": sorted(map(int, np.unique(frame))),
        "available_action_types": sorted(obs.available_actions) if obs.available_actions else [],
        "action_type_counts": Counter(),   # how many times each action type used
        "state_changes": 0,                # actions that changed state
        "no_change": 0,                    # actions that didn't change state
        "states_explored": 0,
        "group_progression": [],           # (action_count, group_level) transitions
        "game_overs": 0,
        "segments_at_start": 0,
        "group_sizes_at_start": [],
        "final_group": 0,
        "frontier_history": [],            # (action_count, frontier_size, states)
        "unique_states_per_1k": [],        # states discovered per 1K actions
        "last_new_state_at": 0,            # last action count when a new state was found
        "actions_used": 0,
    }

    # Initial frame analysis
    mask = detect_status_bar_mask(frame)
    segments = segment_frame(frame.copy())
    groups = classify_segments(segments, mask)
    stats["segments_at_start"] = len(segments)
    stats["group_sizes_at_start"] = [len(g) for g in groups]

    total_actions = 0
    prev_hash = None
    prev_action_idx = None
    prev_group = 0
    prev_states = 0

    while total_actions < max_actions:
        if obs.state.name == "WIN":
            stats["won"] = True
            break

        if obs.state.name in ["NOT_PLAYED", "GAME_OVER"]:
            if obs.state.name == "GAME_OVER":
                stats["game_overs"] += 1
                agent.play_history = []
            obs = env.reset()
            if not obs.frame:
                break
            frame = np.array(obs.frame[-1])
            prev_hash = None
            prev_action_idx = None
            total_actions += 1
            continue

        frame = np.array(obs.frame[-1])
        available = obs.available_actions or [1, 2, 3, 4, 5]

        # Record transition from previous action
        if prev_hash is not None and prev_action_idx is not None:
            new_hash = hash_frame(frame, agent.status_bar_mask)
            if new_hash != prev_hash:
                stats["state_changes"] += 1
                stats["last_new_state_at"] = total_actions
            else:
                stats["no_change"] += 1
            agent.observe_result(prev_hash, prev_action_idx, frame, available)

        # Track group transitions
        cur_group = agent.explorer.active_group
        if cur_group != prev_group:
            stats["group_progression"].append((total_actions, cur_group))
            prev_group = cur_group

        # Track state discovery rate
        cur_states = agent.explorer.num_states
        if cur_states != prev_states:
            prev_states = cur_states

        # Choose and execute action
        fh, _, _, _ = agent.process_frame(frame, available)
        game_action_id, x, y = agent.choose_action(frame, available, obs.levels_completed)
        stats["action_type_counts"][game_action_id] += 1

        # Find internal action index
        action_map = agent.segment_to_action.get(fh, {})
        action_idx = None
        for idx, (gid, data) in action_map.items():
            if gid == game_action_id:
                if data is None or (data.get("x") == x and data.get("y") == y):
                    action_idx = idx
                    break

        prev_hash = fh
        prev_action_idx = action_idx

        # Execute
        action = GameAction.from_id(game_action_id)
        if action.is_complex():
            action.set_data({"x": int(x), "y": int(y)})

        try:
            data = action.action_data.model_dump() if action.is_complex() else None
            obs = env.step(action, data=data)
        except Exception as e:
            total_actions += 1
            continue

        if obs is None:
            break
        total_actions += 1

        # Periodic snapshot
        if total_actions % 1000 == 0:
            stats["frontier_history"].append(
                (total_actions, len(agent.explorer.frontier), agent.explorer.num_states, cur_group)
            )

    stats["actions_used"] = total_actions
    stats["states_explored"] = agent.explorer.num_states
    stats["final_group"] = agent.explorer.active_group
    stats["levels_completed"] = obs.levels_completed if obs else 0
    stats["action_type_counts"] = dict(stats["action_type_counts"])

    # State change rate
    total_tested = stats["state_changes"] + stats["no_change"]
    stats["state_change_rate"] = stats["state_changes"] / max(total_tested, 1)

    return stats


def main():
    from arc_agi import Arcade
    arc = Arcade(arc_api_key=API_KEY)

    print("=" * 80)
    print("DIAGNOSTIC: 14 zero-level ARC-AGI-3 games @ 10K actions each")
    print("=" * 80)

    results = []
    for gid in ZERO_GAMES:
        t0 = time.time()
        print(f"\n--- {gid} ---")
        try:
            r = diagnose_game(arc, gid, max_actions=10000)
            elapsed = time.time() - t0
            r["elapsed_s"] = round(elapsed, 1)
            results.append(r)

            print(f"  Grid: {r['grid_shape']}, Colors: {r['unique_colors']}")
            print(f"  Available actions: {r['available_action_types']}")
            print(f"  Segments at start: {r['segments_at_start']}, Group sizes: {r['group_sizes_at_start']}")
            print(f"  States explored: {r['states_explored']}")
            print(f"  State changes: {r['state_changes']}/{r['state_changes']+r['no_change']} "
                  f"({r['state_change_rate']:.1%})")
            print(f"  Action types used: {r['action_type_counts']}")
            print(f"  Final group: {r['final_group']}, Game overs: {r['game_overs']}")
            print(f"  Group progression: {r['group_progression'][:10]}")
            print(f"  Levels: {r.get('levels_completed', 0)}/{r['total_levels']}")
            print(f"  Last new state at action: {r['last_new_state_at']}")
            print(f"  Frontier history: {r['frontier_history'][-5:]}")
            print(f"  Time: {elapsed:.1f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results.append({"game_id": gid, "error": str(e)})

    # Summary ranking
    print("\n" + "=" * 80)
    print("RANKING by states explored (most promising first)")
    print("=" * 80)
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda r: r["states_explored"], reverse=True)
    for i, r in enumerate(valid):
        print(f"  {i+1}. {r['game_id']}: {r['states_explored']} states, "
              f"{r['state_change_rate']:.0%} change rate, "
              f"group {r['final_group']}, "
              f"{r['game_overs']} game_overs, "
              f"actions={r['available_action_types']}")

    # Save results
    with open("diagnose_zero_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to diagnose_zero_results.json")


if __name__ == "__main__":
    main()
