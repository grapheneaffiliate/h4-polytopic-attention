"""
Explorer v3: Grid-cell click for stuck games.

For games where segmentation finds <20 states in 10K+ actions,
try clicking EVERY cell in the grid, not just segment centroids.
Some games have invisible clickable regions that segmentation misses.
"""

import hashlib
import random
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

# Reuse v2's core components
from olympus.arc3.explorer_v2 import (
    hash_frame, segment_frame, detect_status_bar_mask,
    classify_segments, PriorityGraphExplorerV2, NodeInfo
)


class GridClickAgent:
    """
    For stuck games: instead of clicking segment centroids,
    click every cell in the grid systematically.
    """

    def __init__(self, grid_step: int = 4):
        """grid_step: click every Nth cell (4 = 16x16 grid of click points)."""
        self.grid_step = grid_step
        self.explorer = PriorityGraphExplorerV2(n_groups=3)
        self.status_bar_mask = None
        self.current_level = 0
        self.winning_paths: dict[int, list] = {}
        self.play_history: list[tuple] = []
        self.effective_history: list[tuple] = []
        self.retry_count = 0
        self.replaying = False
        self.replay_queue: list[tuple] = []
        self.episode_lengths: list[int] = []
        self.current_episode_len = 0
        self.estimated_episode_budget = 0

        # Grid click points
        self.click_grid: list[tuple] = []  # (x, y) points to try
        self.frame_action_map: dict[str, dict] = {}

    def _build_click_grid(self, H: int, W: int):
        """Build a grid of click points covering the entire frame."""
        points = []
        for r in range(self.grid_step // 2, H, self.grid_step):
            for c in range(self.grid_step // 2, W, self.grid_step):
                points.append((c, r))  # (x, y) format
        self.click_grid = points

    def process_frame(self, frame: np.ndarray, available_actions: list[int]) -> tuple:
        """Process frame and build action map using grid clicks."""
        if self.status_bar_mask is None:
            self.status_bar_mask = detect_status_bar_mask(frame)

        frame_hash = hash_frame(frame, self.status_bar_mask)

        if not self.click_grid:
            H, W = frame.shape
            self._build_click_grid(H, W)

        if frame_hash in self.frame_action_map:
            action_map = self.frame_action_map[frame_hash]
        else:
            action_map = {}
            idx = 0

            # Grid click actions (one per grid point)
            if 6 in available_actions:
                for x, y in self.click_grid:
                    # Skip status bar clicks
                    if self.status_bar_mask is not None and self.status_bar_mask[y, x]:
                        continue
                    action_map[idx] = (6, {"x": int(x), "y": int(y)})
                    idx += 1

            # Arrow/simple actions
            arrow_start = idx
            for aid in sorted(a for a in available_actions if 1 <= a <= 5):
                action_map[idx] = (aid, None)
                idx += 1

            # Undo
            if 7 in available_actions:
                action_map[idx] = (7, None)
                idx += 1

            self.frame_action_map[frame_hash] = action_map

        # Build groups: all actions in group 0 (try everything)
        num_actions = len(action_map)
        groups = [set(range(num_actions)), set(), set()]

        return frame_hash, num_actions, groups

    def on_game_over(self):
        self.episode_lengths.append(self.current_episode_len)
        self.current_episode_len = 0
        if len(self.episode_lengths) >= 3:
            recent = sorted(self.episode_lengths[-10:])
            self.estimated_episode_budget = recent[len(recent) // 2]
        self.retry_count += 1
        self.replay_queue = []
        for level in sorted(self.winning_paths.keys()):
            self.replay_queue.extend(self.winning_paths[level])
        self.replaying = bool(self.replay_queue)
        self.play_history = []
        self.effective_history = []

    def choose_action(self, frame: np.ndarray, available_actions: list[int],
                     levels_completed: int) -> tuple:
        if levels_completed > self.current_level:
            self.winning_paths[self.current_level] = list(self.effective_history)
            self.current_level = levels_completed
            self.explorer.reset()
            self.frame_action_map.clear()
            self.play_history = []
            self.effective_history = []
            self.status_bar_mask = None
            self.replaying = False
            self.replay_queue = []

        self.current_episode_len += 1

        # Replay mode
        if self.replaying and self.replay_queue:
            action = self.replay_queue.pop(0)
            self.play_history.append(action)
            return action

        self.replaying = False

        frame_hash, num_actions, groups = self.process_frame(frame, available_actions)

        if frame_hash not in self.explorer.nodes:
            self.explorer.add_node(frame_hash, num_actions, groups)

        action_idx = self.explorer.choose_action(frame_hash)

        if action_idx is not None and frame_hash in self.frame_action_map:
            action_map = self.frame_action_map[frame_hash]
            if action_idx in action_map:
                game_action, data = action_map[action_idx]
                if data:
                    result = (game_action, data["x"], data["y"])
                else:
                    result = (game_action, 0, 0)
                self.play_history.append(result)
                return result

        # Fallback
        if 6 in available_actions and self.click_grid:
            x, y = random.choice(self.click_grid)
            return (6, x, y)
        simple = [a for a in available_actions if 1 <= a <= 5]
        if simple:
            return (random.choice(simple), 0, 0)
        return (0, 0, 0)

    def observe_result(self, prev_hash, action_idx, new_frame, available_actions,
                      last_action_tuple=None):
        new_hash = hash_frame(new_frame, self.status_bar_mask)
        success = prev_hash != new_hash
        if success:
            if last_action_tuple and not self.replaying:
                self.effective_history.append(last_action_tuple)
            _, num_actions, groups = self.process_frame(new_frame, available_actions)
            self.explorer.record_transition(
                prev_hash, action_idx, success, new_hash,
                target_actions=num_actions, target_groups=groups
            )
        else:
            self.explorer.record_transition(prev_hash, action_idx, False)

    def observe_death(self, prev_hash, action_idx):
        if prev_hash is not None and action_idx is not None:
            self.explorer.record_death(prev_hash, action_idx)


def solve_game_gridclick(arc, game_id, max_actions=50000, grid_step=4, verbose=True):
    """Run grid-click agent on a game."""
    env = arc.make(game_id, render_mode=None)
    obs = env.reset()

    if not obs.frame:
        return {"game_id": game_id, "error": "no_frame"}

    frame = np.array(obs.frame[-1])
    agent = GridClickAgent(grid_step=grid_step)

    if verbose:
        H, W = frame.shape
        n_clicks = (H // grid_step) * (W // grid_step)
        print(f"\n{'='*60}")
        print(f"[{game_id}] GRID-CLICK mode (step={grid_step}, ~{n_clicks} click points)")
        print(f"Grid: {frame.shape}, Colors: {sorted(map(int, np.unique(frame)))}")
        print(f"Available: {sorted(obs.available_actions or [])}, Levels: {obs.win_levels}")

    from arcengine import GameAction

    total_actions = 0
    prev_hash = None
    prev_action_idx = None
    prev_action_tuple = None

    while total_actions < max_actions:
        if obs.state.name == "WIN":
            break

        if obs.state.name in ["NOT_PLAYED", "GAME_OVER"]:
            if obs.state.name == "GAME_OVER":
                agent.observe_death(prev_hash, prev_action_idx)
                agent.on_game_over()
                if verbose and agent.retry_count <= 5:
                    print(f"  GAME_OVER (retry #{agent.retry_count})")

            obs = env.reset()
            if not obs.frame:
                break
            frame = np.array(obs.frame[-1])
            prev_hash = None
            prev_action_idx = None
            prev_action_tuple = None
            total_actions += 1
            continue

        frame = np.array(obs.frame[-1])
        available = obs.available_actions or [1, 2, 3, 4, 5]

        if prev_hash is not None and prev_action_idx is not None:
            agent.observe_result(prev_hash, prev_action_idx, frame, available,
                               last_action_tuple=prev_action_tuple)

        fh, _, _ = agent.process_frame(frame, available)
        game_action_id, x, y = agent.choose_action(frame, available, obs.levels_completed)

        # Find internal action index
        action_map = agent.frame_action_map.get(fh, {})
        action_idx = None
        for idx, (gid, data) in action_map.items():
            if gid == game_action_id:
                if data is None or (data.get("x") == x and data.get("y") == y):
                    action_idx = idx
                    break

        prev_hash = fh
        prev_action_idx = action_idx
        prev_action_tuple = (game_action_id, x, y)

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

        if verbose and total_actions % 1000 == 0:
            print(f"  [{total_actions}] States: {agent.explorer.num_states}, "
                  f"Depth: {agent.explorer.max_depth}, "
                  f"Group: {agent.explorer.active_group}, "
                  f"Frontier: {len(agent.explorer.frontier)}, "
                  f"Levels: {obs.levels_completed}/{obs.win_levels}")

    result = {
        "game_id": game_id,
        "levels_completed": obs.levels_completed if obs else 0,
        "total_levels": obs.win_levels if obs else 0,
        "actions_used": total_actions,
        "states_explored": agent.explorer.num_states,
        "max_depth": agent.explorer.max_depth,
        "state": obs.state.name if obs else "UNKNOWN",
    }

    if verbose:
        print(f"  Result: {result['levels_completed']}/{result['total_levels']} levels, "
              f"{total_actions} actions, {agent.explorer.num_states} states")

    return result


if __name__ == "__main__":
    import sys, os, time
    api_key = os.environ.get("ARC_API_KEY", "58b421be-5980-4ee8-8e57-0f18dc9369f3")
    from arc_agi import Arcade
    arc = Arcade(arc_api_key=api_key)

    # Test on stuck games
    STUCK_GAMES = ["lf52-271a04aa", "sk48-41055498", "cd82-fb555c5d"]

    for gid in STUCK_GAMES:
        t0 = time.time()
        # Try with step=4 (256 click points), then step=2 (1024) if needed
        for step in [4, 2]:
            print(f"\n--- {gid} with grid_step={step} ---")
            r = solve_game_gridclick(arc, gid, max_actions=20000, grid_step=step)
            elapsed = time.time() - t0
            print(f"  Time: {elapsed:.0f}s")
            if r.get("levels_completed", 0) > 0:
                print(f"  *** SUCCESS: solved {r['levels_completed']} levels! ***")
                break
