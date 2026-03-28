"""
Unified ARC-AGI-3 Explorer — merges v2 + v3 innovations.

- v2 priority-group exploration with lazy rebuilds
- v2 winning path replay (state-changing actions only)
- v3 grid-click fallback: auto-switches when <50 states after 5K actions
- Depth-biased frontier selection
"""

import hashlib
import random
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional


INFINITY = 2**30


def hash_frame(frame: np.ndarray, mask: Optional[np.ndarray] = None) -> str:
    f = frame.copy()
    if mask is not None:
        f[mask] = 0
    flat = f.flatten().astype(np.uint8)
    if len(flat) % 2:
        flat = np.append(flat, 0)
    packed = (flat[0::2] << 4) | flat[1::2]
    h = hashlib.blake2b(packed.tobytes(), digest_size=16,
                        person=f"{frame.shape[0]}x{frame.shape[1]}".encode())
    return h.hexdigest()


def segment_frame(frame: np.ndarray) -> list[dict]:
    H, W = frame.shape
    visited = np.zeros((H, W), dtype=bool)
    segments = []
    for r in range(H):
        for c in range(W):
            if visited[r, c]:
                continue
            color = int(frame[r, c])
            cells = []
            queue = deque([(r, c)])
            visited[r, c] = True
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and int(frame[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            rows = [r for r, c in cells]
            cols = [c for r, c in cells]
            segments.append({
                "color": color, "size": len(cells),
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "centroid": (sum(rows) // len(cells), sum(cols) // len(cells)),
                "cells": cells,
            })
    return segments


def detect_status_bar_mask(frame: np.ndarray) -> np.ndarray:
    H, W = frame.shape
    mask = np.zeros((H, W), dtype=bool)
    for c in range(W - 1, max(W - 15, 0), -1):
        col = frame[:, c]
        dominant = int(np.bincount(col.astype(np.int32)).argmax())
        if np.sum(col == dominant) > H * 0.8:
            mask[:, c] = True
        else:
            break
    for r in range(H - 1, max(H - 8, 0), -1):
        row = frame[r, :]
        dominant = int(np.bincount(row.astype(np.int32)).argmax())
        if np.sum(row == dominant) > W * 0.8:
            mask[r, :] = True
        else:
            break
    for r in range(min(8, H)):
        row = frame[r, :]
        dominant = int(np.bincount(row.astype(np.int32)).argmax())
        if np.sum(row == dominant) > W * 0.8:
            mask[r, :] = True
        else:
            break
    return mask


def classify_segments(segments, status_bar_mask):
    SALIENT_COLORS = set(range(6, 16))
    groups = [set() for _ in range(5)]
    for idx, seg in enumerate(segments):
        r1, c1, r2, c2 = seg["bbox"]
        w, h = c2 - c1 + 1, r2 - r1 + 1
        is_salient = seg["color"] in SALIENT_COLORS
        is_medium = 2 <= w <= 32 and 2 <= h <= 32
        cr, cc = seg["centroid"]
        is_status = bool(status_bar_mask[cr, cc])
        if is_status:
            groups[4].add(idx)
        elif is_salient and is_medium:
            groups[0].add(idx)
        elif is_medium:
            groups[1].add(idx)
        elif is_salient:
            groups[2].add(idx)
        else:
            groups[3].add(idx)
    return groups


# -- Graph Explorer (from v2, with lazy rebuilds) -------------------------

@dataclass
class NodeInfo:
    name: str
    num_actions: int
    groups: list
    tested: dict = field(default_factory=dict)
    closed: bool = False
    distance: int = INFINITY
    depth: int = 0
    death_actions: set = field(default_factory=set)

    def has_open(self, active_group):
        for g in range(active_group + 1):
            if g < len(self.groups) and self.groups[g]:
                return True
        return False

    def get_untested(self, active_group):
        for g in range(active_group + 1):
            if g < len(self.groups) and self.groups[g]:
                return random.choice(list(self.groups[g]))
        return None

    def record_test(self, action_idx, success, target=None):
        self.tested[action_idx] = (success, target)
        for g in self.groups:
            g.discard(action_idx)

    def record_death(self, action_idx):
        self.death_actions.add(action_idx)


class GraphExplorer:
    def __init__(self, n_groups=5):
        self.n_groups = n_groups
        self.nodes = {}
        self.edges = defaultdict(set)
        self.rev_edges = defaultdict(set)
        self.frontier = set()
        self.dist = {}
        self.next_hop = {}
        self.active_group = 0
        self.root = None
        self._dirty = False

    def reset(self):
        self.nodes.clear()
        self.edges.clear()
        self.rev_edges.clear()
        self.frontier.clear()
        self.dist.clear()
        self.next_hop.clear()
        self.active_group = 0
        self.root = None
        self._dirty = False

    def add_node(self, name, num_actions, groups, depth=0):
        if name in self.nodes:
            return
        node = NodeInfo(name=name, num_actions=num_actions,
                       groups=[set(g) for g in groups], depth=depth)
        self.nodes[name] = node
        if self.root is None:
            self.root = name
        if node.has_open(self.active_group):
            self.frontier.add(name)
            self._dirty = True
        else:
            node.closed = True

    def record_transition(self, source, action_idx, success,
                         target=None, target_actions=0, target_groups=None):
        if source not in self.nodes:
            return
        node = self.nodes[source]
        node.record_test(action_idx, success, target)
        if success and target is not None:
            self.edges[source].add((action_idx, target))
            self.rev_edges[target].add((action_idx, source))
            if target not in self.nodes and target_groups is not None:
                self.add_node(target, target_actions, target_groups,
                            depth=node.depth + 1)
        if not node.has_open(self.active_group):
            self._close_node(source)

    def record_death(self, state, action_idx):
        if state in self.nodes:
            self.nodes[state].record_death(action_idx)

    def choose_action(self, current):
        if current not in self.nodes:
            return None
        node = self.nodes[current]
        if node.has_open(self.active_group):
            return node.get_untested(self.active_group)
        if self._dirty:
            self._rebuild_distances()
            self._dirty = False
        if current in self.next_hop:
            return self.next_hop[current][0]
        self._maybe_advance_group()
        if current in self.next_hop:
            return self.next_hop[current][0]
        return None

    def _close_node(self, name):
        node = self.nodes[name]
        if node.closed:
            return
        node.closed = True
        if name in self.frontier:
            self.frontier.discard(name)
            self._dirty = True
        self._maybe_advance_group()

    def _rebuild_distances(self):
        self.dist.clear()
        self.next_hop.clear()
        for name in self.nodes:
            self.dist[name] = INFINITY
            self.nodes[name].distance = INFINITY
        sorted_frontier = sorted(self.frontier,
                                key=lambda f: self.nodes[f].depth, reverse=True)
        queue = deque()
        for f in sorted_frontier:
            self.dist[f] = 0
            self.nodes[f].distance = 0
            queue.append(f)
        while queue:
            v = queue.popleft()
            v_dist = self.dist[v]
            v_depth = self.nodes[v].depth if v in self.nodes else 0
            for action_idx, u in self.rev_edges.get(v, set()):
                new_dist = v_dist + 1
                cur_dist = self.dist.get(u, INFINITY)
                if new_dist < cur_dist:
                    self.dist[u] = new_dist
                    self.nodes[u].distance = new_dist
                    self.next_hop[u] = (action_idx, v)
                    queue.append(u)
                elif new_dist == cur_dist and u in self.next_hop:
                    _, old_next = self.next_hop[u]
                    old_depth = self.nodes[old_next].depth if old_next in self.nodes else 0
                    if v_depth > old_depth:
                        self.next_hop[u] = (action_idx, v)

    def _maybe_advance_group(self):
        while not self.frontier and self.active_group < self.n_groups - 1:
            self.active_group += 1
            for name, node in self.nodes.items():
                node.closed = False
                if node.has_open(self.active_group):
                    self.frontier.add(name)
            self._rebuild_distances()
            self._dirty = False

    @property
    def num_states(self):
        return len(self.nodes)

    @property
    def max_depth(self):
        return max((n.depth for n in self.nodes.values()), default=0)

    @property
    def is_finished(self):
        return not self.frontier and self.active_group >= self.n_groups - 1


# -- Unified Agent ---------------------------------------------------------

class UnifiedAgent:
    """
    Adaptive agent that starts with segment-based clicking,
    then auto-switches to grid-click if stuck.
    """

    GRID_CLICK_THRESHOLD = 50   # switch to grid-click if fewer states than this
    GRID_CLICK_CHECK_AT = 5000  # check after this many actions

    def __init__(self, grid_step=4):
        self.explorer = GraphExplorer(n_groups=5)
        self.status_bar_mask = None
        self.current_level = 0
        self.winning_paths = {}
        self.play_history = []
        self.effective_history = []
        self.retry_count = 0
        self.replaying = False
        self.replay_queue = []
        self.episode_lengths = []
        self.current_episode_len = 0
        self.estimated_episode_budget = 0

        # Mode: "segment" or "grid"
        self.mode = "segment"
        self.grid_step = grid_step
        self.click_grid = []
        self.total_actions_this_level = 0

        # Caches
        self.frame_segments_cache = {}
        self.frame_groups_cache = {}
        self.segment_to_action = {}

    def _build_click_grid(self, H, W):
        points = []
        for r in range(self.grid_step // 2, H, self.grid_step):
            for c in range(self.grid_step // 2, W, self.grid_step):
                points.append((c, r))
        self.click_grid = points

    def _check_switch_to_grid(self):
        """Auto-switch to grid-click if exploration is stuck."""
        if (self.mode == "segment" and
            self.total_actions_this_level >= self.GRID_CLICK_CHECK_AT and
            self.explorer.num_states < self.GRID_CLICK_THRESHOLD):
            self.mode = "grid"
            self.explorer.reset()
            self.frame_segments_cache.clear()
            self.frame_groups_cache.clear()
            self.segment_to_action.clear()
            return True
        return False

    def process_frame_segment(self, frame, available_actions):
        """Segment-based frame processing (v2 style)."""
        if self.status_bar_mask is None:
            self.status_bar_mask = detect_status_bar_mask(frame)

        frame_hash = hash_frame(frame, self.status_bar_mask)

        if frame_hash in self.frame_segments_cache:
            segments = self.frame_segments_cache[frame_hash]
            groups = self.frame_groups_cache[frame_hash]
        else:
            masked = frame.copy()
            if self.status_bar_mask is not None:
                masked[self.status_bar_mask] = 0
            segments = segment_frame(masked)
            groups = classify_segments(segments, self.status_bar_mask)
            self.frame_segments_cache[frame_hash] = segments
            self.frame_groups_cache[frame_hash] = groups

        action_map = {}
        num_click = len(segments)
        if 6 in available_actions:
            for seg_idx, seg in enumerate(segments):
                cr, cc = seg["centroid"]
                action_map[seg_idx] = (6, {"x": int(cc), "y": int(cr)})

        arrow_offset = num_click
        arrow_groups_0 = set()
        for i, aid in enumerate(sorted(a for a in available_actions if 1 <= a <= 5)):
            idx = arrow_offset + i
            action_map[idx] = (aid, None)
            arrow_groups_0.add(idx)

        if 7 in available_actions:
            idx = arrow_offset + sum(1 for a in available_actions if 1 <= a <= 5)
            action_map[idx] = (7, None)
            if len(groups) > 3:
                groups[3].add(idx)

        groups[0] |= arrow_groups_0
        self.segment_to_action[frame_hash] = action_map

        return frame_hash, len(action_map), groups

    def process_frame_grid(self, frame, available_actions):
        """Grid-click frame processing (v3 style)."""
        if self.status_bar_mask is None:
            self.status_bar_mask = detect_status_bar_mask(frame)

        frame_hash = hash_frame(frame, self.status_bar_mask)

        if not self.click_grid:
            H, W = frame.shape
            self._build_click_grid(H, W)

        if frame_hash in self.segment_to_action:
            action_map = self.segment_to_action[frame_hash]
        else:
            action_map = {}
            idx = 0
            if 6 in available_actions:
                for x, y in self.click_grid:
                    if self.status_bar_mask is not None and self.status_bar_mask[y, x]:
                        continue
                    action_map[idx] = (6, {"x": int(x), "y": int(y)})
                    idx += 1
            for aid in sorted(a for a in available_actions if 1 <= a <= 5):
                action_map[idx] = (aid, None)
                idx += 1
            if 7 in available_actions:
                action_map[idx] = (7, None)
                idx += 1
            self.segment_to_action[frame_hash] = action_map

        num_actions = len(action_map)
        groups = [set(range(num_actions)), set(), set(), set(), set()]
        return frame_hash, num_actions, groups

    def process_frame(self, frame, available_actions):
        if self.mode == "grid":
            return self.process_frame_grid(frame, available_actions)
        return self.process_frame_segment(frame, available_actions)

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

    def choose_action(self, frame, available_actions, levels_completed):
        # Level-up
        if levels_completed > self.current_level:
            self.winning_paths[self.current_level] = list(self.effective_history)
            self.current_level = levels_completed
            self.explorer.reset()
            self.frame_segments_cache.clear()
            self.frame_groups_cache.clear()
            self.segment_to_action.clear()
            self.play_history = []
            self.effective_history = []
            self.status_bar_mask = None
            self.replaying = False
            self.replay_queue = []
            self.mode = "segment"  # reset to segment mode for new level
            self.total_actions_this_level = 0

        self.current_episode_len += 1
        self.total_actions_this_level += 1

        # Check grid-click switch
        self._check_switch_to_grid()

        # Replay mode
        if self.replaying and self.replay_queue:
            action = self.replay_queue.pop(0)
            self.play_history.append(action)
            return action
        self.replaying = False

        # Normal exploration
        frame_hash, num_actions, groups = self.process_frame(frame, available_actions)

        if frame_hash not in self.explorer.nodes:
            self.explorer.add_node(frame_hash, num_actions, groups)

        action_idx = self.explorer.choose_action(frame_hash)

        if action_idx is not None and frame_hash in self.segment_to_action:
            action_map = self.segment_to_action[frame_hash]
            if action_idx in action_map:
                game_action, data = action_map[action_idx]
                if data:
                    result = (game_action, data["x"], data["y"])
                else:
                    result = (game_action, 0, 0)
                self.play_history.append(result)
                return result

        # Fallback
        simple = [a for a in available_actions if 1 <= a <= 5]
        if simple:
            return (random.choice(simple), 0, 0)
        if 6 in available_actions:
            return (6, random.randint(0, 63), random.randint(0, 63))
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


# -- Game Runner -----------------------------------------------------------

def solve_game(arc, game_id, max_actions=150000, verbose=True):
    env = arc.make(game_id, render_mode=None)
    obs = env.reset()
    if not obs.frame:
        return {"game_id": game_id, "error": "no_frame"}

    frame = np.array(obs.frame[-1])
    agent = UnifiedAgent(grid_step=4)

    if verbose:
        print(f"[{game_id}] Grid: {frame.shape}, "
              f"Actions: {sorted(obs.available_actions or [])}, "
              f"Levels: {obs.win_levels}")

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

        action_map = agent.segment_to_action.get(fh, {})
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
        except Exception:
            total_actions += 1
            continue

        if obs is None:
            break
        total_actions += 1

        if verbose and total_actions % 2000 == 0:
            print(f"  [{total_actions}] mode={agent.mode} states={agent.explorer.num_states} "
                  f"depth={agent.explorer.max_depth} group={agent.explorer.active_group} "
                  f"levels={obs.levels_completed}/{obs.win_levels}")

    return {
        "game_id": game_id,
        "levels_completed": obs.levels_completed if obs else 0,
        "total_levels": obs.win_levels if obs else 0,
        "actions_used": total_actions,
        "states_explored": agent.explorer.num_states,
        "max_depth": agent.explorer.max_depth,
        "mode": agent.mode,
        "state": obs.state.name if obs else "UNKNOWN",
    }


def run_all(api_key, max_actions=150000, verbose=True):
    from arc_agi import Arcade
    import time
    arc = Arcade(arc_api_key=api_key)
    envs = arc.get_environments()
    total_l = total_c = 0
    t0 = time.time()
    for e in envs:
        try:
            r = solve_game(arc, e.game_id, max_actions, verbose=False)
            lc = r.get("levels_completed", 0)
            tl = r.get("total_levels", 0)
            total_c += lc
            total_l += tl
            status = "WIN" if r.get("state") == "WIN" else f"{lc}/{tl}"
            t = time.time() - t0
            print(f"{e.game_id}: {status} ({r['actions_used']} actions, "
                  f"{r['states_explored']} states, depth={r['max_depth']}, "
                  f"mode={r['mode']}) [{t:.0f}s]", flush=True)
        except Exception as ex:
            print(f"{e.game_id}: ERROR {ex}", flush=True)
    elapsed = time.time() - t0
    print(f"TOTAL: {total_c}/{total_l} levels ({total_c/max(total_l,1)*100:.1f}%) "
          f"in {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    import sys, os
    api_key = os.environ.get("ARC_API_KEY", "58b421be-5980-4ee8-8e57-0f18dc9369f3")
    from arc_agi import Arcade
    arc = Arcade(arc_api_key=api_key)
    if len(sys.argv) > 1:
        game_id = sys.argv[1]
        max_a = int(sys.argv[2]) if len(sys.argv) > 2 else 150000
        r = solve_game(arc, game_id, max_a)
        print(r)
    else:
        run_all(api_key)
