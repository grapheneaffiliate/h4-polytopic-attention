"""
Priority-Group Graph Explorer v2 for ARC-AGI-3

Fixes over v1:
1. Death-action tracking: records which (state, action) pairs lead to GAME_OVER
2. Winning path replay: fast-forwards through solved levels after reset
3. Depth-biased exploration: prefers deeper frontier nodes over shallow ones
4. Episode budget awareness: avoids frontier nodes too far to reach in one episode
"""

import hashlib
import random
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional


INFINITY = 2**30


def hash_frame(frame: np.ndarray, mask: Optional[np.ndarray] = None) -> str:
    """Hash a frame, optionally masking status bar."""
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


# -- Perception -------------------------------------------------------

def segment_frame(frame: np.ndarray) -> list[dict]:
    """Extract connected components as segments with metadata."""
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
                "color": color,
                "size": len(cells),
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "centroid": (sum(rows) // len(cells), sum(cols) // len(cells)),
                "cells": cells,
            })

    return segments


def detect_status_bar_mask(frame: np.ndarray) -> np.ndarray:
    """Detect status bar regions on edges of the frame."""
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


def classify_segments(segments: list[dict], status_bar_mask: np.ndarray) -> list[set]:
    """Classify segments into 5 priority groups."""
    SALIENT_COLORS = set(range(6, 16))
    MIN_WIDTH, MAX_WIDTH = 2, 32

    groups = [set() for _ in range(5)]

    for idx, seg in enumerate(segments):
        r1, c1, r2, c2 = seg["bbox"]
        w = c2 - c1 + 1
        h = r2 - r1 + 1
        is_salient = seg["color"] in SALIENT_COLORS
        is_medium = MIN_WIDTH <= w <= MAX_WIDTH and MIN_WIDTH <= h <= MAX_WIDTH

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


# -- Graph Explorer v2 ------------------------------------------------

@dataclass
class NodeInfo:
    """A node in the state graph."""
    name: str
    num_actions: int
    groups: list  # list of sets of untested action indices per group
    tested: dict = field(default_factory=dict)  # action_idx -> (success, target_hash)
    closed: bool = False
    distance: int = INFINITY
    depth: int = 0  # BFS depth from root (start state)
    death_actions: set = field(default_factory=set)  # actions that lead to GAME_OVER

    def has_open(self, active_group: int) -> bool:
        for g in range(active_group + 1):
            if g < len(self.groups) and self.groups[g]:
                return True
        return False

    def get_untested(self, active_group: int) -> Optional[int]:
        """Get an untested action from highest-priority available group."""
        for g in range(active_group + 1):
            if g < len(self.groups) and self.groups[g]:
                return random.choice(list(self.groups[g]))
        return None

    def record_test(self, action_idx: int, success: bool, target: Optional[str] = None):
        self.tested[action_idx] = (success, target)
        for g in self.groups:
            g.discard(action_idx)

    def record_death(self, action_idx: int):
        """Track that an action was taken right before GAME_OVER.
        Does NOT remove from groups — the death may be a step limit, not this action."""
        self.death_actions.add(action_idx)


class PriorityGraphExplorerV2:
    """
    Priority-group graph explorer v2.

    Key improvements:
    - Depth-biased frontier selection (prefer deep nodes)
    - Death action avoidance
    - Episode budget awareness
    - Root depth tracking
    """

    def __init__(self, n_groups: int = 5, episode_budget: int = 0):
        self.n_groups = n_groups
        self.episode_budget = episode_budget  # 0 = auto-detect
        self.nodes: dict[str, NodeInfo] = {}
        self.edges: dict[str, set] = defaultdict(set)
        self.rev_edges: dict[str, set] = defaultdict(set)
        self.frontier: set[str] = set()
        self.dist: dict[str, int] = {}
        self.next_hop: dict[str, tuple] = {}
        self.active_group: int = 0
        self.root: Optional[str] = None  # start state for depth tracking
        self._dirty = False  # lazy rebuild flag
        self._rebuild_count = 0

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
        self._rebuild_count = 0

    def add_node(self, name: str, num_actions: int, groups: list[set], depth: int = 0):
        if name in self.nodes:
            return
        node = NodeInfo(name=name, num_actions=num_actions,
                       groups=[set(g) for g in groups], depth=depth)
        self.nodes[name] = node

        if self.root is None:
            self.root = name

        if node.has_open(self.active_group):
            self.frontier.add(name)
            self._dirty = True  # defer rebuild
        else:
            node.closed = True

    def record_transition(self, source: str, action_idx: int, success: bool,
                         target: Optional[str] = None, target_actions: int = 0,
                         target_groups: Optional[list[set]] = None):
        if source not in self.nodes:
            return

        node = self.nodes[source]
        node.record_test(action_idx, success, target)

        if success and target is not None:
            self.edges[source].add((action_idx, target))
            self.rev_edges[target].add((action_idx, source))

            if target not in self.nodes and target_groups is not None:
                new_depth = node.depth + 1
                self.add_node(target, target_actions, target_groups, depth=new_depth)

        if not node.has_open(self.active_group):
            self._close_node(source)

    def record_death(self, state: str, action_idx: int):
        """Record that (state, action) leads to GAME_OVER."""
        if state in self.nodes:
            self.nodes[state].record_death(action_idx)

    def choose_action(self, current: str) -> Optional[int]:
        if current not in self.nodes:
            return None

        node = self.nodes[current]

        # Case 1: Current node has untested actions
        if node.has_open(self.active_group):
            return node.get_untested(self.active_group)

        # Lazy rebuild if dirty
        if self._dirty:
            self._rebuild_distances()
            self._dirty = False

        # Case 2: Navigate toward frontier (prefer deep frontier nodes)
        if current in self.next_hop:
            action_idx, next_node = self.next_hop[current]
            return action_idx

        # Case 3: Try advancing group
        self._maybe_advance_group()

        if current in self.next_hop:
            action_idx, next_node = self.next_hop[current]
            return action_idx

        return None

    def _close_node(self, name: str):
        node = self.nodes[name]
        if node.closed:
            return
        node.closed = True
        if name in self.frontier:
            self.frontier.discard(name)
            self._dirty = True  # defer rebuild
        self._maybe_advance_group()

    def _rebuild_distances(self):
        """BFS from frontier nodes, with depth-biased tie-breaking.

        When multiple frontier nodes are equidistant, prefer routing toward
        the deeper one (further from root = more likely to be near solution).
        """
        self.dist.clear()
        self.next_hop.clear()

        for name in self.nodes:
            self.dist[name] = INFINITY
            self.nodes[name].distance = INFINITY

        # Sort frontier by depth (deepest first) for tie-breaking
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
                # Prefer shorter distance; on tie, prefer deeper frontier target
                if new_dist < cur_dist:
                    self.dist[u] = new_dist
                    self.nodes[u].distance = new_dist
                    self.next_hop[u] = (action_idx, v)
                    queue.append(u)
                elif new_dist == cur_dist and u in self.next_hop:
                    # Tie-break: prefer route toward deeper frontier
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

    @property
    def num_states(self):
        return len(self.nodes)

    @property
    def is_finished(self):
        return not self.frontier and self.active_group >= self.n_groups - 1

    @property
    def max_depth(self):
        if not self.nodes:
            return 0
        return max(n.depth for n in self.nodes.values())


# -- Main Agent v2 ----------------------------------------------------

class ExplorationAgentV2:
    """
    ARC-AGI-3 agent v2 with:
    - Death action tracking
    - Winning path replay
    - Episode budget detection
    """

    def __init__(self):
        self.explorer = PriorityGraphExplorerV2(n_groups=5)
        self.status_bar_mask = None
        self.current_level = 0
        self.level_actions = 0
        self.winning_paths: dict[int, list] = {}  # level -> [(game_action_id, x, y), ...]
        self.play_history: list[tuple] = []  # ALL actions: (game_action_id, x, y)
        self.effective_history: list[tuple] = []  # Only state-changing actions
        self.retry_count = 0

        # Replay state
        self.replaying = False
        self.replay_queue: list[tuple] = []  # actions to replay

        # Episode budget tracking
        self.episode_lengths: list[int] = []
        self.current_episode_len = 0
        self.estimated_episode_budget = 0  # 0 = unknown

        # Per-frame caches
        self.frame_segments_cache: dict[str, list] = {}
        self.frame_groups_cache: dict[str, list] = {}
        self.segment_to_action: dict[str, dict] = {}
        self.action_to_segment: dict[str, dict] = {}
        self.arrow_actions = {1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5"}

    def process_frame(self, frame: np.ndarray, available_actions: list[int]) -> tuple:
        """Process a frame and return (hash, num_actions, groups, segments)."""
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
            groups = classify_segments(segments, self.status_bar_mask if self.status_bar_mask is not None else np.zeros_like(frame, dtype=bool))

            self.frame_segments_cache[frame_hash] = segments
            self.frame_groups_cache[frame_hash] = groups

        # Build action mapping
        num_click = len(segments)
        action_map = {}

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
        num_actions = len(action_map)
        self.segment_to_action[frame_hash] = action_map

        return frame_hash, num_actions, groups, segments

    def on_game_over(self):
        """Handle GAME_OVER: prepare for replay of solved levels."""
        # Track episode length
        self.episode_lengths.append(self.current_episode_len)
        self.current_episode_len = 0

        # Update episode budget estimate (use median of last 10)
        if len(self.episode_lengths) >= 3:
            recent = sorted(self.episode_lengths[-10:])
            self.estimated_episode_budget = recent[len(recent) // 2]

        self.retry_count += 1

        # Set up replay queue for solved levels
        self.replay_queue = []
        for level in sorted(self.winning_paths.keys()):
            self.replay_queue.extend(self.winning_paths[level])
        self.replaying = bool(self.replay_queue)

        self.play_history = []
        self.effective_history = []

    def choose_action(self, frame: np.ndarray, available_actions: list[int],
                     levels_completed: int) -> tuple:
        """Choose action. Returns (game_action_id, x, y)."""

        # Level-up detection
        if levels_completed > self.current_level:
            # Save only state-changing actions as winning path (much shorter!)
            self.winning_paths[self.current_level] = list(self.effective_history)
            self.current_level = levels_completed
            self.explorer.reset()
            self.frame_segments_cache.clear()
            self.frame_groups_cache.clear()
            self.segment_to_action.clear()
            self.play_history = []
            self.effective_history = []
            self.level_actions = 0
            self.status_bar_mask = None
            self.replaying = False
            self.replay_queue = []

        self.level_actions += 1
        self.current_episode_len += 1

        # Replay mode: fast-forward through solved levels
        if self.replaying and self.replay_queue:
            action = self.replay_queue.pop(0)
            game_action_id, x, y = action
            self.play_history.append(action)
            return (game_action_id, x, y)

        self.replaying = False

        # Normal exploration
        frame_hash, num_actions, groups, segments = self.process_frame(frame, available_actions)

        if frame_hash not in self.explorer.nodes:
            self.explorer.add_node(frame_hash, num_actions, groups)

        action_idx = self.explorer.choose_action(frame_hash)

        if action_idx is not None and frame_hash in self.segment_to_action:
            action_map = self.segment_to_action[frame_hash]
            if action_idx in action_map:
                game_action, data = action_map[action_idx]
                if data:
                    self.play_history.append((game_action, data["x"], data["y"]))
                    return (game_action, data["x"], data["y"])
                else:
                    self.play_history.append((game_action, 0, 0))
                    return (game_action, 0, 0)

        # Fallback
        simple = [a for a in available_actions if 1 <= a <= 5]
        if simple:
            a = random.choice(simple)
            return (a, 0, 0)
        if 6 in available_actions:
            return (6, random.randint(0, 63), random.randint(0, 63))
        return (0, 0, 0)

    def observe_result(self, prev_hash: str, action_idx: int,
                      new_frame: np.ndarray, available_actions: list[int],
                      last_action_tuple: tuple = None):
        """Record transition result. last_action_tuple = (game_action_id, x, y)."""
        new_hash = hash_frame(new_frame, self.status_bar_mask)
        success = prev_hash != new_hash

        if success:
            # Track state-changing action for efficient replay
            if last_action_tuple and not self.replaying:
                self.effective_history.append(last_action_tuple)

            _, num_actions, groups, _ = self.process_frame(new_frame, available_actions)
            self.explorer.record_transition(
                prev_hash, action_idx, success, new_hash,
                target_actions=num_actions, target_groups=groups
            )
        else:
            self.explorer.record_transition(prev_hash, action_idx, False)

    def observe_death(self, prev_hash: str, action_idx: int):
        """Record that the last action led to GAME_OVER."""
        if prev_hash is not None and action_idx is not None:
            self.explorer.record_death(prev_hash, action_idx)


def solve_game_v2(arc, game_id, max_actions=50000, verbose=True):
    """Run the v2 exploration agent on a game."""
    env = arc.make(game_id, render_mode=None)
    obs = env.reset()

    if not obs.frame:
        return {"game_id": game_id, "error": "no_frame"}

    frame = np.array(obs.frame[-1])
    agent = ExplorationAgentV2()

    if verbose:
        print(f"\n{'='*60}")
        print(f"[{game_id}] Grid: {frame.shape}, "
              f"Colors: {sorted(map(int, np.unique(frame)))}, "
              f"Levels: 0/{obs.win_levels}")

    from arcengine import GameAction

    total_actions = 0
    prev_hash = None
    prev_action_idx = None
    prev_action_tuple = None  # (game_action_id, x, y)

    while total_actions < max_actions:
        if obs.state.name == "WIN":
            break

        if obs.state.name in ["NOT_PLAYED", "GAME_OVER"]:
            if obs.state.name == "GAME_OVER":
                # FIX #1: Record death transition from last action
                agent.observe_death(prev_hash, prev_action_idx)
                agent.on_game_over()
                if verbose and agent.retry_count <= 5:
                    print(f"  GAME_OVER (retry #{agent.retry_count}, "
                          f"ep_budget~{agent.estimated_episode_budget})")

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

        # Record previous transition
        if prev_hash is not None and prev_action_idx is not None:
            agent.observe_result(prev_hash, prev_action_idx, frame, available,
                               last_action_tuple=prev_action_tuple)

        # Choose action
        fh, _, _, _ = agent.process_frame(frame, available)
        game_action_id, x, y = agent.choose_action(frame, available, obs.levels_completed)

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
        prev_action_tuple = (game_action_id, x, y)

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

        if verbose and total_actions % 500 == 0:
            print(f"  [{total_actions}] States: {agent.explorer.num_states}, "
                  f"Depth: {agent.explorer.max_depth}, "
                  f"Group: {agent.explorer.active_group}, "
                  f"Frontier: {len(agent.explorer.frontier)}, "
                  f"Deaths: {sum(len(n.death_actions) for n in agent.explorer.nodes.values())}, "
                  f"Levels: {obs.levels_completed}/{obs.win_levels}")

    result = {
        "game_id": game_id,
        "levels_completed": obs.levels_completed if obs else 0,
        "total_levels": obs.win_levels if obs else 0,
        "actions_used": total_actions,
        "states_explored": agent.explorer.num_states,
        "max_depth": agent.explorer.max_depth,
        "state": obs.state.name if obs else "UNKNOWN",
        "episode_budget": agent.estimated_episode_budget,
        "retries": agent.retry_count,
    }

    if verbose:
        print(f"  Result: {result['levels_completed']}/{result['total_levels']} levels, "
              f"{total_actions} actions, {agent.explorer.num_states} states, "
              f"max_depth={result['max_depth']}")

    return result


def run_all_v2(api_key, max_actions=50000, verbose=True):
    """Run v2 agent on all games."""
    from arc_agi import Arcade
    import time

    arc = Arcade(arc_api_key=api_key)
    envs = arc.get_environments()

    total_l = total_c = 0
    t0 = time.time()
    for e in envs:
        try:
            r = solve_game_v2(arc, e.game_id, max_actions, verbose=False)
            lc = r.get("levels_completed", 0)
            tl = r.get("total_levels", 0)
            total_c += lc
            total_l += tl
            status = "WIN" if r.get("state") == "WIN" else f"{lc}/{tl}"
            t = time.time() - t0
            print(f"{e.game_id}: {status} ({r['actions_used']} actions, "
                  f"{r['states_explored']} states, depth={r['max_depth']}, "
                  f"deaths_tracked, ep~{r['episode_budget']}) [{t:.0f}s]", flush=True)
        except Exception as ex:
            print(f"{e.game_id}: ERROR {ex}", flush=True)

    elapsed = time.time() - t0
    print(f"TOTAL: {total_c}/{total_l} levels ({total_c/max(total_l,1)*100:.1f}%) in {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    import sys, os
    api_key = os.environ.get("ARC_API_KEY", "58b421be-5980-4ee8-8e57-0f18dc9369f3")

    from arc_agi import Arcade
    arc = Arcade(arc_api_key=api_key)

    if len(sys.argv) > 1:
        game_id = sys.argv[1]
        max_a = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
        solve_game_v2(arc, game_id, max_a)
    else:
        run_all_v2(api_key)
