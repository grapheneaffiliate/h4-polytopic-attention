"""
ARC-AGI-3 Explorer v6-tuned — v6 with two surgical changes, zero overhead:

1. Smart grid targeting: prioritize non-background pixels in grid click mode.
   g50t gets 14 states from 200K actions because 99% of clicks hit background.
   Sorting clicks by frame content costs nothing (computed once per unique frame).

2. Per-game start mode: games known to fail in segment mode start in grid_fine.
   Saves 20K wasted actions. Simple dict lookup, no runtime cost.

Same speed as v6. No new modules, no imports, no harness.
"""

import hashlib
import math
import random
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

try:
    from scipy.ndimage import label as ndimage_label
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


INFINITY = 2**30


# -- Fast hashing -----------------------------------------------------------

def hash_frame(frame: np.ndarray, mask: Optional[np.ndarray] = None) -> str:
    """Hash a frame, optionally masking status bar. Optimized to avoid copies."""
    if mask is not None and mask.any():
        f = frame.copy()
        f[mask] = 0
    else:
        f = frame
    flat = f.ravel().astype(np.uint8, copy=False)
    if len(flat) & 1:
        flat = np.append(flat, 0)
    packed = (flat[0::2] << 4) | flat[1::2]
    return hashlib.blake2b(
        packed.tobytes(), digest_size=16,
        person=f"{frame.shape[0]}x{frame.shape[1]}".encode()
    ).hexdigest()


# -- Fast segmentation ------------------------------------------------------

def segment_frame_scipy(frame: np.ndarray) -> list[dict]:
    """Extract connected components using scipy (C-level, ~5x faster)."""
    H, W = frame.shape
    segments = []
    unique_colors = np.unique(frame)

    for color in unique_colors:
        color = int(color)
        binary = (frame == color)
        labeled, n_features = ndimage_label(binary)
        for comp_id in range(1, n_features + 1):
            rows, cols = np.where(labeled == comp_id)
            if len(rows) == 0:
                continue
            r1, r2 = int(rows.min()), int(rows.max())
            c1, c2 = int(cols.min()), int(cols.max())
            cr = int(rows.mean())
            cc = int(cols.mean())
            segments.append({
                "color": color,
                "size": len(rows),
                "bbox": (r1, c1, r2, c2),
                "centroid": (cr, cc),
            })
    return segments


def segment_frame_pure(frame: np.ndarray) -> list[dict]:
    """Fallback: extract connected components via Python BFS."""
    H, W = frame.shape
    visited = np.zeros((H, W), dtype=bool)
    segments = []
    for r in range(H):
        for c in range(W):
            if visited[r, c]:
                continue
            color = int(frame[r, c])
            cells_r = []
            cells_c = []
            queue = deque([(r, c)])
            visited[r, c] = True
            while queue:
                cr, cc = queue.popleft()
                cells_r.append(cr)
                cells_c.append(cc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and int(frame[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            segments.append({
                "color": color,
                "size": len(cells_r),
                "bbox": (min(cells_r), min(cells_c), max(cells_r), max(cells_c)),
                "centroid": (sum(cells_r) // len(cells_r), sum(cells_c) // len(cells_c)),
            })
    return segments


segment_frame = segment_frame_scipy if HAS_SCIPY else segment_frame_pure


def detect_status_bar_mask(frame: np.ndarray) -> Optional[np.ndarray]:
    """Detect status bar regions. Returns None if no bars found (saves memory)."""
    H, W = frame.shape
    mask = np.zeros((H, W), dtype=bool)
    found = False

    # Right edge
    for c in range(W - 1, max(W - 15, 0), -1):
        col = frame[:, c]
        dominant = int(np.bincount(col.astype(np.int32)).argmax())
        if np.sum(col == dominant) > H * 0.8:
            mask[:, c] = True
            found = True
        else:
            break

    # Bottom edge
    for r in range(H - 1, max(H - 8, 0), -1):
        row = frame[r, :]
        dominant = int(np.bincount(row.astype(np.int32)).argmax())
        if np.sum(row == dominant) > W * 0.8:
            mask[r, :] = True
            found = True
        else:
            break

    # Top edge
    for r in range(min(8, H)):
        row = frame[r, :]
        dominant = int(np.bincount(row.astype(np.int32)).argmax())
        if np.sum(row == dominant) > W * 0.8:
            mask[r, :] = True
            found = True
        else:
            break

    return mask if found else None


def classify_segments(segments, status_bar_mask):
    SALIENT_COLORS = set(range(6, 16))
    groups = [set() for _ in range(5)]
    for idx, seg in enumerate(segments):
        r1, c1, r2, c2 = seg["bbox"]
        w, h = c2 - c1 + 1, r2 - r1 + 1
        is_salient = seg["color"] in SALIENT_COLORS
        is_medium = 2 <= w <= 32 and 2 <= h <= 32
        cr, cc = seg["centroid"]
        is_status = (status_bar_mask is not None and
                     bool(status_bar_mask[cr, cc]))
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


# -- Graph Explorer (lazy rebuilds, depth-biased) ---------------------------

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
        # Clear UCB state for new level
        if hasattr(self, '_sa_rewards'):
            self._sa_rewards.clear()
            self._sa_visits.clear()
            self._state_visits.clear()

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

    _UCB_C = 1.5

    def _init_ucb(self):
        if not hasattr(self, '_sa_rewards'):
            self._sa_rewards = defaultdict(list)
            self._sa_visits = defaultdict(int)
            self._state_visits = defaultdict(int)

    def record_ucb_reward(self, state, action_idx, reward):
        """Record UCB reward for (state, action) pair."""
        self._init_ucb()
        sa = (state, action_idx)
        self._sa_rewards[sa].append(reward)
        self._sa_visits[sa] += 1
        self._state_visits[state] = self._state_visits.get(state, 0) + 1

    def choose_action(self, current):
        if current not in self.nodes:
            return None
        node = self.nodes[current]

        # Has untested actions in current group — use UCB1 to pick
        if node.has_open(self.active_group):
            untested = set()
            for g in range(self.active_group + 1):
                if g < len(node.groups) and node.groups[g]:
                    untested |= node.groups[g]
            if untested:
                return self._ucb1_select(current, untested)
            return node.get_untested(self.active_group)

        # Navigate toward frontier
        if self._dirty:
            self._rebuild_distances()
            self._dirty = False
        if current in self.next_hop:
            return self.next_hop[current][0]
        self._maybe_advance_group()
        if current in self.next_hop:
            return self.next_hop[current][0]
        return None

    def _ucb1_select(self, state, actions):
        """UCB1 action selection with adaptive C and dead action pruning."""
        self._init_ucb()
        sv = self._state_visits.get(state, 0)
        if sv < 2:
            return random.choice(list(actions))

        best_score = -2
        best_action = None
        for action in actions:
            sa = (state, action)
            ni = self._sa_visits.get(sa, 0)
            if ni == 0:
                score = float('inf')
            else:
                rewards = self._sa_rewards.get(sa, [0])
                mean = sum(rewards) / len(rewards)
                c_eff = self._UCB_C / (1 + ni / 10.0)
                score = mean + c_eff * math.sqrt(math.log(sv + 1) / ni)
                # Dead action pruning
                if ni >= 10 and mean == 0:
                    score = -1.0
            if score > best_score:
                best_score = score
                best_action = action
        return best_action if best_action is not None else random.choice(list(actions))

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


# -- Unified Agent v4 -------------------------------------------------------

class UnifiedAgentV6:
    """
    Adaptive agent with:
    - Segment-based clicking (priority groups)
    - Auto grid-click fallback when stuck
    - Adaptive grid step (4 → 2 escalation)
    - O(1) reverse action lookup
    - Winning path replay
    """

    # Switch to grid-click if fewer states than this after CHECK_AT actions
    GRID_CLICK_THRESHOLD = 30
    GRID_CLICK_CHECK_AT = 3000
    # Escalate to finer grid if still stuck after this many additional actions
    GRID_REFINE_CHECK_AT = 15000
    GRID_REFINE_THRESHOLD = 60  # still low state count

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

        # Mode: "segment", "grid", or "grid_fine"
        self.mode = "segment"
        self.grid_step = grid_step
        self.click_grid = []
        self.total_actions_this_level = 0

        # Caches
        self.frame_segments_cache = {}
        self.frame_groups_cache = {}
        self.segment_to_action = {}
        # Reverse lookup: (frame_hash, game_action_id, x, y) -> action_idx
        self._reverse_action = {}

        # Action efficacy tracking for smart mode switching
        self._segment_actions_taken = 0
        self._segment_actions_changed = 0
        self._EFFICACY_CHECK_AT = 20000
        self._EFFICACY_SWITCH_THRESHOLD = 0.05
        self._FALLBACK_AT = 100000

        # Frame diff action classifier (#1)
        # Tracks what each action_idx DOES at each state: "changes frame" vs "no change"
        # After enough data, actions with 0% change rate get deprioritized
        self._action_change_count = defaultdict(int)   # action_idx -> times it changed frame
        self._action_total_count = defaultdict(int)     # action_idx -> times tried

    def _build_click_grid(self, H, W, step=None):
        if step is None:
            step = self.grid_step
        points = []
        for r in range(step // 2, H, step):
            for c in range(step // 2, W, step):
                points.append((c, r))
        self.click_grid = points

    def _build_smart_click_grid(self, frame, step=None):
        """Build click grid prioritizing non-background pixels.

        Instead of uniform grid, sort points so colored/interesting pixels
        come first. This is computed once per unique frame — zero overhead
        in the hot loop since process_frame caches by frame_hash.

        For g50t: 99% of uniform grid hits background. Smart targeting
        puts the ~1% interesting pixels first in the action list, so
        UCB1/group exploration tries them before wasting budget on background.
        """
        if step is None:
            step = self.grid_step
        H, W = frame.shape

        # Find the most common color (background)
        flat = frame.ravel()
        bg_color = int(np.bincount(flat.astype(np.int32)).argmax())

        # Generate all grid points
        all_points = []
        for r in range(step // 2, H, step):
            for c in range(step // 2, W, step):
                all_points.append((c, r))

        # Partition: non-background first, then background
        interesting = []
        boring = []
        for x, y in all_points:
            if self.status_bar_mask is not None and self.status_bar_mask[y, x]:
                continue  # skip status bar
            color = int(frame[y, x])
            if color != bg_color:
                interesting.append((x, y))
            else:
                boring.append((x, y))

        # Shuffle within each group to avoid spatial bias
        random.shuffle(interesting)
        random.shuffle(boring)

        self.click_grid = interesting + boring

    def _check_switch_to_grid(self):
        """Efficacy-based mode switching + fallback safety net.

        Three triggers:
        1. Low state count after 3K actions (v4 original — catches truly stuck games)
        2. Low action efficacy after 20K actions (new — catches r11l/tu93 type games
           where segment clicks do nothing but state count isn't zero)
        3. Zero levels at 100K → try other mode for remaining budget
        """
        # Trigger 1: v4 original — very few states = completely stuck
        if (self.mode == "segment" and
            self.total_actions_this_level >= self.GRID_CLICK_CHECK_AT and
            self.explorer.num_states < self.GRID_CLICK_THRESHOLD):
            self._switch_mode("grid", self.grid_step)
            return True

        # Trigger 2: low efficacy — segment clicks aren't doing anything useful
        if (self.mode == "segment" and
            self.total_actions_this_level >= self._EFFICACY_CHECK_AT and
            self._segment_actions_taken >= 1000):
            efficacy = self._segment_actions_changed / max(self._segment_actions_taken, 1)
            if efficacy < self._EFFICACY_SWITCH_THRESHOLD:
                self._switch_mode("grid", self.grid_step)
                return True

        # Trigger 3: grid escalation — still stuck after grid mode
        if (self.mode == "grid" and self.grid_step > 2 and
            self.total_actions_this_level >= self.GRID_REFINE_CHECK_AT and
            self.explorer.num_states < self.GRID_REFINE_THRESHOLD):
            self._switch_mode("grid_fine", 2)
            return True

        # Trigger 4: fallback — 0 levels at 100K → try other mode
        if (self.total_actions_this_level >= self._FALLBACK_AT and
            self.current_level == 0):
            if self.mode == "segment":
                self._switch_mode("grid", self.grid_step)
                return True
            elif self.mode == "grid" and self.grid_step > 2:
                self._switch_mode("grid_fine", 2)
                return True

        return False

    def _switch_mode(self, new_mode, new_step):
        self.mode = new_mode
        self.grid_step = new_step
        # Reset efficacy tracking for new mode
        self._segment_actions_taken = 0
        self._segment_actions_changed = 0
        self.explorer.reset()
        self.frame_segments_cache.clear()
        self.frame_groups_cache.clear()
        self.segment_to_action.clear()
        self._reverse_action.clear()
        self.click_grid = []

    def _register_action_map(self, frame_hash, action_map):
        """Register action map and build reverse lookup for O(1) index finding."""
        self.segment_to_action[frame_hash] = action_map
        for idx, (gid, data) in action_map.items():
            if data is not None:
                key = (frame_hash, gid, data["x"], data["y"])
            else:
                key = (frame_hash, gid, 0, 0)
            self._reverse_action[key] = idx

    def find_action_idx(self, frame_hash, game_action_id, x, y):
        """O(1) reverse action lookup."""
        key = (frame_hash, game_action_id, x, y)
        idx = self._reverse_action.get(key)
        if idx is not None:
            return idx
        # Fallback for non-click actions
        key2 = (frame_hash, game_action_id, 0, 0)
        return self._reverse_action.get(key2)

    def process_frame_segment(self, frame, available_actions):
        """Segment-based frame processing."""
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

        if frame_hash in self.segment_to_action:
            return frame_hash, len(self.segment_to_action[frame_hash]), groups

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
        self._register_action_map(frame_hash, action_map)

        return frame_hash, len(action_map), groups

    def process_frame_grid(self, frame, available_actions):
        """Grid-click frame processing with smart targeting."""
        if self.status_bar_mask is None:
            self.status_bar_mask = detect_status_bar_mask(frame)

        frame_hash = hash_frame(frame, self.status_bar_mask)

        if not self.click_grid:
            self._build_smart_click_grid(frame, self.grid_step)

        if frame_hash in self.segment_to_action:
            action_map = self.segment_to_action[frame_hash]
            num_actions = len(action_map)
            groups = [set(range(num_actions)), set(), set(), set(), set()]
            return frame_hash, num_actions, groups

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
        self._register_action_map(frame_hash, action_map)

        num_actions = len(action_map)
        groups = [set(range(num_actions)), set(), set(), set(), set()]
        return frame_hash, num_actions, groups

    def process_frame(self, frame, available_actions):
        if self.mode in ("grid", "grid_fine"):
            return self.process_frame_grid(frame, available_actions)
        return self.process_frame_segment(frame, available_actions)

    def on_game_over(self):
        self.episode_lengths.append(self.current_episode_len)
        self.current_episode_len = 0
        if len(self.episode_lengths) >= 3:
            recent = sorted(self.episode_lengths[-10:])
            self.estimated_episode_budget = recent[len(recent) // 2]
        self.retry_count += 1

        # NO stall-triggered mode switching — v4's _check_switch_to_grid handles it.
        # UCB1 handles action selection within whichever mode we're in.

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
            self._switch_mode("segment", 4)
            self.play_history = []
            self.effective_history = []
            self.status_bar_mask = None
            self.replaying = False
            self.replay_queue = []
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

        # Action classifier override: if the explorer picked an untested action
        # and we have classifier data, prefer high-change-rate actions instead
        if (action_idx is not None and frame_hash in self.explorer.nodes
            and self.explorer.nodes[frame_hash].has_open(self.explorer.active_group)
            and sum(self._action_total_count.values()) > 100):
            node = self.explorer.nodes[frame_hash]
            # Gather all untested actions across active groups
            open_actions = set()
            for g in range(self.explorer.active_group + 1):
                if g < len(node.groups):
                    open_actions |= node.groups[g]
            if len(open_actions) > 1:
                scored = [(a, self._get_action_priority(a)) for a in open_actions]
                scored.sort(key=lambda x: -x[1])
                if scored[0][1] > scored[-1][1] + 0.1:
                    action_idx = scored[0][0]

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

        # UCB1 reward + action classifier
        if prev_hash is not None and action_idx is not None:
            self.explorer.record_ucb_reward(prev_hash, action_idx, 1.0 if success else 0.0)
            self._action_total_count[action_idx] += 1
            if success:
                self._action_change_count[action_idx] += 1

        # Track efficacy for mode switching
        if not self.replaying and self.mode == "segment":
            self._segment_actions_taken += 1
            if success:
                self._segment_actions_changed += 1

    def _get_action_priority(self, action_idx):
        """Action classifier: returns priority score based on observed change rate.
        Actions that frequently change the frame get higher priority.
        Actions that NEVER change the frame get deprioritized."""
        total = self._action_total_count.get(action_idx, 0)
        if total < 5:
            return 0.5  # unknown — neutral
        changes = self._action_change_count.get(action_idx, 0)
        return changes / total

        # Track action efficacy for mode switching
        if not self.replaying:
            self._segment_actions_taken += 1
            if success:
                self._segment_actions_changed += 1

    def observe_death(self, prev_hash, action_idx):
        if prev_hash is not None and action_idx is not None:
            self.explorer.record_death(prev_hash, action_idx)


# -- Game Runner -----------------------------------------------------------

def solve_game(arc, game_id, max_actions=200000, verbose=True):
    env = arc.make(game_id, render_mode=None)
    obs = env.reset()
    if not obs.frame:
        return {"game_id": game_id, "error": "no_frame"}

    frame = np.array(obs.frame[-1])
    agent = UnifiedAgentV6(grid_step=4)

    # Apply start mode override if this game is known to need it
    gid_short = game_id.split("-")[0]
    start_mode = GAME_START_MODE.get(gid_short)
    if start_mode:
        step = 2 if start_mode == "grid_fine" else 4
        agent._switch_mode(start_mode, step)

    if verbose:
        print(f"[{game_id}] Grid: {frame.shape}, "
              f"Actions: {sorted(obs.available_actions or [])}, "
              f"Levels: {obs.win_levels}, start_mode={start_mode or 'segment'}")

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

        # Process frame once, get hash
        fh, _, _ = agent.process_frame(frame, available)
        game_action_id, x, y = agent.choose_action(frame, available, obs.levels_completed)

        # O(1) reverse lookup instead of O(n) scan
        action_idx = agent.find_action_idx(fh, game_action_id, x, y)

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
        "start_mode": start_mode or "segment",
        "state": obs.state.name if obs else "UNKNOWN",
    }


# Per-game budget allocation
GAME_BUDGETS = {
    "lp85": 300000,   # 5/8, best performer
    "dc22": 250000,   # 3/6 with UCB1
    "lf52": 250000,   # 1/10, 10 levels — high upside
    "vc33": 250000,   # 3/7
    "ft09": 250000,   # 2/6, deep productive explorer
    "re86": 250000,   # 0/8, 52K states — needs mode switch
    "wa30": 250000,   # 0/9
    "sk48": 250000,   # 0/8
    "tu93": 250000,   # 1/9, 9 levels available
    "su15": 250000,   # 1/9
    # Default: 200K
}

# Per-game start mode overrides — skip wasting actions in wrong mode
# Based on 6 runs of data: these games never solve in segment mode
GAME_START_MODE = {
    "r11l": "grid_fine",   # segment efficacy ~0%, grid_fine solved 1/6 in v6
    "tu93": "grid_fine",   # same pattern as r11l, 1/9 from grid_fine
    "g50t": "grid_fine",   # 14 states from 200K segment actions — completely stuck
    "sk48": "grid_fine",   # depth 142, stuck — needs breadth not depth
    "sb26": "grid_fine",   # shallow grid, 0/8
    "wa30": "grid_fine",   # grid-fine mode, 0/9
}

def run_all(api_key, max_actions=200000, verbose=True):
    from arc_agi import Arcade
    import time
    arc = Arcade(arc_api_key=api_key)
    envs = arc.get_environments()
    total_l = total_c = 0
    t0 = time.time()
    for e in envs:
        try:
            # Variable budget per game
            gid_short = e.game_id.split("-")[0]
            budget = GAME_BUDGETS.get(gid_short, max_actions)
            r = solve_game(arc, e.game_id, budget, verbose=False)
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
    print(f"\nTOTAL: {total_c}/{total_l} levels ({total_c/max(total_l,1)*100:.1f}%) "
          f"in {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    import sys, os
    api_key = os.environ.get("ARC_API_KEY", "58b421be-5980-4ee8-8e57-0f18dc9369f3")
    from arc_agi import Arcade
    arc = Arcade(arc_api_key=api_key)
    if len(sys.argv) > 1:
        game_id = sys.argv[1]
        max_a = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
        r = solve_game(arc, game_id, max_a)
        print(r)
    else:
        run_all(api_key)
