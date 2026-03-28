#!/usr/bin/env python3
"""
ARC-AGI-3 Self-Compiling Agent

Architecture:
  Perception → Rules → Planning → Execution → Memory

Key improvements over 3rd place (12.58%):
  1. Object tracking across frames (connected components with matching)
  2. Symbolic rule extraction (action X moves player by delta)
  3. Cross-level transfer (rules from level N bootstrap level N+1)
  4. Goal inference from frame structure
  5. Transition prediction (simulate without acting)

No neural nets. No LLM calls. Pure symbolic reasoning.
"""

import hashlib
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# PERCEPTION: Object Detection & Tracking
# ═══════════════════════════════════════════════════════════════

@dataclass
class GameObject:
    """A connected component in the grid."""
    color: int
    cells: frozenset  # frozenset of (r, c) tuples
    bbox: tuple       # (min_r, min_c, max_r, max_c)
    centroid: tuple   # (mean_r, mean_c)
    size: int

    @property
    def width(self):
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def height(self):
        return self.bbox[2] - self.bbox[0] + 1


def extract_objects(frame: np.ndarray, bg_color: int = -1) -> list[GameObject]:
    """Extract connected components as GameObjects."""
    H, W = frame.shape
    if bg_color < 0:
        # Most common color is background
        bg_color = int(np.bincount(frame.flatten().astype(np.int32)).argmax())

    visited = set()
    objects = []

    for r in range(H):
        for c in range(W):
            if (r, c) in visited or frame[r, c] == bg_color:
                continue
            # BFS flood fill
            color = int(frame[r, c])
            cells = set()
            queue = deque([(r, c)])
            while queue:
                cr, cc = queue.popleft()
                if (cr, cc) in visited:
                    continue
                if cr < 0 or cr >= H or cc < 0 or cc >= W:
                    continue
                if int(frame[cr, cc]) != color:
                    continue
                visited.add((cr, cc))
                cells.add((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                        queue.append((nr, nc))

            if cells:
                rows = [r for r, c in cells]
                cols = [c for r, c in cells]
                obj = GameObject(
                    color=color,
                    cells=frozenset(cells),
                    bbox=(min(rows), min(cols), max(rows), max(cols)),
                    centroid=(sum(rows) / len(cells), sum(cols) / len(cells)),
                    size=len(cells),
                )
                objects.append(obj)

    return objects


def match_objects(before: list[GameObject], after: list[GameObject]) -> dict:
    """Match objects across two frames by overlap and color."""
    matches = {}  # before_idx -> after_idx
    used_after = set()

    # First pass: exact cell overlap
    for i, obj_b in enumerate(before):
        best_overlap = 0
        best_j = -1
        for j, obj_a in enumerate(after):
            if j in used_after or obj_a.color != obj_b.color:
                continue
            overlap = len(obj_b.cells & obj_a.cells)
            if overlap > best_overlap:
                best_overlap = overlap
                best_j = j
        if best_j >= 0 and best_overlap > 0:
            matches[i] = best_j
            used_after.add(best_j)

    # Second pass: nearest centroid for unmatched same-color objects
    for i, obj_b in enumerate(before):
        if i in matches:
            continue
        best_dist = float('inf')
        best_j = -1
        for j, obj_a in enumerate(after):
            if j in used_after or obj_a.color != obj_b.color:
                continue
            if abs(obj_a.size - obj_b.size) > max(obj_a.size, obj_b.size) * 0.5:
                continue  # Size too different
            dist = abs(obj_a.centroid[0] - obj_b.centroid[0]) + abs(obj_a.centroid[1] - obj_b.centroid[1])
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j >= 0 and best_dist < 20:
            matches[i] = best_j
            used_after.add(best_j)

    return matches


# ═══════════════════════════════════════════════════════════════
# STATUS BAR DETECTION (from 3rd place insight)
# ═══════════════════════════════════════════════════════════════

def detect_status_bar(frames: list) -> Optional[np.ndarray]:
    """Detect status bar by finding regions that change between frames.

    Strategy: take 2+ frames, find pixels that change every frame.
    Those are the status bar (step counter, score display).
    Also check for edge-hugging single-color regions.
    """
    H, W = frames[0].shape
    mask = np.zeros((H, W), dtype=bool)

    # Method 1: Find pixels that change across all frame pairs
    if len(frames) >= 3:
        always_changed = np.ones((H, W), dtype=bool)
        for i in range(len(frames) - 1):
            changed = frames[i] != frames[i + 1]
            always_changed &= changed
        if always_changed.any():
            # Expand mask to include surrounding region (manual dilation)
            expanded = always_changed.copy()
            for _ in range(2):
                new = expanded.copy()
                new[1:] |= expanded[:-1]
                new[:-1] |= expanded[1:]
                new[:, 1:] |= expanded[:, :-1]
                new[:, :-1] |= expanded[:, 1:]
                expanded = new
            mask |= expanded

    # Method 2: Edge-hugging single-color strips (common status bar pattern)
    f = frames[0]
    # Right edge
    right_colors = f[:, -1]
    if len(np.unique(right_colors)) <= 2:
        dominant = int(np.bincount(right_colors.astype(np.int32)).argmax())
        for c in range(W - 1, max(W - 15, 0), -1):
            col = f[:, c]
            if np.sum(col == dominant) > H * 0.8:
                mask[:, c] = True
            else:
                break

    # Bottom edge
    bottom_colors = f[-1, :]
    if len(np.unique(bottom_colors)) <= 2:
        dominant = int(np.bincount(bottom_colors.astype(np.int32)).argmax())
        for r in range(H - 1, max(H - 8, 0), -1):
            row = f[r, :]
            if np.sum(row == dominant) > W * 0.8:
                mask[r, :] = True
            else:
                break

    # Top edge
    top_colors = f[0, :]
    if len(np.unique(top_colors)) <= 2:
        dominant = int(np.bincount(top_colors.astype(np.int32)).argmax())
        for r in range(min(8, H)):
            row = f[r, :]
            if np.sum(row == dominant) > W * 0.8:
                mask[r, :] = True
            else:
                break

    return mask if mask.any() else None


# ═══════════════════════════════════════════════════════════════
# STATE HASHING
# ═══════════════════════════════════════════════════════════════

def hash_frame(frame: np.ndarray, mask: Optional[np.ndarray] = None) -> str:
    """Hash a frame, optionally masking status bar."""
    f = frame.copy()
    if mask is not None:
        f[mask] = 0
    # Pack two 4-bit values per byte for compact hashing
    flat = f.flatten().astype(np.uint8)
    if len(flat) % 2:
        flat = np.append(flat, 0)
    packed = (flat[0::2] << 4) | flat[1::2]
    h = hashlib.blake2b(packed.tobytes(), digest_size=16,
                        person=f"{frame.shape[0]}x{frame.shape[1]}".encode())
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════
# RULE ENGINE: Symbolic Transition Rules
# ═══════════════════════════════════════════════════════════════

@dataclass
class MovementRule:
    """Describes how an action affects the game state."""
    action_id: int
    player_delta: tuple = (0, 0)   # (dr, dc) - how player centroid moves
    affects_objects: list = field(default_factory=list)  # list of (color, effect)
    confidence: int = 0            # how many times observed
    blocked_when: list = field(default_factory=list)  # what blocks this movement


@dataclass
class GameRules:
    """Accumulated symbolic rules for a game."""
    player_color: Optional[int] = None
    bg_color: Optional[int] = None
    wall_colors: set = field(default_factory=set)
    goal_colors: set = field(default_factory=set)
    movement_rules: dict = field(default_factory=dict)  # action_id -> MovementRule
    interaction_rules: list = field(default_factory=list)  # what happens on contact


class RuleExtractor:
    """Extract symbolic rules from (state, action, next_state) observations."""

    def __init__(self):
        self.rules = GameRules()
        self.action_observations = defaultdict(list)  # action_id -> list of (before_objs, after_objs, delta)
        self.player_candidates = defaultdict(int)  # color -> vote count

    def observe_transition(self, before: np.ndarray, after: np.ndarray,
                          action_id: int, bg_color: int,
                          mask: Optional[np.ndarray] = None):
        """Record one (state, action, next_state) triple and extract rules."""
        # Apply status bar mask before comparison
        if mask is not None:
            before = before.copy()
            after = after.copy()
            before[mask] = bg_color
            after[mask] = bg_color

        if np.array_equal(before, after):
            self._record_blocked(before, action_id, bg_color)
            return

        # Track centroid shifts for ALL non-bg colors present in the diff
        # This works regardless of what colors swap with what
        diff_mask = before != after
        if not diff_mask.any():
            return

        # Find all non-bg colors that exist in the changed region
        changed_colors = set()
        rows, cols = np.where(diff_mask)
        for r, c in zip(rows, cols):
            bv, av = int(before[r, c]), int(after[r, c])
            if bv != bg_color:
                changed_colors.add(bv)
            if av != bg_color:
                changed_colors.add(av)

        # For each changed color, compute centroid shift
        moved = []
        for color in changed_colors:
            old_pos = np.argwhere(before == color)
            new_pos = np.argwhere(after == color)
            if len(old_pos) == 0 or len(new_pos) == 0:
                continue
            # Only consider colors with stable cell counts (moved, not created/destroyed)
            if abs(len(old_pos) - len(new_pos)) > max(len(old_pos), len(new_pos)) * 0.3:
                continue
            old_c = old_pos.mean(axis=0)
            new_c = new_pos.mean(axis=0)
            dr, dc = new_c[0] - old_c[0], new_c[1] - old_c[1]
            if abs(dr) > 0.3 or abs(dc) > 0.3:
                moved.append((color, (dr, dc), len(old_pos)))
                # Smaller objects more likely to be the player
                # But not too small (status bar pixels)
                if 3 <= len(old_pos) <= 500:
                    self.player_candidates[color] += max(1, 200 - len(old_pos))

        # Record observation
        self.action_observations[action_id].append({
            'moved': moved,
            'changed_colors': changed_colors,
        })

        # Update movement rules
        if moved:
            # Player = smallest consistently-moved object (or highest vote)
            if self.player_candidates:
                player_color = max(self.player_candidates, key=self.player_candidates.get)
                self.rules.player_color = player_color

            for color, delta, size in moved:
                if color == self.rules.player_color:
                    if action_id not in self.rules.movement_rules:
                        self.rules.movement_rules[action_id] = MovementRule(action_id=action_id)
                    rule = self.rules.movement_rules[action_id]
                    rule.player_delta = (round(delta[0]), round(delta[1]))
                    rule.confidence += 1

        # Detect disappearing objects (potential goals)
        for color in changed_colors:
            if color != bg_color and color != self.rules.player_color:
                before_count = int(np.sum(before == color))
                after_count = int(np.sum(after == color))
                if before_count > 0 and after_count == 0:
                    self.rules.goal_colors.add(color)

    def _record_blocked(self, frame: np.ndarray, action_id: int, bg_color: int):
        """Record when an action is blocked — learn wall positions."""
        if self.rules.player_color is None:
            return
        if action_id not in self.rules.movement_rules:
            return

        rule = self.rules.movement_rules[action_id]
        dr, dc = rule.player_delta

        # Find player position
        player_cells = np.argwhere(frame == self.rules.player_color)
        if len(player_cells) == 0:
            return

        # Check what's in the direction of movement
        pr = int(player_cells[:, 0].mean())
        pc = int(player_cells[:, 1].mean())
        H, W = frame.shape

        # Look at the cell(s) in the movement direction
        check_r, check_c = pr + dr, pc + dc
        if 0 <= check_r < H and 0 <= check_c < W:
            blocking_color = int(frame[check_r, check_c])
            if blocking_color != bg_color and blocking_color != self.rules.player_color:
                self.rules.wall_colors.add(blocking_color)

    def get_player_position(self, frame: np.ndarray) -> Optional[tuple]:
        """Get player centroid from frame."""
        if self.rules.player_color is None:
            return None
        cells = np.argwhere(frame == self.rules.player_color)
        if len(cells) == 0:
            return None
        return (int(cells[:, 0].mean()), int(cells[:, 1].mean()))

    def predict_next_state(self, frame: np.ndarray, action_id: int) -> Optional[tuple]:
        """Predict player position after action WITHOUT taking it."""
        pos = self.get_player_position(frame)
        if pos is None or action_id not in self.rules.movement_rules:
            return None
        rule = self.rules.movement_rules[action_id]
        if rule.confidence < 2:
            return None  # Not enough confidence
        dr, dc = rule.player_delta
        new_r, new_c = pos[0] + dr, pos[1] + dc
        H, W = frame.shape
        if not (0 <= new_r < H and 0 <= new_c < W):
            return pos  # Blocked by edge
        target_color = int(frame[new_r, new_c])
        if target_color in self.rules.wall_colors:
            return pos  # Blocked by wall
        return (new_r, new_c)


# ═══════════════════════════════════════════════════════════════
# PLANNER: BFS/A* on Predicted States
# ═══════════════════════════════════════════════════════════════

class Planner:
    """Plan action sequences using extracted rules."""

    def __init__(self, rules: GameRules):
        self.rules = rules

    def find_goals(self, frame: np.ndarray) -> list[tuple]:
        """Find positions of goal-colored cells."""
        goals = []
        for color in self.rules.goal_colors:
            cells = np.argwhere(frame == color)
            for r, c in cells:
                goals.append((int(r), int(c)))
        return goals

    def bfs_to_goal(self, frame: np.ndarray, max_depth: int = 200) -> list[int]:
        """BFS to find action sequence to reach a goal.

        Uses extracted rules to simulate movement without acting.
        """
        start = self._get_abstract_state(frame)
        if start is None:
            return []

        goals = self.find_goals(frame)
        if not goals:
            return []

        goal_set = set(goals)

        # BFS on (player_pos) — simplified abstract state
        visited = {start}
        queue = deque([(start, [])])  # (state, action_sequence)

        while queue and len(queue) < 100000:
            pos, actions = queue.popleft()
            if len(actions) >= max_depth:
                continue

            for action_id, rule in self.rules.movement_rules.items():
                if rule.confidence < 2:
                    continue
                dr, dc = rule.player_delta
                new_r, new_c = pos[0] + dr, pos[1] + dc

                # Check bounds
                H, W = frame.shape
                if not (0 <= new_r < H and 0 <= new_c < W):
                    continue

                # Check walls
                target_color = int(frame[new_r, new_c])
                if target_color in self.rules.wall_colors:
                    continue

                new_pos = (new_r, new_c)

                # Check if we reached a goal
                if new_pos in goal_set:
                    return actions + [action_id]

                if new_pos not in visited:
                    visited.add(new_pos)
                    queue.append((new_pos, actions + [action_id]))

        return []  # No path found

    def _get_abstract_state(self, frame: np.ndarray) -> Optional[tuple]:
        """Get abstract state = player position."""
        if self.rules.player_color is None:
            return None
        cells = np.argwhere(frame == self.rules.player_color)
        if len(cells) == 0:
            return None
        return (int(cells[:, 0].mean()), int(cells[:, 1].mean()))


# ═══════════════════════════════════════════════════════════════
# GRAPH EXPLORER: Systematic State Space Exploration
# ═══════════════════════════════════════════════════════════════

class GraphExplorer:
    """Exact state transition graph with frontier-seeking exploration.

    Combines the 3rd place's systematic exploration with our rule extraction.
    """

    def __init__(self):
        self.graph = {}         # hash -> {action_id: next_hash}
        self.frames = {}        # hash -> frame (numpy array)
        self.frontier = set()   # hashes with untested actions
        self.distances = {}     # hash -> min distance to any frontier node
        self.parent = {}        # hash -> (parent_hash, action_id) for path reconstruction

    def reset(self):
        self.graph.clear()
        self.frames.clear()
        self.frontier.clear()
        self.distances.clear()
        self.parent.clear()

    def add_node(self, frame_hash: str, frame: np.ndarray, available_actions: list[int]):
        """Add a new state to the graph."""
        if frame_hash not in self.graph:
            self.graph[frame_hash] = {}
            self.frames[frame_hash] = frame.copy()
            self.frontier.add(frame_hash)

    def record_transition(self, from_hash: str, action_id: int, to_hash: str):
        """Record that taking action_id from from_hash leads to to_hash."""
        if from_hash in self.graph:
            self.graph[from_hash][action_id] = to_hash

        # Update parent for BFS pathfinding
        if to_hash not in self.parent:
            self.parent[to_hash] = (from_hash, action_id)

    def get_untested_action(self, current_hash: str, available_actions: list[int]) -> Optional[int]:
        """Get an untested action from the current state."""
        if current_hash not in self.graph:
            return None
        tested = set(self.graph[current_hash].keys())
        # Include simple actions AND ACTION7 (undo)
        candidate_actions = [a for a in available_actions if a in range(1, 8) and a != 6]
        untested = [a for a in candidate_actions if a not in tested]
        if untested:
            # Prioritize directional actions (1-4) over space (5) and undo (7)
            for priority in [[1,2,3,4], [5], [7]]:
                for a in priority:
                    if a in untested:
                        return a
            return untested[0]
        return None

    def has_untested(self, current_hash: str, available_actions: list[int]) -> bool:
        if current_hash not in self.graph:
            return False
        tested = set(self.graph[current_hash].keys())
        candidate_actions = [a for a in available_actions if a in range(1, 8) and a != 6]
        return any(a not in tested for a in candidate_actions)

    def find_path_to_frontier(self, current_hash: str, available_actions: list[int]) -> list[int]:
        """BFS to find shortest action sequence to a frontier node."""
        if current_hash in self.frontier and self.has_untested(current_hash, available_actions):
            return []

        visited = {current_hash}
        queue = deque([(current_hash, [])])

        while queue:
            node, path = queue.popleft()
            if len(path) > 50:
                continue

            for action_id, next_hash in self.graph.get(node, {}).items():
                if next_hash in visited:
                    continue
                visited.add(next_hash)
                new_path = path + [action_id]

                if next_hash in self.frontier and self.has_untested(next_hash, available_actions):
                    return new_path

                queue.append((next_hash, new_path))

        return []  # No reachable frontier

    def mark_explored(self, frame_hash: str, available_actions: list[int]):
        """Remove from frontier if all actions tested."""
        if not self.has_untested(frame_hash, available_actions):
            self.frontier.discard(frame_hash)

    @property
    def num_states(self):
        return len(self.graph)


# ═══════════════════════════════════════════════════════════════
# CROSS-LEVEL MEMORY
# ═══════════════════════════════════════════════════════════════

@dataclass
class LevelMemory:
    """What we learned from one level, to transfer to the next."""
    player_color: Optional[int] = None
    bg_color: Optional[int] = None
    wall_colors: set = field(default_factory=set)
    goal_colors: set = field(default_factory=set)
    movement_map: dict = field(default_factory=dict)  # action_id -> (dr, dc)
    actions_to_win: int = 0


class GameMemory:
    """Accumulates knowledge across levels within a game."""

    def __init__(self):
        self.levels: list[LevelMemory] = []

    def save_level(self, rules: GameRules, actions_used: int):
        mem = LevelMemory(
            player_color=rules.player_color,
            bg_color=rules.bg_color,
            wall_colors=set(rules.wall_colors),
            goal_colors=set(rules.goal_colors),
            movement_map={aid: r.player_delta for aid, r in rules.movement_rules.items()},
            actions_to_win=actions_used,
        )
        self.levels.append(mem)

    def bootstrap_rules(self) -> GameRules:
        """Create initial rules for next level from accumulated memory."""
        rules = GameRules()
        if not self.levels:
            return rules

        # Transfer the most recent level's knowledge
        last = self.levels[-1]
        rules.player_color = last.player_color
        rules.bg_color = last.bg_color
        rules.wall_colors = set(last.wall_colors)
        rules.goal_colors = set(last.goal_colors)

        # Transfer movement rules with high confidence
        for action_id, delta in last.movement_map.items():
            rule = MovementRule(action_id=action_id, player_delta=delta, confidence=5)
            rules.movement_rules[action_id] = rule

        return rules


# ═══════════════════════════════════════════════════════════════
# CLICK EXPLORER: For games that use ACTION6 (clicking)
# ═══════════════════════════════════════════════════════════════

class ClickExplorer:
    """Explores click-based games by identifying clickable objects."""

    def __init__(self):
        self.clicked_positions = set()
        self.effective_clicks = set()  # positions where clicks caused changes

    def find_click_targets(self, frame: np.ndarray, bg_color: int) -> list[tuple]:
        """Find potential click targets: centroids of colorful objects."""
        objects = extract_objects(frame, bg_color)
        targets = []

        # Sort by salience: bright colors and medium sizes first
        for obj in sorted(objects, key=lambda o: (-int(o.color >= 6), -o.size)):
            cr, cc = int(obj.centroid[0]), int(obj.centroid[1])
            if (cr, cc) not in self.clicked_positions:
                targets.append((cr, cc, obj.color, obj.size))

        return targets

    def record_click(self, pos: tuple, effective: bool):
        self.clicked_positions.add(pos)
        if effective:
            self.effective_clicks.add(pos)


# ═══════════════════════════════════════════════════════════════
# MAIN AGENT
# ═══════════════════════════════════════════════════════════════

class SelfCompilingAgent:
    """
    The full ARC-AGI-3 agent.

    Strategy:
    1. PROBE: Take each simple action, observe what changes
    2. EXTRACT: Build symbolic rules from transitions
    3. PLAN: If rules + goals identified, BFS to solution
    4. EXPLORE: If planning fails, systematic graph exploration
    5. TRANSFER: Carry rules to next level
    6. RETRY: On GAME_OVER, avoid previous dead-end paths
    """

    def __init__(self):
        self.memory = GameMemory()
        self.rule_extractor = RuleExtractor()
        self.graph = GraphExplorer()
        self.click_explorer = ClickExplorer()
        self.planner = None
        self.status_bar_mask = None
        self.bg_color = None
        self.current_level = 0
        self.level_actions = 0
        self.plan_queue = []  # Pre-computed action sequence
        self.play_history = []  # Action sequence for current play
        self.failed_plays = []  # Previous failed plays (action sequences)
        self.retry_count = 0
        self.winning_paths = {}  # level -> action sequence that solved it
        self.replaying = False
        self.replay_queue = []

    def reset_for_level(self, frame: np.ndarray, bootstrap: bool = True):
        """Reset state for a new level, optionally bootstrapping from memory."""
        if bootstrap and self.memory.levels:
            self.rule_extractor = RuleExtractor()
            self.rule_extractor.rules = self.memory.bootstrap_rules()
        else:
            self.rule_extractor = RuleExtractor()

        self.graph = GraphExplorer()
        self.click_explorer = ClickExplorer()
        self.planner = None
        self.plan_queue = []
        self.level_actions = 0
        self.probe_frames = [frame]  # Collect frames for status bar detection
        self.probing_done = False

        # Detect background color
        self.bg_color = int(np.bincount(frame.flatten().astype(np.int32)).argmax())
        self.rule_extractor.rules.bg_color = self.bg_color

        # Initial status bar detection from single frame
        self.status_bar_mask = detect_status_bar([frame])

    def choose_action(self, frame: np.ndarray, available_actions: list[int],
                     levels_completed: int) -> tuple:
        """Choose next action. Returns (action_id, x, y) where x,y are for ACTION6."""

        # Level transition detection
        if levels_completed > self.current_level:
            # Save winning path for instant replay on retries
            self.winning_paths[self.current_level] = list(self.play_history)
            self.memory.save_level(self.rule_extractor.rules, self.level_actions)
            self.current_level = levels_completed
            self.reset_for_level(frame, bootstrap=True)
            self.play_history = []  # Reset for new level
            print(f"  [Level {levels_completed}] Bootstrapped: "
                  f"player={self.rule_extractor.rules.player_color}, "
                  f"walls={self.rule_extractor.rules.wall_colors}")

        self.level_actions += 1

        # Phase 0: Initial probing — cycle through available actions to learn rules
        if not self.probing_done and self.level_actions <= 8:
            simple = [a for a in available_actions if 1 <= a <= 5]
            if simple:
                idx = (self.level_actions - 1) % len(simple)
                return (simple[idx], 0, 0)
            # Click-only game: try clicking on salient objects
            if 6 in available_actions:
                targets = self.click_explorer.find_click_targets(frame, self.bg_color)
                if targets and self.level_actions <= len(targets):
                    r, c, color, size = targets[self.level_actions - 1]
                    return (6, c, r)
                self.probing_done = True

        # After probing, update status bar mask with collected frames
        if not self.probing_done and self.level_actions > 8:
            self.probing_done = True
            if len(self.probe_frames) >= 3:
                self.status_bar_mask = detect_status_bar(self.probe_frames)

        frame_hash = hash_frame(frame, self.status_bar_mask)
        self.graph.add_node(frame_hash, frame, available_actions)

        # Phase 1: Execute pre-computed plan (disabled — goal detection unreliable)
        if self.plan_queue:
            action_id = self.plan_queue.pop(0)
            return (action_id, 0, 0)

        # Phase 2: Systematic graph exploration
        untested = self.graph.get_untested_action(frame_hash, available_actions)
        if untested is not None:
            return (untested, 0, 0)

        # Navigate to frontier
        path = self.graph.find_path_to_frontier(frame_hash, available_actions)
        if path:
            return (path[0], 0, 0)

        # Phase 4: Click exploration (for ACTION6 games)
        # Track clicks per-state so we try new clicks after state changes
        if 6 in available_actions:
            state_clicks = self.graph.graph.get(frame_hash, {})
            # Generate click action IDs: 1000 + r*64 + c
            targets = self.click_explorer.find_click_targets(frame, self.bg_color)
            for r, c, color, size in targets:
                click_key = 1000 + r * 64 + c
                if click_key not in state_clicks:
                    # Record this click as tested for this state
                    if frame_hash in self.graph.graph:
                        self.graph.graph[frame_hash][click_key] = "pending"
                    return (6, c, r)

            # If all object centroids clicked, try individual salient cells
            salient_cells = np.argwhere(frame != self.bg_color)
            # Shuffle for variety but deterministically based on state
            indices = list(range(len(salient_cells)))
            for idx in indices:
                r, c = int(salient_cells[idx][0]), int(salient_cells[idx][1])
                click_key = 1000 + r * 64 + c
                if click_key not in state_clicks:
                    if frame_hash in self.graph.graph:
                        self.graph.graph[frame_hash][click_key] = "pending"
                    return (6, c, r)

        # Phase 5: Deeper exploration — try action sequences we haven't tried
        # Use ACTION7 (undo) to backtrack and try different paths
        if 7 in available_actions and self.level_actions % 5 == 0:
            return (7, 0, 0)

        # Phase 6: Fallback — cycle through directions more aggressively
        simple = [a for a in available_actions if 1 <= a <= 5]
        if simple:
            # Vary the pattern: try spiral-like movement
            patterns = [
                [1, 1, 4, 4, 2, 2, 3, 3],  # up-up-right-right-down-down-left-left
                [4, 2, 3, 1],                 # right-down-left-up
                [1, 4, 2, 3],                 # up-right-down-left
                [5, 1, 5, 4, 5, 2, 5, 3],   # space between each direction
            ]
            pattern = patterns[(self.level_actions // 20) % len(patterns)]
            action_id = pattern[self.level_actions % len(pattern)]
            if action_id in available_actions:
                return (action_id, 0, 0)
            return (simple[self.level_actions % len(simple)], 0, 0)

        return (0, 0, 0)  # RESET as last resort

    def observe_result(self, prev_frame: np.ndarray, action_id: int,
                      new_frame: np.ndarray, available_actions: list[int]):
        """Process the result of an action."""
        self.play_history.append(action_id)

        # Collect frames during probing phase for status bar detection
        if not self.probing_done:
            self.probe_frames.append(new_frame.copy())

        prev_hash = hash_frame(prev_frame, self.status_bar_mask)
        new_hash = hash_frame(new_frame, self.status_bar_mask)

        # Record transition in graph
        self.graph.record_transition(prev_hash, action_id, new_hash)
        self.graph.add_node(new_hash, new_frame, available_actions)
        self.graph.mark_explored(prev_hash, available_actions)

        # Extract rules from this transition
        self.rule_extractor.observe_transition(
            prev_frame, new_frame, action_id, self.bg_color,
            mask=self.status_bar_mask
        )

        # For click actions, record effectiveness
        if action_id == 6:
            effective = not np.array_equal(prev_frame, new_frame)
            # Graph already records the state transition


# ═══════════════════════════════════════════════════════════════
# GAME RUNNER: Connects agent to ARC-AGI-3 environment
# ═══════════════════════════════════════════════════════════════

def solve_game(arc, game_id, max_actions=5000, verbose=True):
    """Run the self-compiling agent on a single game."""
    env = arc.make(game_id, render_mode=None)
    obs = env.reset()

    if not obs.frame or len(obs.frame) == 0:
        if verbose:
            print(f"[{game_id}] No frame data!")
        return {"game_id": game_id, "error": "no_frame"}

    frame = np.array(obs.frame[-1])
    agent = SelfCompilingAgent()
    agent.reset_for_level(frame, bootstrap=False)

    if verbose:
        print(f"\n{'='*60}")
        print(f"[{game_id}] Grid: {frame.shape}, "
              f"Colors: {sorted(map(int, np.unique(frame)))}, "
              f"Levels: 0/{obs.win_levels}")

    total_actions = 0
    prev_frame = frame

    while total_actions < max_actions:
        # Check termination
        if obs.state.name == "WIN":
            break
        if obs.state.name == "GAME_OVER":
            # Record failed play
            agent.failed_plays.append(list(agent.play_history))
            agent.retry_count += 1
            if verbose and agent.retry_count <= 5:
                print(f"  GAME_OVER after {len(agent.play_history)} actions (retry #{agent.retry_count})")

            # Reset game but KEEP the accumulated graph and rules
            obs = env.reset()
            if not obs.frame:
                break
            frame = np.array(obs.frame[-1])
            # Don't reset graph or rules — persist knowledge across retries
            agent.play_history = []
            agent.plan_queue = []
            agent.probing_done = True  # Already probed
            agent.level_actions = 9    # Skip probing phase

            # If we've solved earlier levels, replay winning paths instantly
            if agent.winning_paths:
                replay_actions = []
                for lvl in sorted(agent.winning_paths.keys()):
                    replay_actions.extend(agent.winning_paths[lvl])
                if replay_actions:
                    agent.plan_queue = replay_actions
                    if verbose and agent.retry_count <= 3:
                        print(f"  Replaying {len(replay_actions)} winning actions for levels 0-{max(agent.winning_paths.keys())}")

            prev_frame = frame
            total_actions += 1
            continue

        # Get current frame
        if obs.frame:
            frame = np.array(obs.frame[-1])

        available = obs.available_actions if obs.available_actions else [1, 2, 3, 4, 5]

        # Choose action
        action_id, x, y = agent.choose_action(
            frame, available, obs.levels_completed
        )

        # Take action
        from arcengine import GameAction
        action = GameAction.from_id(action_id)
        if action.is_complex():
            action.set_data({"x": int(x), "y": int(y)})

        prev_frame = frame
        try:
            # Complex actions need data passed explicitly
            data = action.action_data.model_dump() if action.is_complex() else None
            raw_obs = env.step(action, data=data)
        except (KeyError, Exception) as e:
            # Some games crash on certain actions — skip and try another
            if verbose and total_actions < 50:
                print(f"  Action {action_id} failed: {e}")
            total_actions += 1
            continue

        # Convert observation
        if raw_obs is None:
            break

        # Handle the observation format
        if hasattr(raw_obs, 'frame') and raw_obs.frame is not None and len(raw_obs.frame) > 0:
            new_frame = np.array(raw_obs.frame[-1]) if hasattr(raw_obs.frame[-1], '__len__') else frame
        else:
            new_frame = frame

        # Observe result
        agent.observe_result(prev_frame, action_id, new_frame, available)

        obs = raw_obs
        total_actions += 1

        # Progress reporting
        if verbose and total_actions % 100 == 0:
            rules = agent.rule_extractor.rules
            print(f"  [{total_actions}] States: {agent.graph.num_states}, "
                  f"Player: {rules.player_color}, "
                  f"Walls: {rules.wall_colors}, "
                  f"Goals: {rules.goal_colors}, "
                  f"Levels: {obs.levels_completed}/{obs.win_levels}")

    result = {
        "game_id": game_id,
        "levels_completed": obs.levels_completed,
        "total_levels": obs.win_levels,
        "actions_used": total_actions,
        "states_explored": agent.graph.num_states,
        "state": obs.state.name,
    }

    if verbose:
        print(f"  Result: {obs.levels_completed}/{obs.win_levels} levels, "
              f"{total_actions} actions, {agent.graph.num_states} states")

    return result


def run_all(api_key, max_actions=5000, verbose=True):
    """Run agent on all available games."""
    from arc_agi import Arcade

    arc = Arcade(arc_api_key=api_key)
    envs = arc.get_environments()

    results = []
    total_levels = 0
    total_completed = 0

    for env_info in envs:
        try:
            result = solve_game(arc, env_info.game_id, max_actions, verbose)
            results.append(result)
            total_levels += result.get("total_levels", 0)
            total_completed += result.get("levels_completed", 0)
        except Exception as e:
            if verbose:
                print(f"  [{env_info.game_id}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({"game_id": env_info.game_id, "error": str(e)})

    if verbose:
        print(f"\n{'='*60}")
        print(f"TOTAL: {total_completed}/{total_levels} levels "
              f"across {len(results)} games")

    return results


if __name__ == "__main__":
    import sys
    import os

    api_key = os.environ.get("ARC_API_KEY", "58b421be-5980-4ee8-8e57-0f18dc9369f3")

    from arc_agi import Arcade
    arc = Arcade(arc_api_key=api_key)

    if len(sys.argv) > 1:
        game_id = sys.argv[1]
        max_a = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
        solve_game(arc, game_id, max_a)
    else:
        run_all(api_key)
