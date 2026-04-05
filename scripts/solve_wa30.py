"""
Solver for WA30 - Sokoban-like game with autonomous NPCs.

Strategy per level:
- L0: Pure Sokoban BFS (no NPCs)
- L1-4: kd NPCs help carry boxes to targets. Simulate with forward search.
- L5-6: ys NPCs try to steal boxes. BFS with NPC simulation or destroy NPC first.
- L7-8: Mixed NPCs. Complex.
"""

import json
import time
import sys
import heapq
from collections import deque

CELL = 4

# Sprite pixel dimensions (width x height)
SPRITE_SIZES = {
    "aidclcbjcv": (64, 20), "byigobxzpg": (4, 4), "cwefnfvjhr": (64, 16),
    "doijajrgdi": (8, 12), "geffskzhqq": (4, 8), "ghklglzjuf": (8, 8),
    "jigtxgzhwt": (12, 4), "jqzhxgbmtz": (4, 4), "ktghqrydvd": (8, 16),
    "ofwegeqknn": (4, 8), "ooaamfpvqr": (8, 8), "peimznrlqd": (12, 12),
    "pktgsotzmw": (4, 4), "pmargquscu": (4, 4), "uasmnkbzmm": (4, 4),
    "vikkhnsrzd": (16, 8), "wkmuwhjqyo": (8, 4), "wppuejnwhl": (4, 4),
    "xqaqifquaw": (12, 12), "xxmzyqktqy": (4, 4),
}
SPRITE_TAGS = {
    "aidclcbjcv": [], "byigobxzpg": ["kdweefinfi"], "cwefnfvjhr": [],
    "doijajrgdi": ["fsjjayjoeg"], "geffskzhqq": ["zqxwgacnue"],
    "ghklglzjuf": ["fsjjayjoeg"], "jigtxgzhwt": ["fsjjayjoeg"],
    "jqzhxgbmtz": ["ysysltqlke"], "ktghqrydvd": ["fsjjayjoeg"],
    "ofwegeqknn": ["fsjjayjoeg"], "ooaamfpvqr": ["zqxwgacnue"],
    "peimznrlqd": ["fsjjayjoeg"], "pktgsotzmw": ["geezpjgiyd"],
    "pmargquscu": ["bnzklblgdk"], "uasmnkbzmm": ["debyzcmtnr"],
    "vikkhnsrzd": ["fsjjayjoeg"], "wkmuwhjqyo": ["fsjjayjoeg"],
    "wppuejnwhl": ["wbmdvjhthc"], "xqaqifquaw": ["zqxwgacnue"],
    "xxmzyqktqy": ["fsjjayjoeg"],
}
SPRITE_COLLIDABLE = {
    "aidclcbjcv": True, "byigobxzpg": True, "cwefnfvjhr": True,
    "doijajrgdi": False, "geffskzhqq": False, "ghklglzjuf": False,
    "jigtxgzhwt": False, "jqzhxgbmtz": True, "ktghqrydvd": False,
    "ofwegeqknn": False, "ooaamfpvqr": False, "peimznrlqd": False,
    "pktgsotzmw": True, "pmargquscu": False, "uasmnkbzmm": True,
    "vikkhnsrzd": False, "wkmuwhjqyo": False, "wppuejnwhl": True,
    "xqaqifquaw": False, "xxmzyqktqy": False,
}

LEVEL_DEFS = [
    # Level 0
    [("jigtxgzhwt",28,28),("pktgsotzmw",44,24),("pktgsotzmw",16,28),("pktgsotzmw",32,36),("wppuejnwhl",32,48)],
    # Level 1
    [("byigobxzpg",24,36),("doijajrgdi",12,28),("pktgsotzmw",48,32),("pktgsotzmw",36,28),("pktgsotzmw",40,20),("pktgsotzmw",48,24),("pktgsotzmw",44,40),("wppuejnwhl",12,8)],
    # Level 2
    [("byigobxzpg",48,12),("ktghqrydvd",52,24),("pktgsotzmw",32,32),("pktgsotzmw",20,20),("pktgsotzmw",12,44),("pktgsotzmw",8,16),("pktgsotzmw",32,12),
     ("pmargquscu",32,0),("pmargquscu",32,4),("pmargquscu",32,8),("pmargquscu",32,12),("pmargquscu",32,16),("pmargquscu",32,20),("pmargquscu",32,24),("pmargquscu",32,28),
     ("pmargquscu",32,32),("pmargquscu",32,36),("pmargquscu",32,40),("pmargquscu",32,44),("pmargquscu",32,48),("pmargquscu",32,52),("pmargquscu",32,56),("pmargquscu",32,60),
     ("wppuejnwhl",16,36)],
    # Level 3
    [("byigobxzpg",56,4),("byigobxzpg",8,12),("byigobxzpg",24,56),("ofwegeqknn",4,24),
     ("pktgsotzmw",32,36),("pktgsotzmw",24,24),("pktgsotzmw",36,40),("pktgsotzmw",24,40),("pktgsotzmw",32,24),("pktgsotzmw",36,24),("pktgsotzmw",24,4),
     ("pmargquscu",28,20),("pmargquscu",24,20),("pmargquscu",20,20),("pmargquscu",20,24),("pmargquscu",20,28),("pmargquscu",20,32),("pmargquscu",20,36),("pmargquscu",20,40),("pmargquscu",20,44),
     ("pmargquscu",32,20),("pmargquscu",36,20),("pmargquscu",40,20),("pmargquscu",40,24),("pmargquscu",24,44),("pmargquscu",28,44),("pmargquscu",32,44),("pmargquscu",36,44),("pmargquscu",40,44),
     ("pmargquscu",40,40),("pmargquscu",40,36),("pmargquscu",40,32),("pmargquscu",40,28),
     ("uasmnkbzmm",16,48),("uasmnkbzmm",12,52),("uasmnkbzmm",8,56),("uasmnkbzmm",4,60),("uasmnkbzmm",44,48),("uasmnkbzmm",48,52),("uasmnkbzmm",52,56),("uasmnkbzmm",56,60),
     ("uasmnkbzmm",28,4),("uasmnkbzmm",28,0),("uasmnkbzmm",28,8),("uasmnkbzmm",28,12),("uasmnkbzmm",28,16),
     ("wkmuwhjqyo",36,56),("wppuejnwhl",28,32),("xxmzyqktqy",8,36),("xxmzyqktqy",52,28),("xxmzyqktqy",56,20)],
    # Level 4
    [("byigobxzpg",20,28),("ktghqrydvd",8,24),
     ("pktgsotzmw",52,48),("pktgsotzmw",60,52),("pktgsotzmw",48,4),("pktgsotzmw",56,8),("pktgsotzmw",44,56),("pktgsotzmw",44,28),
     ("uasmnkbzmm",24,24),("uasmnkbzmm",28,24),("uasmnkbzmm",32,24),("uasmnkbzmm",36,24),("uasmnkbzmm",36,20),("uasmnkbzmm",36,16),("uasmnkbzmm",36,12),("uasmnkbzmm",36,8),
     ("uasmnkbzmm",36,4),("uasmnkbzmm",36,0),("uasmnkbzmm",36,60),("uasmnkbzmm",36,56),("uasmnkbzmm",36,52),("uasmnkbzmm",36,48),("uasmnkbzmm",36,44),("uasmnkbzmm",36,40),
     ("uasmnkbzmm",36,36),("uasmnkbzmm",24,36),("uasmnkbzmm",28,36),("uasmnkbzmm",32,36),("wppuejnwhl",44,36)],
    # Level 5
    [("ghklglzjuf",28,12),("jqzhxgbmtz",16,16),("ooaamfpvqr",52,24),
     ("pktgsotzmw",56,28),("pktgsotzmw",28,16),
     ("uasmnkbzmm",44,0),("uasmnkbzmm",44,4),("uasmnkbzmm",44,8),("uasmnkbzmm",44,12),("uasmnkbzmm",44,16),("uasmnkbzmm",44,20),
     ("uasmnkbzmm",48,0),("uasmnkbzmm",48,4),("uasmnkbzmm",48,8),("uasmnkbzmm",48,12),("uasmnkbzmm",48,16),("uasmnkbzmm",48,20),
     ("uasmnkbzmm",44,28),("uasmnkbzmm",44,32),("uasmnkbzmm",44,36),("uasmnkbzmm",44,40),("uasmnkbzmm",44,44),("uasmnkbzmm",44,48),
     ("uasmnkbzmm",48,28),("uasmnkbzmm",48,32),("uasmnkbzmm",48,36),("uasmnkbzmm",48,40),("uasmnkbzmm",48,44),("uasmnkbzmm",48,48),
     ("uasmnkbzmm",44,52),("uasmnkbzmm",44,56),("uasmnkbzmm",48,52),("uasmnkbzmm",48,56),("uasmnkbzmm",44,60),("uasmnkbzmm",48,60),
     ("wppuejnwhl",20,52)],
    # Level 6
    [("aidclcbjcv",0,44),("cwefnfvjhr",0,0),("geffskzhqq",48,28),("jqzhxgbmtz",40,32),("ofwegeqknn",12,28),
     ("pktgsotzmw",32,24),("pktgsotzmw",28,32),
     ("uasmnkbzmm",0,16),("uasmnkbzmm",4,16),("uasmnkbzmm",8,16),("uasmnkbzmm",12,16),("uasmnkbzmm",16,16),("uasmnkbzmm",20,16),("uasmnkbzmm",24,16),("uasmnkbzmm",28,16),
     ("uasmnkbzmm",32,16),("uasmnkbzmm",36,16),("uasmnkbzmm",40,16),("uasmnkbzmm",44,16),("uasmnkbzmm",48,16),("uasmnkbzmm",52,16),("uasmnkbzmm",56,16),("uasmnkbzmm",60,16),
     ("uasmnkbzmm",0,40),("uasmnkbzmm",4,40),("uasmnkbzmm",8,40),("uasmnkbzmm",12,40),("uasmnkbzmm",16,40),("uasmnkbzmm",20,40),("uasmnkbzmm",24,40),("uasmnkbzmm",28,40),
     ("uasmnkbzmm",32,40),("uasmnkbzmm",36,40),("uasmnkbzmm",40,40),("uasmnkbzmm",44,40),("uasmnkbzmm",48,40),("uasmnkbzmm",52,40),("uasmnkbzmm",56,40),("uasmnkbzmm",60,40),
     ("wppuejnwhl",20,32)],
    # Level 7
    [("byigobxzpg",28,4),("byigobxzpg",36,44),("jqzhxgbmtz",32,16),("jqzhxgbmtz",32,56),("peimznrlqd",48,48),
     ("pktgsotzmw",32,12),("pktgsotzmw",28,8),("pktgsotzmw",40,52),("pktgsotzmw",36,48),("pktgsotzmw",28,52),("pktgsotzmw",32,52),("pktgsotzmw",36,8),("pktgsotzmw",24,12),
     ("pktgsotzmw",4,8),("pktgsotzmw",8,12),("pktgsotzmw",8,8),("pktgsotzmw",4,12),("pktgsotzmw",24,44),
     ("uasmnkbzmm",24,24),("uasmnkbzmm",28,24),("uasmnkbzmm",8,24),("uasmnkbzmm",12,24),("uasmnkbzmm",0,24),("uasmnkbzmm",4,24),
     ("uasmnkbzmm",32,24),("uasmnkbzmm",36,24),("uasmnkbzmm",40,24),("uasmnkbzmm",44,24),("uasmnkbzmm",48,24),("uasmnkbzmm",52,24),("uasmnkbzmm",56,24),("uasmnkbzmm",60,24),
     ("uasmnkbzmm",44,36),("uasmnkbzmm",48,36),("uasmnkbzmm",52,36),("uasmnkbzmm",56,36),("uasmnkbzmm",60,36),
     ("uasmnkbzmm",0,36),("uasmnkbzmm",4,36),("uasmnkbzmm",8,36),("uasmnkbzmm",12,36),("uasmnkbzmm",16,36),("uasmnkbzmm",20,36),("uasmnkbzmm",24,36),("uasmnkbzmm",28,36),("uasmnkbzmm",32,36),
     ("vikkhnsrzd",44,8),("wppuejnwhl",4,32),("xqaqifquaw",4,8),("xqaqifquaw",12,48)],
    # Level 8
    [("byigobxzpg",16,28),("byigobxzpg",44,4),("jqzhxgbmtz",60,56),("ooaamfpvqr",4,28),("peimznrlqd",20,12),
     ("pktgsotzmw",12,20),("pktgsotzmw",8,28),("pktgsotzmw",4,20),("pktgsotzmw",4,32),("pktgsotzmw",56,32),("pktgsotzmw",44,20),("pktgsotzmw",8,12),("pktgsotzmw",48,28),("pktgsotzmw",4,28),
     ("pmargquscu",52,12),("pmargquscu",40,12),("pmargquscu",56,12),("pmargquscu",60,12),("pmargquscu",48,12),("pmargquscu",44,12),
     ("uasmnkbzmm",40,40),("uasmnkbzmm",36,40),("uasmnkbzmm",32,40),("uasmnkbzmm",56,40),("uasmnkbzmm",52,40),("uasmnkbzmm",48,40),("uasmnkbzmm",44,40),("uasmnkbzmm",60,40),
     ("uasmnkbzmm",32,44),("uasmnkbzmm",32,48),("uasmnkbzmm",40,56),("uasmnkbzmm",40,60),("uasmnkbzmm",48,44),("uasmnkbzmm",48,48),("uasmnkbzmm",56,48),("uasmnkbzmm",56,52),
     ("uasmnkbzmm",24,56),("uasmnkbzmm",24,60),("uasmnkbzmm",24,52),("uasmnkbzmm",24,48),("uasmnkbzmm",56,56),("uasmnkbzmm",56,60),("uasmnkbzmm",48,52),("uasmnkbzmm",48,56),
     ("uasmnkbzmm",40,48),("uasmnkbzmm",40,52),("uasmnkbzmm",32,56),("uasmnkbzmm",32,52),("uasmnkbzmm",24,40),("uasmnkbzmm",28,40),("uasmnkbzmm",16,40),("uasmnkbzmm",20,40),
     ("uasmnkbzmm",16,44),("uasmnkbzmm",16,48),("uasmnkbzmm",16,56),("uasmnkbzmm",16,52),
     ("uasmnkbzmm",36,0),("uasmnkbzmm",36,4),("uasmnkbzmm",36,8),("uasmnkbzmm",36,16),("uasmnkbzmm",36,12),("uasmnkbzmm",36,20),("uasmnkbzmm",36,24),("uasmnkbzmm",36,28),
     ("uasmnkbzmm",8,60),("uasmnkbzmm",8,56),("uasmnkbzmm",8,52),("uasmnkbzmm",8,48),("uasmnkbzmm",8,44),("uasmnkbzmm",8,40),
     ("wkmuwhjqyo",52,24),("wkmuwhjqyo",52,8),("wppuejnwhl",32,32)],
]

STEP_LIMITS = [200, 70, 100, 100, 125, 75, 125, 150, 70]


def parse_level(level_idx):
    """Parse level into abstract state."""
    level_def = LEVEL_DEFS[level_idx]

    static_walls = set()  # permanent collidable positions (walls, background anchors)
    barriers = set()      # bnzklblgdk blocks
    targets = set()       # wyzquhjerd pixel positions (fsjjayjoeg)
    alt_targets = set()   # lqctaojiby pixel positions (zqxwgacnue)

    player_pos = None
    box_positions = []
    npc_kd = []
    npc_ys = []

    for sprite_name, x, y in level_def:
        tags = SPRITE_TAGS[sprite_name]
        w, h = SPRITE_SIZES[sprite_name]

        if "wbmdvjhthc" in tags:
            player_pos = (x, y)
        elif "geezpjgiyd" in tags:
            box_positions.append((x, y))
        elif "kdweefinfi" in tags:
            npc_kd.append((x, y))
        elif "ysysltqlke" in tags:
            npc_ys.append((x, y))
        elif SPRITE_COLLIDABLE[sprite_name]:
            static_walls.add((x, y))

        if "fsjjayjoeg" in tags:
            for dy in range(h):
                for dx in range(w):
                    targets.add((x + dx, y + dy))
        if "zqxwgacnue" in tags:
            for dy in range(h):
                for dx in range(w):
                    alt_targets.add((x + dx, y + dy))
        if "bnzklblgdk" in tags:
            barriers.add((x, y))

    # Border walls
    for i in range(0, 64, CELL):
        static_walls.add((-CELL, i))
        static_walls.add((64, i))
        static_walls.add((i, -CELL))
        static_walls.add((i, 64))

    return {
        'static_walls': frozenset(static_walls),
        'barriers': frozenset(barriers),
        'targets': frozenset(targets),
        'alt_targets': frozenset(alt_targets),
        'player': player_pos,
        'boxes': list(box_positions),
        'npc_kd': list(npc_kd),
        'npc_ys': list(npc_ys),
        'step_limit': STEP_LIMITS[level_idx],
    }


def adjacent(pos):
    x, y = pos
    return [(x-CELL, y), (x+CELL, y), (x, y-CELL), (x, y+CELL)]


def rotation_from_delta(dx, dy):
    if dy < 0: return 0
    if dx > 0: return 90
    if dy > 0: return 180
    return 270


def facing_pos(pos, rot):
    x, y = pos
    if rot == 0: return (x, y-CELL)
    if rot == 180: return (x, y+CELL)
    if rot == 90: return (x+CELL, y)
    return (x-CELL, y)


class GameSim:
    """Exact simulation of wa30 game logic."""

    def __init__(self, level_data):
        self.static_walls = level_data['static_walls']
        self.barriers = level_data['barriers']
        self.targets = level_data['targets']
        self.alt_targets = level_data['alt_targets']

        # Mutable state
        self.player = level_data['player']
        self.player_rot = 0
        self.boxes = list(level_data['boxes'])  # list of (x,y)
        self.npc_kd = list(level_data['npc_kd'])
        self.npc_ys = list(level_data['npc_ys'])

        # Holdings: maps holder -> box_index, and reverse
        # nsevyuople: holder_sprite -> held_box_sprite
        # zmqreragji: box_sprite -> holder_sprite
        # We represent: holder = ('player',) or ('kd', idx) or ('ys', idx)
        self.holder_to_box = {}  # holder_key -> box_index
        self.box_to_holder = {}  # box_index -> holder_key

        self.steps_left = level_data['step_limit']
        self.won = False
        self.lost = False

        # pkbufziase = all collidable positions
        self._rebuild_collidable()

    def _rebuild_collidable(self):
        self.collidable = set(self.static_walls)
        self.collidable.add(self.player)
        for b in self.boxes:
            self.collidable.add(b)
        for n in self.npc_kd:
            self.collidable.add(n)
        for n in self.npc_ys:
            self.collidable.add(n)

    def _is_free(self, pos):
        """kblzhbvysd: not in collidable and not in barriers."""
        return pos not in self.collidable and pos not in self.barriers

    def _is_on_target(self, pos):
        return pos in self.targets

    def _is_on_alt_target(self, pos):
        return pos in self.alt_targets

    def _move_entity(self, entity_type, entity_idx, new_x, new_y):
        """Move an entity (updating collidable set) with held box if applicable."""
        holder_key = (entity_type, entity_idx) if entity_type != 'player' else ('player',)

        if holder_key in self.holder_to_box:
            held_idx = self.holder_to_box[holder_key]
            old_pos = self._get_entity_pos(entity_type, entity_idx)
            held_box = self.boxes[held_idx]
            dx_hold = held_box[0] - old_pos[0]
            dy_hold = held_box[1] - old_pos[1]
            new_held = (new_x + dx_hold, new_y + dy_hold)

            # Check fuykgiiwit
            if not self._can_move_pair(entity_type, entity_idx, held_idx, (new_x, new_y)):
                return

            # Do the move
            self.collidable.discard(old_pos)
            self.collidable.discard(held_box)
            self._set_entity_pos(entity_type, entity_idx, new_x, new_y)
            self.boxes[held_idx] = new_held
            self.collidable.add((new_x, new_y))
            self.collidable.add(new_held)
        else:
            if self._is_free((new_x, new_y)):
                old_pos = self._get_entity_pos(entity_type, entity_idx)
                self.collidable.discard(old_pos)
                self._set_entity_pos(entity_type, entity_idx, new_x, new_y)
                self.collidable.add((new_x, new_y))

    def _can_move_pair(self, entity_type, entity_idx, held_box_idx, new_pos):
        """fuykgiiwit: can entity+held_box move to new_pos?"""
        holder_key = (entity_type, entity_idx) if entity_type != 'player' else ('player',)
        old_pos = self._get_entity_pos(entity_type, entity_idx)
        held_box = self.boxes[held_box_idx]
        dx = held_box[0] - old_pos[0]
        dy = held_box[1] - old_pos[1]
        new_held = (new_pos[0] + dx, new_pos[1] + dy)

        # new_pos must be free OR == held_box current pos, and not in barriers
        pos_ok = (new_pos not in self.collidable or new_pos == held_box) and new_pos not in self.barriers
        # new_held must be free OR == entity current pos (NO barrier check for held box!)
        held_ok = (new_held not in self.collidable or new_held == old_pos)
        return pos_ok and held_ok

    def _get_entity_pos(self, entity_type, entity_idx):
        if entity_type == 'player': return self.player
        if entity_type == 'kd': return self.npc_kd[entity_idx]
        return self.npc_ys[entity_idx]

    def _set_entity_pos(self, entity_type, entity_idx, x, y):
        if entity_type == 'player': self.player = (x, y)
        elif entity_type == 'kd': self.npc_kd[entity_idx] = (x, y)
        else: self.npc_ys[entity_idx] = (x, y)

    def _grab(self, holder_key, box_idx):
        """xpcvspllwr: holder grabs box."""
        # If box already held by someone, release it first
        if box_idx in self.box_to_holder:
            old_holder = self.box_to_holder[box_idx]
            self._release_by_holder(old_holder)
        self.holder_to_box[holder_key] = box_idx
        self.box_to_holder[box_idx] = holder_key

    def _release_by_holder(self, holder_key):
        """kqrtstlzkg: release box held by holder."""
        if holder_key in self.holder_to_box:
            box_idx = self.holder_to_box[holder_key]
            del self.box_to_holder[box_idx]
            del self.holder_to_box[holder_key]

    def _bfs_to_target_with_held(self, entity_type, entity_idx, target_set):
        """cyjrduhzmz / egqayvffim: BFS for entity holding a box toward target."""
        holder_key = (entity_type, entity_idx) if entity_type != 'player' else ('player',)
        held_idx = self.holder_to_box[holder_key]
        pos = self._get_entity_pos(entity_type, entity_idx)
        held_box = self.boxes[held_idx]
        dx = held_box[0] - pos[0]
        dy = held_box[1] - pos[1]

        # BFS storing (current_pos, first_step_pos)
        collidable = self.collidable
        barriers = self.barriers

        # Check if already there
        box_pos = (pos[0] + dx, pos[1] + dy)
        if box_pos in target_set:
            return [pos]

        visited = {pos}
        # queue entries: (current, first_step)
        queue = deque()
        for npos in adjacent(pos):
            if npos in visited:
                continue
            new_held = (npos[0] + dx, npos[1] + dy)
            pos_ok = (npos not in collidable or npos == held_box) and npos not in barriers
            held_ok = (new_held not in collidable or new_held == pos)
            if pos_ok and held_ok:
                bp = (npos[0] + dx, npos[1] + dy)
                if bp in target_set:
                    return [pos, npos]
                visited.add(npos)
                queue.append((npos, npos))

        while queue:
            current, first_step = queue.popleft()
            for npos in adjacent(current):
                if npos in visited:
                    continue
                new_held = (npos[0] + dx, npos[1] + dy)
                pos_ok = (npos not in collidable or npos == held_box) and npos not in barriers
                held_ok = (new_held not in collidable or new_held == pos)
                if pos_ok and held_ok:
                    bp = (npos[0] + dx, npos[1] + dy)
                    if bp in target_set:
                        return [pos, first_step]
                    visited.add(npos)
                    queue.append((npos, first_step))
        return None

    def _compute_lkvghqfwan(self):
        """Positions adjacent to boxes that are not on target and not held."""
        result = set()
        for bi, bp in enumerate(self.boxes):
            if bi not in self.box_to_holder and not self._is_on_target(bp):
                for ap in adjacent(bp):
                    result.add(ap)
        return result

    def _compute_uuorgjazmj(self):
        """Positions adjacent to boxes that are not held by ysysltqlke and not on alt_target."""
        result = set()
        for bi, bp in enumerate(self.boxes):
            held_by_ys = False
            if bi in self.box_to_holder:
                hk = self.box_to_holder[bi]
                if hk[0] == 'ys':
                    held_by_ys = True
            if not held_by_ys and not self._is_on_alt_target(bp):
                for ap in adjacent(bp):
                    result.add(ap)
        return result

    def _bfs_to_adj_box(self, entity_type, entity_idx, adj_set):
        """czrprbohhe / zauouvdhta: BFS for entity (not holding) to reach adjacent box position."""
        pos = self._get_entity_pos(entity_type, entity_idx)
        if pos in adj_set:
            return [pos]

        visited = {pos}
        collidable = self.collidable
        barriers = self.barriers
        queue = deque()
        for npos in adjacent(pos):
            if npos not in visited and npos not in collidable and npos not in barriers:
                if npos in adj_set:
                    return [pos, npos]
                visited.add(npos)
                queue.append((npos, npos))

        while queue:
            current, first_step = queue.popleft()
            for npos in adjacent(current):
                if npos not in visited and npos not in collidable and npos not in barriers:
                    if npos in adj_set:
                        return [pos, first_step]
                    visited.add(npos)
                    queue.append((npos, first_step))
        return None

    def _sim_kd_npcs(self):
        """ynmgxjqkgh: simulate kdweefinfi NPC behavior."""
        for i in range(len(self.npc_kd)):
            holder_key = ('kd', i)
            if holder_key in self.holder_to_box:
                held_idx = self.holder_to_box[holder_key]
                held_box = self.boxes[held_idx]
                if self._is_on_target(held_box):
                    self._release_by_holder(holder_key)
                else:
                    path = self._bfs_to_target_with_held('kd', i, self.targets)
                    if path and len(path) > 1:
                        next_pos = path[1]
                        self._move_entity('kd', i, next_pos[0], next_pos[1])
            else:
                # Try grab adjacent box
                npc_pos = self.npc_kd[i]
                grabbed = False
                for bi, bp in enumerate(self.boxes):
                    if abs(npc_pos[0]-bp[0]) + abs(npc_pos[1]-bp[1]) == CELL:
                        if bi not in self.box_to_holder and not self._is_on_target(bp):
                            self._grab(holder_key, bi)
                            grabbed = True
                            return  # "return" in original code
                if not grabbed:
                    adj_set = self._compute_lkvghqfwan()
                    path = self._bfs_to_adj_box('kd', i, adj_set)
                    if path and len(path) > 1:
                        next_pos = path[1]
                        self._move_entity('kd', i, next_pos[0], next_pos[1])

    def _sim_ys_npcs(self):
        """aoeyzovteg: simulate ysysltqlke NPC behavior."""
        for i in range(len(self.npc_ys)):
            holder_key = ('ys', i)
            if holder_key in self.holder_to_box:
                held_idx = self.holder_to_box[holder_key]
                held_box = self.boxes[held_idx]
                if self._is_on_alt_target(held_box):
                    self._release_by_holder(holder_key)
                else:
                    path = self._bfs_to_target_with_held('ys', i, self.alt_targets)
                    if path and len(path) > 1:
                        next_pos = path[1]
                        self._move_entity('ys', i, next_pos[0], next_pos[1])
            else:
                npc_pos = self.npc_ys[i]
                grabbed = False
                for bi, bp in enumerate(self.boxes):
                    if abs(npc_pos[0]-bp[0]) + abs(npc_pos[1]-bp[1]) == CELL:
                        held_by_ys = bi in self.box_to_holder and self.box_to_holder[bi][0] == 'ys'
                        if not held_by_ys and not self._is_on_alt_target(bp):
                            self._grab(holder_key, bi)
                            grabbed = True
                            return
                if not grabbed:
                    adj_set = self._compute_uuorgjazmj()
                    path = self._bfs_to_adj_box('ys', i, adj_set)
                    if path and len(path) > 1:
                        next_pos = path[1]
                        self._move_entity('ys', i, next_pos[0], next_pos[1])

    def _check_win(self):
        """ymzfopzgbq: all boxes on target and not held by anyone."""
        for bi, bp in enumerate(self.boxes):
            if not self._is_on_target(bp):
                return False
            if bi in self.box_to_holder:
                return False
        return True

    def do_action(self, action_id):
        """Execute one action and return if game is won."""
        self.steps_left -= 1

        if action_id in [1, 2, 3, 4]:
            deltas = {1: (0, -CELL), 2: (0, CELL), 3: (-CELL, 0), 4: (CELL, 0)}
            dx, dy = deltas[action_id]
            new_rot = rotation_from_delta(dx, dy)

            holder_key = ('player',)
            if holder_key not in self.holder_to_box:
                self.player_rot = new_rot

            new_pos = (self.player[0] + dx, self.player[1] + dy)
            self._move_entity('player', 0, new_pos[0], new_pos[1])
            if holder_key not in self.holder_to_box:
                self.player_rot = new_rot

        elif action_id == 5:
            holder_key = ('player',)
            if holder_key in self.holder_to_box:
                self._release_by_holder(holder_key)
            else:
                # Try pick up box
                face = facing_pos(self.player, self.player_rot)
                picked = False
                for bi, bp in enumerate(self.boxes):
                    if bp == face:
                        self._grab(holder_key, bi)
                        picked = True
                        break
                # Try destroy ys NPC
                if not picked:
                    for i, np_pos in enumerate(self.npc_ys):
                        if np_pos == face:
                            self._release_by_holder(('ys', i))
                            self.collidable.discard(np_pos)
                            self.npc_ys.pop(i)
                            # Remap ys holder keys
                            new_h2b = {}
                            new_b2h = {}
                            for hk, bi in self.holder_to_box.items():
                                if hk[0] == 'ys':
                                    if hk[1] > i:
                                        new_hk = ('ys', hk[1] - 1)
                                    elif hk[1] == i:
                                        continue  # already released
                                    else:
                                        new_hk = hk
                                    new_h2b[new_hk] = bi
                                    new_b2h[bi] = new_hk
                                else:
                                    new_h2b[hk] = bi
                                    new_b2h[bi] = hk
                            self.holder_to_box = new_h2b
                            self.box_to_holder = new_b2h
                            break

        # NPC simulation
        self._sim_kd_npcs()
        self._sim_ys_npcs()

        if self._check_win():
            self.won = True
            return True
        if self.steps_left <= 0:
            self.lost = True
        return False

    def state_key(self):
        """Hashable state for BFS dedup."""
        return (
            self.player, self.player_rot,
            tuple(self.boxes),
            tuple(self.npc_kd), tuple(self.npc_ys),
            tuple(sorted(self.holder_to_box.items())),
        )

    def clone(self):
        """Create a copy of this game state."""
        import copy
        g = GameSim.__new__(GameSim)
        g.static_walls = self.static_walls
        g.barriers = self.barriers
        g.targets = self.targets
        g.alt_targets = self.alt_targets
        g.player = self.player
        g.player_rot = self.player_rot
        g.boxes = list(self.boxes)
        g.npc_kd = list(self.npc_kd)
        g.npc_ys = list(self.npc_ys)
        g.holder_to_box = dict(self.holder_to_box)
        g.box_to_holder = dict(self.box_to_holder)
        g.steps_left = self.steps_left
        g.won = self.won
        g.lost = self.lost
        g.collidable = set(self.collidable)
        return g


def solve_pick_and_carry(level_data):
    """Solve using pick-and-carry mechanics for levels without NPCs.

    Strategy: for each box, find path to carry it to a target.
    1. Player BFS to position adjacent to box (correct rotation to face it)
    2. ACTION5 to pick up
    3. Player BFS (while carrying) to position where box is on target
    4. ACTION5 to drop
    Repeat for each box.
    """
    static_walls = level_data['static_walls']
    barriers = level_data['barriers']
    targets = level_data['targets']
    boxes = list(level_data['boxes'])
    player = level_data['player']
    step_limit = level_data['step_limit']

    blocked = static_walls | barriers

    # Get target cells (4-aligned positions within target zones)
    target_cells = set()
    for t in targets:
        if t[0] % CELL == 0 and t[1] % CELL == 0:
            target_cells.add(t)

    if all(b in targets for b in boxes):
        return []

    all_actions = []
    current_player = player
    current_rot = 0
    remaining_boxes = list(range(len(boxes)))
    current_boxes = list(boxes)

    # Greedy: pick closest box to a target, carry it there
    while remaining_boxes:
        # Find available target cells
        used_targets = set(current_boxes[i] for i in range(len(current_boxes))
                          if i not in remaining_boxes and current_boxes[i] in targets)
        avail_targets = target_cells - used_targets

        best_plan = None
        best_cost = float('inf')

        for bi in remaining_boxes:
            box_pos = current_boxes[bi]
            for tgt in avail_targets:
                # Plan: walk to box, pick up, carry to target, drop
                plan = plan_carry_box(current_player, current_rot, box_pos, tgt,
                                      current_boxes, bi, static_walls, barriers, blocked)
                if plan is not None and len(plan) < best_cost:
                    best_plan = plan
                    best_cost = len(plan)
                    best_bi = bi
                    best_tgt = tgt

        if best_plan is None:
            print(f"    No plan found for remaining boxes: {[current_boxes[i] for i in remaining_boxes]}")
            break

        all_actions.extend(best_plan)
        if len(all_actions) > step_limit:
            print(f"    Exceeded step limit ({len(all_actions)} > {step_limit})")
            return None

        current_boxes[best_bi] = best_tgt
        remaining_boxes.remove(best_bi)

        # Update player position and rotation after executing plan
        sim_p, sim_r = simulate_actions_pos(current_player, current_rot, current_boxes, best_bi, best_plan, static_walls, barriers)
        current_player = sim_p
        current_rot = sim_r

        # Actually, after dropping, player is at the drop position (adjacent to target)
        # Let me just track it through the plan
        print(f"    Box {best_bi} -> {best_tgt}: {len(best_plan)} actions (total: {len(all_actions)})")

    if remaining_boxes:
        return None

    if len(all_actions) <= step_limit:
        print(f"    Solved: {len(all_actions)} moves")
        return all_actions
    return None


def plan_carry_box(player_pos, player_rot, box_pos, target_pos, all_boxes, box_idx,
                   static_walls, barriers, blocked):
    """Plan to carry a specific box to a target position.
    Returns action sequence or None.
    Tries all 4 pickup directions.
    """
    box_set = set(all_boxes)

    # For each pickup direction, try: walk to adjacent pos, pick up, carry to target, drop
    # The pickup direction determines the box_offset
    best_plan = None

    for dx, dy, rot in [(0, -CELL, 180), (0, CELL, 0), (-CELL, 0, 90), (CELL, 0, 270)]:
        adj = (box_pos[0] + dx, box_pos[1] + dy)
        if adj in blocked or adj in box_set:
            continue

        walk_blocked = blocked | box_set
        walk = bfs_walk(player_pos, player_rot, adj, rot, walk_blocked)
        if walk is None:
            continue

        box_offset = (box_pos[0] - adj[0], box_pos[1] - adj[1])

        carry = bfs_carry(adj, box_offset, target_pos,
                          all_boxes, box_idx, static_walls, barriers)
        if carry is None:
            continue

        plan = walk + [5] + carry + [5]
        if best_plan is None or len(plan) < len(best_plan):
            best_plan = plan

    return best_plan


def bfs_walk(start_pos, start_rot, target_pos, target_rot, blocked):
    """BFS for player walking (not carrying) to target position with target rotation."""
    if start_pos == target_pos and start_rot == target_rot:
        return []

    start = (start_pos, start_rot)
    target = (target_pos, target_rot)
    queue = deque([(start, [])])
    visited = {start}

    while queue:
        (pos, rot), actions = queue.popleft()
        if len(actions) > 200:
            continue

        for aid in [1, 2, 3, 4]:
            deltas = {1:(0,-CELL),2:(0,CELL),3:(-CELL,0),4:(CELL,0)}
            dx, dy = deltas[aid]
            new_rot = rotation_from_delta(dx, dy)
            tp = (pos[0]+dx, pos[1]+dy)
            if tp not in blocked:
                new_pos = tp
            else:
                new_pos = pos  # blocked, stay but rotate
            new_state = (new_pos, new_rot)
            if new_state == target:
                return actions + [aid]
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, actions + [aid]))

    return None


def bfs_carry(start_pos, box_offset, target_box_pos, all_boxes, box_idx,
              static_walls, barriers):
    """BFS for player carrying a box to get box to target position."""
    # Player needs to reach a position where player + offset = target
    # Actually, the player can approach from different directions to the target
    # But the offset is fixed. So target_player = target_box - offset
    target_player = (target_box_pos[0] - box_offset[0], target_box_pos[1] - box_offset[1])

    box_set = set(all_boxes)

    start = start_pos
    queue = deque([(start, [])])
    visited = {start}

    while queue:
        pos, actions = queue.popleft()
        if len(actions) > 200:
            continue

        for aid in [1, 2, 3, 4]:
            deltas = {1:(0,-CELL),2:(0,CELL),3:(-CELL,0),4:(CELL,0)}
            dx, dy = deltas[aid]
            tp = (pos[0]+dx, pos[1]+dy)
            new_box = (tp[0]+box_offset[0], tp[1]+box_offset[1])

            # Check entity movement
            tp_ok = tp not in static_walls and tp not in barriers
            # tp not in box_set (except the held box)
            for bi, bp in enumerate(all_boxes):
                if bi != box_idx and bp == tp:
                    tp_ok = False
                    break

            # Check box movement (no barrier check for box!)
            hb_ok = new_box not in static_walls
            for bi, bp in enumerate(all_boxes):
                if bi != box_idx and bp == new_box:
                    hb_ok = False
                    break

            if tp_ok and hb_ok:
                if tp == target_player:
                    return actions + [aid]
                if tp not in visited:
                    visited.add(tp)
                    queue.append((tp, actions + [aid]))

    return None


def simulate_actions_pos(player_pos, player_rot, boxes, box_idx, actions, static_walls, barriers):
    """Simulate actions to get final player position and rotation."""
    pos = player_pos
    rot = player_rot
    held = -1
    box_pos = boxes[box_idx]
    box_offset = None

    for a in actions:
        if a == 5:
            if held >= 0:
                held = -1
                box_offset = None
            else:
                held = box_idx
                box_offset = (box_pos[0] - pos[0], box_pos[1] - pos[1])
        elif a <= 4:
            deltas = {1:(0,-CELL),2:(0,CELL),3:(-CELL,0),4:(CELL,0)}
            dx, dy = deltas[a]
            if held < 0:
                rot = rotation_from_delta(dx, dy)
                tp = (pos[0]+dx, pos[1]+dy)
                # Simplified check
                pos = tp
            else:
                tp = (pos[0]+dx, pos[1]+dy)
                pos = tp
                box_pos = (tp[0]+box_offset[0], tp[1]+box_offset[1])

    return pos, rot


def solve_with_sim_bfs(level_data, max_iter=2000000):
    """BFS using full game simulation."""
    initial = GameSim(level_data)

    if initial._check_win():
        return []

    start_key = initial.state_key()
    queue = deque([(initial, [])])
    visited = {start_key}

    iters = 0
    while queue:
        iters += 1
        if iters % 50000 == 0:
            print(f"    SimBFS: {iters} iters, queue={len(queue)}, depth={len(queue[0][1])}")
        if iters > max_iter:
            print(f"    SimBFS exceeded {max_iter}")
            return None

        state, actions = queue.popleft()
        if state.steps_left <= 0:
            continue

        for aid in [1, 2, 3, 4, 5]:
            new_state = state.clone()
            won = new_state.do_action(aid)

            if won:
                print(f"    Solved: {len(actions)+1} moves, {iters} iters")
                return actions + [aid]

            if new_state.lost:
                continue

            key = new_state.state_key()
            if key in visited:
                continue
            visited.add(key)

            queue.append((new_state, actions + [aid]))

    print(f"    No solution, {iters} iters")
    return None


def heuristic(sim):
    """Heuristic: sum of min distances from each box to nearest target."""
    targets = sim.targets
    # Get 4-aligned target positions
    target_cells = set()
    for t in targets:
        if t[0] % CELL == 0 and t[1] % CELL == 0:
            target_cells.add(t)

    if not target_cells:
        return 0

    h = 0
    used = set()
    for bi, bp in enumerate(sim.boxes):
        if bp in targets:
            continue  # already on target
        min_d = float('inf')
        best_t = None
        for t in target_cells:
            if t not in used:
                d = abs(bp[0]-t[0]) + abs(bp[1]-t[1])
                if d < min_d:
                    min_d = d
                    best_t = t
        if best_t is not None:
            used.add(best_t)
        h += min_d // CELL if min_d != float('inf') else 100
    return h


def solve_with_beam_search(level_data, beam_width=500, max_steps=None, time_limit=60):
    """Beam search using heuristic."""
    if max_steps is None:
        max_steps = level_data['step_limit']

    initial = GameSim(level_data)
    if initial._check_win():
        return []

    beam = [(heuristic(initial), initial, [])]
    start_time = time.time()
    best_h = float('inf')

    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            print(f"    Beam search timeout at step {step}")
            break

        candidates = []
        seen = set()

        for h, state, actions in beam:
            if state.steps_left <= 0:
                continue
            for aid in [1, 2, 3, 4, 5]:
                ns = state.clone()
                won = ns.do_action(aid)
                if won:
                    print(f"    Beam search solved: {len(actions)+1} moves at step {step}")
                    return actions + [aid]
                if ns.lost:
                    continue
                key = ns.state_key()
                if key in seen:
                    continue
                seen.add(key)
                new_h = heuristic(ns)
                candidates.append((new_h, ns, actions + [aid]))

        if not candidates:
            print(f"    Beam search: no candidates at step {step}")
            break

        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]

        if beam[0][0] < best_h:
            best_h = beam[0][0]
            if step % 10 == 0:
                print(f"    Step {step}: best_h={best_h}, beam_size={len(beam)}")

    print(f"    Beam search failed")
    return None


import random

def solve_with_random(level_data, max_tries=50000, time_limit=120):
    """Random search with restarts."""
    start_time = time.time()
    best_score = float('inf')
    best_solution = None

    for trial in range(max_tries):
        if time.time() - start_time > time_limit:
            break

        sim = GameSim(level_data)
        actions = []

        for step in range(level_data['step_limit']):
            # Weighted random: prefer movement over ACTION5
            aid = random.choices([1,2,3,4,5], weights=[3,3,3,3,1])[0]
            won = sim.do_action(aid)
            actions.append(aid)
            if won:
                print(f"    Random trial {trial}: solved in {len(actions)} moves!")
                return actions
            if sim.lost:
                break

        on_t = sum(1 for b in sim.boxes if sim._is_on_target(b))
        score = len(sim.boxes) - on_t
        if score < best_score:
            best_score = score
            if trial % 5000 == 0:
                print(f"    Trial {trial}: best_score={best_score} ({on_t}/{len(sim.boxes)} on target)")

    print(f"    Random search: best score={best_score}")
    return None


def solve_with_greedy_random(level_data, max_tries=100000, time_limit=120):
    """Random search biased by heuristic - pick best of N random continuations."""
    start_time = time.time()

    for trial in range(max_tries):
        if time.time() - start_time > time_limit:
            break

        sim = GameSim(level_data)
        actions = []

        for step in range(level_data['step_limit']):
            # Try all 5 actions, pick the one with best heuristic (with some randomness)
            best_aid = None
            best_h = float('inf')
            candidates = []

            for aid in [1, 2, 3, 4, 5]:
                ns = sim.clone()
                won = ns.do_action(aid)
                if won:
                    return actions + [aid]
                if ns.lost:
                    continue
                h = heuristic(ns)
                candidates.append((h, aid, ns))

            if not candidates:
                break

            # Pick from top candidates with some randomness
            candidates.sort(key=lambda x: x[0])
            # 70% chance pick best, 20% second best, 10% random
            r = random.random()
            if r < 0.6 and len(candidates) >= 1:
                idx = 0
            elif r < 0.8 and len(candidates) >= 2:
                idx = 1
            elif r < 0.9 and len(candidates) >= 3:
                idx = 2
            else:
                idx = random.randint(0, len(candidates)-1)

            h, aid, ns = candidates[idx]
            actions.append(aid)
            sim = ns

        if trial % 1000 == 0 and trial > 0:
            print(f"    Greedy random trial {trial}")

    print(f"    Greedy random: no solution found")
    return None


def try_manual_plan(level_idx, level_data):
    """Try a hand-crafted plan for known levels."""
    plans = {
        2: [1,1,1,1, 4, 5, 4,4,4, 5,  # carry (20,20) to (32,20)
            3,3,3,3,3,3, 1, 4, 5, 4,4,4,4,4,4, 5,  # carry (8,16) to (32,16)
            3,3,3,3,3, 2,2,2,2,2,2,2, 4, 5, 4,4,4,4,4, 5]  # carry (12,44) to (32,44)
            + [1]*54,  # let NPC work
        3: [4, 2, 5, 2, 2, 5, 1, 1, 1, 5, 1, 5, 3, 5, 3, 5, 4, 4, 5, 2, 4, 5,
            2, 2, 5, 2, 5, 3, 3, 5, 1, 3, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        8: [4, 4, 1, 4, 1, 5, 1, 1, 5, 2, 4, 2, 5, 1, 4, 5, 4, 2, 2, 5, 1, 1, 5,
            3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 4, 1, 1, 1, 1, 5, 2,
            3, 5, 2, 2, 3, 5, 1, 4, 4, 4, 4, 4, 1, 5],
    }

    if level_idx not in plans:
        return None

    actions = plans[level_idx]
    # Verify
    sim = GameSim(level_data)
    for i, a in enumerate(actions):
        won = sim.do_action(a)
        if won:
            return actions[:i+1]
        if sim.lost:
            return None
    return None


def solve_level(level_idx):
    """Solve a single level."""
    print(f"\nLevel {level_idx}:")
    level_data = parse_level(level_idx)

    has_kd = len(level_data['npc_kd']) > 0
    has_ys = len(level_data['npc_ys']) > 0
    n_boxes = len(level_data['boxes'])

    print(f"  {n_boxes} boxes, {len(level_data['npc_kd'])} kd NPCs, {len(level_data['npc_ys'])} ys NPCs")
    print(f"  Step limit: {level_data['step_limit']}")

    start_time = time.time()
    solution = None

    # Try manual plan first
    solution = try_manual_plan(level_idx, level_data)
    if solution:
        print(f"  Solved with manual plan: {len(solution)} moves")
        elapsed = time.time() - start_time
        return solution, elapsed

    if not has_kd and not has_ys:
        print("  Strategy: pick-and-carry greedy")
        solution = solve_pick_and_carry(level_data)
    else:
        # Try beam search first
        print("  Strategy: beam search")
        solution = solve_with_beam_search(level_data, beam_width=1000, time_limit=30)

        if solution is None:
            print("  Strategy: simulation BFS")
            solution = solve_with_sim_bfs(level_data, max_iter=2000000)

        if solution is None:
            print("  Strategy: greedy random")
            solution = solve_with_greedy_random(level_data, max_tries=50000, time_limit=60)

        if solution is None:
            print("  Strategy: pure random")
            solution = solve_with_random(level_data, max_tries=50000, time_limit=30)

    elapsed = time.time() - start_time

    if solution is not None:
        print(f"  Result: {len(solution)} actions in {elapsed:.2f}s")
        return solution, elapsed
    else:
        print(f"  Failed in {elapsed:.2f}s")
        return None, elapsed


def main():
    results = {
        "game_id": "wa30",
        "total_levels": 9,
        "levels": [],
        "solved_levels": 0,
    }

    if len(sys.argv) > 1:
        levels_to_solve = [int(x) for x in sys.argv[1:]]
    else:
        levels_to_solve = list(range(9))

    all_solutions = {}

    for level_idx in levels_to_solve:
        solution, elapsed = solve_level(level_idx)
        if solution is not None:
            all_solutions[level_idx] = (solution, elapsed)

    for level_idx in range(9):
        if level_idx in all_solutions:
            solution, elapsed = all_solutions[level_idx]
            results["levels"].append({
                "level": level_idx,
                "solved": True,
                "num_actions": len(solution),
                "actions": [{"id": a} for a in solution],
            })
            results["solved_levels"] += 1
        else:
            results["levels"].append({
                "level": level_idx,
                "solved": False,
            })

    import os
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solutions", "wa30.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Solved {results['solved_levels']}/{results['total_levels']} levels")


if __name__ == "__main__":
    main()
