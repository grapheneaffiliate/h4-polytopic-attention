"""
All environments for Agent Zero, implementing the generic Env interface.

14 environments total:
- 10 ported from CGE (same logic, new interface)
- 4 new: TextReasoning, CodeEnv, FeatureTransfer, CompositeEnv
"""

import random
import re
from .env_interface import Env


# ── Ported from CGE ──────────────────────────────────────────

class LinearPuzzle(Env):
    """Press correct actions in sequence. Tests action efficacy learning."""
    def __init__(self, n_levels=3, sol_len=5, n_actions=15, budget=25):
        self._level = 0; self._total = n_levels; self._budget = budget
        self._n_actions = n_actions; self._steps = 0; self._pos = 0
        self._effective = set(random.sample(range(n_actions), sol_len))
        self._solutions = {l: random.sample(list(self._effective), sol_len) for l in range(n_levels)}
    @property
    def name(self): return "LinearPuzzle"
    @property
    def total_levels(self): return self._total
    @property
    def current_level(self): return self._level
    def reset(self):
        self._pos = 0; self._steps = 0
        return f"L{self._level}_S0", set(range(self._n_actions))
    def step(self, action):
        self._steps += 1
        if self._steps >= self._budget:
            return f"L{self._level}_S{self._pos}", set(range(self._n_actions)), 0, False, True
        sol = self._solutions[self._level]
        if self._pos < len(sol) and action == sol[self._pos]:
            self._pos += 1
            if self._pos >= len(sol):
                self._level += 1; self._pos = 0
                return f"L{self._level}_S0", set(range(self._n_actions)), 100, True, self._level >= self._total
            return f"L{self._level}_S{self._pos}", set(range(self._n_actions)), 1, False, False
        return f"L{self._level}_S{self._pos}", set(range(self._n_actions)), 0, False, False


class MazeNavigation(Env):
    """Grid maze. 4 arrows work, 4-6 don't. Tests spatial exploration."""
    def __init__(self, w=7, h=7, n_levels=3, budget=50):
        self._w = w; self._h = h; self._levels = n_levels; self._budget = budget
        self._level = 0; self._pos = (0,0); self._steps = 0
        self._blocks = {}
        for lv in range(n_levels):
            b = set()
            for _ in range(w*h//4):
                r,c = random.randint(0,h-1), random.randint(0,w-1)
                if (r,c) != (0,0) and (r,c) != (w-1,h-1): b.add((c,r))
            for i in range(w): b.discard((i,0))
            for j in range(h): b.discard((w-1,j))
            self._blocks[lv] = b
    @property
    def name(self): return "MazeNavigation"
    @property
    def total_levels(self): return self._levels
    @property
    def current_level(self): return self._level
    def reset(self):
        self._pos = (0,0); self._steps = 0
        return self._state(), set(range(10))
    def step(self, action):
        self._steps += 1
        if self._steps >= self._budget:
            return self._state(), set(range(10)), 0, False, True
        dx,dy = 0,0
        if action==0: dy=-1
        elif action==1: dy=1
        elif action==2: dx=-1
        elif action==3: dx=1
        else: return self._state(), set(range(10)), 0, False, False
        nx,ny = self._pos[0]+dx, self._pos[1]+dy
        if 0<=nx<self._w and 0<=ny<self._h and (nx,ny) not in self._blocks[self._level]:
            self._pos = (nx,ny)
            if self._pos == (self._w-1, self._h-1):
                self._level += 1; self._pos = (0,0)
                return self._state(), set(range(10)), 100, True, self._level >= self._levels
            return self._state(), set(range(10)), 1, False, False
        return self._state(), set(range(10)), 0, False, False
    def _state(self): return f"L{self._level}_{self._pos[0]}_{self._pos[1]}"


class BottleneckPuzzle(Env):
    """Tree with 1 correct branch. Tests dead-end avoidance."""
    def __init__(self, n_branches=5, depth=4, n_levels=2, budget=60):
        self._nb = n_branches; self._depth = depth; self._levels = n_levels
        self._budget = budget; self._level = 0; self._state_v = "root"; self._steps = 0
        self._na = n_branches + 2
        self._graphs = {l: self._build(l) for l in range(n_levels)}
    def _build(self, level):
        rng = random.Random(level*137+42); wb = rng.randint(0, self._nb-1)
        g = {"root": {b: f"L{level}_B{b}_0" for b in range(self._nb)}}
        for b in range(self._nb):
            if b == wb: continue
            for d in range(self._depth):
                s = f"L{level}_B{b}_{d}"
                if d < self._depth-1: g[s] = {self._nb: f"L{level}_B{b}_{d+1}"}
                else: g[s] = {}
        for d in range(self._depth):
            s = f"L{level}_B{wb}_{d}"
            g[s] = {self._nb: f"L{level}_B{wb}_{d+1}"}
        bt = f"L{level}_B{wb}_{self._depth}"
        g[bt] = {self._nb+1: "GOAL"}; g["GOAL"] = {}; self._wb = wb
        return g
    @property
    def name(self): return "BottleneckPuzzle"
    @property
    def total_levels(self): return self._levels
    @property
    def current_level(self): return self._level
    def reset(self):
        self._state_v = "root"; self._steps = 0
        return "root", set(range(self._na))
    def step(self, action):
        self._steps += 1
        if self._steps >= self._budget:
            return self._state_v, set(range(self._na)), 0, False, True
        tr = self._graphs[self._level].get(self._state_v, {})
        if action in tr:
            ns = tr[action]
            if ns == "GOAL":
                self._level += 1; self._state_v = "root"
                return "root", set(range(self._na)), 100, True, self._level >= self._levels
            self._state_v = ns
            return self._state_v, set(range(self._na)), 1, False, False
        return self._state_v, set(range(self._na)), 0, False, False


class HiddenPatternPuzzle(Env):
    """State features determine correct action. Tests feature→action learning."""
    def __init__(self, n_steps=8, n_levels=3, n_actions=8, budget=35):
        self._ns = n_steps; self._levels = n_levels; self._na = n_actions
        self._budget = budget; self._level = 0; self._pos = 0; self._steps = 0
        acts = list(range(n_actions)); random.shuffle(acts)
        self._rule = {c: acts[c] for c in range(4)}
        self._seqs = {l: [random.randint(0,3) for _ in range(n_steps)] for l in range(n_levels)}
    @property
    def name(self): return "HiddenPatternPuzzle"
    @property
    def total_levels(self): return self._levels
    @property
    def current_level(self): return self._level
    def reset(self):
        self._pos = 0; self._steps = 0; return self._state(), set(range(self._na))
    def step(self, action):
        self._steps += 1
        if self._steps >= self._budget:
            return self._state(), set(range(self._na)), 0, False, True
        seq = self._seqs[self._level]
        if self._pos < len(seq) and action == self._rule[seq[self._pos]]:
            self._pos += 1
            if self._pos >= len(seq):
                self._level += 1; self._pos = 0
                return self._state(), set(range(self._na)), 100, True, self._level >= self._levels
            return self._state(), set(range(self._na)), 1, False, False
        return self._state(), set(range(self._na)), 0, False, False
    def _state(self):
        seq = self._seqs.get(self._level, [])
        c = seq[self._pos] if self._pos < len(seq) else -1
        return f"L{self._level}_P{self._pos}_C{c}"


class LargeStateSpace(Env):
    """Big grid, few useful actions. Tests budget focusing."""
    def __init__(self, gs=8, n_levels=1, budget=150):
        self._gs = gs; self._levels = n_levels; self._budget = budget
        self._level = 0; self._pos = (0,0); self._steps = 0
    @property
    def name(self): return "LargeStateSpace"
    @property
    def total_levels(self): return self._levels
    @property
    def current_level(self): return self._level
    def reset(self):
        self._pos = (0,0); self._steps = 0; return self._state(), set(range(6))
    def step(self, action):
        self._steps += 1
        if self._steps >= self._budget:
            return self._state(), set(range(6)), 0, False, True
        x,y = self._pos; ch = False
        if action==0 and y>0: self._pos=(x,y-1); ch=True
        elif action==1 and y<self._gs-1: self._pos=(x,y+1); ch=True
        elif action==2 and x>0: self._pos=(x-1,y); ch=True
        elif action==3 and x<self._gs-1: self._pos=(x+1,y); ch=True
        if self._pos == (self._gs-1, self._gs-1):
            self._level += 1; self._pos = (0,0)
            return self._state(), set(range(6)), 100, True, self._level >= self._levels
        return self._state(), set(range(6)), 1 if ch else 0, False, False
    def _state(self): return f"L{self._level}_{self._pos[0]}_{self._pos[1]}"


class DeepTreeSearch(Env):
    """Deep tree with many dead ends. Tests MCTS + tree detection."""
    def __init__(self, n_branches=8, depth=4, n_levels=4, budget=60):
        self._nb = n_branches; self._depth = depth; self._levels = n_levels
        self._budget = budget; self._level = 0; self._steps = 0
        self._na = n_branches + 1; self._state_v = f"L0"
        self._graphs = {l: self._build(l) for l in range(n_levels)}
    def _build(self, level):
        rng = random.Random(level*137+42); g = {}
        good = [rng.choice([i for i in range(self._nb) if i%2==0]) for _ in range(self._depth)]
        def build(pfx, d):
            if d >= self._depth: return
            tr = {b: f"{pfx}_B{b}" for b in range(self._nb)}
            g[pfx] = tr
            for b in range(self._nb):
                if b == good[d]:
                    build(f"{pfx}_B{b}", d+1)
                else:
                    dd = rng.randint(1,2); cur = f"{pfx}_B{b}"
                    for i in range(dd):
                        nxt = f"{cur}_D{i}"; g[cur] = {self._nb: nxt}; cur = nxt
                    g[cur] = {}
            if d == self._depth - 1:
                deep = f"{pfx}_B{good[d]}"; cur = deep
                for i in range(2):
                    nxt = f"{cur}_F{i}"; g[cur] = {self._nb: nxt}; cur = nxt
                g[cur] = {self._nb: "GOAL"}
        build(f"L{level}", 0); g["GOAL"] = {}; return g
    @property
    def name(self): return "DeepTreeSearch"
    @property
    def total_levels(self): return self._levels
    @property
    def current_level(self): return self._level
    def reset(self):
        self._state_v = f"L{self._level}"; self._steps = 0
        return self._state_v, set(range(self._na))
    def step(self, action):
        self._steps += 1
        if self._steps >= self._budget:
            return self._state_v, set(range(self._na)), 0, False, True
        tr = self._graphs[self._level].get(self._state_v, {})
        if action in tr:
            ns = tr[action]
            if ns == "GOAL":
                self._level += 1; self._state_v = f"L{self._level}"
                return self._state_v, set(range(self._na)), 100, True, self._level >= self._levels
            self._state_v = ns
            return self._state_v, set(range(self._na)), 1, False, False
        return self._state_v, set(range(self._na)), 0, False, False


class NeedleInHaystack(Env):
    """Big grid with waypoints. Tests dead-action pruning."""
    def __init__(self, gs=12, n_wp=3, n_levels=2, budget=120):
        self._gs = gs; self._levels = n_levels; self._budget = budget
        self._level = 0; self._pos = (0,0); self._wi = 0; self._steps = 0
        self._wps = {}
        for l in range(n_levels):
            rng = random.Random(l*97+13)
            w = [(rng.randint(0,gs-1), rng.randint(0,gs-1)) for _ in range(n_wp)]
            w.append((gs-1,gs-1)); self._wps[l] = w
    @property
    def name(self): return "NeedleInHaystack"
    @property
    def total_levels(self): return self._levels
    @property
    def current_level(self): return self._level
    def reset(self):
        self._pos=(0,0); self._wi=0; self._steps=0; return self._state(), set(range(8))
    def step(self, action):
        self._steps += 1
        if self._steps >= self._budget:
            return self._state(), set(range(8)), 0, False, True
        x,y = self._pos; ch = False
        if action==0 and y>0: self._pos=(x,y-1); ch=True
        elif action==1 and y<self._gs-1: self._pos=(x,y+1); ch=True
        elif action==2 and x>0: self._pos=(x-1,y); ch=True
        elif action==3 and x<self._gs-1: self._pos=(x+1,y); ch=True
        wps = self._wps[self._level]
        if self._wi < len(wps) and self._pos == wps[self._wi]:
            self._wi += 1; ch = True
            if self._wi >= len(wps):
                self._level += 1; self._pos=(0,0); self._wi=0
                return self._state(), set(range(8)), 100, True, self._level >= self._levels
        return self._state(), set(range(8)), 1 if ch else 0, False, False
    def _state(self): return f"L{self._level}_{self._pos[0]}_{self._pos[1]}_W{self._wi}"


class StuckGame(Env):
    """Most actions useless. Tests finding the 2 that work."""
    def __init__(self, n_pos=5, n_levels=2, budget=80):
        self._np = n_pos; self._levels = n_levels; self._budget = budget
        self._level = 0; self._pos = 0; self._act = set(); self._steps = 0
        self._amap = {}
        for l in range(n_levels):
            rng = random.Random(l*53+7)
            self._amap[l] = {p: rng.choice([10,11]) for p in range(n_pos)}
    @property
    def name(self): return "StuckGame"
    @property
    def total_levels(self): return self._levels
    @property
    def current_level(self): return self._level
    def reset(self):
        self._pos=0; self._act=set(); self._steps=0; return self._state(), set(range(12))
    def step(self, action):
        self._steps += 1
        if self._steps >= self._budget:
            return self._state(), set(range(12)), 0, False, True
        ch = False; am = self._amap[self._level]
        if action in (10,11):
            if self._pos not in self._act and am.get(self._pos)==action:
                self._act.add(self._pos); ch=True
                if len(self._act) >= self._np:
                    self._level += 1; self._pos=0; self._act=set()
                    return self._state(), set(range(12)), 100, True, self._level >= self._levels
            self._pos = (self._pos+1) % self._np; ch=True
        elif action < 4:
            if action==0 and self._pos>0: self._pos-=1; ch=True
            elif action==1 and self._pos<self._np-1: self._pos+=1; ch=True
        return self._state(), set(range(12)), 1 if ch else 0, False, False
    def _state(self):
        a = "".join(str(int(i in self._act)) for i in range(self._np))
        return f"L{self._level}_P{self._pos}_A{a}"


class CausalChain(Env):
    """Must discover multi-step action sequences."""
    def __init__(self, chain_len=3, n_actions=8, n_levels=3, budget=60):
        self._cl = chain_len; self._na = n_actions; self._levels = n_levels
        self._budget = budget; self._level = 0; self._prog = 0; self._recent = []
        self._steps = 0
        eff = random.sample(range(n_actions), min(chain_len+1, n_actions))
        self._chains = {l: [random.Random(l*71+29).choice(eff) for _ in range(chain_len)] for l in range(n_levels)}
    @property
    def name(self): return "CausalChain"
    @property
    def total_levels(self): return self._levels
    @property
    def current_level(self): return self._level
    def reset(self):
        self._prog=0; self._recent=[]; self._steps=0; return self._state(), set(range(self._na))
    def step(self, action):
        self._steps += 1
        if self._steps >= self._budget:
            return self._state(), set(range(self._na)), 0, False, True
        chain = self._chains[self._level]
        self._recent.append(action)
        if len(self._recent) > self._cl: self._recent = self._recent[-self._cl:]
        ch = False
        if len(self._recent) >= self._cl and self._recent[-self._cl:] == chain:
            self._prog += 1; self._recent = []; ch = True
            if self._prog >= 3:
                self._level += 1; self._prog = 0
                return self._state(), set(range(self._na)), 100, True, self._level >= self._levels
        elif action == chain[min(self._prog, len(chain)-1) % len(chain)]:
            ch = True
        return self._state(), set(range(self._na)), 1 if ch else 0, False, False
    def _state(self): return f"L{self._level}_P{self._prog}_R{''.join(map(str, self._recent[-3:]))}"


class RuleLearning(Env):
    """Feature→action rule transfers across levels."""
    def __init__(self, n_feat=3, n_val=4, n_actions=8, n_steps=10, n_levels=5, budget=60):
        self._nf=n_feat; self._nv=n_val; self._na=n_actions; self._ns=n_steps
        self._levels=n_levels; self._budget=budget; self._level=0; self._pos=0; self._steps=0
        self._rf = random.randint(0, n_feat-1)
        acts = list(range(n_actions)); random.shuffle(acts)
        self._rule = {v: acts[v % n_actions] for v in range(n_val)}
        self._seqs = {l: [tuple(random.Random(l*83+17).randint(0,n_val-1) for _ in range(n_feat))
                          for _ in range(n_steps)] for l in range(n_levels)}
    @property
    def name(self): return "RuleLearning"
    @property
    def total_levels(self): return self._levels
    @property
    def current_level(self): return self._level
    def reset(self):
        self._pos=0; self._steps=0; return self._state(), set(range(self._na))
    def step(self, action):
        self._steps += 1
        if self._steps >= self._budget:
            return self._state(), set(range(self._na)), 0, False, True
        seq = self._seqs[self._level]
        if self._pos < len(seq) and action == self._rule[seq[self._pos][self._rf]]:
            self._pos += 1
            if self._pos >= len(seq):
                self._level += 1; self._pos = 0
                return self._state(), set(range(self._na)), 100, True, self._level >= self._levels
            return self._state(), set(range(self._na)), 1, False, False
        return self._state(), set(range(self._na)), 0, False, False
    def _state(self):
        seq = self._seqs.get(self._level, [])
        f = "_".join(f"F{i}V{v}" for i,v in enumerate(seq[self._pos])) if self._pos < len(seq) else "done"
        return f"L{self._level}_P{self._pos}_{f}"


# ── NEW Environments ─────────────────────────────────────────

class TextReasoning(Env):
    """
    Text puzzle: given a simple pattern, pick the answer.
    Actions = candidate answers (integers). Only one is correct per step.
    Tests: can the agent learn a pattern faster than random search?

    Pattern: answer = (step_number * multiplier + offset) % n_actions
    The agent doesn't know the formula — must infer from exploration.
    """
    def __init__(self, n_actions=10, n_steps=8, n_levels=3, budget=40):
        self._na = n_actions; self._ns = n_steps; self._levels = n_levels
        self._budget = budget; self._level = 0; self._pos = 0; self._steps = 0
        self._mult = random.randint(1, 5)
        self._off = random.randint(0, n_actions-1)
    @property
    def name(self): return "TextReasoning"
    @property
    def total_levels(self): return self._levels
    @property
    def current_level(self): return self._level
    def reset(self):
        self._pos=0; self._steps=0; return self._state(), set(range(self._na))
    def step(self, action):
        self._steps += 1
        if self._steps >= self._budget:
            return self._state(), set(range(self._na)), 0, False, True
        correct = (self._pos * self._mult + self._off + self._level) % self._na
        if action == correct:
            self._pos += 1
            if self._pos >= self._ns:
                self._level += 1; self._pos = 0
                return self._state(), set(range(self._na)), 100, True, self._level >= self._levels
            return self._state(), set(range(self._na)), 1, False, False
        return self._state(), set(range(self._na)), 0, False, False
    def _state(self): return f"L{self._level}_P{self._pos}"


class CodeEnv(Env):
    """
    Given a target sequence, actions are "set position i to value v".
    Reward = number of positions matching target.
    Solve = all positions match.

    Models code editing: each action modifies one part of the output,
    reward is test pass rate.
    """
    def __init__(self, length=4, n_values=4, n_levels=2, budget=60):
        self._len = length; self._nv = n_values; self._levels = n_levels
        self._budget = budget; self._level = 0; self._steps = 0
        self._current = [0] * length
        self._targets = {l: [random.randint(0, n_values-1) for _ in range(length)] for l in range(n_levels)}
        self._na = length * n_values  # action i*n_values + v = set position i to value v
    @property
    def name(self): return "CodeEnv"
    @property
    def total_levels(self): return self._levels
    @property
    def current_level(self): return self._level
    def reset(self):
        self._current = [0] * self._len; self._steps = 0
        return self._state(), set(range(self._na))
    def step(self, action):
        self._steps += 1
        if self._steps >= self._budget:
            return self._state(), set(range(self._na)), 0, False, True
        pos = action // self._nv; val = action % self._nv
        if pos < self._len:
            old = list(self._current); self._current[pos] = val
            target = self._targets[self._level]
            old_match = sum(1 for a,b in zip(old, target) if a==b)
            new_match = sum(1 for a,b in zip(self._current, target) if a==b)
            if new_match == self._len:
                self._level += 1; self._current = [0] * self._len
                done = self._level >= self._levels
                return f"L{self._level}_{''.join('0'*self._len)}", set(range(self._na)), 100, True, done
            reward = max(0, new_match - old_match)
            return self._state(), set(range(self._na)), reward, False, False
        return self._state(), set(range(self._na)), 0, False, False
    def _state(self): return f"L{self._level}_{''.join(map(str, self._current))}"


class FeatureTransfer(Env):
    """
    Like RuleLearning, but action indices SHUFFLE every level while
    the feature→reward mapping persists. Tests true generalization:
    can the agent learn "feature X means high reward" even when the
    action that achieves it changes?
    """
    def __init__(self, n_feat=3, n_val=3, n_actions=6, n_steps=6, n_levels=4, budget=50):
        self._nf=n_feat; self._nv=n_val; self._na=n_actions; self._ns=n_steps
        self._levels=n_levels; self._budget=budget; self._level=0; self._pos=0; self._steps=0
        # The RULE: feature 0's value determines correct action BASE
        # But the mapping shuffles per level
        self._base_rule = {v: v % n_actions for v in range(n_val)}
        self._shuffles = {}
        for l in range(n_levels):
            perm = list(range(n_actions))
            random.Random(l*41+7).shuffle(perm)
            self._shuffles[l] = perm  # perm[original_action] = shuffled_action
        self._seqs = {l: [tuple(random.Random(l*67+11).randint(0,n_val-1)
                          for _ in range(n_feat)) for _ in range(n_steps)]
                     for l in range(n_levels)}
    @property
    def name(self): return "FeatureTransfer"
    @property
    def total_levels(self): return self._levels
    @property
    def current_level(self): return self._level
    def reset(self):
        self._pos=0; self._steps=0; return self._state(), set(range(self._na))
    def step(self, action):
        self._steps += 1
        if self._steps >= self._budget:
            return self._state(), set(range(self._na)), 0, False, True
        seq = self._seqs[self._level]
        if self._pos < len(seq):
            feat_val = seq[self._pos][0]  # feature 0 determines answer
            base_action = self._base_rule[feat_val]
            correct = self._shuffles[self._level][base_action]
            if action == correct:
                self._pos += 1
                if self._pos >= len(seq):
                    self._level += 1; self._pos = 0
                    return self._state(), set(range(self._na)), 100, True, self._level >= self._levels
                return self._state(), set(range(self._na)), 1, False, False
        return self._state(), set(range(self._na)), 0, False, False
    def _state(self):
        seq = self._seqs.get(self._level, [])
        f = "_".join(f"F{i}V{v}" for i,v in enumerate(seq[self._pos])) if self._pos < len(seq) else "done"
        return f"L{self._level}_P{self._pos}_{f}"


class CompositeEnv(Env):
    """
    Chains 2 sub-environments. Solve env1 to unlock env2.
    Tests planning across environment boundaries.
    """
    def __init__(self):
        self._envs = [
            LargeStateSpace(gs=5, n_levels=1, budget=60),
            LinearPuzzle(n_levels=1, sol_len=3, n_actions=6, budget=20),
        ]
        self._idx = 0; self._total = 2
    @property
    def name(self): return "CompositeEnv"
    @property
    def total_levels(self): return self._total
    @property
    def current_level(self): return self._idx
    def reset(self):
        return self._envs[self._idx].reset()
    def step(self, action):
        state, actions, reward, level_up, done = self._envs[self._idx].step(action)
        if level_up:
            self._idx += 1
            if self._idx >= len(self._envs):
                return state, actions, 100, True, True
            return self._envs[self._idx].reset()[0], self._envs[self._idx].reset()[1], 100, True, False
        return state, actions, reward, False, done


# ── Factory ──────────────────────────────────────────────────

def get_all_environments(seed=42) -> list[Env]:
    random.seed(seed)
    return [
        LinearPuzzle(), MazeNavigation(), BottleneckPuzzle(),
        HiddenPatternPuzzle(), LargeStateSpace(),
        DeepTreeSearch(), NeedleInHaystack(), StuckGame(),
        CausalChain(), RuleLearning(),
        TextReasoning(), CodeEnv(), FeatureTransfer(), CompositeEnv(),
    ]
