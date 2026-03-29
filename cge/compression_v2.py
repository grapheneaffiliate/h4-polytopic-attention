"""
Compression Layer v2 — the intelligence core.

Three new capabilities beyond v1:
1. Dead-end pattern learning: recognize branch signatures that lead nowhere
2. Feature-action rule mining: learn which state features predict which actions work
3. Action sequence memory: remember multi-step sequences that made progress

These directly address the three hardest test cases:
- DeepTreeSearch: dead-end avoidance
- RuleLearning: feature-action rules
- CausalChain: sequence memory
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import math
import re


@dataclass
class ActionStats:
    attempts: int = 0
    successes: int = 0

    @property
    def efficacy(self) -> float:
        if self.attempts == 0:
            return 0.5
        return self.successes / self.attempts

    @property
    def confidence(self) -> float:
        return 1.0 - 1.0 / (1.0 + self.attempts)


@dataclass
class StateSig:
    change_rate: float
    fanout: int
    depth: int
    total_tested: int
    is_dead_end: bool = False  # NEW: true if this state leads nowhere


class CompressionLayerV2:
    """
    Compression Layer v2 with dead-end avoidance, rule mining, and sequence memory.
    """

    def __init__(self):
        # --- From v1 ---
        self.action_stats: dict[object, ActionStats] = defaultdict(ActionStats)
        self.state_sigs: dict[str, StateSig] = {}
        self.bottlenecks: set[str] = set()
        self.winning_paths: list[list[str]] = []
        self.winning_actions: list[list[object]] = []
        self._action_ranking: list[object] = []
        self._analysis_count = 0

        # --- NEW: Dead-end pattern learning ---
        self.dead_end_states: set[str] = set()
        # Features of states that are dead ends (for generalization)
        self._dead_end_features: dict[tuple, int] = defaultdict(int)  # feature -> count
        self._live_features: dict[tuple, int] = defaultdict(int)
        self._dead_end_model_ready = False

        # --- NEW: Feature-action rule mining ---
        # For each feature value, track which actions succeed
        self._feature_action_success: dict[tuple, ActionStats] = defaultdict(ActionStats)
        # (feature_name, feature_value, action) -> stats
        self._best_rules: dict[tuple, object] = {}  # (feature_name, value) -> best action

        # --- NEW: Action sequence memory ---
        self._action_history: list[tuple] = []  # (state, action, changed)
        self._successful_sequences: dict[tuple, int] = defaultdict(int)  # seq -> count
        self._sequence_length = 3  # look for 3-action patterns
        self._best_sequences: list[tuple] = []  # top sequences by success rate
        self._current_sequence_attempt: list = []  # current sequence being tried

        # --- NEW: State-dependent action learning (from v1, improved) ---
        self._state_action_stats: dict[tuple, dict] = defaultdict(lambda: defaultdict(ActionStats))

    # ===== Analysis =====

    def analyze(self, explorer):
        """Full analysis pass. O(V + E + history)."""
        self._analysis_count += 1

        # --- State signatures + global action stats ---
        action_counts = defaultdict(lambda: [0, 0])

        for name, node in explorer.nodes.items():
            self.state_sigs[name] = StateSig(
                change_rate=node.change_rate,
                fanout=node.fanout,
                depth=node.depth,
                total_tested=len(node.tested),
            )
            for action, (changed, target) in node.tested.items():
                action_counts[action][0] += 1
                if changed:
                    action_counts[action][1] += 1

        for action, (attempts, successes) in action_counts.items():
            self.action_stats[action].attempts = attempts
            self.action_stats[action].successes = successes

        self._action_ranking = sorted(
            self.action_stats.keys(),
            key=lambda a: (self.action_stats[a].efficacy, self.action_stats[a].attempts),
            reverse=True,
        )

        # --- Dead-end detection ---
        self._detect_dead_ends(explorer)

        # --- Feature-action rule mining ---
        self._mine_feature_rules(explorer)

        # --- State-dependent action learning ---
        self._state_action_stats.clear()
        for name, node in explorer.nodes.items():
            state_type = self._classify_state(name)
            for action, (changed, target) in node.tested.items():
                sa = self._state_action_stats[state_type][action]
                sa.attempts += 1
                if changed:
                    sa.successes += 1

        # --- Sequence analysis ---
        self._analyze_sequences()

        # --- Bottleneck detection ---
        self.bottlenecks.clear()
        for name, node in explorer.nodes.items():
            novel_successors = set()
            for action, (changed, target) in node.tested.items():
                if changed and target is not None and target != name:
                    novel_successors.add(target)
            if len(novel_successors) == 1 and node.depth > 0:
                self.bottlenecks.add(name)

    # ===== Dead-End Learning =====

    def _detect_dead_ends(self, explorer):
        """
        Identify true dead-end states using reachability analysis.

        A state is a dead end ONLY if:
        1. It has no path (through any chain of successors) to any frontier node
        2. It is not on a known winning path
        3. It is fully explored (closed)

        This avoids false positives: states that are closed but lead to
        other states with untested actions are NOT dead ends.
        """
        self.dead_end_states.clear()
        self._dead_end_features.clear()
        self._live_features.clear()

        winning_states = set()
        for path in self.winning_paths:
            winning_states.update(path)

        # BFS backward from frontier to find all states that can reach frontier
        can_reach_frontier = set(explorer.frontier)
        queue = list(explorer.frontier)
        while queue:
            v = queue.pop(0)
            for action, u in explorer.rev_edges.get(v, set()):
                if u not in can_reach_frontier:
                    can_reach_frontier.add(u)
                    queue.append(u)

        for name, node in explorer.nodes.items():
            if name in winning_states:
                for feat in self._extract_features(name):
                    self._live_features[feat] += 1
                continue

            if node.closed and name not in can_reach_frontier:
                # Truly dead: closed AND cannot reach any frontier node
                self.dead_end_states.add(name)
                if name in self.state_sigs:
                    self.state_sigs[name].is_dead_end = True
                for feat in self._extract_features(name):
                    self._dead_end_features[feat] += 1
            else:
                for feat in self._extract_features(name):
                    self._live_features[feat] += 1

        # Need substantial evidence before using dead-end predictions
        self._dead_end_model_ready = len(self.dead_end_states) >= 10

    def _extract_features(self, state_name: str) -> list[tuple]:
        """Extract generalizable features from a state name."""
        features = []
        parts = state_name.split("_")
        for i, part in enumerate(parts):
            # Feature: specific part value
            features.append(("part", i, part))
            # Feature: part prefix (e.g., "B" from "B3")
            if part and part[0].isalpha():
                features.append(("prefix", i, part[0]))
            # Feature: part numeric value (if it has one)
            nums = re.findall(r'\d+', part)
            if nums:
                val = int(nums[-1])
                features.append(("num", i, val))
                features.append(("parity", i, val % 2))
                features.append(("range", i, val // 3))  # group into ranges

        # Depth feature
        if state_name in self.state_sigs:
            sig = self.state_sigs[state_name]
            features.append(("depth_range", sig.depth // 3))
        return features

    def predict_dead_end(self, state_name: str) -> float:
        """
        Predict probability that a state is a dead end based on learned patterns.
        Returns 0.0 (definitely live) to 1.0 (definitely dead).

        Conservative: requires STRONG evidence (multiple features agreeing,
        high sample counts). False positives are much worse than false negatives.
        """
        if not self._dead_end_model_ready:
            return 0.0

        features = self._extract_features(state_name)
        if not features:
            return 0.0

        dead_score = 0.0
        live_score = 0.0
        n_features = 0

        for feat in features:
            dead_count = self._dead_end_features.get(feat, 0)
            live_count = self._live_features.get(feat, 0)
            total = dead_count + live_count
            if total >= 10:  # need substantial evidence
                ratio = dead_count / total
                if ratio > 0.7:  # only count strongly dead-associated features
                    dead_score += ratio
                    n_features += 1
                elif ratio < 0.3:
                    live_score += 1.0
                    n_features += 1

        if n_features < 2:  # need multiple features agreeing
            return 0.0

        if live_score > 0:
            return 0.0  # any live-associated feature vetoes

        return min(dead_score / n_features, 0.8)  # cap at 0.8, never 100% sure

    # ===== Feature-Action Rule Mining =====

    def _mine_feature_rules(self, explorer):
        """
        Learn rules: which state features predict which actions succeed.
        For each (feature, value) pair, find the action with highest success rate.
        """
        self._feature_action_success.clear()
        self._best_rules.clear()

        for name, node in explorer.nodes.items():
            features = self._extract_features(name)
            for action, (changed, target) in node.tested.items():
                for feat in features:
                    key = (feat, action)
                    stats = self._feature_action_success[key]
                    stats.attempts += 1
                    if changed:
                        stats.successes += 1

        # Find best action per feature
        feature_best = defaultdict(lambda: (None, 0.0, 0))
        for (feat, action), stats in self._feature_action_success.items():
            if stats.attempts >= 3:
                if stats.efficacy > feature_best[feat][1]:
                    feature_best[feat] = (action, stats.efficacy, stats.attempts)

        for feat, (action, eff, count) in feature_best.items():
            if eff > 0.6 and count >= 8:  # only high-confidence rules
                self._best_rules[feat] = action

    def get_rule_actions(self, state_name: str, available: set) -> list:
        """Get actions suggested by learned feature-action rules."""
        features = self._extract_features(state_name)
        action_scores = defaultdict(float)

        for feat in features:
            if feat in self._best_rules:
                action = self._best_rules[feat]
                if action in available:
                    # Weight by how specific/confident the rule is
                    key = (feat, action)
                    stats = self._feature_action_success.get(key)
                    if stats and stats.attempts >= 3:
                        action_scores[action] += stats.efficacy * math.log(stats.attempts + 1)

        if not action_scores:
            return []

        return sorted(action_scores.keys(), key=lambda a: action_scores[a], reverse=True)

    # ===== Action Sequence Memory =====

    def record_action(self, state: str, action: object, changed: bool):
        """Record an action for sequence analysis."""
        self._action_history.append((state, action, changed))

    def _analyze_sequences(self):
        """Find action sequences that reliably produce state changes."""
        self._successful_sequences.clear()
        seq_attempts = defaultdict(int)

        history = self._action_history
        slen = self._sequence_length

        for i in range(len(history) - slen + 1):
            window = history[i:i+slen]
            seq = tuple(a for _, a, _ in window)
            # Did the last action in the sequence produce a change?
            if window[-1][2]:  # changed
                self._successful_sequences[seq] += 1
            seq_attempts[seq] += 1

        # Rank sequences by success rate (with minimum attempts)
        scored = []
        for seq, successes in self._successful_sequences.items():
            attempts = seq_attempts.get(seq, 1)
            if attempts >= 3:
                rate = successes / attempts
                scored.append((seq, rate, attempts))

        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        self._best_sequences = [s[0] for s in scored[:10]]

    def get_sequence_suggestion(self, recent_actions: list) -> Optional[object]:
        """
        Given recent actions, suggest the next action to complete a known-good sequence.
        """
        if not self._best_sequences:
            return None

        slen = self._sequence_length
        if len(recent_actions) < slen - 1:
            return None

        recent = tuple(recent_actions[-(slen-1):])
        for seq in self._best_sequences:
            if seq[:slen-1] == recent:
                return seq[-1]  # complete the sequence

        return None

    # ===== State Classification (improved from v1) =====

    def _classify_state(self, state_name: str) -> tuple:
        parts = state_name.split("_")
        type_parts = []
        for p in parts:
            if p.startswith("L"):
                continue  # skip level for cross-level transfer
            type_parts.append(p)
        return tuple(type_parts) if type_parts else ("default",)

    # ===== Main Interface =====

    def rank_actions(self, state: str, available: set, recent_actions: list = None) -> list:
        """
        Rank available actions using all learned knowledge:
        1. Sequence completion (if in the middle of a known-good sequence)
        2. Feature-action rules (state features predict best action)
        3. State-dependent efficacy (what works in similar states)
        4. Global efficacy (what works overall)
        """
        if not self._action_ranking and not self._best_rules:
            return list(available)

        scored = defaultdict(float)

        # Layer 1: Sequence completion (highest priority)
        if recent_actions:
            seq_action = self.get_sequence_suggestion(recent_actions)
            if seq_action is not None and seq_action in available:
                scored[seq_action] += 100.0

        # Layer 2: Feature-action rules (only high confidence)
        rule_actions = self.get_rule_actions(state, available)
        for i, a in enumerate(rule_actions[:3]):  # top 3 only
            scored[a] += 8.0 / (i + 1)

        # Layer 3: State-dependent efficacy (need enough data)
        state_type = self._classify_state(state)
        if state_type in self._state_action_stats:
            sa_stats = self._state_action_stats[state_type]
            total = sum(s.attempts for s in sa_stats.values())
            if total >= 8:  # need enough state-specific data
                for a in available:
                    if a in sa_stats and sa_stats[a].attempts >= 4:
                        scored[a] += 5.0 * sa_stats[a].efficacy

        # Layer 4: Global efficacy (always available, low weight)
        for a in available:
            if a in self.action_stats and self.action_stats[a].attempts >= 5:
                scored[a] += 2.0 * self.action_stats[a].efficacy

        # Penalize actions that lead to predicted dead ends
        # (We don't know targets before acting, but we can down-rank
        # actions with historically low efficacy in this state type)

        # Sort by score
        ranked = sorted(available, key=lambda a: scored.get(a, 0), reverse=True)
        return ranked

    def score_state(self, state: str) -> float:
        """Score a state for frontier prioritization. Penalize predicted dead ends."""
        sig = self.state_sigs.get(state)
        score = 1.0  # base

        if sig:
            score += sig.depth * 0.5
            if sig.total_tested == 0:
                score += 3.0
            if sig.is_dead_end:
                score -= 5.0  # avoid known dead ends

        # Only use predicted dead-end penalty when model is confident
        dead_prob = self.predict_dead_end(state)
        if dead_prob > 0.5:
            score -= dead_prob * 3.0

        # Bottleneck bonus
        if state in self.bottlenecks:
            score += 3.0

        return score

    def record_win(self, path_states: list[str], path_actions: list[object]):
        self.winning_paths.append(path_states)
        self.winning_actions.append(path_actions)

    def classify_environment(self) -> dict:
        if not self.action_stats:
            return {"type": "unknown"}
        total_eff = {a: s.efficacy for a, s in self.action_stats.items()}
        avg_eff = sum(total_eff.values()) / max(len(total_eff), 1)
        n_effective = sum(1 for e in total_eff.values() if e > 0.1)
        n_states = len(self.state_sigs)
        n_dead = len(self.dead_end_states)
        return {
            "n_states": n_states,
            "n_dead_ends": n_dead,
            "dead_end_ratio": n_dead / max(n_states, 1),
            "n_effective_actions": n_effective,
            "avg_efficacy": avg_eff,
            "n_rules_learned": len(self._best_rules),
            "n_sequences_learned": len(self._best_sequences),
        }

    def get_summary(self) -> str:
        lines = [f"CompressionV2 (analyzed {self._analysis_count}x):"]
        if self._action_ranking:
            top3 = self._action_ranking[:3]
            lines.append(f"  Actions: {[f'{a}({self.action_stats[a].efficacy:.0%})' for a in top3]}")
        env = self.classify_environment()
        lines.append(f"  States: {env.get('n_states',0)}, Dead-ends: {env.get('n_dead_ends',0)} "
                    f"({env.get('dead_end_ratio',0):.0%})")
        lines.append(f"  Rules: {env.get('n_rules_learned',0)}, "
                    f"Sequences: {env.get('n_sequences_learned',0)}")
        if self._best_sequences:
            lines.append(f"  Top seq: {self._best_sequences[:3]}")
        return "\n".join(lines)
