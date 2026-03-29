"""
Compression Layer — learns patterns from the exploration graph.

This is the core innovation: instead of treating every state as equally unknown,
the compression layer discovers:
1. Action efficacy (which action types actually work in this environment)
2. State signatures (fingerprints that cluster states into types)
3. Bottlenecks (key decision points on the path to solutions)
4. Progress gradients (which direction leads to solutions)
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class ActionStats:
    """Tracks success rate for an action type across all states."""
    attempts: int = 0
    successes: int = 0

    @property
    def efficacy(self) -> float:
        if self.attempts == 0:
            return 0.5  # prior: unknown = maybe useful
        return self.successes / self.attempts

    @property
    def confidence(self) -> float:
        """How confident are we in the efficacy estimate? [0, 1]"""
        return 1.0 - 1.0 / (1.0 + self.attempts)


@dataclass
class StateSig:
    """Signature (fingerprint) of a state based on its transition behavior."""
    change_rate: float    # fraction of tested actions that changed state
    fanout: int           # number of distinct successors
    depth: int            # graph depth from root
    total_tested: int     # how many actions tested here


class CompressionLayer:
    """
    Learns patterns from the exploration graph to guide future search.

    Call `analyze(explorer)` periodically. Then use:
    - `rank_actions(state, available)` for action ordering
    - `score_state(state)` for frontier prioritization
    - `get_bottlenecks()` for key decision points
    """

    def __init__(self, learning_rate: float = 0.1):
        self.action_stats: dict[object, ActionStats] = defaultdict(ActionStats)
        self.state_sigs: dict[str, StateSig] = {}
        self.bottlenecks: set[str] = set()
        self.winning_paths: list[list[str]] = []  # state sequences that led to wins
        self.winning_actions: list[list[object]] = []  # action sequences that led to wins
        self.winning_sigs: list[list[StateSig]] = []  # signature sequences along wins
        self._analysis_count = 0
        self.learning_rate = learning_rate

        # Learned patterns
        self._action_ranking: list[object] = []  # actions sorted by global efficacy
        self._progress_direction: Optional[dict] = None  # what "progress" looks like

        # State-dependent action learning: which actions work in which state types
        # Key = state_type (discretized signature), Value = ActionStats per action
        self._state_action_stats: dict[tuple, dict] = defaultdict(lambda: defaultdict(ActionStats))

    def analyze(self, explorer):
        """
        Scan the graph and update all learned patterns.
        O(V + E) — call every ~1000 actions.
        """
        self._analysis_count += 1

        # Reset per-analysis stats
        action_counts = defaultdict(lambda: [0, 0])  # action -> [attempts, successes]

        for name, node in explorer.nodes.items():
            # 1. Compute state signature
            self.state_sigs[name] = StateSig(
                change_rate=node.change_rate,
                fanout=node.fanout,
                depth=node.depth,
                total_tested=len(node.tested),
            )

            # 2. Accumulate action stats
            for action, (changed, target) in node.tested.items():
                action_counts[action][0] += 1
                if changed:
                    action_counts[action][1] += 1

        # Update global action efficacy with exponential moving average
        for action, (attempts, successes) in action_counts.items():
            stats = self.action_stats[action]
            if stats.attempts == 0:
                stats.attempts = attempts
                stats.successes = successes
            else:
                lr = self.learning_rate
                stats.attempts = int(stats.attempts * (1 - lr) + attempts * lr)
                stats.successes = int(stats.successes * (1 - lr) + successes * lr)
                stats.attempts = max(stats.attempts, 1)

        # 2b. State-dependent action learning
        #     Classify each state into a "type" based on its name pattern,
        #     then learn which actions work for each type.
        #     Rebuild from scratch each time (avoid double-counting).
        self._state_action_stats.clear()
        for name, node in explorer.nodes.items():
            state_type = self._classify_state(name)
            for action, (changed, target) in node.tested.items():
                sa = self._state_action_stats[state_type][action]
                sa.attempts += 1
                if changed:
                    sa.successes += 1

        # 3. Rank actions by efficacy
        self._action_ranking = sorted(
            self.action_stats.keys(),
            key=lambda a: (self.action_stats[a].efficacy, -self.action_stats[a].attempts),
            reverse=True,
        )

        # 4. Find bottlenecks: nodes with exactly 1 novel successor
        self.bottlenecks.clear()
        for name, node in explorer.nodes.items():
            novel_successors = set()
            for action, (changed, target) in node.tested.items():
                if changed and target is not None and target != name:
                    novel_successors.add(target)
            if len(novel_successors) == 1 and node.depth > 0:
                self.bottlenecks.add(name)

        # 5. Compute progress direction from winning paths
        if self.winning_sigs:
            self._compute_progress_direction()

    def record_win(self, path_states: list[str], path_actions: list[object]):
        """Record a winning path (state sequence + action sequence)."""
        self.winning_paths.append(path_states)
        self.winning_actions.append(path_actions)
        # Record signatures along winning path
        sigs = [self.state_sigs.get(s) for s in path_states]
        sigs = [s for s in sigs if s is not None]
        if sigs:
            self.winning_sigs.append(sigs)

    def _classify_state(self, state_name: str) -> tuple:
        """
        Classify a state into a "type" for state-dependent action learning.

        Extracts structural features from the state name. In ARC-AGI-3 this would
        use frame features; here we use the state name pattern as a proxy.
        The key insight: states with similar names tend to need similar actions.
        """
        # Extract any embedded identifiers (e.g., "L0_P2_C3" -> type based on C3)
        parts = state_name.split("_")
        # Use the last distinctive part as the type key
        # This captures things like color (C3), position type, etc.
        type_parts = []
        for p in parts:
            if p.startswith("C") or p.startswith("B"):
                type_parts.append(p)
            elif p.startswith("L"):
                pass  # skip level (we want cross-level transfer)
            else:
                type_parts.append(p)
        return tuple(type_parts) if type_parts else ("default",)

    def rank_actions(self, state: str, available: set) -> list:
        """
        Return available actions ordered by learned efficacy.

        Uses state-dependent ranking when available (better for environments
        where the correct action depends on state features).
        Falls back to global ranking, then random.
        """
        if not self._action_ranking and not self._state_action_stats:
            return list(available)

        # Try state-dependent ranking first
        state_type = self._classify_state(state)
        if state_type in self._state_action_stats:
            sa_stats = self._state_action_stats[state_type]
            # Only use if we have enough data for this state type
            total_attempts = sum(s.attempts for s in sa_stats.values())
            if total_attempts >= 3:
                ranked = sorted(
                    [a for a in available if a in sa_stats],
                    key=lambda a: (sa_stats[a].efficacy, -sa_stats[a].attempts),
                    reverse=True,
                )
                remaining = [a for a in available if a not in sa_stats]
                if ranked:
                    return ranked + remaining

        # Fall back to global ranking
        ranked = [a for a in self._action_ranking if a in available]
        remaining = [a for a in available if a not in self.action_stats]
        return ranked + remaining

    def score_state(self, state: str) -> float:
        """
        Score a state for frontier prioritization.
        Higher = more promising to explore.
        """
        sig = self.state_sigs.get(state)
        if sig is None:
            return 1.0  # unknown states are interesting

        score = 0.0

        # Deeper states are more promising (closer to solution)
        score += sig.depth * 1.0

        # Untested states (total_tested == 0) are highly promising
        if sig.total_tested == 0:
            score += 3.0
        else:
            # States with moderate change rate are more promising than
            # fully explored or dead-end states
            score += (1.0 - abs(sig.change_rate - 0.5)) * 2.0

        # States similar to winning path states get a bonus
        if self._progress_direction:
            similarity = self._similarity_to_progress(sig)
            score += similarity * 5.0

        # Bottleneck bonus: exploring past bottlenecks is high-value
        if state in self.bottlenecks:
            score += 3.0

        return score

    def classify_environment(self) -> dict:
        """
        Infer what kind of environment this is based on learned patterns.
        Returns a dict of environment properties.
        """
        if not self.action_stats:
            return {"type": "unknown", "confidence": 0.0}

        total_efficacy = {}
        for action, stats in self.action_stats.items():
            total_efficacy[action] = stats.efficacy

        avg_efficacy = sum(total_efficacy.values()) / max(len(total_efficacy), 1)
        max_efficacy = max(total_efficacy.values()) if total_efficacy else 0
        n_effective = sum(1 for e in total_efficacy.values() if e > 0.1)

        # State space analysis
        n_states = len(self.state_sigs)
        avg_change = sum(s.change_rate for s in self.state_sigs.values()) / max(n_states, 1)
        max_depth = max((s.depth for s in self.state_sigs.values()), default=0)

        return {
            "n_states": n_states,
            "max_depth": max_depth,
            "avg_change_rate": avg_change,
            "avg_action_efficacy": avg_efficacy,
            "n_effective_actions": n_effective,
            "n_bottlenecks": len(self.bottlenecks),
            "has_winning_data": len(self.winning_paths) > 0,
        }

    def _compute_progress_direction(self):
        """
        From winning paths, learn what "progress" looks like as a signature trajectory.
        """
        if not self.winning_sigs:
            self._progress_direction = None
            return

        # Average the signature deltas along winning paths
        n = 0
        avg_delta_cr = 0.0
        avg_delta_fanout = 0.0
        avg_depth_at_win = 0.0

        for sigs in self.winning_sigs:
            if len(sigs) < 2:
                continue
            for i in range(1, len(sigs)):
                avg_delta_cr += sigs[i].change_rate - sigs[i-1].change_rate
                avg_delta_fanout += sigs[i].fanout - sigs[i-1].fanout
                n += 1
            avg_depth_at_win += sigs[-1].depth

        if n > 0:
            self._progress_direction = {
                "delta_change_rate": avg_delta_cr / n,
                "delta_fanout": avg_delta_fanout / n,
                "typical_win_depth": avg_depth_at_win / len(self.winning_sigs),
            }
        else:
            self._progress_direction = None

    def _similarity_to_progress(self, sig: StateSig) -> float:
        """How similar is this state's signature to the progress direction?"""
        if self._progress_direction is None:
            return 0.0

        # Score based on depth proximity to typical win depth
        win_depth = self._progress_direction["typical_win_depth"]
        if win_depth > 0:
            depth_sim = 1.0 - abs(sig.depth - win_depth) / (win_depth + 1)
            return max(0.0, depth_sim)
        return 0.0

    def get_summary(self) -> str:
        """Human-readable summary of what's been learned."""
        lines = [f"CompressionLayer (analyzed {self._analysis_count}x):"]

        if self._action_ranking:
            top3 = self._action_ranking[:3]
            lines.append(f"  Top actions: {[f'{a}({self.action_stats[a].efficacy:.0%})' for a in top3]}")

        lines.append(f"  States: {len(self.state_sigs)}, Bottlenecks: {len(self.bottlenecks)}")

        if self._progress_direction:
            pd = self._progress_direction
            lines.append(f"  Progress: depth~{pd['typical_win_depth']:.0f}, "
                        f"Δcr={pd['delta_change_rate']:+.3f}")

        env = self.classify_environment()
        lines.append(f"  Env: {env['n_effective_actions']} effective actions, "
                    f"avg_change={env['avg_change_rate']:.1%}")

        return "\n".join(lines)
