"""
Action-Effect Memory — remembers what actions DO across episodes and levels.

The explorer has amnesia: re86 explores 52K states but learns nothing about
WHAT actions do. This module fixes that.

Storage: (state_features, action, effect_type, magnitude)
Lookup: given current state features, find similar past states and what
        actions did there → return priors for UCB1.

Two tiers:
1. Exact action model: "action 3 always rotates" (compiled after N observations)
2. Similarity lookup: "states with high symmetry + action 3 → rotation" (cosine)

The 3B model eventually replaces tier 2 with learned embeddings.
For now: hand-crafted feature vectors + cosine similarity.
"""

import math
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ── Effect Types ────────────────────────────────────────────

class EffectType:
    """Categorical effect labels derived from frame diffs."""
    NO_CHANGE = "no_change"
    COLOR_CHANGE = "color_change"
    SHIFT = "shift"
    ROTATION = "rotation"
    GRAVITY = "gravity"
    FILL = "fill"
    SWAP = "swap"
    COMPLEX = "complex"
    UNKNOWN = "unknown"


@dataclass
class ActionEffect:
    """One observed (state, action) → effect."""
    action: int
    effect_type: str
    magnitude: float  # 0-1, how much of the frame changed
    state_features: Optional[np.ndarray] = None
    level: int = 0
    episode: int = 0


@dataclass
class ActionModel:
    """Compiled model for a single action: what it does across observations."""
    action: int
    effect_counts: dict = field(default_factory=lambda: defaultdict(int))
    total_observations: int = 0
    avg_magnitude: float = 0.0
    _magnitude_sum: float = 0.0

    @property
    def dominant_effect(self) -> str:
        if not self.effect_counts:
            return EffectType.UNKNOWN
        return max(self.effect_counts, key=self.effect_counts.get)

    @property
    def consistency(self) -> float:
        """How consistent is this action? 1.0 = always same effect."""
        if self.total_observations == 0:
            return 0.0
        top = max(self.effect_counts.values())
        return top / self.total_observations

    @property
    def change_rate(self) -> float:
        """How often does this action change the frame at all?"""
        if self.total_observations == 0:
            return 0.0
        no_change = self.effect_counts.get(EffectType.NO_CHANGE, 0)
        return 1.0 - (no_change / self.total_observations)

    def record(self, effect_type: str, magnitude: float):
        self.effect_counts[effect_type] += 1
        self.total_observations += 1
        self._magnitude_sum += magnitude
        self.avg_magnitude = self._magnitude_sum / self.total_observations

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "dominant_effect": self.dominant_effect,
            "consistency": round(self.consistency, 3),
            "change_rate": round(self.change_rate, 3),
            "avg_magnitude": round(self.avg_magnitude, 4),
            "observations": self.total_observations,
            "effects": dict(self.effect_counts),
        }


# ── Feature Extraction ─────────────────────────────────────

def extract_frame_features(frame: np.ndarray) -> np.ndarray:
    """
    Extract a fixed-size feature vector from a game frame.

    Features (20-dim):
    - Color histogram (16 bins, normalized)
    - Symmetry scores (horizontal, vertical)
    - Entropy
    - Edge density
    """
    H, W = frame.shape

    # Color histogram (16 bins)
    hist = np.bincount(frame.ravel().astype(int), minlength=16)[:16]
    hist = hist.astype(float)
    total = hist.sum()
    if total > 0:
        hist /= total

    # Horizontal symmetry
    flipped_h = np.fliplr(frame)
    h_sym = np.mean(frame == flipped_h)

    # Vertical symmetry
    flipped_v = np.flipud(frame)
    v_sym = np.mean(frame == flipped_v)

    # Shannon entropy
    probs = hist[hist > 0]
    entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 1 else 0.0
    entropy /= 4.0  # normalize to ~0-1 range

    # Edge density (adjacent pixels differ)
    h_edges = np.mean(frame[:, 1:] != frame[:, :-1]) if W > 1 else 0.0
    v_edges = np.mean(frame[1:, :] != frame[:-1, :]) if H > 1 else 0.0

    features = np.concatenate([
        hist,                          # 16
        [h_sym, v_sym],               # 2
        [entropy],                     # 1
        [(h_edges + v_edges) / 2],    # 1
    ])
    return features  # 20-dim


def classify_frame_diff(prev_frame: np.ndarray, new_frame: np.ndarray) -> tuple[str, float]:
    """
    Classify the effect of an action by comparing frames.

    Returns (effect_type, magnitude) where magnitude is fraction of pixels changed.
    """
    if prev_frame.shape != new_frame.shape:
        return EffectType.COMPLEX, 1.0

    diff = prev_frame != new_frame
    magnitude = np.mean(diff)

    if magnitude == 0:
        return EffectType.NO_CHANGE, 0.0

    if magnitude < 0.001:
        return EffectType.NO_CHANGE, magnitude

    changed_rows, changed_cols = np.where(diff)
    if len(changed_rows) == 0:
        return EffectType.NO_CHANGE, 0.0

    # Check for shift pattern: all changed pixels have same displacement
    # Compare a small sample for efficiency
    n_sample = min(50, len(changed_rows))
    if n_sample >= 3:
        idx = np.random.choice(len(changed_rows), n_sample, replace=False)
        sample_r = changed_rows[idx]
        sample_c = changed_cols[idx]

        # Check if prev_frame values at changed positions appear elsewhere in new_frame
        # This is a heuristic — not perfect
        prev_colors = set(int(prev_frame[r, c]) for r, c in zip(sample_r, sample_c))
        new_colors = set(int(new_frame[r, c]) for r, c in zip(sample_r, sample_c))

        color_overlap = len(prev_colors & new_colors) / max(len(prev_colors | new_colors), 1)

        if color_overlap > 0.8:
            # Colors preserved — likely a geometric transform
            # Check rotation: compare with 90° rotated version
            rot90 = np.rot90(prev_frame)
            if rot90.shape == new_frame.shape and np.mean(rot90 == new_frame) > 0.8:
                return EffectType.ROTATION, magnitude

            # Check shift: bbox of changes is rectangular and coherent
            r_range = changed_rows.max() - changed_rows.min()
            c_range = changed_cols.max() - changed_cols.min()
            area = (r_range + 1) * (c_range + 1)
            density = len(changed_rows) / max(area, 1)

            if density > 0.3:
                # Check gravity pattern: changes concentrated in rows
                row_counts = np.bincount(changed_rows, minlength=prev_frame.shape[0])
                col_counts = np.bincount(changed_cols, minlength=prev_frame.shape[1])

                row_spread = np.std(row_counts[row_counts > 0])
                col_spread = np.std(col_counts[col_counts > 0])

                if row_spread > col_spread * 2:
                    return EffectType.GRAVITY, magnitude

                return EffectType.SHIFT, magnitude

        elif len(new_colors) <= 2 and magnitude > 0.05:
            return EffectType.FILL, magnitude

        elif color_overlap < 0.3:
            return EffectType.COLOR_CHANGE, magnitude

    # Check swap: exactly 2 regions exchanged
    if 0.02 < magnitude < 0.5:
        unique_prev = set(int(x) for x in np.unique(prev_frame[diff]))
        unique_new = set(int(x) for x in np.unique(new_frame[diff]))
        if len(unique_prev) == 2 and unique_prev == unique_new:
            return EffectType.SWAP, magnitude

    if magnitude > 0.5:
        return EffectType.COMPLEX, magnitude

    return EffectType.UNKNOWN, magnitude


# ── Cosine Similarity ───────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ── Action-Effect Memory ───────────────────────────────────

class ActionEffectMemory:
    """
    Stores and retrieves action effects across episodes and levels.

    Two lookup modes:
    1. Action model: "what does action X do in general?"
    2. Similarity search: "what worked in states similar to this one?"

    Usage:
        memory = ActionEffectMemory()

        # Record observations
        memory.record(action=3, prev_frame=f1, new_frame=f2, level=0, episode=5)

        # Query action model
        model = memory.get_action_model(action=3)
        print(model.dominant_effect)  # "rotation"

        # Query by state similarity
        priors = memory.suggest_actions(current_frame, available_actions=[1,2,3,4,5])
        # Returns {action: weight} for UCB1
    """

    def __init__(self, max_observations: int = 2000, similarity_threshold: float = 0.85):
        self.max_observations = max_observations
        self.similarity_threshold = similarity_threshold

        # Tier 1: per-action compiled models (cheap — just counters)
        self.action_models: dict[int, ActionModel] = {}

        # Tier 2: sampled observations for similarity lookup (expensive — has features)
        # Only store 1 in N observations to keep memory bounded
        self.observations: list[ActionEffect] = []
        self._feature_matrix: Optional[np.ndarray] = None  # lazy-built
        self._feature_dirty = True
        self._sample_rate = 50  # store 1 in 50 observations for similarity

        # Per-level action models (don't pollute across very different games)
        self.level_action_models: dict[int, dict[int, ActionModel]] = defaultdict(dict)

        # Stats
        self.total_records = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def record(self, action: int, prev_frame: np.ndarray, new_frame: np.ndarray,
               level: int = 0, episode: int = 0):
        """Record an action's effect.

        Tier 1 (action models) records every observation — it's just counter updates.
        Tier 2 (similarity) samples 1 in N to keep memory bounded on Actions runners.
        """
        effect_type, magnitude = classify_frame_diff(prev_frame, new_frame)

        # Tier 1: always update action models (cheap — just counters)
        if action not in self.action_models:
            self.action_models[action] = ActionModel(action=action)
        self.action_models[action].record(effect_type, magnitude)

        if action not in self.level_action_models[level]:
            self.level_action_models[level][action] = ActionModel(action=action)
        self.level_action_models[level][action].record(effect_type, magnitude)

        # Tier 2: sample for similarity lookup (expensive — stores feature vector)
        self.total_records += 1
        if (len(self.observations) < self.max_observations
            and self.total_records % self._sample_rate == 0
            and effect_type != EffectType.NO_CHANGE):  # only store interesting effects
            features = extract_frame_features(prev_frame)
            obs = ActionEffect(
                action=action,
                effect_type=effect_type,
                magnitude=magnitude,
                state_features=features,
                level=level,
                episode=episode,
            )
            self.observations.append(obs)
            self._feature_dirty = True

    def get_action_model(self, action: int, level: Optional[int] = None) -> Optional[ActionModel]:
        """Get compiled action model. Level-specific if available."""
        if level is not None and level in self.level_action_models:
            model = self.level_action_models[level].get(action)
            if model and model.total_observations >= 5:
                return model
        return self.action_models.get(action)

    def suggest_actions(self, current_frame: np.ndarray,
                       available_actions: list[int],
                       level: int = 0,
                       prefer_effects: Optional[list[str]] = None) -> dict[int, float]:
        """
        Suggest action weights based on memory.

        Returns {action: weight} for UCB1 priors.

        Strategy:
        1. Boost actions with high change rate (they DO something)
        2. If prefer_effects specified, boost actions whose dominant effect matches
        3. Penalize actions with 0 change rate (dead actions)
        4. Use similarity to boost actions that worked in similar states
        """
        weights = {}

        for action in available_actions:
            model = self.get_action_model(action, level)
            if model is None or model.total_observations < 3:
                weights[action] = 0.0  # no data — neutral
                continue

            w = 0.0

            # Base weight from change rate
            w += model.change_rate * 2.0

            # Penalty for dead actions
            if model.change_rate == 0 and model.total_observations >= 10:
                w = -2.0

            # Bonus for preferred effects
            if prefer_effects and model.dominant_effect in prefer_effects:
                w += model.consistency * 3.0

            # Bonus for consistent actions (predictable = plannable)
            if model.consistency > 0.8:
                w += 1.0

            weights[action] = w

        # Tier 2: similarity-based lookup
        if self.observations and current_frame is not None:
            sim_weights = self._similarity_lookup(current_frame, available_actions)
            for action, sw in sim_weights.items():
                weights[action] = weights.get(action, 0.0) + sw * 0.5

        return weights

    def _similarity_lookup(self, current_frame: np.ndarray,
                           available_actions: list[int]) -> dict[int, float]:
        """Find similar past states and what actions worked there."""
        current_features = extract_frame_features(current_frame)

        # Build feature matrix if dirty
        if self._feature_dirty or self._feature_matrix is None:
            if self.observations:
                feats = [obs.state_features for obs in self.observations
                         if obs.state_features is not None]
                if feats:
                    self._feature_matrix = np.array(feats)
                else:
                    return {}
            else:
                return {}
            self._feature_dirty = False

        # Batch cosine similarity
        norms = np.linalg.norm(self._feature_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = self._feature_matrix / norms

        c_norm = np.linalg.norm(current_features)
        if c_norm == 0:
            return {}
        c_normalized = current_features / c_norm

        similarities = normalized @ c_normalized  # (N,)

        # Find similar states above threshold
        mask = similarities >= self.similarity_threshold
        if not mask.any():
            self.cache_misses += 1
            return {}

        self.cache_hits += 1

        # Aggregate: for each action, weight by similarity × magnitude
        action_scores = defaultdict(float)
        action_counts = defaultdict(int)

        indices = np.where(mask)[0]
        for idx in indices:
            obs = self.observations[idx]
            if obs.action in available_actions and obs.effect_type != EffectType.NO_CHANGE:
                sim = similarities[idx]
                action_scores[obs.action] += sim * obs.magnitude
                action_counts[obs.action] += 1

        # Normalize
        weights = {}
        for action in action_scores:
            if action_counts[action] > 0:
                weights[action] = action_scores[action] / action_counts[action]

        return weights

    def get_action_summary(self, level: Optional[int] = None) -> dict:
        """Human-readable summary of what we know about each action."""
        models = self.level_action_models.get(level, {}) if level is not None else {}
        if not models:
            models = self.action_models

        summary = {}
        for action, model in sorted(models.items()):
            summary[action] = model.to_dict()
        return summary

    def compile_hypotheses(self) -> list[str]:
        """
        Generate natural-language hypotheses about action effects.
        These become sprint contract inputs and 3B model prompts.
        """
        hypotheses = []
        for action, model in sorted(self.action_models.items()):
            if model.total_observations < 5:
                continue
            if model.consistency > 0.7 and model.dominant_effect != EffectType.NO_CHANGE:
                hypotheses.append(
                    f"Action {action} = {model.dominant_effect} "
                    f"({model.consistency:.0%} consistent, "
                    f"avg magnitude {model.avg_magnitude:.3f})"
                )
            elif model.change_rate == 0:
                hypotheses.append(f"Action {action} = dead (never changes frame)")
            elif model.change_rate < 0.1:
                hypotheses.append(
                    f"Action {action} = mostly inert "
                    f"({model.change_rate:.0%} change rate)"
                )
        return hypotheses

    def on_level_complete(self, level: int):
        """Called when a level is solved. Marks level models as validated."""
        pass  # Future: weight validated models higher in transfer

    def on_episode_reset(self):
        """Called on episode reset. No-op for now (memory persists)."""
        pass

    def get_stats(self) -> dict:
        return {
            "total_records": self.total_records,
            "observations_stored": len(self.observations),
            "actions_modeled": len(self.action_models),
            "levels_modeled": len(self.level_action_models),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }
