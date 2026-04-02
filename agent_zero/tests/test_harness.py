"""
Tests for harness modules: memory, evaluator, sprint_contract, handoff.
"""

import numpy as np
import random
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from agent_zero.memory import (
    ActionEffectMemory,
    classify_frame_diff,
    extract_frame_features,
    EffectType,
    ActionModel,
    cosine_similarity,
)
from agent_zero.evaluator import Evaluator, EvalGrades, EvalResult
from agent_zero.sprint_contract import SprintPlanner, SprintContract
from agent_zero.handoff import HandoffManager, LevelHandoff, GameHandoff


# ── Memory Tests ────────────────────────────────────────────

class TestFrameFeatures:
    def test_feature_shape(self):
        frame = np.random.randint(0, 16, (32, 32))
        features = extract_frame_features(frame)
        assert features.shape == (20,)

    def test_features_normalized(self):
        frame = np.zeros((10, 10), dtype=int)
        features = extract_frame_features(frame)
        # Color histogram should sum to 1
        assert abs(features[:16].sum() - 1.0) < 1e-6

    def test_symmetric_frame_high_symmetry(self):
        frame = np.random.randint(0, 4, (10, 5))
        symmetric = np.hstack([frame, np.fliplr(frame)])
        features = extract_frame_features(symmetric)
        h_sym = features[16]
        assert h_sym > 0.9

    def test_uniform_frame_low_entropy(self):
        frame = np.ones((10, 10), dtype=int) * 3
        features = extract_frame_features(frame)
        entropy = features[18]
        assert entropy == 0.0


class TestFrameDiff:
    def test_no_change(self):
        frame = np.random.randint(0, 16, (10, 10))
        effect, mag = classify_frame_diff(frame, frame.copy())
        assert effect == EffectType.NO_CHANGE
        assert mag == 0.0

    def test_rotation_detected(self):
        frame = np.random.randint(0, 4, (10, 10))
        rotated = np.rot90(frame)
        effect, mag = classify_frame_diff(frame, rotated)
        assert effect == EffectType.ROTATION

    def test_different_shapes(self):
        f1 = np.zeros((10, 10), dtype=int)
        f2 = np.zeros((10, 15), dtype=int)
        effect, mag = classify_frame_diff(f1, f2)
        assert effect == EffectType.COMPLEX

    def test_large_change(self):
        f1 = np.zeros((10, 10), dtype=int)
        f2 = np.ones((10, 10), dtype=int) * 5
        effect, mag = classify_frame_diff(f1, f2)
        assert mag == 1.0


class TestActionModel:
    def test_empty_model(self):
        m = ActionModel(action=3)
        assert m.dominant_effect == EffectType.UNKNOWN
        assert m.consistency == 0.0
        assert m.change_rate == 0.0

    def test_consistent_action(self):
        m = ActionModel(action=3)
        for _ in range(10):
            m.record(EffectType.ROTATION, 0.3)
        assert m.dominant_effect == EffectType.ROTATION
        assert m.consistency == 1.0
        assert m.change_rate == 1.0

    def test_dead_action(self):
        m = ActionModel(action=5)
        for _ in range(10):
            m.record(EffectType.NO_CHANGE, 0.0)
        assert m.change_rate == 0.0
        assert m.dominant_effect == EffectType.NO_CHANGE

    def test_mixed_effects(self):
        m = ActionModel(action=1)
        for _ in range(7):
            m.record(EffectType.SHIFT, 0.1)
        for _ in range(3):
            m.record(EffectType.NO_CHANGE, 0.0)
        assert m.dominant_effect == EffectType.SHIFT
        assert m.change_rate == 0.7
        assert m.consistency == 0.7


class TestActionEffectMemory:
    def test_record_and_query(self):
        mem = ActionEffectMemory()
        f1 = np.zeros((10, 10), dtype=int)
        f2 = np.ones((10, 10), dtype=int) * 3
        mem.record(action=3, prev_frame=f1, new_frame=f2, level=0)
        assert mem.total_records == 1
        model = mem.get_action_model(3)
        assert model is not None
        assert model.total_observations == 1

    def test_suggest_actions(self):
        mem = ActionEffectMemory()
        f1 = np.zeros((10, 10), dtype=int)
        # Record action 3 as always changing
        for _ in range(5):
            f2 = np.random.randint(0, 4, (10, 10))
            mem.record(action=3, prev_frame=f1, new_frame=f2, level=0)
        # Record action 5 as dead
        for _ in range(10):
            mem.record(action=5, prev_frame=f1, new_frame=f1, level=0)

        weights = mem.suggest_actions(f1, [3, 5], level=0)
        assert weights[3] > weights[5]

    def test_compile_hypotheses(self):
        mem = ActionEffectMemory()
        f1 = np.zeros((10, 10), dtype=int)
        for _ in range(10):
            f2 = np.rot90(f1) if f1.shape[0] == f1.shape[1] else f1
            mem.record(action=3, prev_frame=f1, new_frame=f2, level=0)
        hyps = mem.compile_hypotheses()
        assert isinstance(hyps, list)

    def test_level_specific_models(self):
        mem = ActionEffectMemory()
        f1 = np.zeros((10, 10), dtype=int)
        f2 = np.ones((10, 10), dtype=int)
        mem.record(action=3, prev_frame=f1, new_frame=f2, level=0)
        mem.record(action=3, prev_frame=f1, new_frame=f1, level=1)

        model_l0 = mem.get_action_model(3, level=0)
        model_l1 = mem.get_action_model(3, level=1)
        # Both should exist but level models need 5+ obs to be preferred
        assert model_l0 is not None


class TestCosineSimilarity:
    def test_identical(self):
        a = np.array([1, 2, 3], dtype=float)
        assert abs(cosine_similarity(a, a) - 1.0) < 1e-6

    def test_orthogonal(self):
        a = np.array([1, 0], dtype=float)
        b = np.array([0, 1], dtype=float)
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_zero_vector(self):
        a = np.array([0, 0], dtype=float)
        b = np.array([1, 2], dtype=float)
        assert cosine_similarity(a, b) == 0.0


# ── Evaluator Tests ─────────────────────────────────────────

class TestEvaluator:
    def _make_context(self, **kwargs):
        ctx = {
            "states_explored": 100,
            "actions_taken": 10000,
            "levels_completed": 0,
            "total_levels": 6,
            "current_mode": "segment",
            "efficacy": 0.05,
            "unique_actions_used": 20,
            "available_action_count": 50,
            "action_models": {},
            "recent_progress": [10, 8, 5, 3, 1],
            "max_depth": 20,
            "budget_total": 200000,
        }
        ctx.update(kwargs)
        return ctx

    def test_healthy_exploration(self):
        ev = Evaluator()
        ctx = self._make_context(states_explored=500, efficacy=0.08)
        result = ev.evaluate(ctx)
        assert result.grades.overall > 0.3
        assert not result.should_abort

    def test_stuck_exploration(self):
        ev = Evaluator()
        ctx = self._make_context(
            states_explored=5,
            actions_taken=60000,
            efficacy=0.001,
            recent_progress=[0, 0, 0, 0, 0],
        )
        result = ev.evaluate(ctx)
        assert result.grades.exploration_efficiency < 0.2
        assert result.grades.mode_appropriateness < 0.1

    def test_should_abort(self):
        ev = Evaluator()
        ctx = self._make_context(
            states_explored=3,
            actions_taken=60000,
            levels_completed=0,
            efficacy=0.001,
        )
        result = ev.evaluate(ctx)
        assert result.should_abort

    def test_mode_switch_recommendation(self):
        ev = Evaluator()
        ctx = self._make_context(
            actions_taken=15000,
            efficacy=0.001,
            current_mode="segment",
        )
        result = ev.evaluate(ctx)
        assert result.should_switch_mode
        assert result.suggested_mode == "grid"

    def test_eval_interval(self):
        ev = Evaluator(eval_interval=100)
        assert ev.should_evaluate(100)
        assert not ev.should_evaluate(50)

    def test_trend(self):
        ev = Evaluator(eval_interval=1)
        for i in range(5):
            ctx = self._make_context(actions_taken=i * 100, states_explored=i * 10)
            ev.evaluate(ctx)
        trend = ev.get_trend(5)
        assert len(trend) == 5

    def test_dead_action_recommendation(self):
        ev = Evaluator()
        ctx = self._make_context(
            action_models={
                3: {"change_rate": 0, "observations": 15},
                5: {"change_rate": 0.5, "observations": 20},
            }
        )
        result = ev.evaluate(ctx)
        recs = " ".join(result.recommendations)
        assert "dead" in recs.lower() or "3" in recs


# ── Sprint Contract Tests ───────────────────────────────────

class TestSprintPlanner:
    def test_basic_contract(self):
        planner = SprintPlanner()
        contract = planner.plan(level=0, game_id="test")
        assert contract.level == 0
        assert contract.budget > 0
        assert contract.checkpoint_at > 0

    def test_contract_with_memory(self):
        planner = SprintPlanner()
        mem = ActionEffectMemory()
        f1 = np.zeros((10, 10), dtype=int)
        # Build some memory
        for _ in range(10):
            f2 = np.random.randint(0, 4, (10, 10))
            mem.record(action=3, prev_frame=f1, new_frame=f2, level=0)
        for _ in range(10):
            mem.record(action=5, prev_frame=f1, new_frame=f1, level=0)

        contract = planner.plan(level=1, memory=mem)
        assert 5 in contract.dead_actions
        assert 3 in contract.preferred_actions

    def test_contract_to_prompt(self):
        planner = SprintPlanner()
        contract = planner.plan(level=2, game_id="dc22")
        prompt = contract.to_prompt()
        assert "Level 2" in prompt
        assert "dc22" in prompt

    def test_confidence_with_solved_levels(self):
        planner = SprintPlanner()
        mem = ActionEffectMemory()
        f1 = np.zeros((10, 10), dtype=int)
        for _ in range(10):
            f2 = np.random.randint(0, 4, (10, 10))
            mem.record(action=3, prev_frame=f1, new_frame=f2, level=0)

        contract = planner.plan(
            level=1,
            memory=mem,
            solved_levels={0: [(f1, 3)]},
            prev_frames={0: (f1, np.rot90(f1))},
        )
        assert contract.confidence > 0


# ── Handoff Tests ───────────────────────────────────────────

class TestHandoffManager:
    def test_basic_handoff(self):
        mgr = HandoffManager(game_id="test", total_levels=6)
        mgr.update_from_episode(
            level=0,
            states_explored=100,
            max_depth=10,
            actions_taken=5000,
            mode="segment",
            efficacy=0.05,
        )
        lh = mgr.get_handoff(level=0)
        assert lh.states_explored == 100
        assert lh.episodes_attempted == 1

    def test_multiple_episodes(self):
        mgr = HandoffManager(game_id="test", total_levels=6)
        mgr.update_from_episode(level=0, states_explored=100, actions_taken=5000)
        mgr.update_from_episode(level=0, states_explored=200, actions_taken=3000)
        lh = mgr.get_handoff(level=0)
        assert lh.episodes_attempted == 2
        assert lh.states_explored == 200  # max
        assert lh.actions_taken == 8000   # sum

    def test_level_complete(self):
        mgr = HandoffManager(game_id="test", total_levels=6)
        mgr.on_level_complete(level=0, winning_path=[("s1", 3), ("s2", 5)])
        assert mgr.game.levels_completed == 1
        assert len(mgr.game.winning_patterns) == 1

    def test_handoff_to_prompt(self):
        mgr = HandoffManager(game_id="dc22", total_levels=6)
        mgr.update_from_episode(level=0, states_explored=1000, actions_taken=50000,
                               mode="segment", efficacy=0.03)
        lh = mgr.get_handoff(level=0)
        prompt = lh.to_prompt()
        assert "Level 0" in prompt
        assert "1000" in prompt

    def test_game_handoff_prompt(self):
        mgr = HandoffManager(game_id="dc22", total_levels=6)
        mgr.update_from_episode(level=0, states_explored=1000, actions_taken=50000)
        mgr.on_level_complete(level=0, winning_path=[("s1", 3)])
        game = mgr.get_game_handoff()
        prompt = game.to_prompt()
        assert "dc22" in prompt

    def test_failed_mode_tracking(self):
        mgr = HandoffManager(game_id="test", total_levels=6)
        mgr.update_from_episode(
            level=0,
            states_explored=5,
            actions_taken=10000,
            mode="segment",
            efficacy=0.001,
        )
        lh = mgr.get_handoff(level=0)
        assert "segment" in lh.failed_modes

    def test_memory_integration(self):
        mgr = HandoffManager(game_id="test", total_levels=6)
        mem = ActionEffectMemory()
        f1 = np.zeros((10, 10), dtype=int)
        for _ in range(10):
            f2 = np.random.randint(0, 4, (10, 10))
            mem.record(action=3, prev_frame=f1, new_frame=f2, level=0)

        mgr.update_from_memory(mem, level=1)
        lh = mgr.get_handoff(level=1)
        # Should have populated action model or goal hypothesis
        assert lh.goal_hypothesis != "unknown" or len(lh.action_model) > 0


# ── Integration ─────────────────────────────────────────────

class TestIntegration:
    """Test that all harness modules work together."""

    def test_full_loop(self):
        """Simulate a mini exploration loop with all harness components."""
        mem = ActionEffectMemory()
        ev = Evaluator(eval_interval=10)
        planner = SprintPlanner()
        handoff_mgr = HandoffManager(game_id="test", total_levels=3)

        # Simulate level 0
        f1 = np.zeros((10, 10), dtype=int)
        for i in range(20):
            f2 = np.random.randint(0, 4, (10, 10)) if i % 3 == 0 else f1
            action = random.choice([1, 2, 3, 4, 5])
            mem.record(action=action, prev_frame=f1, new_frame=f2, level=0)

        # Evaluate
        ctx = {
            "states_explored": 50,
            "actions_taken": 20,
            "levels_completed": 0,
            "total_levels": 3,
            "current_mode": "segment",
            "efficacy": 0.1,
            "unique_actions_used": 5,
            "available_action_count": 5,
            "action_models": {},
            "recent_progress": [5, 3, 2],
            "max_depth": 8,
            "budget_total": 200,
        }
        result = ev.evaluate(ctx)
        assert result.grades.overall >= 0

        # Level complete
        handoff_mgr.on_level_complete(0, winning_path=[("s1", 3), ("s2", 5)])
        handoff_mgr.update_from_memory(mem, level=1)

        # Plan next level
        contract = planner.plan(
            level=1,
            game_id="test",
            memory=mem,
            solved_levels={0: [("s1", 3)]},
            prev_frames={0: (f1, np.rot90(f1))},
        )
        assert contract.level == 1
        assert contract.confidence > 0

        # Handoff prompt is readable
        game_handoff = handoff_mgr.get_game_handoff()
        prompt = game_handoff.to_prompt()
        assert len(prompt) > 0


if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)
    pytest.main([__file__, "-v"])
