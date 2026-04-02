"""
Evaluator — the generator-evaluator separation pattern from Anthropic's harness design.

The explorer explores. The evaluator judges whether that exploration is productive.
Separation prevents the agent from praising its own mediocre work.

Runs every N actions (default 100). Outputs:
1. Grades: efficiency, diversity, mode appropriateness, progress
2. Critique: specific text about what's wrong
3. Recommendations: concrete changes (switch mode, change budget, try different actions)

Heuristic now. The 3B model fills this role later — same interface, different backend.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalGrades:
    """Numeric grades (0-1) for exploration quality."""
    exploration_efficiency: float = 0.0   # new states / actions taken
    action_diversity: float = 0.0         # unique actions used / available actions
    mode_appropriateness: float = 0.0     # is current mode working?
    progress_rate: float = 0.0            # levels solved / budget used
    overall: float = 0.0                  # weighted average

    def to_dict(self) -> dict:
        return {
            "exploration_efficiency": round(self.exploration_efficiency, 3),
            "action_diversity": round(self.action_diversity, 3),
            "mode_appropriateness": round(self.mode_appropriateness, 3),
            "progress_rate": round(self.progress_rate, 3),
            "overall": round(self.overall, 3),
        }


@dataclass
class EvalResult:
    """Full evaluation output."""
    grades: EvalGrades
    critique: str                          # What's wrong
    recommendations: list[str]             # What to change
    should_switch_mode: bool = False        # Strong signal: change mode
    suggested_mode: Optional[str] = None   # "segment", "grid", "grid_fine"
    should_increase_budget: bool = False
    should_abort: bool = False             # Give up on this level

    def to_dict(self) -> dict:
        return {
            "grades": self.grades.to_dict(),
            "critique": self.critique,
            "recommendations": self.recommendations,
            "should_switch_mode": self.should_switch_mode,
            "suggested_mode": self.suggested_mode,
            "should_increase_budget": self.should_increase_budget,
            "should_abort": self.should_abort,
        }


class Evaluator:
    """
    Heuristic evaluator. Judges exploration quality from stats.

    Interface matches what the 3B model will provide:
        result = evaluator.evaluate(context)
        if result.should_switch_mode:
            agent.switch_mode(result.suggested_mode)

    Context keys:
        states_explored: int
        actions_taken: int
        levels_completed: int
        total_levels: int
        current_mode: str ("segment", "grid", "grid_fine")
        actions_per_new_state: float (rolling average)
        unique_actions_used: int
        available_action_count: int
        efficacy: float (frame changes / actions)
        action_models: dict from memory module
        recent_progress: list[int] (new states per 100-action window)
        max_depth: int
        budget_remaining: int
        budget_total: int
    """

    def __init__(self, eval_interval: int = 100):
        self.eval_interval = eval_interval
        self._last_eval_action = 0
        self._eval_history: list[EvalResult] = []

    def should_evaluate(self, actions_taken: int) -> bool:
        """Check if it's time to evaluate."""
        return actions_taken - self._last_eval_action >= self.eval_interval

    def evaluate(self, context: dict) -> EvalResult:
        """Run evaluation on current exploration state."""
        self._last_eval_action = context.get("actions_taken", 0)

        grades = self._compute_grades(context)
        critique = self._generate_critique(context, grades)
        recommendations = self._generate_recommendations(context, grades)

        should_switch, suggested = self._should_switch_mode(context, grades)
        should_increase = self._should_increase_budget(context, grades)
        should_abort = self._should_abort(context, grades)

        result = EvalResult(
            grades=grades,
            critique=critique,
            recommendations=recommendations,
            should_switch_mode=should_switch,
            suggested_mode=suggested,
            should_increase_budget=should_increase,
            should_abort=should_abort,
        )
        self._eval_history.append(result)
        return result

    def _compute_grades(self, ctx: dict) -> EvalGrades:
        g = EvalGrades()

        # Exploration efficiency: new states per action
        states = ctx.get("states_explored", 0)
        actions = ctx.get("actions_taken", 1)
        raw_eff = states / max(actions, 1)
        # Normalize: 0.01 states/action is decent, 0.001 is bad
        g.exploration_efficiency = min(1.0, raw_eff / 0.01)

        # Action diversity: how many unique actions used vs available
        unique = ctx.get("unique_actions_used", 0)
        available = ctx.get("available_action_count", 1)
        g.action_diversity = unique / max(available, 1)

        # Mode appropriateness: is current mode's efficacy acceptable?
        efficacy = ctx.get("efficacy", 0.0)
        mode = ctx.get("current_mode", "segment")
        if mode == "segment":
            # Segment mode: 5%+ efficacy is fine
            g.mode_appropriateness = min(1.0, efficacy / 0.05)
        else:
            # Grid mode: 1%+ efficacy is fine (grid is naturally lower)
            g.mode_appropriateness = min(1.0, efficacy / 0.01)

        # Progress rate: levels solved relative to budget
        levels = ctx.get("levels_completed", 0)
        total_levels = ctx.get("total_levels", 1)
        budget_used = ctx.get("actions_taken", 0) / max(ctx.get("budget_total", 200000), 1)
        if total_levels > 0:
            level_rate = levels / total_levels
            # If we've used 50% budget and solved 50% levels, that's perfect
            g.progress_rate = min(1.0, level_rate / max(budget_used, 0.01))
        else:
            g.progress_rate = 0.0

        # Overall: weighted average
        g.overall = (
            g.exploration_efficiency * 0.3 +
            g.action_diversity * 0.1 +
            g.mode_appropriateness * 0.3 +
            g.progress_rate * 0.3
        )

        return g

    def _generate_critique(self, ctx: dict, grades: EvalGrades) -> str:
        """Generate specific critique text."""
        problems = []

        if grades.exploration_efficiency < 0.2:
            states = ctx.get("states_explored", 0)
            actions = ctx.get("actions_taken", 0)
            problems.append(
                f"Low exploration efficiency: {states} states from {actions} actions "
                f"({states/max(actions,1):.4f} states/action). "
                f"Most actions are wasted."
            )

        if grades.action_diversity < 0.3:
            unique = ctx.get("unique_actions_used", 0)
            available = ctx.get("available_action_count", 0)
            problems.append(
                f"Low action diversity: only {unique}/{available} actions used. "
                f"You keep clicking the same segments."
            )

        if grades.mode_appropriateness < 0.2:
            mode = ctx.get("current_mode", "segment")
            efficacy = ctx.get("efficacy", 0)
            problems.append(
                f"Mode '{mode}' is ineffective: {efficacy:.1%} efficacy. "
                f"Consider switching."
            )

        # Check for stall
        recent = ctx.get("recent_progress", [])
        if len(recent) >= 5 and all(p == 0 for p in recent[-5:]):
            problems.append(
                "Stalled: no new states in last 500 actions. "
                "Exploration is going nowhere."
            )

        if not problems:
            return "Exploration looks healthy."

        return " ".join(problems)

    def _generate_recommendations(self, ctx: dict, grades: EvalGrades) -> list[str]:
        """Generate actionable recommendations."""
        recs = []

        if grades.exploration_efficiency < 0.2:
            recs.append("Try unexplored action types instead of repeating known-dead actions.")

        if grades.mode_appropriateness < 0.2:
            mode = ctx.get("current_mode", "segment")
            if mode == "segment":
                recs.append("Switch to grid-click mode — segment clicking isn't working.")
            elif mode == "grid":
                recs.append("Try finer grid (step=2) or switch back to segment mode.")

        if grades.action_diversity < 0.3:
            recs.append("Increase exploration constant (C) to try more diverse actions.")

        # Check action models for insights
        action_models = ctx.get("action_models", {})
        dead_actions = [a for a, m in action_models.items()
                       if isinstance(m, dict) and m.get("change_rate", 1) == 0
                       and m.get("observations", 0) >= 10]
        if dead_actions:
            recs.append(f"Actions {dead_actions} are dead (0% change rate). Stop trying them.")

        useful_actions = [a for a, m in action_models.items()
                         if isinstance(m, dict) and m.get("change_rate", 0) > 0.3]
        if useful_actions:
            recs.append(f"Focus on actions {useful_actions} — they have high change rates.")

        return recs

    def _should_switch_mode(self, ctx: dict, grades: EvalGrades) -> tuple[bool, Optional[str]]:
        """Strong signal: should we switch exploration mode?"""
        if grades.mode_appropriateness >= 0.3:
            return False, None

        mode = ctx.get("current_mode", "segment")
        actions = ctx.get("actions_taken", 0)

        # Need enough data before recommending switch
        if actions < 5000:
            return False, None

        if mode == "segment":
            return True, "grid"
        elif mode == "grid":
            return True, "grid_fine"
        return False, None

    def _should_increase_budget(self, ctx: dict, grades: EvalGrades) -> bool:
        """Should we allocate more budget to this game?"""
        levels = ctx.get("levels_completed", 0)
        total = ctx.get("total_levels", 1)
        budget_frac = ctx.get("actions_taken", 0) / max(ctx.get("budget_total", 200000), 1)

        # If we're solving levels at a good rate and running out of budget
        if levels > 0 and levels < total and budget_frac > 0.7:
            return True
        return False

    def _should_abort(self, ctx: dict, grades: EvalGrades) -> bool:
        """Should we give up on this level?"""
        actions = ctx.get("actions_taken", 0)
        states = ctx.get("states_explored", 0)

        # Explored >50K actions with <10 states = hopeless
        if actions > 50000 and states < 10:
            return True

        # Budget 80%+ used, no levels, bad grades
        budget_frac = actions / max(ctx.get("budget_total", 200000), 1)
        levels = ctx.get("levels_completed", 0)
        if budget_frac > 0.8 and levels == 0 and grades.overall < 0.1:
            return True

        return False

    def get_latest(self) -> Optional[EvalResult]:
        """Get most recent evaluation."""
        return self._eval_history[-1] if self._eval_history else None

    def get_trend(self, n: int = 5) -> list[float]:
        """Get trend of overall grades (last N evaluations)."""
        return [e.grades.overall for e in self._eval_history[-n:]]
