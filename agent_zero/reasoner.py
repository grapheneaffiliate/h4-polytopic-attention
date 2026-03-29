"""
Reasoner module — pluggable intelligence for when search stalls.

When UCB1 has explored N states with no progress, the reasoner:
1. Summarizes what's been tried
2. Generates action recommendations
3. Returns them as soft priors for UCB1 (Bayesian update, not override)

Interface: Reasoner.suggest(context) -> dict[action, weight]

Implementations:
- HeuristicReasoner: rule-based, no external dependencies
- LLMReasoner: wraps a callable(prompt) -> response for LLM integration
"""

from abc import ABC, abstractmethod
from typing import Any


class Reasoner(ABC):
    """Base class for reasoning modules."""

    @abstractmethod
    def suggest(self, context: dict) -> dict:
        """
        Given context about the current state of exploration,
        return a dict of {action: weight} suggesting which actions to try.

        Context keys:
            state: current state string
            available_actions: list of available actions
            states_explored: number of unique states found
            level: current level number
            actions_since_progress: actions since last state change
            top_actions: list of (action, successes, attempts) tuples
            total_actions: total actions taken so far

        Returns: {action: float} where higher weight = more recommended
        """


class HeuristicReasoner(Reasoner):
    """
    Rule-based reasoner. No external dependencies.

    Heuristics:
    1. Prefer actions with high success rate (from top_actions)
    2. Prefer actions NOT in the bottom of efficacy ranking
    3. If stuck for very long, suggest unexplored action types
    """

    def suggest(self, context: dict) -> dict:
        suggestions = {}
        available = context.get("available_actions", [])
        top_actions = context.get("top_actions", [])
        stuck = context.get("actions_since_progress", 0)

        if not available:
            return suggestions

        # Heuristic 1: boost high-efficacy actions
        for action, successes, attempts in top_actions:
            if attempts > 0 and action in available:
                rate = successes / attempts
                suggestions[action] = rate * 2.0

        # Heuristic 2: if very stuck, try actions we haven't used much
        if stuck > 500:
            tried_actions = set(a for a, _, _ in top_actions)
            untried = [a for a in available if a not in tried_actions]
            for a in untried:
                suggestions[a] = 3.0  # strong bias toward novelty

        return suggestions


class LLMReasoner(Reasoner):
    """
    LLM-backed reasoner. Wraps any callable(prompt: str) -> str.

    Usage:
        reasoner = LLMReasoner(lambda prompt: my_llm_call(prompt))
        agent = AgentZero(reasoner=reasoner)

    The LLM receives a text summary of the exploration state and
    should return action recommendations as "action_id: weight" lines.
    """

    def __init__(self, llm_fn: callable):
        """llm_fn: callable that takes a prompt string, returns response string."""
        self.llm_fn = llm_fn

    def suggest(self, context: dict) -> dict:
        prompt = self._build_prompt(context)
        try:
            response = self.llm_fn(prompt)
            return self._parse_response(response, context.get("available_actions", []))
        except Exception:
            return {}

    def _build_prompt(self, context: dict) -> str:
        lines = [
            "You are helping an exploration agent decide which action to take.",
            f"Current state: {context.get('state', '?')}",
            f"Available actions: {context.get('available_actions', [])}",
            f"States explored so far: {context.get('states_explored', 0)}",
            f"Current level: {context.get('level', 0)}",
            f"Actions since last progress: {context.get('actions_since_progress', 0)}",
            "",
            "Top actions by success rate:",
        ]
        for action, successes, attempts in context.get("top_actions", []):
            rate = successes / max(attempts, 1)
            lines.append(f"  Action {action}: {successes}/{attempts} ({rate:.0%})")
        lines.extend([
            "",
            "Which actions should I try? Reply with lines like:",
            "action_id: weight",
            "where weight is 1.0-5.0 (higher = stronger recommendation)",
        ])
        return "\n".join(lines)

    def _parse_response(self, response: str, available: list) -> dict:
        suggestions = {}
        available_set = set(available)
        for line in response.strip().split("\n"):
            line = line.strip()
            if ":" in line:
                try:
                    parts = line.split(":")
                    action_str = parts[0].strip()
                    weight = float(parts[1].strip())
                    # Try to parse action as int
                    try:
                        action = int(action_str)
                    except ValueError:
                        action = action_str
                    if action in available_set:
                        suggestions[action] = max(0, min(weight, 10.0))
                except (ValueError, IndexError):
                    continue
        return suggestions
