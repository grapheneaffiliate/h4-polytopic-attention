"""
LLM Reasoner — calls Claude (or any LLM) when search gets stuck.

Three modes:
  1. --dry-run: prints prompts, returns empty suggestions (for testing)
  2. HeuristicARC: rule-based ARC-specific reasoning (no API needed)
  3. AnthropicReasoner: calls Claude API (needs ANTHROPIC_API_KEY)

The reasoner receives:
  - Current state (grid)
  - What's been tried (actions, rewards)
  - Training examples (for ARC tasks)

It returns action priors: {action: weight} that become soft UCB biases.

Usage:
  # Dry run (testing)
  reasoner = DryRunReasoner()

  # Heuristic (no API)
  reasoner = HeuristicARCReasoner(task_data)

  # Claude API
  reasoner = AnthropicReasoner(api_key="sk-...")
"""

import json
import hashlib
from typing import Optional
from .reasoner import Reasoner


class DryRunReasoner(Reasoner):
    """Prints prompts instead of calling an API. For testing."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.call_count = 0

    def suggest(self, context: dict) -> dict:
        self.call_count += 1
        if self.verbose:
            print(f"\n[DryRun Reasoner call #{self.call_count}]")
            print(f"  State: {str(context.get('state', '?'))[:80]}")
            print(f"  Stuck for: {context.get('actions_since_progress', 0)} actions")
            print(f"  States explored: {context.get('states_explored', 0)}")
        return {}


class HeuristicARCReasoner(Reasoner):
    """
    ARC-specific heuristic reasoner. No API needed.

    Analyzes training examples to infer:
    1. Output grid dimensions
    2. Common colors in output
    3. Spatial patterns (symmetry, tiling, color mapping)

    Returns action priors biased toward likely-correct cell values.
    """

    def __init__(self, task_data: dict):
        self.task = task_data
        self._analyzed = False
        self._output_colors = {}  # position -> most_common_color
        self._color_map = {}      # input_color -> output_color
        self._analyze()

    def _analyze(self):
        """Analyze training examples to extract patterns."""
        if self._analyzed:
            return
        self._analyzed = True

        # Color mapping: which input colors map to which output colors?
        color_transitions = {}  # (in_color, position_type) -> [out_colors]
        for example in self.task.get("train", []):
            inp = example["input"]
            out = example["output"]
            h_in, w_in = len(inp), len(inp[0]) if inp else 0
            h_out, w_out = len(out), len(out[0]) if out else 0

            # Direct color mapping (when grids are same size)
            if h_in == h_out and w_in == w_out:
                for r in range(h_in):
                    for c in range(w_in):
                        ic = inp[r][c]
                        oc = out[r][c]
                        if ic not in color_transitions:
                            color_transitions[ic] = []
                        color_transitions[ic].append(oc)

            # Output color frequencies
            for r in range(h_out):
                for c in range(w_out):
                    pos = (r % 3, c % 3)  # position modulo for tiling
                    if pos not in self._output_colors:
                        self._output_colors[pos] = {}
                    color = out[r][c]
                    self._output_colors[pos][color] = \
                        self._output_colors[pos].get(color, 0) + 1

        # Build color map: most common output per input
        for ic, ocs in color_transitions.items():
            from collections import Counter
            most_common = Counter(ocs).most_common(1)
            if most_common:
                self._color_map[ic] = most_common[0][0]

    def suggest(self, context: dict) -> dict:
        """
        Suggest actions based on ARC pattern analysis.
        Actions = row * W * 10 + col * 10 + color
        """
        suggestions = {}
        available = context.get("available_actions", [])

        if not available:
            return suggestions

        # Infer grid dimensions from action count
        n_actions = max(available) + 1 if available else 0
        # n_actions = H * W * 10
        n_cells = n_actions // 10
        # Try to figure out W and H
        # Use training output dims as hint
        if self.task.get("train"):
            sample_out = self.task["train"][0]["output"]
            H = len(sample_out)
            W = len(sample_out[0]) if sample_out else 1
        else:
            H = W = int(n_cells ** 0.5) or 1

        # Strategy 1: bias toward colors from color_map
        for action in available:
            color = action % 10
            remainder = action // 10
            col = remainder % W if W > 0 else 0
            row = remainder // W if W > 0 else 0

            # Bias toward output colors at this position (from training)
            pos = (row % 3, col % 3)
            if pos in self._output_colors:
                freq = self._output_colors[pos]
                if color in freq:
                    suggestions[action] = freq[color] * 2.0

            # Bias toward color-mapped values
            # (We'd need the current grid state to apply this, but context
            #  has the state as a string — parse it if needed)

        # Strategy 2: if very stuck, suggest actions that change to the most
        # common output color overall
        stuck = context.get("actions_since_progress", 0)
        if stuck > 300:
            all_colors = {}
            for pos_colors in self._output_colors.values():
                for c, freq in pos_colors.items():
                    all_colors[c] = all_colors.get(c, 0) + freq
            if all_colors:
                best_color = max(all_colors, key=all_colors.get)
                for action in available:
                    if action % 10 == best_color:
                        suggestions[action] = suggestions.get(action, 0) + 1.5

        return suggestions


class AnthropicReasoner(Reasoner):
    """
    Claude API-backed reasoner. Requires ANTHROPIC_API_KEY.

    Caches responses by context hash to avoid duplicate API calls.
    """

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514",
                 cache: bool = True):
        self.api_key = api_key
        self.model = model
        self._cache = {} if cache else None

    def suggest(self, context: dict) -> dict:
        # Check cache
        ctx_hash = hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()
        if self._cache is not None and ctx_hash in self._cache:
            return self._cache[ctx_hash]

        prompt = self._build_prompt(context)

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            suggestions = self._parse_response(text, context.get("available_actions", []))
        except Exception as e:
            suggestions = {}

        if self._cache is not None:
            self._cache[ctx_hash] = suggestions
        return suggestions

    def _build_prompt(self, context: dict) -> str:
        lines = [
            "You are helping a search agent explore an environment.",
            f"Current state (grid hash): {str(context.get('state', '?'))[:200]}",
            f"Available actions: {len(context.get('available_actions', []))} total",
            f"States explored: {context.get('states_explored', 0)}",
            f"Actions since progress: {context.get('actions_since_progress', 0)}",
            "",
            "Top actions by success rate:",
        ]
        for action, successes, attempts in context.get("top_actions", [])[:5]:
            rate = successes / max(attempts, 1)
            lines.append(f"  Action {action}: {successes}/{attempts} ({rate:.0%})")
        lines.extend([
            "",
            "Suggest 3-5 actions to try. Format: action_id: weight (1.0-5.0)",
            "Higher weight = stronger recommendation. Think about why.",
        ])
        return "\n".join(lines)

    def _parse_response(self, text: str, available: list) -> dict:
        suggestions = {}
        available_set = set(available)
        for line in text.strip().split("\n"):
            if ":" in line:
                try:
                    parts = line.split(":")
                    action = int(parts[0].strip())
                    weight = float(parts[1].strip().split()[0])
                    if action in available_set:
                        suggestions[action] = max(0, min(weight, 10.0))
                except (ValueError, IndexError):
                    continue
        return suggestions
