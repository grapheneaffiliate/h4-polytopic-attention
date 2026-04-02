"""
ARC-AGI-3 Explorer v7 — v6 + harness (memory, evaluator, sprint contract, handoff).

All v6 features PLUS:
1. Action-effect memory: remembers what actions DO across episodes
2. Evaluator: grades exploration every 100 actions, recommends changes
3. Sprint contract: negotiated plan before each level
4. Handoff: structured state compression between episodes

The harness is the skeleton that the 3B model plugs into.
All four components are heuristic-based now.
"""

import hashlib
import math
import random
import sys
import os
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

# Import v6 components (reuse everything except solve_game and run_all)
from olympus.arc3.explorer_v6_adaptive import (
    hash_frame,
    segment_frame,
    detect_status_bar_mask,
    classify_segments,
    NodeInfo,
    GraphExplorer,
    UnifiedAgentV6,
    INFINITY,
    GAME_BUDGETS,
)

# Import harness modules
from agent_zero.memory import ActionEffectMemory, extract_frame_features
from agent_zero.evaluator import Evaluator
from agent_zero.sprint_contract import SprintPlanner, SprintContract
from agent_zero.handoff import HandoffManager


class HarnessAgent(UnifiedAgentV6):
    """
    V6 agent + harness integration.

    Adds:
    - Memory recording on every observe_result
    - Evaluator check every 100 actions
    - Sprint contract on level transitions
    - Handoff on episode resets
    """

    def __init__(self, game_id: str = "", total_levels: int = 0, grid_step: int = 4):
        super().__init__(grid_step=grid_step)

        # Harness components
        self.memory = ActionEffectMemory(max_observations=20000)
        self.evaluator = Evaluator(eval_interval=500)  # every 500 actions
        self.planner = SprintPlanner()
        self.handoff_mgr = HandoffManager(game_id=game_id, total_levels=total_levels)

        # Current contract
        self.contract: Optional[SprintContract] = None

        # Frame cache for memory (prev_frame storage)
        self._prev_frame: Optional[np.ndarray] = None
        self._episode_actions = 0
        self._episode_new_states = 0
        self._unique_actions_used: set = set()
        self._recent_progress: list = []  # new states per 500-action window
        self._window_new_states = 0

        # Solved level frames for goal inference
        self._level_start_frames: dict = {}  # {level: frame}
        self._level_end_frames: dict = {}    # {level: frame}

    def choose_action(self, frame, available_actions, levels_completed):
        """Override to add harness hooks."""
        # Level-up detection (before parent handles it)
        if levels_completed > self.current_level:
            self._on_level_complete_harness(frame)

        result = super().choose_action(frame, available_actions, levels_completed)

        # Store frame for memory recording
        self._prev_frame = frame.copy()

        # Track unique actions
        if isinstance(result, tuple) and len(result) >= 1:
            self._unique_actions_used.add(result[0])

        # Evaluator check
        self.total_actions_this_level  # already incremented by parent
        if self.evaluator.should_evaluate(self.total_actions_this_level):
            self._run_evaluation()

        # Contract checkpoint
        if self.contract and self.total_actions_this_level == self.contract.checkpoint_at:
            self._contract_checkpoint()

        return result

    def observe_result(self, prev_hash, action_idx, new_frame, available_actions,
                      last_action_tuple=None):
        """Override to record action effects in memory."""
        # Call parent
        super().observe_result(prev_hash, action_idx, new_frame, available_actions,
                              last_action_tuple)

        # Record in memory
        if self._prev_frame is not None and action_idx is not None:
            # Use game action ID for memory (more meaningful than internal idx)
            game_action = 0
            if last_action_tuple and isinstance(last_action_tuple, tuple):
                game_action = last_action_tuple[0]
            elif action_idx is not None and prev_hash in self.segment_to_action:
                action_map = self.segment_to_action[prev_hash]
                if action_idx in action_map:
                    game_action = action_map[action_idx][0]

            self.memory.record(
                action=game_action,
                prev_frame=self._prev_frame,
                new_frame=new_frame,
                level=self.current_level,
                episode=self.retry_count,
            )

        # Track new states for evaluation
        new_hash = hash_frame(new_frame, self.status_bar_mask)
        if prev_hash != new_hash:
            self._episode_new_states += 1
            self._window_new_states += 1
        self._episode_actions += 1

        # Record start frame for goal inference
        if self.current_level not in self._level_start_frames:
            self._level_start_frames[self.current_level] = new_frame.copy()

    def on_game_over(self):
        """Override to add handoff on episode reset."""
        # Compute efficacy before parent resets state
        efficacy = 0.0
        if self._segment_actions_taken > 0:
            efficacy = self._segment_actions_changed / self._segment_actions_taken

        # Update handoff
        self.handoff_mgr.update_from_episode(
            level=self.current_level,
            states_explored=self.explorer.num_states,
            max_depth=self.explorer.max_depth,
            actions_taken=self._episode_actions,
            mode=self.mode,
            efficacy=efficacy,
            effective_history=list(self.effective_history),
        )

        # Update handoff from memory
        self.handoff_mgr.update_from_memory(self.memory, self.current_level)

        # Track progress windows
        self._recent_progress.append(self._window_new_states)
        self._window_new_states = 0

        # Reset episode counters
        self._episode_actions = 0
        self._episode_new_states = 0

        # Call parent
        super().on_game_over()

    def _on_level_complete_harness(self, frame):
        """Called when a level is completed, before parent processes it."""
        level = self.current_level

        # Record end frame
        self._level_end_frames[level] = frame.copy()

        # Notify handoff
        self.handoff_mgr.on_level_complete(
            level=level,
            winning_path=list(self.effective_history),
        )

        # Notify memory
        self.memory.on_level_complete(level)

        # Generate contract for next level
        prev_frames = {}
        for lvl in self._level_start_frames:
            if lvl in self._level_end_frames:
                prev_frames[lvl] = (self._level_start_frames[lvl],
                                    self._level_end_frames[lvl])

        self.contract = self.planner.plan(
            level=level + 1,
            game_id=self.handoff_mgr.game.game_id,
            memory=self.memory,
            solved_levels=self.winning_paths,
            prev_frames=prev_frames,
            total_budget=200000,  # will be overridden by GAME_BUDGETS
            current_mode=self.mode,
        )

        # Update handoff with contract goal
        self.handoff_mgr.update_goal_from_contract(level + 1, self.contract)

    def _run_evaluation(self):
        """Run the evaluator and potentially act on recommendations."""
        efficacy = 0.0
        if self._segment_actions_taken > 0:
            efficacy = self._segment_actions_changed / self._segment_actions_taken

        # Build context
        action_summaries = {}
        for action, model in self.memory.action_models.items():
            action_summaries[action] = model.to_dict()

        context = {
            "states_explored": self.explorer.num_states,
            "actions_taken": self.total_actions_this_level,
            "levels_completed": self.current_level,
            "total_levels": self.handoff_mgr.game.total_levels,
            "current_mode": self.mode,
            "efficacy": efficacy,
            "unique_actions_used": len(self._unique_actions_used),
            "available_action_count": max(len(self._unique_actions_used), 10),
            "action_models": action_summaries,
            "recent_progress": self._recent_progress[-10:],
            "max_depth": self.explorer.max_depth,
            "budget_total": 200000,
        }

        result = self.evaluator.evaluate(context)

        # Act on strong signals only (don't override v6's own mode switching)
        if result.should_switch_mode and result.suggested_mode:
            # Only switch if v6 hasn't already switched and we have enough evidence
            if self.total_actions_this_level > 10000:
                # Let v6's own logic handle it — evaluator's signal is informational
                pass

        # Memory-based action suggestions feed into UCB priors
        if self.memory.action_models:
            priors = self.memory.suggest_actions(
                current_frame=self._prev_frame if self._prev_frame is not None else np.zeros((1, 1)),
                available_actions=list(self._unique_actions_used),
                level=self.current_level,
            )
            # These priors are available but we don't force them —
            # the UCB1 in v6's GraphExplorer handles selection.
            # Future: wire these into GraphExplorer._ucb1_select as soft priors.

    def _contract_checkpoint(self):
        """Check progress against sprint contract."""
        if self.contract is None:
            return

        levels = self.current_level
        states = self.explorer.num_states

        # If contract says abort and we have no levels, consider giving up
        if (self.total_actions_this_level >= self.contract.abort_at and
            levels == 0 and states < 50):
            # Signal to the evaluator but don't force abort
            pass

    def get_harness_stats(self) -> dict:
        """Get stats from all harness components."""
        stats = {
            "memory": self.memory.get_stats(),
            "hypotheses": self.memory.compile_hypotheses(),
        }

        eval_result = self.evaluator.get_latest()
        if eval_result:
            stats["evaluation"] = eval_result.to_dict()

        if self.contract:
            stats["contract"] = self.contract.to_dict()

        handoff = self.handoff_mgr.get_game_handoff()
        stats["handoff"] = {
            "game_type": handoff.game_type,
            "levels_completed": handoff.levels_completed,
            "total_episodes": handoff.total_episodes,
        }

        return stats


# -- Game Runner (v7) --------------------------------------------------------

def solve_game(arc, game_id, max_actions=200000, verbose=True):
    """Solve a game using HarnessAgent (v7)."""
    env = arc.make(game_id, render_mode=None)
    obs = env.reset()
    if not obs.frame:
        return {"game_id": game_id, "error": "no_frame"}

    frame = np.array(obs.frame[-1])

    # Create harness agent
    gid_short = game_id.split("-")[0]
    agent = HarnessAgent(
        game_id=game_id,
        total_levels=obs.win_levels or 10,
        grid_step=4,
    )

    if verbose:
        print(f"[{game_id}] Grid: {frame.shape}, "
              f"Actions: {sorted(obs.available_actions or [])}, "
              f"Levels: {obs.win_levels}")

    from arcengine import GameAction

    total_actions = 0
    prev_hash = None
    prev_action_idx = None
    prev_action_tuple = None

    while total_actions < max_actions:
        if obs.state.name == "WIN":
            break

        if obs.state.name in ["NOT_PLAYED", "GAME_OVER"]:
            if obs.state.name == "GAME_OVER":
                agent.observe_death(prev_hash, prev_action_idx)
                agent.on_game_over()
            obs = env.reset()
            if not obs.frame:
                break
            frame = np.array(obs.frame[-1])
            prev_hash = None
            prev_action_idx = None
            prev_action_tuple = None
            total_actions += 1
            continue

        frame = np.array(obs.frame[-1])
        available = obs.available_actions or [1, 2, 3, 4, 5]

        if prev_hash is not None and prev_action_idx is not None:
            agent.observe_result(prev_hash, prev_action_idx, frame, available,
                               last_action_tuple=prev_action_tuple)

        # Process frame once, get hash
        fh, _, _ = agent.process_frame(frame, available)
        game_action_id, x, y = agent.choose_action(frame, available, obs.levels_completed)

        # O(1) reverse lookup
        action_idx = agent.find_action_idx(fh, game_action_id, x, y)

        prev_hash = fh
        prev_action_idx = action_idx
        prev_action_tuple = (game_action_id, x, y)

        action = GameAction.from_id(game_action_id)
        if action.is_complex():
            action.set_data({"x": int(x), "y": int(y)})

        try:
            data = action.action_data.model_dump() if action.is_complex() else None
            obs = env.step(action, data=data)
        except Exception:
            total_actions += 1
            continue

        if obs is None:
            break
        total_actions += 1

        if verbose and total_actions % 5000 == 0:
            harness_stats = agent.get_harness_stats()
            hyps = harness_stats.get("hypotheses", [])
            eval_info = harness_stats.get("evaluation", {})
            overall = eval_info.get("grades", {}).get("overall", 0)
            print(f"  [{total_actions}] mode={agent.mode} states={agent.explorer.num_states} "
                  f"depth={agent.explorer.max_depth} levels={obs.levels_completed}/{obs.win_levels} "
                  f"eval={overall:.2f} hyps={len(hyps)} mem={agent.memory.total_records}")

    # Final handoff
    agent.handoff_mgr.infer_game_type()
    game_handoff = agent.handoff_mgr.get_game_handoff()

    result = {
        "game_id": game_id,
        "levels_completed": obs.levels_completed if obs else 0,
        "total_levels": obs.win_levels if obs else 0,
        "actions_used": total_actions,
        "states_explored": agent.explorer.num_states,
        "max_depth": agent.explorer.max_depth,
        "mode": agent.mode,
        "state": obs.state.name if obs else "UNKNOWN",
        # Harness extras
        "game_type": game_handoff.game_type,
        "memory_records": agent.memory.total_records,
        "hypotheses": agent.memory.compile_hypotheses(),
    }

    if verbose:
        print(f"\n  Harness summary for {game_id}:")
        print(f"    Game type: {game_handoff.game_type}")
        print(f"    Memory: {agent.memory.total_records} records")
        hyps = agent.memory.compile_hypotheses()
        if hyps:
            for h in hyps[:5]:
                print(f"    Hypothesis: {h}")
        eval_result = agent.evaluator.get_latest()
        if eval_result:
            print(f"    Last eval: {eval_result.grades.overall:.2f} — {eval_result.critique[:100]}")

    return result


def run_all(api_key, max_actions=200000, verbose=True):
    """Run all games with harness agent."""
    from arc_agi import Arcade
    import time
    arc = Arcade(arc_api_key=api_key)
    envs = arc.get_environments()
    total_l = total_c = 0
    t0 = time.time()
    all_results = []

    for e in envs:
        try:
            gid_short = e.game_id.split("-")[0]
            budget = GAME_BUDGETS.get(gid_short, max_actions)
            r = solve_game(arc, e.game_id, budget, verbose=False)
            lc = r.get("levels_completed", 0)
            tl = r.get("total_levels", 0)
            total_c += lc
            total_l += tl
            status = "WIN" if r.get("state") == "WIN" else f"{lc}/{tl}"
            t = time.time() - t0
            hyps = r.get("hypotheses", [])
            print(f"{e.game_id}: {status} ({r['actions_used']} actions, "
                  f"{r['states_explored']} states, depth={r['max_depth']}, "
                  f"mode={r['mode']}, type={r.get('game_type', '?')}, "
                  f"hyps={len(hyps)}) [{t:.0f}s]", flush=True)
            all_results.append(r)
        except Exception as ex:
            print(f"{e.game_id}: ERROR {ex}", flush=True)

    elapsed = time.time() - t0
    print(f"\nTOTAL: {total_c}/{total_l} levels ({total_c/max(total_l,1)*100:.1f}%) "
          f"in {elapsed:.0f}s", flush=True)

    # Summary: game types discovered
    type_counts = defaultdict(int)
    for r in all_results:
        gt = r.get("game_type", "unknown")
        type_counts[gt] += 1
    print(f"\nGame types: {dict(type_counts)}")

    return all_results


if __name__ == "__main__":
    api_key = os.environ.get("ARC_API_KEY", "58b421be-5980-4ee8-8e57-0f18dc9369f3")
    from arc_agi import Arcade
    arc = Arcade(arc_api_key=api_key)
    if len(sys.argv) > 1:
        game_id = sys.argv[1]
        max_a = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
        r = solve_game(arc, game_id, max_a)
        print(r)
    else:
        run_all(api_key)
