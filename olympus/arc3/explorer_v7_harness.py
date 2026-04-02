"""
ARC-AGI-3 Explorer v7 — v6 + active harness.

All v6 features PLUS actively driving decisions:
1. Action-effect memory → UCB1 priors (dead actions pruned, good actions boosted)
2. Evaluator → mode switching (faster reaction than v6's fixed thresholds)
3. Sprint contract → dead action removal from groups before exploration
4. Handoff → cross-episode learning (episode N reads episode N-1's memory)

Target: 50%+ (up from v6's 30/182 = 16.5%)
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
from agent_zero.skills import SkillInjector


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

        # Memory-driven priors (updated on each evaluation cycle)
        self._memory_priors: dict = {}  # {game_action_id: weight}

        # Track game_action_id -> set of action_idx mappings for prior injection
        self._game_action_to_indices: dict = defaultdict(set)  # {game_action_id: {idx, idx, ...}}

        # Evaluator-driven mode switch tracking
        self._evaluator_switched = False

        # Skills — persistent cross-run memory
        self.skill_injector = SkillInjector(game_id=game_id)
        self.skill_injector.load()

        # Apply skill overrides at init
        skill_start = self.skill_injector.get_start_mode()
        if skill_start and skill_start != self.mode:
            step = 2 if skill_start == "grid_fine" else 4
            self._switch_mode(skill_start, step)

        # Stash dead/boost from skills for use during exploration
        self._skill_dead_actions = set(self.skill_injector.get_dead_actions())
        self._skill_boost_actions = self.skill_injector.get_boost_actions()

    def choose_action(self, frame, available_actions, levels_completed):
        """Override to inject memory priors and evaluator-driven decisions."""
        # Level-up detection (before parent handles it)
        if levels_completed > self.current_level:
            self._on_level_complete_harness(frame)

        # Inject memory priors into GraphExplorer's UCB BEFORE parent chooses
        self._inject_memory_priors()

        # Prune dead actions from explorer node groups BEFORE parent chooses
        self._prune_dead_actions_from_groups(frame, available_actions)

        result = super().choose_action(frame, available_actions, levels_completed)

        # Store frame for memory recording
        self._prev_frame = frame.copy()

        # Track unique actions
        if isinstance(result, tuple) and len(result) >= 1:
            self._unique_actions_used.add(result[0])

        # Skill-based mode switching (check every 1000 actions, cheap)
        if self.total_actions_this_level % 1000 == 0 and not self._evaluator_switched:
            efficacy = 0.0
            if self._segment_actions_taken > 0:
                efficacy = self._segment_actions_changed / self._segment_actions_taken
            skill_switch = self.skill_injector.should_switch_mode(
                self.total_actions_this_level, self.mode, efficacy)
            if skill_switch:
                step = 2 if skill_switch == "grid_fine" else 4
                self._switch_mode(skill_switch, step)
                self._evaluator_switched = True

        # Evaluator check — actively drives mode switching
        if self.evaluator.should_evaluate(self.total_actions_this_level):
            self._run_evaluation()

        # Contract checkpoint
        if self.contract and self.total_actions_this_level == self.contract.checkpoint_at:
            self._contract_checkpoint()

        return result

    def _inject_memory_priors(self):
        """Inject memory-based priors into GraphExplorer's UCB1 reward tracking.

        For each action the memory knows about:
        - Dead actions (0% change rate, 10+ observations): inject negative rewards
        - High change rate actions: inject positive rewards as a prior
        This biases UCB1 without overriding it — the agent can still explore.
        """
        if not self.memory.action_models:
            return
        if self.total_actions_this_level < 50:
            return  # let the agent explore freely at start

        self.explorer._init_ucb()

        for game_action_id, model in self.memory.action_models.items():
            if model.total_observations < 5:
                continue

            # Find all action_idx that map to this game_action_id
            indices = self._game_action_to_indices.get(game_action_id, set())
            if not indices:
                continue

            for action_idx in indices:
                for state in list(self.explorer.nodes.keys())[-20:]:  # recent states only
                    sa = (state, action_idx)
                    existing = self.explorer._sa_visits.get(sa, 0)
                    if existing > 0:
                        continue  # don't override real observations

                    # Inject synthetic prior (1 observation worth)
                    if model.change_rate == 0 and model.total_observations >= 10:
                        # Dead action: inject 0 reward
                        self.explorer._sa_rewards[sa] = [0.0]
                        self.explorer._sa_visits[sa] = 3  # enough to trigger dead-action pruning
                        self.explorer._state_visits[state] = max(
                            self.explorer._state_visits.get(state, 0), 3)
                    elif model.change_rate > 0.3:
                        # Good action: inject positive reward as weak prior
                        self.explorer._sa_rewards[sa] = [model.change_rate]
                        self.explorer._sa_visits[sa] = 1
                        self.explorer._state_visits[state] = max(
                            self.explorer._state_visits.get(state, 0), 1)

    def _prune_dead_actions_from_groups(self, frame, available_actions):
        """Remove known-dead actions from explorer node groups.

        If memory says action X never changes the frame (10+ observations),
        remove all action_idx mapping to X from untested groups.
        This prevents the explorer from wasting time on them.
        """
        if not self.memory.action_models:
            return

        # Get dead game actions
        dead_game_actions = set()
        for game_action_id, model in self.memory.action_models.items():
            if model.change_rate == 0 and model.total_observations >= 15:
                dead_game_actions.add(game_action_id)

        if not dead_game_actions:
            return

        # Find action_idx values for dead game actions and remove from groups
        dead_indices = set()
        for ga in dead_game_actions:
            dead_indices.update(self._game_action_to_indices.get(ga, set()))

        if not dead_indices:
            return

        # Remove from current node's groups
        fh = hash_frame(frame, self.status_bar_mask) if self.status_bar_mask is not None else hash_frame(frame)
        if fh in self.explorer.nodes:
            node = self.explorer.nodes[fh]
            for g in node.groups:
                g -= dead_indices

    def observe_result(self, prev_hash, action_idx, new_frame, available_actions,
                      last_action_tuple=None):
        """Override to record action effects and build action mappings."""
        # Call parent
        super().observe_result(prev_hash, action_idx, new_frame, available_actions,
                              last_action_tuple)

        # Resolve game_action_id and build mapping
        game_action = 0
        if last_action_tuple and isinstance(last_action_tuple, tuple):
            game_action = last_action_tuple[0]
        elif action_idx is not None and prev_hash in self.segment_to_action:
            action_map = self.segment_to_action[prev_hash]
            if action_idx in action_map:
                game_action = action_map[action_idx][0]

        # Build game_action → action_idx mapping for prior injection
        if action_idx is not None and game_action > 0:
            self._game_action_to_indices[game_action].add(action_idx)

        # Record in memory
        if self._prev_frame is not None and action_idx is not None:
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
        self._evaluator_switched = False  # reset for next level

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

        # ACTIVE: evaluator drives mode switching
        # Override v6's fixed thresholds with memory-informed decisions
        if (result.should_switch_mode and result.suggested_mode
            and not self._evaluator_switched
            and self.mode != result.suggested_mode):
            # Evaluator says switch — do it, but only once per level
            self._switch_mode(result.suggested_mode,
                            2 if result.suggested_mode == "grid_fine" else self.grid_step)
            self._evaluator_switched = True

        # Update memory priors for next choose_action cycle
        if self.memory.action_models and self._prev_frame is not None:
            self._memory_priors = self.memory.suggest_actions(
                current_frame=self._prev_frame,
                available_actions=list(self._unique_actions_used),
                level=self.current_level,
            )

    def _contract_checkpoint(self):
        """Check progress against sprint contract and act on fallback strategy."""
        if self.contract is None:
            return

        states = self.explorer.num_states

        # Fallback: if contract says try different mode and we're stuck
        if (self.contract.fallback_mode
            and self.mode != self.contract.fallback_mode
            and not self._evaluator_switched
            and states < 30):
            self._switch_mode(
                self.contract.fallback_mode,
                2 if self.contract.fallback_mode == "grid_fine" else self.grid_step)
            self._evaluator_switched = True

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

        stats["skills"] = self.skill_injector.get_summary()

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
        "skills_applied": agent.skill_injector.applied,
        # Full diagnostics for meta-harness optimization
        "diagnostics": {
            "action_models": agent.memory.get_action_summary(),
            "evaluator_trend": agent.evaluator.get_trend(10),
            "handoff": game_handoff.to_dict(),
            "contract": agent.contract.to_dict() if agent.contract else None,
            "dead_actions_pruned": list(agent.contract.dead_actions) if agent.contract else [],
            "evaluator_switched_mode": agent._evaluator_switched,
            "memory_stats": agent.memory.get_stats(),
        },
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

    # Write per-game diagnostic JSON (for meta-harness optimization)
    diag_dir = os.path.join(os.path.dirname(__file__), "..", "..", "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    import json
    gid_short = game_id.split("-")[0]
    diag_path = os.path.join(diag_dir, f"{gid_short}.json")
    try:
        with open(diag_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception:
        pass  # don't crash on diagnostic write failure

    return result


# v7 budgets: more aggressive than v6
# Give zero-score games more budget — memory might crack them
# Give high-potential games even more
V7_GAME_BUDGETS = {
    # Top performers — push harder
    "lp85": 400000,   # 5/8, might hit 6-7
    "dc22": 350000,   # 3/6 with UCB1, memory should help
    "lf52": 350000,   # 1/10, 10 levels available — huge upside
    "vc33": 300000,   # 3/7
    "ft09": 350000,   # 2/6, deep explorer
    # Zero-score games — memory + evaluator might flip them
    "re86": 350000,   # 0/8, 52K states — needs memory
    "wa30": 300000,   # 0/9, grid-fine mode
    "sb26": 300000,   # 0/8, shallow grid
    "sk48": 300000,   # 0/8, deep but stuck
    "g50t": 300000,   # 0/7, barely exploring
    # Near-solved — squeeze more levels
    "ar25": 250000,   # 2/8
    "m0r0": 250000,   # 2/6
    # Default: 200K (from GAME_BUDGETS)
}


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
            # Budget priority: skill override > V7 budget > V6 budget > default
            skill_check = SkillInjector(game_id=e.game_id)
            skill_check.load()
            skill_budget = skill_check.get_budget_override()
            budget = skill_budget or V7_GAME_BUDGETS.get(gid_short, GAME_BUDGETS.get(gid_short, max_actions))
            r = solve_game(arc, e.game_id, budget, verbose=False)
            lc = r.get("levels_completed", 0)
            tl = r.get("total_levels", 0)
            total_c += lc
            total_l += tl
            status = "WIN" if r.get("state") == "WIN" else f"{lc}/{tl}"
            t = time.time() - t0
            hyps = r.get("hypotheses", [])
            skills = r.get("skills_applied", [])
            print(f"{e.game_id}: {status} ({r['actions_used']} actions, "
                  f"{r['states_explored']} states, depth={r['max_depth']}, "
                  f"mode={r['mode']}, type={r.get('game_type', '?')}, "
                  f"hyps={len(hyps)}, skills={len(skills)}) [{t:.0f}s]", flush=True)
            if skills:
                for s in skills[:3]:
                    print(f"    SKILL: {s}", flush=True)
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

    # Write run summary JSON (meta-harness: uncompressed feedback for next iteration)
    import json
    summary = {
        "version": "v7.1",
        "total_levels_completed": total_c,
        "total_levels": total_l,
        "score_pct": round(total_c / max(total_l, 1) * 100, 1),
        "elapsed_seconds": round(elapsed),
        "game_types": dict(type_counts),
        "per_game": [],
    }
    for r in all_results:
        gid = r.get("game_id", "?")
        gid_short = gid.split("-")[0]
        summary["per_game"].append({
            "game_id": gid,
            "game_short": gid_short,
            "levels": f"{r.get('levels_completed', 0)}/{r.get('total_levels', 0)}",
            "actions": r.get("actions_used", 0),
            "states": r.get("states_explored", 0),
            "depth": r.get("max_depth", 0),
            "mode": r.get("mode", "?"),
            "game_type": r.get("game_type", "unknown"),
            "memory_records": r.get("memory_records", 0),
            "hypotheses_count": len(r.get("hypotheses", [])),
            "hypotheses": r.get("hypotheses", [])[:5],
            "evaluator_switched": r.get("diagnostics", {}).get("evaluator_switched_mode", False),
        })

    diag_dir = os.path.join(os.path.dirname(__file__), "..", "..", "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    try:
        with open(os.path.join(diag_dir, "_run_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
    except Exception:
        pass

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
