"""
Skills — persistent cross-run memory extracted from success and failure.

Inspired by MetaClaw: every failed run is training data, even without a model.
Convert failures into text-based rules, inject them next time, improve without retraining.

Skills persist in agent_zero/skills/ as markdown files.
Before each game, matching skills are loaded and converted to:
- UCB1 priors (boost/penalize specific actions)
- Mode overrides (start in grid instead of segment)
- Budget adjustments (give more/less time)
- Fallback triggers (switch mode at specific thresholds)

No LLM needed. Skills are hand-written or auto-extracted from run diagnostics.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


SKILLS_DIR = Path(__file__).parent / "skills"


@dataclass
class Skill:
    """One learned skill — a lesson from a past run."""
    name: str
    game_id: str = "global"          # specific game or "global"
    tags: list = field(default_factory=list)  # pattern tags for similarity matching
    confidence: float = 0.5          # 0-1, how sure we are
    runs_seen: int = 1               # how many runs support this

    # The lesson
    observation: str = ""            # what was observed
    lesson: str = ""                 # what it means
    action: str = ""                 # what to do about it

    # Concrete directives (machine-readable)
    start_mode: Optional[str] = None          # override starting mode
    switch_mode_at: Optional[int] = None      # switch mode at this action count
    switch_mode_to: Optional[str] = None      # target mode for switch
    dead_game_actions: list = field(default_factory=list)  # game action IDs to skip
    boost_game_actions: list = field(default_factory=list)  # game action IDs to prioritize
    budget_override: Optional[int] = None     # override budget
    min_efficacy: Optional[float] = None      # switch mode if efficacy below this

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "game_id": self.game_id,
            "tags": self.tags,
            "confidence": self.confidence,
            "runs_seen": self.runs_seen,
            "observation": self.observation,
            "lesson": self.lesson,
            "action": self.action,
        }
        if self.start_mode:
            d["start_mode"] = self.start_mode
        if self.switch_mode_at is not None:
            d["switch_mode_at"] = self.switch_mode_at
            d["switch_mode_to"] = self.switch_mode_to
        if self.dead_game_actions:
            d["dead_game_actions"] = self.dead_game_actions
        if self.boost_game_actions:
            d["boost_game_actions"] = self.boost_game_actions
        if self.budget_override:
            d["budget_override"] = self.budget_override
        if self.min_efficacy is not None:
            d["min_efficacy"] = self.min_efficacy
        return d


def parse_skill_file(path: Path) -> Optional[Skill]:
    """Parse a skill markdown file into a Skill object."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    skill = Skill(name=path.stem)

    # Parse YAML-ish frontmatter between --- markers
    fm_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.DOTALL)
    body = text
    if fm_match:
        frontmatter = fm_match.group(1)
        body = text[fm_match.end():]

        for line in frontmatter.strip().split("\n"):
            line = line.strip()
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip().lower()
            val = val.strip().strip('"').strip("'")

            if key == "game_id":
                skill.game_id = val
            elif key == "tags":
                skill.tags = [t.strip() for t in val.split(",")]
            elif key == "confidence":
                try:
                    skill.confidence = float(val)
                except ValueError:
                    pass
            elif key == "runs_seen":
                try:
                    skill.runs_seen = int(val)
                except ValueError:
                    pass
            elif key == "start_mode":
                skill.start_mode = val
            elif key == "switch_mode_at":
                try:
                    skill.switch_mode_at = int(val)
                except ValueError:
                    pass
            elif key == "switch_mode_to":
                skill.switch_mode_to = val
            elif key == "dead_game_actions":
                skill.dead_game_actions = [int(x.strip()) for x in val.split(",") if x.strip().isdigit()]
            elif key == "boost_game_actions":
                skill.boost_game_actions = [int(x.strip()) for x in val.split(",") if x.strip().isdigit()]
            elif key == "budget_override":
                try:
                    skill.budget_override = int(val)
                except ValueError:
                    pass
            elif key == "min_efficacy":
                try:
                    skill.min_efficacy = float(val)
                except ValueError:
                    pass

    # Parse body sections
    sections = {}
    current = None
    for line in body.split("\n"):
        header = re.match(r'^##\s+(.+)', line)
        if header:
            current = header.group(1).lower().strip()
            sections[current] = []
        elif current:
            sections[current].append(line)

    skill.observation = "\n".join(sections.get("observation", [])).strip()
    skill.lesson = "\n".join(sections.get("lesson", [])).strip()
    skill.action = "\n".join(sections.get("action", [])).strip()

    return skill


def load_skills(game_id: str = "", tags: Optional[list] = None) -> list[Skill]:
    """
    Load matching skills for a game.

    Matching order:
    1. Exact game_id match (highest priority)
    2. Tag overlap (pattern matching)
    3. Global skills (game_id == "global")
    """
    if not SKILLS_DIR.exists():
        return []

    all_skills = []
    for f in SKILLS_DIR.glob("*.md"):
        skill = parse_skill_file(f)
        if skill:
            all_skills.append(skill)

    if not all_skills:
        return []

    # Match
    game_short = game_id.split("-")[0] if game_id else ""
    matched = []

    for skill in all_skills:
        score = 0.0

        # Exact game match
        if skill.game_id == game_short or skill.game_id == game_id:
            score = 2.0
        # Global
        elif skill.game_id == "global":
            score = 0.5
        # Tag match
        elif tags:
            overlap = len(set(skill.tags) & set(tags))
            if overlap > 0:
                score = 0.5 + overlap * 0.3

        if score > 0:
            matched.append((score * skill.confidence, skill))

    # Sort by relevance * confidence
    matched.sort(key=lambda x: -x[0])
    return [skill for _, skill in matched]


class SkillInjector:
    """
    Applies loaded skills to agent configuration before and during a game.

    Usage:
        injector = SkillInjector(game_id="re86-...")
        injector.load()

        # Before game: get overrides
        start_mode = injector.get_start_mode()       # "grid_fine" or None
        budget = injector.get_budget_override()        # 350000 or None
        dead_actions = injector.get_dead_actions()     # [7] or []
        boost_actions = injector.get_boost_actions()   # [3, 5] or []

        # During game: check action-count triggers
        if injector.should_switch_mode(actions_taken=25000, current_mode="segment"):
            agent.switch_mode("grid_fine")
    """

    def __init__(self, game_id: str = "", game_tags: Optional[list] = None):
        self.game_id = game_id
        self.game_tags = game_tags or []
        self.skills: list[Skill] = []
        self.applied: list[str] = []  # track which skills were applied

    def load(self):
        """Load matching skills from disk."""
        self.skills = load_skills(self.game_id, self.game_tags)

    def get_start_mode(self) -> Optional[str]:
        """Get mode override for game start."""
        for skill in self.skills:
            if skill.start_mode:
                self.applied.append(f"start_mode={skill.start_mode} (from {skill.name})")
                return skill.start_mode
        return None

    def get_budget_override(self) -> Optional[int]:
        """Get budget override."""
        for skill in self.skills:
            if skill.budget_override:
                self.applied.append(f"budget={skill.budget_override} (from {skill.name})")
                return skill.budget_override
        return None

    def get_dead_actions(self) -> list[int]:
        """Get game action IDs that should be deprioritized."""
        dead = set()
        for skill in self.skills:
            if skill.dead_game_actions:
                dead.update(skill.dead_game_actions)
                self.applied.append(f"dead_actions={skill.dead_game_actions} (from {skill.name})")
        return list(dead)

    def get_boost_actions(self) -> list[int]:
        """Get game action IDs that should be prioritized."""
        boost = []
        seen = set()
        for skill in self.skills:
            for a in skill.boost_game_actions:
                if a not in seen:
                    boost.append(a)
                    seen.add(a)
            if skill.boost_game_actions:
                self.applied.append(f"boost_actions={skill.boost_game_actions} (from {skill.name})")
        return boost

    def should_switch_mode(self, actions_taken: int, current_mode: str,
                           efficacy: float = 1.0) -> Optional[str]:
        """Check if any skill says to switch mode at this point."""
        for skill in self.skills:
            # Action-count trigger
            if (skill.switch_mode_at is not None
                and actions_taken >= skill.switch_mode_at
                and skill.switch_mode_to
                and current_mode != skill.switch_mode_to):
                self.applied.append(
                    f"switch_mode at {actions_taken} → {skill.switch_mode_to} (from {skill.name})")
                return skill.switch_mode_to

            # Efficacy trigger
            if (skill.min_efficacy is not None
                and efficacy < skill.min_efficacy
                and actions_taken > 5000
                and skill.switch_mode_to
                and current_mode != skill.switch_mode_to):
                self.applied.append(
                    f"low_efficacy {efficacy:.3f} → {skill.switch_mode_to} (from {skill.name})")
                return skill.switch_mode_to

        return None

    def get_summary(self) -> dict:
        return {
            "skills_loaded": len(self.skills),
            "skills_applied": self.applied,
            "skill_names": [s.name for s in self.skills],
        }
