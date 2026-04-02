"""Local game runner for ARC-AGI-3 games.

Loads game classes from environment_files/ and provides a clean interface
for BFS/A* solvers to run games locally at full speed without the API.
"""

import copy
import glob
import hashlib
import importlib.util
import json
import os
import sys
from typing import Optional

import numpy as np
from arcengine import ARCBaseGame, ActionInput, GameAction, GameState


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_game_source(game_id: str, base_dir: str = None) -> str:
    """Find the Python source file for a game."""
    if base_dir is None:
        base_dir = os.path.join(REPO_ROOT, "environment_files")
    pattern = os.path.join(base_dir, game_id, "*", f"{game_id}.py")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No source file found for game {game_id}")
    return matches[0]


def find_game_metadata(game_id: str, base_dir: str = None) -> dict:
    """Load metadata.json for a game."""
    if base_dir is None:
        base_dir = os.path.join(REPO_ROOT, "environment_files")
    pattern = os.path.join(base_dir, game_id, "*", "metadata.json")
    matches = glob.glob(pattern)
    if not matches:
        return {}
    with open(matches[0]) as f:
        return json.load(f)


def load_game_class(game_id: str, base_dir: str = None) -> type:
    """Load a game class from its source file."""
    source_path = find_game_source(game_id, base_dir)
    module_name = f"arc_game_{game_id}"

    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Find the ARCBaseGame subclass
    for name in dir(module):
        obj = getattr(module, name)
        if (isinstance(obj, type)
                and issubclass(obj, ARCBaseGame)
                and obj is not ARCBaseGame):
            return obj

    raise ValueError(f"No ARCBaseGame subclass found in {source_path}")


def create_game(game_class: type) -> ARCBaseGame:
    """Create and reset a game instance."""
    game = game_class()
    reset_action = ActionInput(id=GameAction.RESET)
    game.perform_action(reset_action, raw=True)
    return game


def step_game(game: ARCBaseGame, action: ActionInput):
    """Perform one action and return the result."""
    return game.perform_action(action, raw=True)


def get_valid_actions(game: ARCBaseGame) -> list:
    """Get the finite set of valid actions at the current state."""
    return game._get_valid_actions()


def hash_state(game: ARCBaseGame) -> bytes:
    """Hash the current game state (frame + score + level + hidden state)."""
    frame = game.camera.render(game.current_level.get_sprites())
    h = hashlib.blake2b(frame.tobytes(), digest_size=16)
    h.update(game._score.to_bytes(2, 'little'))
    h.update(game._current_level_index.to_bytes(2, 'little'))
    # Include hidden state for games with non-visual state
    hidden = game._get_hidden_state()
    h.update(hidden.tobytes())
    return h.digest()


def clone_game(game: ARCBaseGame) -> ARCBaseGame:
    """Deep copy a game for tree search branching."""
    return copy.deepcopy(game)


def render_frame(game: ARCBaseGame) -> np.ndarray:
    """Render the current frame."""
    return game.camera.render(game.current_level.get_sprites())


def list_all_games(base_dir: str = None) -> list:
    """List all available game IDs."""
    if base_dir is None:
        base_dir = os.path.join(REPO_ROOT, "environment_files")
    games = []
    for d in sorted(glob.glob(os.path.join(base_dir, "*/"))):
        game_id = os.path.basename(os.path.normpath(d))
        games.append(game_id)
    return games


def action_to_dict(action: ActionInput) -> dict:
    """Serialize an ActionInput for storage."""
    d = {"id": action.id.value}
    if action.data:
        d["data"] = action.data
    return d


def dict_to_action(d: dict) -> ActionInput:
    """Deserialize an ActionInput from storage."""
    return ActionInput(id=GameAction.from_id(d["id"]), data=d.get("data", {}))


if __name__ == "__main__":
    # Quick test: load and probe LP85
    print("Loading LP85...")
    cls = load_game_class("lp85")
    game = create_game(cls)
    print(f"  Game ID: {game.game_id}")
    print(f"  Win score: {game.win_score}")
    print(f"  State: {game._state}")
    print(f"  Score: {game._score}")
    print(f"  Level: {game._current_level_index}")

    actions = get_valid_actions(game)
    print(f"  Valid actions: {len(actions)}")
    for a in actions[:5]:
        print(f"    {a.id.name} data={a.data}")

    h1 = hash_state(game)
    print(f"  State hash: {h1.hex()}")

    # Test deepcopy
    clone = clone_game(game)
    h2 = hash_state(clone)
    print(f"  Clone hash: {h2.hex()} (match={h1 == h2})")

    # Take one action on clone, verify original unchanged
    if actions:
        step_game(clone, actions[0])
        h3 = hash_state(clone)
        h4 = hash_state(game)
        print(f"  After action on clone: clone={h3.hex()}, original={h4.hex()}")
        print(f"  Original unchanged: {h4 == h1}")

    print("\nAll games:")
    for gid in list_all_games():
        try:
            meta = find_game_metadata(gid)
            ba = meta.get("baseline_actions", [])
            tag = meta.get("tags", ["?"])[0] if meta.get("tags") else "?"
            print(f"  {gid:5s}  {tag:15s}  levels={len(ba)}")
        except Exception as e:
            print(f"  {gid:5s}  ERROR: {e}")
