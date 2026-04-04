#!/usr/bin/env python3
"""
Build training data for 3B model fine-tuning from solved games.

Generates (frame_features, action) pairs from pre-computed solutions.
Each solved level provides a sequence of (state, optimal_action) demonstrations.

Output: JSONL file for fine-tuning, plus statistics.
"""

import json
import os
import sys
import glob
import hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_runner import load_game_class, create_game, step_game, find_game_metadata
from arcengine import ActionInput, GameAction, GameState

SOLUTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solutions")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def encode_frame(frame):
    """Encode a 64x64 frame into compact feature vector."""
    features = {}

    # Color histogram (13 bins)
    hist = np.bincount(frame.flatten().astype(int), minlength=13)[:13]
    features["color_hist"] = (hist / frame.size).tolist()

    # Quadrant histograms
    h, w = frame.shape
    for qi in range(2):
        for qj in range(2):
            quad = frame[qi*h//2:(qi+1)*h//2, qj*w//2:(qj+1)*w//2]
            qhist = np.bincount(quad.flatten().astype(int), minlength=13)[:13]
            features[f"quad_{qi}{qj}"] = (qhist / quad.size).tolist()

    # Spatial moments for top 3 colors
    flat_hist = np.bincount(frame.flatten().astype(int), minlength=13)[:13]
    top3 = np.argsort(-flat_hist)[:3].tolist()
    for c in top3:
        mask = (frame == c)
        if mask.sum() > 0:
            ys, xs = np.where(mask)
            features[f"color_{c}_center"] = [float(ys.mean()/h), float(xs.mean()/w)]
            features[f"color_{c}_spread"] = [float(ys.std()/h), float(xs.std()/w)]

    return features


def extract_training_pairs(game_id, solution):
    """Extract (state, action) training pairs from a solved game."""
    pairs = []

    try:
        for mod in list(sys.modules.keys()):
            if "arc_game" in mod:
                del sys.modules[mod]

        cls = load_game_class(game_id)
        game = create_game(cls)
        meta = find_game_metadata(game_id)

        for level_data in solution.get("levels", []):
            if not level_data.get("solved"):
                continue
            if not level_data.get("actions"):
                continue

            level_idx = level_data["level"]
            actions = level_data["actions"]

            for step_idx, act_data in enumerate(actions):
                # Capture current frame
                frame = game.camera.render(game.current_level.get_sprites())
                features = encode_frame(frame)

                # Record the action taken
                action_id = act_data.get("id", 1)
                action_data = act_data.get("data", {})

                pair = {
                    "game_id": game_id,
                    "level": level_idx,
                    "step": step_idx,
                    "total_steps": len(actions),
                    "score": game._score,
                    "action_id": action_id,
                    "action_data": action_data,
                    "frame_features": features,
                    "frame_hash": hashlib.blake2b(
                        frame.tobytes(), digest_size=8
                    ).hexdigest(),
                    "available_actions": list(game._available_actions),
                }
                pairs.append(pair)

                # Execute the action
                action = ActionInput(
                    id=GameAction.from_id(action_id),
                    data=action_data if action_data else {},
                )
                step_game(game, action)

    except Exception as e:
        print(f"  Error extracting {game_id}: {e}")

    return pairs


def build_game_analysis():
    """Build per-game analysis data from source code examination."""
    analyses = {}

    for gid in sorted(os.listdir(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "environment_files"
    ))):
        try:
            meta = find_game_metadata(gid)
            analyses[gid] = {
                "game_id": meta.get("game_id", gid),
                "baseline_actions": meta.get("baseline_actions", []),
                "tags": meta.get("tags", []),
                "total_levels": len(meta.get("baseline_actions", [])),
            }
        except:
            pass

    return analyses


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Building Training Data for 3B Model")
    print("=" * 60)

    # Load all solutions
    all_pairs = []
    stats = {"games": {}, "total_pairs": 0, "total_levels": 0}

    for sol_file in sorted(glob.glob(os.path.join(SOLUTIONS_DIR, "*.json"))):
        gid = os.path.splitext(os.path.basename(sol_file))[0]
        if gid.startswith("_"):
            continue

        with open(sol_file) as f:
            solution = json.load(f)

        solved = solution.get("solved_levels", 0)
        if solved == 0:
            continue

        print(f"  {gid}: {solved} levels...", end="", flush=True)

        pairs = extract_training_pairs(gid, solution)
        all_pairs.extend(pairs)

        stats["games"][gid] = {
            "solved_levels": solved,
            "training_pairs": len(pairs),
        }
        stats["total_pairs"] += len(pairs)
        stats["total_levels"] += solved

        print(f" {len(pairs)} pairs")

    # Save training data as JSONL
    output_path = os.path.join(OUTPUT_DIR, "arc3_training_pairs.jsonl")
    with open(output_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    # Save game analysis
    analyses = build_game_analysis()
    with open(os.path.join(OUTPUT_DIR, "arc3_game_analysis.json"), "w") as f:
        json.dump(analyses, f, indent=2)

    # Save stats
    stats_path = os.path.join(OUTPUT_DIR, "arc3_training_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Total: {stats['total_pairs']} training pairs from "
          f"{stats['total_levels']} solved levels")
    print(f"  Saved to: {output_path}")
    print(f"  Game analysis: {os.path.join(OUTPUT_DIR, 'arc3_game_analysis.json')}")


if __name__ == "__main__":
    main()
