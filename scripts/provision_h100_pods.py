#!/usr/bin/env python3
"""
RunPod Fine-Tuning Setup for ARC-AGI-3
=======================================
Prepares data and generates training script for 3B model fine-tuning.

Training data: data/arc3_training_pairs.jsonl
Model: Qwen-2.5-3B-Instruct with LoRA

Usage:
  python scripts/provision_h100_pods.py   # Prep data + generate script
  # Then on RunPod: python scripts/runpod_train.py
"""

import json
import os
import sys


def build_finetune_dataset(data_path):
    """Convert training pairs to instruction format for fine-tuning."""
    samples = []
    with open(data_path) as f:
        for line in f:
            pair = json.loads(line)
            features = pair["frame_features"]
            hist = features.get("color_hist", [])

            input_text = (
                f"Game: {pair['game_id']} Level: {pair['level']} "
                f"Step: {pair['step']}/{pair['total_steps']}\n"
                f"Actions: {pair['available_actions']}\n"
                f"Colors: {[round(h,3) for h in hist]}\n"
            )
            for qi in range(2):
                for qj in range(2):
                    qh = features.get(f"quad_{qi}{qj}", [])
                    if qh:
                        top = sorted(range(len(qh)), key=lambda i: -qh[i])[:3]
                        input_text += f"Q{qi}{qj}: {[round(qh[t],3) for t in top]}\n"

            action_id = pair["action_id"]
            action_data = pair.get("action_data", {})
            output_text = f"ACTION{action_id}"
            if action_data:
                output_text += f" x={action_data.get('x',0)} y={action_data.get('y',0)}"

            samples.append({"input": input_text, "output": output_text,
                          "game_id": pair["game_id"], "level": pair["level"]})
    return samples


def main():
    data_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "arc3_training_pairs.jsonl")

    if not os.path.exists(data_path):
        print("Run build_training_data.py first")
        return

    samples = build_finetune_dataset(data_path)
    print(f"{len(samples)} training samples")

    output_path = data_path.replace("training_pairs", "finetune")
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"Saved to {output_path}")

    games = {}
    for s in samples:
        games[s["game_id"]] = games.get(s["game_id"], 0) + 1
    for gid in sorted(games):
        print(f"  {gid}: {games[gid]} samples")


if __name__ == "__main__":
    main()
