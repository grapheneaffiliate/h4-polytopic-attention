"""
Train a specialist model with QLoRA + optional H4 attention swap.

Supports both CPU and GPU training. GPU (H100) is ~50-100x faster.

Usage:
    # CPU training (slow but free)
    python olympus/train_specialist.py --specialist code --device cpu

    # GPU training (fast, ~$2-3/hr on RunPod H100)
    python olympus/train_specialist.py --specialist code --device cuda

    # All specialists sequentially
    python olympus/train_specialist.py --specialist all --device cuda

Each specialist goes through:
1. Load SmolLM3-3B-Instruct base model
2. Apply QLoRA (only 1-2% of params trainable)
3. Fine-tune on specialist-specific data
4. Optionally: progressive H4 attention swap
5. Save checkpoint
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))


# Specialist configurations
SPECIALIST_CONFIGS = {
    'general': {
        'base_model': 'HuggingFaceTB/SmolLM3-3B-Instruct',
        'datasets': [],  # Use as-is, no fine-tuning needed
        'max_tokens': 0,
        'description': 'General conversation, instructions, creative (no FT needed)',
    },
    'code': {
        'base_model': 'HuggingFaceTB/SmolLM3-3B-Instruct',
        'datasets': [
            ('sahil2801/CodeAlpaca-20k', 'train'),
            ('m-a-p/CodeFeedback-Filtered-Instruction', 'train'),
        ],
        'max_tokens': 200_000_000,
        'lr': 2e-5,
        'epochs': 3,
        'max_seq_len': 2048,
        'description': 'Code generation, debugging, explanation',
    },
    'math': {
        'base_model': 'HuggingFaceTB/SmolLM3-3B-Instruct',
        'datasets': [
            ('meta-math/MetaMathQA', 'train'),
            ('openai/gsm8k', 'train'),
        ],
        'max_tokens': 100_000_000,
        'lr': 5e-5,
        'epochs': 5,
        'max_seq_len': 1024,
        'description': 'Math problem solving, logical reasoning',
    },
    'qa': {
        'base_model': 'HuggingFaceTB/SmolLM3-3B-Instruct',
        'datasets': [
            ('rajpurkar/squad_v2', 'train'),
            ('google-research-datasets/nq_open', 'train'),
        ],
        'max_tokens': 150_000_000,
        'lr': 5e-5,
        'epochs': 3,
        'max_seq_len': 1024,
        'description': 'Factual QA from retrieved context',
    },
}


def train_specialist(specialist_name: str, device: str = 'cpu', output_dir: str = 'checkpoints'):
    """
    Train a single specialist.

    This is the scaffold — full implementation requires:
    - pip install transformers datasets peft bitsandbytes accelerate
    - For GPU: CUDA-compatible PyTorch
    - For CPU: patience (3-6x slower than GPU, but free)
    """
    config = SPECIALIST_CONFIGS[specialist_name]

    print(f"\n{'='*60}")
    print(f"  Training specialist: {specialist_name}")
    print(f"  {config['description']}")
    print(f"  Base model: {config['base_model']}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    if not config['datasets']:
        print("  This specialist uses the base model as-is (no fine-tuning).")
        print("  Just download and convert to ternary.")
        return

    # Check dependencies
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  Install: pip install torch transformers datasets peft")
        return

    print(f"  Max tokens: {config['max_tokens']/1e6:.0f}M")
    print(f"  Datasets: {[d[0] for d in config['datasets']]}")
    print(f"  Learning rate: {config.get('lr', 'N/A')}")
    print(f"  Max seq len: {config.get('max_seq_len', 'N/A')}")

    # Step 1: Load base model
    print("\n  Step 1: Loading base model...")
    print(f"  This downloads SmolLM3-3B (~6GB) on first run.")
    print(f"  Subsequent runs use cached model.")

    # Step 2: Apply QLoRA
    print("\n  Step 2: QLoRA setup...")
    print(f"  LoRA rank: 16, target modules: q/k/v/o/gate/up/down")
    print(f"  Trainable params: ~20-50M out of 3B (1-2%)")

    # Step 3: Load datasets
    print("\n  Step 3: Loading datasets...")
    for ds_name, split in config['datasets']:
        print(f"    {ds_name} ({split})")

    # Step 4: Train
    print("\n  Step 4: Training...")
    if device == 'cuda':
        print(f"  Estimated time: 1-2 hours on H100")
        print(f"  Estimated cost: ~$3-5 on RunPod")
    else:
        print(f"  Estimated time: 1-2 days on CPU")
        print(f"  Estimated cost: $0 (electricity only)")

    print(f"\n  [Training not started — this is the scaffold.]")
    print(f"  [Run with --execute to actually train.]")
    print(f"  [Requires: transformers, datasets, peft, bitsandbytes]")

    # Step 5: Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'olympus_{specialist_name}.pt')
    print(f"\n  Output: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Olympus specialist')
    parser.add_argument('--specialist', required=True,
                        choices=list(SPECIALIST_CONFIGS.keys()) + ['all'],
                        help='Which specialist to train')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                        help='Training device (cpu or cuda)')
    parser.add_argument('--output', default='checkpoints',
                        help='Output directory for checkpoints')
    args = parser.parse_args()

    if args.specialist == 'all':
        for name in SPECIALIST_CONFIGS:
            train_specialist(name, args.device, args.output)
    else:
        train_specialist(args.specialist, args.device, args.output)


if __name__ == '__main__':
    main()
