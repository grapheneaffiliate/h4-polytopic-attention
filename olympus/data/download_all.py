"""
Download all Olympus training data from HuggingFace.

Run this ON the GPU pod — it has fast internet (1-10 Gbps).
Do not download locally and upload — that's slow and wastes bandwidth.

Usage:
    python olympus/data/download_all.py
    python olympus/data/download_all.py --specialist code
    python olympus/data/download_all.py --specialist math
    python olympus/data/download_all.py --specialist qa
    python olympus/data/download_all.py --specialist all
    python olympus/data/download_all.py --base-model  # just SmolLM3
"""

import argparse
import os
import time


def download_base_model():
    """Download SmolLM3-3B-Instruct (~6GB)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "HuggingFaceTB/SmolLM3-3B-Instruct"
    print(f"\nDownloading {model_id}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"  Tokenizer downloaded ({time.time()-t0:.0f}s)")

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    print(f"  Model downloaded ({time.time()-t0:.0f}s)")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")

    # Quick verify
    import torch
    inputs = tokenizer("Hello, I am", return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)
    print(f"  Verify: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

    del model  # free memory
    return tokenizer


def download_code_data():
    """Download code specialist training data."""
    from datasets import load_dataset

    print("\n--- Code Specialist Data ---")

    print("  CodeAlpaca-20k...")
    ds = load_dataset("sahil2801/CodeAlpaca-20k")
    print(f"    {len(ds['train'])} examples")

    print("  CodeFeedback...")
    ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction")
    print(f"    {len(ds['train'])} examples")

    print("  Evol-Instruct-Code (subset)...")
    ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1")
    print(f"    {len(ds['train'])} examples")


def download_math_data():
    """Download math specialist training data."""
    from datasets import load_dataset

    print("\n--- Math Specialist Data ---")

    print("  MetaMathQA...")
    ds = load_dataset("meta-math/MetaMathQA")
    print(f"    {len(ds['train'])} examples")

    print("  GSM8K...")
    ds = load_dataset("openai/gsm8k", "main")
    print(f"    {len(ds['train'])} train, {len(ds['test'])} test")

    print("  ARC...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    print(f"    {len(ds['train'])} train, {len(ds['test'])} test")


def download_qa_data():
    """Download QA specialist training data."""
    from datasets import load_dataset

    print("\n--- QA Specialist Data ---")

    print("  SQuAD 2.0...")
    ds = load_dataset("rajpurkar/squad_v2")
    print(f"    {len(ds['train'])} train, {len(ds['validation'])} val")

    print("  Natural Questions (open)...")
    ds = load_dataset("google-research-datasets/nq_open")
    print(f"    {len(ds['train'])} train, {len(ds['validation'])} val")

    print("  TriviaQA...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
    print(f"    {len(ds['train'])} train")


def main():
    parser = argparse.ArgumentParser(description='Download Olympus training data')
    parser.add_argument('--specialist', default='all',
                        choices=['all', 'code', 'math', 'qa'])
    parser.add_argument('--base-model', action='store_true',
                        help='Download SmolLM3-3B only')
    args = parser.parse_args()

    # Always need these
    print("Installing dependencies...")
    os.system("pip install -q transformers datasets accelerate peft bitsandbytes sentencepiece protobuf")

    t_start = time.time()

    if args.base_model or args.specialist == 'all':
        download_base_model()

    if args.specialist in ('all', 'code'):
        download_code_data()

    if args.specialist in ('all', 'math'):
        download_math_data()

    if args.specialist in ('all', 'qa'):
        download_qa_data()

    print(f"\nAll downloads complete in {(time.time()-t_start)/60:.1f} minutes")


if __name__ == '__main__':
    main()
