"""
GGUF Conversion Pipeline for Olympus Specialists

Three steps:
  1. Merge LoRA adapters into base SmolLM3-3B model
  2. Convert merged model to GGUF format
  3. Quantize to Q4_K_M for fast CPU inference (3B model -> ~1.8GB)

Usage:
  python olympus/convert_gguf.py                    # Convert all specialists
  python olympus/convert_gguf.py --specialist code   # Convert one specialist
  python olympus/convert_gguf.py --quantize-only     # Re-quantize existing merges
  python olympus/convert_gguf.py --check             # Verify all outputs exist

Prerequisites:
  pip install transformers peft torch
  pip install llama-cpp-python        # For inference
  git clone https://github.com/ggerganov/llama.cpp  # For conversion
  cd llama.cpp && make                               # Build quantize tool

ETA per specialist: ~3-5 min merge, ~2 min convert, ~1 min quantize
"""

import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
GGUF_DIR = CHECKPOINTS_DIR / "gguf"
LLAMA_CPP_DIR = PROJECT_ROOT / "llama.cpp"

BASE_MODEL_ID = "HuggingFaceTB/SmolLM3-3B"

SPECIALISTS = {
    "code": CHECKPOINTS_DIR / "olympus_code" / "final",
    "math": CHECKPOINTS_DIR / "olympus_math" / "final",
    "qa":   CHECKPOINTS_DIR / "olympus_qa"   / "final",
}

# Q4_K_M: best quality-per-bit for 3B models. ~1.8GB output.
# Q5_K_M: slightly better quality, ~2.1GB. Use if disk isn't tight.
DEFAULT_QUANT = "Q4_K_M"


def check_prerequisites():
    """Verify everything needed is installed."""
    errors = []

    # Check Python packages
    try:
        import torch
    except ImportError:
        errors.append("torch not installed: pip install torch")

    try:
        import transformers
    except ImportError:
        errors.append("transformers not installed: pip install transformers")

    try:
        import peft
    except ImportError:
        errors.append("peft not installed: pip install peft")

    # Check llama.cpp
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        errors.append(
            f"llama.cpp not found at {LLAMA_CPP_DIR}\n"
            f"  Fix: git clone https://github.com/ggerganov/llama.cpp {LLAMA_CPP_DIR}"
        )

    # Check quantize binary
    quantize_bin = find_quantize_binary()
    if quantize_bin is None:
        errors.append(
            f"llama-quantize not found. Build llama.cpp:\n"
            f"  cd {LLAMA_CPP_DIR} && cmake -B build && cmake --build build --config Release"
        )

    return errors


def find_quantize_binary():
    """Find the llama-quantize binary (different paths on different builds)."""
    candidates = [
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize",
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize.exe",
        LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-quantize.exe",
        LLAMA_CPP_DIR / "build" / "llama-quantize",
        LLAMA_CPP_DIR / "llama-quantize",
        LLAMA_CPP_DIR / "llama-quantize.exe",
        LLAMA_CPP_DIR / "quantize",
        LLAMA_CPP_DIR / "quantize.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Try PATH
    if shutil.which("llama-quantize"):
        return Path(shutil.which("llama-quantize"))
    return None


def merge_lora(specialist_name, adapter_dir, output_dir):
    """Merge LoRA adapter into base model, save as full HF model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"\n{'='*60}")
    print(f"MERGING: {specialist_name}")
    print(f"  Base:    {BASE_MODEL_ID}")
    print(f"  Adapter: {adapter_dir}")
    print(f"  Output:  {output_dir}")
    print(f"{'='*60}")

    if output_dir.exists() and (output_dir / "model.safetensors").exists():
        print(f"  Merged model already exists, skipping. Use --force to re-merge.")
        return True

    # Verify adapter exists
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.exists():
        print(f"  ERROR: Adapter not found at {adapter_file}")
        print(f"  Have you downloaded checkpoints from RunPod?")
        return False

    print(f"  Loading base model ({BASE_MODEL_ID})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    print(f"  Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    print(f"  Merging weights (this takes ~2 min)...")
    model = model.merge_and_unload()

    print(f"  Saving merged model to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    # Verify
    merged_size = sum(f.stat().st_size for f in output_dir.iterdir()) / (1024**3)
    print(f"  Merged model: {merged_size:.1f}GB")

    # Free memory
    del model, base_model
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True


def convert_to_gguf(specialist_name, merged_dir, gguf_path):
    """Convert HF model to GGUF format using llama.cpp."""
    print(f"\n{'='*60}")
    print(f"CONVERTING TO GGUF: {specialist_name}")
    print(f"  Input:  {merged_dir}")
    print(f"  Output: {gguf_path}")
    print(f"{'='*60}")

    if gguf_path.exists():
        print(f"  GGUF already exists, skipping. Use --force to reconvert.")
        return True

    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"  ERROR: {convert_script} not found")
        return False

    gguf_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(convert_script),
        str(merged_dir),
        "--outfile", str(gguf_path),
        "--outtype", "f16",
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: Conversion failed")
        print(f"  stdout: {result.stdout[-500:]}")
        print(f"  stderr: {result.stderr[-500:]}")
        return False

    size_gb = gguf_path.stat().st_size / (1024**3)
    print(f"  GGUF created: {size_gb:.1f}GB")
    return True


def quantize_gguf(specialist_name, f16_gguf_path, quant_gguf_path, quant_type):
    """Quantize f16 GGUF to smaller quantized GGUF."""
    print(f"\n{'='*60}")
    print(f"QUANTIZING: {specialist_name} -> {quant_type}")
    print(f"  Input:  {f16_gguf_path}")
    print(f"  Output: {quant_gguf_path}")
    print(f"{'='*60}")

    if quant_gguf_path.exists():
        print(f"  Quantized GGUF already exists, skipping. Use --force to requantize.")
        return True

    quantize_bin = find_quantize_binary()
    if quantize_bin is None:
        print(f"  ERROR: llama-quantize not found")
        return False

    quant_gguf_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [str(quantize_bin), str(f16_gguf_path), str(quant_gguf_path), quant_type]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: Quantization failed")
        print(f"  stderr: {result.stderr[-500:]}")
        return False

    size_mb = quant_gguf_path.stat().st_size / (1024**2)
    print(f"  Quantized GGUF: {size_mb:.0f}MB")
    return True


def convert_specialist(name, adapter_dir, quant_type=DEFAULT_QUANT, force=False):
    """Full pipeline for one specialist: merge -> convert -> quantize."""
    merged_dir = CHECKPOINTS_DIR / f"olympus_{name}" / "merged"
    f16_gguf = GGUF_DIR / f"olympus-{name}-f16.gguf"
    quant_gguf = GGUF_DIR / f"olympus-{name}-{quant_type.lower()}.gguf"

    if force:
        # Clean existing outputs
        for p in [f16_gguf, quant_gguf]:
            if p.exists():
                p.unlink()
        if merged_dir.exists():
            shutil.rmtree(merged_dir)

    # Step 1: Merge
    if not merge_lora(name, adapter_dir, merged_dir):
        return False

    # Step 2: Convert to f16 GGUF
    if not convert_to_gguf(name, merged_dir, f16_gguf):
        return False

    # Step 3: Quantize
    if not quantize_gguf(name, f16_gguf, quant_gguf, quant_type):
        return False

    # Step 4: Clean up f16 GGUF (large, only needed as intermediate)
    if quant_gguf.exists() and f16_gguf.exists():
        print(f"\n  Cleaning up f16 intermediate ({f16_gguf.stat().st_size / (1024**3):.1f}GB)...")
        f16_gguf.unlink()

    # Optionally clean merged HF model too (saves ~6GB per specialist)
    # Uncomment if disk is tight:
    # if merged_dir.exists():
    #     shutil.rmtree(merged_dir)
    #     print(f"  Cleaned merged model directory")

    print(f"\n  DONE: {quant_gguf} ({quant_gguf.stat().st_size / (1024**2):.0f}MB)")
    return True


def check_outputs(quant_type=DEFAULT_QUANT):
    """Show status of all conversion outputs."""
    print(f"\n{'='*60}")
    print(f"GGUF CONVERSION STATUS")
    print(f"{'='*60}")

    all_good = True
    for name, adapter_dir in SPECIALISTS.items():
        adapter_ok = (adapter_dir / "adapter_model.safetensors").exists()
        merged_ok = (CHECKPOINTS_DIR / f"olympus_{name}" / "merged" / "model.safetensors").exists()
        quant_path = GGUF_DIR / f"olympus-{name}-{quant_type.lower()}.gguf"
        quant_ok = quant_path.exists()

        status = "READY" if quant_ok else "MISSING"
        if not quant_ok:
            all_good = False

        size_str = f"({quant_path.stat().st_size / (1024**2):.0f}MB)" if quant_ok else ""

        print(f"\n  {name}:")
        print(f"    Adapter:   {'OK' if adapter_ok else 'MISSING'} ({adapter_dir})")
        print(f"    Merged:    {'OK' if merged_ok else 'MISSING'}")
        print(f"    GGUF:      {status} {size_str}")

    print(f"\n  {'All specialists converted!' if all_good else 'Some specialists need conversion.'}")
    return all_good


def main():
    parser = argparse.ArgumentParser(description="Convert Olympus specialists to GGUF")
    parser.add_argument("--specialist", choices=list(SPECIALISTS.keys()),
                        help="Convert only this specialist")
    parser.add_argument("--quant", default=DEFAULT_QUANT,
                        help=f"Quantization type (default: {DEFAULT_QUANT})")
    parser.add_argument("--force", action="store_true",
                        help="Force reconversion even if outputs exist")
    parser.add_argument("--check", action="store_true",
                        help="Only check status, don't convert")
    parser.add_argument("--skip-prereq", action="store_true",
                        help="Skip prerequisite checks")
    args = parser.parse_args()

    if args.check:
        check_outputs(args.quant)
        return

    # Check prerequisites
    if not args.skip_prereq:
        errors = check_prerequisites()
        if errors:
            print("PREREQUISITES MISSING:")
            for e in errors:
                print(f"  - {e}")
            print("\nFix the above, then re-run. Or use --skip-prereq to skip checks.")
            sys.exit(1)

    # Convert
    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    if args.specialist:
        specialists = {args.specialist: SPECIALISTS[args.specialist]}
    else:
        specialists = SPECIALISTS

    results = {}
    for name, adapter_dir in specialists.items():
        results[name] = convert_specialist(name, adapter_dir, args.quant, args.force)

    # Summary
    print(f"\n\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")
    for name, success in results.items():
        icon = "OK" if success else "FAILED"
        gguf_path = GGUF_DIR / f"olympus-{name}-{args.quant.lower()}.gguf"
        size = f"({gguf_path.stat().st_size / (1024**2):.0f}MB)" if gguf_path.exists() else ""
        print(f"  {icon}  {name} {size}")

    if all(results.values()):
        print(f"\nAll conversions complete! GGUF files in: {GGUF_DIR}")
        print(f"\nNext steps:")
        print(f"  1. pip install llama-cpp-python")
        print(f"  2. python olympus/app.py   (will auto-detect GGUF files)")
    else:
        print(f"\nSome conversions failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
