#!/usr/bin/env python3
"""
Fine-tune SmolLM3-3B on solved ARC tasks (task description → C code).

Usage:
    python finetune_arc.py \
        --train-data /runpod-volume/arc_finetune.jsonl \
        --output-dir /runpod-volume/arc_lora_r1 \
        --epochs 5 --lr 2e-4 --batch-size 2

The model learns to generate C code bodies that solve ARC puzzles.
Each round of the self-compiling loop adds more training data.
"""

import argparse
import json
import os
import torch
from pathlib import Path


def train(train_data, output_dir, base_model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
          epochs=5, lr=2e-4, batch_size=2, max_length=2048, lora_r=16, lora_alpha=32):
    """Fine-tune with LoRA on the ARC training data."""
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load training data
    print(f"Loading training data from {train_data}")
    examples = []
    with open(train_data) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"  {len(examples)} examples")

    # Format as text
    def format_example(ex):
        msgs = ex["messages"]
        text = ""
        for msg in msgs:
            role = msg["role"]
            content = msg["content"]
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return {"text": text}

    dataset = Dataset.from_list([format_example(ex) for ex in examples])
    print(f"  Dataset: {len(dataset)} rows")

    # Training
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 4 // batch_size),
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
    )

    # Use SFTTrainer with basic config
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"Starting training: {epochs} epochs, lr={lr}, batch={batch_size}")
    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    args = parser.parse_args()

    train(
        train_data=args.train_data,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        lora_r=args.lora_r,
    )
