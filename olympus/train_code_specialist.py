"""
Train code specialist: SmolLM3-3B + QLoRA on code data.

Run on GPU pod:
    python olympus/train_code_specialist.py

Expects SmolLM3-3B and datasets already cached from download_all.py.
"""

import torch
import time
import os
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, concatenate_datasets


def format_code_example(example):
    """Format CodeAlpaca/CodeFeedback examples as instruction-response pairs."""
    # Different datasets have different field names
    instruction = example.get("instruction", example.get("query", example.get("prompt", "")))
    output = example.get("output", example.get("answer", example.get("response", "")))
    inp = example.get("input", "")

    if inp and inp.strip():
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    return {"text": text}


def main():
    t_start = time.time()
    model_id = "HuggingFaceTB/SmolLM3-3B"

    print("=" * 60)
    print("  OLYMPUS CODE SPECIALIST — QLoRA Training")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in fp16 (4-bit has conversion issues with SmolLM3)
    # fp16 uses 6.2GB VRAM, leaves ~10GB for LoRA training on 16GB GPU
    print("Loading SmolLM3-3B in fp16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Freeze base model, only train LoRA adapters
    for param in model.parameters():
        param.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Base model: {n_params/1e9:.1f}B params")

    # Apply LoRA
    print("Applying LoRA (r=16)...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare data
    print("\nLoading code datasets...")
    ds1 = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    ds2 = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")

    print(f"  CodeAlpaca: {len(ds1)} examples")
    print(f"  CodeFeedback: {len(ds2)} examples")

    # Format and combine
    ds1 = ds1.map(format_code_example, remove_columns=ds1.column_names)
    ds2 = ds2.map(format_code_example, remove_columns=ds2.column_names)
    dataset = concatenate_datasets([ds1, ds2]).shuffle(seed=42)

    # Take a subset for faster training (can increase later)
    max_examples = 50000
    if len(dataset) > max_examples:
        dataset = dataset.select(range(max_examples))
    print(f"  Combined: {len(dataset)} examples (capped at {max_examples})")

    # Tokenize
    print("Tokenizing...")
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            padding=False,
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Split
    split = dataset.train_test_split(test_size=0.02, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Training arguments
    output_dir = "/runpod-volume/olympus_code_specialist"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # effective batch = 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=2,
        max_grad_norm=1.0,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch: {training_args.per_device_train_batch_size} x {training_args.gradient_accumulation_steps} = {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  LR: {training_args.learning_rate}")
    print(f"  Output: {output_dir}")

    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))

    # Evaluate with generation
    print("\n" + "=" * 60)
    print("  CODE GENERATION TEST")
    print("=" * 60)

    model.eval()
    test_prompts = [
        "### Instruction:\nWrite a Python function that sorts a list using bubble sort.\n\n### Response:\n",
        "### Instruction:\nWrite a Python function to check if a number is prime.\n\n### Response:\n",
        "### Instruction:\nWrite a Python function that reverses a string.\n\n### Response:\n",
        "### Instruction:\nWrite a Python class for a binary search tree with insert and search methods.\n\n### Response:\n",
        "### Instruction:\nWrite a JavaScript function that debounces another function.\n\n### Response:\n",
    ]

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        print(f"\nQ: {prompt.split('Instruction:')[1].split('Response:')[0].strip()}")
        print(f"A: {response[:300]}")
        print("-" * 40)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/3600:.1f} hours")
    print(f"  Output: {output_dir}/final")

    # Save metrics
    metrics = {
        "specialist": "code",
        "base_model": model_id,
        "training_hours": elapsed / 3600,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "lora_rank": 16,
        "epochs": training_args.num_train_epochs,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
