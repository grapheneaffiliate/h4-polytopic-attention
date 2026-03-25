"""
Train math/reasoning specialist: SmolLM3-3B + LoRA on math data.
Run on GPU pod: python olympus/train_math_specialist.py
"""

import torch
import time
import os
import json
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, concatenate_datasets


def format_math_example(example):
    query = example.get("query", example.get("question", example.get("problem", "")))
    response = example.get("response", example.get("answer", example.get("solution", "")))
    if not query or not response:
        return {"text": ""}
    return {"text": f"### Problem:\n{query}\n\n### Solution:\n{response}"}


def main():
    t_start = time.time()
    model_id = "HuggingFaceTB/SmolLM3-3B"

    print("=" * 60)
    print("  OLYMPUS MATH SPECIALIST — LoRA Training")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading SmolLM3-3B in fp16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16)
    for param in model.parameters():
        param.requires_grad = False

    print("Applying LoRA (r=16)...")
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\nLoading math datasets...")
    ds1 = load_dataset("meta-math/MetaMathQA", split="train")
    ds2 = load_dataset("openai/gsm8k", "main", split="train")

    print(f"  MetaMathQA: {len(ds1)} examples")
    print(f"  GSM8K: {len(ds2)} examples")

    ds1 = ds1.map(format_math_example, remove_columns=ds1.column_names)
    ds2 = ds2.map(lambda x: {"text": f"### Problem:\n{x['question']}\n\n### Solution:\n{x['answer']}"},
                   remove_columns=ds2.column_names)

    dataset = concatenate_datasets([ds1, ds2]).shuffle(seed=42)
    dataset = dataset.filter(lambda x: len(x["text"]) > 20)

    max_examples = 50000
    if len(dataset) > max_examples:
        dataset = dataset.select(range(max_examples))
    print(f"  Combined: {len(dataset)} examples")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024, padding=False)

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    split = dataset.train_test_split(test_size=0.02, seed=42)

    output_dir = "/runpod-volume/olympus_math_specialist"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=3,
        per_device_train_batch_size=2, per_device_eval_batch_size=2,
        gradient_accumulation_steps=8, learning_rate=2e-4,
        lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.01,
        logging_steps=50, eval_strategy="steps", eval_steps=500,
        save_strategy="steps", save_steps=500, save_total_limit=3,
        fp16=True, gradient_checkpointing=True, report_to="none",
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=split["train"], eval_dataset=split["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print(f"\nTraining: {training_args.num_train_epochs} epochs, {len(split['train'])} examples")
    trainer.train()

    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))

    print("\n" + "=" * 60)
    print("  MATH GENERATION TEST")
    print("=" * 60)

    model.eval()
    test_prompts = [
        "### Problem:\nWhat is 15 * 23?\n\n### Solution:\n",
        "### Problem:\nSolve for x: 2x + 5 = 17\n\n### Solution:\n",
        "### Problem:\nA store sells apples for $2 each. If John buys 5 apples and pays with a $20 bill, how much change does he get?\n\n### Solution:\n",
    ]

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.3, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### Solution:" in response:
            response = response.split("### Solution:")[-1].strip()
        print(f"\nQ: {prompt.split('Problem:')[1].split('Solution:')[0].strip()}")
        print(f"A: {response[:200]}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/3600:.1f} hours")
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({"specialist": "math", "hours": elapsed/3600, "examples": len(split["train"])}, f)


if __name__ == "__main__":
    main()
