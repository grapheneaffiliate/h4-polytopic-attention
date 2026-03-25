"""
Train QA/retrieval specialist: SmolLM3-3B + LoRA on QA data.
Run on GPU pod: python olympus/train_qa_specialist.py
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


def format_squad(example):
    context = example.get("context", "")
    question = example.get("question", "")
    answers = example.get("answers", {})
    if isinstance(answers, dict):
        answer_text = answers.get("text", [""])[0] if answers.get("text") else ""
    elif isinstance(answers, list) and answers:
        answer_text = answers[0] if isinstance(answers[0], str) else answers[0].get("text", "")
    else:
        answer_text = ""
    if not answer_text:
        return {"text": ""}
    return {"text": f"### Context:\n{context[:800]}\n\n### Question:\n{question}\n\n### Answer:\n{answer_text}"}


def format_nq(example):
    question = example.get("question", "")
    answers = example.get("answer", [])
    if isinstance(answers, list) and answers:
        answer_text = answers[0]
    elif isinstance(answers, str):
        answer_text = answers
    else:
        return {"text": ""}
    return {"text": f"### Question:\n{question}\n\n### Answer:\n{answer_text}"}


def main():
    t_start = time.time()
    model_id = "HuggingFaceTB/SmolLM3-3B"

    print("=" * 60)
    print("  OLYMPUS QA SPECIALIST — LoRA Training")
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

    print("\nLoading QA datasets...")
    ds1 = load_dataset("rajpurkar/squad_v2", split="train")
    ds2 = load_dataset("google-research-datasets/nq_open", split="train")

    print(f"  SQuAD 2.0: {len(ds1)} examples")
    print(f"  NQ Open: {len(ds2)} examples")

    ds1 = ds1.map(format_squad, remove_columns=ds1.column_names)
    ds2 = ds2.map(format_nq, remove_columns=ds2.column_names)

    dataset = concatenate_datasets([ds1, ds2]).shuffle(seed=42)
    dataset = dataset.filter(lambda x: len(x["text"]) > 20)

    max_examples = 80000
    if len(dataset) > max_examples:
        dataset = dataset.select(range(max_examples))
    print(f"  Combined: {len(dataset)} examples")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024, padding=False)

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    split = dataset.train_test_split(test_size=0.02, seed=42)

    output_dir = "/runpod-volume/olympus_qa_specialist"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=2,
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
    print("  QA GENERATION TEST")
    print("=" * 60)

    model.eval()
    test_prompts = [
        "### Context:\nThe Eiffel Tower was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair. It was designed by Gustave Eiffel's engineering company.\n\n### Question:\nWhen was the Eiffel Tower built?\n\n### Answer:\n",
        "### Context:\nPython is a high-level programming language created by Guido van Rossum and first released in 1991. It emphasizes code readability.\n\n### Question:\nWho created Python?\n\n### Answer:\n",
        "### Question:\nWhat is the capital of France?\n\n### Answer:\n",
    ]

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.3, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### Answer:" in response:
            response = response.split("### Answer:")[-1].strip()
        q = prompt.split("Question:")[1].split("Answer:")[0].strip() if "Question:" in prompt else prompt[:60]
        print(f"\nQ: {q}")
        print(f"A: {response[:150]}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/3600:.1f} hours")
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({"specialist": "qa", "hours": elapsed/3600, "examples": len(split["train"])}, f)


if __name__ == "__main__":
    main()
