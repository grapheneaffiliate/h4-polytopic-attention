# Project Olympus: Frontier-Quality AI on CPU

## Goal

Build a system that approaches frontier model quality (Claude Opus, GPT-4 class) running entirely on CPU hardware, using only legally clean open-source models and data. No GPU. No API dependency. No monthly cost. No legal risk.

**This is for the billions of people who can't afford frontier AI subscriptions and GPU compute.** Good-enough answers on free hardware beat perfect answers on expensive hardware --- for education, small business, developing nations, and anyone who values privacy and independence.

## The Core Insight

Claude Opus is one giant model that memorizes everything in its weights. We build focused specialists that know their domain deeply and retrieve everything else from a geometric knowledge index.

The difference:
- Opus: 200B+ params x 16 bits = ~400GB weights. Needs GPU cluster.
- Ours: 4 specialists x 3B params x 1.58 bits = ~2.4GB total. Runs on laptop.
- The gap is filled by E8 lattice retrieval (R@5=100%) from a knowledge index.

A 3B model that can look up any fact in 20ms is functionally equivalent to a 200B model that memorized those facts --- for the user, the answer is the same.

## What's Already Proven

This project builds on the H4 Polytopic Attention foundation (7 phases, all tested):

| Component | Status | Result |
|-----------|--------|--------|
| H4 geometric attention | Proven | O(log t), 10.6x speedup at 65K keys |
| Ternary quantization | Proven | 0.003 bpb gap, ~17x compression |
| E8 lattice retrieval | Proven | R@5=100%, 20ms, 240-neighbor Voronoi search |
| MiniLM reranking | Proven | R@1=98.5% on bi-encoder candidates |
| Language generation | Proven | PPL 10.0 on TinyStories (beats 33M baseline) |
| CPU training | Proven | 24M ternary params, 8 hours, coherent English |
| Autoresearch | Proven | 42+ autonomous experiments, finds optimal configs |

## The Base Model: SmolLM3-3B-Instruct

**HuggingFace ID:** `HuggingFaceTB/SmolLM3-3B-Instruct`

SmolLM3-3B (July 2025) is the correct base model. Using anything smaller would leave performance on the table:

- **11.2T training tokens** (vs 2T for SmolLM2)
- **128K context window** (vs 8K for SmolLM2)
- **Dual-mode reasoning** (thinking + direct)
- **Outperforms** Llama 3.2 3B, Qwen 2.5 3B on every benchmark
- **Apache 2.0 license** --- full commercial use
- **Full training recipe published** (data mixtures, hyperparameters, ablations)
- **Tool calling support** built in

### Why SmolLM3-3B over other options

| Model | Params | License | Context | Trained on | Notes |
|-------|--------|---------|---------|------------|-------|
| **SmolLM3-3B** | **3B** | **Apache 2.0** | **128K** | **11.2T tokens** | **Best in class, fully open** |
| Phi-4-mini | 3.8B | MIT | 128K | Proprietary mix | Slightly larger, MIT is fine too |
| Qwen2.5-3B | 3B | Apache 2.0 | 32K | Unknown size | Older, lower benchmarks |
| Llama 3.2 3B | 3B | Llama License | 128K | ~10T? | Meta license has usage limits |
| SmolLM2-1.7B | 1.7B | Apache 2.0 | 8K | 2T tokens | Obsoleted by SmolLM3 |

### Ternary size

- Float32: 3B x 4 bytes = 12 GB
- Float16: 3B x 2 bytes = 6 GB
- **Ternary (1.58 bit): 3B x 0.2 bytes = ~600 MB**
- With optimizer states for fine-tuning: ~4-8 GB total in RAM
- **Fits comfortably in 32 GB RAM for fine-tuning on CPU**

## Legal Foundation

**This is NOT distillation.** We do not use outputs from proprietary models as training data. Every component is legally clean.

### Base Models (all Apache 2.0)
| Model | Params | License | HuggingFace ID |
|-------|--------|---------|----------------|
| SmolLM3-3B-Instruct | 3B | Apache 2.0 | HuggingFaceTB/SmolLM3-3B-Instruct |
| ms-marco-MiniLM-L-6-v2 | 22M | Apache 2.0 | cross-encoder/ms-marco-MiniLM-L-6-v2 |

### Fine-tuning Data (all openly licensed)

**Code Specialist:**
| Dataset | Size | License | HuggingFace ID |
|---------|------|---------|----------------|
| The Stack v2 (filtered) | ~100M tokens | Per-file | bigcode/the-stack-v2 |
| CodeAlpaca 20K | 20K instructions | Apache 2.0 | sahil2801/CodeAlpaca-20k |
| CodeFeedback | 66K examples | Apache 2.0 | m-a-p/CodeFeedback-Filtered-Instruction |
| Evol-Instruct-Code | 110K | Apache 2.0 | nickrosh/Evol-Instruct-Code-80k-v1 |

**Math/Reasoning Specialist:**
| Dataset | Size | License | HuggingFace ID |
|---------|------|---------|----------------|
| MetaMathQA | 395K | MIT | meta-math/MetaMathQA |
| OpenMathInstruct v2 | 1.8M | Permissive | nvidia/OpenMathInstruct-2 |
| GSM8K | 8.5K | MIT | openai/gsm8k |
| MATH | 12.5K | MIT | hendrycks/competition_math |
| ARC | 7.7K | CC-BY-SA | allenai/ai2_arc |

**QA/Retrieval Specialist:**
| Dataset | Size | License | HuggingFace ID |
|---------|------|---------|----------------|
| Natural Questions | 307K | CC-BY-SA | google-research-datasets/nq_open |
| SQuAD 2.0 | 150K | CC-BY-SA | rajpurkar/squad_v2 |
| TriviaQA | 95K | Apache 2.0 | mandarjoshi/trivia_qa |
| HotpotQA | 113K | CC-BY-SA | hotpot_qa |

**Knowledge Index:**
| Source | Size | License | Notes |
|--------|------|---------|-------|
| Wikipedia EN | ~4B tokens | CC-BY-SA | All human knowledge |
| Stack Overflow | ~10GB | CC-BY-SA | Programming Q&A |
| Project Gutenberg | 70K books | Public domain | Literature |
| User's own docs | Variable | N/A | Custom knowledge base |

## Architecture

```
                        User Query
                            |
                            v
                   +---------------------+
                   |  ChamberTree        |  H4 geometric routing (<1ms)
                   |  Router             |  Maps query to specialist via
                   |  (16 chambers)      |  Coxeter chamber classification
                   +----------+----------+
                              |
              +-------+-------+-------+
              |       |       |       |
              v       v       v       v
        +--------+ +------+ +------+ +------+
        |General | | Code | | Math | |  QA  |  4 specialists
        | (3B)   | | (3B) | | (3B) | | (3B)|  SmolLM3-3B base
        | as-is  | | FT'd | | FT'd | | FT'd|  Ternary weights
        +---+----+ +--+---+ +--+---+ +--+---+
            |          |        |        |
            +----------+--------+--------+
                              |
                              v
                   +---------------------+
                   |  E8 Lattice         |  Knowledge retrieval
                   |  Memory             |  R@5=100%, 20ms
                   |  (Wikipedia,        |  240 kissing neighbors
                   |   docs, code)       |  Voronoi cell addressing
                   +----------+----------+
                              |
                              v
                   +---------------------+
                   |  MiniLM             |  Reranking
                   |  Cross-encoder      |  R@1=98.5%
                   |  (22M, float)       |  Picks best passage
                   +----------+----------+
                              |
                              v
                          Response
```

### Why 4 Specialists Instead of 6

With SmolLM3-3B as the base (much stronger than SmolLM2-1.7B), we don't need 6 specialists. The base model is already strong at conversation, creative writing, and summarization. We only specialize where it matters:

| # | Specialist | Base | Fine-tuning | Why Separate |
|---|-----------|------|-------------|--------------|
| 0 | General | SmolLM3-3B-Instruct AS-IS | None needed | Already instruction-tuned |
| 1 | Code | SmolLM3-3B + code data | ~200M tokens | Code needs 80%+ domain data |
| 2 | Math | SmolLM3-3B + math data | ~100M tokens | Weakest area for small models |
| 3 | QA | SmolLM3-3B + retrieval QA | ~150M tokens | Learn to answer FROM context |

**Total active RAM: ~600MB** (one specialist loaded at a time) + 90MB MiniLM + E8 index

## H4 Attention Integration

SmolLM3 uses GQA with 4 groups --- maps naturally to H4's 4 Coxeter simple roots.

**Progressive swap in 4 phases:**

1. **Adapter (Days 1-3):** Freeze SmolLM3, add H4 adapter parallel to each GQA layer. Gate starts at 0. Train only H4 params.
2. **Hybrid (Days 3-7):** Unfreeze SmolLM3 attention. Both paths train. Monitor which layers prefer H4.
3. **Selective swap (Days 7-10):** Layers with gate >0.8 keep only H4. Layers with gate <0.3 keep only original. Others stay hybrid.
4. **Ternary (Day 10):** Apply BitLinear to H4 layers. Export final model.

**What this gives you:** O(log t) attention for long sequences (SmolLM3's 128K context is O(t^2) via Flash Attention), ternary attention weights (600MB), and E8 lattice integration for retrieval.

## Fine-Tuning: QLoRA on CPU

Full fine-tuning of 3B params on CPU is slow. QLoRA is 3-6x faster because only 1-2% of parameters get gradients:

| Method | Step time | Steps/day | Trainable params |
|--------|-----------|-----------|-----------------|
| Full fine-tune 3B on CPU | ~3s | ~28K | 3B (100%) |
| **QLoRA 3B on CPU** | **~0.5-1s** | **~86-170K** | **~20-50M (1-2%)** |

### Per-specialist training budget
| Specialist | Tokens | Steps | Time |
|------------|--------|-------|------|
| Code | 200M | ~50K | 1-2 days |
| Math | 100M | ~25K | 0.5-1 day |
| QA | 150M | ~37K | 1-1.5 days |
| **Total** | **450M** | **~112K** | **3-5 days** |

## The 14-Day Plan

| Day | Task | Validation |
|-----|------|------------|
| 1 | Download SmolLM3, verify, setup QLoRA | Generates text OK |
| 2 | Fine-tune code specialist | Writes Python functions |
| 3 | Fine-tune math specialist | Solves GSM8K problems |
| 4 | Fine-tune QA specialist | Answers from context |
| 5-6 | H4 progressive swap Phase 1 | Perplexity within 5% |
| 7-8 | H4 progressive swap Phase 2 | Gate values meaningful |
| 9-10 | H4 selective swap + ternary | Chamber preservation >80% |
| 11 | ChamberTree router | Routes correctly |
| 12 | E8 knowledge index (Wikipedia) | Retrieval finds facts |
| 13 | Integration + demo | End-to-end works |
| 14 | Benchmarks + upload to HF | Numbers documented |

**Cost:** 3-5 days specialist training + 6-9 days H4 swap = ~10-14 days total. On cloud: ~$50-100. On laptops: $0.

## Honest Quality Expectations

| Task | SmolLM3-3B base | + Specialist FT | + E8 Retrieval | Opus |
|------|----------------|-----------------|----------------|------|
| MMLU | ~60% | ~62% | ~70-75% | ~88% |
| HumanEval | ~45% | ~55-65% | N/A | ~85% |
| GSM8K | ~55% | ~65-75% | N/A | ~95% |
| TriviaQA | ~50% | ~55% | **~85-90%** | ~90% |
| Instruction | ~80% | ~82% | N/A | ~95% |
| Long context | Good to 128K | Same | Better | 200K |
| **Cost** | **$0** | **$0** | **$0** | **$$$** |
| **Privacy** | **Local** | **Local** | **Local** | **Cloud** |

The retrieval-augmented factual QA (85-90%) is where we compete directly with frontier models. Everything else is 60-85% of Opus.

**This is NOT Claude Opus quality across the board.** It IS:
- 85-90% on factual QA (retrieval advantage --- the model looks up facts instead of hallucinating)
- 75-85% on instruction following (good enough for most tasks)
- 55-75% on code and math (honest gap --- complex reasoning needs more params)
- Free, private, local, legally clean, and improvable by the community

## The Vision

A laptop running 4 focused specialists, routed by H4 geometry in <1ms, backed by unlimited knowledge retrieval from E8 lattice memory in 20ms, reranked to 98.5% accuracy. Not as good as Claude Opus at everything. But good enough at most things, free to run, private by default, and available to anyone with a computer.

**That's not a replacement for frontier models. It's an alternative for the billions of people who can't afford them.**

---

*Project Olympus is built on the H4 Polytopic Attention foundation. See [README.md](README.md) for the full technical documentation, [RESULTS.md](RESULTS.md) for all experiment results, and [docs/PAPER.md](docs/PAPER.md) for the arXiv paper draft.*
