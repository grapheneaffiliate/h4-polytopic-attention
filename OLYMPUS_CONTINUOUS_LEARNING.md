# Olympus Continuous Learning: Self-Improving Specialist System

## The Core Idea

The system identifies its own weaknesses, generates its own training data, trains its own specialists, and integrates them — with zero human intervention. The autoresearch pattern that discovered optimal configs in 30 experiments now discovers and fills capability gaps autonomously.

Individual specialists are frozen after training. But the system as a whole evolves continuously — new specialists appear, the router adapts, the knowledge index grows. Like a brain where individual neurons stabilize but circuits reshape constantly.

## What Already Exists

Every component of this loop is already built and proven:

| Component | Status | What it does |
|-----------|--------|-------------|
| Autoresearch loop | Proven (42+ experiments) | Autonomous try → measure → keep/discard |
| QLoRA training | Proven (3 specialists training now) | Fine-tune SmolLM3-3B on any domain |
| Router | Proven (100% on test set) | Classify queries to specialists |
| E8 knowledge index | Proven (R@5=100%) | Store and retrieve any knowledge |
| Confidence scoring | Proven (MRR, R@1, perplexity) | Measure response quality |
| ChamberTree geometry | Proven (16 chambers, <1ms) | Geometric sub-routing |

## The Continuous Learning Loop

```
FOREVER:
    1. SERVE — Answer queries, track confidence on every response
    2. DETECT — Identify weak domains (low confidence, user corrections)
    3. CURATE — Generate training data for weak domains
    4. TRAIN — Fine-tune new specialist (QLoRA, automated)
    5. VALIDATE — Does the specialist outperform the general model?
    6. DEPLOY — If yes: add to router. If no: discard, try again.
    7. ADAPT — ChamberTree reorganizes for new specialist
    8. GOTO 1
```

### Step 1: SERVE — Confidence Tracking

Every response includes a confidence score computed from:

```python
def compute_confidence(query, response, retrieval_results):
    signals = {
        # Model confidence: how sure is the LM about its tokens?
        'generation_entropy': mean_token_entropy(response),

        # Retrieval confidence: did we find good context?
        'retrieval_score': retrieval_results.top_score,
        'retrieval_gap': retrieval_results.score[0] - retrieval_results.score[1],

        # Router confidence: was the specialist choice clear?
        'router_confidence': router_result.confidence,

        # Length signal: very short responses often mean uncertainty
        'response_length': len(response.tokens),
    }

    # Weighted combination
    confidence = (
        0.3 * (1 - signals['generation_entropy']) +
        0.3 * signals['retrieval_score'] +
        0.2 * signals['router_confidence'] +
        0.2 * min(signals['response_length'] / 50, 1.0)
    )

    return confidence, signals
```

Low confidence responses get logged with the query, domain, and failure signals.

### Step 2: DETECT — Gap Identification

```python
def detect_gaps(confidence_log, threshold=0.5, min_failures=20):
    """
    Identify domains where the system consistently underperforms.

    A 'domain' is identified by:
    - Keyword clustering of low-confidence queries
    - Router chamber distribution of failures
    - User correction patterns (if available)
    """
    # Cluster low-confidence queries by topic
    weak_queries = [q for q, conf in confidence_log if conf < threshold]

    # Simple keyword extraction for domain identification
    domain_counts = Counter()
    for query in weak_queries:
        keywords = extract_keywords(query)  # TF-IDF or simple frequency
        for kw in keywords:
            domain_counts[kw] += 1

    # Domains with enough failures to justify a specialist
    gaps = [
        domain for domain, count in domain_counts.most_common(10)
        if count >= min_failures
    ]

    return gaps  # e.g., ['chemistry', 'legal', 'spanish']
```

### Step 3: CURATE — Automated Data Collection

```python
def curate_training_data(domain, target_examples=10000):
    """
    Automatically gather training data for a new specialist.

    Sources (in order of preference):
    1. Existing QA datasets on HuggingFace for this domain
    2. Wikipedia articles on this topic (already in E8 index)
    3. Filtered web text from open datasets (FineWeb-Edu, etc.)
    """
    data = []

    # Check HuggingFace for domain-specific datasets
    hf_datasets = search_huggingface(f"{domain} QA instruction")
    for ds_name in hf_datasets[:3]:
        ds = load_dataset(ds_name)
        data.extend(format_as_instruction_pairs(ds))

    # Pull relevant passages from E8 knowledge index
    domain_passages = knowledge_index.query(domain, k=1000)
    data.extend(generate_qa_from_passages(domain_passages))

    # Filter for quality and dedup
    data = deduplicate(data)
    data = filter_quality(data, min_length=50)

    return data[:target_examples]
```

### Step 4: TRAIN — Automated QLoRA

```python
def train_specialist(domain, training_data):
    """
    Same recipe as the 3 specialists training now.
    QLoRA on SmolLM3-3B, automated, no human intervention.
    """
    # Identical to olympus/train_specialist.py
    config = {
        'base_model': 'HuggingFaceTB/SmolLM3-3B',
        'lora_r': 16,
        'lr': 2e-4,
        'epochs': 2,
        'max_seq_len': 1024,
    }

    # Train (GPU: ~2 hours, CPU: ~2 days)
    checkpoint = run_qlora_training(config, training_data)

    return checkpoint
```

### Step 5: VALIDATE — Does It Actually Help?

```python
def validate_specialist(new_specialist, domain, test_queries):
    """
    Compare new specialist vs general model on domain-specific queries.

    The specialist must BEAT the general model to be deployed.
    This prevents regression — bad training data doesn't ship.
    """
    general_scores = []
    specialist_scores = []

    for query in test_queries:
        # Score both responses
        general_response = general_model.generate(query)
        specialist_response = new_specialist.generate(query)

        # Compare on multiple metrics
        general_scores.append(score_response(query, general_response))
        specialist_scores.append(score_response(query, specialist_response))

    improvement = mean(specialist_scores) - mean(general_scores)

    if improvement > 0.05:  # 5% threshold
        return 'deploy', improvement
    else:
        return 'discard', improvement
```

### Step 6: DEPLOY — Hot-Add to Router

```python
def deploy_specialist(domain, checkpoint):
    """
    Add new specialist to the running system.

    1. Add domain keywords to router
    2. Assign ChamberTree chambers
    3. Load specialist (or keep on disk for lazy loading)
    """
    # Update router keywords
    router.add_domain(domain, keywords=extract_domain_keywords(domain))

    # Assign chambers (take from general's allocation or split)
    router.assign_chambers(domain, chambers=[next_available_chamber()])

    # Register checkpoint path
    specialist_registry[domain] = checkpoint

    print(f"Deployed {domain} specialist: {checkpoint}")
```

### Step 7: ADAPT — ChamberTree Reorganization

As new specialists are added, the 16-chamber space gets redistributed. The ChamberTree geometry naturally supports this — each specialist gets the chambers whose geometric encoding best matches its domain queries.

Over time, the chamber assignments are learned from real routing data rather than hard-coded. A tiny classifier trained on (query_chamber, correct_specialist) pairs replaces the static mapping.

## Example: System Learns Chemistry

**Week 1:** User asks chemistry questions. System routes to general specialist. Answers are mediocre. Confidence scores average 0.35 on chemistry queries.

**Week 2:** Gap detection triggers: "chemistry" has 50+ low-confidence queries. System searches HuggingFace, finds chemistry QA datasets. Curates 8,000 instruction-response pairs.

**Week 3:** System runs QLoRA training on SmolLM3-3B with chemistry data. Takes 3 hours on GPU or 2 days on CPU. Validation shows 15% improvement over general model on chemistry questions.

**Week 3 (deploy):** Chemistry specialist added to router. Keywords: "molecule", "element", "reaction", "compound", "pH", "electron", "bond", etc. Assigned ChamberTree chambers 10-11 (split from creative's allocation).

**Week 4+:** Chemistry questions route to specialist. Confidence scores average 0.75. System works on the next gap (maybe legal, maybe medical, maybe the user's specific codebase).

## The Scaling Path

| Specialists | RAM (ternary) | Disk | Coverage |
|-------------|--------------|------|----------|
| 4 (initial) | 600MB active | 2.4GB | General + code + math + QA |
| 8 | 600MB active | 4.8GB | + chemistry + legal + medical + language |
| 16 | 600MB active | 9.6GB | + domain-specific (user's docs, codebase) |
| 32 | 600MB active | 19.2GB | Comprehensive coverage |

Active RAM never grows — only one specialist loads at a time. Disk grows linearly. A 1TB drive holds 166 specialists. Each one costs $2-3 in GPU time to train.

## What Needs to Be Built

1. **Confidence scorer** — wrap every response with quality signals (partially exists in eval code)
2. **Gap detector** — cluster low-confidence queries by domain (new, ~100 lines)
3. **Data curator** — search HuggingFace + pull from E8 index (new, ~200 lines)
4. **Training trigger** — auto-launch QLoRA when gap exceeds threshold (new, wraps existing training script)
5. **Validation pipeline** — A/B test new specialist vs general (new, ~150 lines)
6. **Hot deployment** — add specialist to router without restart (router already supports dynamic keyword addition)

Total new code: ~500 lines. Everything else reuses existing infrastructure.

## Why This Works

The autoresearch loop already proved the pattern:
- **Autonomous experimentation**: 42+ experiments, zero human intervention
- **Quality-gated deployment**: keep what works, discard what doesn't
- **Incremental improvement**: each experiment builds on the best so far

Continuous specialist learning is the same pattern applied at the system level instead of the hyperparameter level. The infrastructure is identical. The difference is what's being optimized: not learning rate, but capability.

**The system doesn't just answer questions. It learns to answer questions it couldn't answer before.**

---

*This document describes the continuous learning roadmap for Project Olympus. The foundation (specialists, router, retrieval, training pipeline) is built. The continuous learning loop is the next phase.*
