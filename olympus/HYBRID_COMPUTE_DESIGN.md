# Hybrid Compute Architecture: H4 Language + 2D Exact Execution

## The Problem

The math specialist (SmolLM3-3B + QLoRA, loss 0.23) learned beautiful mathematical
reasoning patterns from MetaMathQA but can't do reliable arithmetic. It sees "15 × 23"
and pattern-matches to training data instead of calculating. This is a fundamental
limitation of language models: they learn statistical patterns, not computation.

## The Solution

Percepta (Tzamos et al., 2026) proved that compiled programs can execute inside
transformer weights with zero error at 32,000 tok/s. Their key insight: 2D attention
heads turn max-dot-product queries into convex hull lookups, enabling O(log t) exact
execution of arbitrary programs.

We integrate this as a **dual-path architecture**: H4 4D heads for language reasoning,
2D execution heads for exact computation. Same transformer, different heads for
different jobs.

## Architecture

```
User: "What is 15 × 23?"
        |
        v
  [Math Specialist — SmolLM3-3B hybrid]
        |
   +----+----+
   |         |
   v         v
 H4 4D    2D Exec
 Heads     Heads
 (reason)  (compute)
   |         |
   |    [WASM: i32.const 15
   |     i32.const 23
   |     i32.mul → 345]
   |         |
   +----+----+
        |
        v
  "15 × 23 = 345. To see why, note that
   15 × 20 = 300 and 15 × 3 = 45,
   so 300 + 45 = 345."
```

The model:
1. Recognizes "15 × 23" needs computation (language understanding, H4 heads)
2. Emits a micro-program: `i32.const 15; i32.const 23; i32.mul` (routing decision)
3. Executes it in the 2D execution path (exact, O(log t), zero error)
4. Reads the result (345) back into the language path
5. Formats the explanation (language generation, H4 heads)

## Head Allocation

In a transformer with d_model=512 and mixed heads:

| Head Type | Dimension | Count | Purpose |
|-----------|-----------|-------|---------|
| H4 attention | 4D | ~100 heads | Language understanding, generation, retrieval |
| 2D execution | 2D | ~28 heads | Exact arithmetic, logic, program execution |
| Standard | varies | ~128 heads total | FFN, embeddings (unchanged) |

The 2D execution heads only activate when the model detects a computation. For pure
language queries, they contribute nothing (gated off). For math, they provide exact answers.

## What We Can Build Now (Without Percepta's Weights)

Percepta's full WASM interpreter is not publicly available yet. But we can build
a simpler version that handles the operations the math specialist actually needs:

### Tier 1: Basic Arithmetic (buildable now)
- Integer addition, subtraction, multiplication, division
- Modular arithmetic
- Comparison operators
- These can be compiled into 2D attention heads analytically

### Tier 2: Extended Math (needs more work)
- Floating point operations
- Exponentiation, logarithms (iterative)
- Square roots (Newton's method compiled)
- Matrix operations (compiled loops)

### Tier 3: Full WASM (needs Percepta cooperation or reimplementation)
- Arbitrary C programs compiled to WASM
- Full interpreter in weights
- Memory management, control flow, function calls

## Integration with Olympus Math Specialist

The math specialist currently has two failure modes:
1. **Arithmetic errors**: 15 × 23 = wrong (pattern matching, not computing)
2. **Reasoning errors**: wrong approach to word problems (limitation of 3B params)

The hybrid architecture fixes #1 completely. It doesn't fix #2 — that requires
better reasoning training data or larger models. But #1 is the embarrassing failure
(a model that can't multiply) while #2 is the expected limitation (small model,
hard problems).

## Implementation Plan

### Phase 1: Detect computation needs (can build now)
```python
def needs_computation(query):
    """Detect if a query requires exact arithmetic."""
    patterns = [
        r'\d+\s*[\+\-\*\/\^]\s*\d+',  # 15 * 23
        r'calculate|compute|evaluate',   # explicit requests
        r'how much is|what is \d',       # implicit math
    ]
    return any(re.search(p, query, re.I) for p in patterns)
```

### Phase 2: Compile arithmetic into 2D heads (research task)
Following Percepta's method:
- Each arithmetic operation becomes a sequence of tokens
- 2D attention heads execute the operation via convex hull queries
- Result is read back as a token

### Phase 3: Train switching mechanism
The model learns when to use language (H4 4D) vs computation (2D exec):
- During training, annotate examples with mode tags
- Model learns to emit mode-switch tokens
- Gating mechanism routes to appropriate heads

## Timeline

- **Now**: Document design, implement computation detection
- **This month**: Implement basic arithmetic in 2D heads (add, mul, div)
- **When Percepta publishes**: Integrate their full WASM interpreter
- **After integration**: The math specialist never hallucinates on arithmetic again

## References

- Percepta: "Can LLMs Be Computers?" (Tzamos et al., 2026)
  - Key paper: 2D convex hull attention, O(log t) exact execution
  - Their HullKVCache is the 2D analog of our ChamberTree
- H4 Polytopic Attention (this project)
  - 4D Coxeter chamber attention, O(log t) language generation
  - ChamberTree is the 4D analog of their HullKVCache
