# ARC-AGI: Self-Compiling Intelligence Test

## The Idea

Use the self-compiling loop to solve ARC-AGI puzzles:
1. Model sees training input/output pairs
2. Hypothesizes the transformation rule
3. Compiles rule as C program → transformer-vm executes
4. Verifies against ALL training outputs (exact match required)
5. If all match → apply to test input (answer)
6. If any mismatch → generate new hypothesis → recompile → retry

## Why This Is Perfect For Our System

- ARC requires abstract reasoning (model's job)
- ARC requires exact execution (transformer-vm's job)
- ARC has built-in verification (training pairs = ground truth)
- ARC bans cloud APIs (local-only = our advantage)
- The $2M prize rewards exactly what we built

## SDK

```bash
pip install arc-agi
```

```python
# ARC-AGI 1/2 dataset (400 training tasks, 400 eval tasks)
import arc_agi
# Tasks are JSON: {"train": [{"input": grid, "output": grid}], "test": [...]}
# Grids are 2D arrays of ints 0-9 (colors)
```

ARC-AGI-3 uses a separate game SDK with `Arcade()` class (needs API key).

## Task Format

Each task has 2-5 training pairs showing input→output transformation.
The agent must infer the rule and apply it to the test input.

Example task 0a938d79:
- Input: mostly-zero grid with 2 colored dots
- Output: columns/rows through the dots are tiled across the grid
- Rule: "find non-zero cells, extend their pattern periodically"

## Implementation Plan

### Phase 1: Rule Hypothesizer
Use the 3B/7B model to look at training pairs and generate
hypotheses as natural language:
- "Tile the colored columns across the width"
- "Fill rows containing non-zero cells"
- "Rotate the grid 90 degrees"

### Phase 2: Rule Compiler
Convert each hypothesis to a C program:
```c
void compute(const char *input) {
    // Parse grid from input
    // Apply transformation rule
    // Output transformed grid
}
```

### Phase 3: Verifier
Run compiled rule on ALL training inputs.
Compare output grids cell-by-cell with expected outputs.
If all match → hypothesis is correct.

### Phase 4: Search
If hypothesis fails, analyze WHERE it fails:
- Which cells are wrong?
- What's the pattern in the errors?
Feed this back to the model for a refined hypothesis.

## Key Insight

ARC is essentially PROGRAM SYNTHESIS:
- Training pairs = input/output examples
- The "rule" = a program that transforms input to output
- Verification = running the program on training inputs

Our system does program synthesis via:
- LLM generates candidate programs
- transformer-vm executes them exactly
- Property checker verifies against ground truth
- Auto-compiler registers successful programs

## Competition Details

- ARC Prize 2026: $2M+ prize pool
- Kaggle: kaggle.com/competitions/arc-prize-2026-arc-agi-3
- No internet during eval (local models only)
- Must open-source solution
- Deadlines: June 30, Sept 30 milestones, Nov 2 final
