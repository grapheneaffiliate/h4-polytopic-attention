# Self-Compiling Intelligence: Making Small Models Smart

## The Core Insight

A 3B model can't match a 400B model in raw pattern storage. But it
can match it in CAPABILITY if it converts soft knowledge into hard
computation — permanently, exactly, and cumulatively.

The loop:
```
1. Model attempts a task
2. Property checker verifies the result
3. If correct: compile the solution into transformer-vm (exact, permanent)
4. If wrong: the FAILURE tells us what to compile next
5. The compiled tool becomes available for harder tasks
6. Repeat from 1, but now the model is more capable
```

Each iteration, the model gets a new exact tool. The tools compound.
After 100 iterations, the model has 100 perfect subroutines that a
400B model would have to approximate from weights every time.

## Why This Is Different From Normal Tools

ChatGPT can call a calculator. That's not what we're doing.

The difference:
- **ChatGPT + calculator**: Human decides what tool to call, tool is external
- **Self-compiling**: Model discovers what it can't do, compiles the fix,
  the fix becomes part of the model's capability forever

The model doesn't just USE tools — it GROWS tools from its own failures.

## The Five Components (all built)

### 1. The Learner: SmolLM3-3B specialist
Understands algorithms conceptually. Gets ~90% of code right.
The 10% it gets wrong reveals systematic weaknesses.

### 2. The Verifier: Property checker
Catches the failures AUTOMATICALLY. No human needed.
"Output is not increasing" → the model can't do backtracking.

### 3. The Compiler: transformer-vm
Converts any C program into exact execution. 10.7K tok/s.
When the model can't implement something, we compile it.

### 4. The Index: E8 lattice + ChamberTree
Decides which compiled tool to use for which query.
O(log n) lookup. Routes queries to the right tool automatically.

### 5. The Memory: E8 knowledge lattice
Stores what the model has learned across sessions.
"LIS queries → use compiled lis.c" is a permanent memory.

## The Bootstrap Ladder

### Level 0: Built-in capabilities (today)
- Basic arithmetic (compiled: arithmetic.c)
- Fibonacci, GCD, primality (compiled)
- LIS with correct backtracking (compiled)
- Simple code generation (specialist LLM)
- Math reasoning (specialist LLM)
- QA (specialist LLM)

### Level 1: Self-discovered compilations
The model attempts harder problems. Property checker catches failures.
Each failure class gets a compiled fix:
- Sorting algorithms → compile sort.c (exact)
- Graph traversal → compile bfs.c, dfs.c (exact)
- String matching → compile kmp.c (exact)
- Matrix operations → compile matmul.c (exact)

### Level 2: Composed tools
Compiled tools can CALL each other:
- "Find the LIS, then sort the remaining elements" → lis.c + sort.c
- "Find shortest path, then compute path cost" → dijkstra.c + arithmetic.c
The model orchestrates, compiled tools execute.

### Level 3: Self-optimizing
The self-improving compiler (already built at ~/self_improving_compiler)
optimizes the compiled tools themselves:
- Discover that sort + filter can be fused into one pass
- Transfer optimization patterns via E8 lattice
- Each optimization makes the tool faster AND teaches the model

### Level 4: Reasoning scaffolds
The hardest part: compile REASONING PATTERNS, not just algorithms.
- "When the user asks 'why', retrieve context then generate explanation"
- "When code fails property check, identify the violated invariant"
- "When uncertain, say so instead of hallucinating"
These are control flow patterns, compilable into transformer-vm.

## What Makes This Novel

Nobody has combined these pieces:

| Component | Exists elsewhere? | Our version |
|-----------|------------------|-------------|
| Small model | Yes (everyone) | SmolLM3-3B with LoRA specialists |
| Property checking | Partially (unit tests) | Mathematical invariant verification |
| Exact compilation | NO (transformer-vm is unique) | Any C program → exact transformer execution |
| Geometric routing | NO (ChamberTree is unique) | O(log n) tool selection via 600-cell |
| Self-improving | Partially (AutoML) | E8 lattice transfer learning |
| Formal verification | Partially (Lean) | Machine-verified tool correctness |

The unique piece is #3: **the ability to compile arbitrary computation into
the transformer itself.** Everyone else uses external tools. We compile the
tool INTO the model's execution path.

## The Metric That Matters

Not perplexity. Not MMLU score. Not tok/s.

**Capability accumulation rate: how many new exact tools per week?**

If the model discovers 1 failure class per day, and each gets compiled:
- Week 1: 7 tools (arithmetic, fib, prime, gcd, collatz, lis, sort)
- Week 4: 28 tools (+ graph, string, matrix, date, logic...)
- Week 12: 84 tools (+ composed tools, reasoning scaffolds)

At 84 compiled tools + 3B reasoning model + E8 retrieval, the system
handles most practical queries with exact computation. The 3B model
only needs to handle NOVEL problems — and its failure on those
generates the next tool to compile.

## The Endgame

A 3B model that:
- Has 1000 compiled exact tools covering common computation patterns
- Routes queries to the right tool in O(log n) via ChamberTree
- Retrieves relevant knowledge from E8 lattice (not memorized in weights)
- Only uses the neural network for genuinely novel reasoning
- Formally verifies each new compiled tool in Lean
- Runs entirely on a consumer CPU at useful speeds

This isn't Opus. Opus does everything with one giant neural network.
This is a HYBRID SYSTEM where the neural network handles understanding
and novel reasoning, while compiled tools handle everything that can be
computed exactly. The division of labor means the neural network can be
small (3-7B) while the system capability can be enormous.

## Implementation: What to Build Next

### Phase 1: Automatic failure detection (1 week)
Extend the property checker to categorize failures:
- Backtracking errors → "needs compiled DP tool"
- Off-by-one errors → "needs compiled iteration tool"
- Wrong data structure → "needs compiled container tool"
- Hallucinated facts → "needs E8 knowledge retrieval"

### Phase 2: Automatic compilation (2 weeks)
When a failure class is detected, automatically:
1. Generate a C implementation (use Qwen 7B)
2. Verify it against the property checker
3. If verification fails, use a reference implementation
4. Compile via transformer-vm
5. Register in ChamberTree

### Phase 3: Tool composition (2 weeks)
Build a planner that decomposes complex queries into sequences
of compiled tool calls. The model generates the plan, compiled
tools execute the steps.

### Phase 4: Reasoning compilation (4 weeks)
Identify recurring reasoning patterns and compile them:
- "Retrieve, then synthesize" for QA
- "Parse, transform, format" for code
- "Decompose, solve parts, combine" for math
Each compiled pattern is a control flow graph, not a neural network.

## The Key Equation

```
System_capability = Neural_reasoning × Compiled_tools × Retrieved_knowledge

Opus:    10 × 1 × 1 = 10    (all neural, no compilation, no retrieval)
Us now:  1 × 7 × 1 = 7      (weak neural, 7 tools, no retrieval yet)
Us goal: 2 × 100 × 10 = 2000 (moderate neural, many tools, good retrieval)
```

The multiplication means you don't need a giant neural network.
You need a moderate one with excellent tools and excellent retrieval.
