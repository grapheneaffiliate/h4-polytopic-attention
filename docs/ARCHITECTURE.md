# Architecture Guide

## System Overview

H4 Polytopic Attention is a transformer executor where:
- Programs compile into analytically constructed transformer weights
- Attention heads operate in 4D using H4 (600-cell) geometry
- Memory is indexed via E8 lattice Voronoi cells
- The E8 -> H4 projection unifies memory and attention into one geometric system

This document covers the internal architecture in detail.

---

## 1. Execution Model

### Von Neumann Mapping

```
+------------------+     +------------------+
|    FETCH         |     |   StateEncoder   |
|  Read instr at   | --> | Encode IP, regs, |
|  current IP      |     | opcode as d_model|
+------------------+     | vector on S^3    |
                          +------------------+
                                  |
                                  v
+------------------+     +------------------+
|    DECODE        |     |  Attention Heads  |
|  Determine       | --> | H4 ChamberTree   |
|  operation       |     | lookup in trace   |
+------------------+     +------------------+
                                  |
                                  v
+------------------+     +------------------+
|    EXECUTE       |     |    FFN Layers     |
|  Perform the     | --> | Opcode decode +  |
|  operation       |     | ALU operations    |
+------------------+     +------------------+
                                  |
                                  v
+------------------+     +------------------+
|    WRITEBACK     |     |  Update trace,   |
|  Store results   | --> | registers, and   |
|                  |     | lattice memory   |
+------------------+     +------------------+
```

### Execution Loop (per step)

```python
while not halted:
    instr = program[ip]
    state_vec = encoder.encode_state(ip, registers, instr, step)
    trace.append(state_vec)

    # Transformer forward pass over full trace
    output = transformer.forward(trace)

    # Execute the instruction (updates registers, ip)
    execute_instruction(instr)
```

The transformer's forward pass is run at every step, giving it access to the full execution history. Attention heads look back at prior states to resolve memory lookups and control flow patterns.

---

## 2. State Encoding

### Vector Layout (d_model = 32)

```
Byte range    Contents                    Encoding method
---------     --------                    ---------------
[0:4]         Instruction pointer         Golden-angle spiral on S^3
[4:8]         Opcode                      600-cell vertex (10 vertices apart)
[8:16]        Register file (8 regs)      tanh(reg / 100) normalization
[16:20]       Operand A                   Golden-angle spiral on S^3
[20:24]       Operand B                   Golden-angle spiral on S^3
[24:28]       Destination register        Golden-angle spiral on S^3
[28:32]       Step counter / phase        phi-scaled trigonometric
```

### Golden-Angle Spiral

Each integer (IP, register index, address) maps to a well-separated direction on S^3:

```python
theta1 = n * 2 * pi * phi_inv        # golden angle in first plane
theta2 = n * pi * phi_inv * 0.7      # golden angle in second plane
r1, r2 = cos(theta2), sin(theta2)
vec = [r1*cos(theta1), r1*sin(theta1), r2*cos(theta1*phi), r2*sin(theta1*phi)]
```

The golden angle (2*pi*phi_inv ~ 137.5 degrees) ensures maximum angular separation between consecutive indices, so nearby IPs don't interfere in attention.

### Opcode Encoding

Each opcode maps to a specific 600-cell vertex, spaced 10 apart:

```
LOAD      -> vertex[0]      STORE_MEM -> vertex[80]
ADD       -> vertex[10]     LOAD_MEM  -> vertex[90]
SUB       -> vertex[20]
MUL       -> vertex[30]
STORE     -> vertex[40]
JMP       -> vertex[50]
JNZ       -> vertex[60]
HALT      -> vertex[70]
```

With 120 vertices, spacing of 10 ensures ~36 degrees angular separation between opcodes --- well above the Coxeter chamber resolution.

---

## 3. Attention Architecture

### Head Allocation

With d_model = 32 and d_head = 4, we get 8 attention heads:

```
Heads 0-1:  IP Lookup
            Keys encode instruction pointers
            Queries find matching IPs in the execution history
            Purpose: "What happened last time we were at this IP?"

Heads 2-3:  Register Lookup
            Keys encode register indices
            Queries find register values at prior steps
            Purpose: "What was the value of R[a] when we last wrote it?"

Heads 4-5:  Operand Fetch
            Keys encode operand values
            Queries fetch operand data from the trace
            Purpose: "What operands were used with this instruction?"

Heads 6-7:  Control Flow
            Keys encode branch conditions
            Queries predict branch direction
            Purpose: "Was the branch taken last time?"
```

### ChamberTree Structure

Each head maintains a ChamberTree for its 4D key-value pairs:

```
             Root (4 simple roots of H4)
              /    |    |    \
           [0] [1] ... [14] [15]     <- 16 buckets (level 0)
            |
         Rotated roots (by pi/5)
          /    |    |    \
       [0] [1] ... [14] [15]        <- 16 sub-buckets (level 1)
            |
         Rotated roots (by pi/5 * phi)
          /    |    |    \
       [0] [1] ... [14] [15]        <- 16 leaf buckets (level 2)
            |
         [k1, v1, ts1]              <- Leaf entries (max 64 per leaf)
         [k2, v2, ts2]
         ...
```

**Bucket index computation:**
```rust
fn bucket_index(roots: &[Vec4; 4], key: Vec4) -> usize {
    let mut idx = 0;
    if key.dot(roots[0]) >= 0.0 { idx |= 1; }
    if key.dot(roots[1]) >= 0.0 { idx |= 2; }
    if key.dot(roots[2]) >= 0.0 { idx |= 4; }
    if key.dot(roots[3]) >= 0.0 { idx |= 8; }
    idx  // 0..15
}
```

**Approximate query:** At each level, visit the primary bucket plus 4 Hamming-1 neighbors (flip one bit each). This gives 5/16 = 31.25% scan per level. Over 3 levels: (5/16)^3 = 3.05%.

---

## 4. E8 Lattice Memory (Phase 4)

### Address Translation Pipeline

```
Linear address (u64)
    |
    | _address_to_embedding()
    v
8D golden-angle embedding (Vec8)
    |
    | decode_to_e8()
    v
E8 lattice point (LatticePoint, [i32; 8])
    |
    | hash
    v
HashMap bucket (Vec<MemoryEntry>)
    |
    | project_to_h4()
    v
4D H4 vector (Vec4)
    |
    | ChamberTree.insert()
    v
Attention-queryable cache
```

### E8 Closest-Point Decoder

The E8 lattice decomposes as D8 + (D8 + [1/2]^8). The decoder finds which coset point is closest:

```
Input: point p in R^8

Coset 1 (D8): integers with even sum
  1. f1 = round(p) componentwise
  2. If sum(f1) is odd:
     - Find component with largest rounding error
     - Flip it to the other nearest integer
  3. dist1 = ||p - f1||^2

Coset 2 (D8 + [1/2]^8): half-integers with even sum
  1. f2 = floor(p) + 0.5 componentwise
  2. If sum(f2) is not an integer (parity check):
     - Find component with largest error
     - Shift it by +-1
  3. dist2 = ||p - f2||^2

Return: f1 if dist1 <= dist2, else f2
```

**Representation:** All coordinates stored as 2x integers ([i32; 8]) so both integer and half-integer lattice points use the same type. Integer point (1, 0, ...) becomes (2, 0, ...); half-integer (1/2, 1/2, ...) becomes (1, 1, ...).

### Memory Operations

**STORE_MEM (R[a] -> mem[R[b]]):**
1. Convert address R[b] to 8D embedding via golden-angle spiral
2. Decode embedding to E8 lattice point
3. Insert (embedding, value, address, timestamp) into that cell's bucket
4. Project embedding to 4D via E8->H4 and insert into ChamberTree
5. If bucket exceeds 240 entries (kissing number), evict oldest (LRU)

**LOAD_MEM (mem[R[a]] -> R[dest]):**
1. Convert address R[a] to 8D embedding
2. Decode to E8 lattice point (primary cell)
3. Search primary cell for closest embedding match
4. Search 240 kissing neighbors for closer matches
5. Return value of closest match

### Utilization Characteristics

With structured (Wasm-like) embeddings at 10,000 steps:
- **106 occupied cells** out of ~10,000 entries
- **Max bucket size: 240** (saturated cells hit the kissing number bound)
- **Primary cell hit rate: 100%** (exact queries always find their target in the primary cell)
- **Average bucket size: 83.7** entries

The clustering behavior is expected: structured execution traces have correlated embeddings that map to nearby E8 lattice points. The 240 cap prevents any cell from growing unbounded.

---

## 5. Phi-Recursive State Encoding

### Fibonacci Checkpoint Levels

Long traces are compressed using Fibonacci-spaced checkpoints:

```
Level 0: checkpoint at every step          spacing = F(1) = 1
Level 1: checkpoint at every phi steps     spacing = F(2) = 1
Level 2: checkpoint at every phi^2 steps   spacing = F(3) = 2
Level 3: checkpoint at every phi^3 steps   spacing = F(4) = 3
Level 4: checkpoint at every phi^4 steps   spacing = F(5) = 5
...
Level k: checkpoint at every phi^k steps   spacing = F(k+1)
```

### Zeckendorf Retrieval

To retrieve state at step t, use Zeckendorf decomposition:
- Every positive integer is a unique sum of non-consecutive Fibonacci numbers
- Decompose distance to target step into Fibonacci components
- Traverse checkpoint levels to reconstruct state

**Example:** Retrieve step 20 = F(8) + F(4) + F(2) = 21 - 1? No: 20 = 13 + 5 + 2 = F(7) + F(5) + F(3). Jump to level 7 checkpoint closest to target, then level 5, then level 3.

---

## 6. Weight Construction

### Analytical (No Training)

Transformer weights are computed directly from the geometry:

**Attention weights:**
```python
# W_K for IP-lookup heads: project to first 4 dims (IP encoding)
W_K[0] = I_4   (4x32, first 4 columns of identity)
W_K[1] = I_4   (duplicate for robustness)

# W_K for register-lookup heads: project to dims 8:16 (register file)
W_K[2][8:12, :] = I_4
W_K[3][12:16, :] = I_4
```

**FFN weights:**
```python
# Layer 1: opcode detection
# Each opcode vertex direction becomes a row in W1
# ReLU activation selects the matching opcode

# Layer 2: operation execution
# Maps detected opcode to register update
```

The key insight is that H4 geometry provides enough angular resolution (14,400 chambers) that each instruction type occupies a distinct Coxeter chamber, making the attention lookup exact rather than approximate.

---

## 7. Rust Implementation Details

### SIMD Alignment

```rust
#[repr(align(32))]   // Vec4: one 256-bit AVX2 register
pub struct Vec4(pub [f64; 4]);

#[repr(align(64))]   // Vec8: two 256-bit AVX2 registers
pub struct Vec8(pub [f64; 8]);
```

Dot products are written as single expressions so LLVM can emit `vmulpd` + horizontal add:
```rust
pub fn dot(self, other: Self) -> f64 {
    (a[0]*b[0] + a[1]*b[1]) + (a[2]*b[2] + a[3]*b[3])
}
```

### Thread Safety

ChamberTree is `Send + Sync` for rayon parallelism across attention heads:
```rust
unsafe impl Send for ChamberTree {}
unsafe impl Sync for ChamberTree {}
```

Parallel queries distribute heads across threads:
```rust
pub fn query_approx_par(&self, embedding: &[f64]) -> Vec<Option<(f64, u64)>> {
    self.heads.par_iter().enumerate().map(|(h, head)| {
        head.query_approx(query_vec)
    }).collect()
}
```

### Build Configuration

```toml
[profile.release]
opt-level = 3    # Maximum optimization (enables auto-vectorization)
lto = true       # Link-time optimization (enables cross-crate inlining)
```

---

## 8. MCP Server Architecture

```
Claude Code <--stdio--> h4_mcp_server.py <--import--> weight_compiler.py
                                          <--import--> h4_polytopic_attention.py
```

The server runs as a subprocess of Claude Code, communicating via stdin/stdout JSON-RPC (MCP protocol). It uses Max plan OAuth --- the compute happens locally, and Claude's API calls use the existing subscription.

### Tool Dispatch

```python
@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "h4_fibonacci":       return await _h4_fibonacci(arguments)
    elif name == "h4_compile_and_run": return await _h4_compile_and_run(arguments)
    elif name == "h4_geometry_info":   return await _h4_geometry_info(arguments)
    elif name == "h4_benchmark":       return await _h4_benchmark(arguments)
    elif name == "h4_lattice_memory":  return await _h4_lattice_memory(arguments)
```

Each tool creates a fresh H4Executor instance, suppresses stdout (the executor is chatty), runs the program, and returns structured JSON results.

---

## 9. Phase 5: Trainable Hybrid Attention (PyTorch)

### Frozen + Trainable Architecture

The key insight from the Fibonacci proof-of-concept (26 trainable params on a frozen H4 backbone): the geometric backbone provides such strong inductive bias that only small trainable adapters are needed. The learned W_nudge matrix converged to rank-1, aligning 93% with a 600-cell vertex. Phase 5 generalizes this to full language modeling.

```
Input tokens
    |
    v
Token Embedding + Golden-Angle Positional Encoding (phi-inverse spacing)
    |
    v
+-------------------------------------------------------------------+
|  H4TransformerBlock (repeated N times)                            |
|                                                                    |
|  LayerNorm -> H4AttentionLayer -> Residual                        |
|                |                                                   |
|                +-- FROZEN: 600-cell vertices, simple roots,       |
|                |           E8->H4 projection, ChamberTree         |
|                |                                                   |
|                +-- TRAINABLE: W_q/k/v_proj, W_nudge (4x4/head), |
|                |              chamber_bonus (16/head), W_out      |
|                |                                                   |
|                +-- Forward path:                                  |
|                    1. Project input -> Q,K,V per head             |
|                    2. Normalize Q,K to S^3 (unit 4-sphere)        |
|                    3. Apply W_nudge (small query rotation)        |
|                    4. ChamberTree top-k lookup OR full attention  |
|                    5. Soft chamber bonus on keys (differentiable) |
|                    6. Softmax over candidates -> weighted V sum   |
|                    7. Concatenate heads -> W_out                  |
|                                                                    |
|  LayerNorm -> FFN (Linear -> GELU -> Linear) -> Residual         |
+-------------------------------------------------------------------+
    |
    v
LayerNorm -> LM Head (weight-tied with token embedding)
    |
    v
Logits (vocab_size)
```

### ChamberTree as Preprocessing Filter

The ChamberTree is not differentiable --- it's a discrete tree traversal. The trick:

1. **Build:** Insert all K vectors into ChamberTree (numpy, per sequence)
2. **Query:** For each Q, tree returns top-k candidate key indices in O(log t)
3. **Gather:** Use indices to select candidate K, V tensors in PyTorch
4. **Attend:** Compute differentiable softmax attention over k candidates only
5. **Gradients:** Flow through W_q_proj, W_k_proj, W_nudge, chamber_bonus, W_v_proj normally

This gives O(k) attention per query where k << t. The tree is just a fast filter.

### Soft Chamber Bonus

The chamber_bonus parameter adds a per-key attention bias based on which Coxeter chamber the key lands in. Since softmax is shift-invariant, the bonus must be per-key (not per-query).

For gradient flow, chamber membership uses soft assignment:
```
k_dots = K @ simple_roots^T            # (B, T, H, 4)
soft_signs = sigmoid(k_dots * 3.0)     # soft bit per root
chamber_weights = prod over 4 bits     # (B, T, H, 16) soft membership
k_bonus = sum(chamber_weights * bonus) # (B, T, H) per-key bonus
scores += k_bonus                      # differentiable
```

### Golden-Angle Positional Encoding

Instead of sinusoidal or RoPE, positions are encoded using phi-inverse spacing:

```
Position n -> angle = n * 2*pi * phi^-1
```

phi^-1 ~ 0.618 is the most irrational number (hardest to approximate by rationals). This guarantees:
- Consecutive positions are maximally separated (~137.5 degrees apart)
- No position vectors repeat or nearly repeat at any scale
- Long-range positions compress via Zeckendorf decomposition (sum of Fibonacci-indexed embeddings)

Frequency scales across dimension pairs use phi powers: phi^(-k/n_pairs), giving geometrically spaced frequencies anchored to the golden ratio.

### Diagnostic Metrics

Phase 5 tracks geometric health during training:

| Metric | What it measures | Healthy range |
|--------|-----------------|---------------|
| `chamber_entropy` | Shannon entropy of chamber utilization | > 2.0 (max ~2.77 for 16 chambers) |
| `nudge_rank` | Ratio sigma_1/sigma_2 of W_nudge SVD | > 2.0 (rank-1 = focused direction) |
| `geo_alignment` | Max dot of nudge direction with 600-cell vertex | > 0.9 (geometry attracts learning) |
| `scan_ratio` | Fraction of keys examined per query (tree mode) | << 1.0 (lower = more pruning) |

### Autoresearch Protocol

`h4_program.md` defines an autonomous experiment loop:
- Agent modifies ONLY `train_cpu.py` (trainable adapters + hyperparameters)
- Frozen geometry is off-limits
- 2-minute CPU budget per experiment
- Keep/discard based on val_bpb improvement
- Git commit on keep, checkout on discard
- Results tracked in `results.tsv` (untracked)

---

## 10. Phase 6: BitNet b1.58 Ternary Weights

### Why Ternary Fits H4 Geometry

The ChamberTree routing already performs 1-bit quantization: `sign(dot(query, root_i))` for 4 roots gives a 4-bit chamber index. BitNet b1.58 extends this to ternary {-1, 0, +1} for all trainable weights.

The key insight: ternary preserves signs, and chamber assignments only depend on signs. Therefore ternary-quantized attention heads navigate the same Coxeter chambers. At initialization this was confirmed at **97.9% chamber preservation**. After training, preservation drops to ~76% as the model finds its own geometric routing in the ternary-constrained space --- different from the float routing but equally effective (0.003 bpb gap).

### BitLinear: Straight-Through Estimator

```
Training forward pass:
    1. Quantize weights:  w_q = RoundClip(w / mean(|w|), -1, +1)
    2. STE:  w_ste = w + (w_q * scale - w).detach()
       → forward sees quantized value, backward sees float shadow
    3. Quantize activations: x_q = RoundClip(x * 127 / max(|x|), -127, 127)
    4. STE:  x_ste = x + (x_q * scale / 127 - x).detach()
    5. y = Linear(x_ste, w_ste)  → full gradient flow to w and x

Inference forward pass (after freeze()):
    1. Load frozen int8 weights (only {-1, 0, +1} values)
    2. y = x @ w_frozen.float() * scale  → matmul is add/sub only
```

### What Stays Float32

| Component | Reason |
|-----------|--------|
| 600-cell vertices (120 x 4) | Frozen buffer, static lookup |
| H4 simple roots (4 x 4) | Frozen buffer, need exact signs for chamber routing |
| E8->H4 projection (4 x 8) | Frozen buffer |
| chamber_bonus (n_heads x 16) | Too small to quantize (128 values), needs continuous gradients |
| Token embeddings | Lookup table, not a matmul |
| LayerNorm parameters | Small, need float precision |
| LM head | Weight-tied with embeddings |

### What Becomes Ternary

| Component | Params (per layer) | Ternary Size (per layer) |
|-----------|-------------------|------------------------|
| W_q_proj, W_k_proj | 2x d_model*32 | varies by d_model |
| W_v_proj, W_out | 2x d_model*128 | varies by d_model |
| FFN up + down | 2x d_model*d_ffn | varies by d_model |

Example configs from autoresearch (best results):
- **d_model=128, 6 layers** (float best): ~348K params, ~1.4 MB float32
- **d_model=256, 4 layers** (ternary best): ~1.1M params, ~310 KB at 1.58 bits (~17x compression)

### Ternary Inference Path

```
Token → Embedding (float16 lookup)
      → BitLinear Q/K/V (ternary: add/sub only)
      → Normalize to S³ (float, 4-dim per head)
      → W_nudge rotation (float, 4x4 per head — only 16 values)
      → ChamberTree lookup (integer sign comparisons, 3.1% scan)
      → Softmax over k candidates (float, k << t)
      → Weighted sum of values (float x ternary)
      → BitLinear output (ternary: add/sub only)
      → BitLinear FFN (ternary: add/sub only)
      → LM head → next token
```

Float multiplies in the critical path: root dot products (4x4), softmax, dequant scales (one per layer). Everything else is integer addition/subtraction.

---

## 11. Phase 7: Unified Geometric RAG

### E8 Retrieval + H4 Ranking + Ternary Generation

Phase 7 unifies document retrieval and generation through the same E8→H4 projection. In standard RAG systems, retrieval (embedding model + vector database) and generation (LLM) are separate systems with different geometries. H4 RAG uses one geometry for both:

1. **Encode:** Documents chunk into 8D embeddings via golden-angle spiral placement, stored in E8 Voronoi cells (`rag/encoder.py`)
2. **Retrieve:** Questions project to 8D, E8 lattice lookup finds nearest chunks in O(1) + 240 neighbors (`h4_polytopic_attention.py: E8LatticeIndex`)
3. **Rank:** Question and passage both encode to 4D vectors on S³ via H4 attention. Relevance = dot product in H4 space — the same metric as attention routing (`rag/ranking_model.py`)
4. **Generate:** Ternary H4 model reads retrieved context and produces answers (`h4_language_model.py`)

### Contrastive Ranking in H4 Space

The ranker uses InfoNCE loss with in-batch negatives. For a batch of B questions, each paired with its correct passage, the similarity matrix is B×B where entry [i,j] = dot(q_i, p_j) on S³. The diagonal entries (correct pairs) should dominate each row.

Key finding: ternary models need ~2x higher contrastive temperature (0.15 vs 0.07) because BitLinear produces noisier similarity scores that require softer probability distributions for gradient flow.

### Two-Stage Retrieval Pipeline

The production pipeline uses both architectures:

1. **Bi-encoder** (fast, broad): Encodes question and passage separately, scores by H4 dot product. At 3.7M params: R@5=100%, MRR=0.93. The answer is always in the top 5 results. Cost: ~20ms for all documents.

2. **Pre-trained cross-encoder** (precise, narrow): A pre-trained reranker (ms-marco-MiniLM-L-6-v2, 22M params) scores each candidate by reading question + passage jointly. Achieves **98.5% R@1** on the bi-encoder's top-5 candidates. Cost: ~500ms for 5 candidates.

Pipeline: H4 bi-encoder retrieves top-5 (100% recall, 20ms) → MiniLM reranks top-5 (98.5% precision, ~500ms) → return best.

Our H4 cross-encoder (25M ternary params, 8h overnight on 5.9K SQuAD pairs) reached **80% R@1 peak** (69% final eval) — a breakthrough showing H4 cross-attention learns question-to-passage alignment through Coxeter chambers. Trajectory: 29% (1hr) → 52% (3hr) → 80% (7hr). The system ships with MiniLM (98.5%) for production accuracy, with the H4 cross-encoder as a fully geometric, zero-dependency alternative.

### Full-Scale Results

- **Language modeling:** 24M ternary params, PPL 10.0 on TinyStories (8h CPU), beats published 33M baseline
- **Bi-encoder retrieval:** R@5=100%, MRR=0.93 at 3.7M params (answer always in results)
- **Combined reranking:** 98.5% R@1 (H4 bi-encoder retrieval + pre-trained MiniLM reranker)
- **Passage ranking (min viable):** 41.5% R@1 at 870K params (12x random chance)
- **Retrieval:** 7.8ms per query via E8 lattice
- **Cost:** $0/month (replaces $500-2,000/month enterprise RAG stack)
