# Compilation Roadmap: Exact Algorithms in Tensor Circuits

## The Principle

If a task has a known algorithm, don't train a neural network to approximate it. **Compile the algorithm** into the same tensor primitives (ReLU + linear) the network already uses. The network handles the fuzzy parts (understanding the question, formatting the answer). The compiled circuit handles the exact parts (computing the result).

## What's Built

| Domain | Compiled Operations | Status |
|--------|-------------------|--------|
| **Arithmetic** | add, sub, mul, div, pow, mod, expressions | **30/30 exact, deployed** |

## What's Next (by specialist)

### Math Specialist
| Operation | Algorithm | Difficulty | Impact |
|-----------|-----------|------------|--------|
| Unit conversion | mul/div by constants | Easy | Every conversion exact |
| Percentage | mul + div | Easy | "25% of 200" = exact |
| Date arithmetic | Zeller's congruence, day counting | Medium | Day-of-week, days between dates |
| Statistics | sum, mean, std via compiled add/mul | Medium | Descriptive stats exact |
| Matrix operations | compiled binary matmul | Hard | Linear algebra exact |

### Code Specialist
| Operation | Algorithm | Difficulty | Impact |
|-----------|-----------|------------|--------|
| String comparison | byte-level XOR gates | Easy | Exact character matching |
| String matching | compiled KMP/Boyer-Moore | Medium | Find/replace exact |
| Regex matching | compiled NFA state machine | Medium | Pattern matching exact |
| JSON/XML parsing | compiled state machine | Medium | Structure extraction exact |
| Type checking | compiled pattern matching | Hard | Formal verification |

### QA Specialist
| Operation | Algorithm | Difficulty | Impact |
|-----------|-----------|------------|--------|
| Boolean logic | AND/OR/NOT cascades | Easy | "If A and B, then C" exact |
| Graph reachability | BFS via binary matrix | Medium | "Is there a path?" exact |
| Table lookup | compiled binary search | Medium | Fact retrieval exact |
| Multi-hop reasoning | chained graph queries | Hard | "Who is the president of the country where X was born?" |

### General Specialist
| Operation | Algorithm | Difficulty | Impact |
|-----------|-----------|------------|--------|
| Counting | compiled binary counter | Easy | "How many X in this list?" exact |
| Sorting | compiled merge sort on binary keys | Medium | Ranked lists exact |
| Deduplication | hash + comparison circuits | Medium | "Remove duplicates" exact |

## Implementation Pattern

Every compiled operation follows the same pattern:

```python
class CompiledOperation:
    def can_handle(self, query: str) -> bool:
        """Does this query contain something I can compute exactly?"""

    def extract_and_compute(self, query: str) -> Optional[Result]:
        """Extract the computable part, compute it, return exact result."""

    def format_result(self, query: str, result: Result) -> str:
        """Format the exact result for the language model to explain."""
```

The router checks all compiled operations before invoking the language model:

```
Query → compiled_arithmetic.can_handle? → exact answer
      → compiled_strings.can_handle?    → exact answer
      → compiled_logic.can_handle?      → exact answer
      → (none matched) → language model → approximate answer
```

Each new compiled operation makes the system more exact without changing the language model.

## The Endgame

A system where:
- **Language model** handles understanding, reasoning, creativity, explanation
- **Compiled circuits** handle every operation with a known algorithm
- **The boundary** moves over time as more algorithms get compiled
- **The user** sees a single system that's both creative and exact

The math specialist that hallucinated on 15 × 23 this morning now computes it exactly. Tomorrow the code specialist does exact string matching. Next week the QA specialist does exact graph traversal. The system gets more reliable every week, not through training, but through compilation.
