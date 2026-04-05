# Abstract BFS

## Concept
Read game source code, extract the state transition function as pure Python operations on tuples/arrays, then BFS at 20-30K states/s without touching the game engine.

## Speed Comparison
| Method | States/s | Notes |
|--------|----------|-------|
| Deepcopy BFS | 30-400 | Limited by deepcopy (15ms) |
| Reset+replay BFS | 17-80 | Limited by replay length |
| Abstract tuple BFS | 20,000-30,000 | 100x faster |

## Games Using This
- **LP85**: Button permutations → BFS on position tuples. 5/8 in 22min.
- **TN36**: Opcodes → algebraic computation. 5/7 in 0.4s.
- **FT09**: Clicks → GF(p) linear algebra. 6/6 instant.

## How to Build One
1. Read the game's `step()` method
2. Identify the minimal state: which variables change?
3. Identify the win condition: what does it check?
4. Implement state transition as pure tuple operations
5. BFS with visited set (tuples are hashable)

## Common Pitfalls
- **Dict key collision** (LP85): duplicate sprite names as dict keys → use lists
- **Deepcopy closures** (TN36): lambdas reference wrong object → avoid deepcopy
- **Camera scaling** (LP85): coordinates differ between local engine and API
- **Per-level variation**: button layouts, sprite counts, and camera change per level
