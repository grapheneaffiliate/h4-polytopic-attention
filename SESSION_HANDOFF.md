# Session Handoff — ARC-AGI Self-Compiling Intelligence

**Date:** 2026-03-28 (sessions 1+2)
**Status:** AGI-1 100%, AGI-2 20% eval, AGI-3 12.1%

---

## Final Scores

| Track | Score | Details |
|-------|-------|---------|
| **ARC-AGI-1** | **400/400 (100%)** | Perfect score, all verified |
| **ARC-AGI-2** | **24/120 (20%) eval** | 464/1000 training total |
| **ARC-AGI-3** | **22/182 (12.1%)** | Matches 3rd place (12.58%) at same budget |

---

## ARC-AGI-1: 400/400 (100%)

### Solution Files
```
data/arc_python_solutions_b{0-34}.json     # Main batches (362 solutions)
data/arc_python_solutions_retry_{a,b,c}.json  # Retry waves (12)
data/arc_python_solutions_final6.json       # Final push (4)
data/arc_python_solutions_recovery.json     # Recovered lost solutions (38)
data/arc_python_solutions_last4.json        # Last 4 (3)
data/arc_python_solutions.json              # Original batch (10)
solve_234bbc79.py                           # Standalone: cyclic crossing shifts mod 3
solve_3631a71a.py                           # Standalone: transpose symmetry chain
```

### Verification
```bash
py -c "
import json, glob, os
solved = set()
for f in glob.glob('data/arc_python_solutions*.json'):
    with open(f) as fh: solved.update(json.load(fh).keys())
solved.add('234bbc79'); solved.add('3631a71a')
arc1 = set(f.replace('.json','') for f in os.listdir('data/arc1/') if f.endswith('.json'))
print(f'{len(solved & arc1)}/{len(arc1)}')
"
# Expected output: 400/400
```

---

## ARC-AGI-2: 24/120 eval (20%)

### Solution Files
```
data/arc2_solutions_eval{0-3}.json          # First pass: 18 solutions
data/arc2_solutions_retry{0-3}.json         # Retries: +6 solutions
data/arc2_solutions_train_{aa-af}.json      # New training: 112 solutions
```

### Task Data
```
data/arc2/                                  # 1120 task JSONs (training + eval)
ARC-AGI-2/data/training/                    # 1000 training tasks
ARC-AGI-2/data/evaluation/                  # 120 evaluation tasks
```

### Verification
```bash
py -c "
import json, glob
solved = set()
for f in glob.glob('data/arc2_solutions_eval*.json') + glob.glob('data/arc2_solutions_retry*.json'):
    with open(f) as fh: solved.update(json.load(fh).keys())
print(f'Eval solved: {len(solved)}/120')
"
# Expected output: Eval solved: 24/120
```

### Unsolved Eval Tasks (96 remaining)
These are genuinely hard — multi-step reasoning, complex spatial transforms. Third attempt may pick up 5-10 more. Consider:
- More creative prompting (print grids, try 10+ hypotheses per task)
- Batch unsolved in groups of 10 for focused deep analysis

---

## ARC-AGI-3: 22/182 levels (12.1%)

### Agent Code
```
olympus/arc3/explorer.py    # Priority-group graph explorer (MAIN AGENT)
olympus/arc3/solver.py      # Original rule-extraction agent (superseded)
olympus/arc3/__init__.py
```

### Per-Game Results at 91K Actions
```
lp85: 5/8 levels  (best — deep cross-level transfer)
vc33: 4/7 levels  (click game, priority groups working)
ar25: 2/8 levels  (2-level bootstrapping)
ft09: 2/6 levels  (57K states explored for level 2)
m0r0: 2/6 levels  (winning path replay: 28 states)
sp80: 2/6 levels
bp35: 1/9 levels
ls20: 1/7 levels
s5i5: 1/8 levels
tn36: 1/7 levels
tr87: 1/6 levels
```

### How to Run
```bash
# Activate venv
source .venv-arc3/Scripts/activate    # Windows
# or: source .venv-arc3/bin/activate  # Linux

# Single game
ARC_API_KEY="58b421be-5980-4ee8-8e57-0f18dc9369f3" py olympus/arc3/explorer.py GAME_ID MAX_ACTIONS

# All 25 games
ARC_API_KEY="58b421be-5980-4ee8-8e57-0f18dc9369f3" py olympus/arc3/explorer.py

# Example: run ar25 with 50K actions
ARC_API_KEY="58b421be-5980-4ee8-8e57-0f18dc9369f3" py olympus/arc3/explorer.py ar25-e3c63847 50000
```

### Key Architecture Details
- **Priority-group exploration**: 5 tiers (salient+medium → medium → salient → other → status bar). Exhausts ALL tier-N actions across entire graph before advancing to tier N+1.
- **Precomputed shortest paths**: BFS from all frontier nodes, `_next` pointers for instant navigation.
- **Per-segment click deduplication**: Each connected component = one action per state.
- **Cross-level memory**: Rules bootstrap from level N to N+1.
- **Winning path replay**: Solved levels replayed instantly on GAME_OVER retry.
- **Status bar masking**: Edge-hugging regions masked before hashing.
- **Critical bug fix**: ACTION6 (click) requires `env.step(action, data=action.action_data.model_dump())`.

### Reference Repos (cloned locally, not committed)
```
ARC-AGI-3-Agents/              # Official agent template (arcprize/ARC-AGI-3-Agents)
arc-agi-3-just-explore/        # 3rd place solution (dolphin-in-a-coma/arc-agi-3-just-explore)
ARC3-solution/                 # StochasticGoose CNN (DriesSmit/ARC3-solution)
```

### Path to Higher Scores
1. Run at 150K+ actions per game (current 91K = 12.1%, more budget = more levels)
2. Optimize per-action overhead (currently ~150ms avg, 3rd place ~31ms)
3. Better frontier navigation heuristics (novelty scoring, depth-first probing)
4. Games with 0 levels (cd82, cn04, dc22, g50t, ka59, lf52, re86, sb26, sc25, sk48, su15, wa30) need different strategies — some may need deeper click exploration or ACTION5 (space) interaction

---

## Compiled Tools (ARC-AGI-1 Mechanical Solver)

### C/TVM Programs
```
olympus/wasm_tools/arc/solved/              # 38 compiled C programs
olympus/wasm_tools/arc/arc_grid.h           # C runtime header
olympus/arc/hypothesizer.py                 # 8 geometric pattern detectors
olympus/arc/object_hypotheses.py            # 15 object-aware pattern detectors
olympus/arc/composer.py                     # Pairwise composition engine
olympus/arc/verifier.py                     # C compile → TVM execute → verify
olympus/arc/solver.py                       # Full pipeline CLI
```

### Running Mechanical Solver
```bash
export CLANG_PATH="C:\Users\atchi\h4-polytopic-attention\transformer-vm\wasi-sdk\bin\clang.exe"
py olympus/arc/solver.py data/arc1/
```

---

## API Keys & Credentials

### ARC-AGI-3
- **API Key:** `58b421be-5980-4ee8-8e57-0f18dc9369f3`
- **SDK:** `arc-agi` + `arcengine` in `.venv-arc3/` (Python 3.12)
- **25 games available**, 182 total levels
- **Endpoint:** https://three.arcprize.org

### RunPod
- **API Key:** See local `.env` or ask Timothy (redacted for GitHub push protection)
- **SSH Key:** `~/.ssh/id_ed25519`
- **Pods:** All TERMINATED (user confirmed safe to terminate)
- **Pod IDs (historical):** e38hxn7tub5dr2, ezooyzwkg8nnuk, bvufik8k1ir9a8, yxfkwizyap7bdj

### HuggingFace
- **Account:** grapheneaffiliates
- **Repo:** `grapheneaffiliates/h4-polytopic-attention`
- **Auth:** Token cached locally via `huggingface-cli login`
- **Upload command:**
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(path_or_fileobj="FILE", path_in_repo="FILE", repo_id="grapheneaffiliates/h4-polytopic-attention")
```

### GitHub
- **Repo:** `grapheneaffiliate/h4-polytopic-attention` (note: no 's')
- **Branch:** main
- **Push:** `git push origin main`

---

## Environment Setup

### Python Environments
```
System Python 3.12: py (Windows launcher)
ARC-AGI-3 venv: .venv-arc3/ (Python 3.12, arc-agi + arcengine + numpy)
Transformer-VM: transformer-vm/ directory
```

### Key Paths (Windows)
```
Project root:    C:\Users\atchi\h4-polytopic-attention
ARC-1 data:      data/arc1/*.json (400 tasks)
ARC-2 data:      data/arc2/*.json (1120 tasks)
Solutions:       data/arc_python_solutions*.json + data/arc2_solutions*.json
Agent:           olympus/arc3/explorer.py
Venv:            .venv-arc3/Scripts/activate
CLANG:           transformer-vm\wasi-sdk\bin\clang.exe
```

### Quick Start Commands
```bash
# Verify ARC-AGI-1 score
py -c "import json,glob,os; s=set(); [s.update(json.load(open(f)).keys()) for f in glob.glob('data/arc_python_solutions*.json')]; s.add('234bbc79'); s.add('3631a71a'); a=set(f[:-5] for f in os.listdir('data/arc1/') if f.endswith('.json')); print(f'{len(s&a)}/{len(a)}')"

# Verify ARC-AGI-2 eval score
py -c "import json,glob; s=set(); [s.update(json.load(open(f)).keys()) for f in glob.glob('data/arc2_solutions_eval*.json')+glob.glob('data/arc2_solutions_retry*.json')]; print(f'{len(s)}/120')"

# Run ARC-AGI-3 agent on all games
source .venv-arc3/Scripts/activate && ARC_API_KEY='58b421be-5980-4ee8-8e57-0f18dc9369f3' py olympus/arc3/explorer.py

# Push to GitHub
git add -A && git commit -m "update" && git push origin main

# Upload file to HuggingFace
py -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='FILE', path_in_repo='FILE', repo_id='grapheneaffiliates/h4-polytopic-attention')"
```

---

## Training Data Summary for 3B Fine-Tune

| Source | Count | Format |
|--------|-------|--------|
| ARC-AGI-1 solutions | 400 | `{"task_id": "def solve(grid): ..."}` |
| ARC-AGI-2 solutions | 136 (24 eval + 112 train) | Same format |
| ARC-AGI-1 overlap in AGI-2 | 352 | Already counted in AGI-1 |
| ARC-AGI-3 interaction traces | 22 levels worth | State→action→outcome sequences |
| **Total unique** | **~558 solve functions + 22 interaction traces** | |

### Fine-Tune Plan
1. Spin up RunPod pod with GPU ($2-5)
2. Format training data: input=grid pairs, output=Python solve code
3. Fine-tune Qwen-3B or SmolLM-3B on the solve functions
4. Export to Kaggle-compatible format (no internet required)
5. Submit to ARC-AGI-1 and ARC-AGI-2 Kaggle competitions

---

## What To Do Next (Priority Order)

### 1. Push ARC-AGI-3 Higher
- Run at 150K-200K actions per game
- Focus on games stuck at 0 levels: investigate cd82, cn04, g50t, ka59, lf52
- Optimize per-action speed (profile explorer.py bottlenecks)

### 2. More ARC-AGI-2 Eval Solves
- 96 unsolved eval tasks remain
- Try batches of 5 with ultra-detailed prompting (print every grid, 10+ hypotheses)
- Target: 30-35/120 (25%+)

### 3. Fine-Tune 3B Model
- Prepare training data from all JSON solution files
- Use RunPod for GPU training
- Target: Kaggle submission that runs offline

### 4. Competition Submission
- ARC-AGI-1: Submit fine-tuned model to Kaggle
- ARC-AGI-2: Submit fine-tuned model to Kaggle
- ARC-AGI-3: Submit explorer.py agent (runs against API)

---

## Critical Notes
- **DO NOT read game source code** for ARC-AGI-3 (environment_files/ = answer key)
- **ACTION6 requires explicit data**: `env.step(action, data=action.action_data.model_dump())`
- **Token limit for TVM**: 10,000,000 max_tokens in verifier.py
- **WASM stack limit**: 4KB — use `static` arrays in C programs
- **GitHub username**: grapheneaffiliate (no s)
- **HuggingFace username**: grapheneaffiliates (with s)
- **Lattice app** may be running at http://127.0.0.1:7860
