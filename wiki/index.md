# ARC-AGI-3 Wiki

Knowledge base for solving all 25 ARC-AGI-3 games. Updated by LLM sessions.

## Score: 45/163 (27.6%) confirmed on live API

## Games (25)
| Game | Local | API | Page |
|------|-------|-----|------|
| [lp85](games/lp85.md) | 5/8 | 5/8 | Abstract permutation BFS |
| [tu93](games/tu93.md) | 9/9 | 9/9 | Frame-hash BFS |
| [ft09](games/ft09.md) | 6/6 | 6/6 | GF(p) constraint solver |
| [tn36](games/tn36.md) | 5/7 | 5/7 | Abstract opcode solver |
| [tr87](games/tr87.md) | 4/6 | 1/6 | Rule extraction solver (NEW) |
| [vc33](games/vc33.md) | 3/7 | 3/7 | Grid-scan BFS |
| [dc22](games/dc22.md) | 0/6 | 3/6 | Explorer only |
| [ar25](games/ar25.md) | 2/8 | 2/8 | Dynamic click BFS |
| [m0r0](games/m0r0.md) | 2/6 | 2/6 | Dynamic click BFS |
| [sp80](games/sp80.md) | 2/6 | 2/6 | Dynamic click BFS |
| [cn04](games/cn04.md) | 1/5 | 1/5 | Dynamic click BFS |
| [ka59](games/ka59.md) | 1/7 | 1/7 | Dynamic click BFS |
| [cd82](games/cd82.md) | 1/6 | 1/6 | Keyboard BFS |
| [ls20](games/ls20.md) | 1/7 | 1/7 | Keyboard BFS |
| [sk48](games/sk48.md) | 1/8 | 1/8 | Dynamic click BFS |
| [s5i5](games/s5i5.md) | 1/8 | 1/8 | Grid-scan BFS |
| [su15](games/su15.md) | 0/9 | 1/9 | Explorer only |
| [bp35](games/bp35.md) | 0/9 | 0/9 | Unsolved (timeout) |
| [g50t](games/g50t.md) | 0/7 | 0/7 | Unsolved (deepcopy broken) |
| [lf52](games/lf52.md) | 0/10 | 0/10 | Unsolved (timeout) |
| [r11l](games/r11l.md) | 0/6 | 0/6 | Unsolved (3K click targets) |
| [re86](games/re86.md) | 0/8 | 0/8 | Unsolved (pixel matching) |
| [sb26](games/sb26.md) | 0/8 | 0/8 | Unsolved (tile matching) |
| [sc25](games/sc25.md) | 0/6 | 0/6 | Unsolved (deepcopy broken) |
| [wa30](games/wa30.md) | 0/9 | 0/9 | Unsolved (Sokoban, deep) |

## Techniques
- [Abstract BFS](techniques/abstract-bfs.md) — bypass game engine, BFS on tuples
- [Source Code Analysis](techniques/source-code-analysis.md) — read step(), extract mechanics
- [Dynamic Click Refresh](techniques/dynamic-clicks.md) — per-state _get_valid_clickable_actions()
- [Grid-Scan Clicks](techniques/grid-scan.md) — find effective click positions empirically
- [Deepcopy Bug](techniques/deepcopy-bug.md) — lambda closure issue and workarounds
- [GF(p) Algebra](techniques/gfp-algebra.md) — Lights-Out constraint solving

## Infrastructure
- [Combined Runner](techniques/combined-runner.md) — run_combined.py architecture
- [Training Pipeline](techniques/training-pipeline.md) — 3B model fine-tuning data

## Unattached Add-ons
- [CGE UCB1](techniques/cge-ucb1.md) — proven 29% over BFS in simulation
- [Action-Effect Memory](techniques/action-effect-memory.md) — designed, not implemented
- [Goal Inference](techniques/goal-inference.md) — designed, not implemented
- [FLASH World Model](techniques/flash-model.md) — tested, not integrated
- [CHRYSALIS](techniques/chrysalis.md) — research architecture, not competition-ready
