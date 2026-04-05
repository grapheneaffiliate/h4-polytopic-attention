# ARC-AGI-3 Overview

## Current Score: 45/163 (27.6%) on live API
## Expected Next Run: 48+ (with TR87 4/6)

## Strategy
Pre-computed solutions (instant replay) + explorer v6 fallback. Precomputed covers 41 levels across 15 games. Explorer adds 4 levels from 3 games.

## Biggest Gains Available
| Action | Levels | Effort |
|--------|--------|--------|
| Deploy TR87 solver to API | +3 | Low (already built) |
| Fix TN36 levels 5-6 (wall pathfinding) | +2 | Medium |
| Extend LP85 levels 5-7 (bigger BFS) | +3 | High (11M+ states) |
| Build WA30 Sokoban solver | +1-3 | High |
| Build R11L click-drag solver | +1-3 | Medium |
| Build SB26 CSP solver | +1-3 | Medium |
| Train 3B model (all games) | +10-30 | High (RunPod) |

## What's NOT Working
- Generic BFS caps at ~400 states/s with deepcopy — can't reach depth 15+
- Random walk (1000 episodes) found 0 solutions on any unsolved game
- IDDFS with reset+replay: 80 nodes/s — too slow for baselines >20
- Explorer v6 hangs on LP85, BP35, LF52 (timeout or crash)
- CHRYSALIS phi-annealed self-play loses to UCB1 20/20 vs 1/20

## What IS Working
- Abstract solvers: 100-1000x faster than engine-based BFS
- Per-state dynamic click refresh: solved 5 games
- Grid-scan click discovery: solved 2 games
- Pre-computed replay: 41 levels in <10 seconds total
- LP85 dict collision fix: +5 levels from a one-line change
