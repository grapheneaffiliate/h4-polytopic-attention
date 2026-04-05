# Activity Log

## 2026-04-04 (CHRYSALIS + solver session)
- TR87 solver built (`scripts/solve_tr87.py`): 4/6 levels solved
  - L0: simple rule matching (14 actions)
  - L1: multi-output rules (25 actions)
  - L2: multi-input rules (21 actions)
  - L3: double_translation fixed (21 actions)
  - L4-5: alter_rules mode blocks — cursor iterates flattened rule sprites, not bottom row
- TR87 source fully decoded: constants, sprite tags, level flags, randomization seeds
- TN36 button-to-action mapping discovered: action i=btn[i][0], action i+n+1=btn[i][1], action n=play
- TN36 abstract solver now outputs click actions (was setting buttons directly via internal API)
- WA30 source fully analyzed: carry mechanic, blocked set management, per-level data extracted
- R11L partially analyzed: click-only leg positioning, 60 action limit, collision detection
- All 10 unsolved games ranked by ROI (TR87 > WA30 > SU15 > R11L > SB26 > G50T > RE86 > SC25 > BP35 > LF52)
- Wiki pages updated with detailed decoded mechanics for TR87 and WA30
- BitNet 2.4B model running locally at 15 tok/s via bitnet.cpp (separate work)
- CHRYSALIS framework built (GeometricMoE, H4 attention, ternary engine) — research, not ARC-relevant

## 2026-04-04 (earlier)
- **LP85 5/8 breakthrough**: dict key collision fix in check_win
- Combined runner confirmed 45/163 (27.6%) on live API
- LP85 explorer crashes all runners — precomputed only
- Training data pipeline: 511 pairs for 3B model

## 2026-04-03
- Combined runner built and debugged
- ACTION1 init bug found and fixed (+4 levels: CN04, SK48, KA59, AR25)
- Force_explorer for LP85 (subprocess, then in-process — both failed)
- Multiple workflow runs debugging LP85 hang

## 2026-04-02
- TN36 7/7 abstract opcode solver (5/7 with action format for API)
- TU93 extended to 9/9 with bigger BFS budget
- FT09 extended to 6/6 with GF(3) + NTi fix from other session
- VC33 3/7, M0R0 2/6, SK48 1/8, S5I5 1/8 via grid-scan and dynamic clicks
- Deepcopy closure bug discovered (affects G50T, SC25, TN36)
- All 11 unsolved games analyzed
