# Activity Log

## 2026-04-04
- TR87 solver built: 4/6 levels via rule extraction from game state
- TR87 levels 4-5 blocked by alter_rules (seed-randomized rules)
- TN36 solve_tn36.py updated with improved button mapping
- Wiki created with compiled knowledge for all 25 games

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
