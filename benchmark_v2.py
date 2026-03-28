"""Benchmark v2 explorer on all 25 ARC-AGI-3 games at 50K actions."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from olympus.arc3.explorer_v2 import solve_game_v2, run_all_v2

API_KEY = os.environ.get("ARC_API_KEY", "58b421be-5980-4ee8-8e57-0f18dc9369f3")

if __name__ == "__main__":
    run_all_v2(API_KEY, max_actions=100000, verbose=False)
