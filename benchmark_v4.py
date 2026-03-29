"""Benchmark v4 unified explorer on all 25 ARC-AGI-3 games at 200K actions."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from olympus.arc3.explorer_v4 import run_all

API_KEY = os.environ.get("ARC_API_KEY", "58b421be-5980-4ee8-8e57-0f18dc9369f3")

if __name__ == "__main__":
    max_actions = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    print(f"Running v4 unified explorer at {max_actions} actions per game...")
    run_all(API_KEY, max_actions=max_actions, verbose=False)
