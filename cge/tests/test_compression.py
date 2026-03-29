"""Tests for CompressionLayer."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from cge.core import GraphExplorer
from cge.compression import CompressionLayer


def _build_test_graph():
    """Build a small graph with known properties."""
    g = GraphExplorer()
    g.add_node("A", {0, 1, 2, 3})
    # Action 0 and 1 change state, 2 and 3 don't
    g.record_transition("A", 0, True, target="B", target_actions={0, 1, 2})
    g.record_transition("A", 1, True, target="C", target_actions={0, 1, 2})
    g.record_transition("A", 2, False)
    g.record_transition("A", 3, False)
    # In B: action 0 changes, 1 doesn't
    g.record_transition("B", 0, True, target="D", target_actions={0, 1})
    g.record_transition("B", 1, False)
    return g


def test_analyze_builds_signatures():
    g = _build_test_graph()
    c = CompressionLayer()
    c.analyze(g)
    assert "A" in c.state_sigs
    assert "B" in c.state_sigs
    # A: 4 tested, 2 changed -> change_rate = 0.5
    assert abs(c.state_sigs["A"].change_rate - 0.5) < 0.01


def test_action_ranking():
    g = _build_test_graph()
    c = CompressionLayer()
    c.analyze(g)
    # Actions 0 and 1 succeed more often than 2 and 3
    ranking = c.rank_actions("A", {0, 1, 2, 3})
    # 0 succeeds in A and B (2/2), 1 succeeds in A but not B (1/2)
    assert ranking[0] == 0  # highest efficacy


def test_bottleneck_detection():
    g = GraphExplorer()
    g.add_node("start", {0, 1})
    g.record_transition("start", 0, True, target="dead1", target_actions={0})
    g.record_transition("start", 1, True, target="bottle", target_actions={0})
    g.record_transition("dead1", 0, False)
    # bottle has exactly 1 novel successor
    g.record_transition("bottle", 0, True, target="goal", target_actions={0})
    c = CompressionLayer()
    c.analyze(g)
    assert "bottle" in c.bottlenecks


def test_winning_path_recording():
    g = _build_test_graph()
    c = CompressionLayer()
    c.analyze(g)
    c.record_win(["A", "B", "D"], [0, 0])
    assert len(c.winning_paths) == 1
    assert len(c.winning_sigs) == 1


def test_progress_direction():
    g = _build_test_graph()
    c = CompressionLayer()
    c.analyze(g)
    c.record_win(["A", "B", "D"], [0, 0])
    c.analyze(g)  # re-analyze with winning data
    assert c._progress_direction is not None
    assert "typical_win_depth" in c._progress_direction


def test_environment_classification():
    g = _build_test_graph()
    c = CompressionLayer()
    c.analyze(g)
    env = c.classify_environment()
    assert env["n_states"] == 4  # A, B, C, D
    assert env["n_effective_actions"] > 0


def test_state_scoring():
    g = _build_test_graph()
    c = CompressionLayer()
    c.analyze(g)
    # Deeper states should score higher
    score_a = c.score_state("A")  # depth 0
    score_d = c.score_state("D")  # depth 2
    assert score_d > score_a


def test_summary():
    g = _build_test_graph()
    c = CompressionLayer()
    c.analyze(g)
    s = c.get_summary()
    assert "CompressionLayer" in s
    assert "States:" in s


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"  PASS {name}")
            except AssertionError as e:
                print(f"  FAIL {name}: {e}")
            except Exception as e:
                print(f"  ERROR {name}: {e}")
    print("Done.")
