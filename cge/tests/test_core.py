"""Tests for GraphExplorer."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from cge.core import GraphExplorer


def test_add_node():
    g = GraphExplorer()
    g.add_node("A", {0, 1, 2})
    assert g.num_states == 1
    assert "A" in g.nodes
    assert g.nodes["A"].untested == {0, 1, 2}
    assert g.root == "A"


def test_basic_exploration():
    g = GraphExplorer()
    g.add_node("A", {0, 1})
    # Should return an untested action
    action = g.choose_action("A")
    assert action in {0, 1}


def test_transition_creates_node():
    g = GraphExplorer()
    g.add_node("A", {0, 1})
    g.record_transition("A", 0, True, target="B", target_actions={0, 1})
    assert g.num_states == 2
    assert "B" in g.nodes
    assert g.nodes["A"].tested[0] == (True, "B")


def test_no_change_transition():
    g = GraphExplorer()
    g.add_node("A", {0, 1})
    g.record_transition("A", 0, False)
    assert g.num_states == 1
    assert g.nodes["A"].tested[0] == (False, None)
    assert g.nodes["A"].untested == {1}


def test_navigation_to_frontier():
    g = GraphExplorer()
    g.add_node("A", {0, 1})
    g.record_transition("A", 0, True, target="B", target_actions={0, 1})
    g.record_transition("A", 1, False)
    # A is closed (all tested), B has untested — should navigate A->B
    action = g.choose_action("A")
    assert action == 0  # the edge from A to B uses action 0


def test_deep_navigation():
    g = GraphExplorer()
    g.add_node("A", {0})
    g.record_transition("A", 0, True, target="B", target_actions={0})
    g.record_transition("B", 0, True, target="C", target_actions={0, 1})
    # A and B are closed, C has untested
    # From A, should navigate: A->B->C
    action = g.choose_action("A")
    assert action == 0


def test_reset():
    g = GraphExplorer()
    g.add_node("A", {0})
    g.record_transition("A", 0, True, target="B", target_actions={0})
    g.reset()
    assert g.num_states == 0
    assert g.root is None


def test_action_order_respected():
    g = GraphExplorer()
    g.add_node("A", {0, 1, 2, 3, 4})
    # With action_order, should pick the first available
    action = g.choose_action("A", action_order=[3, 1, 0, 2, 4])
    assert action == 3


def test_node_change_rate():
    g = GraphExplorer()
    g.add_node("A", {0, 1, 2, 3})
    g.record_transition("A", 0, True, target="B", target_actions={0})
    g.record_transition("A", 1, False)
    g.record_transition("A", 2, True, target="C", target_actions={0})
    # 3 tested, 2 changed -> change_rate = 2/3
    assert abs(g.nodes["A"].change_rate - 2/3) < 0.01


def test_max_depth():
    g = GraphExplorer()
    g.add_node("A", {0})
    g.record_transition("A", 0, True, target="B", target_actions={0})
    g.record_transition("B", 0, True, target="C", target_actions={0})
    assert g.max_depth == 2
    assert g.nodes["C"].depth == 2


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
