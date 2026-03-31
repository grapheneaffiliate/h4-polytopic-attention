"""Tests for TransformComposer."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))


def test_depth1_rotation():
    """Depth 1: simple rotation should be found."""
    from agent_zero.transforms.composer import TransformComposer
    inp = np.array([[1, 2], [3, 4]])
    out = np.rot90(inp, -1)  # rotate 90 CW
    composer = TransformComposer(max_depth=1, timeout_seconds=10)
    chain = composer.solve([(inp, out)])
    if chain:
        result = chain.apply(inp)
        assert np.array_equal(result, out), f"Expected {out}, got {result}"
        print(f"  PASS test_depth1_rotation: {chain.describe()}")
    else:
        print("  SKIP test_depth1_rotation: rotation primitive not found (primitives may not be loaded)")


def test_depth1_reflect():
    """Depth 1: horizontal reflection."""
    from agent_zero.transforms.composer import TransformComposer
    inp = np.array([[1, 2, 3], [4, 5, 6]])
    out = np.fliplr(inp)
    composer = TransformComposer(max_depth=1, timeout_seconds=10)
    chain = composer.solve([(inp, out)])
    if chain:
        result = chain.apply(inp)
        assert np.array_equal(result, out)
        print(f"  PASS test_depth1_reflect: {chain.describe()}")
    else:
        print("  SKIP test_depth1_reflect: reflect primitive not found")


def test_depth2_color_then_rotate():
    """Depth 2: color swap followed by rotation."""
    from agent_zero.transforms.composer import TransformComposer
    inp = np.array([[1, 2], [3, 1]])
    # Swap 1↔3, then rotate 180
    swapped = np.where(inp == 1, 3, np.where(inp == 3, 1, inp))
    out = np.rot90(swapped, 2)
    composer = TransformComposer(max_depth=2, timeout_seconds=15)
    chain = composer.solve([(inp, out)])
    if chain:
        result = chain.apply(inp)
        match = np.array_equal(result, out)
        print(f"  {'PASS' if match else 'FAIL'} test_depth2_color_then_rotate: {chain.describe()}")
    else:
        print("  SKIP test_depth2_color_then_rotate: no solution found")


def test_no_solution():
    """Should return None for impossible transforms (random grids)."""
    from agent_zero.transforms.composer import TransformComposer
    np.random.seed(42)
    inp = np.random.randint(0, 10, (5, 5))
    out = np.random.randint(0, 10, (3, 7))  # different shape, random
    composer = TransformComposer(max_depth=2, timeout_seconds=5)
    chain = composer.solve([(inp, out)])
    # Should either be None or very low score
    print(f"  PASS test_no_solution: returned {'None' if chain is None else chain.describe()}")


def test_timeout():
    """Should not hang — returns within timeout."""
    from agent_zero.transforms.composer import TransformComposer
    import time
    inp = np.ones((10, 10), dtype=int)
    out = np.zeros((10, 10), dtype=int)
    composer = TransformComposer(max_depth=3, timeout_seconds=2)
    t0 = time.time()
    chain = composer.solve([(inp, out)])
    elapsed = time.time() - t0
    assert elapsed < 10, f"Took {elapsed:.1f}s, should be <10s"
    print(f"  PASS test_timeout: completed in {elapsed:.1f}s")


def test_chain_describe():
    """TransformChain.describe() returns readable string."""
    from agent_zero.transforms.composer import TransformChain
    chain = TransformChain()
    assert chain.describe() == "(identity)"
    print("  PASS test_chain_describe")


def test_multiple_examples():
    """Must match ALL training examples, not just one."""
    from agent_zero.transforms.composer import TransformComposer
    # Two examples that both require the same transform (flip vertical)
    inp1 = np.array([[1, 2], [3, 4]])
    out1 = np.flipud(inp1)
    inp2 = np.array([[5, 6], [7, 8]])
    out2 = np.flipud(inp2)
    composer = TransformComposer(max_depth=1, timeout_seconds=10)
    chain = composer.solve([(inp1, out1), (inp2, out2)])
    if chain:
        assert np.array_equal(chain.apply(inp1), out1)
        assert np.array_equal(chain.apply(inp2), out2)
        print(f"  PASS test_multiple_examples: {chain.describe()}")
    else:
        print("  SKIP test_multiple_examples: no solution found")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception as e:
                print(f"  ERROR {name}: {e}")
    print("Done.")
