"""
Exact computation engine for the math specialist.

Detects arithmetic expressions in queries, computes them exactly,
and returns results that the language model can incorporate into
its response. This is the bridge between language reasoning (H4 4D)
and exact computation (inspired by Percepta's 2D execution path).

Current implementation: Python-based exact computation.
Future: compile arithmetic into 2D attention heads (Percepta method).

The key insight: the model doesn't need to be good at arithmetic.
It needs to be good at RECOGNIZING arithmetic and DELEGATING to
exact computation. That's a much easier task.
"""

import re
import math
import operator
from typing import Optional, Tuple, List


# Supported operations
OPS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '×': operator.mul,
    'x': operator.mul,  # lowercase x as multiply
    '/': operator.truediv,
    '÷': operator.truediv,
    '^': operator.pow,
    '**': operator.pow,
    '%': operator.mod,
}


def detect_arithmetic(text: str) -> List[dict]:
    """
    Detect arithmetic expressions in text.

    Returns list of {expression, start, end} for each detected expression.
    Handles: 15 * 23, 100/4, 2^10, 15% of 200, sqrt(16), etc.
    """
    expressions = []

    # Pattern: number op number (possibly chained: 1 + 2 + 3)
    arith_pattern = r'(\d+(?:\.\d+)?)\s*([\+\-\*\/\^×÷%]|(?:\*\*))\s*(\d+(?:\.\d+)?)'
    for m in re.finditer(arith_pattern, text):
        expressions.append({
            'expression': m.group(0),
            'type': 'binary',
            'a': float(m.group(1)),
            'op': m.group(2),
            'b': float(m.group(3)),
            'start': m.start(),
            'end': m.end(),
        })

    # Pattern: X% of Y
    pct_pattern = r'(\d+(?:\.\d+)?)\s*%\s*(?:of)\s*(\d+(?:\.\d+)?)'
    for m in re.finditer(pct_pattern, text, re.IGNORECASE):
        expressions.append({
            'expression': m.group(0),
            'type': 'percentage',
            'a': float(m.group(1)),
            'b': float(m.group(2)),
            'start': m.start(),
            'end': m.end(),
        })

    # Pattern: sqrt(X) or square root of X
    sqrt_pattern = r'(?:sqrt|square\s+root\s+of)\s*\(?\s*(\d+(?:\.\d+)?)\s*\)?'
    for m in re.finditer(sqrt_pattern, text, re.IGNORECASE):
        expressions.append({
            'expression': m.group(0),
            'type': 'sqrt',
            'a': float(m.group(1)),
            'start': m.start(),
            'end': m.end(),
        })

    # Pattern: factorial (X! or factorial of X)
    fact_pattern = r'(\d+)\s*!'
    for m in re.finditer(fact_pattern, text):
        expressions.append({
            'expression': m.group(0),
            'type': 'factorial',
            'a': int(m.group(1)),
            'start': m.start(),
            'end': m.end(),
        })

    return expressions


def compute(expr: dict) -> Optional[float]:
    """Compute a detected expression exactly."""
    try:
        if expr['type'] == 'binary':
            op_func = OPS.get(expr['op'])
            if op_func is None:
                return None
            result = op_func(expr['a'], expr['b'])
            # Return int if result is whole number
            if isinstance(result, float) and result == int(result):
                return int(result)
            return result

        elif expr['type'] == 'percentage':
            return expr['a'] / 100 * expr['b']

        elif expr['type'] == 'sqrt':
            return math.sqrt(expr['a'])

        elif expr['type'] == 'factorial':
            if expr['a'] > 170:  # overflow protection
                return None
            return math.factorial(int(expr['a']))

    except (ZeroDivisionError, OverflowError, ValueError):
        return None


def compute_all(text: str) -> List[Tuple[str, str]]:
    """
    Find and compute all arithmetic expressions in text.

    Returns list of (expression_string, result_string) tuples.
    """
    expressions = detect_arithmetic(text)
    results = []
    for expr in expressions:
        result = compute(expr)
        if result is not None:
            results.append((expr['expression'], str(result)))
    return results


def augment_query(query: str) -> str:
    """
    Augment a query with computed results.

    If the query contains arithmetic, compute it and append the results
    so the language model can reference exact values.

    Example:
        "What is 15 * 23?" → "What is 15 * 23? [COMPUTE: 15 * 23 = 345]"
    """
    results = compute_all(query)
    if not results:
        return query

    compute_block = " ".join(f"[COMPUTE: {expr} = {val}]" for expr, val in results)
    return f"{query} {compute_block}"


def test():
    """Test the computation engine."""
    test_cases = [
        ("What is 15 * 23?", [("15 * 23", "345")]),
        ("Calculate 100 / 4", [("100 / 4", "25")]),
        ("What is 25% of 200?", [("25% of 200", "50.0")]),
        ("Compute 2^10", [("2^10", "1024")]),
        ("What is 7!", [("7!", "5040")]),
        ("Add 123 + 456", [("123 + 456", "579")]),
        ("15 * 23 + 10", [("15 * 23", "345"), ("23 + 10", "33")]),
        ("No math here", []),
        ("The year 2026 was interesting", []),
    ]

    print("Compute Engine Tests:")
    print("-" * 60)
    passed = 0
    for query, expected in test_cases:
        results = compute_all(query)
        result_tuples = [(e, v) for e, v in results]
        match = result_tuples == expected
        passed += match
        status = "OK" if match else "FAIL"
        print(f"  {status}: {query}")
        if not match:
            print(f"    Expected: {expected}")
            print(f"    Got:      {result_tuples}")

    print(f"\n{passed}/{len(test_cases)} passed")

    # Demo augmented queries
    print("\nAugmented queries:")
    for query in ["What is 15 * 23?", "If I have 500 - 123 dollars", "Calculate sqrt(144)"]:
        print(f"  In:  {query}")
        print(f"  Out: {augment_query(query)}")


if __name__ == '__main__':
    test()
