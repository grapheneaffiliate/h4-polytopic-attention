#!/usr/bin/env python3
"""
E8 Theta Series Decomposition Under H4 Projection

The E8 theta function Theta_E8(q) = 1 + 240q + 2160q^2 + 6720q^3 + ...
counts lattice vectors at each squared norm (using norm^2/2 as index).

Under E8 = H4 + H4', each vector v decomposes as v = v_H4 + v_H4'.
Its norm splits: |v|^2 = |v_H4|^2 + |v_H4'|^2.

But H4 norms live in Q(sqrt(5)), not in Q. So the decomposition
produces a TWO-VARIABLE theta series over the golden ring Z[phi].

This connects E8 modular forms to HILBERT MODULAR FORMS of Q(sqrt(5)).
The explicit decomposition may reveal structure nobody has written down.

Strategy: enumerate E8 lattice vectors at norms 2,4,6,8,10,
project each to H4+H4', record the (norm_H4, norm_H4') pairs
in exact Q(sqrt(5)) arithmetic.
"""

import os
import time
import json
from collections import Counter
from itertools import product as cartprod

os.environ["OMP_NUM_THREADS"] = "2"

print("=" * 65)
print("  E8 THETA SERIES: H4 DECOMPOSITION")
print("  Exact Q(sqrt(5)) arithmetic")
print("=" * 65)


# ── Q(sqrt(5)) arithmetic ────────────────────────────────────────
# Represent a + b*sqrt(5) as (a, b) with a, b rational (we use
# integers * 4 to avoid fractions from the 2x-scaled E8 roots).

class QSqrt5:
    """Exact element of Q(sqrt(5)): a + b*sqrt(5), stored as (a, b)
    with integer numerators (denominator tracked separately)."""
    __slots__ = ('a', 'b')

    def __init__(self, a=0, b=0):
        self.a = a  # rational part (integer)
        self.b = b  # sqrt(5) coefficient (integer)

    def __add__(self, other):
        return QSqrt5(self.a + other.a, self.b + other.b)

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b

    def __hash__(self):
        return hash((self.a, self.b))

    def __repr__(self):
        if self.b == 0:
            return f"{self.a}"
        if self.a == 0:
            return f"{self.b}*sqrt5"
        return f"({self.a}+{self.b}*sqrt5)"

    def approx(self):
        return self.a + self.b * 2.2360679774997896  # sqrt(5)


# ── E8 lattice vector enumeration ─────────────────────────────────

def enumerate_e8_shell(norm_sq):
    """Enumerate all E8 lattice vectors with given squared norm.

    E8 lattice (D8+ form): vectors are either
      Type A: all integers, even sum
      Type B: all half-integers, even sum
    Squared norm = sum of squares.

    We use 2x-scaling: actual coordinates * 2 -> all integers.
    Scaled squared norm = 4 * actual squared norm.
    So norm_sq=2 (actual) -> scaled_norm_sq=8.
    """
    target = norm_sq * 4  # scaled
    vectors = []

    # Type A: coordinates are even integers (0, +-2, +-4, ...)
    # Sum must be even (always true since all even)
    # Sum of squares = target
    _enumerate_type_a([], 8, target, vectors)

    # Type B: coordinates are odd integers (+-1, +-3, +-5, ...)
    # Sum must be even
    _enumerate_type_b([], 8, target, vectors)

    return vectors


def _enumerate_type_a(partial, remaining, target, results):
    """Enumerate 8D vectors with even integer coords, sum of squares = target."""
    if remaining == 0:
        if target == 0:
            # Check even sum (always true for even integers, but verify)
            s = sum(partial)
            if s % 4 == 0:  # in 2x scaling, "even sum" means sum divisible by 4
                results.append(tuple(partial))
        return

    # Maximum absolute value for remaining coordinates
    max_val = int(target ** 0.5)
    # Only even values
    for v in range(0, max_val + 1, 2):
        v_sq = v * v
        if v_sq > target:
            break
        if v == 0:
            _enumerate_type_a(partial + [0], remaining - 1, target, results)
        else:
            for sign in [v, -v]:
                _enumerate_type_a(partial + [sign], remaining - 1, target - v_sq, results)


def _enumerate_type_b(partial, remaining, target, results):
    """Enumerate 8D vectors with odd integer coords, sum of squares = target."""
    if remaining == 0:
        if target == 0:
            s = sum(partial)
            if s % 4 == 0:  # even sum condition in 2x scaling
                results.append(tuple(partial))
        return

    max_val = int(target ** 0.5)
    # Only odd values
    for v in range(1, max_val + 1, 2):
        v_sq = v * v
        if v_sq > target:
            break
        for sign in [v, -v]:
            _enumerate_type_b(partial + [sign], remaining - 1, target - v_sq, results)


# ── H4 projection with exact Q(sqrt(5)) arithmetic ───────────────

def project_exact(vec):
    """Project 8D vector to H4 using exact Q(sqrt(5)) arithmetic.

    Projection: (x1,...,x8) -> (x1 + phi*x5, ..., x4 + phi*x8)
    where phi = (1+sqrt(5))/2.

    In our 2x-scaled coords, xi are integers.
    Result: 4 components, each of form (a + b*sqrt(5)) where
    a = 2*xi + xj  (integer, from 2*xi + (1/2)*2*xj = 2*xi + xj)
    Wait, let me be careful.

    phi = (1+sqrt(5))/2
    Component i: vec[i] + phi * vec[i+4]
               = vec[i] + (1+sqrt(5))/2 * vec[i+4]
               = vec[i] + vec[i+4]/2 + vec[i+4]*sqrt(5)/2

    In 2x-scaled: actual vec[i] = scaled[i]/2
    So: scaled[i]/2 + phi * scaled[i+4]/2
      = scaled[i]/2 + (1+sqrt(5))/2 * scaled[i+4]/2
      = (scaled[i] + scaled[i+4])/4 + scaled[i+4]*sqrt(5)/4

    To stay in integers, multiply everything by 4:
    4 * component_i = (scaled[i] + scaled[i+4]) + scaled[i+4]*sqrt(5)

    So norm in Q(sqrt(5)):
    |4*comp_i|^2 = (a + b*sqrt(5))^2 = a^2 + 5b^2 + 2ab*sqrt(5)
    Total 4^2 * |proj|^2 = sum_i (a_i^2 + 5*b_i^2) + sqrt(5)*sum_i(2*a_i*b_i)
    """
    components = []
    for i in range(4):
        a = vec[i] + vec[i + 4]  # rational part * 4
        b = vec[i + 4]            # sqrt(5) part * 4
        components.append((a, b))

    # Compute |proj|^2 * 16 in Q(sqrt(5))
    rat_part = 0
    sqrt5_part = 0
    for a, b in components:
        rat_part += a * a + 5 * b * b
        sqrt5_part += 2 * a * b

    return QSqrt5(rat_part, sqrt5_part), components


def project_conjugate_exact(vec):
    """Project to H4' using phi_bar = (1-sqrt(5))/2.

    4 * component_i = (scaled[i] + scaled[i+4]) - scaled[i+4]*sqrt(5)
    -> just negate the sqrt(5) coefficient.
    """
    rat_part = 0
    sqrt5_part = 0
    for i in range(4):
        a = vec[i] + vec[i + 4]
        b = -vec[i + 4]  # negated!
        rat_part += a * a + 5 * b * b
        sqrt5_part += 2 * a * b

    return QSqrt5(rat_part, sqrt5_part)


# ── Main computation ──────────────────────────────────────────────

# Known theta series coefficients: vectors at norm^2 = 2n
# a(n) for E8: 1, 240, 2160, 6720, 17520, 30240, 60480, 82560, ...
expected_counts = {
    1: 240,
    2: 2160,
    3: 6720,
    4: 17520,
    5: 30240,
}

all_decompositions = {}

for shell in range(1, 6):
    norm_sq = 2 * shell

    print(f"\n--- Shell {shell}: norm^2 = {norm_sq} ---")
    t0 = time.time()
    vectors = enumerate_e8_shell(norm_sq)
    elapsed = time.time() - t0
    print(f"  Found {len(vectors)} vectors ({elapsed:.1f}s)")

    if shell in expected_counts:
        expected = expected_counts[shell]
        status = "OK" if len(vectors) == expected else f"MISMATCH (expected {expected})"
        print(f"  Expected: {expected} -> {status}")

    if len(vectors) == 0:
        continue

    # Project each vector and record (norm_H4, norm_H4') in Q(sqrt(5))
    decomp = Counter()  # (norm_h4, norm_h4_bar) -> count
    for v in vectors:
        nh4, _ = project_exact(v)
        nh4bar = project_conjugate_exact(v)
        decomp[(nh4, nh4bar)] += 1

    print(f"  Distinct (norm_H4, norm_H4') pairs: {len(decomp)}")
    print(f"  Decomposition:")
    for (nh4, nh4bar), count in sorted(decomp.items(),
            key=lambda x: x[0][0].approx()):
        # The two norms should sum to the original norm * 16
        total = QSqrt5(nh4.a + nh4bar.a, nh4.b + nh4bar.b)
        print(f"    H4={nh4!r:>20s}  H4'={nh4bar!r:>20s}  "
              f"sum={total!r:>15s}  count={count:>5d}")

    all_decompositions[shell] = {
        str((str(k[0]), str(k[1]))): v
        for k, v in decomp.items()
    }

    # Safety: don't run too long
    if time.time() - t0 > 300:  # 5 min max per shell
        print("  Time limit reached, stopping")
        break


# ── Analysis: look for patterns ───────────────────────────────────

print(f"\n\n" + "=" * 65)
print(f"  PATTERN ANALYSIS")
print(f"=" * 65)

print(f"""
The norm decomposition n = n_H4 + n_H4' lives in Q(sqrt(5)).
If the counts at each (n_H4, n_H4') pair follow a pattern
related to Hilbert modular forms of Q(sqrt(5)), that would
connect E8 theta series to the arithmetic of the golden field.

Key question: do the counts factorize? I.e., is
  #{'{'}v : |v_H4|^2=a, |v_H4'|^2=b{'}'} = f(a) * g(b)
for some functions f, g? If yes, Theta_E8 = Theta_H4 * Theta_H4'
as Hilbert modular forms. If no, there's a non-trivial mixing term.
""")

# Save results
out_path = os.path.join(os.path.dirname(__file__), "e8_theta_decomp.json")
with open(out_path, "w") as f:
    json.dump(all_decompositions, f, indent=2)
print(f"Results saved to {out_path}")
