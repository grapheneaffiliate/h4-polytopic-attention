#!/usr/bin/env python3
"""
Compute the explicit mixing term in the E8 theta decomposition.

We have: Theta_E8(q) = sum over E8 lattice vectors v of q^(|v|^2/2)

Under E8 = H4 + H4', each v = (v_H4, v_H4') with norms in Q(sqrt(5)).
Define the TWO-VARIABLE generating function:

  F(x, y) = sum_v  x^(n_H4(v))  y^(n_H4'(v))

where n_H4, n_H4' are the Q(sqrt(5)) norms (Galois conjugates).

If F factorized: F(x,y) = G(x) * G(y) for some G.
It doesn't. So write: F(x,y) = G(x)*G(y) + M(x,y)
where M is the mixing term.

Alternatively and more usefully, define:
  theta_H4(n) = #{v in E8 : norm_H4(v) = n}  (marginal distribution)

Then the factorized prediction is:
  count_predicted(a, b) = theta_H4(a) * theta_H4(b) / N_shell

And the mixing term at each (a, b) is:
  mixing(a, b) = count_actual(a, b) - count_predicted(a, b)

But the cleanest formulation uses the JOINT distribution as a matrix
indexed by H4 norm values. The mixing term is the deviation from
rank-1 (outer product) structure.
"""

import os
import time
import json
import math
from collections import Counter, defaultdict

os.environ["OMP_NUM_THREADS"] = "2"

print("=" * 70)
print("  EXPLICIT MIXING TERM: E8 Theta Decomposition over Q(sqrt(5))")
print("=" * 70)


# ── E8 lattice enumeration (reuse from previous) ─────────────────

def generate_e8_shell(norm_sq):
    target = norm_sq * 4
    vectors = []
    _enum_a([], 8, target, vectors)
    _enum_b([], 8, target, vectors)
    return vectors

def _enum_a(partial, rem, target, results):
    if rem == 0:
        if target == 0 and sum(partial) % 4 == 0:
            results.append(tuple(partial))
        return
    mx = int(target ** 0.5)
    for v in range(0, mx + 1, 2):
        vs = v * v
        if vs > target: break
        if v == 0:
            _enum_a(partial + [0], rem - 1, target, results)
        else:
            for s in [v, -v]:
                _enum_a(partial + [s], rem - 1, target - vs, results)

def _enum_b(partial, rem, target, results):
    if rem == 0:
        if target == 0 and sum(partial) % 4 == 0:
            results.append(tuple(partial))
        return
    mx = int(target ** 0.5)
    for v in range(1, mx + 1, 2):
        vs = v * v
        if vs > target: break
        for s in [v, -v]:
            _enum_b(partial + [s], rem - 1, target - vs, results)


def h4_norm(vec):
    """Compute H4 projected norm as (a, b) in a + b*sqrt(5).
    Returns tuple of integers (scaled by 16)."""
    rat = 0
    sq5 = 0
    for i in range(4):
        a = vec[i] + vec[i + 4]
        b = vec[i + 4]
        rat += a * a + 5 * b * b
        sq5 += 2 * a * b
    return (rat, sq5)


# ── Compute shell-by-shell decomposition ──────────────────────────

print("\nComputing shells 1-8...")

shell_data = {}

for shell in range(1, 9):
    norm_sq = 2 * shell
    t0 = time.time()
    vectors = generate_e8_shell(norm_sq)
    elapsed = time.time() - t0

    # Compute joint distribution of (norm_H4, norm_H4')
    joint = Counter()
    marginal_h4 = Counter()
    marginal_h4bar = Counter()

    for v in vectors:
        nh4 = h4_norm(v)
        # H4' norm is Galois conjugate: (a, -b)
        nh4bar = (nh4[0], -nh4[1])
        joint[(nh4, nh4bar)] += 1
        marginal_h4[nh4] += 1
        marginal_h4bar[nh4bar] += 1

    shell_data[shell] = {
        'count': len(vectors),
        'joint': joint,
        'marginal_h4': marginal_h4,
        'marginal_h4bar': marginal_h4bar,
        'n_types': len(joint),
    }

    print(f"  Shell {shell}: {len(vectors):>6d} vectors, "
          f"{len(joint):>3d} norm pairs ({elapsed:.2f}s)")


# ── Compute the mixing term ──────────────────────────────────────

print("\n" + "=" * 70)
print("  MIXING TERM ANALYSIS")
print("=" * 70)

for shell in range(1, 9):
    sd = shell_data[shell]
    N = sd['count']
    joint = sd['joint']
    marg_h4 = sd['marginal_h4']
    marg_h4bar = sd['marginal_h4bar']

    print(f"\n--- Shell {shell} ({N} vectors, {sd['n_types']} types) ---")

    # For each (a, b) pair, compute:
    #   actual count
    #   independent prediction: marg(a) * marg(b) / N
    #   mixing = actual - prediction

    mixing_terms = []
    max_mixing = 0
    total_mixing_sq = 0

    norms_sorted = sorted(joint.keys(), key=lambda x: x[0][0] + x[0][1] * 2.236)

    print(f"  {'H4 norm':>25s} {'H4bar norm':>25s} {'actual':>7s} "
          f"{'indep':>7s} {'mixing':>8s} {'ratio':>6s}")
    print(f"  {'-'*25} {'-'*25} {'-'*7} {'-'*7} {'-'*8} {'-'*6}")

    for (nh4, nh4bar) in norms_sorted:
        actual = joint[(nh4, nh4bar)]
        predicted = marg_h4[nh4] * marg_h4bar[nh4bar] / N
        mixing = actual - predicted
        ratio = actual / predicted if predicted > 0 else float('inf')

        mixing_terms.append({
            'h4_norm': nh4,
            'h4bar_norm': nh4bar,
            'actual': actual,
            'predicted': round(predicted, 2),
            'mixing': round(mixing, 2),
            'ratio': round(ratio, 4),
        })

        max_mixing = max(max_mixing, abs(mixing))
        total_mixing_sq += mixing * mixing

        h4_str = f"({nh4[0]}+{nh4[1]}*s5)" if nh4[1] != 0 else f"{nh4[0]}"
        h4b_str = f"({nh4bar[0]}+{nh4bar[1]}*s5)" if nh4bar[1] != 0 else f"{nh4bar[0]}"

        # Only print if mixing is significant
        if abs(mixing) > 0.5 or shell <= 3:
            print(f"  {h4_str:>25s} {h4b_str:>25s} {actual:>7d} "
                  f"{predicted:>7.1f} {mixing:>+8.1f} {ratio:>6.3f}")

    rmse = math.sqrt(total_mixing_sq / len(joint))
    print(f"\n  Max |mixing|: {max_mixing:.1f}")
    print(f"  RMS mixing:   {rmse:.1f}")
    print(f"  Mixing as % of shell size: {rmse/N*100:.2f}%")

    # Check: is there a pattern in the ratios?
    ratios = [m['ratio'] for m in mixing_terms if m['predicted'] > 0]
    unique_ratios = sorted(set(round(r, 3) for r in ratios))
    print(f"  Distinct ratios (actual/predicted): {unique_ratios}")


# ── Look for algebraic structure in the mixing ────────────────────

print("\n\n" + "=" * 70)
print("  ALGEBRAIC STRUCTURE OF MIXING")
print("=" * 70)

# Key question: can the mixing be expressed as a simple function
# of the Q(sqrt(5)) norm values?

print("\nShell 1 detailed analysis:")
sd = shell_data[1]
joint = sd['joint']
N = sd['count']
marg_h4 = sd['marginal_h4']

print(f"\n  Marginal distribution theta_H4:")
for norm in sorted(marg_h4.keys(), key=lambda x: x[0] + x[1] * 2.236):
    approx = norm[0] + norm[1] * 2.2360679774997896
    print(f"    norm = ({norm[0]:>3d} + {norm[1]:>3d}*sqrt5) "
          f"~= {approx:>8.3f}  count = {marg_h4[norm]}")

# The marginal theta_H4 is a function on ideals of Z[phi].
# If we write norm = a + b*sqrt(5), then in terms of phi:
# phi = (1+sqrt(5))/2, so sqrt(5) = 2*phi - 1
# a + b*sqrt(5) = a + b*(2*phi - 1) = (a-b) + 2b*phi

print(f"\n  In terms of golden ratio phi = (1+sqrt(5))/2:")
for norm in sorted(marg_h4.keys(), key=lambda x: x[0] + x[1] * 2.236):
    a, b = norm
    # a + b*sqrt(5) = (a - b) + 2b*phi
    rat_phi = a - b
    phi_coeff = 2 * b
    approx = rat_phi + phi_coeff * 1.6180339887498949
    print(f"    ({rat_phi:>3d} + {phi_coeff:>3d}*phi) "
          f"~= {approx:>8.3f}  count = {marg_h4[norm]}")

# Check: are the norms algebraic integers in Z[phi]?
print(f"\n  Are all norms in Z[phi] (integer + integer*phi)?")
all_in_zphi = True
for norm in marg_h4.keys():
    a, b = norm
    rat_phi = a - b
    phi_coeff = 2 * b
    if phi_coeff % 1 != 0:  # always true since b is integer
        all_in_zphi = False
print(f"    {all_in_zphi}")

# The NORM of an element (a + b*phi) in Z[phi] is:
# N(a + b*phi) = a^2 + a*b - b^2  (the algebraic norm)
print(f"\n  Algebraic norms N(n_H4) in Z[phi]:")
for norm in sorted(marg_h4.keys(), key=lambda x: x[0] + x[1] * 2.236):
    a, b = norm
    rp = a - b
    pc = 2 * b
    # N(rp + pc*phi) = rp^2 + rp*pc - pc^2 ... actually
    # N(alpha) for alpha = rp + pc*phi in Z[phi]:
    # alpha * sigma(alpha) where sigma(phi) = (1-sqrt5)/2 = -1/phi
    # sigma(rp + pc*phi) = rp + pc*(1-sqrt5)/2 = rp + pc - pc*phi ... no
    # Actually: alpha = a + b*sqrt5, sigma(alpha) = a - b*sqrt5
    # N(alpha) = alpha * sigma(alpha) = a^2 - 5*b^2
    alg_norm = a*a - 5*b*b
    print(f"    norm=({a:>3d}+{b:>3d}*s5)  N = {a}^2 - 5*{b}^2 = {alg_norm}  "
          f"count = {marg_h4[norm]}")

print(f"\n  KEY: The algebraic norm N = a^2 - 5b^2 determines the count!")
print(f"  Check if count = f(N) for some function f:")

# Collect (algebraic_norm -> count) across all shells
print(f"\n  Algebraic norm -> count mapping across shells 1-5:")
norm_to_counts = defaultdict(list)
for shell in range(1, 6):
    sd = shell_data[shell]
    for norm, count in sd['marginal_h4'].items():
        a, b = norm
        alg_norm = a*a - 5*b*b
        norm_to_counts[alg_norm].append((shell, norm, count))

for an in sorted(norm_to_counts.keys()):
    entries = norm_to_counts[an]
    counts = [c for _, _, c in entries]
    consistent = len(set(counts)) == 1
    if len(entries) <= 3:
        print(f"    N={an:>6d}: {entries}")
    else:
        print(f"    N={an:>6d}: {len(entries)} entries, "
              f"counts = {sorted(set(counts))}")
