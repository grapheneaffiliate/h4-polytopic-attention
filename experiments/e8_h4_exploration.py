#!/usr/bin/env python3
"""
E8 -> H4 Algebraic Structure Exploration

Question: When E8 root system triples (α + β = γ) are projected onto the
H4 subspace, which triples survive as valid 600-cell relationships, and
which break? Is there a pattern?

This is exact integer/rational arithmetic. No floating point. No GPU.

If we find structure here that nobody has catalogued, that's a publishable
result. If the counts match known integer sequences in OEIS, that reveals
unexpected connections.
"""

import numpy as np
from itertools import combinations
from collections import Counter
import time
import json

# Limit CPU
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

print("=" * 65)
print("  E8 -> H4 ALGEBRAIC STRUCTURE EXPLORATION")
print("  Hunting for unknown structure in the projection")
print("=" * 65)


# ── Step 1: Generate all 240 E8 root vectors ─────────────────────

def generate_e8_roots():
    """Generate all 240 roots of E8.

    Type 1: All permutations of (±1, ±1, 0, 0, 0, 0, 0, 0) — 112 roots
    Type 2: (±1/2, ±1/2, ..., ±1/2) with even number of minus signs — 128 roots

    We use 2x scaling to stay in integers: multiply everything by 2.
    So Type 1 becomes (±2, ±2, 0, 0, 0, 0, 0, 0)
    And Type 2 becomes (±1, ±1, ..., ±1) with even minus count.
    """
    roots = []

    # Type 1: pick 2 positions out of 8, assign ±2
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [2, -2]:
                for sj in [2, -2]:
                    v = [0] * 8
                    v[i] = si
                    v[j] = sj
                    roots.append(tuple(v))

    # Type 2: all (±1)^8 with even number of -1s
    for mask in range(256):
        v = []
        neg_count = 0
        for bit in range(8):
            if mask & (1 << bit):
                v.append(-1)
                neg_count += 1
            else:
                v.append(1)
        if neg_count % 2 == 0:
            roots.append(tuple(v))

    return roots


t0 = time.time()
roots = generate_e8_roots()
print(f"\nStep 1: Generated {len(roots)} E8 roots ({time.time()-t0:.3f}s)")
assert len(roots) == 240, f"Expected 240, got {len(roots)}"


# ── Step 2: Compute inner product structure ──────────────────────

def inner_product(a, b):
    """Integer inner product (scaled by 4 due to 2x scaling)."""
    return sum(x * y for x, y in zip(a, b))


t0 = time.time()
# Compute all pairwise inner products
ip_counts = Counter()
for i in range(len(roots)):
    for j in range(i + 1, len(roots)):
        ip = inner_product(roots[i], roots[j])
        ip_counts[ip] += 1

print(f"\nStep 2: Inner product distribution ({time.time()-t0:.3f}s)")
print(f"  (Remember: scaled by 4, so ip=4 means actual ip=1)")
for ip in sorted(ip_counts.keys()):
    actual_ip = ip / 4.0
    print(f"  ip={ip:3d} (actual {actual_ip:+5.2f}): {ip_counts[ip]:5d} pairs")


# ── Step 3: Find all root triples (α + β = γ) ───────────────────

t0 = time.time()
root_set = set(roots)
triples = []  # (i, j, k) where roots[i] + roots[j] = roots[k]

root_to_idx = {r: i for i, r in enumerate(roots)}

for i in range(len(roots)):
    for j in range(i + 1, len(roots)):
        s = tuple(a + b for a, b in zip(roots[i], roots[j]))
        if s in root_set:
            k = root_to_idx[s]
            triples.append((i, j, k))

print(f"\nStep 3: Found {len(triples)} root addition triples ({time.time()-t0:.3f}s)")
print(f"  (α + β = γ where all three are E8 roots)")

# Categorize triples by the inner product of α and β
triple_by_ip = Counter()
for i, j, k in triples:
    ip = inner_product(roots[i], roots[j])
    triple_by_ip[ip] += 1

print(f"\n  Triples by <α,β>:")
for ip in sorted(triple_by_ip.keys()):
    print(f"    <α,β>={ip:3d} (actual {ip/4:+.2f}): {triple_by_ip[ip]} triples")


# ── Step 4: H4 projection ───────────────────────────────────────

def build_h4_projection():
    """Build the 8D -> 4D projection matrix for E8 -> H4.

    The H4 subspace is defined by the golden ratio:
    φ = (1+√5)/2. The projection uses the eigenspaces of
    the E8 Coxeter element.

    We use the standard projection where the first 4 coordinates
    capture the H4 structure.

    For exact arithmetic, we work with (a + b*√5) representation.
    """
    # Standard projection: E8 decomposes as H4 ⊕ H4' under the
    # Coxeter element. The projection picks out the H4 part.
    #
    # For now, use the simple projection that maps:
    #   (x1,x2,x3,x4,x5,x6,x7,x8) -> (x1+φ*x5, x2+φ*x6, x3+φ*x7, x4+φ*x8)
    # where φ = (1+√5)/2
    #
    # This maps E8 roots to 600-cell vertices (up to scaling).
    phi = (1 + np.sqrt(5)) / 2
    return phi


phi = build_h4_projection()

def project_to_h4(root):
    """Project an E8 root to 4D H4 space.

    Returns (a1+φ*a5, a2+φ*a6, a3+φ*a7, a4+φ*a8) as a tuple.
    For exact arithmetic, we return (rational_part, phi_part) pairs.
    """
    # In our 2x-scaled coordinates:
    # The projection is (root[0] + φ*root[4], ..., root[3] + φ*root[7])
    return tuple(
        (root[i], root[i + 4])  # (rational_part, phi_coefficient)
        for i in range(4)
    )


def h4_inner_product(a, b):
    """Inner product in H4 using exact Q(√5) arithmetic.

    a and b are each 4 tuples of (rational, phi_coeff).
    <a,b> = Σ (a_r + a_φ*φ)(b_r + b_φ*φ)
           = Σ (a_r*b_r + a_φ*b_φ*φ²) + (a_r*b_φ + a_φ*b_r)*φ
    where φ² = φ + 1.
    """
    rat_part = 0  # coefficient of 1
    phi_part = 0  # coefficient of φ

    for (ar, ap), (br, bp) in zip(a, b):
        # (ar + ap*φ)(br + bp*φ) = ar*br + ap*bp*φ² + (ar*bp + ap*br)*φ
        # φ² = φ + 1, so ap*bp*φ² = ap*bp + ap*bp*φ
        rat_part += ar * br + ap * bp  # ar*br + ap*bp*(1)
        phi_part += ar * bp + ap * br + ap * bp  # (ar*bp + ap*br) + ap*bp from φ²

    return (rat_part, phi_part)


t0 = time.time()

# Project all roots
h4_roots = [project_to_h4(r) for r in roots]

# Compute H4 inner product distribution
h4_ip_counts = Counter()
for i in range(len(h4_roots)):
    for j in range(i + 1, len(h4_roots)):
        ip = h4_inner_product(h4_roots[i], h4_roots[j])
        h4_ip_counts[ip] += 1

print(f"\nStep 4: H4 projection inner product distribution ({time.time()-t0:.3f}s)")
print(f"  Inner products as (rational + phi_coeff * φ):")
# Sort by approximate value for readability
sorted_ips = sorted(h4_ip_counts.keys(), key=lambda x: x[0] + x[1] * 1.618)
for ip in sorted_ips:
    approx = ip[0] + ip[1] * (1 + np.sqrt(5)) / 2
    print(f"  ({ip[0]:3d} + {ip[1]:3d}φ) ~= {approx:+8.3f}: {h4_ip_counts[ip]:5d} pairs")


# ── Step 5: Which triples survive projection? ────────────────────

t0 = time.time()

# Check: for each E8 triple (α+β=γ), does proj(α)+proj(β)=proj(γ)?
# In exact Q(√5) arithmetic, this is just checking component-wise.
surviving = 0
broken = 0
survival_by_type = Counter()

for idx, (i, j, k) in enumerate(triples):
    pa, pb, pk = h4_roots[i], h4_roots[j], h4_roots[k]

    # Check if proj(α) + proj(β) = proj(γ) in Q(√5)
    matches = True
    for d in range(4):
        sum_rat = pa[d][0] + pb[d][0]
        sum_phi = pa[d][1] + pb[d][1]
        if sum_rat != pk[d][0] or sum_phi != pk[d][1]:
            matches = False
            break

    ip_ab = inner_product(roots[i], roots[j])

    if matches:
        surviving += 1
        survival_by_type[('survive', ip_ab)] += 1
    else:
        broken += 1
        survival_by_type[('broken', ip_ab)] += 1

print(f"\nStep 5: Triple survival under H4 projection ({time.time()-t0:.3f}s)")
print(f"  Total triples:    {len(triples)}")
print(f"  Surviving:        {surviving}")
print(f"  Broken:           {broken}")
print(f"  Survival rate:    {surviving/len(triples)*100:.1f}%")

print(f"\n  Breakdown by <α,β> type:")
for status in ['survive', 'broken']:
    for ip in sorted(set(ip for (s, ip) in survival_by_type if s == status)):
        key = (status, ip)
        if key in survival_by_type:
            total = survival_by_type.get(('survive', ip), 0) + survival_by_type.get(('broken', ip), 0)
            rate = survival_by_type.get(('survive', ip), 0) / total * 100 if total > 0 else 0
            if status == 'survive':
                print(f"    <α,β>={ip:3d}: {survival_by_type[key]:4d} survive / "
                      f"{total} total ({rate:.0f}%)")


# ── Step 6: Count graph structures ───────────────────────────────

t0 = time.time()

# Build adjacency by inner product value
# E8 root graph: connect roots with specific inner products
# The "addition graph": connect α-β if α+β is also a root

addition_neighbors = {}
for i, j, k in triples:
    addition_neighbors.setdefault(i, set()).add(j)
    addition_neighbors.setdefault(j, set()).add(i)

# Count triangles in the addition graph
triangles = 0
for i in range(len(roots)):
    neighbors_i = addition_neighbors.get(i, set())
    for j in neighbors_i:
        if j > i:
            neighbors_j = addition_neighbors.get(j, set())
            common = neighbors_i & neighbors_j
            triangles += len([k for k in common if k > j])

print(f"\nStep 6: Graph structures ({time.time()-t0:.3f}s)")
print(f"  Addition graph edges: {sum(len(v) for v in addition_neighbors.values()) // 2}")
print(f"  Triangles in addition graph: {triangles}")

# Degree distribution
degrees = Counter()
for i in range(len(roots)):
    d = len(addition_neighbors.get(i, set()))
    degrees[d] += 1

print(f"  Degree distribution:")
for d in sorted(degrees.keys()):
    print(f"    degree {d:3d}: {degrees[d]:3d} vertices")


# ── Step 7: Unique H4 projected points ───────────────────────────

t0 = time.time()

unique_h4 = set()
for h in h4_roots:
    unique_h4.add(h)

print(f"\nStep 7: Projected geometry ({time.time()-t0:.3f}s)")
print(f"  240 E8 roots project to {len(unique_h4)} unique H4 points")

# How many distinct norms?
h4_norms = Counter()
for h in h4_roots:
    norm = h4_inner_product(h, h)
    h4_norms[norm] += 1

print(f"  Distinct H4 norms: {len(h4_norms)}")
for norm in sorted(h4_norms.keys(), key=lambda x: x[0] + x[1] * 1.618):
    approx = norm[0] + norm[1] * (1 + np.sqrt(5)) / 2
    print(f"    norm=({norm[0]}+{norm[1]}φ) ~= {approx:.3f}: {h4_norms[norm]} roots")


# ── Step 8: Key integers to check against OEIS ──────────────────

print(f"\n" + "=" * 65)
print(f"  KEY INTEGERS (check against OEIS)")
print(f"=" * 65)

key_numbers = {
    "E8 roots": 240,
    "Root addition triples": len(triples),
    "Triples surviving H4 projection": surviving,
    "Triples broken by projection": broken,
    "Unique H4 projected points": len(unique_h4),
    "Addition graph triangles": triangles,
    "Addition graph edges": sum(len(v) for v in addition_neighbors.values()) // 2,
}

for name, val in key_numbers.items():
    print(f"  {val:8d}  {name}")

print(f"\n  Search these on https://oeis.org/ for unexpected connections.")
print(f"  If any count is NOT in OEIS, it may be a new sequence.")


# ── Save results ─────────────────────────────────────────────────

results = {
    "e8_roots": len(roots),
    "triples_total": len(triples),
    "triples_surviving": surviving,
    "triples_broken": broken,
    "survival_rate": surviving / len(triples),
    "unique_h4_points": len(unique_h4),
    "addition_graph_triangles": triangles,
    "addition_graph_edges": sum(len(v) for v in addition_neighbors.values()) // 2,
    "degree_distribution": {str(k): v for k, v in sorted(degrees.items())},
    "ip_distribution_e8": {str(k): v for k, v in sorted(ip_counts.items())},
    "key_integers": key_numbers,
}

out_path = os.path.join(os.path.dirname(__file__), "e8_h4_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to {out_path}")
print(f"  Total time: {time.time() - t0:.1f}s")
