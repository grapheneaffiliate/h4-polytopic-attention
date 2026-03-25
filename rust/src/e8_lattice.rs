//! E₈ lattice decoder and Voronoi cell geometry.
//!
//! The E₈ lattice is the densest sphere packing in 8D (Viazovska 2016).
//! It decomposes as E₈ = D₈ ∪ (D₈ + [½]⁸), where D₈ = {x ∈ Z⁸ : Σxᵢ ≡ 0 mod 2}.
//!
//! The closest-lattice-point decoder gives O(1) Voronoi cell lookup:
//! given any point in R⁸, find which E₈ lattice point's Voronoi cell
//! contains it. This becomes the memory address.
//!
//! The kissing number of E₈ is 240 — each lattice point has exactly
//! 240 nearest neighbors. This bounds the neighbor shell search for
//! approximate queries.

use crate::vec8::Vec8;

/// E₈ lattice point represented as integer coordinates.
/// For the D₈ coset these are integers; for D₈+½ they're doubled
/// (so half-integers become odd integers, stored as 2x).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LatticePoint {
    /// Coordinates × 2 (so half-integers are representable as odd ints).
    pub coords: [i32; 8],
}

impl LatticePoint {
    pub fn zero() -> Self {
        LatticePoint { coords: [0; 8] }
    }

    /// Convert back to floating-point Vec8.
    #[inline]
    pub fn to_vec8(self) -> Vec8 {
        Vec8([
            self.coords[0] as f64 / 2.0,
            self.coords[1] as f64 / 2.0,
            self.coords[2] as f64 / 2.0,
            self.coords[3] as f64 / 2.0,
            self.coords[4] as f64 / 2.0,
            self.coords[5] as f64 / 2.0,
            self.coords[6] as f64 / 2.0,
            self.coords[7] as f64 / 2.0,
        ])
    }

    /// Squared distance from this lattice point to a Vec8.
    #[inline]
    pub fn dist_sq_to(self, point: Vec8) -> f64 {
        self.to_vec8().dist_sq(point)
    }
}

/// Decode a point in R⁸ to the nearest E₈ lattice point.
///
/// Algorithm:
/// 1. Find closest D₈ point (integer coords with even sum)
/// 2. Find closest D₈ + [½]⁸ point (half-integer coords with even sum)
/// 3. Return whichever is closer
///
/// This is O(1) — just rounding + parity correction.
pub fn decode_to_e8(point: Vec8) -> LatticePoint {
    // Coset 1: D₈ (integers with even coordinate sum)
    let d8 = decode_to_d8(point);

    // Coset 2: D₈ + [½]⁸ (half-integers with even sum)
    let d8_half = decode_to_d8_half(point);

    // Pick the closer one
    let dist_d8 = d8.dist_sq_to(point);
    let dist_half = d8_half.dist_sq_to(point);

    if dist_d8 <= dist_half { d8 } else { d8_half }
}

/// Decode to closest D₈ lattice point.
/// D₈ = {x ∈ Z⁸ : x₁ + x₂ + ... + x₈ ≡ 0 (mod 2)}
fn decode_to_d8(point: Vec8) -> LatticePoint {
    let mut rounded = [0i32; 8];
    let mut errors = [0.0f64; 8];

    for i in 0..8 {
        let r = point.0[i].round();
        rounded[i] = r as i32;
        errors[i] = (point.0[i] - r).abs();
    }

    // Check parity: sum must be even
    let sum: i32 = rounded.iter().sum();
    if sum % 2 != 0 {
        // Flip the coordinate with largest rounding error
        let max_idx = errors.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap();

        if point.0[max_idx] > rounded[max_idx] as f64 {
            rounded[max_idx] += 1;
        } else {
            rounded[max_idx] -= 1;
        }
    }

    // Store as 2× for uniform representation with half-integer coset
    let mut coords = [0i32; 8];
    for i in 0..8 {
        coords[i] = rounded[i] * 2;
    }
    LatticePoint { coords }
}

/// Decode to closest D₈ + [½]⁸ lattice point.
/// These are half-integer points with even sum.
fn decode_to_d8_half(point: Vec8) -> LatticePoint {
    let mut rounded = [0i32; 8];
    let mut errors = [0.0f64; 8];

    for i in 0..8 {
        // Round to nearest half-integer: floor + 0.5
        let half = point.0[i].floor() + 0.5;
        // Convert to doubled representation: 2 * half = 2*floor + 1 (always odd)
        rounded[i] = (half * 2.0).round() as i32;
        errors[i] = (point.0[i] - half).abs();
    }

    // Check parity of the half-integer sum:
    // sum of (rounded[i]/2) must be integer, i.e., sum of rounded[i] must be even
    let sum: i32 = rounded.iter().sum();
    if sum % 4 != 0 {
        // Flip the coordinate with largest rounding error
        let max_idx = errors.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap();

        if point.0[max_idx] > (rounded[max_idx] as f64 / 2.0) {
            rounded[max_idx] += 2;
        } else {
            rounded[max_idx] -= 2;
        }
    }

    LatticePoint { coords: rounded }
}

/// The 240 nearest neighbors of the origin in E₈.
///
/// These are the E₈ root vectors — the kissing configuration.
/// They come in three orbits:
///   - 112 vectors: permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
///   -  128 vectors: (±½)⁸ with even number of minus signs
///
/// All have norm² = 2 (or norm² = 2 in the ×2 representation = coords² sum = 8).
///
/// These define the neighbor shells for Voronoi cell traversal.
pub fn kissing_vectors() -> Vec<LatticePoint> {
    let mut neighbors = Vec::with_capacity(240);

    // Orbit 1: ±eᵢ ± eⱼ for i < j — 112 vectors
    // In ×2 representation: coords with two ±2 entries, rest 0
    for i in 0..8 {
        for j in (i + 1)..8 {
            for &si in &[2i32, -2] {
                for &sj in &[2i32, -2] {
                    let mut coords = [0i32; 8];
                    coords[i] = si;
                    coords[j] = sj;
                    neighbors.push(LatticePoint { coords });
                }
            }
        }
    }

    // Orbit 2: (±½)⁸ with even number of minus signs — 128 vectors
    // In ×2 representation: all coords ±1, with even number of -1s
    for mask in 0..256u32 {
        let minus_count = mask.count_ones();
        if minus_count % 2 != 0 {
            continue;
        }
        let mut coords = [1i32; 8];
        for k in 0..8 {
            if mask & (1 << k) != 0 {
                coords[k] = -1;
            }
        }
        neighbors.push(LatticePoint { coords });
    }

    debug_assert_eq!(neighbors.len(), 240);
    neighbors
}

/// Add two lattice points (in ×2 representation).
pub fn lattice_add(a: LatticePoint, b: LatticePoint) -> LatticePoint {
    let mut coords = [0i32; 8];
    for i in 0..8 {
        coords[i] = a.coords[i] + b.coords[i];
    }
    LatticePoint { coords }
}

/// Generate the neighbor shell of a lattice point:
/// the 240 kissing vectors translated to that point.
pub fn neighbor_shell(center: LatticePoint) -> Vec<LatticePoint> {
    kissing_vectors().into_iter()
        .map(|kv| lattice_add(center, kv))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_origin() {
        let p = Vec8([0.1, -0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let lp = decode_to_e8(p);
        // Should decode to origin (all zeros in ×2 representation)
        assert_eq!(lp.coords, [0; 8]);
    }

    #[test]
    fn test_decode_integer_point() {
        let p = Vec8([1.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let lp = decode_to_e8(p);
        let v = lp.to_vec8();
        // Should snap to (2, 0, 0, ...) — coords [4, 0, 0, ...]
        assert_eq!(v.0[0], 2.0);
        assert_eq!(v.0[1], 0.0);
    }

    #[test]
    fn test_kissing_count() {
        let kv = kissing_vectors();
        assert_eq!(kv.len(), 240);
    }

    #[test]
    fn test_kissing_norms() {
        let kv = kissing_vectors();
        for v in &kv {
            let norm_sq: i32 = v.coords.iter().map(|c| c * c).sum();
            // In ×2 representation, norm² should be 8 (= 4 × actual norm² of 2)
            assert!(norm_sq == 8, "kissing vector norm² = {} (expected 8)", norm_sq);
        }
    }

    #[test]
    fn test_roundtrip() {
        // A known E₈ lattice point should decode to itself
        let lp = LatticePoint { coords: [2, 2, 0, 0, 0, 0, 0, 0] }; // = (1,1,0,...,0) ∈ D₈
        let v = lp.to_vec8();
        let decoded = decode_to_e8(v);
        assert_eq!(lp.coords, decoded.coords);
    }
}
