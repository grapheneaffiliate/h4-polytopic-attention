//! 8D vector type for E₈ lattice embeddings.
//!
//! Two AVX2 registers (2×[f64;4]) fit a single 8D vector.
//! All operations are branchless and vectorization-friendly.
//! This is the "fat" vector that lives in E₈ space before
//! projection down to 4D H₄ space.

use std::ops::{Add, Sub};
use crate::vec4::Vec4;

/// An 8D vector, aligned for SIMD.
#[derive(Clone, Copy, Debug)]
#[repr(align(64))]
pub struct Vec8(pub [f64; 8]);

impl Vec8 {
    #[inline(always)]
    pub fn new(v: [f64; 8]) -> Self {
        Vec8(v)
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Vec8([0.0; 8])
    }

    /// Dot product — two 4-wide fused multiply-adds.
    #[inline(always)]
    pub fn dot(self, other: Self) -> f64 {
        let a = self.0;
        let b = other.0;
        // Two independent 4-wide reductions for ILP
        let lo = (a[0] * b[0] + a[1] * b[1]) + (a[2] * b[2] + a[3] * b[3]);
        let hi = (a[4] * b[4] + a[5] * b[5]) + (a[6] * b[6] + a[7] * b[7]);
        lo + hi
    }

    #[inline(always)]
    pub fn norm_sq(self) -> f64 {
        self.dot(self)
    }

    #[inline(always)]
    pub fn norm(self) -> f64 {
        self.norm_sq().sqrt()
    }

    #[inline(always)]
    pub fn normalized(self) -> Self {
        let n = self.norm();
        if n < 1e-12 {
            return Self::zero();
        }
        self.scale(1.0 / n)
    }

    #[inline(always)]
    pub fn scale(self, s: f64) -> Self {
        Vec8([
            self.0[0] * s, self.0[1] * s, self.0[2] * s, self.0[3] * s,
            self.0[4] * s, self.0[5] * s, self.0[6] * s, self.0[7] * s,
        ])
    }

    #[inline(always)]
    pub fn dist_sq(self, other: Self) -> f64 {
        (self - other).norm_sq()
    }

    /// Project 8D → 4D via the E₈ → H₄ projection matrix.
    /// The projection uses cos(π/5) = φ/2 eigenvalues of the Coxeter element.
    #[inline(always)]
    pub fn project_to_h4(self, proj: &E8H4Projection) -> Vec4 {
        Vec4([
            proj.row_dot(0, self),
            proj.row_dot(1, self),
            proj.row_dot(2, self),
            proj.row_dot(3, self),
        ])
    }

    /// Round each component to nearest integer.
    #[inline(always)]
    pub fn round(self) -> Self {
        Vec8([
            self.0[0].round(), self.0[1].round(), self.0[2].round(), self.0[3].round(),
            self.0[4].round(), self.0[5].round(), self.0[6].round(), self.0[7].round(),
        ])
    }

    /// Floor each component and add 0.5 (half-integer lattice).
    #[inline(always)]
    pub fn half_integer(self) -> Self {
        Vec8([
            self.0[0].floor() + 0.5, self.0[1].floor() + 0.5,
            self.0[2].floor() + 0.5, self.0[3].floor() + 0.5,
            self.0[4].floor() + 0.5, self.0[5].floor() + 0.5,
            self.0[6].floor() + 0.5, self.0[7].floor() + 0.5,
        ])
    }

    /// Sum of all components.
    #[inline(always)]
    pub fn component_sum(self) -> f64 {
        (self.0[0] + self.0[1] + self.0[2] + self.0[3])
            + (self.0[4] + self.0[5] + self.0[6] + self.0[7])
    }

    /// Convert to integer key for hashing (multiply by scale, round).
    #[inline(always)]
    pub fn to_lattice_key(self) -> [i32; 8] {
        [
            self.0[0].round() as i32, self.0[1].round() as i32,
            self.0[2].round() as i32, self.0[3].round() as i32,
            self.0[4].round() as i32, self.0[5].round() as i32,
            self.0[6].round() as i32, self.0[7].round() as i32,
        ]
    }
}

impl Add for Vec8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Vec8([
            self.0[0] + rhs.0[0], self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2], self.0[3] + rhs.0[3],
            self.0[4] + rhs.0[4], self.0[5] + rhs.0[5],
            self.0[6] + rhs.0[6], self.0[7] + rhs.0[7],
        ])
    }
}

impl Sub for Vec8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Vec8([
            self.0[0] - rhs.0[0], self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2], self.0[3] - rhs.0[3],
            self.0[4] - rhs.0[4], self.0[5] - rhs.0[5],
            self.0[6] - rhs.0[6], self.0[7] - rhs.0[7],
        ])
    }
}

/// The E₈ → H₄ projection matrix (4×8).
///
/// Uses the eigenvalues of the H₄ Coxeter element:
///   cos(π/5) = φ/2,  sin(π/5)
///   cos(2π/5) = 1/(2φ),  sin(2π/5)
///
/// Two 2D rotation blocks map 8D coordinates to 4D,
/// preserving the golden-ratio structure that connects
/// E₈ lattice geometry to H₄ attention.
pub struct E8H4Projection {
    pub rows: [[f64; 8]; 4],
}

impl E8H4Projection {
    pub fn new() -> Self {
        let c1 = std::f64::consts::PI / 5.0;
        let c2 = 2.0 * std::f64::consts::PI / 5.0;
        let (s1, c1) = (c1.sin(), c1.cos());
        let (s2, c2) = (c2.sin(), c2.cos());

        E8H4Projection {
            rows: [
                [ c1,  s1,  c2,  s2,  0.0, 0.0, 0.0, 0.0],
                [-s1,  c1, -s2,  c2,  0.0, 0.0, 0.0, 0.0],
                [ 0.0, 0.0, 0.0, 0.0,  c1,  s1,  c2,  s2],
                [ 0.0, 0.0, 0.0, 0.0, -s1,  c1, -s2,  c2],
            ],
        }
    }

    /// Dot product of one row with a Vec8.
    #[inline(always)]
    fn row_dot(&self, row: usize, v: Vec8) -> f64 {
        let r = &self.rows[row];
        (r[0] * v.0[0] + r[1] * v.0[1] + r[2] * v.0[2] + r[3] * v.0[3])
            + (r[4] * v.0[4] + r[5] * v.0[5] + r[6] * v.0[6] + r[7] * v.0[7])
    }
}
