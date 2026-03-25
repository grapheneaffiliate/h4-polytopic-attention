//! 4D vector type optimized for SIMD-friendly layout.
//!
//! On x86_64 with AVX2, a [f64; 4] fits in a single 256-bit register.
//! All operations are branchless and vectorization-friendly.

use std::ops::{Add, Sub, Mul, Neg};

/// A 4D vector on the stack, aligned for SIMD.
#[derive(Clone, Copy, Debug)]
#[repr(align(32))]
pub struct Vec4(pub [f64; 4]);

/// The golden ratio phi = (1+sqrt(5))/2
pub const PHI: f64 = 1.618033988749895;
/// 1/phi = phi - 1
pub const PHI_INV: f64 = 0.6180339887498949;

impl Vec4 {
    #[inline(always)]
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Vec4([x, y, z, w])
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Vec4([0.0; 4])
    }

    /// Dot product — the core operation. With `repr(align(32))` and
    /// `opt-level=3`, LLVM will emit a single `vmulpd` + horizontal add
    /// on AVX2 targets.
    #[inline(always)]
    pub fn dot(self, other: Self) -> f64 {
        // Encourage SIMD: multiply all 4 lanes, then reduce.
        // Written as a single expression so LLVM sees the full reduction.
        let a = self.0;
        let b = other.0;
        (a[0] * b[0] + a[1] * b[1]) + (a[2] * b[2] + a[3] * b[3])
    }

    /// Dot product of 4 Vec4s against the same query, returning 4 scores.
    /// This lets LLVM interleave the multiplies across 4 independent chains.
    #[inline(always)]
    pub fn dot_4(query: Self, keys: &[Vec4; 4]) -> [f64; 4] {
        [
            query.dot(keys[0]),
            query.dot(keys[1]),
            query.dot(keys[2]),
            query.dot(keys[3]),
        ]
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
        Vec4([
            self.0[0] * s,
            self.0[1] * s,
            self.0[2] * s,
            self.0[3] * s,
        ])
    }

    /// Reflect self across the hyperplane orthogonal to `normal`.
    #[inline(always)]
    pub fn reflect(self, normal: Self) -> Self {
        let d = 2.0 * self.dot(normal);
        Vec4([
            self.0[0] - d * normal.0[0],
            self.0[1] - d * normal.0[1],
            self.0[2] - d * normal.0[2],
            self.0[3] - d * normal.0[3],
        ])
    }

    #[inline(always)]
    pub fn dist_sq(self, other: Self) -> f64 {
        (self - other).norm_sq()
    }

    /// Component-wise max of two Vec4s (for bounding).
    #[inline(always)]
    pub fn max_components(self, other: Self) -> Self {
        Vec4([
            self.0[0].max(other.0[0]),
            self.0[1].max(other.0[1]),
            self.0[2].max(other.0[2]),
            self.0[3].max(other.0[3]),
        ])
    }
}

impl Add for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Vec4([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
        ])
    }
}

impl Sub for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Vec4([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
        ])
    }
}

impl Mul<f64> for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        self.scale(rhs)
    }
}

impl Neg for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Vec4([-self.0[0], -self.0[1], -self.0[2], -self.0[3]])
    }
}
