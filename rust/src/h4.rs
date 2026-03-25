//! H₄ geometry: 600-cell vertices and Coxeter chamber structure.
//!
//! The 600-cell has 120 vertices on S³, organized by the H₄ reflection
//! group with 14,400 elements. The 4 simple roots define reflection
//! hyperplanes that partition S³ into Coxeter chambers.

use crate::vec4::{Vec4, PHI, PHI_INV};

/// The 4 simple roots of H₄, normalized.
/// These define the walls of the fundamental Coxeter chamber.
pub fn simple_roots() -> [Vec4; 4] {
    let mut roots = [
        Vec4::new(1.0, -1.0, 0.0, 0.0),                                          // α₁
        Vec4::new(0.0, 1.0, -1.0, 0.0),                                           // α₂
        Vec4::new(0.0, 0.0, 1.0, 0.0),                                            // α₃
        Vec4::new(-0.5, -0.5, -0.5, -0.5 * PHI_INV + 0.5 * PHI),                  // α₄
    ];
    for r in &mut roots {
        *r = r.normalized();
    }
    roots
}

/// Generate all 120 vertices of the 600-cell on the unit 3-sphere.
pub fn generate_600_cell() -> Vec<Vec4> {
    let mut vertices = Vec::with_capacity(120);

    // Orbit 1: permutations of (±1, 0, 0, 0) — 8 vertices
    for i in 0..4 {
        for &sign in &[1.0_f64, -1.0] {
            let mut v = [0.0; 4];
            v[i] = sign;
            vertices.push(Vec4(v));
        }
    }

    // Orbit 2: (±½, ±½, ±½, ±½) — 16 vertices
    for mask in 0..16u32 {
        let v = [
            if mask & 1 != 0 { -0.5 } else { 0.5 },
            if mask & 2 != 0 { -0.5 } else { 0.5 },
            if mask & 4 != 0 { -0.5 } else { 0.5 },
            if mask & 8 != 0 { -0.5 } else { 0.5 },
        ];
        vertices.push(Vec4(v));
    }

    // Orbit 3: even permutations of (0, ±½, ±φ/2, ±1/(2φ)) — 96 vertices
    let base = [0.0, 0.5, PHI / 2.0, PHI_INV / 2.0];
    let even_perms: [(usize, usize, usize, usize); 12] = [
        (0,1,2,3), (0,2,3,1), (0,3,1,2),
        (1,0,3,2), (1,2,0,3), (1,3,2,0),
        (2,0,1,3), (2,1,3,0), (2,3,0,1),
        (3,0,2,1), (3,1,0,2), (3,2,1,0),
    ];

    for &(a, b, c, d) in &even_perms {
        let coords = [base[a], base[b], base[c], base[d]];
        // Find non-zero positions
        let non_zero: Vec<usize> = (0..4).filter(|&i| coords[i].abs() > 1e-12).collect();
        let n = non_zero.len();

        for sign_mask in 0..(1u32 << n) {
            let mut v = coords;
            for (j, &idx) in non_zero.iter().enumerate() {
                if sign_mask & (1 << j) != 0 {
                    v[idx] = -v[idx];
                }
            }
            vertices.push(Vec4(v));
        }
    }

    // Normalize all to unit sphere
    for v in &mut vertices {
        *v = v.normalized();
    }

    // Deduplicate
    let mut unique: Vec<Vec4> = Vec::with_capacity(120);
    'outer: for v in &vertices {
        for u in &unique {
            if v.dist_sq(*u) < 1e-14 {
                continue 'outer;
            }
        }
        unique.push(*v);
    }

    unique
}

/// Verify the 600-cell structure: should have exactly 120 vertices,
/// all on the unit sphere, with specific dot product values involving φ.
pub fn verify_600_cell(vertices: &[Vec4]) -> bool {
    if vertices.len() != 120 {
        eprintln!("Expected 120 vertices, got {}", vertices.len());
        return false;
    }

    for (i, v) in vertices.iter().enumerate() {
        let n = v.norm();
        if (n - 1.0).abs() > 1e-10 {
            eprintln!("Vertex {} not on unit sphere: norm = {}", i, n);
            return false;
        }
    }

    // Check that φ/2 appears among dot products
    let target = PHI / 2.0;
    let mut found = false;
    'search: for i in 0..vertices.len() {
        for j in (i+1)..vertices.len() {
            let d = vertices[i].dot(vertices[j]);
            if (d - target).abs() < 0.01 {
                found = true;
                break 'search;
            }
        }
    }

    if !found {
        eprintln!("φ/2 not found in dot products");
        return false;
    }

    true
}
