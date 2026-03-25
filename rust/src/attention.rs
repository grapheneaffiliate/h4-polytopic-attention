//! H₄ Polytopic Attention: multi-head 4D attention with parallel head queries.

use rayon::prelude::*;
use crate::vec4::{Vec4, PHI};
use crate::h4;
use crate::chamber_tree::ChamberTree;

pub struct AttentionHead {
    cache: ChamberTree,
}

impl AttentionHead {
    pub fn new(simple_roots: [Vec4; 4]) -> Self {
        AttentionHead {
            cache: ChamberTree::new(simple_roots),
        }
    }

    pub fn insert(&mut self, key: Vec4, value: [f64; 4], ts: u64) {
        self.cache.insert(key, value, ts);
    }

    pub fn query_exact(&self, query: Vec4) -> Option<(f64, u64)> {
        self.cache.query_max_exact(query).map(|(s, _, t)| (s, t))
    }

    pub fn query_approx(&self, query: Vec4) -> Option<(f64, u64)> {
        self.cache.query_max_approx(query).map(|(s, _, t)| (s, t))
    }

    pub fn size(&self) -> u64 {
        self.cache.size
    }
}

pub struct H4Attention {
    heads: Vec<AttentionHead>,
    pub n_heads: usize,
    step: u64,
}

impl H4Attention {
    pub fn new(d_model: usize) -> Self {
        assert!(d_model % 4 == 0);
        let n_heads = d_model / 4;
        let roots = h4::simple_roots();
        let heads = (0..n_heads).map(|_| AttentionHead::new(roots)).collect();
        H4Attention { heads, n_heads, step: 0 }
    }

    pub fn insert(&mut self, embedding: &[f64]) {
        let ts = self.step;
        self.step += 1;
        for h in 0..self.n_heads {
            let o = h * 4;
            let key = Vec4::new(embedding[o], embedding[o+1], embedding[o+2], embedding[o+3]);
            let value = [embedding[o], embedding[o+1], embedding[o+2], embedding[o+3]];
            self.heads[h].insert(key, value, ts);
        }
    }

    /// Serial exact query across all heads.
    pub fn query_exact(&self, embedding: &[f64]) -> Vec<Option<(f64, u64)>> {
        (0..self.n_heads).map(|h| {
            let o = h * 4;
            let q = Vec4::new(
                embedding[o] * PHI, embedding[o+1] * PHI,
                embedding[o+2] * PHI, embedding[o+3] * PHI,
            );
            self.heads[h].query_exact(q)
        }).collect()
    }

    /// Serial approximate query across all heads.
    pub fn query_approx(&self, embedding: &[f64]) -> Vec<Option<(f64, u64)>> {
        (0..self.n_heads).map(|h| {
            let o = h * 4;
            let q = Vec4::new(
                embedding[o] * PHI, embedding[o+1] * PHI,
                embedding[o+2] * PHI, embedding[o+3] * PHI,
            );
            self.heads[h].query_approx(q)
        }).collect()
    }

    /// Parallel exact query — distributes heads across threads via rayon.
    pub fn query_exact_par(&self, embedding: &[f64]) -> Vec<Option<(f64, u64)>> {
        self.heads.par_iter().enumerate().map(|(h, head)| {
            let o = h * 4;
            let q = Vec4::new(
                embedding[o] * PHI, embedding[o+1] * PHI,
                embedding[o+2] * PHI, embedding[o+3] * PHI,
            );
            head.query_exact(q)
        }).collect()
    }

    /// Parallel approximate query — distributes heads across threads via rayon.
    pub fn query_approx_par(&self, embedding: &[f64]) -> Vec<Option<(f64, u64)>> {
        self.heads.par_iter().enumerate().map(|(h, head)| {
            let o = h * 4;
            let q = Vec4::new(
                embedding[o] * PHI, embedding[o+1] * PHI,
                embedding[o+2] * PHI, embedding[o+3] * PHI,
            );
            head.query_approx(q)
        }).collect()
    }

    pub fn cache_size(&self) -> u64 {
        self.heads.first().map(|h| h.size()).unwrap_or(0)
    }
}
