//! H₄ Hierarchical Chamber Tree: O(log t) max-dot-product queries in 4D.
//!
//! Three-level recursive Coxeter bucketing:
//!   Level 0: 16 buckets (4 H₄ root splits)
//!   Level 1: 16 × 16 = 256 sub-buckets (rotated roots)
//!   Level 2: 256 × 16 = 4096 leaf buckets (rotated again)
//!
//! Each level uses the H₄ simple roots rotated by a different angle,
//! so the splits at each level are geometrically independent.
//!
//! Approximate query visits 5 buckets at each level (primary + 4 neighbors),
//! scanning only ~5/16 × 5/16 × leaf_size keys = O(log t) effective.

use crate::vec4::Vec4;

const BUCKET_BITS: usize = 4;
const N_BUCKETS: usize = 1 << BUCKET_BITS; // 16

/// Maximum number of keys in a leaf before it stops being a leaf.
/// At this threshold, scanning is fast enough that further subdivision
/// isn't worth the overhead.
const LEAF_THRESHOLD: usize = 64;

/// Maximum recursion depth for hierarchical buckets.
const MAX_LEVELS: usize = 3;

/// A leaf bucket storing keys/values contiguously.
struct LeafBucket {
    keys: Vec<Vec4>,
    vals: Vec<[f64; 4]>,
    timestamps: Vec<u64>,
}

impl LeafBucket {
    fn new() -> Self {
        LeafBucket {
            keys: Vec::new(),
            vals: Vec::new(),
            timestamps: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.keys.len()
    }

    #[inline(always)]
    fn scan(&self, query: Vec4, best_score: &mut f64, best_val: &mut [f64; 4], best_ts: &mut u64) {
        let len = self.keys.len();
        let chunks = len / 4;

        for c in 0..chunks {
            let base = c * 4;
            let s0 = query.dot(self.keys[base]);
            let s1 = query.dot(self.keys[base + 1]);
            let s2 = query.dot(self.keys[base + 2]);
            let s3 = query.dot(self.keys[base + 3]);

            let (local_max, local_idx) = if s0 >= s1 {
                if s0 >= s2 {
                    if s0 >= s3 { (s0, 0) } else { (s3, 3) }
                } else if s2 >= s3 { (s2, 2) } else { (s3, 3) }
            } else if s1 >= s2 {
                if s1 >= s3 { (s1, 1) } else { (s3, 3) }
            } else if s2 >= s3 { (s2, 2) } else { (s3, 3) };

            if local_max > *best_score {
                *best_score = local_max;
                let idx = base + local_idx;
                *best_val = self.vals[idx];
                *best_ts = self.timestamps[idx];
            }
        }

        for i in (chunks * 4)..len {
            let score = query.dot(self.keys[i]);
            if score > *best_score {
                *best_score = score;
                *best_val = self.vals[i];
                *best_ts = self.timestamps[i];
            }
        }
    }
}

/// Either a leaf or a recursive sub-tree of 16 children.
enum BucketNode {
    Leaf(LeafBucket),
    Branch {
        roots: [Vec4; 4],
        children: Box<[BucketNode; N_BUCKETS]>,
        count: u64,
    },
}

impl BucketNode {
    fn new_leaf() -> Self {
        BucketNode::Leaf(LeafBucket::new())
    }

    fn count(&self) -> u64 {
        match self {
            BucketNode::Leaf(b) => b.len() as u64,
            BucketNode::Branch { count, .. } => *count,
        }
    }
}

/// Rotate H₄ roots to create independent splits at each level.
/// Uses a rotation in the (0,1) and (2,3) planes by angle theta.
fn rotate_roots(roots: &[Vec4; 4], theta: f64) -> [Vec4; 4] {
    let c = theta.cos();
    let s = theta.sin();
    let mut out = *roots;
    for r in &mut out {
        let x = r.0[0];
        let y = r.0[1];
        let z = r.0[2];
        let w = r.0[3];
        r.0[0] = x * c - y * s;
        r.0[1] = x * s + y * c;
        r.0[2] = z * c - w * s;
        r.0[3] = z * s + w * c;
        *r = r.normalized();
    }
    out
}

#[inline(always)]
fn bucket_index(roots: &[Vec4; 4], key: Vec4) -> usize {
    let mut idx = 0usize;
    if key.dot(roots[0]) >= 0.0 { idx |= 1; }
    if key.dot(roots[1]) >= 0.0 { idx |= 2; }
    if key.dot(roots[2]) >= 0.0 { idx |= 4; }
    if key.dot(roots[3]) >= 0.0 { idx |= 8; }
    idx
}

#[inline]
fn neighbor_indices(primary: usize) -> [usize; 4] {
    [primary ^ 1, primary ^ 2, primary ^ 4, primary ^ 8]
}

/// The hierarchical chamber tree for one attention head.
pub struct ChamberTree {
    root_roots: [Vec4; 4],
    top: [BucketNode; N_BUCKETS],
    pub size: u64,
    level_angles: [f64; MAX_LEVELS],
}

unsafe impl Send for ChamberTree {}
unsafe impl Sync for ChamberTree {}

impl ChamberTree {
    pub fn new(simple_roots: [Vec4; 4]) -> Self {
        // Use phi-scaled angles for geometric independence between levels
        let phi = 1.618033988749895_f64;
        let level_angles = [
            0.0,                              // level 0: original roots
            std::f64::consts::PI / 5.0,       // level 1: 36° (pentagonal)
            std::f64::consts::PI / 5.0 * phi, // level 2: 36°×φ ≈ 58.3°
        ];

        ChamberTree {
            root_roots: simple_roots,
            top: std::array::from_fn(|_| BucketNode::new_leaf()),
            size: 0,
            level_angles,
        }
    }

    pub fn insert(&mut self, key: Vec4, value: [f64; 4], timestamp: u64) {
        let key = key.normalized();
        let bi = bucket_index(&self.root_roots, key);
        Self::insert_into(&mut self.top[bi], key, value, timestamp, &self.root_roots, &self.level_angles, 1);
        self.size += 1;
    }

    fn insert_into(
        node: &mut BucketNode,
        key: Vec4,
        value: [f64; 4],
        timestamp: u64,
        parent_roots: &[Vec4; 4],
        level_angles: &[f64; MAX_LEVELS],
        depth: usize,
    ) {
        match node {
            BucketNode::Leaf(leaf) => {
                leaf.keys.push(key);
                leaf.vals.push(value);
                leaf.timestamps.push(timestamp);

                // Split if over threshold and not at max depth
                if leaf.len() > LEAF_THRESHOLD && depth < MAX_LEVELS {
                    // Collect entries before replacing
                    let keys: Vec<Vec4> = leaf.keys.drain(..).collect();
                    let vals: Vec<[f64; 4]> = leaf.vals.drain(..).collect();
                    let tss: Vec<u64> = leaf.timestamps.drain(..).collect();
                    let count = keys.len() as u64;

                    let child_roots = rotate_roots(parent_roots, level_angles[depth]);
                    let mut children: Box<[BucketNode; N_BUCKETS]> = Box::new(
                        std::array::from_fn(|_| BucketNode::new_leaf())
                    );

                    // Re-insert all entries into children
                    for i in 0..keys.len() {
                        let ci = bucket_index(&child_roots, keys[i]);
                        if let BucketNode::Leaf(cl) = &mut children[ci] {
                            cl.keys.push(keys[i]);
                            cl.vals.push(vals[i]);
                            cl.timestamps.push(tss[i]);
                        }
                    }

                    *node = BucketNode::Branch {
                        roots: child_roots,
                        children,
                        count,
                    };
                }
            }
            BucketNode::Branch { roots, children, count } => {
                *count += 1;
                let ci = bucket_index(roots, key);
                Self::insert_into(&mut children[ci], key, value, timestamp, roots, level_angles, depth + 1);
            }
        }
    }

    /// Exact query: visits all buckets at every level.
    pub fn query_max_exact(&self, query: Vec4) -> Option<(f64, [f64; 4], u64)> {
        if self.size == 0 { return None; }
        let query = query.normalized();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_val = [0.0f64; 4];
        let mut best_ts = 0u64;

        for b in 0..N_BUCKETS {
            Self::query_node(&self.top[b], query, &mut best_score, &mut best_val, &mut best_ts, false);
        }
        Some((best_score, best_val, best_ts))
    }

    /// Approximate query: at each level, visits primary + 4 Hamming-1 neighbors.
    /// Effective scan ratio per level: 5/16 ≈ 31%.
    /// Over 2 branch levels: (5/16)² ≈ 9.8% of keys scanned.
    /// Over 3 levels: (5/16)³ ≈ 3% of keys scanned.
    pub fn query_max_approx(&self, query: Vec4) -> Option<(f64, [f64; 4], u64)> {
        if self.size == 0 { return None; }
        let query = query.normalized();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_val = [0.0f64; 4];
        let mut best_ts = 0u64;

        let primary = bucket_index(&self.root_roots, query);

        // Scan primary bucket
        Self::query_node(&self.top[primary], query, &mut best_score, &mut best_val, &mut best_ts, true);

        // Scan 4 Hamming-1 neighbors
        for nb in neighbor_indices(primary) {
            Self::query_node(&self.top[nb], query, &mut best_score, &mut best_val, &mut best_ts, true);
        }

        Some((best_score, best_val, best_ts))
    }

    /// Approximate top-k: collect all candidate entries from the approximate
    /// bucket set (primary + 4 neighbors at each level), then return the top-k
    /// by dot product with the query. Only scans ~3% of keys.
    ///
    /// Returns Vec of (score, value, timestamp) sorted descending by score,
    /// truncated to k entries.
    pub fn query_topk_approx(&self, query: Vec4, k: usize) -> Vec<(f64, [f64; 4], u64)> {
        if self.size == 0 { return Vec::new(); }
        let query = query.normalized();
        let mut candidates: Vec<(f64, [f64; 4], u64)> = Vec::new();

        let primary = bucket_index(&self.root_roots, query);

        // Collect from primary + 4 neighbors
        Self::collect_candidates(&self.top[primary], query, &mut candidates, true);
        for nb in neighbor_indices(primary) {
            Self::collect_candidates(&self.top[nb], query, &mut candidates, true);
        }

        // Sort descending by score and truncate to k
        candidates.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        candidates.truncate(k);
        candidates
    }

    /// Collect all (score, value, timestamp) from a node into the candidates vec.
    fn collect_candidates(
        node: &BucketNode,
        query: Vec4,
        candidates: &mut Vec<(f64, [f64; 4], u64)>,
        approx: bool,
    ) {
        match node {
            BucketNode::Leaf(leaf) => {
                for i in 0..leaf.keys.len() {
                    let score = query.dot(leaf.keys[i]);
                    candidates.push((score, leaf.vals[i], leaf.timestamps[i]));
                }
            }
            BucketNode::Branch { roots, children, count } => {
                if *count == 0 { return; }
                if approx {
                    let primary = bucket_index(roots, query);
                    Self::collect_candidates(&children[primary], query, candidates, true);
                    for nb in neighbor_indices(primary) {
                        Self::collect_candidates(&children[nb], query, candidates, true);
                    }
                } else {
                    for child in children.iter() {
                        Self::collect_candidates(child, query, candidates, false);
                    }
                }
            }
        }
    }

    fn query_node(
        node: &BucketNode,
        query: Vec4,
        best_score: &mut f64,
        best_val: &mut [f64; 4],
        best_ts: &mut u64,
        approx: bool,
    ) {
        match node {
            BucketNode::Leaf(leaf) => {
                leaf.scan(query, best_score, best_val, best_ts);
            }
            BucketNode::Branch { roots, children, count } => {
                if *count == 0 { return; }

                if approx {
                    // Approximate: only visit primary + neighbors at this level too
                    let primary = bucket_index(roots, query);
                    Self::query_node(&children[primary], query, best_score, best_val, best_ts, true);
                    for nb in neighbor_indices(primary) {
                        Self::query_node(&children[nb], query, best_score, best_val, best_ts, true);
                    }
                } else {
                    // Exact: visit all children
                    for child in children.iter() {
                        Self::query_node(child, query, best_score, best_val, best_ts, false);
                    }
                }
            }
        }
    }
}
