//! E₈ Lattice-Indexed RAM: hierarchical memory backend for the H₄ executor.
//!
//! Memory operations (read/write) use E₈ lattice decoding to bucket-sort
//! entries into Voronoi cells. This gives O(1) approximate nearest-neighbor
//! lookup for any memory address, replacing linear scans.
//!
//! Architecture:
//!   Store: 8D embedding → decode to E₈ lattice point → bucket insert
//!   Load:  8D query → decode → primary cell + neighbor shells → best match
//!   Attn:  8D embedding → project to 4D via E₈→H₄ → ChamberTree query
//!
//! The E₈→H₄ projection ensures memory access and state attention share
//! the same φ-structured geometry — unified through the cos(π/5) eigenvalues
//! of the Coxeter element.

use std::collections::HashMap;
use crate::vec4::Vec4;
use crate::vec8::{Vec8, E8H4Projection};
use crate::e8_lattice::{self, LatticePoint, decode_to_e8, kissing_vectors, lattice_add};
use crate::chamber_tree::ChamberTree;
use crate::h4;

/// A single memory cell within a Voronoi bucket.
#[derive(Clone)]
struct MemoryEntry {
    /// Full 8D embedding (stored for precise distance computation).
    key: Vec8,
    /// 4D value (same dimensionality as attention values).
    value: [f64; 4],
    /// Write timestamp for recency tracking.
    timestamp: u64,
    /// Linear address (for Wasm memory compatibility).
    address: u64,
}

/// Statistics for monitoring lattice memory utilization.
#[derive(Clone, Debug, Default)]
pub struct LatticeMemoryStats {
    pub total_entries: u64,
    pub occupied_cells: u64,
    pub total_reads: u64,
    pub total_writes: u64,
    pub neighbor_shell_queries: u64,
    pub primary_cell_hits: u64,
    pub max_bucket_size: usize,
    pub avg_bucket_size: f64,
}

/// E₈ lattice-indexed RAM.
///
/// Each Voronoi cell of the E₈ lattice is a memory bucket.
/// The 240 kissing vectors define the neighbor shell for approximate queries.
/// The E₈→H₄ projection connects this 8D memory to 4D attention.
pub struct LatticeMemory {
    /// Voronoi cell buckets: E₈ lattice point → entries.
    cells: HashMap<[i32; 8], Vec<MemoryEntry>>,
    /// Precomputed 240 kissing vectors for neighbor traversal.
    kissing: Vec<LatticePoint>,
    /// E₈ → H₄ projection matrix.
    projection: E8H4Projection,
    /// H₄ ChamberTree for projected 4D attention queries.
    chamber_tree: ChamberTree,
    /// Global timestamp counter.
    timestamp: u64,
    /// Running statistics.
    stats: LatticeMemoryStats,
    /// Maximum entries per cell before we stop inserting into that cell.
    /// The kissing number (240) is a natural bound.
    max_cell_size: usize,
}

impl LatticeMemory {
    pub fn new() -> Self {
        let kissing = kissing_vectors();
        let projection = E8H4Projection::new();
        let roots = h4::simple_roots();
        let chamber_tree = ChamberTree::new(roots);

        LatticeMemory {
            cells: HashMap::new(),
            kissing,
            projection,
            chamber_tree,
            timestamp: 0,
            stats: LatticeMemoryStats::default(),
            max_cell_size: 240, // Kissing number bound
        }
    }

    /// Write a value to lattice memory.
    ///
    /// The 8D embedding is decoded to its E₈ Voronoi cell and stored there.
    /// The 4D projection is also inserted into the ChamberTree for
    /// attention-compatible queries.
    pub fn store(&mut self, embedding: Vec8, value: [f64; 4], address: u64) {
        let ts = self.timestamp;
        self.timestamp += 1;
        self.stats.total_writes += 1;

        // Decode to E₈ lattice point
        let lattice_pt = decode_to_e8(embedding);
        let key = lattice_pt.coords;

        // Insert into Voronoi cell bucket
        let bucket = self.cells.entry(key).or_insert_with(Vec::new);
        if bucket.len() < self.max_cell_size {
            bucket.push(MemoryEntry {
                key: embedding,
                value,
                timestamp: ts,
                address,
            });
        } else {
            // Evict oldest entry (LRU within cell)
            let oldest_idx = bucket.iter().enumerate()
                .min_by_key(|(_, e)| e.timestamp)
                .map(|(i, _)| i).unwrap();
            bucket[oldest_idx] = MemoryEntry {
                key: embedding,
                value,
                timestamp: ts,
                address,
            };
        }

        // Project to 4D and insert into ChamberTree for attention
        let projected = embedding.project_to_h4(&self.projection);
        self.chamber_tree.insert(projected, value, ts);

        self.stats.total_entries = self.cells.values().map(|b| b.len() as u64).sum();
        self.stats.occupied_cells = self.cells.len() as u64;
    }

    /// Load from lattice memory: find the best match for a query embedding.
    ///
    /// Strategy:
    /// 1. Decode query to E₈ lattice point (primary cell)
    /// 2. Search primary cell for exact/nearest match
    /// 3. If needed, search 240 neighbor cells (one kissing shell)
    ///
    /// Returns (value, address, distance², timestamp).
    pub fn load(&mut self, query: Vec8) -> Option<(f64, [f64; 4], u64, u64)> {
        self.stats.total_reads += 1;

        let primary = decode_to_e8(query);

        // Search primary cell first
        let mut best: Option<(f64, [f64; 4], u64, u64)> = None;

        if let Some(bucket) = self.cells.get(&primary.coords) {
            for entry in bucket {
                let dist = query.dist_sq(entry.key);
                if best.is_none() || dist < best.unwrap().0 {
                    best = Some((dist, entry.value, entry.address, entry.timestamp));
                }
            }
            if best.is_some() {
                self.stats.primary_cell_hits += 1;
            }
        }

        // Search neighbor shell (240 kissing vectors)
        self.stats.neighbor_shell_queries += 1;
        for kv in &self.kissing {
            let neighbor = lattice_add(primary, *kv);
            if let Some(bucket) = self.cells.get(&neighbor.coords) {
                for entry in bucket {
                    let dist = query.dist_sq(entry.key);
                    if best.is_none() || dist < best.unwrap().0 {
                        best = Some((dist, entry.value, entry.address, entry.timestamp));
                    }
                }
            }
        }

        best
    }

    /// Load by linear address (exact match).
    /// Falls back to linear scan over all cells — use sparingly.
    pub fn load_by_address(&self, address: u64) -> Option<([f64; 4], u64)> {
        for bucket in self.cells.values() {
            for entry in bucket {
                if entry.address == address {
                    return Some((entry.value, entry.timestamp));
                }
            }
        }
        None
    }

    /// Query via H₄ attention (4D projected space).
    ///
    /// Projects the 8D query to 4D and queries the ChamberTree.
    /// This gives O(log t) approximate max-dot-product search
    /// in the same geometric space as the attention heads.
    pub fn query_attention_exact(&self, query_8d: Vec8) -> Option<(f64, [f64; 4], u64)> {
        let q4 = query_8d.project_to_h4(&self.projection);
        self.chamber_tree.query_max_exact(q4)
    }

    /// Approximate attention query (5/16 bucket scan per level).
    pub fn query_attention_approx(&self, query_8d: Vec8) -> Option<(f64, [f64; 4], u64)> {
        let q4 = query_8d.project_to_h4(&self.projection);
        self.chamber_tree.query_max_approx(q4)
    }

    /// Project an 8D embedding to 4D H₄ space.
    pub fn project(&self, embedding: Vec8) -> Vec4 {
        embedding.project_to_h4(&self.projection)
    }

    /// Get current utilization statistics.
    pub fn stats(&self) -> LatticeMemoryStats {
        let mut s = self.stats.clone();
        if !self.cells.is_empty() {
            s.max_bucket_size = self.cells.values().map(|b| b.len()).max().unwrap_or(0);
            s.avg_bucket_size = s.total_entries as f64 / s.occupied_cells.max(1) as f64;
        }
        s
    }

    /// Number of entries in the lattice memory.
    pub fn size(&self) -> u64 {
        self.stats.total_entries
    }

    /// Number of occupied Voronoi cells.
    pub fn occupied_cells(&self) -> u64 {
        self.stats.occupied_cells
    }

    /// Bucket utilization: fraction of occupied cells vs total entries.
    /// Higher is better — means entries are well-distributed across cells.
    pub fn utilization(&self) -> f64 {
        if self.stats.total_entries == 0 {
            return 0.0;
        }
        self.stats.occupied_cells as f64 / self.stats.total_entries as f64
    }

    /// ChamberTree cache size (4D projected entries).
    pub fn chamber_cache_size(&self) -> u64 {
        self.chamber_tree.size
    }
}

/// Unified H₄ attention layer with E₈ lattice memory backing.
///
/// This is the Phase 4 integration point: attention heads operate in 4D
/// (via ChamberTree), but memory is stored and addressed in 8D (via E₈ lattice).
/// The E₈→H₄ projection unifies both.
pub struct LatticeAttention {
    /// Per-head attention caches (4D, H₄ ChamberTree).
    heads: Vec<ChamberTree>,
    /// Shared lattice memory (8D, E₈ Voronoi cells).
    pub memory: LatticeMemory,
    /// Number of 4D attention heads.
    n_heads: usize,
    /// Global step counter.
    step: u64,
}

impl LatticeAttention {
    pub fn new(d_model: usize) -> Self {
        assert!(d_model % 4 == 0);
        let n_heads = d_model / 4;
        let roots = h4::simple_roots();
        let heads = (0..n_heads).map(|_| ChamberTree::new(roots)).collect();

        LatticeAttention {
            heads,
            memory: LatticeMemory::new(),
            n_heads,
            step: 0,
        }
    }

    /// Insert a full embedding: 4D chunks go to per-head ChamberTrees,
    /// and the first 8 dimensions go to E₈ lattice memory.
    pub fn insert(&mut self, embedding: &[f64]) {
        let ts = self.step;
        self.step += 1;

        // Per-head 4D insertion (existing attention mechanism)
        for h in 0..self.n_heads {
            let o = h * 4;
            let key = Vec4::new(embedding[o], embedding[o+1], embedding[o+2], embedding[o+3]);
            let value = [embedding[o], embedding[o+1], embedding[o+2], embedding[o+3]];
            self.heads[h].insert(key, value, ts);
        }

        // E₈ lattice memory insertion (first 8 dims = first 2 heads)
        if embedding.len() >= 8 {
            let e8_key = Vec8::new([
                embedding[0], embedding[1], embedding[2], embedding[3],
                embedding[4], embedding[5], embedding[6], embedding[7],
            ]);
            let value = if embedding.len() >= 12 {
                [embedding[8], embedding[9], embedding[10], embedding[11]]
            } else {
                [embedding[0], embedding[1], embedding[2], embedding[3]]
            };
            self.memory.store(e8_key, value, ts);
        }
    }

    /// Store to lattice memory at a specific address (for STORE_MEM instruction).
    pub fn store_mem(&mut self, embedding_8d: [f64; 8], value: [f64; 4], address: u64) {
        self.memory.store(Vec8::new(embedding_8d), value, address);
    }

    /// Load from lattice memory by embedding similarity (for LOAD_MEM instruction).
    pub fn load_mem(&mut self, query_8d: [f64; 8]) -> Option<(f64, [f64; 4], u64, u64)> {
        self.memory.load(Vec8::new(query_8d))
    }

    /// Load from lattice memory by linear address.
    pub fn load_mem_addr(&self, address: u64) -> Option<([f64; 4], u64)> {
        self.memory.load_by_address(address)
    }

    /// Query attention heads (4D, exact).
    pub fn query_exact(&self, embedding: &[f64]) -> Vec<Option<(f64, u64)>> {
        use crate::vec4::PHI;
        (0..self.n_heads).map(|h| {
            let o = h * 4;
            let q = Vec4::new(
                embedding[o] * PHI, embedding[o+1] * PHI,
                embedding[o+2] * PHI, embedding[o+3] * PHI,
            );
            self.heads[h].query_max_exact(q).map(|(s, _, t)| (s, t))
        }).collect()
    }

    /// Query attention heads (4D, approximate).
    pub fn query_approx(&self, embedding: &[f64]) -> Vec<Option<(f64, u64)>> {
        use crate::vec4::PHI;
        (0..self.n_heads).map(|h| {
            let o = h * 4;
            let q = Vec4::new(
                embedding[o] * PHI, embedding[o+1] * PHI,
                embedding[o+2] * PHI, embedding[o+3] * PHI,
            );
            self.heads[h].query_max_approx(q).map(|(s, _, t)| (s, t))
        }).collect()
    }

    /// Query via unified E₈→H₄ path: 8D query projects to 4D attention.
    pub fn query_unified(&self, query_8d: [f64; 8]) -> Option<(f64, [f64; 4], u64)> {
        self.memory.query_attention_approx(Vec8::new(query_8d))
    }

    pub fn cache_size(&self) -> u64 {
        self.heads.first().map(|h| h.size).unwrap_or(0)
    }

    pub fn memory_stats(&self) -> LatticeMemoryStats {
        self.memory.stats()
    }
}
