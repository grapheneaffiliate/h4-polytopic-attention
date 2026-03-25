mod vec4;
mod vec8;
mod h4;
mod chamber_tree;
mod e8_lattice;
mod lattice_memory;
mod attention;

use std::time::Instant;
use vec4::{PHI, PHI_INV};
use attention::H4Attention;
use lattice_memory::LatticeAttention;

/// Simple xorshift64 PRNG.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self { Rng(seed) }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64 / u64::MAX as f64) * 2.0 - 1.0
    }

    fn random_vec(&mut self, d: usize) -> Vec<f64> {
        (0..d).map(|_| self.next_f64()).collect()
    }

    /// Generate a structured embedding that simulates Wasm execution traces.
    /// Keys cluster around a few "instruction pointer" directions with
    /// phi-scaled perturbations, mimicking real execution patterns.
    fn structured_vec(&mut self, d: usize, step: usize) -> Vec<f64> {
        let mut v = vec![0.0f64; d];
        // Simulate ~8 distinct instruction classes
        let instr_class = step % 8;
        // Base direction rotates slowly (like an instruction pointer)
        let base_angle = (step as f64) * 0.01;
        // Phase encodes the operation type
        let phase = (instr_class as f64) * std::f64::consts::PI / 4.0;

        for h in 0..(d / 4) {
            let head_offset = h as f64 * 0.7;
            // Structured: direction from instruction class + slow drift
            v[h * 4]     = (base_angle + head_offset).cos() * (phase).cos();
            v[h * 4 + 1] = (base_angle + head_offset).sin() * (phase).cos();
            v[h * 4 + 2] = (base_angle * PHI + head_offset).cos() * (phase).sin();
            v[h * 4 + 3] = (base_angle * PHI + head_offset).sin() * (phase).sin();
            // Small perturbation (register state variation)
            for k in 0..4 {
                v[h * 4 + k] += self.next_f64() * 0.1;
            }
        }
        v
    }
}

fn verify_h4() -> bool {
    println!("Golden ratio phi  = {:.15}", PHI);
    println!("1/phi = phi - 1   = {:.15}", PHI_INV);
    println!("phi + 1/phi       = {:.15} (sqrt(5) = {:.15})", PHI + PHI_INV, 5.0_f64.sqrt());
    println!();

    let vertices = h4::generate_600_cell();
    println!("600-cell vertices: {} (expected: 120)", vertices.len());

    if !h4::verify_600_cell(&vertices) {
        println!("600-cell verification: FAILED");
        return false;
    }
    println!("600-cell verification: PASSED");

    let mut dots: Vec<i64> = Vec::new();
    for i in 0..vertices.len() {
        for j in (i+1)..vertices.len() {
            let d = vertices[i].dot(vertices[j]);
            let d_rounded = (d * 1_000_000.0).round() as i64;
            if !dots.contains(&d_rounded) {
                dots.push(d_rounded);
            }
        }
    }
    println!("Unique dot products: {} (expected: ~8 for H4)", dots.len());

    let has_phi_half = dots.iter().any(|&d| {
        (d as f64 / 1_000_000.0 - PHI / 2.0).abs() < 0.01
    });
    println!("phi/2 in dot products: {}", if has_phi_half { "yes" } else { "no" });
    true
}

struct BenchResult {
    name: String,
    total_s: f64,
    rate: f64,
    n_steps: usize,
}

fn run_benchmark(
    name: &str,
    n_steps: usize,
    d_model: usize,
    n_layers: usize,
    embeddings: &[Vec<f64>],
    mode: &str, // "exact", "approx", "exact_par", "approx_par"
) -> BenchResult {
    let mut layers: Vec<H4Attention> = (0..n_layers)
        .map(|_| H4Attention::new(d_model))
        .collect();

    println!("\n--- {} ({} steps) ---", name, n_steps);
    let start = Instant::now();
    let mut last_report = start;

    for (i, emb) in embeddings.iter().take(n_steps).enumerate() {
        for layer in &mut layers {
            match mode {
                "exact"      => { let _ = layer.query_exact(emb); },
                "approx"     => { let _ = layer.query_approx(emb); },
                "exact_par"  => { let _ = layer.query_exact_par(emb); },
                "approx_par" => { let _ = layer.query_approx_par(emb); },
                _ => unreachable!(),
            }
            layer.insert(emb);
        }

        let now = Instant::now();
        if now.duration_since(last_report).as_secs() >= 3 || i == n_steps - 1 {
            let elapsed = now.duration_since(start).as_secs_f64();
            let rate = (i + 1) as f64 / elapsed;
            println!("  Step {}/{}: {:.0} steps/s (cache: {})", i+1, n_steps, rate, layers[0].cache_size());
            last_report = now;
        }
    }

    let total = start.elapsed().as_secs_f64();
    let rate = n_steps as f64 / total;
    println!("  => {:.0} steps/s avg", rate);

    BenchResult { name: name.to_string(), total_s: total, rate, n_steps }
}

fn main() {
    println!("H4 Polytopic Attention -- Rust Implementation");
    println!("=============================================\n");

    if !verify_h4() { return; }

    let d_model = 72;
    let n_layers = 3;
    let n_steps = 50_000;

    println!("\n===============================================");
    println!("  BENCHMARK SUITE");
    println!("  d_model={}, n_heads={}, n_layers={}", d_model, d_model/4, n_layers);
    println!("===============================================");

    let mut rng = Rng::new(42);

    // Generate random embeddings
    let random_embs: Vec<Vec<f64>> = (0..n_steps).map(|_| rng.random_vec(d_model)).collect();

    // Generate structured Wasm-like embeddings
    let mut rng2 = Rng::new(123);
    let structured_embs: Vec<Vec<f64>> = (0..n_steps)
        .map(|i| rng2.structured_vec(d_model, i))
        .collect();

    let mut results = Vec::new();

    // 1. Random keys, exact scan (baseline)
    results.push(run_benchmark(
        "Random keys, exact (all 16 buckets)",
        n_steps, d_model, n_layers, &random_embs, "exact",
    ));

    // 2. Random keys, approximate (primary + 4 neighbors)
    results.push(run_benchmark(
        "Random keys, approx (5/16 buckets)",
        n_steps, d_model, n_layers, &random_embs, "approx",
    ));

    // 3. Random keys, parallel exact
    results.push(run_benchmark(
        "Random keys, exact + rayon parallel",
        n_steps, d_model, n_layers, &random_embs, "exact_par",
    ));

    // 4. Random keys, parallel approximate
    results.push(run_benchmark(
        "Random keys, approx + rayon parallel",
        n_steps, d_model, n_layers, &random_embs, "approx_par",
    ));

    // 5. Structured keys, exact
    results.push(run_benchmark(
        "Structured (Wasm-like), exact",
        n_steps, d_model, n_layers, &structured_embs, "exact",
    ));

    // 6. Structured keys, approximate
    results.push(run_benchmark(
        "Structured (Wasm-like), approx",
        n_steps, d_model, n_layers, &structured_embs, "approx",
    ));

    // 7. Structured keys, parallel approximate
    results.push(run_benchmark(
        "Structured (Wasm-like), approx + rayon parallel",
        n_steps, d_model, n_layers, &structured_embs, "approx_par",
    ));

    // Summary
    println!("\n===============================================");
    println!("  RESULTS SUMMARY ({} steps each)", n_steps);
    println!("===============================================");
    println!("{:<48} {:>10} {:>8}", "Benchmark", "steps/s", "vs Python");
    println!("{}", "-".repeat(70));
    for r in &results {
        println!("{:<48} {:>10.0} {:>7.0}x", r.name, r.rate, r.rate / 34.0);
    }
    println!("{}", "-".repeat(70));
    println!("Python PoC baseline: ~34 steps/s");

    // Theoretical speedup
    let linear_work = (n_steps as f64) * (n_steps as f64 + 1.0) / 2.0;
    let hull_work: f64 = (1..=n_steps).map(|t| (t as f64).log2().max(1.0)).sum();
    println!("Theoretical O(log t) speedup vs O(t): {:.0}x at {} steps", linear_work / hull_work, n_steps);
    println!("===============================================");

    // ========================================================
    // Phase 4: E₈ Lattice Memory Benchmarks
    // ========================================================
    println!("\n===============================================");
    println!("  PHASE 4: E₈ LATTICE MEMORY");
    println!("===============================================");

    // Verify E₈ lattice
    let kissing = e8_lattice::kissing_vectors();
    println!("E₈ kissing vectors: {} (expected: 240)", kissing.len());
    let all_norm_2: bool = kissing.iter().all(|v| {
        let n: i32 = v.coords.iter().map(|c| c * c).sum();
        n == 8 // ×2 representation: actual norm² = 8/4 = 2
    });
    println!("All kissing vectors norm² = 2: {}", if all_norm_2 { "PASS" } else { "FAIL" });

    // Verify E₈ → H₄ projection
    let proj = vec8::E8H4Projection::new();
    println!("E₈→H₄ projection matrix:");
    println!("  cos(π/5) = φ/2 = {:.6} (φ/2 = {:.6})", proj.rows[0][0], PHI / 2.0);
    println!("  cos(2π/5) = 1/(2φ) = {:.6} (1/(2φ) = {:.6})", proj.rows[0][2], PHI_INV / 2.0);

    // Benchmark: Lattice memory store + load
    let lattice_steps = 10_000;
    println!("\n--- E₈ Lattice Memory: store + load ({} steps) ---", lattice_steps);
    let mut lattice_attn = LatticeAttention::new(d_model);
    let mut rng3 = Rng::new(777);

    let start = Instant::now();
    for i in 0..lattice_steps {
        let emb = rng3.structured_vec(d_model, i);
        lattice_attn.insert(&emb);
    }
    let store_time = start.elapsed().as_secs_f64();
    let store_rate = lattice_steps as f64 / store_time;

    // Load benchmark: query each embedding
    let mut rng4 = Rng::new(777); // Same seed to query same embeddings
    let start = Instant::now();
    let mut hits = 0u64;
    for i in 0..lattice_steps {
        let emb = rng4.structured_vec(d_model, i);
        let q = [emb[0], emb[1], emb[2], emb[3], emb[4], emb[5], emb[6], emb[7]];
        if lattice_attn.load_mem(q).is_some() {
            hits += 1;
        }
    }
    let load_time = start.elapsed().as_secs_f64();
    let load_rate = lattice_steps as f64 / load_time;

    // Unified 8D→4D query benchmark
    let start = Instant::now();
    let mut unified_hits = 0u64;
    let mut rng5 = Rng::new(777);
    for i in 0..lattice_steps {
        let emb = rng5.structured_vec(d_model, i);
        let q = [emb[0], emb[1], emb[2], emb[3], emb[4], emb[5], emb[6], emb[7]];
        if lattice_attn.query_unified(q).is_some() {
            unified_hits += 1;
        }
    }
    let unified_time = start.elapsed().as_secs_f64();
    let unified_rate = lattice_steps as f64 / unified_time;

    let stats = lattice_attn.memory_stats();
    println!("  Store: {:.0} ops/s", store_rate);
    println!("  Load (E₈ Voronoi):   {:.0} ops/s, {}/{} hits ({:.1}%)",
        load_rate, hits, lattice_steps, hits as f64 / lattice_steps as f64 * 100.0);
    println!("  Query (E₈→H₄ unified): {:.0} ops/s, {}/{} hits ({:.1}%)",
        unified_rate, unified_hits, lattice_steps, unified_hits as f64 / lattice_steps as f64 * 100.0);

    println!("\n  Lattice Memory Stats:");
    println!("    Total entries:      {}", stats.total_entries);
    println!("    Occupied cells:     {}", stats.occupied_cells);
    println!("    Utilization:        {:.1}%", lattice_attn.memory.utilization() * 100.0);
    println!("    Max bucket size:    {}", stats.max_bucket_size);
    println!("    Avg bucket size:    {:.1}", stats.avg_bucket_size);
    println!("    Primary cell hits:  {} ({:.1}% of reads)",
        stats.primary_cell_hits,
        stats.primary_cell_hits as f64 / stats.total_reads.max(1) as f64 * 100.0);
    println!("    Chamber cache (4D): {}", lattice_attn.memory.chamber_cache_size());
    println!("    H₄ attn cache (4D): {}", lattice_attn.cache_size());

    println!("\n===============================================");
    println!("  PHASE 4 COMPLETE");
    println!("  E₈ (8D) → H₄ (4D) unified memory+attention");
    println!("  Voronoi cells: O(1) address decode");
    println!("  Neighbor shells: 240 kissing vectors");
    println!("  Projection: cos(π/5) = φ/2 eigenvalues");
    println!("===============================================");
}
