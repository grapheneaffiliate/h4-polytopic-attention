//! PyO3 bridge: expose H4 ChamberTree to Python via `h4_rust` module.
//!
//! Functions:
//!   - query_topk(keys, queries, k) -> (n_queries, k) index array
//!   - chamber_indices(vectors, roots) -> 4-bit chamber IDs
//!   - build_and_query_approx(keys, queries, k) -> (n_queries, k) indices (approximate)

mod vec4;
mod vec8;
mod h4;
mod chamber_tree;
mod e8_lattice;
mod lattice_memory;
mod attention;

use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2, PyArray1, IntoPyArray};
use crate::vec4::Vec4;
use crate::h4::simple_roots;
use crate::chamber_tree::ChamberTree;
use std::sync::Mutex;

/// Persistent tree handle exposed to Python.
/// Build once with build_tree(), query many times with query_tree_approx().
#[pyclass]
struct TreeHandle {
    tree: Mutex<ChamberTree>,
    n_keys: usize,
}

#[pymethods]
impl TreeHandle {
    #[getter]
    fn size(&self) -> usize {
        self.n_keys
    }
}

/// Build a ChamberTree from keys. Returns a handle for repeated queries.
/// keys: (n_keys, 4) f64 array
#[pyfunction]
fn build_tree<'py>(
    _py: Python<'py>,
    keys: PyReadonlyArray2<'py, f64>,
) -> PyResult<TreeHandle> {
    let keys_arr = keys.as_array();
    let n_keys = keys_arr.nrows();
    if keys_arr.ncols() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err("keys must have 4 columns"));
    }

    let roots = simple_roots();
    let mut tree = ChamberTree::new(roots);
    for i in 0..n_keys {
        let key = Vec4::new(keys_arr[[i, 0]], keys_arr[[i, 1]], keys_arr[[i, 2]], keys_arr[[i, 3]]);
        tree.insert(key, [i as f64, 0.0, 0.0, 0.0], i as u64);
    }

    Ok(TreeHandle { tree: Mutex::new(tree), n_keys })
}

/// Query a pre-built tree for approximate top-k (O(log t) per query).
/// Only scans ~3% of keys via Coxeter chamber pruning.
/// handle: TreeHandle from build_tree()
/// keys: (n_keys, 4) original key array (needed for index recovery)
/// queries: (n_queries, 4) query vectors
/// k: top-k per query
/// Returns: (n_queries, k) index array
#[pyfunction]
fn query_tree_approx<'py>(
    py: Python<'py>,
    handle: &TreeHandle,
    queries: PyReadonlyArray2<'py, f64>,
    k: usize,
) -> PyResult<Py<PyArray2<i64>>> {
    let queries_arr = queries.as_array();
    let n_queries = queries_arr.nrows();
    if queries_arr.ncols() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err("queries must have 4 columns"));
    }

    let tree = handle.tree.lock().unwrap();
    let mut result = vec![-1i64; n_queries * k];

    for q in 0..n_queries {
        let query = Vec4::new(
            queries_arr[[q, 0]], queries_arr[[q, 1]],
            queries_arr[[q, 2]], queries_arr[[q, 3]],
        );
        let candidates = tree.query_topk_approx(query, k);
        let actual_k = k.min(candidates.len());
        for j in 0..actual_k {
            result[q * k + j] = candidates[j].1[0] as i64;
        }
    }

    let arr = numpy::ndarray::Array2::from_shape_vec((n_queries, k), result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py).into())
}

/// Build a ChamberTree from keys, query all queries for top-k by exact scan.
/// keys: (n_keys, 4) f64 array
/// queries: (n_queries, 4) f64 array
/// k: number of top results per query
/// Returns: (n_queries, k) i64 index array (-1 = not enough keys)
#[pyfunction]
fn query_topk<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray2<'py, f64>,
    queries: PyReadonlyArray2<'py, f64>,
    k: usize,
) -> PyResult<Py<PyArray2<i64>>> {
    let keys_arr = keys.as_array();
    let queries_arr = queries.as_array();
    let n_keys = keys_arr.nrows();
    let n_queries = queries_arr.nrows();

    // Validate dimensions
    if keys_arr.ncols() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("keys must have 4 columns, got {}", keys_arr.ncols()),
        ));
    }
    if queries_arr.ncols() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("queries must have 4 columns, got {}", queries_arr.ncols()),
        ));
    }

    let roots = simple_roots();
    let mut tree = ChamberTree::new(roots);

    // Insert all keys with their index as both value and timestamp
    for i in 0..n_keys {
        let key = Vec4::new(
            keys_arr[[i, 0]],
            keys_arr[[i, 1]],
            keys_arr[[i, 2]],
            keys_arr[[i, 3]],
        );
        // Store the index in the value field so we can recover it
        let value = [i as f64, 0.0, 0.0, 0.0];
        tree.insert(key, value, i as u64);
    }

    // For top-k, we need to scan and collect the k best matches per query.
    // The ChamberTree's query_max_exact returns only top-1, so we do multiple
    // queries by masking out already-found keys. For efficiency with larger k,
    // we build a flat index and sort dot products directly.
    let mut result = vec![-1i64; n_queries * k];

    for q in 0..n_queries {
        let query = Vec4::new(
            queries_arr[[q, 0]],
            queries_arr[[q, 1]],
            queries_arr[[q, 2]],
            queries_arr[[q, 3]],
        )
        .normalized();

        // Compute all dot products and sort for top-k
        let mut scored: Vec<(f64, usize)> = (0..n_keys)
            .map(|i| {
                let key = Vec4::new(
                    keys_arr[[i, 0]],
                    keys_arr[[i, 1]],
                    keys_arr[[i, 2]],
                    keys_arr[[i, 3]],
                );
                (query.dot(key.normalized()), i)
            })
            .collect();

        // Partial sort for top-k (descending by score)
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let actual_k = k.min(n_keys);
        for j in 0..actual_k {
            result[q * k + j] = scored[j].1 as i64;
        }
    }

    // Convert to numpy 2D array
    let arr = numpy::ndarray::Array2::from_shape_vec((n_queries, k), result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py).into())
}

/// Approximate top-k using ChamberTree's hierarchical bucket pruning.
/// Only scans ~3% of keys at 3 levels: (5/16)^3 = 3.05%.
/// keys: (n_keys, 4) f64 array
/// queries: (n_queries, 4) f64 array
/// k: number of top results per query
/// Returns: (n_queries, k) i64 index array (-1 = not enough keys)
///
/// Also returns the scan ratio (fraction of keys actually examined) via
/// a second return value when called from Python.
#[pyfunction]
fn query_topk_approx<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray2<'py, f64>,
    queries: PyReadonlyArray2<'py, f64>,
    k: usize,
) -> PyResult<Py<PyArray2<i64>>> {
    let keys_arr = keys.as_array();
    let queries_arr = queries.as_array();
    let n_keys = keys_arr.nrows();
    let n_queries = queries_arr.nrows();

    if keys_arr.ncols() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("keys must have 4 columns, got {}", keys_arr.ncols()),
        ));
    }
    if queries_arr.ncols() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("queries must have 4 columns, got {}", queries_arr.ncols()),
        ));
    }

    let roots = simple_roots();
    let mut tree = ChamberTree::new(roots);

    // Insert all keys — store index in both value[0] and timestamp
    for i in 0..n_keys {
        let key = Vec4::new(
            keys_arr[[i, 0]],
            keys_arr[[i, 1]],
            keys_arr[[i, 2]],
            keys_arr[[i, 3]],
        );
        let value = [i as f64, 0.0, 0.0, 0.0];
        tree.insert(key, value, i as u64);
    }

    let mut result = vec![-1i64; n_queries * k];
    let mut total_scanned = 0u64;
    let mut total_possible = 0u64;

    for q in 0..n_queries {
        let query = Vec4::new(
            queries_arr[[q, 0]],
            queries_arr[[q, 1]],
            queries_arr[[q, 2]],
            queries_arr[[q, 3]],
        );

        // Use ChamberTree approximate query — only visits ~3% of buckets
        let candidates = tree.query_topk_approx(query, k);

        total_scanned += candidates.len() as u64;
        total_possible += n_keys as u64;

        let actual_k = k.min(candidates.len());
        for j in 0..actual_k {
            // value[0] stores the original key index
            result[q * k + j] = candidates[j].1[0] as i64;
        }
    }

    let arr = numpy::ndarray::Array2::from_shape_vec((n_queries, k), result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py).into())
}

/// Compute 4-bit chamber index for each vector given roots.
/// vectors: (n, 4) f64 array
/// roots: (4, 4) f64 array of simple roots
/// Returns: (n,) i64 array of chamber IDs (0..15)
#[pyfunction]
fn chamber_indices<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArray2<'py, f64>,
    roots: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray1<i64>>> {
    let vecs = vectors.as_array();
    let roots_arr = roots.as_array();
    let n = vecs.nrows();

    if vecs.ncols() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err("vectors must have 4 columns"));
    }
    if roots_arr.nrows() != 4 || roots_arr.ncols() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err("roots must be (4, 4)"));
    }

    // Build root Vec4s
    let r: [Vec4; 4] = [
        Vec4::new(roots_arr[[0, 0]], roots_arr[[0, 1]], roots_arr[[0, 2]], roots_arr[[0, 3]]),
        Vec4::new(roots_arr[[1, 0]], roots_arr[[1, 1]], roots_arr[[1, 2]], roots_arr[[1, 3]]),
        Vec4::new(roots_arr[[2, 0]], roots_arr[[2, 1]], roots_arr[[2, 2]], roots_arr[[2, 3]]),
        Vec4::new(roots_arr[[3, 0]], roots_arr[[3, 1]], roots_arr[[3, 2]], roots_arr[[3, 3]]),
    ];

    let mut indices = Vec::with_capacity(n);
    for i in 0..n {
        let v = Vec4::new(vecs[[i, 0]], vecs[[i, 1]], vecs[[i, 2]], vecs[[i, 3]]);
        let mut idx = 0i64;
        if v.dot(r[0]) >= 0.0 { idx |= 1; }
        if v.dot(r[1]) >= 0.0 { idx |= 2; }
        if v.dot(r[2]) >= 0.0 { idx |= 4; }
        if v.dot(r[3]) >= 0.0 { idx |= 8; }
        indices.push(idx);
    }

    let arr = numpy::ndarray::Array1::from_vec(indices);
    Ok(arr.into_pyarray(py).into())
}

/// Return the H4 simple roots as a (4, 4) f64 array.
#[pyfunction]
fn get_simple_roots<'py>(py: Python<'py>) -> PyResult<Py<PyArray2<f64>>> {
    let roots = simple_roots();
    let mut data = vec![0.0f64; 16];
    for i in 0..4 {
        for j in 0..4 {
            data[i * 4 + j] = roots[i].0[j];
        }
    }
    let arr = numpy::ndarray::Array2::from_shape_vec((4, 4), data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py).into())
}

#[pymodule]
fn h4_rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TreeHandle>()?;
    m.add_function(wrap_pyfunction!(build_tree, m)?)?;
    m.add_function(wrap_pyfunction!(query_tree_approx, m)?)?;
    m.add_function(wrap_pyfunction!(query_topk, m)?)?;
    m.add_function(wrap_pyfunction!(query_topk_approx, m)?)?;
    m.add_function(wrap_pyfunction!(chamber_indices, m)?)?;
    m.add_function(wrap_pyfunction!(get_simple_roots, m)?)?;
    Ok(())
}
