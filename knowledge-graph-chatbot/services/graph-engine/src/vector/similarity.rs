// vector/similarity.rs — Vector similarity functions.
// Provides cosine similarity and dot-product computations for embedding comparison.
// Used by both the HNSW index (internally) and the hybrid search re-ranker.
//
// These functions operate on raw f32 slices for zero-copy performance.
// SIMD vectorization is left to the compiler via opt-level=3 in release builds.

/// Compute cosine similarity between two vectors.
///
/// Returns a value in [-1.0, 1.0] where 1.0 = identical direction.
/// Returns 0.0 if either vector has zero magnitude (degenerate case).
///
/// Formula: cos(θ) = (A · B) / (||A|| × ||B||)
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    // Single-pass computation: dot product and both norms simultaneously
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denominator = norm_a.sqrt() * norm_b.sqrt();

    if denominator < f32::EPSILON {
        // Avoid division by zero for zero-magnitude vectors
        return 0.0;
    }

    dot / denominator
}

/// Compute cosine distance between two vectors.
///
/// Returns a value in [0.0, 2.0] where 0.0 = identical direction.
/// This is used as the distance metric for HNSW nearest neighbor search.
///
/// Formula: distance = 1.0 - cosine_similarity(A, B)
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

/// Compute dot product between two vectors.
///
/// Returns the raw inner product without normalization.
/// Useful when vectors are already L2-normalized (equivalent to cosine similarity).
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}

/// L2 (Euclidean) distance between two vectors.
///
/// Returns sqrt(Σ(a_i - b_i)²). Used as a fallback distance metric.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Normalize a vector to unit length (L2 normalization).
///
/// After normalization, dot_product equals cosine_similarity.
/// Returns a new vector; does not mutate the input.
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm < f32::EPSILON {
        return vec![0.0; v.len()];
    }

    v.iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        let v = vec![1.0, 2.0, 3.0];
        let dist = cosine_distance(&v, &v);
        assert!(dist.abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_zero_vector() {
        let zero = vec![0.0, 0.0, 0.0];
        let other = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&zero, &other), 0.0);
    }
}
