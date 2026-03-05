// tests/vector_tests.rs — Unit tests for HNSW vector index and similarity functions.

#[cfg(test)]
mod tests {
    use crate::vector::similarity::*;

    #[test]
    fn test_cosine_similarity_with_real_embeddings() {
        // Simulate two similar 384-dim embeddings (just using 8-dim for test brevity)
        let a = vec![0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7];
        let b = vec![0.15, 0.48, 0.32, 0.79, 0.22, 0.58, 0.41, 0.68];

        let sim = cosine_similarity(&a, &b);
        assert!(sim > 0.99, "Similar vectors should have high cosine similarity");
    }

    #[test]
    fn test_cosine_distance_is_one_minus_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        let sim = cosine_similarity(&a, &b);
        let dist = cosine_distance(&a, &b);
        assert!((sim + dist - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_normalized_equals_cosine() {
        let a = vec![3.0, 4.0];
        let b = vec![1.0, 2.0];

        let norm_a = normalize(&a);
        let norm_b = normalize(&b);

        let dot = dot_product(&norm_a, &norm_b);
        let cos = cosine_similarity(&a, &b);

        assert!((dot - cos).abs() < 1e-5, "Dot product of normalized = cosine sim");
    }

    #[test]
    fn test_euclidean_distance_zero_for_same() {
        let v = vec![1.0, 2.0, 3.0];
        assert!(euclidean_distance(&v, &v).abs() < 1e-6);
    }
}
