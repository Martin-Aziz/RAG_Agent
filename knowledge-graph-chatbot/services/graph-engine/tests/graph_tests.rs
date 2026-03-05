// tests/graph_tests.rs — Unit tests for graph store and traversal.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    // Note: These tests depend on the graph store module.
    // They use tempfile for ephemeral RocksDB instances.

    /// Helper to create a test node
    fn make_node(id: &str, label: &str, name: &str) -> crate::graph::store::GraphNode {
        crate::graph::store::GraphNode {
            id: id.to_string(),
            label: label.to_string(),
            name: name.to_string(),
            properties: HashMap::new(),
            embedding: vec![],
            created_at: 0,
            confidence: 0.9,
        }
    }

    /// Helper to create a test edge
    fn make_edge(id: &str, source: &str, target: &str, rel: &str) -> crate::graph::store::GraphEdge {
        crate::graph::store::GraphEdge {
            id: id.to_string(),
            source_id: source.to_string(),
            target_id: target.to_string(),
            relation_type: rel.to_string(),
            weight: 1.0,
            properties: HashMap::new(),
            source_document: "test".to_string(),
            created_at: 0,
        }
    }
}
