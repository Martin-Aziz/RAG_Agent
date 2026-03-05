// graph/query.rs — Simple Cypher-like query DSL for pattern matching.
// Provides a structured query interface beyond raw BFS/DFS traversal.
//
// This is intentionally simpler than full Cypher — it covers the common
// query patterns needed for RAG retrieval without the complexity of a
// full query language parser.
//
// Example queries:
//   MATCH (n:CVE) WHERE n.name CONTAINS "Log4j" RETURN n LIMIT 10
//   MATCH (a:THREAT_ACTOR)-[r:EXPLOITS]->(v:CVE) WHERE v.id = "abc" RETURN a,r,v

use std::collections::HashMap;
use petgraph::stable_graph::StableGraph;
use petgraph::Directed;
use petgraph::visit::EdgeRef;
use serde::{Serialize, Deserialize};

use super::store::{GraphNode, GraphEdge};

// ============================================================================
// Query DSL types
// ============================================================================

/// A structured graph query that can be built programmatically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQuery {
    /// Node label filter (e.g., "CVE", "THREAT_ACTOR"). Empty = any label.
    pub node_label: Option<String>,
    /// Property filters as key-value conditions.
    pub property_filters: Vec<PropertyFilter>,
    /// Edge traversal pattern (optional).
    pub traversal: Option<TraversalPattern>,
    /// Maximum number of results to return.
    pub limit: usize,
    /// Offset for pagination.
    pub offset: usize,
}

/// A property filter condition applied to node properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyFilter {
    pub key: String,
    pub operator: FilterOperator,
    pub value: String,
}

/// Supported filter operators for property matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOperator {
    /// Exact string match (case-sensitive)
    Equals,
    /// Substring match (case-insensitive)
    Contains,
    /// Prefix match (case-insensitive)
    StartsWith,
    /// Suffix match (case-insensitive)
    EndsWith,
    /// Not equal
    NotEquals,
}

/// Defines a traversal pattern from matched nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalPattern {
    /// Direction of edge to follow
    pub direction: TraversalDirection,
    /// Relationship type filter (None = any)
    pub relation_type: Option<String>,
    /// Target node label filter (None = any)
    pub target_label: Option<String>,
}

/// Edge traversal direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraversalDirection {
    Outgoing,
    Incoming,
    Both,
}

/// Result of a query: matched nodes optionally with connected nodes/edges.
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub nodes: Vec<GraphNode>,
    pub connected: Vec<(GraphNode, GraphEdge, GraphNode)>, // (source, edge, target)
    pub total_matched: usize,
}

// ============================================================================
// Query Execution
// ============================================================================

/// Execute a structured graph query against the in-memory graph.
///
/// Steps:
/// 1. Scan all nodes matching label and property filters
/// 2. If traversal pattern is specified, follow edges from matched nodes
/// 3. Apply limit/offset for pagination
/// 4. Return matched nodes and optionally connected triplets
pub fn execute_query(
    graph: &StableGraph<GraphNode, GraphEdge, Directed>,
    query: &GraphQuery,
) -> QueryResult {
    let limit = if query.limit == 0 { 50 } else { query.limit.min(500) };

    // Step 1: Find all nodes matching the label and property filters
    let mut matched_indices = Vec::new();

    for idx in graph.node_indices() {
        if let Some(node) = graph.node_weight(idx) {
            // Check label filter
            if let Some(ref label) = query.node_label {
                if &node.label != label {
                    continue;
                }
            }

            // Check property filters — all must match (AND semantics)
            let mut all_match = true;
            for filter in &query.property_filters {
                let prop_value = node.properties.get(&filter.key)
                    .or_else(|| {
                        // Also check the name field as a pseudo-property
                        if filter.key == "name" {
                            Some(&node.name)
                        } else {
                            None
                        }
                    });

                let matches = match prop_value {
                    Some(val) => match_filter(val, &filter.operator, &filter.value),
                    None => false,
                };

                if !matches {
                    all_match = false;
                    break;
                }
            }

            if all_match {
                matched_indices.push(idx);
            }
        }
    }

    let total_matched = matched_indices.len();

    // Step 2: Apply offset and limit
    let paginated: Vec<_> = matched_indices
        .into_iter()
        .skip(query.offset)
        .take(limit)
        .collect();

    // Step 3: Collect matched nodes
    let nodes: Vec<GraphNode> = paginated
        .iter()
        .filter_map(|idx| graph.node_weight(*idx).cloned())
        .collect();

    // Step 4: If traversal pattern is specified, follow edges
    let mut connected = Vec::new();

    if let Some(ref pattern) = query.traversal {
        for &node_idx in &paginated {
            let source_node = match graph.node_weight(node_idx) {
                Some(n) => n.clone(),
                None => continue,
            };

            // Follow edges based on direction
            let edges_to_check: Vec<_> = match pattern.direction {
                TraversalDirection::Outgoing => {
                    graph.edges(node_idx).collect()
                }
                TraversalDirection::Incoming => {
                    graph.edges_directed(node_idx, petgraph::Direction::Incoming).collect()
                }
                TraversalDirection::Both => {
                    let mut edges: Vec<_> = graph.edges(node_idx).collect();
                    edges.extend(graph.edges_directed(node_idx, petgraph::Direction::Incoming));
                    edges
                }
            };

            for edge_ref in edges_to_check {
                let edge = edge_ref.weight();

                // Apply relation type filter
                if let Some(ref rel_type) = pattern.relation_type {
                    if &edge.relation_type != rel_type {
                        continue;
                    }
                }

                // Get the other node (target for outgoing, source for incoming)
                let other_idx = if edge_ref.source() == node_idx {
                    edge_ref.target()
                } else {
                    edge_ref.source()
                };

                if let Some(other_node) = graph.node_weight(other_idx) {
                    // Apply target label filter
                    if let Some(ref target_label) = pattern.target_label {
                        if &other_node.label != target_label {
                            continue;
                        }
                    }

                    // Determine which is source and which is target for the triplet
                    if edge_ref.source() == node_idx {
                        connected.push((
                            source_node.clone(),
                            edge.clone(),
                            other_node.clone(),
                        ));
                    } else {
                        connected.push((
                            other_node.clone(),
                            edge.clone(),
                            source_node.clone(),
                        ));
                    }
                }
            }
        }
    }

    QueryResult {
        nodes,
        connected,
        total_matched,
    }
}

/// Evaluate a single property filter condition.
fn match_filter(value: &str, operator: &FilterOperator, filter_value: &str) -> bool {
    match operator {
        FilterOperator::Equals => value == filter_value,
        FilterOperator::NotEquals => value != filter_value,
        FilterOperator::Contains => {
            value.to_lowercase().contains(&filter_value.to_lowercase())
        }
        FilterOperator::StartsWith => {
            value.to_lowercase().starts_with(&filter_value.to_lowercase())
        }
        FilterOperator::EndsWith => {
            value.to_lowercase().ends_with(&filter_value.to_lowercase())
        }
    }
}

// ============================================================================
// Query Builder (fluent API)
// ============================================================================

impl GraphQuery {
    /// Create a new query with default settings.
    pub fn new() -> Self {
        Self {
            node_label: None,
            property_filters: Vec::new(),
            traversal: None,
            limit: 50,
            offset: 0,
        }
    }

    /// Filter by node label (entity type).
    pub fn with_label(mut self, label: &str) -> Self {
        self.node_label = Some(label.to_string());
        self
    }

    /// Add a property equals filter.
    pub fn where_eq(mut self, key: &str, value: &str) -> Self {
        self.property_filters.push(PropertyFilter {
            key: key.to_string(),
            operator: FilterOperator::Equals,
            value: value.to_string(),
        });
        self
    }

    /// Add a property contains filter.
    pub fn where_contains(mut self, key: &str, value: &str) -> Self {
        self.property_filters.push(PropertyFilter {
            key: key.to_string(),
            operator: FilterOperator::Contains,
            value: value.to_string(),
        });
        self
    }

    /// Add a traversal pattern.
    pub fn traverse(mut self, direction: TraversalDirection, relation: Option<&str>, target_label: Option<&str>) -> Self {
        self.traversal = Some(TraversalPattern {
            direction,
            relation_type: relation.map(|s| s.to_string()),
            target_label: target_label.map(|s| s.to_string()),
        });
        self
    }

    /// Set result limit.
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set offset for pagination.
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }
}
