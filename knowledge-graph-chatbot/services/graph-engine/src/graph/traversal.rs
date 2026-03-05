// graph/traversal.rs — Graph traversal algorithms: BFS, DFS, shortest path.
// Operates on petgraph StableGraph with configurable depth limits and edge filters.
//
// Performance target: 2-hop traversal on 50 nodes < 10ms.
// All functions are pure (no side effects), operating on immutable graph references.

use std::collections::{HashSet, VecDeque, HashMap};
use petgraph::stable_graph::{NodeIndex, EdgeIndex, StableGraph};
use petgraph::Directed;
use petgraph::visit::EdgeRef;

use super::store::{GraphNode, GraphEdge};

/// BFS subgraph extraction from seed nodes.
///
/// Returns (visited_node_indices, visited_edge_indices, total_nodes_visited).
/// Respects max_hops depth limit, optional relation_filter, and max_nodes cap.
///
/// The relation_filter, when provided, acts as a whitelist: only edges whose
/// relation_type is in the filter are traversed. If None, all edges are followed.
pub fn bfs_subgraph(
    graph: &StableGraph<GraphNode, GraphEdge, Directed>,
    seed_indices: &[NodeIndex],
    max_hops: u32,
    relation_filter: Option<&[String]>,
    max_nodes: usize,
) -> (Vec<NodeIndex>, Vec<EdgeIndex>, usize) {
    let mut visited_nodes: HashSet<NodeIndex> = HashSet::new();
    let mut visited_edges: HashSet<EdgeIndex> = HashSet::new();
    let mut total_visited: usize = 0;

    // BFS queue stores (node_index, current_depth)
    let mut queue: VecDeque<(NodeIndex, u32)> = VecDeque::new();

    // Initialize queue with all seed nodes at depth 0
    for &seed in seed_indices {
        if visited_nodes.len() >= max_nodes {
            break;
        }
        if visited_nodes.insert(seed) {
            queue.push_back((seed, 0));
        }
    }

    // Standard BFS with depth tracking
    while let Some((current_node, depth)) = queue.pop_front() {
        total_visited += 1;

        // Stop expanding if we've reached max depth
        if depth >= max_hops {
            continue;
        }

        // Stop if we've collected enough nodes
        if visited_nodes.len() >= max_nodes {
            break;
        }

        // Traverse outgoing edges
        for edge in graph.edges(current_node) {
            let edge_data = edge.weight();

            // Apply relation type filter if specified
            if let Some(filter) = relation_filter {
                if !filter.iter().any(|f| f == &edge_data.relation_type) {
                    continue;
                }
            }

            let target = edge.target();
            visited_edges.insert(edge.id());

            if visited_nodes.insert(target) {
                if visited_nodes.len() >= max_nodes {
                    break;
                }
                queue.push_back((target, depth + 1));
            }
        }

        // Also traverse incoming edges for bidirectional exploration
        for edge in graph.edges_directed(current_node, petgraph::Direction::Incoming) {
            let edge_data = edge.weight();

            if let Some(filter) = relation_filter {
                if !filter.iter().any(|f| f == &edge_data.relation_type) {
                    continue;
                }
            }

            let source = edge.source();
            visited_edges.insert(edge.id());

            if visited_nodes.insert(source) {
                if visited_nodes.len() >= max_nodes {
                    break;
                }
                queue.push_back((source, depth + 1));
            }
        }
    }

    (
        visited_nodes.into_iter().collect(),
        visited_edges.into_iter().collect(),
        total_visited,
    )
}

/// DFS subgraph extraction from a single seed node.
///
/// Returns (visited_node_indices, visited_edge_indices).
/// Useful for deep exploration along specific paths (e.g., attack chains).
pub fn dfs_subgraph(
    graph: &StableGraph<GraphNode, GraphEdge, Directed>,
    seed: NodeIndex,
    max_depth: u32,
    relation_filter: Option<&[String]>,
    max_nodes: usize,
) -> (Vec<NodeIndex>, Vec<EdgeIndex>) {
    let mut visited_nodes: HashSet<NodeIndex> = HashSet::new();
    let mut visited_edges: HashSet<EdgeIndex> = HashSet::new();

    // DFS stack stores (node_index, current_depth)
    let mut stack: Vec<(NodeIndex, u32)> = vec![(seed, 0)];
    visited_nodes.insert(seed);

    while let Some((current_node, depth)) = stack.pop() {
        if depth >= max_depth || visited_nodes.len() >= max_nodes {
            continue;
        }

        for edge in graph.edges(current_node) {
            let edge_data = edge.weight();

            if let Some(filter) = relation_filter {
                if !filter.iter().any(|f| f == &edge_data.relation_type) {
                    continue;
                }
            }

            let target = edge.target();
            visited_edges.insert(edge.id());

            if visited_nodes.insert(target) {
                if visited_nodes.len() >= max_nodes {
                    break;
                }
                stack.push((target, depth + 1));
            }
        }
    }

    (
        visited_nodes.into_iter().collect(),
        visited_edges.into_iter().collect(),
    )
}

/// Compute shortest path distances from a set of anchor nodes to all reachable nodes.
///
/// Returns a map of NodeIndex → minimum hop distance to any anchor.
/// Used by HybridSearch to compute graph proximity scores.
///
/// Uses multi-source BFS: all anchors start at distance 0.
pub fn shortest_path_distances(
    graph: &StableGraph<GraphNode, GraphEdge, Directed>,
    anchor_indices: &[NodeIndex],
    max_distance: u32,
) -> HashMap<NodeIndex, u32> {
    let mut distances: HashMap<NodeIndex, u32> = HashMap::new();
    let mut queue: VecDeque<(NodeIndex, u32)> = VecDeque::new();

    // Initialize all anchors at distance 0
    for &anchor in anchor_indices {
        distances.insert(anchor, 0);
        queue.push_back((anchor, 0));
    }

    // Multi-source BFS — finds shortest distance to any anchor
    while let Some((current, dist)) = queue.pop_front() {
        if dist >= max_distance {
            continue;
        }

        let next_dist = dist + 1;

        // Outgoing edges
        for edge in graph.edges(current) {
            let target = edge.target();
            if !distances.contains_key(&target) || distances[&target] > next_dist {
                distances.insert(target, next_dist);
                queue.push_back((target, next_dist));
            }
        }

        // Incoming edges (treat graph as undirected for proximity)
        for edge in graph.edges_directed(current, petgraph::Direction::Incoming) {
            let source = edge.source();
            if !distances.contains_key(&source) || distances[&source] > next_dist {
                distances.insert(source, next_dist);
                queue.push_back((source, next_dist));
            }
        }
    }

    distances
}
