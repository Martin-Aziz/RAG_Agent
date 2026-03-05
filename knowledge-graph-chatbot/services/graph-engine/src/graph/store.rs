// graph/store.rs — RocksDB-backed petgraph storage engine.
// Provides persistent, thread-safe graph storage with in-memory petgraph for
// fast traversal and RocksDB for durable persistence across restarts.
//
// Design decisions:
// - StableGraph (not Graph) to preserve NodeIndex/EdgeIndex across removals
// - DashMap for lock-free concurrent node_id → NodeIndex lookups
// - RocksDB column families separate nodes, edges, embeddings, and metadata
// - All mutations are write-through: petgraph + RocksDB in same operation

use std::sync::Arc;
use dashmap::DashMap;
use petgraph::stable_graph::{NodeIndex, EdgeIndex, StableGraph};
use petgraph::Directed;
use rocksdb::{DB, Options, ColumnFamilyDescriptor, WriteBatch};
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::error::{GraphError, StorageError};

// ============================================================================
// Serializable graph node/edge types (internal representation)
// ============================================================================

/// Internal node representation stored in petgraph and RocksDB.
/// Maps 1:1 to the protobuf Node message but uses native Rust types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub name: String,
    pub properties: std::collections::HashMap<String, String>,
    pub embedding: Vec<f32>,
    pub created_at: i64,
    pub confidence: f32,
}

/// Internal edge representation stored in petgraph and RocksDB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub relation_type: String,
    pub weight: f32,
    pub properties: std::collections::HashMap<String, String>,
    pub source_document: String,
    pub created_at: i64,
}

/// Result of a subgraph query: a set of nodes and edges.
#[derive(Debug, Clone)]
pub struct SubgraphResult {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub total_nodes_visited: u32,
    pub traversal_time_ms: f32,
}

// ============================================================================
// Column family names for RocksDB — each stores a different data type
// ============================================================================
const CF_NODES: &str = "nodes";
const CF_EDGES: &str = "edges";
const CF_EMBEDDINGS: &str = "embeddings";
const CF_METADATA: &str = "metadata";

// ============================================================================
// GraphStore — Core storage engine
// ============================================================================

/// Thread-safe, persistent graph storage engine.
/// Combines petgraph (in-memory, fast traversal) with RocksDB (durable persistence).
pub struct GraphStore {
    /// In-memory directed graph for fast traversal algorithms.
    /// RwLock allows concurrent reads with exclusive writes.
    graph: Arc<RwLock<StableGraph<GraphNode, GraphEdge, Directed>>>,

    /// Concurrent map: node UUID → petgraph NodeIndex.
    /// DashMap provides lock-free reads for the hot lookup path.
    node_index: Arc<DashMap<String, NodeIndex>>,

    /// Concurrent map: edge UUID → petgraph EdgeIndex.
    edge_index: Arc<DashMap<String, EdgeIndex>>,

    /// RocksDB instance with column families for persistent storage.
    db: Arc<DB>,
}

impl GraphStore {
    /// Initialize the graph store with RocksDB at the given path.
    /// Creates column families if they don't exist, then loads all persisted
    /// nodes and edges back into the in-memory petgraph.
    pub fn new(db_path: &str) -> Result<Self, GraphError> {
        // Configure RocksDB with column families for data separation.
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        opts.set_max_open_files(256);
        opts.set_keep_log_file_num(5);

        let cf_descriptors = vec![
            ColumnFamilyDescriptor::new(CF_NODES, Options::default()),
            ColumnFamilyDescriptor::new(CF_EDGES, Options::default()),
            ColumnFamilyDescriptor::new(CF_EMBEDDINGS, Options::default()),
            ColumnFamilyDescriptor::new(CF_METADATA, Options::default()),
        ];

        let db = DB::open_cf_descriptors(&opts, db_path, cf_descriptors)
            .map_err(|e| StorageError::RocksDb(e.to_string()))?;

        let store = Self {
            graph: Arc::new(RwLock::new(StableGraph::new())),
            node_index: Arc::new(DashMap::new()),
            edge_index: Arc::new(DashMap::new()),
            db: Arc::new(db),
        };

        // Reload persisted data into in-memory graph on startup.
        // This ensures the graph is immediately queryable after restart.
        store.load_from_disk()?;

        tracing::info!(
            nodes = store.node_index.len(),
            "Graph store initialized from disk"
        );

        Ok(store)
    }

    /// Upsert a node: insert if new, update if exists.
    /// Returns the NodeIndex and whether the node was newly created.
    pub async fn upsert_node(&self, node: GraphNode) -> Result<(NodeIndex, bool), GraphError> {
        // Validate input: node ID must not be empty
        if node.id.is_empty() {
            return Err(GraphError::ValidationError("Node ID cannot be empty".into()));
        }

        let node_id = node.id.clone();
        let mut graph = self.graph.write().await;

        let (idx, created) = if let Some(existing_idx) = self.node_index.get(&node_id) {
            // Update existing node: replace data in petgraph
            let idx = *existing_idx;
            if let Some(existing) = graph.node_weight_mut(idx) {
                existing.label = node.label.clone();
                existing.name = node.name.clone();
                existing.properties = node.properties.clone();
                if !node.embedding.is_empty() {
                    existing.embedding = node.embedding.clone();
                }
                existing.confidence = node.confidence;
            }
            (idx, false)
        } else {
            // Insert new node into petgraph and update the index
            let idx = graph.add_node(node.clone());
            self.node_index.insert(node_id.clone(), idx);
            (idx, true)
        };

        // Persist to RocksDB — serialize node as JSON to the "nodes" CF
        self.persist_node(&node)?;

        tracing::debug!(
            node_id = %node_id,
            created = created,
            "Node upserted"
        );

        Ok((idx, created))
    }

    /// Upsert an edge between two existing nodes.
    /// Returns NOT_FOUND if either the source or target node doesn't exist.
    pub async fn upsert_edge(&self, edge: GraphEdge) -> Result<(EdgeIndex, bool), GraphError> {
        // Validate: both source and target must exist
        let source_idx = self.node_index.get(&edge.source_id)
            .ok_or_else(|| GraphError::NodeNotFound(edge.source_id.clone()))?;
        let target_idx = self.node_index.get(&edge.target_id)
            .ok_or_else(|| GraphError::NodeNotFound(edge.target_id.clone()))?;

        if edge.id.is_empty() {
            return Err(GraphError::ValidationError("Edge ID cannot be empty".into()));
        }

        let edge_id = edge.id.clone();
        let mut graph = self.graph.write().await;

        let (idx, created) = if let Some(existing_idx) = self.edge_index.get(&edge_id) {
            // Update existing edge
            let idx = *existing_idx;
            if let Some(existing) = graph.edge_weight_mut(idx) {
                existing.relation_type = edge.relation_type.clone();
                existing.weight = edge.weight;
                existing.properties = edge.properties.clone();
                existing.source_document = edge.source_document.clone();
            }
            (idx, false)
        } else {
            // Insert new edge between the source and target nodes
            let idx = graph.add_edge(*source_idx, *target_idx, edge.clone());
            self.edge_index.insert(edge_id.clone(), idx);
            (idx, true)
        };

        // Persist to RocksDB
        self.persist_edge(&edge)?;

        tracing::debug!(
            edge_id = %edge_id,
            source = %edge.source_id,
            target = %edge.target_id,
            relation = %edge.relation_type,
            created = created,
            "Edge upserted"
        );

        Ok((idx, created))
    }

    /// Retrieve a node by its UUID.
    pub async fn get_node(&self, node_id: &str) -> Result<GraphNode, GraphError> {
        let idx = self.node_index.get(node_id)
            .ok_or_else(|| GraphError::NodeNotFound(node_id.to_string()))?;

        let graph = self.graph.read().await;
        graph.node_weight(*idx)
            .cloned()
            .ok_or_else(|| GraphError::NodeNotFound(node_id.to_string()))
    }

    /// Delete a node and all its connected edges.
    /// Returns the number of edges that were also removed.
    pub async fn delete_node(&self, node_id: &str) -> Result<u32, GraphError> {
        let idx = self.node_index.get(node_id)
            .ok_or_else(|| GraphError::NodeNotFound(node_id.to_string()))?;
        let idx = *idx;

        let mut graph = self.graph.write().await;

        // Collect all edges connected to this node (both incoming and outgoing)
        let connected_edges: Vec<EdgeIndex> = graph
            .edges(idx)
            .map(|e| e.id())
            .collect();

        // Also collect incoming edges (edges() only returns outgoing)
        let incoming_edges: Vec<EdgeIndex> = graph
            .edges_directed(idx, petgraph::Direction::Incoming)
            .map(|e| e.id())
            .collect();

        let mut edges_removed = 0u32;

        // Remove all connected edges and their index entries
        for edge_idx in connected_edges.iter().chain(incoming_edges.iter()) {
            if let Some(edge) = graph.remove_edge(*edge_idx) {
                self.edge_index.remove(&edge.id);
                self.delete_edge_from_disk(&edge.id)?;
                edges_removed += 1;
            }
        }

        // Remove the node itself
        graph.remove_node(idx);
        self.node_index.remove(node_id);
        self.delete_node_from_disk(node_id)?;

        tracing::info!(
            node_id = %node_id,
            edges_removed = edges_removed,
            "Node deleted"
        );

        Ok(edges_removed)
    }

    /// Query a subgraph by performing BFS from seed nodes.
    /// Delegates to the traversal module for the actual BFS logic.
    pub async fn query_subgraph(
        &self,
        seed_ids: &[String],
        max_hops: u32,
        relation_filter: Option<&[String]>,
        max_nodes: usize,
    ) -> Result<SubgraphResult, GraphError> {
        let start = std::time::Instant::now();
        let graph = self.graph.read().await;

        // Resolve seed IDs to NodeIndex values
        let seed_indices: Vec<NodeIndex> = seed_ids
            .iter()
            .filter_map(|id| self.node_index.get(id).map(|idx| *idx))
            .collect();

        if seed_indices.is_empty() {
            return Err(GraphError::ValidationError(
                "No valid seed node IDs provided".into()
            ));
        }

        // Cap max_hops at 3 to prevent performance explosion
        let capped_hops = max_hops.min(3);
        let capped_nodes = if max_nodes == 0 { 50 } else { max_nodes.min(500) };

        // Perform BFS traversal using the traversal module
        let (visited_nodes, visited_edges, total_visited) =
            crate::graph::traversal::bfs_subgraph(
                &graph,
                &seed_indices,
                capped_hops,
                relation_filter,
                capped_nodes,
            );

        // Collect node and edge data from the graph
        let nodes: Vec<GraphNode> = visited_nodes
            .iter()
            .filter_map(|idx| graph.node_weight(*idx).cloned())
            .collect();

        let edges: Vec<GraphEdge> = visited_edges
            .iter()
            .filter_map(|idx| graph.edge_weight(*idx).cloned())
            .collect();

        let elapsed = start.elapsed();

        tracing::info!(
            seed_count = seed_ids.len(),
            max_hops = capped_hops,
            result_nodes = nodes.len(),
            result_edges = edges.len(),
            traversal_ms = elapsed.as_secs_f32() * 1000.0,
            "Subgraph query completed"
        );

        Ok(SubgraphResult {
            nodes,
            edges,
            total_nodes_visited: total_visited as u32,
            traversal_time_ms: elapsed.as_secs_f32() * 1000.0,
        })
    }

    /// Get graph statistics: total node/edge counts by type.
    pub async fn get_stats(&self) -> (usize, usize, std::collections::HashMap<String, usize>, std::collections::HashMap<String, usize>) {
        let graph = self.graph.read().await;

        let mut nodes_by_label: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut edges_by_type: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        for idx in graph.node_indices() {
            if let Some(node) = graph.node_weight(idx) {
                *nodes_by_label.entry(node.label.clone()).or_insert(0) += 1;
            }
        }

        for idx in graph.edge_indices() {
            if let Some(edge) = graph.edge_weight(idx) {
                *edges_by_type.entry(edge.relation_type.clone()).or_insert(0) += 1;
            }
        }

        let node_count = graph.node_count();
        let edge_count = graph.edge_count();

        (node_count, edge_count, nodes_by_label, edges_by_type)
    }

    /// Get read access to the underlying graph for traversal operations.
    pub fn graph(&self) -> &Arc<RwLock<StableGraph<GraphNode, GraphEdge, Directed>>> {
        &self.graph
    }

    /// Get a reference to the node index for ID lookups.
    pub fn node_index_map(&self) -> &Arc<DashMap<String, NodeIndex>> {
        &self.node_index
    }

    // ========================================================================
    // Persistence helpers — RocksDB read/write operations
    // ========================================================================

    /// Serialize and persist a node to the "nodes" column family.
    fn persist_node(&self, node: &GraphNode) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_NODES)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound(CF_NODES.into()))?;

        let serialized = serde_json::to_vec(node)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;

        self.db.put_cf(&cf, node.id.as_bytes(), &serialized)
            .map_err(|e| StorageError::RocksDb(e.to_string()))?;

        Ok(())
    }

    /// Serialize and persist an edge to the "edges" column family.
    fn persist_edge(&self, edge: &GraphEdge) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_EDGES)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound(CF_EDGES.into()))?;

        let serialized = serde_json::to_vec(edge)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;

        self.db.put_cf(&cf, edge.id.as_bytes(), &serialized)
            .map_err(|e| StorageError::RocksDb(e.to_string()))?;

        Ok(())
    }

    /// Delete a node from the "nodes" column family.
    fn delete_node_from_disk(&self, node_id: &str) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_NODES)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound(CF_NODES.into()))?;

        self.db.delete_cf(&cf, node_id.as_bytes())
            .map_err(|e| StorageError::RocksDb(e.to_string()))?;

        Ok(())
    }

    /// Delete an edge from the "edges" column family.
    fn delete_edge_from_disk(&self, edge_id: &str) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_EDGES)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound(CF_EDGES.into()))?;

        self.db.delete_cf(&cf, edge_id.as_bytes())
            .map_err(|e| StorageError::RocksDb(e.to_string()))?;

        Ok(())
    }

    /// Load all persisted nodes and edges from RocksDB into the in-memory graph.
    /// Called once during initialization to restore state after restart.
    fn load_from_disk(&self) -> Result<(), GraphError> {
        // We need blocking access since this runs in the constructor (not async).
        // Using a new Tokio runtime would be wasteful; instead we directly
        // manipulate the inner graph via try_write (no contention at startup).
        let mut graph = self.graph.blocking_write();

        // Load all nodes from the "nodes" column family
        let nodes_cf = self.db.cf_handle(CF_NODES)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound(CF_NODES.into()))?;

        let iter = self.db.iterator_cf(&nodes_cf, rocksdb::IteratorMode::Start);
        for item in iter {
            let (_, value) = item.map_err(|e| StorageError::RocksDb(e.to_string()))?;
            let node: GraphNode = serde_json::from_slice(&value)
                .map_err(|e| StorageError::Deserialization(e.to_string()))?;
            let node_id = node.id.clone();
            let idx = graph.add_node(node);
            self.node_index.insert(node_id, idx);
        }

        // Load all edges from the "edges" column family
        let edges_cf = self.db.cf_handle(CF_EDGES)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound(CF_EDGES.into()))?;

        let iter = self.db.iterator_cf(&edges_cf, rocksdb::IteratorMode::Start);
        for item in iter {
            let (_, value) = item.map_err(|e| StorageError::RocksDb(e.to_string()))?;
            let edge: GraphEdge = serde_json::from_slice(&value)
                .map_err(|e| StorageError::Deserialization(e.to_string()))?;

            // Resolve source and target node indices
            let source_idx = self.node_index.get(&edge.source_id);
            let target_idx = self.node_index.get(&edge.target_id);

            if let (Some(src), Some(tgt)) = (source_idx, target_idx) {
                let edge_id = edge.id.clone();
                let idx = graph.add_edge(*src, *tgt, edge);
                self.edge_index.insert(edge_id, idx);
            } else {
                // Orphaned edge — source or target node missing. Log and skip.
                tracing::warn!(
                    edge_id = %edge.id,
                    source = %edge.source_id,
                    target = %edge.target_id,
                    "Skipping orphaned edge during load"
                );
            }
        }

        tracing::info!(
            nodes = self.node_index.len(),
            edges = self.edge_index.len(),
            "Loaded graph from RocksDB"
        );

        Ok(())
    }
}
