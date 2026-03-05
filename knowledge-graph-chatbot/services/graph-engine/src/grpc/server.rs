// grpc/server.rs — tonic gRPC server implementing all GraphService RPCs.
// Maps protobuf messages to internal types, delegates to GraphStore and VectorIndex,
// and returns proper gRPC status codes on errors.
//
// Every RPC call is instrumented with tracing spans for observability:
// - Request parameters logged at info level
// - Latency measured and logged on completion
// - Errors logged at error level before returning gRPC Status

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::graph::store::{GraphStore, GraphNode, GraphEdge};
use crate::graph::traversal;
use crate::vector::hnsw::VectorIndex;
use crate::error::GraphError;

// Import the generated protobuf types from tonic-build.
// These are placed in the `kgchat` module by prost based on the proto package name.
pub mod proto {
    tonic::include_proto!("kgchat");
}

use proto::graph_service_server::GraphService;

/// GraphServiceImpl holds shared references to the graph store and vector index.
/// All methods are &self (shared immutable reference) — thread safety is internal.
pub struct GraphServiceImpl {
    store: Arc<GraphStore>,
    vector_index: Arc<VectorIndex>,
}

impl GraphServiceImpl {
    pub fn new(store: Arc<GraphStore>, vector_index: Arc<VectorIndex>) -> Self {
        Self { store, vector_index }
    }

    /// Convert protobuf Node to internal GraphNode.
    fn proto_to_node(node: &proto::Node) -> GraphNode {
        GraphNode {
            id: node.id.clone(),
            label: node.label.clone(),
            name: node.name.clone(),
            properties: node.properties.clone(),
            embedding: node.embedding.clone(),
            created_at: node.created_at,
            confidence: node.confidence,
        }
    }

    /// Convert internal GraphNode to protobuf Node.
    fn node_to_proto(node: &GraphNode) -> proto::Node {
        proto::Node {
            id: node.id.clone(),
            label: node.label.clone(),
            name: node.name.clone(),
            properties: node.properties.clone(),
            embedding: node.embedding.clone(),
            created_at: node.created_at,
            confidence: node.confidence,
        }
    }

    /// Convert protobuf Edge to internal GraphEdge.
    fn proto_to_edge(edge: &proto::Edge) -> GraphEdge {
        GraphEdge {
            id: edge.id.clone(),
            source_id: edge.source_id.clone(),
            target_id: edge.target_id.clone(),
            relation_type: edge.relation_type.clone(),
            weight: edge.weight,
            properties: edge.properties.clone(),
            source_document: edge.source_document.clone(),
            created_at: edge.created_at,
        }
    }

    /// Convert internal GraphEdge to protobuf Edge.
    fn edge_to_proto(edge: &GraphEdge) -> proto::Edge {
        proto::Edge {
            id: edge.id.clone(),
            source_id: edge.source_id.clone(),
            target_id: edge.target_id.clone(),
            relation_type: edge.relation_type.clone(),
            weight: edge.weight,
            properties: edge.properties.clone(),
            source_document: edge.source_document.clone(),
            created_at: edge.created_at,
        }
    }
}

#[tonic::async_trait]
impl GraphService for GraphServiceImpl {
    /// UpsertNode — Create or update a node in the knowledge graph.
    async fn upsert_node(
        &self,
        request: Request<proto::UpsertNodeRequest>,
    ) -> Result<Response<proto::NodeResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        let node = req.node
            .ok_or_else(|| Status::invalid_argument("Node is required"))?;

        tracing::info!(node_id = %node.id, label = %node.label, name = %node.name, "UpsertNode request");

        let graph_node = Self::proto_to_node(&node);
        let has_embedding = !graph_node.embedding.is_empty();
        let embedding_clone = graph_node.embedding.clone();
        let node_id_clone = graph_node.id.clone();

        // Upsert into graph store
        let (_, created) = self.store.upsert_node(graph_node).await
            .map_err(|e| Status::from(e))?;

        // If the node has an embedding, also index it in the vector store
        if has_embedding {
            self.vector_index.insert_embedding(&node_id_clone, &embedding_clone)
                .map_err(|e| {
                    tracing::error!(error = %e, "Failed to index embedding");
                    Status::internal(format!("Vector indexing failed: {}", e))
                })?;
        }

        let elapsed = start.elapsed();
        tracing::info!(
            node_id = %node.id,
            created = created,
            latency_ms = elapsed.as_secs_f64() * 1000.0,
            "UpsertNode completed"
        );

        Ok(Response::new(proto::NodeResponse {
            node: Some(node),
            created,
        }))
    }

    /// UpsertEdge — Create or update a directed edge between two nodes.
    async fn upsert_edge(
        &self,
        request: Request<proto::UpsertEdgeRequest>,
    ) -> Result<Response<proto::EdgeResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        let edge = req.edge
            .ok_or_else(|| Status::invalid_argument("Edge is required"))?;

        tracing::info!(
            edge_id = %edge.id,
            source = %edge.source_id,
            target = %edge.target_id,
            relation = %edge.relation_type,
            "UpsertEdge request"
        );

        let graph_edge = Self::proto_to_edge(&edge);
        let (_, created) = self.store.upsert_edge(graph_edge).await
            .map_err(|e| Status::from(e))?;

        let elapsed = start.elapsed();
        tracing::info!(
            edge_id = %edge.id,
            created = created,
            latency_ms = elapsed.as_secs_f64() * 1000.0,
            "UpsertEdge completed"
        );

        Ok(Response::new(proto::EdgeResponse {
            edge: Some(edge),
            created,
        }))
    }

    /// QuerySubgraph — BFS traversal from seed nodes with depth and filter constraints.
    async fn query_subgraph(
        &self,
        request: Request<proto::SubgraphQuery>,
    ) -> Result<Response<proto::SubgraphResult>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        tracing::info!(
            seeds = req.seed_node_ids.len(),
            max_hops = req.max_hops,
            max_nodes = req.max_nodes,
            "QuerySubgraph request"
        );

        let relation_filter = if req.relation_filter.is_empty() {
            None
        } else {
            Some(req.relation_filter.as_slice())
        };

        let max_nodes = if req.max_nodes <= 0 { 50 } else { req.max_nodes as usize };

        let result = self.store.query_subgraph(
            &req.seed_node_ids,
            req.max_hops as u32,
            relation_filter.map(|f| f.iter().map(|s| s.as_str()).collect::<Vec<_>>()).as_deref(),
            max_nodes,
        ).await.map_err(|e| Status::from(e))?;

        let elapsed = start.elapsed();
        tracing::info!(
            result_nodes = result.nodes.len(),
            result_edges = result.edges.len(),
            latency_ms = elapsed.as_secs_f64() * 1000.0,
            "QuerySubgraph completed"
        );

        Ok(Response::new(proto::SubgraphResult {
            nodes: result.nodes.iter().map(Self::node_to_proto).collect(),
            edges: result.edges.iter().map(Self::edge_to_proto).collect(),
            total_nodes_visited: result.total_nodes_visited as i32,
            traversal_time_ms: result.traversal_time_ms,
        }))
    }

    /// VectorSearch — Top-K nearest neighbor search by embedding similarity.
    async fn vector_search(
        &self,
        request: Request<proto::VectorSearchRequest>,
    ) -> Result<Response<proto::VectorSearchResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        let k = if req.k <= 0 { 10 } else { req.k as usize };

        tracing::info!(
            embedding_dim = req.embedding.len(),
            k = k,
            "VectorSearch request"
        );

        let results = self.vector_index.search_knn(&req.embedding, k)
            .map_err(|e| {
                tracing::error!(error = %e, "Vector search failed");
                Status::internal(format!("Vector search error: {}", e))
            })?;

        // Optionally enrich results with full node data
        let mut search_results = Vec::with_capacity(results.len());
        for (node_id, distance) in &results {
            // Apply minimum score filter
            if req.min_score > 0.0 {
                let similarity = 1.0 - distance;
                if similarity < req.min_score {
                    continue;
                }
            }

            // Apply label filter if specified
            let node = self.store.get_node(node_id).await.ok();
            if let Some(ref node_data) = node {
                if !req.label_filter.is_empty() {
                    if !req.label_filter.contains(&node_data.label) {
                        continue;
                    }
                }
            }

            search_results.push(proto::VectorSearchResult {
                node_id: node_id.clone(),
                distance: *distance,
                node: node.map(|n| Self::node_to_proto(&n)),
            });
        }

        let elapsed = start.elapsed();
        tracing::info!(
            results = search_results.len(),
            latency_ms = elapsed.as_secs_f64() * 1000.0,
            "VectorSearch completed"
        );

        Ok(Response::new(proto::VectorSearchResponse {
            results: search_results,
            search_time_ms: elapsed.as_secs_f32() * 1000.0,
        }))
    }

    /// HybridSearch — Combine vector similarity with graph path distance.
    async fn hybrid_search(
        &self,
        request: Request<proto::HybridSearchRequest>,
    ) -> Result<Response<proto::HybridSearchResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        let k = if req.k <= 0 { 10 } else { req.k as usize };
        let vector_weight = if req.vector_weight == 0.0 { 0.7 } else { req.vector_weight };
        let graph_weight = if req.graph_weight == 0.0 { 0.3 } else { req.graph_weight };

        tracing::info!(
            k = k,
            anchors = req.anchor_ids.len(),
            vector_weight = vector_weight,
            graph_weight = graph_weight,
            "HybridSearch request"
        );

        // Step 1: Vector search for candidate nodes (fetch more than K for re-ranking)
        let vector_results = self.vector_index.search_knn(&req.embedding, k * 3)
            .map_err(|e| Status::internal(format!("Vector search: {}", e)))?;

        // Step 2: Compute shortest path distances from anchor nodes
        let graph = self.store.graph().read().await;
        let anchor_indices: Vec<_> = req.anchor_ids.iter()
            .filter_map(|id| self.store.node_index_map().get(id).map(|idx| *idx))
            .collect();

        let distances = traversal::shortest_path_distances(
            &graph, &anchor_indices, 5
        );
        drop(graph);

        // Step 3: Re-rank by combining vector + graph scores
        let mut hybrid_results: Vec<proto::HybridResult> = Vec::new();

        for (node_id, vector_distance) in &vector_results {
            let vector_score = 1.0 - vector_distance; // Convert distance to similarity

            // Get graph proximity score (inverse of hop distance)
            let (graph_score, hops) = if let Some(idx) = self.store.node_index_map().get(node_id) {
                if let Some(&hop_dist) = distances.get(&*idx) {
                    (1.0 / (1.0 + hop_dist as f32), hop_dist as i32) // Inverse distance
                } else {
                    (0.0, -1) // Not reachable from anchors
                }
            } else {
                (0.0, -1)
            };

            let combined = vector_weight * vector_score + graph_weight * graph_score;

            let node = self.store.get_node(node_id).await.ok();

            hybrid_results.push(proto::HybridResult {
                node_id: node_id.clone(),
                vector_score,
                graph_score,
                combined_score: combined,
                node: node.map(|n| Self::node_to_proto(&n)),
                shortest_path_hops: hops,
            });
        }

        // Sort by combined score (descending)
        hybrid_results.sort_by(|a, b| {
            b.combined_score.partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to K results
        hybrid_results.truncate(k);

        let elapsed = start.elapsed();
        tracing::info!(
            results = hybrid_results.len(),
            latency_ms = elapsed.as_secs_f64() * 1000.0,
            "HybridSearch completed"
        );

        Ok(Response::new(proto::HybridSearchResponse {
            results: hybrid_results,
            search_time_ms: elapsed.as_secs_f32() * 1000.0,
        }))
    }

    /// BatchIngest — Client-streaming RPC for bulk node/edge ingestion.
    type BatchIngestStream = ReceiverStream<Result<proto::BatchIngestResponse, Status>>;

    async fn batch_ingest(
        &self,
        request: Request<tonic::Streaming<proto::BatchIngestRequest>>,
    ) -> Result<Response<proto::BatchIngestResponse>, Status> {
        let start = Instant::now();
        let mut stream = request.into_inner();

        let mut nodes_upserted: i32 = 0;
        let mut edges_upserted: i32 = 0;
        let mut errors: i32 = 0;
        let mut error_messages: Vec<String> = Vec::new();

        tracing::info!("BatchIngest stream started");

        // Process each item in the stream
        while let Some(item) = stream.message().await
            .map_err(|e| Status::internal(format!("Stream error: {}", e)))? {

            match item.item {
                Some(proto::batch_ingest_request::Item::Node(node)) => {
                    let graph_node = Self::proto_to_node(&node);
                    let has_embedding = !graph_node.embedding.is_empty();
                    let embedding = graph_node.embedding.clone();
                    let node_id = graph_node.id.clone();

                    match self.store.upsert_node(graph_node).await {
                        Ok(_) => {
                            nodes_upserted += 1;
                            // Index embedding if present
                            if has_embedding {
                                if let Err(e) = self.vector_index.insert_embedding(&node_id, &embedding) {
                                    tracing::warn!(node_id = %node_id, error = %e, "Embedding indexing failed in batch");
                                }
                            }
                        }
                        Err(e) => {
                            errors += 1;
                            error_messages.push(format!("Node {}: {}", node.id, e));
                        }
                    }
                }
                Some(proto::batch_ingest_request::Item::Edge(edge)) => {
                    let graph_edge = Self::proto_to_edge(&edge);
                    match self.store.upsert_edge(graph_edge).await {
                        Ok(_) => edges_upserted += 1,
                        Err(e) => {
                            errors += 1;
                            error_messages.push(format!("Edge {}: {}", edge.id, e));
                        }
                    }
                }
                None => {
                    tracing::warn!("Received empty batch ingest item");
                }
            }
        }

        let elapsed = start.elapsed();
        tracing::info!(
            nodes = nodes_upserted,
            edges = edges_upserted,
            errors = errors,
            latency_ms = elapsed.as_secs_f64() * 1000.0,
            "BatchIngest completed"
        );

        // Save vector index to disk after bulk ingestion
        if nodes_upserted > 0 {
            if let Err(e) = self.vector_index.save_index() {
                tracing::error!(error = %e, "Failed to persist vector index after batch ingest");
            }
        }

        Ok(Response::new(proto::BatchIngestResponse {
            nodes_upserted,
            edges_upserted,
            errors,
            error_messages,
            total_time_ms: elapsed.as_secs_f32() * 1000.0,
        }))
    }

    /// GetNode — Retrieve a single node by ID.
    async fn get_node(
        &self,
        request: Request<proto::GetNodeRequest>,
    ) -> Result<Response<proto::NodeResponse>, Status> {
        let req = request.into_inner();
        tracing::info!(node_id = %req.node_id, "GetNode request");

        let node = self.store.get_node(&req.node_id).await
            .map_err(|e| Status::from(e))?;

        Ok(Response::new(proto::NodeResponse {
            node: Some(Self::node_to_proto(&node)),
            created: false,
        }))
    }

    /// DeleteNode — Remove a node and all its connected edges.
    async fn delete_node(
        &self,
        request: Request<proto::DeleteNodeRequest>,
    ) -> Result<Response<proto::DeleteNodeResponse>, Status> {
        let req = request.into_inner();
        tracing::info!(node_id = %req.node_id, "DeleteNode request");

        let edges_removed = self.store.delete_node(&req.node_id).await
            .map_err(|e| Status::from(e))?;

        Ok(Response::new(proto::DeleteNodeResponse {
            deleted: true,
            edges_removed: edges_removed as i32,
        }))
    }

    /// GetStats — Return graph statistics.
    async fn get_stats(
        &self,
        _request: Request<proto::StatsRequest>,
    ) -> Result<Response<proto::StatsResponse>, Status> {
        let (node_count, edge_count, nodes_by_label, edges_by_type) =
            self.store.get_stats().await;

        Ok(Response::new(proto::StatsResponse {
            total_nodes: node_count as i64,
            total_edges: edge_count as i64,
            vector_index_size: self.vector_index.len() as i64,
            nodes_by_label: nodes_by_label.into_iter()
                .map(|(k, v)| (k, v as i64))
                .collect(),
            edges_by_type: edges_by_type.into_iter()
                .map(|(k, v)| (k, v as i64))
                .collect(),
        }))
    }
}
