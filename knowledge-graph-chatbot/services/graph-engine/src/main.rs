// main.rs — Graph Engine service entrypoint.
// Initializes RocksDB storage, HNSW vector index, and starts the gRPC server.
//
// Configuration is read from environment variables:
//   ROCKSDB_PATH     — path to RocksDB data directory (default: /data/graph)
//   HNSW_INDEX_PATH  — path to HNSW index directory (default: /data/vectors)
//   GRPC_PORT        — port for the gRPC server (default: 50051)
//   LOG_LEVEL        — tracing log level (default: info)
//   EMBEDDING_DIM    — embedding dimension (default: 384)

pub mod error;
pub mod graph;
pub mod vector;
pub mod grpc;

use std::env;
use std::net::SocketAddr;
use std::sync::Arc;

use tonic::transport::Server;
use tracing_subscriber::{fmt, EnvFilter};

use crate::graph::store::GraphStore;
use crate::vector::hnsw::VectorIndex;
use crate::grpc::server::{GraphServiceImpl, proto};

use proto::graph_service_server::GraphServiceServer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ========================================================================
    // 1. Initialize structured logging
    // ========================================================================
    let log_level = env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string());

    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(&log_level))
        )
        .json()
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    tracing::info!("Starting graph-engine service");

    // ========================================================================
    // 2. Load configuration from environment
    // ========================================================================
    let rocksdb_path = env::var("ROCKSDB_PATH")
        .unwrap_or_else(|_| "/data/graph".to_string());
    let hnsw_path = env::var("HNSW_INDEX_PATH")
        .unwrap_or_else(|_| "/data/vectors".to_string());
    let grpc_port: u16 = env::var("GRPC_PORT")
        .unwrap_or_else(|_| "50051".to_string())
        .parse()
        .expect("GRPC_PORT must be a valid port number");
    let embedding_dim: usize = env::var("EMBEDDING_DIM")
        .unwrap_or_else(|_| "384".to_string())
        .parse()
        .expect("EMBEDDING_DIM must be a valid integer");

    tracing::info!(
        rocksdb_path = %rocksdb_path,
        hnsw_path = %hnsw_path,
        grpc_port = grpc_port,
        embedding_dim = embedding_dim,
        "Configuration loaded"
    );

    // ========================================================================
    // 3. Initialize storage layers
    // ========================================================================

    // Initialize RocksDB-backed graph store.
    // This loads any persisted nodes/edges from disk into the in-memory petgraph.
    let graph_store = Arc::new(
        GraphStore::new(&rocksdb_path)
            .expect("Failed to initialize graph store")
    );
    tracing::info!("Graph store initialized");

    // Initialize HNSW vector index.
    // Loads existing index from disk if available, or creates a new empty one.
    let vector_index = Arc::new(
        VectorIndex::load_or_create(&hnsw_path, Some(embedding_dim))
            .expect("Failed to initialize vector index")
    );
    tracing::info!(
        indexed_vectors = vector_index.len(),
        dimension = vector_index.dimension(),
        "Vector index initialized"
    );

    // ========================================================================
    // 4. Create gRPC service implementation
    // ========================================================================
    let graph_service = GraphServiceImpl::new(
        graph_store.clone(),
        vector_index.clone(),
    );

    // ========================================================================
    // 5. Start gRPC server
    // ========================================================================
    let addr: SocketAddr = format!("0.0.0.0:{}", grpc_port)
        .parse()
        .expect("Invalid socket address");

    tracing::info!(address = %addr, "Starting gRPC server");

    // Set up graceful shutdown handler
    let vector_index_shutdown = vector_index.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        tracing::info!("Shutdown signal received — persisting vector index");

        // Save the HNSW index to disk before exiting
        if let Err(e) = vector_index_shutdown.save_index() {
            tracing::error!(error = %e, "Failed to persist vector index on shutdown");
        }

        tracing::info!("Vector index persisted. Shutting down.");
        std::process::exit(0);
    });

    // Start the tonic gRPC server with the GraphService implementation
    Server::builder()
        .add_service(GraphServiceServer::new(graph_service))
        .serve(addr)
        .await?;

    Ok(())
}
