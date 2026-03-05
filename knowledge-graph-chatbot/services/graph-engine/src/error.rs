// error.rs — Typed error hierarchy for the graph-engine service.
// Uses `thiserror` for ergonomic error definitions with automatic Display/From impls.
// Maps to appropriate gRPC status codes via the From<GraphError> for tonic::Status impl.

use thiserror::Error;

/// Top-level error type for all graph-engine operations.
/// Each variant maps to a specific gRPC status code for proper client error handling.
#[derive(Error, Debug)]
pub enum GraphError {
    /// Node or edge not found by ID.
    /// Maps to gRPC NOT_FOUND.
    #[error("Node not found: {0}")]
    NodeNotFound(String),

    /// Edge not found by ID.
    #[error("Edge not found: {0}")]
    EdgeNotFound(String),

    /// Duplicate node or edge insertion attempt.
    /// Maps to gRPC ALREADY_EXISTS.
    #[error("Duplicate entity: {0}")]
    DuplicateEntity(String),

    /// Validation error for invalid input data (e.g., empty IDs, wrong dimensions).
    /// Maps to gRPC INVALID_ARGUMENT.
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// RocksDB storage operation failure.
    /// Maps to gRPC INTERNAL.
    #[error("Storage error: {0}")]
    StorageError(#[from] StorageError),

    /// HNSW vector index operation failure.
    /// Maps to gRPC INTERNAL.
    #[error("Vector index error: {0}")]
    VectorError(#[from] VectorError),

    /// Serialization/deserialization failure.
    /// Maps to gRPC INTERNAL.
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Internal unexpected error.
    /// Maps to gRPC INTERNAL.
    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Storage-layer errors for RocksDB operations.
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("RocksDB error: {0}")]
    RocksDb(String),

    #[error("Column family not found: {0}")]
    ColumnFamilyNotFound(String),

    #[error("Serialization failed: {0}")]
    Serialization(String),

    #[error("Deserialization failed: {0}")]
    Deserialization(String),
}

/// Vector index errors for HNSW operations.
#[derive(Error, Debug)]
pub enum VectorError {
    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Vector index not initialized")]
    NotInitialized,

    #[error("Index persistence error: {0}")]
    PersistenceError(String),

    #[error("Search failed: {0}")]
    SearchFailed(String),
}

// ============================================================================
// gRPC Status Mapping
// ============================================================================

impl From<GraphError> for tonic::Status {
    fn from(err: GraphError) -> Self {
        match &err {
            GraphError::NodeNotFound(_) | GraphError::EdgeNotFound(_) => {
                tonic::Status::not_found(err.to_string())
            }
            GraphError::DuplicateEntity(_) => {
                tonic::Status::already_exists(err.to_string())
            }
            GraphError::ValidationError(_) => {
                tonic::Status::invalid_argument(err.to_string())
            }
            GraphError::StorageError(_)
            | GraphError::VectorError(_)
            | GraphError::SerializationError(_)
            | GraphError::InternalError(_) => {
                // Log internal errors before converting — the gRPC client
                // should see a generic message, not internal details in production.
                tracing::error!(error = %err, "Internal graph-engine error");
                tonic::Status::internal(err.to_string())
            }
        }
    }
}

// ============================================================================
// Convenience conversions
// ============================================================================

impl From<serde_json::Error> for GraphError {
    fn from(err: serde_json::Error) -> Self {
        GraphError::SerializationError(err.to_string())
    }
}

impl From<rocksdb::Error> for StorageError {
    fn from(err: rocksdb::Error) -> Self {
        StorageError::RocksDb(err.to_string())
    }
}
