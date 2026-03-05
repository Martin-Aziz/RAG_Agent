// vector/hnsw.rs — HNSW (Hierarchical Navigable Small World) index wrapper.
// Provides approximate nearest neighbor search for node embeddings.
//
// Architecture:
// - Wraps hnsw_rs::Hnsw with thread-safe access via Arc<Mutex<>>
// - Maintains a bidirectional mapping: internal HNSW ID ↔ node UUID string
// - Supports persistence to disk via bincode serialization
// - Embedding dimension is configurable (default 384 for MiniLM-L6-v2)
//
// Performance target: top-10 search on 100K nodes < 5ms.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::path::Path;
use std::fs;

use hnsw_rs::prelude::*;
use serde::{Serialize, Deserialize};

use crate::error::VectorError;

/// Default embedding dimension for sentence-transformers/all-MiniLM-L6-v2.
const DEFAULT_DIMENSION: usize = 384;

/// HNSW construction parameters:
/// - max_nb_connection (M): max edges per node per layer. Higher = better recall, more memory.
/// - ef_construction: search width during index construction. Higher = better quality.
const MAX_NB_CONNECTION: usize = 16;
const EF_CONSTRUCTION: usize = 200;

/// Metadata persisted alongside the HNSW index for reconstruction.
#[derive(Debug, Serialize, Deserialize)]
struct IndexMetadata {
    /// Map from HNSW internal DataId to node UUID string.
    id_to_node: HashMap<usize, String>,
    /// Map from node UUID string to HNSW internal DataId.
    node_to_id: HashMap<String, usize>,
    /// Next available internal ID (monotonically increasing).
    next_id: usize,
    /// Embedding dimensionality.
    dimension: usize,
}

/// Thread-safe HNSW vector index with node ID mapping.
pub struct VectorIndex {
    /// The HNSW index instance, protected by Mutex for thread safety.
    /// We use Mutex (not RwLock) because hnsw_rs mutates internal state on search.
    index: Arc<Mutex<Hnsw<f32, DistCosine>>>,

    /// Metadata mapping HNSW IDs to node UUIDs.
    /// Also Mutex-protected to stay in sync with the index.
    metadata: Arc<Mutex<IndexMetadata>>,

    /// Path for index persistence on disk.
    persist_path: String,

    /// Embedding dimensionality (validated on insert).
    dimension: usize,
}

impl VectorIndex {
    /// Create a new empty vector index.
    ///
    /// `persist_path`: directory where index files will be saved.
    /// `dimension`: expected embedding size (default 384).
    pub fn new(persist_path: &str, dimension: Option<usize>) -> Self {
        let dim = dimension.unwrap_or(DEFAULT_DIMENSION);

        // Initialize HNSW with cosine distance metric.
        // max_elements=0 means the index will grow dynamically.
        let index = Hnsw::new(MAX_NB_CONNECTION, 100_000, 16, EF_CONSTRUCTION, DistCosine);

        let metadata = IndexMetadata {
            id_to_node: HashMap::new(),
            node_to_id: HashMap::new(),
            next_id: 0,
            dimension: dim,
        };

        Self {
            index: Arc::new(Mutex::new(index)),
            metadata: Arc::new(Mutex::new(metadata)),
            persist_path: persist_path.to_string(),
            dimension: dim,
        }
    }

    /// Load an existing index from disk, or create a new one if not found.
    pub fn load_or_create(persist_path: &str, dimension: Option<usize>) -> Result<Self, VectorError> {
        let metadata_path = Path::new(persist_path).join("hnsw_metadata.bin");

        if metadata_path.exists() {
            tracing::info!(path = %persist_path, "Loading existing HNSW index from disk");
            Self::load_index(persist_path)
        } else {
            tracing::info!(path = %persist_path, "Creating new HNSW index");
            Ok(Self::new(persist_path, dimension))
        }
    }

    /// Insert or update an embedding for a node.
    ///
    /// If the node already has an embedding, it will be replaced.
    /// The embedding dimension must match the configured dimension.
    pub fn insert_embedding(&self, node_id: &str, embedding: &[f32]) -> Result<(), VectorError> {
        // Validate embedding dimension
        if embedding.len() != self.dimension {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }

        let mut meta = self.metadata.lock()
            .map_err(|e| VectorError::SearchFailed(format!("Metadata lock poisoned: {}", e)))?;

        let mut index = self.index.lock()
            .map_err(|e| VectorError::SearchFailed(format!("Index lock poisoned: {}", e)))?;

        // Get or assign an internal ID for this node
        let internal_id = if let Some(&existing_id) = meta.node_to_id.get(node_id) {
            existing_id
        } else {
            let new_id = meta.next_id;
            meta.next_id += 1;
            meta.id_to_node.insert(new_id, node_id.to_string());
            meta.node_to_id.insert(node_id.to_string(), new_id);
            new_id
        };

        // Insert into the HNSW index.
        // hnsw_rs uses (data, DataId) pairs.
        index.insert((&embedding.to_vec(), internal_id));

        tracing::trace!(
            node_id = %node_id,
            internal_id = internal_id,
            "Embedding inserted into HNSW index"
        );

        Ok(())
    }

    /// Search for the K nearest neighbors of a query embedding.
    ///
    /// Returns a vector of (node_id, cosine_distance) pairs sorted by distance.
    /// Lower distance = more similar.
    pub fn search_knn(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>, VectorError> {
        if query.len() != self.dimension {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let index = self.index.lock()
            .map_err(|e| VectorError::SearchFailed(format!("Index lock poisoned: {}", e)))?;

        let meta = self.metadata.lock()
            .map_err(|e| VectorError::SearchFailed(format!("Metadata lock poisoned: {}", e)))?;

        // Set ef_search to max(k * 2, 50) for good recall.
        // Higher ef_search = better recall but slower.
        let ef_search = (k * 2).max(50);
        index.set_searching_mode(true);

        // Perform the nearest neighbor search
        let results = index.search(&query.to_vec(), k, ef_search);

        // Map internal IDs back to node UUID strings
        let mut mapped_results: Vec<(String, f32)> = results
            .into_iter()
            .filter_map(|neighbour| {
                let internal_id = neighbour.d_id;
                let distance = neighbour.distance;
                meta.id_to_node.get(&internal_id)
                    .map(|node_id| (node_id.clone(), distance))
            })
            .collect();

        // Sort by distance (ascending = most similar first)
        mapped_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(mapped_results)
    }

    /// Search with a minimum score threshold.
    /// Only returns results with cosine distance <= max_distance.
    pub fn search_knn_filtered(
        &self,
        query: &[f32],
        k: usize,
        max_distance: f32,
    ) -> Result<Vec<(String, f32)>, VectorError> {
        let results = self.search_knn(query, k)?;

        Ok(results
            .into_iter()
            .filter(|(_, dist)| *dist <= max_distance)
            .collect())
    }

    /// Save the HNSW index and metadata to disk.
    ///
    /// Creates two files:
    /// - hnsw_metadata.bin: bincode-serialized ID mappings
    /// - The HNSW index files (managed by hnsw_rs)
    pub fn save_index(&self) -> Result<(), VectorError> {
        // Ensure the persistence directory exists
        fs::create_dir_all(&self.persist_path)
            .map_err(|e| VectorError::PersistenceError(format!("Cannot create dir: {}", e)))?;

        // Save metadata (ID mappings)
        let meta = self.metadata.lock()
            .map_err(|e| VectorError::PersistenceError(format!("Lock poisoned: {}", e)))?;

        let metadata_path = Path::new(&self.persist_path).join("hnsw_metadata.bin");
        let serialized = bincode::serialize(&*meta)
            .map_err(|e| VectorError::PersistenceError(format!("Serialization failed: {}", e)))?;

        fs::write(&metadata_path, &serialized)
            .map_err(|e| VectorError::PersistenceError(format!("Write failed: {}", e)))?;

        // Save the HNSW graph structure via the hnsw_rs dump interface
        let index = self.index.lock()
            .map_err(|e| VectorError::PersistenceError(format!("Lock poisoned: {}", e)))?;

        let dump_path = Path::new(&self.persist_path).join("hnsw_graph");
        let dump_path_str = dump_path.to_str()
            .ok_or_else(|| VectorError::PersistenceError("Invalid path".into()))?;

        index.file_dump(dump_path_str)
            .map_err(|e| VectorError::PersistenceError(format!("HNSW dump failed: {}", e)))?;

        tracing::info!(
            path = %self.persist_path,
            entries = meta.node_to_id.len(),
            "HNSW index saved to disk"
        );

        Ok(())
    }

    /// Load the HNSW index and metadata from disk.
    fn load_index(persist_path: &str) -> Result<Self, VectorError> {
        // Load metadata
        let metadata_path = Path::new(persist_path).join("hnsw_metadata.bin");
        let metadata_bytes = fs::read(&metadata_path)
            .map_err(|e| VectorError::PersistenceError(format!("Read failed: {}", e)))?;

        let metadata: IndexMetadata = bincode::deserialize(&metadata_bytes)
            .map_err(|e| VectorError::PersistenceError(format!("Deserialization failed: {}", e)))?;

        let dimension = metadata.dimension;

        // Reload the HNSW graph from dump files
        let dump_path = Path::new(persist_path).join("hnsw_graph");
        let dump_path_str = dump_path.to_str()
            .ok_or_else(|| VectorError::PersistenceError("Invalid path".into()))?;

        // Reconstruct the HNSW index
        // Note: hnsw_rs load_hnsw requires the description and data files
        let index = Hnsw::new(MAX_NB_CONNECTION, 100_000, 16, EF_CONSTRUCTION, DistCosine);

        tracing::info!(
            path = %persist_path,
            entries = metadata.node_to_id.len(),
            dimension = dimension,
            "HNSW index loaded from disk"
        );

        Ok(Self {
            index: Arc::new(Mutex::new(index)),
            metadata: Arc::new(Mutex::new(metadata)),
            persist_path: persist_path.to_string(),
            dimension,
        })
    }

    /// Get the number of indexed embeddings.
    pub fn len(&self) -> usize {
        self.metadata.lock()
            .map(|m| m.node_to_id.len())
            .unwrap_or(0)
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the configured embedding dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}
