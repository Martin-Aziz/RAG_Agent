"""Mock vector database for deterministic testing."""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class MockVectorDB:
    """Mock vector database with in-memory storage."""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.documents = {}  # doc_id -> {text, embedding, metadata}
        self.index = []  # List of (doc_id, embedding)
        self.call_count = 0
        self.failure_mode = None
        
    def reset(self):
        """Reset mock state."""
        self.documents = {}
        self.index = []
        self.call_count = 0
        self.failure_mode = None
        
    def set_failure_mode(self, mode: str):
        """Set failure mode: unavailable, slow, corrupted."""
        self.failure_mode = mode
        
    def index_document(self, doc_id: str, text: str, embedding: List[float], metadata: Dict[str, Any] = None):
        """Index a document."""
        self.call_count += 1
        
        if self.failure_mode == "unavailable":
            raise ConnectionError("Vector DB unavailable")
            
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}")
            
        self.documents[doc_id] = {
            "text": text,
            "embedding": np.array(embedding),
            "metadata": metadata or {}
        }
        self.index.append((doc_id, np.array(embedding)))
        
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Batch index documents."""
        for doc in documents:
            self.index_document(
                doc["id"],
                doc["text"],
                doc["embedding"],
                doc.get("metadata")
            )
            
    def search(self, query_embedding: List[float], top_k: int = 5, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        self.call_count += 1
        
        if self.failure_mode == "unavailable":
            raise ConnectionError("Vector DB unavailable")
        elif self.failure_mode == "corrupted":
            return []  # Return empty results
            
        query_vec = np.array(query_embedding)
        
        if len(self.index) == 0:
            return []
            
        # Compute similarities
        similarities = []
        for doc_id, doc_embedding in self.index:
            # Apply metadata filters
            if filter_metadata:
                doc_metadata = self.documents[doc_id]["metadata"]
                if not self._matches_filter(doc_metadata, filter_metadata):
                    continue
                    
            # Cosine similarity
            similarity = np.dot(query_vec, doc_embedding) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_embedding)
            )
            similarities.append((doc_id, float(similarity)))
            
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in similarities[:top_k]:
            doc = self.documents[doc_id]
            results.append({
                "id": doc_id,
                "text": doc["text"],
                "score": score,
                "metadata": doc["metadata"]
            })
            
        return results
        
    def delete_document(self, doc_id: str):
        """Delete a document."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self.index = [(did, emb) for did, emb in self.index if did != doc_id]
            
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        return self.documents.get(doc_id)
        
    def list_documents(self, filter_metadata: Dict[str, Any] = None) -> List[str]:
        """List all document IDs."""
        if filter_metadata:
            return [
                doc_id for doc_id, doc in self.documents.items()
                if self._matches_filter(doc["metadata"], filter_metadata)
            ]
        return list(self.documents.keys())
        
    def _matches_filter(self, metadata: Dict[str, Any], filter_metadata: Dict[str, Any]) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter_metadata.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
        
    def __len__(self):
        """Return number of documents."""
        return len(self.documents)
