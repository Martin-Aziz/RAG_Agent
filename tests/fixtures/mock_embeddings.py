"""Mock embeddings service for deterministic testing."""
import hashlib
import numpy as np
from typing import List, Union


class MockEmbeddings:
    """Mock embeddings service that returns deterministic vectors."""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.call_count = 0
        self.embedding_cache = {}
        self.custom_embeddings = {}
        
    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.embedding_cache = {}
        
    def set_embedding(self, text: str, embedding: List[float]):
        """Set custom embedding for specific text."""
        self.custom_embeddings[text] = embedding
        
    def embed_query(self, text: str) -> List[float]:
        """Embed query text into vector."""
        self.call_count += 1
        
        # Check custom embeddings first
        if text in self.custom_embeddings:
            return self.custom_embeddings[text]
            
        # Check cache
        if text in self.embedding_cache:
            return self.embedding_cache[text]
            
        # Generate deterministic embedding based on text hash
        embedding = self._generate_deterministic_embedding(text)
        self.embedding_cache[text] = embedding
        return embedding
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        return [self.embed_query(text) for text in texts]
        
    def _generate_deterministic_embedding(self, text: str) -> List[float]:
        """Generate deterministic embedding from text using hash."""
        # Use hash to seed random generator for deterministic output
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        rng = np.random.RandomState(hash_val % (2**32))
        
        # Generate random vector and normalize
        embedding = rng.randn(self.dimension)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding.tolist()
        
    def stub_embeddings_for(self, text: str, similar_to: str = None):
        """Stub embeddings to make text similar to another text."""
        if similar_to is None:
            # Just ensure embedding exists
            self.embed_query(text)
        else:
            # Make embeddings similar
            base_embedding = self.embed_query(similar_to)
            # Add small random noise
            noise = np.random.randn(self.dimension) * 0.1
            similar_embedding = np.array(base_embedding) + noise
            similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)
            self.set_embedding(text, similar_embedding.tolist())
            
    def __call__(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Make instance callable."""
        if isinstance(text, str):
            return self.embed_query(text)
        return self.embed_documents(text)
