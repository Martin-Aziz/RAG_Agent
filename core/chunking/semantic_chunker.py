"""Semantic chunking implementation using embedding-based similarity.

Groups sentences into coherent chunks based on semantic similarity between
consecutive sentences, splitting when similarity drops below threshold.
"""
from typing import List, Dict, Any, Optional
import re


class SemanticChunker:
    """Chunks text using embedding-based sentence similarity.
    
    Args:
        embedder: Object with embed(texts: List[str]) -> List[List[float]] method
        similarity_threshold: Split when cosine similarity drops below this (0.6-0.75)
        min_chunk_size: Minimum tokens per chunk
        max_chunk_size: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
    """
    
    def __init__(
        self,
        embedder,
        similarity_threshold: float = 0.7,
        min_chunk_size: int = 100,
        max_chunk_size: int = 500,
        overlap_tokens: int = 50
    ):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_tokens = overlap_tokens
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Handle common abbreviations
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr)\.\s', r'\1<PERIOD> ', text)
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Restore periods in abbreviations
        sentences = [s.replace('<PERIOD>', '.') for s in sentences]
        return [s.strip() for s in sentences if s.strip()]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        try:
            import numpy as np
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        except Exception:
            # Fallback pure Python
            import math
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)
    
    def _count_tokens(self, text: str) -> int:
        """Approximate token count (words * 1.3 for subwords)."""
        return int(len(text.split()) * 1.3)
    
    def chunk(self, text: str, doc_id: str = "doc") -> List[Dict[str, Any]]:
        """Chunk text into semantically coherent passages with overlap.
        
        Returns:
            List of chunks with metadata: [{"text": str, "doc_id": str, "chunk_id": int, ...}]
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        
        # Get embeddings for all sentences
        try:
            sentence_embeddings = self.embedder.embed(sentences)
        except Exception as e:
            # Fallback: return fixed-size chunks
            return self._fallback_chunk(text, doc_id)
        
        # Group sentences into chunks based on similarity
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_tokens = self._count_tokens(sentences[0])
        
        for i in range(1, len(sentences)):
            # Compute similarity between consecutive sentences
            similarity = self._cosine_similarity(
                sentence_embeddings[i-1],
                sentence_embeddings[i]
            )
            
            sentence_tokens = self._count_tokens(sentences[i])
            
            # Split conditions:
            # 1. Similarity drops below threshold (semantic boundary)
            # 2. Chunk would exceed max size
            should_split = (
                similarity < self.similarity_threshold or
                current_chunk_tokens + sentence_tokens > self.max_chunk_size
            )
            
            if should_split and current_chunk_tokens >= self.min_chunk_size:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append({
                    "text": chunk_text,
                    "doc_id": doc_id,
                    "chunk_id": len(chunks),
                    "num_sentences": len(current_chunk_sentences),
                    "num_tokens": current_chunk_tokens,
                    "boundary_similarity": similarity
                })
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk_sentences,
                    self.overlap_tokens
                )
                current_chunk_sentences = overlap_sentences + [sentences[i]]
                current_chunk_tokens = sum(self._count_tokens(s) for s in current_chunk_sentences)
            else:
                # Continue building current chunk
                current_chunk_sentences.append(sentences[i])
                current_chunk_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append({
                "text": chunk_text,
                "doc_id": doc_id,
                "chunk_id": len(chunks),
                "num_sentences": len(current_chunk_sentences),
                "num_tokens": current_chunk_tokens,
                "boundary_similarity": 1.0  # Last chunk
            })
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str], target_tokens: int) -> List[str]:
        """Get last N sentences that fit within target_tokens for overlap."""
        overlap = []
        token_count = 0
        for s in reversed(sentences):
            s_tokens = self._count_tokens(s)
            if token_count + s_tokens > target_tokens:
                break
            overlap.insert(0, s)
            token_count += s_tokens
        return overlap
    
    def _fallback_chunk(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Fallback to fixed-size chunking if embeddings fail."""
        words = text.split()
        chunks = []
        chunk_size_words = int(self.max_chunk_size / 1.3)
        overlap_words = int(self.overlap_tokens / 1.3)
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size_words]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "text": chunk_text,
                "doc_id": doc_id,
                "chunk_id": len(chunks),
                "num_tokens": len(chunk_words),
                "fallback": True
            })
            i += chunk_size_words - overlap_words
        
        return chunks


class HierarchicalChunker:
    """Creates parent-child document relationships for context preservation.
    
    Retrieves small child chunks for precision, but includes parent context
    for LLM synthesis.
    """
    
    def __init__(
        self,
        semantic_chunker: SemanticChunker,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 500
    ):
        self.semantic_chunker = semantic_chunker
        # Adjust chunker settings for parent vs child
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
    
    def chunk_hierarchical(self, text: str, doc_id: str = "doc") -> Dict[str, Any]:
        """Create hierarchical chunks with parent-child relationships.
        
        Returns:
            {
                "parents": [{"text": str, "parent_id": str, ...}],
                "children": [{"text": str, "child_id": str, "parent_id": str, ...}]
            }
        """
        # Create parent chunks (larger)
        original_max = self.semantic_chunker.max_chunk_size
        self.semantic_chunker.max_chunk_size = self.parent_chunk_size
        parents = self.semantic_chunker.chunk(text, doc_id)
        
        # Create child chunks within each parent
        children = []
        self.semantic_chunker.max_chunk_size = self.child_chunk_size
        
        for parent_idx, parent in enumerate(parents):
            parent_id = f"{doc_id}_parent_{parent_idx}"
            parent["parent_id"] = parent_id
            
            # Chunk parent text into children
            child_chunks = self.semantic_chunker.chunk(parent["text"], doc_id)
            for child in child_chunks:
                child["parent_id"] = parent_id
                child["child_id"] = f"{parent_id}_child_{child['chunk_id']}"
                children.append(child)
        
        # Restore original setting
        self.semantic_chunker.max_chunk_size = original_max
        
        return {
            "parents": parents,
            "children": children
        }
