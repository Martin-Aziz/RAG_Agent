"""Hybrid retrieval with BM25 + Dense vector search and RRF fusion."""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
from rank_bm25 import BM25Okapi
import numpy as np


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    doc_id: str
    passage_id: str
    text: str
    score: float
    source: str  # 'bm25', 'vector', 'fused'
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HybridRetriever:
    """Hybrid retrieval combining BM25 and dense vector search with RRF fusion."""
    
    def __init__(
        self,
        bm25_retriever,
        vector_retriever,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        rrf_k: int = 60,
        candidate_pool_size: int = 50,
        timeout_seconds: int = 10
    ):
        """Initialize hybrid retriever.
        
        Args:
            bm25_retriever: BM25 retriever instance
            vector_retriever: Vector retriever instance
            bm25_weight: Weight for BM25 scores (used in linear fusion fallback)
            vector_weight: Weight for vector scores
            rrf_k: RRF constant (typically 60)
            candidate_pool_size: Number of candidates to retrieve before fusion
            timeout_seconds: Timeout for retrieval operations
        """
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k
        self.candidate_pool_size = candidate_pool_size
        self.timeout = timeout_seconds
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_rrf: bool = True
    ) -> List[RetrievalResult]:
        """Retrieve documents using hybrid search with RRF fusion.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            use_rrf: Use RRF fusion (True) or linear fusion (False)
        
        Returns:
            List of retrieval results sorted by fused score
        """
        # Retrieve from both sources in parallel with timeout
        try:
            bm25_results, vector_results = await asyncio.wait_for(
                asyncio.gather(
                    self._retrieve_bm25(query),
                    self._retrieve_vector(query)
                ),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            # Fallback to vector only
            vector_results = await self._retrieve_vector(query)
            bm25_results = []
        
        # Fuse results
        if use_rrf:
            fused = self._reciprocal_rank_fusion(bm25_results, vector_results)
        else:
            fused = self._linear_fusion(bm25_results, vector_results)
        
        return fused[:top_k]
    
    async def _retrieve_bm25(self, query: str) -> List[RetrievalResult]:
        """Retrieve using BM25."""
        try:
            raw_results = self.bm25.retrieve(query, k=self.candidate_pool_size)
            return [
                RetrievalResult(
                    doc_id=r.get("doc_id", "unknown"),
                    passage_id=r.get("passage_id", "p0"),
                    text=r.get("text", ""),
                    score=r.get("score", 0.0),
                    source="bm25"
                )
                for r in raw_results
            ]
        except Exception as e:
            print(f"BM25 retrieval failed: {e}")
            return []
    
    async def _retrieve_vector(self, query: str) -> List[RetrievalResult]:
        """Retrieve using vector search."""
        try:
            raw_results = self.vector.retrieve(query, k=self.candidate_pool_size)
            return [
                RetrievalResult(
                    doc_id=r.get("doc_id", "unknown"),
                    passage_id=r.get("passage_id", "p0"),
                    text=r.get("text", ""),
                    score=r.get("score", 0.0),
                    source="vector"
                )
                for r in raw_results
            ]
        except Exception as e:
            print(f"Vector retrieval failed: {e}")
            return []
    
    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[RetrievalResult],
        vector_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Fuse results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank_i)) across all result lists
        """
        # Build rank mappings
        bm25_ranks = {
            self._get_doc_key(r): idx
            for idx, r in enumerate(bm25_results)
        }
        vector_ranks = {
            self._get_doc_key(r): idx
            for idx, r in enumerate(vector_results)
        }
        
        # Collect all unique documents
        doc_map = {}
        for r in bm25_results:
            key = self._get_doc_key(r)
            doc_map[key] = r
        for r in vector_results:
            key = self._get_doc_key(r)
            if key not in doc_map:
                doc_map[key] = r
        
        # Compute RRF scores
        fused = []
        for key, doc in doc_map.items():
            rrf_score = 0.0
            
            if key in bm25_ranks:
                rrf_score += 1.0 / (self.rrf_k + bm25_ranks[key])
            
            if key in vector_ranks:
                rrf_score += 1.0 / (self.rrf_k + vector_ranks[key])
            
            fused_doc = RetrievalResult(
                doc_id=doc.doc_id,
                passage_id=doc.passage_id,
                text=doc.text,
                score=rrf_score,
                source="fused",
                metadata={
                    "bm25_rank": bm25_ranks.get(key),
                    "vector_rank": vector_ranks.get(key),
                    "rrf_score": rrf_score
                }
            )
            fused.append(fused_doc)
        
        # Sort by RRF score descending
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused
    
    def _linear_fusion(
        self,
        bm25_results: List[RetrievalResult],
        vector_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Fuse results using linear combination of normalized scores."""
        # Normalize scores
        bm25_normalized = self._normalize_scores(bm25_results)
        vector_normalized = self._normalize_scores(vector_results)
        
        # Build score maps
        bm25_scores = {
            self._get_doc_key(r): r.score
            for r in bm25_normalized
        }
        vector_scores = {
            self._get_doc_key(r): r.score
            for r in vector_normalized
        }
        
        # Collect all unique documents
        doc_map = {}
        for r in bm25_results:
            key = self._get_doc_key(r)
            doc_map[key] = r
        for r in vector_results:
            key = self._get_doc_key(r)
            if key not in doc_map:
                doc_map[key] = r
        
        # Compute weighted scores
        fused = []
        for key, doc in doc_map.items():
            bm25_score = bm25_scores.get(key, 0.0)
            vector_score = vector_scores.get(key, 0.0)
            
            combined_score = (
                self.bm25_weight * bm25_score +
                self.vector_weight * vector_score
            )
            
            fused_doc = RetrievalResult(
                doc_id=doc.doc_id,
                passage_id=doc.passage_id,
                text=doc.text,
                score=combined_score,
                source="fused",
                metadata={
                    "bm25_score": bm25_score,
                    "vector_score": vector_score,
                    "combined_score": combined_score
                }
            )
            fused.append(fused_doc)
        
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused
    
    def _normalize_scores(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Min-max normalize scores to [0, 1]."""
        if not results:
            return results
        
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores equal
            for r in results:
                r.score = 1.0
            return results
        
        normalized = []
        for r in results:
            normalized_score = (r.score - min_score) / (max_score - min_score)
            normalized_result = RetrievalResult(
                doc_id=r.doc_id,
                passage_id=r.passage_id,
                text=r.text,
                score=normalized_score,
                source=r.source,
                metadata=r.metadata
            )
            normalized.append(normalized_result)
        
        return normalized
    
    def _get_doc_key(self, result: RetrievalResult) -> str:
        """Generate unique key for document."""
        return f"{result.doc_id}_{result.passage_id}"


class AdaptiveFusionRetriever(HybridRetriever):
    """Hybrid retriever with adaptive fusion based on query characteristics."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_type_classifier = None  # Can be added later
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_rrf: bool = None  # Auto-select if None
    ) -> List[RetrievalResult]:
        """Retrieve with adaptive fusion strategy."""
        # Auto-select fusion method based on query
        if use_rrf is None:
            use_rrf = self._should_use_rrf(query)
        
        return await super().retrieve(query, top_k, use_rrf)
    
    def _should_use_rrf(self, query: str) -> bool:
        """Determine if RRF should be used based on query characteristics.
        
        Heuristic: Use RRF for longer, more complex queries.
        Use linear fusion for short, keyword-focused queries.
        """
        words = query.split()
        
        # Use RRF for longer queries (better for complex multi-hop)
        if len(words) > 10:
            return True
        
        # Use linear fusion for short queries (better for keyword matching)
        return False
