"""Hybrid search with Reciprocal Rank Fusion (RRF) and cross-encoder reranking."""
from typing import List, Dict, Any, Optional
import math


class HybridRetriever:
    """Combines BM25 and vector retrieval using Reciprocal Rank Fusion.
    
    Runs both retrievers in parallel and fuses scores using RRF algorithm
    for optimal ranking combining lexical and semantic signals.
    """
    
    def __init__(self, vector_retriever, bm25_retriever, k: int = 60, alpha: float = 0.5):
        """
        Args:
            vector_retriever: Semantic vector retriever
            bm25_retriever: Lexical BM25 retriever
            k: RRF constant (typically 60)
            alpha: Weight for linear fusion fallback (0=all BM25, 1=all vector)
        """
        self.vector = vector_retriever
        self.bm25 = bm25_retriever
        self.k = k
        self.alpha = alpha
    
    def retrieve(self, query: str, top_k: int = 10, method: str = "rrf") -> List[Dict[str, Any]]:
        """Retrieve using hybrid search with score fusion.
        
        Args:
            query: Search query
            top_k: Number of results to return
            method: Fusion method - "rrf" (Reciprocal Rank Fusion) or "linear" (weighted)
        
        Returns:
            List of documents with fused scores
        """
        # Retrieve from both in parallel (async would be better in production)
        # Retrieve more candidates for better fusion
        candidate_k = min(top_k * 3, 50)
        
        vector_results = self.vector.retrieve(query, k=candidate_k)
        bm25_results = self.bm25.retrieve(query, k=candidate_k)
        
        if method == "rrf":
            return self._reciprocal_rank_fusion(vector_results, bm25_results, top_k)
        elif method == "linear":
            return self._linear_fusion(vector_results, bm25_results, top_k)
        else:
            raise ValueError(f"Unknown fusion method: {method}")
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Implement Reciprocal Rank Fusion: score = sum(1/(k + rank_i))."""
        # Build rank maps
        vector_ranks = {
            self._get_doc_key(doc): idx for idx, doc in enumerate(vector_results)
        }
        bm25_ranks = {
            self._get_doc_key(doc): idx for idx, doc in enumerate(bm25_results)
        }
        
        # Collect all unique documents
        all_docs = {}
        for doc in vector_results:
            key = self._get_doc_key(doc)
            all_docs[key] = doc
        for doc in bm25_results:
            key = self._get_doc_key(doc)
            if key not in all_docs:
                all_docs[key] = doc
        
        # Compute RRF scores
        scored = []
        for key, doc in all_docs.items():
            rrf_score = 0.0
            
            # Add vector contribution
            if key in vector_ranks:
                rrf_score += 1.0 / (self.k + vector_ranks[key])
            
            # Add BM25 contribution
            if key in bm25_ranks:
                rrf_score += 1.0 / (self.k + bm25_ranks[key])
            
            doc_copy = doc.copy()
            doc_copy["rrf_score"] = rrf_score
            doc_copy["score"] = rrf_score  # Unify score field
            doc_copy["fusion_method"] = "rrf"
            scored.append(doc_copy)
        
        # Sort by RRF score descending
        scored.sort(key=lambda x: x["rrf_score"], reverse=True)
        return scored[:top_k]
    
    def _linear_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Linear combination: score = α·vector + (1-α)·bm25."""
        # Normalize scores to [0, 1]
        vector_normalized = self._normalize_scores(vector_results)
        bm25_normalized = self._normalize_scores(bm25_results)
        
        # Build score maps
        vector_scores = {
            self._get_doc_key(doc): doc["normalized_score"]
            for doc in vector_normalized
        }
        bm25_scores = {
            self._get_doc_key(doc): doc["normalized_score"]
            for doc in bm25_normalized
        }
        
        # Collect all unique documents
        all_docs = {}
        for doc in vector_results:
            key = self._get_doc_key(doc)
            all_docs[key] = doc
        for doc in bm25_results:
            key = self._get_doc_key(doc)
            if key not in all_docs:
                all_docs[key] = doc
        
        # Compute linear fusion
        scored = []
        for key, doc in all_docs.items():
            v_score = vector_scores.get(key, 0.0)
            b_score = bm25_scores.get(key, 0.0)
            
            fused_score = self.alpha * v_score + (1 - self.alpha) * b_score
            
            doc_copy = doc.copy()
            doc_copy["fused_score"] = fused_score
            doc_copy["score"] = fused_score
            doc_copy["fusion_method"] = "linear"
            scored.append(doc_copy)
        
        scored.sort(key=lambda x: x["fused_score"], reverse=True)
        return scored[:top_k]
    
    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Min-max normalize scores to [0, 1]."""
        if not results:
            return []
        
        scores = [doc.get("score", 0.0) for doc in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores equal
            for doc in results:
                doc["normalized_score"] = 1.0
            return results
        
        for doc in results:
            original = doc.get("score", 0.0)
            doc["normalized_score"] = (original - min_score) / (max_score - min_score)
        
        return results
    
    def _get_doc_key(self, doc: Dict[str, Any]) -> str:
        """Generate unique key for document."""
        return f"{doc.get('doc_id', 'unknown')}_{doc.get('passage_id', 'p0')}"


class CrossEncoderReranker:
    """Rerank retrieved documents using cross-encoder model.
    
    Cross-encoders jointly encode query+document pairs for superior
    relevance scoring (92%+ accuracy) compared to bi-encoders.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        """
        Args:
            model_name: HuggingFace cross-encoder model
            device: 'cpu' or 'cuda'
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Lazy load cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device=self.device)
        except ImportError:
            print("Warning: sentence-transformers not installed. Reranking will use fallback scoring.")
            self.model = None
        except Exception as e:
            print(f"Warning: Failed to load cross-encoder: {e}")
            self.model = None
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder.
        
        Args:
            query: Search query
            documents: List of retrieved documents
            top_k: Return top K after reranking (None = return all)
        
        Returns:
            Documents sorted by cross-encoder score
        """
        if not documents:
            return []
        
        if self.model is None:
            # Fallback: return documents as-is
            return documents[:top_k] if top_k else documents
        
        # Prepare query-document pairs
        pairs = [[query, doc.get("text", "")] for doc in documents]
        
        # Score pairs
        try:
            scores = self.model.predict(pairs)
            
            # Attach scores to documents
            for doc, score in zip(documents, scores):
                doc["cross_encoder_score"] = float(score)
                doc["score"] = float(score)  # Override with reranker score
            
            # Sort by cross-encoder score
            documents.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
            
            return documents[:top_k] if top_k else documents
        
        except Exception as e:
            print(f"Warning: Cross-encoder reranking failed: {e}")
            return documents[:top_k] if top_k else documents
