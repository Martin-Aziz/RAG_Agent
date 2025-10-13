"""Advanced query processing: multi-query expansion and HyDE."""
from typing import List, Dict, Any, Optional
import asyncio


class MultiQueryExpander:
    """Generates multiple query reformulations for improved recall.
    
    Creates 3-5 variants with different perspectives, specificity levels,
    and synonyms, then fuses results using RRF.
    """
    
    def __init__(self, model_adapter, num_variants: int = 3):
        """
        Args:
            model_adapter: LLM adapter for query generation
            num_variants: Number of query variants to generate
        """
        self.model = model_adapter
        self.num_variants = num_variants
    
    async def expand_query(self, query: str) -> List[str]:
        """Generate multiple reformulations of the query.
        
        Returns:
            List of query variants (original + generated)
        """
        prompt = f"""Generate {self.num_variants} different ways to ask the following question. 
Each variant should:
1. Use different wording but preserve the core intent
2. Vary in specificity (some more specific, some broader)
3. Use synonyms and related terms

Original question: {query}

Generate {self.num_variants} alternative phrasings, one per line:
"""
        
        try:
            if hasattr(self.model, 'generate_answer_async'):
                response = await self.model.generate_answer_async(prompt, [])
            else:
                response = self.model.generate_answer(prompt, [])
            
            # Parse variants from response
            variants = [query]  # Include original
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                # Remove numbering/bullets
                line = line.lstrip('0123456789.-) ')
                if line and len(line) > 10 and line not in variants:
                    variants.append(line)
            
            return variants[:self.num_variants + 1]
        
        except Exception as e:
            print(f"Query expansion failed: {e}")
            return [query]  # Fallback to original
    
    async def retrieve_with_expansion(
        self,
        query: str,
        retriever,
        k: int = 10,
        fusion_k: int = 60
    ) -> List[Dict[str, Any]]:
        """Retrieve using expanded queries and fuse results with RRF.
        
        Args:
            query: Original query
            retriever: Retriever with retrieve(query, k) method
            k: Documents per query
            fusion_k: RRF constant
        
        Returns:
            Fused and deduplicated results
        """
        # Generate query variants
        variants = await self.expand_query(query)
        
        # Retrieve for each variant in parallel
        retrieval_tasks = [
            asyncio.create_task(self._safe_retrieve(retriever, var, k))
            for var in variants
        ]
        all_results = await asyncio.gather(*retrieval_tasks)
        
        # Reciprocal Rank Fusion across all query results
        return self._reciprocal_rank_fusion(all_results, fusion_k)
    
    async def _safe_retrieve(self, retriever, query: str, k: int) -> List[Dict[str, Any]]:
        """Safe retrieval wrapper."""
        try:
            # Check if retriever has async method
            if hasattr(retriever, 'retrieve_async'):
                return await retriever.retrieve_async(query, k=k)
            else:
                # Sync retriever
                return retriever.retrieve(query, k=k)
        except Exception as e:
            print(f"Retrieval failed for query '{query[:50]}': {e}")
            return []
    
    def _reciprocal_rank_fusion(
        self,
        results_list: List[List[Dict[str, Any]]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """Fuse multiple retrieval results using RRF."""
        # Build document key -> doc mapping and rank lists
        all_docs = {}
        doc_ranks = {}  # doc_key -> [rank_in_query_0, rank_in_query_1, ...]
        
        for query_results in results_list:
            for rank, doc in enumerate(query_results):
                doc_key = self._get_doc_key(doc)
                if doc_key not in all_docs:
                    all_docs[doc_key] = doc
                if doc_key not in doc_ranks:
                    doc_ranks[doc_key] = []
                doc_ranks[doc_key].append(rank)
        
        # Compute RRF scores
        scored = []
        for doc_key, doc in all_docs.items():
            ranks = doc_ranks[doc_key]
            rrf_score = sum(1.0 / (k + rank) for rank in ranks)
            doc_copy = doc.copy()
            doc_copy["rrf_score"] = rrf_score
            doc_copy["score"] = rrf_score
            doc_copy["num_queries_found"] = len(ranks)
            scored.append(doc_copy)
        
        # Sort by RRF score
        scored.sort(key=lambda x: x["rrf_score"], reverse=True)
        return scored
    
    def _get_doc_key(self, doc: Dict[str, Any]) -> str:
        """Generate unique key for document."""
        return f"{doc.get('doc_id', 'unknown')}_{doc.get('passage_id', 'p0')}"


class HyDERetriever:
    """Hypothetical Document Embeddings (HyDE) for improved semantic matching.
    
    Generates a hypothetical answer to the query using LLM, then uses
    that answer's embedding for retrieval instead of the query.
    """
    
    def __init__(self, model_adapter, embedder):
        """
        Args:
            model_adapter: LLM for generating hypothetical answer
            embedder: Embedding model
        """
        self.model = model_adapter
        self.embedder = embedder
    
    async def generate_hypothetical_document(self, query: str) -> str:
        """Generate hypothetical answer that would answer the query."""
        prompt = f"""Generate a detailed, factual answer to this question as if you had perfect knowledge. 
Write in a encyclopedic style with specific facts and details.

Question: {query}

Hypothetical answer (2-3 sentences with specific facts):"""
        
        try:
            if hasattr(self.model, 'generate_answer_async'):
                response = await self.model.generate_answer_async(prompt, [])
            else:
                response = self.model.generate_answer(prompt, [])
            
            return response.strip()
        
        except Exception as e:
            print(f"HyDE generation failed: {e}")
            return query  # Fallback to original query
    
    async def retrieve_with_hyde(
        self,
        query: str,
        retriever,
        k: int = 10,
        use_both: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve using HyDE approach.
        
        Args:
            query: Original query
            retriever: Vector retriever with retrieve method
            k: Number of results
            use_both: If True, combine query and hypothetical doc results
        
        Returns:
            Retrieved documents
        """
        # Generate hypothetical document
        hypo_doc = await self.generate_hypothetical_document(query)
        
        if use_both:
            # Retrieve with both query and hypothetical doc, then merge
            query_results = retriever.retrieve(query, k=k)
            hypo_results = retriever.retrieve(hypo_doc, k=k)
            
            # Deduplicate and merge scores
            merged = self._merge_results(query_results, hypo_results)
            return merged[:k]
        else:
            # Only use hypothetical document
            return retriever.retrieve(hypo_doc, k=k)
    
    def _merge_results(
        self,
        results_a: List[Dict[str, Any]],
        results_b: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge two result lists, averaging scores for duplicates."""
        doc_map = {}
        
        for doc in results_a:
            key = self._get_doc_key(doc)
            doc_map[key] = {"doc": doc, "scores": [doc.get("score", 0.0)]}
        
        for doc in results_b:
            key = self._get_doc_key(doc)
            if key in doc_map:
                doc_map[key]["scores"].append(doc.get("score", 0.0))
            else:
                doc_map[key] = {"doc": doc, "scores": [doc.get("score", 0.0)]}
        
        # Average scores
        merged = []
        for key, data in doc_map.items():
            doc = data["doc"].copy()
            doc["score"] = sum(data["scores"]) / len(data["scores"])
            doc["hyde_merged"] = len(data["scores"]) > 1
            merged.append(doc)
        
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged
    
    def _get_doc_key(self, doc: Dict[str, Any]) -> str:
        """Generate unique key for document."""
        return f"{doc.get('doc_id', 'unknown')}_{doc.get('passage_id', 'p0')}"
