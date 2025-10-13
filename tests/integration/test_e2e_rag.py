"""Integration tests for end-to-end RAG pipeline."""
import pytest
from typing import Dict, Any


@pytest.mark.integration
class TestEndToEndRAG:
    """Integration tests for complete RAG workflow."""
    
    def test_end_to_end_rag_returns_grounded_answer(self, app_client, mock_embedding, mock_llm, mock_vectordb):
        """Test complete RAG pipeline with grounded answer and citations.
        
        Required test from specification - validates:
        - Query embedding
        - Document retrieval
        - Context assembly
        - LLM generation
        - Citation attachment
        """
        # Index a synthetic doc that contains the answer
        doc_text = "In legal section: 42 is the law"
        doc_embedding = mock_embedding.embed_query(doc_text)
        mock_vectordb.index_document(
            "doc1",
            doc_text,
            doc_embedding,
            metadata={"source": "doc1"}
        )
        
        # Stub embeddings for query
        mock_embedding.stub_embeddings_for("What is 42?", similar_to=doc_text)
        
        # Set LLM to return grounded response
        mock_llm.set_response(
            "42 is the law",
            "According to doc1, 42 is the law. [sources: doc1]"
        )
        
        # Make request
        resp = app_client.post("/query", json={"query": "What is 42?"})
        
        # Assertions
        assert resp.status_code == 200
        data = resp.json()
        
        # Verify answer contains the fact
        assert "42 is the law" in data["answer"]
        
        # Verify sources are cited
        assert "sources" in data
        assert any(s["id"] == "doc1" for s in data["sources"])
        
        # Verify LLM was called
        assert mock_llm.call_count > 0
        
    def test_e2e_with_multiple_sources(self, app_client, mock_embedding, mock_llm, mock_vectordb):
        """Test RAG pipeline retrieves and cites multiple relevant sources."""
        # Index multiple documents
        docs = [
            ("doc1", "Python was created by Guido van Rossum."),
            ("doc2", "Python is a high-level programming language."),
            ("doc3", "Python emphasizes code readability."),
        ]
        
        for doc_id, text in docs:
            embedding = mock_embedding.embed_query(text)
            mock_vectordb.index_document(doc_id, text, embedding, metadata={"source": doc_id})
            
        # Stub query embedding
        query = "Tell me about Python"
        mock_embedding.stub_embeddings_for(query, similar_to="Python programming language")
        
        # Set LLM response
        mock_llm.set_response(
            "Python",
            "Python is a high-level programming language created by Guido van Rossum, "
            "emphasizing code readability. [sources: doc1, doc2, doc3]"
        )
        
        # Make request
        resp = app_client.post("/query", json={"query": query})
        
        assert resp.status_code == 200
        data = resp.json()
        
        # Should retrieve multiple sources
        assert len(data["sources"]) >= 2
        
        # All sources should have doc IDs
        for source in data["sources"]:
            assert "id" in source
            assert source["id"].startswith("doc")
            
    def test_e2e_with_no_relevant_documents(self, app_client, mock_embedding, mock_llm, mock_vectordb):
        """Test RAG handles queries with no relevant documents."""
        # Index unrelated document
        doc_text = "The weather is sunny today."
        doc_embedding = mock_embedding.embed_query(doc_text)
        mock_vectordb.index_document("doc1", doc_text, doc_embedding)
        
        # Query about something else
        query = "What is quantum physics?"
        query_embedding = mock_embedding.embed_query(query)
        
        # Set LLM to handle low-relevance case
        mock_llm.set_response(
            "quantum physics",
            "I don't have specific information about quantum physics in my sources."
        )
        
        resp = app_client.post("/query", json={"query": query})
        
        assert resp.status_code == 200
        data = resp.json()
        
        # Should indicate lack of information
        assert "don't have" in data["answer"].lower() or "no information" in data["answer"].lower()
        
    def test_e2e_with_metadata_context(self, app_client, mock_embedding, mock_llm, mock_vectordb):
        """Test RAG pipeline includes metadata in context."""
        # Index document with rich metadata
        doc_text = "Machine learning is a subset of AI."
        doc_embedding = mock_embedding.embed_query(doc_text)
        mock_vectordb.index_document(
            "doc1",
            doc_text,
            doc_embedding,
            metadata={
                "source": "ai_textbook.pdf",
                "author": "Dr. Smith",
                "date": "2025-01-01",
                "page": 42
            }
        )
        
        mock_embedding.stub_embeddings_for("What is ML?", similar_to=doc_text)
        mock_llm.set_response(
            "Machine learning",
            "According to ai_textbook.pdf (page 42), machine learning is a subset of AI. [sources: doc1]"
        )
        
        resp = app_client.post("/query", json={"query": "What is ML?"})
        
        assert resp.status_code == 200
        data = resp.json()
        
        # Verify metadata is accessible
        assert len(data["sources"]) > 0
        source = data["sources"][0]
        assert "metadata" in source
        
    def test_e2e_latency_within_threshold(self, app_client, mock_embedding, mock_llm, mock_vectordb, test_config):
        """Test E2E latency is within acceptable threshold."""
        import time
        
        # Index document
        doc_text = "Test document for latency."
        doc_embedding = mock_embedding.embed_query(doc_text)
        mock_vectordb.index_document("doc1", doc_text, doc_embedding)
        
        mock_embedding.stub_embeddings_for("test query", similar_to=doc_text)
        mock_llm.set_response("Test", "Test response. [sources: doc1]")
        
        # Measure latency
        start = time.time()
        resp = app_client.post("/query", json={"query": "test query"})
        latency_ms = (time.time() - start) * 1000
        
        assert resp.status_code == 200
        
        # With mocks, should be fast
        threshold_ms = test_config["LATENCY_THRESHOLD_MS"]
        assert latency_ms < threshold_ms, f"Latency {latency_ms}ms exceeds threshold {threshold_ms}ms"
        
    def test_e2e_error_handling(self, app_client, mock_embedding, mock_llm, mock_vectordb):
        """Test E2E pipeline handles errors gracefully."""
        # Set LLM to fail
        mock_llm.set_failure_mode("error")
        
        resp = app_client.post("/query", json={"query": "test query"})
        
        # Should return error status
        assert resp.status_code in [500, 503]
        
        # Should have error message
        data = resp.json()
        assert "error" in data or "detail" in data
        
    def test_e2e_with_empty_query(self, app_client):
        """Test E2E handles empty query gracefully."""
        resp = app_client.post("/query", json={"query": ""})
        
        # Should return bad request
        assert resp.status_code == 400
        
    def test_e2e_with_very_long_query(self, app_client, mock_embedding, mock_llm, mock_vectordb):
        """Test E2E handles very long queries."""
        # Create very long query
        long_query = "What is " + "A" * 10000 + "?"
        
        mock_llm.set_response("A", "The query is too long to process effectively.")
        
        resp = app_client.post("/query", json={"query": long_query})
        
        # Should either handle it or reject gracefully
        assert resp.status_code in [200, 400, 413]
        
        if resp.status_code == 200:
            data = resp.json()
            assert "answer" in data
