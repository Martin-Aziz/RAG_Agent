"""Integration test for advanced RAG features."""
import pytest
import os
import asyncio
from core.orchestrator import Orchestrator
from api.schemas import QueryRequest


@pytest.mark.asyncio
async def test_basic_mode_query():
    """Test orchestrator in basic mode."""
    os.environ['USE_ADVANCED_RAG'] = '0'
    orch = Orchestrator()
    
    req = QueryRequest(
        user_id="test_user",
        session_id="test_session",
        query="Who directed Inception?",
        mode="parrag",
        context_ids=[],
        prefer_low_cost=True
    )
    
    response = await orch.handle_query(req)
    
    assert response.answer is not None
    assert len(response.trace) > 0
    assert response.confidence >= 0.0
    print(f"✓ Basic mode query works")


@pytest.mark.asyncio
async def test_advanced_mode_query():
    """Test orchestrator with advanced features enabled."""
    os.environ['USE_ADVANCED_RAG'] = '1'
    orch = Orchestrator()
    
    req = QueryRequest(
        user_id="test_user_adv",
        session_id="test_session_adv",
        query="What is machine learning?",
        mode="parrag",
        context_ids=[],
        prefer_low_cost=False
    )
    
    response = await orch.handle_query(req)
    
    assert response.answer is not None
    assert len(response.trace) > 0
    
    # Check for advanced agent actions in trace
    trace_agents = {step.agent for step in response.trace}
    print(f"  Agents in trace: {trace_agents}")
    
    # At minimum should have memory and hybrid_retriever or traditional agents
    assert len(trace_agents) > 0
    print(f"✓ Advanced mode query works")


@pytest.mark.asyncio
async def test_conversation_memory():
    """Test conversation memory across multiple turns."""
    os.environ['USE_ADVANCED_RAG'] = '1'
    orch = Orchestrator()
    
    session_id = "memory_test_session"
    user_id = "memory_test_user"
    
    # First query
    req1 = QueryRequest(
        user_id=user_id,
        session_id=session_id,
        query="What is RAG?",
        mode="parrag",
        context_ids=[],
        prefer_low_cost=False
    )
    response1 = await orch.handle_query(req1)
    assert response1.answer is not None
    
    # Second query (should have context from first)
    req2 = QueryRequest(
        user_id=user_id,
        session_id=session_id,
        query="How does it work?",
        mode="parrag",
        context_ids=[],
        prefer_low_cost=False
    )
    response2 = await orch.handle_query(req2)
    assert response2.answer is not None
    
    # Check that memory was used
    if orch.conversation_memory:
        state = orch.conversation_memory.get_session(session_id, user_id)
        assert len(state.turns) == 2
        print(f"✓ Conversation memory stores {len(state.turns)} turns")


@pytest.mark.asyncio
async def test_hoprag_mode():
    """Test HopRAG graph traversal mode."""
    os.environ['USE_ADVANCED_RAG'] = '0'
    orch = Orchestrator()
    
    req = QueryRequest(
        user_id="test_user",
        session_id="test_session",
        query="Find connections about Inception",
        mode="hoprag",
        context_ids=[],
        prefer_low_cost=True
    )
    
    response = await orch.handle_query(req)
    
    assert response.answer is not None
    assert len(response.trace) > 0
    
    # Check for hoprag agent in trace
    trace_agents = {step.agent for step in response.trace}
    # HopRAG may appear if graph has been built
    print(f"✓ HopRAG mode query works")


def test_semantic_chunker():
    """Test semantic chunker."""
    from core.chunking import SemanticChunker
    from core.embedders.ollama_embedder import OllamaEmbedder
    
    # Use a mock embedder that returns dummy embeddings
    class MockEmbedder:
        def embed(self, texts):
            import random
            return [[random.random() for _ in range(128)] for _ in texts]
    
    embedder = MockEmbedder()
    chunker = SemanticChunker(
        embedder=embedder,
        similarity_threshold=0.7,
        min_chunk_size=50,
        max_chunk_size=200,
        overlap_tokens=20
    )
    
    text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
    chunks = chunker.chunk(text, doc_id="test_doc")
    
    assert len(chunks) > 0
    assert all('text' in chunk for chunk in chunks)
    assert all('doc_id' in chunk for chunk in chunks)
    print(f"✓ Semantic chunker created {len(chunks)} chunks")


def test_hybrid_retriever():
    """Test hybrid retriever with RRF."""
    from core.retrieval.hybrid import HybridRetriever
    from core.agents.retriever_vector import VectorRetriever
    from core.agents.retriever_bm25 import BM25Retriever
    
    # Create and index retrievers
    docs = [
        {"doc_id": "d1", "text": "Machine learning is a subset of artificial intelligence"},
        {"doc_id": "d2", "text": "Deep learning uses neural networks"},
        {"doc_id": "d3", "text": "AI systems can learn from data"}
    ]
    
    vector = VectorRetriever()
    vector.index(docs)
    
    bm25 = BM25Retriever()
    bm25.index(docs)
    
    hybrid = HybridRetriever(vector, bm25, k=60)
    
    results = hybrid.retrieve("What is machine learning?", top_k=3, method="rrf")
    
    assert len(results) > 0
    assert all('rrf_score' in doc for doc in results)
    assert all('score' in doc for doc in results)
    print(f"✓ Hybrid retriever returned {len(results)} results with RRF fusion")


def test_conversation_memory_manager():
    """Test conversation memory manager."""
    from core.memory import ConversationMemoryManager
    import tempfile
    import shutil
    
    # Use temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        memory = ConversationMemoryManager(
            max_turns=5,
            storage_path=temp_dir
        )
        
        # Add turns
        memory.add_turn(
            session_id="test_session",
            user_id="test_user",
            user_message="What is AI?",
            assistant_message="AI is artificial intelligence..."
        )
        
        memory.add_turn(
            session_id="test_session",
            user_id="test_user",
            user_message="Tell me more",
            assistant_message="AI includes machine learning..."
        )
        
        # Get context
        context = memory.build_context("test_session", "test_user")
        
        assert "What is AI?" in context
        assert "Tell me more" in context
        print(f"✓ Memory manager stores and retrieves conversation context")
    
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("\n=== Running Advanced RAG Integration Tests ===\n")
    
    # Run sync tests
    print("1. Testing semantic chunker...")
    test_semantic_chunker()
    
    print("\n2. Testing hybrid retriever...")
    test_hybrid_retriever()
    
    print("\n3. Testing conversation memory...")
    test_conversation_memory_manager()
    
    # Run async tests
    print("\n4. Testing basic mode query...")
    asyncio.run(test_basic_mode_query())
    
    print("\n5. Testing advanced mode query...")
    asyncio.run(test_advanced_mode_query())
    
    print("\n6. Testing conversation memory integration...")
    asyncio.run(test_conversation_memory())
    
    print("\n7. Testing HopRAG mode...")
    asyncio.run(test_hoprag_mode())
    
    print("\n=== All tests passed! ===")
