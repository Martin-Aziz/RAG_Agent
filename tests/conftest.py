"""Root conftest.py for RAG Agent test suite.

This module provides shared fixtures, mocks, and configuration for all tests.
"""
import os
import sys
import pytest
import tempfile
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import test fixtures
from tests.fixtures.mock_llm import MockLLM
from tests.fixtures.mock_embeddings import MockEmbeddings
from tests.fixtures.mock_vectordb import MockVectorDB
from tests.fixtures.synthetic_corpus_generator import generate_synthetic_corpus

# Test configuration from environment
TEST_CONFIG = {
    "RAG_API_BASE_URL": os.getenv("RAG_API_BASE_URL", "http://localhost:8000"),
    "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "test-embeddings-v1"),
    "LLM_MODEL": os.getenv("LLM_MODEL", "gpt-5-codecs-test"),
    "VECTOR_DB": os.getenv("VECTOR_DB", "faiss"),
    "TOP_K": int(os.getenv("TOP_K", "5")),
    "CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", "1000")),
    "CHUNK_OVERLAP": int(os.getenv("CHUNK_OVERLAP", "200")),
    "SIMILARITY_THRESHOLD": float(os.getenv("SIMILARITY_THRESHOLD", "0.85")),
    "F1_THRESHOLD": float(os.getenv("F1_THRESHOLD", "0.75")),
    "LATENCY_THRESHOLD_MS": int(os.getenv("LATENCY_THRESHOLD_MS", "800")),
    "COVERAGE_THRESHOLD": int(os.getenv("COVERAGE_THRESHOLD", "85")),
}


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return TEST_CONFIG


@pytest.fixture(scope="session")
def temp_data_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def synthetic_corpus(temp_data_dir):
    """Generate synthetic document corpus for tests."""
    corpus_dir = temp_data_dir / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    documents = generate_synthetic_corpus(corpus_dir, num_docs=200)
    return documents


@pytest.fixture
def mock_llm():
    """Provide mock LLM client."""
    return MockLLM()


@pytest.fixture
def mock_embeddings():
    """Provide mock embeddings service."""
    return MockEmbeddings(dimension=768)


@pytest.fixture
def mock_vectordb(mock_embeddings):
    """Provide mock vector database."""
    return MockVectorDB(embedding_dim=768)


@pytest.fixture
def app_client(mock_llm, mock_embeddings, mock_vectordb, monkeypatch):
    """Provide FastAPI test client with mocked dependencies."""
    from fastapi.testclient import TestClient
    from api.main import app
    
    # Patch dependencies with mocks
    monkeypatch.setattr("core.model_adapters.OllamaAdapter", lambda: mock_llm)
    monkeypatch.setattr("core.embedders.ollama_embedder.OllamaEmbedder", lambda **kw: mock_embeddings)
    
    client = TestClient(app)
    return client


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        {
            "id": "doc1",
            "text": "Python is a high-level programming language. It was created by Guido van Rossum in 1991.",
            "metadata": {"source": "programming_guide.pdf", "page": 1, "author": "Tech Writer"}
        },
        {
            "id": "doc2",
            "text": "Machine learning is a subset of artificial intelligence. It focuses on learning from data.",
            "metadata": {"source": "ml_intro.pdf", "page": 1, "author": "AI Researcher"}
        },
        {
            "id": "doc3",
            "text": "The legal system protects individual rights. Section 42 defines fundamental freedoms.",
            "metadata": {"source": "legal_code.pdf", "page": 42, "author": "Legal Council"}
        },
    ]


@pytest.fixture
def sample_query():
    """Provide sample query for testing."""
    return "What is Python and who created it?"


@pytest.fixture
def golden_qa_pairs():
    """Provide golden question-answer pairs for regression testing."""
    return [
        {
            "query": "What is Python?",
            "expected_answer": "Python is a high-level programming language created by Guido van Rossum in 1991.",
            "required_sources": ["doc1"],
            "similarity_threshold": 0.85
        },
        {
            "query": "What is machine learning?",
            "expected_answer": "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
            "required_sources": ["doc2"],
            "similarity_threshold": 0.85
        },
        {
            "query": "What does Section 42 define?",
            "expected_answer": "Section 42 defines fundamental freedoms in the legal code.",
            "required_sources": ["doc3"],
            "similarity_threshold": 0.85
        },
    ]


@pytest.fixture(autouse=True)
def reset_mocks(mock_llm, mock_embeddings, mock_vectordb):
    """Reset all mocks before each test."""
    yield
    mock_llm.reset()
    mock_embeddings.reset()
    mock_vectordb.reset()


@pytest.fixture
def mock_auth_token():
    """Provide mock authentication token."""
    return "test_token_12345"


@pytest.fixture
def mock_user():
    """Provide mock user with permissions."""
    return {
        "user_id": "test_user_001",
        "roles": ["user", "reader"],
        "permissions": ["read_docs", "query_rag"]
    }


# Pytest hooks for custom behavior
def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "requires_external: mark test as requiring external services"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip external tests by default."""
    skip_external = pytest.mark.skip(reason="Requires external services (use --run-external)")
    
    for item in items:
        if "requires_external" in item.keywords and not config.getoption("--run-external", default=False):
            item.add_marker(skip_external)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-external",
        action="store_true",
        default=False,
        help="Run tests that require external services"
    )
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="Run performance tests (slow)"
    )


# Helper functions for test assertions
def assert_valid_embedding(embedding, expected_dim=768):
    """Assert that embedding is valid."""
    import numpy as np
    assert isinstance(embedding, (list, np.ndarray))
    if isinstance(embedding, list):
        embedding = np.array(embedding)
    assert embedding.shape[0] == expected_dim
    assert not np.isnan(embedding).any()
    assert not np.isinf(embedding).any()


def assert_valid_response_schema(response_data: Dict[str, Any]):
    """Assert that API response matches expected schema."""
    assert "answer" in response_data
    assert isinstance(response_data["answer"], str)
    assert len(response_data["answer"]) > 0
    
    if "sources" in response_data:
        assert isinstance(response_data["sources"], list)
        for source in response_data["sources"]:
            assert "id" in source
            assert "score" in source or "relevance" in source
    
    if "metadata" in response_data:
        assert isinstance(response_data["metadata"], dict)


def compute_semantic_similarity(text1: str, text2: str, mock_embeddings) -> float:
    """Compute semantic similarity between two texts using mock embeddings."""
    import numpy as np
    emb1 = np.array(mock_embeddings.embed_query(text1))
    emb2 = np.array(mock_embeddings.embed_query(text2))
    
    # Cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(similarity)


def compute_token_f1(reference: str, candidate: str) -> float:
    """Compute token-level F1 score."""
    ref_tokens = set(reference.lower().split())
    cand_tokens = set(candidate.lower().split())
    
    if len(cand_tokens) == 0:
        return 0.0
    
    overlap = ref_tokens & cand_tokens
    precision = len(overlap) / len(cand_tokens)
    recall = len(overlap) / len(ref_tokens) if len(ref_tokens) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
