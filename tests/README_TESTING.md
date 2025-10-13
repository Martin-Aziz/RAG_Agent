# 🧪 RAG Agent Test Suite Documentation

## Overview

This comprehensive test suite validates all functionality of the RAG Agent chatbot system. It includes unit tests, integration tests, black-box API tests, behavioral tests, adversarial/security tests, performance tests, and monitoring tests.

## 📋 Test Coverage

| Category | Test Count | Coverage | Status |
|----------|-----------|----------|--------|
| **Unit Tests** | 50+ | Component-level | ✅ Complete |
| **Integration Tests** | 25+ | Pipeline-level | ✅ Complete |
| **Black-box/API Tests** | 20+ | HTTP endpoints | ✅ Complete |
| **Behavioral Tests** | 15+ | Model quality | ✅ Complete |
| **Adversarial Tests** | 10+ | Security | ✅ Complete |
| **Performance Tests** | 8+ | Load/throughput | ✅ Complete |
| **Monitoring Tests** | 5+ | Observability | ✅ Complete |
| **Regression Tests** | Golden queries | Stability | ✅ Complete |

**Total Test Coverage**: **>85%** (enforced by CI)

---

## 🚀 Quick Start

### Installation

```bash
# Install test dependencies
cd /path/to/RAG_Agent
pip install -r tests/requirements.txt

# Generate synthetic test corpus (optional - auto-generated if missing)
python tests/fixtures/synthetic_corpus_generator.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=core --cov=app --cov=api --cov-report=html

# Run specific test category
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests only
pytest tests/behavior/                # Behavioral tests only

# Run tests matching pattern
pytest -k "test_chunker"              # All chunker tests
pytest -k "test_e2e"                  # All end-to-end tests

# Run with specific markers
pytest -m unit                        # Unit tests
pytest -m "not slow"                  # Exclude slow tests
pytest -m "behavior and not flaky"    # Behavioral, exclude flaky

# Run in parallel (faster)
pytest -n auto                        # Auto-detect CPU count
pytest -n 4                           # Use 4 workers

# Run with verbose output
pytest -v                             # Verbose
pytest -vv                            # Extra verbose
pytest -s                             # Show print statements
```

---

## 📁 Test Structure

```
tests/
├── conftest.py                      # Shared fixtures and configuration
├── pytest.ini                       # Pytest configuration
├── requirements.txt                 # Test dependencies
│
├── unit/                            # Component-level tests
│   ├── test_chunker.py             # Text chunking (10 tests)
│   ├── test_embedding_wrapper.py   # Embeddings (12 tests)
│   ├── test_vectordb_client.py     # Vector DB operations (15 tests)
│   ├── test_retriever.py           # Document retrieval (12 tests)
│   ├── test_reranker.py            # Result reranking (8 tests)
│   ├── test_document_parser.py     # PDF/TXT parsing (10 tests)
│   ├── test_citation_generator.py  # Citation formatting (6 tests)
│   ├── test_response_formatter.py  # JSON schema (8 tests)
│   └── test_rate_limiter.py        # Rate limiting (7 tests)
│
├── integration/                     # Pipeline-level tests
│   ├── test_e2e_rag.py             # End-to-end RAG flow (8 tests)
│   ├── test_multiturn_context.py   # Conversation state (6 tests)
│   ├── test_long_document_handling.py  # Large docs (5 tests)
│   ├── test_document_upload.py     # Upload & indexing (7 tests)
│   └── test_metadata_filtering.py  # Metadata queries (6 tests)
│
├── blackbox/                        # HTTP API tests
│   ├── test_api_contract.py        # Endpoint schemas (12 tests)
│   ├── test_auth_and_rate_limits.py  # Auth & limits (10 tests)
│   ├── test_concurrency.py         # Parallel requests (5 tests)
│   └── test_fault_injection.py     # Failure handling (8 tests)
│
├── behavior/                        # Model quality tests
│   ├── test_grounding_and_citations.py   # Grounding (8 tests)
│   ├── test_hallucination_detection.py   # Hallucination (6 tests)
│   ├── test_conservative_answering.py    # Safe responses (5 tests)
│   ├── test_consistency.py               # Answer stability (4 tests)
│   └── test_tone_and_policy.py          # Safety policy (6 tests)
│
├── adversarial/                     # Security tests
│   ├── test_prompt_injection.py    # Injection attacks (8 tests)
│   ├── test_pii_leakage.py         # PII protection (6 tests)
│   ├── test_input_fuzzing.py       # Malformed inputs (10 tests)
│   └── test_access_control.py      # Permission checks (7 tests)
│
├── performance/                     # Load & performance
│   ├── test_throughput.py          # Queries/sec (4 tests)
│   ├── test_cache_performance.py   # Cache effectiveness (5 tests)
│   ├── test_memory_leak.py         # Resource leaks (3 tests)
│   └── load_test_k6_script.js      # K6 load test
│
├── monitoring/                      # Observability
│   ├── test_telemetry.py           # Telemetry events (6 tests)
│   └── test_health_check.py        # Health endpoints (4 tests)
│
├── regression/                      # Stability tests
│   ├── test_golden_queries.py      # Golden query set (20 queries)
│   └── golden_queries.json         # Expected answers
│
├── fixtures/                        # Test utilities
│   ├── conftest.py                 # Fixture definitions
│   ├── mock_llm.py                 # Mock LLM client
│   ├── mock_embeddings.py          # Mock embeddings
│   ├── mock_vectordb.py            # Mock vector DB
│   ├── synthetic_corpus_generator.py  # Test data generator
│   └── corpus/                     # Generated test documents
│
└── scripts/                         # Test utilities
    ├── test_reporter.py            # Aggregate test metrics
    ├── generate_test_data.py       # Data generation
    ├── mutation_test_config.py     # Mutation testing
    ├── canary_tests.py             # Deployment canaries
    ├── human_review_harness.py     # Manual review UI
    ├── dataset_drift_monitor.py    # Corpus drift detection
    └── auto_repair_suggestions.py  # Test failure guidance
```

---

## 🎯 Test Categories Explained

### 1. Unit Tests (Component-Level)

Tests individual components in isolation with mocked dependencies.

**Key Tests:**
- `test_chunker.py` - Text chunking with overlap and metadata preservation
- `test_embedding_wrapper.py` - Embedding generation, shape validation, determinism
- `test_vectordb_client.py` - Index/search/delete operations, error handling
- `test_retriever.py` - Document retrieval, top_k, score ordering
- `test_reranker.py` - Result reordering based on relevance

**Run:** `pytest tests/unit/ -m unit`

### 2. Integration Tests (Pipeline-Level)

Tests complete workflows with mocked external services.

**Key Tests:**
- `test_e2e_rag.py` - Full RAG pipeline: query → embed → retrieve → generate → cite
- `test_multiturn_context.py` - Conversation state management
- `test_long_document_handling.py` - Documents >10k tokens
- `test_document_upload.py` - PDF/TXT upload → index → retrieve
- `test_metadata_filtering.py` - Filter by date/author/source

**Run:** `pytest tests/integration/ -m integration`

### 3. Black-box/API Tests

Tests HTTP endpoints without knowledge of internal implementation.

**Key Tests:**
- `test_api_contract.py` - Schema validation, status codes, headers
- `test_auth_and_rate_limits.py` - Authentication, authorization, rate limiting
- `test_concurrency.py` - Parallel requests, race conditions
- `test_fault_injection.py` - Graceful degradation when services fail

**Run:** `pytest tests/blackbox/ -m blackbox`

### 4. Behavioral Tests

Tests model output quality and grounding accuracy.

**Key Tests:**
- `test_grounding_and_citations.py` - Verifies answers cite correct sources
- `test_hallucination_detection.py` - Detects fabricated information
- `test_conservative_answering.py` - Safe refusals for sensitive queries
- `test_consistency.py` - Stable answers for repeated queries
- `test_tone_and_policy.py` - Appropriate tone and safety policy

**Run:** `pytest tests/behavior/ -m behavior`

**Note:** Some behavioral tests are marked `@pytest.mark.flaky` due to acceptable model variance. CI allows 1 retry for flaky tests.

### 5. Adversarial/Security Tests

Tests system resilience against attacks and abuse.

**Key Tests:**
- `test_prompt_injection.py` - Prevents system prompt override
- `test_pii_leakage.py` - Protects personal information
- `test_input_fuzzing.py` - Handles malformed inputs gracefully
- `test_access_control.py` - Enforces document permissions

**Run:** `pytest tests/adversarial/ -m adversarial`

### 6. Performance Tests

Tests system performance, throughput, and resource usage.

**Key Tests:**
- `test_throughput.py` - Measures queries/second
- `test_cache_performance.py` - Cold vs. warm cache latency
- `test_memory_leak.py` - Detects memory leaks over time
- `load_test_k6_script.js` - K6 load testing script

**Run:** `pytest tests/performance/ -m performance`

**Note:** Performance tests are marked `@pytest.mark.slow` and may be skipped in quick test runs.

### 7. Monitoring Tests

Tests observability, telemetry, and health checks.

**Key Tests:**
- `test_telemetry.py` - Verifies telemetry events are emitted
- `test_health_check.py` - Health endpoint returns subsystem status

**Run:** `pytest tests/monitoring/ -m monitoring`

### 8. Regression Tests

Golden query tests that ensure stable behavior across deployments.

**Key Tests:**
- `test_golden_queries.py` - 20 golden queries with expected answers
- Compares semantic similarity (threshold: 0.85) and F1 score (threshold: 0.75)

**Run:** `pytest tests/regression/ -m regression`

---

## ⚙️ Configuration

### Environment Variables

Configure test behavior via environment variables:

```bash
# API Configuration
export RAG_API_BASE_URL="http://localhost:8000"

# Model Configuration
export EMBEDDING_MODEL="test-embeddings-v1"
export LLM_MODEL="gpt-5-codecs-test"
export VECTOR_DB="faiss"  # faiss, weaviate, pinecone

# Retrieval Configuration
export TOP_K=5
export CHUNK_SIZE=1000
export CHUNK_OVERLAP=200

# Test Thresholds
export SIMILARITY_THRESHOLD=0.85     # Semantic similarity
export F1_THRESHOLD=0.75             # Token F1 score
export LATENCY_THRESHOLD_MS=800      # Max latency (mocked tests)
export COVERAGE_THRESHOLD=85         # Minimum code coverage %

# Test Behavior
export PYTEST_WORKERS=auto           # Parallel workers
export SKIP_SLOW_TESTS=false         # Skip performance tests
export SKIP_EXTERNAL_TESTS=true      # Skip tests requiring external services
```

### pytest.ini Configuration

Key settings in `pytest.ini`:

```ini
[pytest]
markers =
    unit: Unit tests
    integration: Integration tests
    blackbox: API tests
    behavior: Behavioral tests
    adversarial: Security tests
    performance: Performance tests
    slow: Tests taking >1 second
    flaky: Tests with acceptable variance

addopts =
    -v
    --cov=core --cov=app --cov=api
    --cov-report=html:reports/coverage_html
    --junitxml=reports/junit.xml
    --html=reports/test_report.html

[coverage:report]
fail_under = 85  # Minimum 85% coverage
```

---

## 📊 Test Metrics & Thresholds

### Coverage Thresholds

| Component | Target | Enforced |
|-----------|--------|----------|
| Overall | ≥85% | ✅ Yes |
| Core modules | ≥90% | ✅ Yes |
| API endpoints | ≥80% | ✅ Yes |
| Utilities | ≥75% | ⚠️ Warning |

### Performance Thresholds

| Metric | Target | Measurement |
|--------|--------|-------------|
| Query latency (mocked) | <800ms | 95th percentile |
| Query latency (full stack) | <2s | 95th percentile |
| Throughput | >100 qps | Sustained load |
| Memory usage | <500MB | Per session |

### Quality Thresholds

| Metric | Target | Test Type |
|--------|--------|-----------|
| Grounding F1 | ≥0.75 | Behavioral |
| Semantic similarity | ≥0.85 | Regression |
| Citation accuracy | 100% | Behavioral |
| Hallucination rate | <5% | Behavioral |

---

## 🔧 Common Test Scenarios

### Scenario 1: Run Quick Smoke Tests

```bash
# Run fast tests only
pytest -m "not slow" --maxfail=5
```

### Scenario 2: Run Full Test Suite (CI)

```bash
# Comprehensive test run with reports
pytest --cov --cov-report=html --junitxml=reports/junit.xml
```

### Scenario 3: Debug Failing Test

```bash
# Run single test with full output
pytest tests/unit/test_chunker.py::TestChunker::test_chunker_preserves_metadata -vv -s
```

### Scenario 4: Run Performance Benchmarks

```bash
# Run performance tests with benchmarks
pytest tests/performance/ -m performance --benchmark-only
```

### Scenario 5: Check Behavioral Quality

```bash
# Run all behavioral and regression tests
pytest tests/behavior/ tests/regression/ -m "behavior or regression"
```

---

## 🤖 Mocks & Fixtures

### Mock LLM (`mock_llm.py`)

Deterministic LLM responses for stable testing:

```python
def test_with_mock_llm(mock_llm):
    mock_llm.set_response("Python", "Python is a programming language")
    response = mock_llm.generate("What is Python?")
    assert "programming language" in response
```

**Features:**
- Deterministic responses based on prompt content
- Configurable failure modes (timeout, error, rate_limit)
- Call history tracking
- Custom response stubs

### Mock Embeddings (`mock_embeddings.py`)

Deterministic embeddings using text hashing:

```python
def test_with_mock_embeddings(mock_embeddings):
    emb1 = mock_embeddings.embed_query("Python")
    emb2 = mock_embeddings.embed_query("Python")
    assert emb1 == emb2  # Deterministic
```

**Features:**
- Deterministic vectors from text hash
- Configurable dimension (default: 768)
- Embedding cache for performance
- Custom embedding stubs for similarity tests

### Mock Vector DB (`mock_vectordb.py`)

In-memory vector database:

```python
def test_with_mock_vectordb(mock_vectordb, mock_embeddings):
    # Index document
    embedding = mock_embeddings.embed_query("test doc")
    mock_vectordb.index_document("doc1", "test doc", embedding)
    
    # Search
    results = mock_vectordb.search(embedding, top_k=5)
    assert len(results) == 1
    assert results[0]["id"] == "doc1"
```

**Features:**
- In-memory storage (fast, no external dependencies)
- Full CRUD operations
- Metadata filtering
- Configurable failure modes

### Synthetic Corpus Generator

Generate 200 diverse test documents:

```python
from tests.fixtures.synthetic_corpus_generator import generate_synthetic_corpus

documents = generate_synthetic_corpus(output_dir, num_docs=200)
```

**Features:**
- Various lengths (short, medium, long, very long)
- Multiple domains (programming, legal, medical, business, science)
- Overlapping facts for reranking tests
- Near-duplicates for de-duplication tests
- Simulated PII for redaction tests

---

## 🚨 Troubleshooting

### Test Failures

#### ImportError: Module not found
```bash
# Ensure project root is in PYTHONPATH
export PYTHONPATH=/path/to/RAG_Agent:$PYTHONPATH
pytest
```

#### Fixture not found
```bash
# Verify conftest.py is in tests/ directory
ls tests/conftest.py

# Re-install test requirements
pip install -r tests/requirements.txt
```

#### Flaky behavioral tests
```bash
# Run with retries (CI does this automatically)
pytest tests/behavior/ --reruns 2 --reruns-delay 1
```

### Performance Issues

#### Tests taking too long
```bash
# Run in parallel
pytest -n auto

# Skip slow tests
pytest -m "not slow"

# Skip performance tests
pytest -m "not performance"
```

#### Coverage calculation slow
```bash
# Run coverage on subset
pytest tests/unit/ --cov=core --no-cov-on-fail
```

---

## 📈 CI/CD Integration

### GitHub Actions Workflow

See `.github/workflows/test-suite.yml` for full CI configuration.

**CI runs:**
1. Unit tests (parallel)
2. Integration tests
3. Black-box API tests
4. Behavioral tests (with retry)
5. Security tests
6. Coverage report (enforces ≥85%)
7. Performance benchmarks (informational)

**Artifacts uploaded:**
- `reports/junit.xml` - Test results
- `reports/coverage.xml` - Coverage data
- `reports/coverage_html/` - HTML coverage report
- `reports/test_report.html` - HTML test report

### Local CI Simulation

```bash
# Run same tests as CI
./tests/scripts/run_ci_locally.sh
```

---

## 📚 Additional Resources

### Test Utilities

- **test_reporter.py** - Aggregate test metrics into JSON summary
- **generate_test_data.py** - Generate additional test data
- **mutation_test_config.py** - Run mutation tests with mutmut
- **canary_tests.py** - Run canary tests for deployments
- **human_review_harness.py** - Generate CSV for manual review
- **dataset_drift_monitor.py** - Detect corpus distribution changes
- **auto_repair_suggestions.py** - Get suggestions for test failures

### Running Mutation Tests

```bash
# Install mutmut
pip install mutmut

# Run mutation tests
mutmut run --paths-to-mutate=core,app,api

# View results
mutmut show
```

### Canary Tests

```bash
# Run canary tests before deployment
python tests/scripts/canary_tests.py --env production
```

### Human Review Harness

```bash
# Generate review CSV for low-confidence answers
python tests/scripts/human_review_harness.py --output reviews.csv
```

---

## 🤝 Contributing

When adding new tests:

1. **Follow naming convention**: `test_<component>_<behavior>.py`
2. **Add docstrings**: Explain what the test validates
3. **Use appropriate markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, etc.
4. **Maintain coverage**: Ensure new code has ≥85% test coverage
5. **Add to CI**: Update `.github/workflows/test-suite.yml` if needed
6. **Update docs**: Add test description to this README

---

## 📞 Support

For issues or questions:

- **GitHub Issues**: [RAG_Agent/issues](https://github.com/Martin-Aziz/RAG_Agent/issues)
- **Documentation**: [docs/](../docs/)
- **CI Logs**: Check GitHub Actions for test failures

---

**Last Updated**: 2025-10-13  
**Test Suite Version**: 1.0.0  
**Total Tests**: 150+  
**Minimum Coverage**: 85%
