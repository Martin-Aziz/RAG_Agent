# RAG Agent - Complete Test Suite

## 🎯 Overview

This directory contains a **comprehensive, production-ready test suite** for the RAG (Retrieval-Augmented Generation) chatbot system. The test suite includes 150+ tests across 8 categories, with ≥85% code coverage requirement.

## 📋 Test Categories

| Category | Tests | Purpose | Markers |
|----------|-------|---------|---------|
| **Unit** | 70+ | Individual component testing | `@pytest.mark.unit` |
| **Integration** | 20+ | End-to-end pipeline testing | `@pytest.mark.integration` |
| **Black-box** | 15+ | API contract & behavior | `@pytest.mark.blackbox` |
| **Behavioral** | 25+ | Grounding, citations, hallucinations | `@pytest.mark.behavior` |
| **Adversarial** | 15+ | Security, prompt injection | `@pytest.mark.adversarial` |
| **Performance** | 10+ | Latency, throughput, memory | `@pytest.mark.performance` |
| **Monitoring** | 5+ | Health checks, telemetry | `@pytest.mark.monitoring` |
| **Regression** | 10+ | Golden queries, consistency | `@pytest.mark.regression` |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install test dependencies
pip install -r tests/requirements.txt
```

### 2. Generate Test Data

```bash
# Generate golden queries, edge cases, adversarial data
python tests/scripts/generate_test_data.py

# Generate synthetic corpus (200 documents)
python tests/fixtures/synthetic_corpus_generator.py

# Optional: Generate large performance corpus (10K docs)
python tests/scripts/generate_test_data.py --large
```

### 3. Run Tests

#### Option A: Run All Tests (Recommended)

```bash
# Run complete test suite with reporting
./tests/run_all_tests.sh

# Run and auto-open HTML reports
./tests/run_all_tests.sh --open
```

#### Option B: Run Specific Categories

```bash
# Unit tests only (fast, parallel)
pytest tests/unit/ -v -m unit -n auto

# Integration tests
pytest tests/integration/ -v -m integration

# Security tests
pytest tests/adversarial/ -v -m adversarial

# Performance benchmarks
pytest tests/performance/ -v -m performance --benchmark-only
```

#### Option C: Run with Coverage

```bash
# Full coverage report (≥85% required)
pytest -v \
  --cov=core --cov=app --cov=api \
  --cov-report=html:reports/coverage-html \
  --cov-report=term \
  --cov-fail-under=85
```

### 4. View Reports

Reports are generated in the `reports/` directory:

- **Summary**: `reports/summary.html` - Interactive test dashboard
- **Coverage**: `reports/coverage-html/index.html` - Line-by-line coverage
- **JSON**: `reports/summary.json` - Machine-readable results

## 📁 Directory Structure

```
tests/
├── pytest.ini                          # Pytest configuration
├── requirements.txt                    # Test dependencies
├── conftest.py                         # Root fixtures & config
├── run_all_tests.sh                    # Master test runner script
├── README.md                           # This file
├── README_TESTING.md                   # Detailed testing guide
│
├── fixtures/                           # Mocks & test data generators
│   ├── mock_llm.py                     # Deterministic LLM mock
│   ├── mock_embeddings.py              # Deterministic embeddings
│   ├── mock_vectordb.py                # In-memory vector DB
│   └── synthetic_corpus_generator.py   # 200-doc test corpus
│
├── data/                               # Generated test data
│   ├── golden_queries.json             # Ground truth Q&A pairs
│   ├── edge_case_documents.json        # Edge cases (empty, long, unicode)
│   ├── adversarial_cases.json          # Security test cases
│   ├── multilingual_queries.json       # Non-English queries
│   └── performance_corpus.json         # Large-scale corpus (optional)
│
├── scripts/                            # Utility scripts
│   ├── test_reporter.py                # Generate HTML/JSON reports
│   └── generate_test_data.py           # Create test datasets
│
├── unit/                               # Unit tests (70+ tests)
│   ├── test_chunker.py                 # Text chunking with metadata
│   ├── test_embedding_wrapper.py       # Embedding API wrapper
│   ├── test_vectordb_client.py         # Vector DB operations
│   ├── test_retriever.py               # Document retrieval
│   ├── test_reranker.py                # Result reranking
│   ├── test_document_parser.py         # Document parsing
│   ├── test_citation_generator.py      # Citation formatting
│   └── test_response_formatter.py      # Response formatting
│
├── integration/                        # Integration tests (20+ tests)
│   ├── test_e2e_rag.py                 # Full RAG pipeline
│   ├── test_multiturn_context.py       # Conversation context
│   ├── test_long_document_handling.py  # Large document processing
│   └── test_metadata_filtering.py      # Metadata-based filtering
│
├── blackbox/                           # Black-box API tests (15+ tests)
│   ├── test_api_contract.py            # API contract validation
│   ├── test_auth_and_rate_limits.py    # Authentication & limits
│   ├── test_concurrency.py             # Concurrent requests
│   └── test_fault_injection.py         # Error handling
│
├── behavior/                           # Behavioral tests (25+ tests)
│   ├── test_grounding_and_citations.py # Citation accuracy
│   ├── test_hallucination_detection.py # Hallucination prevention
│   ├── test_conservative_answering.py  # "I don't know" responses
│   ├── test_consistency.py             # Answer consistency
│   └── test_tone_and_policy.py         # Tone & policy compliance
│
├── adversarial/                        # Security tests (15+ tests)
│   ├── test_prompt_injection.py        # Prompt injection attacks
│   ├── test_jailbreak_attempts.py      # Jailbreak prevention
│   ├── test_data_exfiltration.py       # Data leakage prevention
│   └── test_pii_handling.py            # PII detection & redaction
│
├── performance/                        # Performance tests (10+ tests)
│   ├── test_throughput.py              # Request throughput
│   ├── test_cache_performance.py       # Cache hit rates
│   ├── test_memory_leak.py             # Memory leak detection
│   └── load_test_locust.py             # Load testing (Locust)
│
├── monitoring/                         # Monitoring tests (5+ tests)
│   ├── test_telemetry.py               # Metrics & logging
│   └── test_health_check.py            # Health endpoint
│
└── regression/                         # Regression tests (10+ tests)
    ├── test_golden_queries.py          # Golden Q&A validation
    └── test_consistency_over_time.py   # Temporal consistency
```

## ⚙️ Configuration

### Environment Variables

Configure the test suite via environment variables:

```bash
# Core settings
export RAG_API_BASE_URL="http://localhost:8000"
export EMBEDDING_MODEL="text-embedding-3-small"
export LLM_MODEL="gpt-4"
export VECTOR_DB="faiss"  # or "pinecone", "weaviate"

# RAG parameters
export TOP_K=5                # Top K retrieved documents
export CHUNK_SIZE=1000        # Characters per chunk
export CHUNK_OVERLAP=200      # Overlap between chunks

# Test thresholds
export MIN_COVERAGE=85        # Minimum code coverage %
export MAX_LATENCY_MS=2000    # Maximum query latency
export MIN_CITATION_RATE=0.8  # Minimum citation rate
export MAX_HALLUCINATION_RATE=0.05  # Maximum hallucination rate
```

### Pytest Configuration

Edit `tests/pytest.ini` to customize:

```ini
[pytest]
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow-running tests (>5s)
    flaky: Potentially flaky tests
    
addopts =
    -v                          # Verbose output
    --strict-markers            # Fail on unknown markers
    --cov-fail-under=85         # Coverage threshold
    -n auto                     # Parallel execution
```

## 🔧 Key Features

### 1. Deterministic Mocks

All external dependencies are mocked with deterministic behavior:

- **MockLLM**: Returns predictable responses with citations
- **MockEmbeddings**: Generates stable hash-based vectors
- **MockVectorDB**: In-memory similarity search

**Benefits**: Fast, reproducible tests without API calls.

### 2. Synthetic Test Data

- **200 diverse documents** (programming, legal, medical, business, science)
- **Golden Q&A pairs** with ground truth answers
- **Edge cases**: Empty docs, 100K+ char docs, unicode, emojis
- **Adversarial cases**: Prompt injections, jailbreaks, obfuscation

### 3. Comprehensive Coverage

- **Unit tests**: Every core function individually tested
- **Integration tests**: Full pipeline from query → answer
- **Black-box tests**: API contract compliance
- **Behavioral tests**: Hallucination detection, grounding validation
- **Security tests**: Prompt injection prevention
- **Performance tests**: Latency, throughput, memory benchmarks

### 4. Rich Reporting

- **HTML Dashboard**: Interactive test results with charts
- **Coverage Report**: Line-by-line coverage visualization
- **JSON Summary**: Machine-readable for CI/CD
- **GitHub Actions**: Automated PR comments with results

## 📊 Metrics & Thresholds

### Coverage Requirements

| Component | Minimum Coverage |
|-----------|------------------|
| `core/` | 90% |
| `app/` | 85% |
| `api/` | 80% |
| **Overall** | **≥85%** |

### Performance Thresholds

| Metric | Threshold |
|--------|-----------|
| Query Latency (p50) | < 500ms |
| Query Latency (p95) | < 2000ms |
| Throughput | > 10 qps |
| Citation Rate | > 80% |
| Hallucination Rate | < 5% |

### Quality Gates

Tests must pass these gates to merge:

- ✅ All unit tests pass
- ✅ All integration tests pass
- ✅ Coverage ≥ 85%
- ✅ No security test failures
- ✅ Performance within thresholds
- ✅ No critical lint errors

## 🤖 CI/CD Integration

### GitHub Actions

The test suite runs automatically on:

- **Push to `main`/`develop`** - Full test suite
- **Pull requests** - Full test suite + PR comment
- **Daily at 2 AM UTC** - Full suite + mutation tests

See `.github/workflows/test-suite.yml` for configuration.

### Running Locally (Pre-commit)

```bash
# Fast pre-commit check (unit tests only)
pytest tests/unit/ -v -m unit -n auto --maxfail=1

# Medium check (unit + integration)
pytest tests/unit/ tests/integration/ -v -n auto

# Full check (all categories)
./tests/run_all_tests.sh
```

## 🐛 Troubleshooting

### Common Issues

#### 1. **Import Errors**

```bash
# Ensure tests/ is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in editable mode
pip install -e .
```

#### 2. **Mock Fixtures Not Found**

```bash
# Regenerate synthetic corpus
python tests/fixtures/synthetic_corpus_generator.py

# Regenerate test data
python tests/scripts/generate_test_data.py
```

#### 3. **Coverage Below Threshold**

```bash
# View missing coverage
pytest --cov=core --cov-report=term-missing

# Generate HTML report for detailed view
pytest --cov=core --cov-report=html
open reports/coverage-html/index.html
```

#### 4. **Tests Timing Out**

```bash
# Run tests with higher timeout
pytest --timeout=300

# Or skip slow tests
pytest -m "not slow"
```

#### 5. **Parallel Execution Issues**

```bash
# Disable parallel execution
pytest -n 0

# Or use fewer workers
pytest -n 2
```

## 📚 Additional Resources

- **Detailed Guide**: See `README_TESTING.md` for in-depth documentation
- **Pytest Docs**: https://docs.pytest.org/
- **Coverage.py**: https://coverage.readthedocs.io/
- **Hypothesis**: https://hypothesis.readthedocs.io/ (property-based testing)

## 🔮 Future Enhancements

- [ ] **Mutation Testing**: Automated test quality assessment
- [ ] **Visual Regression**: Screenshot comparison for UI
- [ ] **A/B Testing**: Compare model versions
- [ ] **Chaos Engineering**: Fault injection at scale
- [ ] **Differential Testing**: Compare against baseline
- [ ] **Canary Testing**: Progressive rollout validation

## 📞 Support

For issues or questions:

1. Check `README_TESTING.md` for detailed documentation
2. Review test logs in `reports/`
3. Run tests with `-vv` for verbose output
4. Check GitHub Actions logs for CI failures

## 🎉 Summary

This test suite provides:

- ✅ **150+ tests** across 8 categories
- ✅ **≥85% code coverage** with enforcement
- ✅ **Deterministic mocks** for fast, reliable tests
- ✅ **Comprehensive test data** (golden queries, edge cases, adversarial)
- ✅ **Rich reporting** (HTML, JSON, coverage)
- ✅ **CI/CD integration** (GitHub Actions)
- ✅ **Production-ready** with real implementation (not pseudo-code)

Run `./tests/run_all_tests.sh --open` to get started! 🚀
