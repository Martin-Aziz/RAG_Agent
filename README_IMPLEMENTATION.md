# Production RAG System Upgrade - Implementation Guide

## 🎯 Overview

This guide documents the complete upgrade of the RAG chatbot into a production-grade, stateful, multi-path agent system with:

- **LangGraph Orchestration**: Stateful workflows with checkpointing
- **Intent Routing**: Pre-retrieval classification for efficiency and safety  
- **Hybrid Retrieval**: BM25 + Vector with RRF fusion
- **Cross-Encoder Reranking**: MS MARCO models for precision
- **Self-RAG Verification**: Retrieve-generate-critique loops
- **GraphRAG Integration**: Neo4j-backed multi-hop reasoning
- **Hierarchical Memory**: Short/session/long-term with semantic switching
- **Multi-Agent Mode**: Optional AutoGen/CrewAI specialization
- **Full Observability**: Tracing, metrics, artifacts

##📁 Project Structure

```
RAG_Agent/
├── app/                          # New modular implementation
│   ├── config.py                 # Configuration management
│   ├── router/                   # Intent classification & routing
│   │   └── __init__.py
│   ├── retrieval/                # Hybrid BM25 + vector retrieval
│   │   └── __init__.py
│   ├── rerank/                   # Cross-encoder reranking
│   │   └── __init__.py
│   ├── graphrag/                 # Neo4j GraphRAG integration
│   │   ├── __init__.py
│   │   ├── entity_extraction.py
│   │   ├── graph_store.py
│   │   └── traversal.py
│   ├── generation/               # Prompt templates & generation
│   │   └── __init__.py
│   ├── verifier/                 # Self-RAG verification
│   │   └── __init__.py
│   ├── memory/                   # Hierarchical memory
│   │   └── __init__.py
│   ├── orchestration/            # LangGraph workflows
│   │   ├── __init__.py
│   │   ├── graph.py
│   │   └── nodes.py
│   ├── agents/                   # Multi-agent (AutoGen/CrewAI)
│   │   ├── __init__.py
│   │   ├── autogen_agents.py
│   │   └── crewai_agents.py
│   └── observability/            # Tracing & metrics
│       └── __init__.py
├── core/                         # Legacy implementation (kept for compatibility)
├── configs/
│   ├── default.yaml              # Main configuration
│   ├── local.yaml                # Local dev overrides
│   └── production.yaml           # Production settings
├── scripts/
│   ├── install.sh                # Dependency installation
│   ├── setup_neo4j.sh            # Neo4j bootstrap
│   └── run_evals.py              # Evaluation runner
├── tests/
│   ├── unit/                     # Unit tests
│   ├── scenario/                 # Acceptance tests
│   └── integration/              # Integration tests
├── docker-compose.yml            # Local services (Neo4j, etc.)
├── requirements.txt              # Core dependencies
├── requirements-langgraph.txt    # LangGraph & agent frameworks
└── README_IMPLEMENTATION.md      # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# LangGraph + Agent frameworks
pip install -r requirements-langgraph.txt

# Verify installations
python -c "import langgraph, langchain, autogen, crewai; print('✅ All frameworks installed')"
```

### 2. Start Neo4j (for GraphRAG)

```bash
# Using Docker Compose
docker-compose up -d neo4j

# Verify
curl http://localhost:7474
```

### 3. Configure System

```bash
# Copy and edit config
cp configs/default.yaml configs/local.yaml

# Set environment variables
export RAG_CONFIG_PATH=configs/local.yaml
export OLLAMA_MODEL=llama2
export OLLAMA_EMBED_MODEL=qwen3-embedding:latest
```

### 4. Seed Knowledge Base

```bash
python seeds/seed_data.py
```

### 5. Run Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Acceptance tests
pytest tests/scenario/ -v

# Full test suite
pytest tests/ -v --cov=app
```

### 6. Start API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Access Web UI

Navigate to `http://localhost:8000` to interact with the upgraded system.

---

## 🧩 Module Implementation Status

| Module | Status | Description |
|--------|--------|-------------|
| ✅ `app/config.py` | Complete | YAML/ENV configuration with Pydantic models |
| ✅ `app/router/` | Complete | Intent classification, FAQ matching, safety checks |
| ✅ `app/retrieval/` | Complete | Hybrid BM25+vector with RRF fusion |
| ⏳ `app/rerank/` | In Progress | Cross-encoder reranking (see below) |
| ⏳ `app/graphrag/` | In Progress | Neo4j integration (see below) |
| ⏳ `app/verifier/` | In Progress | Self-RAG verification loops (see below) |
| ⏳ `app/memory/` | In Progress | Hierarchical memory system (see below) |
| ⏳ `app/orchestration/` | In Progress | LangGraph workflow (see below) |
| ⏳ `app/agents/` | In Progress | Multi-agent mode (see below) |
| ⏳ `app/observability/` | In Progress | Tracing & metrics (see below) |

---

## 📝 Implementation Details

### Intent Router (`app/router/`)

**Status:** ✅ Complete

**Features:**
- Pattern-based smalltalk detection (hello, goodbye, thanks)
- Jailbreak pattern detection for safety
- FAQ similarity matching with confidence thresholds
- Configurable routing thresholds

**Usage:**
```python
from app.router import IntentRouter, Intent

router = IntentRouter()
intent, confidence, metadata = router.route("Hello, how are you?")

if intent == Intent.SMALLTALK:
    response = router.get_smalltalk_response(query)
elif intent == Intent.FAQ:
    response = metadata["answer"]
elif intent == Intent.RAG:
    # Full RAG pipeline
    pass
```

**Configuration:**
```yaml
router:
  intent_classifier:
    enabled: true
    confidence_threshold: 0.75
  faq_matcher:
    enabled: true
    similarity_threshold: 0.85
  safety:
    jailbreak_detection: true
    max_query_length: 1000
```

---

### Hybrid Retrieval (`app/retrieval/`)

**Status:** ✅ Complete

**Features:**
- Parallel BM25 + vector retrieval
- Reciprocal Rank Fusion (RRF) with configurable k
- Linear fusion fallback
- Adaptive fusion based on query characteristics
- Timeout handling and circuit breaking

**Usage:**
```python
from app.retrieval import HybridRetriever

hybrid = HybridRetriever(
    bm25_retriever=bm25,
    vector_retriever=vector,
    rrf_k=60,
    candidate_pool_size=50
)

results = await hybrid.retrieve(query, top_k=10, use_rrf=True)
```

**Configuration:**
```yaml
retrieval:
  hybrid:
    bm25_weight: 0.5
    vector_weight: 0.5
    rrf_k: 60
    candidate_pool_size: 50
```

---

### Next Steps for Full Implementation

The following modules need to be completed. I'll create them now:

1. **Cross-Encoder Reranking** (`app/rerank/`)
2. **GraphRAG with Neo4j** (`app/graphrag/`)
3. **Self-RAG Verifier** (`app/verifier/`)
4. **Hierarchical Memory** (`app/memory/`)
5. **LangGraph Orchestration** (`app/orchestration/`)
6. **Multi-Agent Support** (`app/agents/`)
7. **Observability** (`app/observability/`)
8. **Comprehensive Tests** (`tests/`)

---

## 🧪 Acceptance Test Checklist

- [ ] **Smalltalk routing**: Greetings return < 200ms without RAG
- [ ] **FAQ routing**: FAQ matches return < 300ms with citations
- [ ] **Hybrid retrieval**: BM25 + vector fusion works correctly
- [ ] **RRF fusion**: Results properly ranked by reciprocal rank
- [ ] **Reranking**: Cross-encoder improves top-k precision
- [ ] **Self-RAG**: Low-confidence triggers re-retrieval
- [ ] **GraphRAG**: Multi-hop queries traverse Neo4j graph
- [ ] **Memory**: Long sessions show semantic switching
- [ ] **Observability**: Traces captured for all runs
- [ ] **Multi-agent**: Optional mode works with feature flag

---

## 🔧 Configuration Guide

### Feature Flags

```yaml
features:
  multi_agent_mode: false       # Enable AutoGen/CrewAI
  graphrag_enabled: true        # Enable Neo4j GraphRAG
  self_rag_verification: true   # Enable verification loops
  intent_routing: true          # Enable pre-retrieval routing
  hybrid_retrieval: true        # Enable BM25+vector hybrid
  cross_encoder_reranking: true # Enable reranking
```

### Performance Tuning

**RRF Parameter (`rrf_k`)**:
- Default: 60
- Lower (20-40): More weight on top ranks
- Higher (80-100): More weight on lower ranks
- **Tune first** for optimal recall vs latency

**Candidate Pool Size**:
- Default: 50
- Increase for better recall (100+)
- Decrease for lower latency (20-30)

**Reranker Top-K**:
- Default: 10
- Precision-focused: 5-10
- Recall-focused: 15-20

### Environment Variables

```bash
# Override any config value
export RAG_FEATURES_GRAPHRAG_ENABLED=true
export RAG_RETRIEVAL_HYBRID_RRF_K=80
export RAG_MODELS_LLM_MODEL_NAME=llama3
```

---

## 📊 Observability

### Metrics Tracked

- Accuracy per query type
- Hallucination rate
- Latency (p50, p95, p99)
- Cost per query
- Fallback/refusal rate
- Route distribution

### Artifacts Saved

- Retrieved passages with scores
- Reranker scores
- Verification grades
- Prompts and completions
- Decision logs

### Prometheus Endpoint

```bash
curl http://localhost:8000/metrics
```

---

## 🐛 Troubleshooting

### Issue: Intent router not detecting smalltalk

**Solution**: Lower `confidence_threshold` in `router.intent_classifier`

### Issue: RRF fusion returning poor results

**Solution**: Tune `rrf_k` parameter (try 40-80 range) and verify both retrievers return results

### Issue: Cross-encoder reranking slow

**Solution**: Reduce `candidate_pool_size` or use smaller reranker model

### Issue: Neo4j connection failures

**Solution**: Check `docker-compose logs neo4j` and verify credentials in config

---

## 📚 References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Self-RAG Paper](https://arxiv.org/abs/2310.11511)
- [RRF Fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)
- [MS MARCO Cross-Encoders](https://www.sbert.net/docs/pretrained-models/ce-msmarco.html)
- [Neo4j GraphRAG](https://neo4j.com/developer/genai-ecosystem/)

---

## 🤝 Contributing

See implementation TODOs in each module. Priority areas:

1. Complete reranker module
2. Implement GraphRAG entity extraction
3. Build Self-RAG verification loops
4. Create LangGraph orchestration graph
5. Add comprehensive test coverage

---

## 📄 License

Same as parent project
