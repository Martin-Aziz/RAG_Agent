# Advanced RAG Quick Reference Card

## 🚀 Quick Start

### Basic Mode (Default)
```bash
python seeds/seed_data.py
uvicorn api.main:app --reload
```

### Advanced Mode (Production)
```bash
export USE_ADVANCED_RAG=1
export ENABLE_FAISS=1
pip install sentence-transformers torch transformers
uvicorn api.main:app --reload
```

## 🎛️ Environment Variables

| Variable | Values | Effect |
|----------|--------|--------|
| `USE_ADVANCED_RAG` | `0`/`1` | Enable all advanced features |
| `ENABLE_FAISS` | `0`/`1` | Use FAISS vector search |
| `USE_QUERY_EXPANSION` | `0`/`1` | Multi-query expansion (+latency) |
| `USE_OLLAMA` | `0`/`1` | Use Ollama LLM |
| `OLLAMA_MODEL` | `llama2` | LLM model name |
| `OLLAMA_EMBED_MODEL` | `qwen3-embedding:latest` | Embedding model |

## 📦 Feature Matrix

| Feature | Module | Requires | Latency Impact |
|---------|--------|----------|----------------|
| Semantic Chunking | `core.chunking` | embedder | - |
| Hybrid Search (RRF) | `core.retrieval.hybrid` | - | +50ms |
| Cross-Encoder Rerank | `core.retrieval.hybrid` | sentence-transformers | +200-500ms |
| Conversation Memory | `core.memory` | - | +10ms |
| Self-RAG Grading | `core.self_rag` | - | +300-800ms |
| Corrective RAG | `core.self_rag` | - | Variable |
| Multi-Query Expansion | `core.query_processing` | - | +200-500ms |
| HyDE | `core.query_processing` | embedder | +100-300ms |

## 🔧 Common Tasks

### Query with Memory
```python
# First turn
POST /query {"user_id": "u1", "session_id": "s1", "query": "What is AI?", ...}

# Follow-up (context preserved)
POST /query {"user_id": "u1", "session_id": "s1", "query": "How does it work?", ...}
```

### Manual Feature Usage

**Semantic Chunking:**
```python
from core.chunking import SemanticChunker
from core.embedders.ollama_embedder import OllamaEmbedder

embedder = OllamaEmbedder(model="qwen3-embedding:latest")
chunker = SemanticChunker(embedder, similarity_threshold=0.7)
chunks = chunker.chunk(text, doc_id="doc_123")
```

**Hybrid Search:**
```python
from core.retrieval.hybrid import HybridRetriever

hybrid = HybridRetriever(vector_retriever, bm25_retriever, k=60)
results = hybrid.retrieve(query, top_k=10, method="rrf")
```

**Cross-Encoder Reranking:**
```python
from core.retrieval.hybrid import CrossEncoderReranker

reranker = CrossEncoderReranker()
reranked = reranker.rerank(query, documents, top_k=10)
```

**Conversation Memory:**
```python
from core.memory import ConversationMemoryManager

memory = ConversationMemoryManager(max_turns=10, max_tokens=4000)
memory.add_turn(session_id, user_id, user_msg, assistant_msg)
context = memory.build_context(session_id, user_id)
```

**Self-RAG Verification:**
```python
from core.self_rag import SelfRAGVerifier

verifier = SelfRAGVerifier(model_adapter)
is_relevant, conf, reason = await verifier.grade_retrieval(query, doc)
is_supported, conf, reason = await verifier.check_hallucination(query, answer, evidence)
```

**Multi-Query Expansion:**
```python
from core.query_processing import MultiQueryExpander

expander = MultiQueryExpander(model_adapter, num_variants=3)
results = await expander.retrieve_with_expansion(query, retriever, k=10)
```

**HyDE:**
```python
from core.query_processing import HyDERetriever

hyde = HyDERetriever(model_adapter, embedder)
results = await hyde.retrieve_with_hyde(query, retriever, k=10, use_both=True)
```

## 🐛 Troubleshooting

**"Advanced features not available"**
```bash
pip install sentence-transformers torch transformers
```

**Cross-encoder slow**
```python
# Use smaller model
CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-2-v2")
```

**Memory context too long**
```python
# Reduce turns or tokens
ConversationMemoryManager(max_turns=5, max_tokens=2000)
```

**Zero retrieval results**
```bash
# Check seeding
python seeds/seed_data.py
ls data/docs.json  # Should exist
```

## 📊 Trace Analysis

**Look for these agents in response trace:**

Basic mode:
- `planner`, `router`, `retriever`, `verifier`, `synthesizer`

Advanced mode:
- `memory`, `hybrid_retriever`, `reranker`, `corrective_rag`, `self_rag`

**Example trace inspection:**
```python
response = await orch.handle_query(req)
agents = {step.agent for step in response.trace}
print(f"Active agents: {agents}")
```

## 🎯 Performance Tuning

**For low latency (<500ms):**
```bash
USE_ADVANCED_RAG=0  # Basic mode
```

**For balanced (<1s):**
```bash
USE_ADVANCED_RAG=1
USE_QUERY_EXPANSION=0  # Skip expansion
# Use CPU cross-encoder
```

**For maximum quality (1-2s):**
```bash
USE_ADVANCED_RAG=1
USE_QUERY_EXPANSION=1
# Use GPU cross-encoder
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `core/orchestrator.py` | Main integration point |
| `core/chunking/semantic_chunker.py` | Semantic chunking |
| `core/retrieval/hybrid.py` | Hybrid search + reranking |
| `core/memory/conversation_memory.py` | Multi-turn memory |
| `core/self_rag/reflection.py` | Self-RAG + CRAG |
| `core/query_processing/advanced.py` | Expansion + HyDE |
| `docs/ADVANCED_FEATURES.md` | Full documentation |
| `tests/integration/test_advanced_rag.py` | Integration tests |

## 🔗 Quick Links

- [Full Documentation](ADVANCED_FEATURES.md)
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md)
- [CLI Guide](CHAT.md)
- [Main README](../README_NEW.md)

## 💡 Tips

1. **Start with basic mode** - validate setup before enabling advanced features
2. **Test incrementally** - enable one feature at a time to isolate issues
3. **Monitor trace** - use trace to debug which agent is causing issues
4. **Check logs** - orchestrator prints initialization status
5. **Use fallbacks** - system designed to work without optional deps

## 🎓 Learning Path

1. ✅ Run basic mode query - understand baseline
2. ✅ Enable `USE_ADVANCED_RAG=1` - see new agents in trace
3. ✅ Test conversation memory - send multi-turn dialogue
4. ✅ Try query expansion - compare recall with/without
5. ✅ Inspect trace details - understand agent interactions
6. ✅ Tune thresholds - optimize for your use case

---

**For detailed examples and API reference, see [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md)**
