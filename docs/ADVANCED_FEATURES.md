# Advanced RAG Features Guide

## Overview

This RAG agent now includes state-of-the-art 2025 techniques for production-grade retrieval-augmented generation:

1. **Semantic Chunking** - Embedding-based boundary detection
2. **Hybrid Search with RRF** - Combines BM25 + vector search
3. **Cross-Encoder Reranking** - Neural reranking for 92%+ accuracy
4. **Conversation Memory** - Multi-turn dialogue with context preservation
5. **Self-RAG** - Reflection loops with hallucination detection
6. **Corrective RAG (CRAG)** - Dynamic correction strategies
7. **Multi-Query Expansion** - Generate query variants for better recall
8. **HyDE** - Hypothetical Document Embeddings

## Quick Start

### Enable Advanced Features

```bash
# Set environment variable to activate advanced RAG
export USE_ADVANCED_RAG=1

# Enable FAISS for vector search (required for some features)
export ENABLE_FAISS=1
export OLLAMA_EMBED_MODEL=qwen3-embedding:latest

# Optional: Enable query expansion
export USE_QUERY_EXPANSION=1

# Install advanced dependencies
pip install sentence-transformers torch transformers
```

### Start the API

```bash
uvicorn api.main:app --reload
```

When advanced features are enabled, you'll see:
```
Advanced RAG features enabled: hybrid search, reranking, self-RAG, memory
```

## Feature Details

### 1. Semantic Chunking

**What it does:** Splits documents into coherent chunks based on semantic similarity between consecutive sentences, not fixed token counts.

**Usage:**
```python
from core.chunking import SemanticChunker
from core.embedders.ollama_embedder import OllamaEmbedder

embedder = OllamaEmbedder(model="qwen3-embedding:latest")
chunker = SemanticChunker(
    embedder=embedder,
    similarity_threshold=0.7,  # Split when similarity drops below 0.7
    min_chunk_size=100,
    max_chunk_size=500,
    overlap_tokens=50
)

chunks = chunker.chunk(document_text, doc_id="doc_123")
# Returns: [{"text": "...", "doc_id": "...", "chunk_id": 0, ...}, ...]
```

**Benefits:**
- Preserves semantic coherence across chunk boundaries
- Better retrieval relevance (chunks contain complete thoughts)
- Configurable overlap prevents information loss

### 2. Hybrid Search with Reciprocal Rank Fusion

**What it does:** Runs BM25 (lexical) and vector (semantic) search in parallel, then fuses results using RRF algorithm.

**Automatic:** When `USE_ADVANCED_RAG=1`, all PAR-RAG queries use hybrid retrieval.

**Manual usage:**
```python
from core.retrieval.hybrid import HybridRetriever

hybrid = HybridRetriever(vector_retriever, bm25_retriever, k=60)

# RRF fusion (recommended)
results = hybrid.retrieve(query, top_k=10, method="rrf")

# Or linear fusion with tunable weight
results = hybrid.retrieve(query, top_k=10, method="linear")  # alpha=0.5 default
```

**RRF Formula:** `score = sum(1 / (k + rank_i))` where k=60

**Benefits:**
- Captures both keyword matching (BM25) and semantic similarity (vector)
- Robust across diverse query types
- Better than either method alone

### 3. Cross-Encoder Reranking

**What it does:** Applies a BERT-based cross-encoder to jointly score query+document pairs for superior relevance.

**Automatic:** Reranking is applied after hybrid retrieval when advanced features enabled.

**Model:** Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` by default (92%+ accuracy on MS MARCO).

**Manual usage:**
```python
from core.retrieval.hybrid import CrossEncoderReranker

reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu"  # or "cuda"
)

# Rerank retrieved documents
reranked = reranker.rerank(query, documents, top_k=10)
```

**Benefits:**
- Dramatically improves relevance (bi-encoders retrieve, cross-encoders rerank)
- Reduces false positives from initial retrieval
- Production-grade accuracy

### 4. Conversation Memory

**What it does:** Manages multi-turn conversations with automatic summarization and user profile tracking.

**Features:**
- **Short-term memory:** Last N turns per session
- **Long-term memory:** User preferences and learned facts across sessions
- **Automatic summarization:** Older turns compressed to fit token budget
- **Persistence:** Sessions and profiles saved to `data/memory/`

**Automatic:** Context from previous turns automatically prepended to new queries when `USE_ADVANCED_RAG=1`.

**Manual usage:**
```python
from core.memory import ConversationMemoryManager

memory = ConversationMemoryManager(
    max_turns=10,
    max_tokens=4000,
    summarize_threshold=7,
    storage_path="data/memory"
)

# Add a turn
memory.add_turn(
    session_id="abc123",
    user_id="user_1",
    user_message="What is RAG?",
    assistant_message="RAG is..."
)

# Build context for next query
context = memory.build_context("abc123", "user_1", include_user_profile=True)

# Update user preferences
memory.update_user_preference("user_1", "response_style", "technical")
memory.add_learned_fact("user_1", "Works on AI research")
```

**Example context output:**
```
User Profile:
Preferences: {"response_style": "technical"}
Known facts: Works on AI research

Conversation Summary:
User asked about: What is RAG? | User asked about: How to implement?

Recent Conversation:
User: What is RAG?
Assistant: RAG is Retrieval-Augmented Generation...
User: How does it work?
Assistant: RAG combines retrieval...

Current query: Tell me more about vector databases
```

### 5. Self-RAG with Reflection

**What it does:** Uses LLM to grade retrieved documents and check generated answers for hallucinations.

**Automatic:** When `USE_ADVANCED_RAG=1`, every retrieval is graded and answers are checked.

**Grading prompts:**
- **Retrieval grading:** "Does this document contain relevant information?"
- **Hallucination check:** "Is this answer fully supported by the evidence?"

**Manual usage:**
```python
from core.self_rag import SelfRAGVerifier

verifier = SelfRAGVerifier(model_adapter, confidence_threshold=0.7)

# Grade a retrieved document
is_relevant, confidence, reasoning = await verifier.grade_retrieval(query, document)

# Check for hallucinations
is_supported, confidence, reasoning = await verifier.check_hallucination(
    query,
    answer,
    evidence_list
)
```

**Benefits:**
- Filters out irrelevant retrieved documents
- Detects and flags unsupported claims
- Provides confidence scores for downstream decisions

### 6. Corrective RAG (CRAG)

**What it does:** Evaluates retrieval quality and applies dynamic correction strategies:
- **High confidence (>0.8):** Use as-is
- **Medium confidence (0.5-0.8):** Refine knowledge (extract key sentences)
- **Low confidence (<0.5):** Web search fallback (if available)

**Automatic:** Applied in advanced PAR-RAG mode.

**Manual usage:**
```python
from core.self_rag import CorrectiveRAGEngine, SelfRAGVerifier

verifier = SelfRAGVerifier(model_adapter)
crag = CorrectiveRAGEngine(
    verifier=verifier,
    web_search_tool=my_web_search,
    high_threshold=0.8,
    low_threshold=0.5
)

# Correct retrieval based on quality
corrected_docs, strategy = await crag.correct_retrieval(query, retrieved_docs)
# strategy: "high_confidence", "knowledge_refinement", "web_search_fallback"
```

**Benefits:**
- Adaptive correction based on retrieval quality
- Prevents low-quality results from reaching LLM
- Automatic fallback to external sources

### 7. Multi-Query Expansion

**What it does:** Generates 3-5 query reformulations with different perspectives and specificity, then fuses results with RRF.

**Enable:**
```bash
export USE_QUERY_EXPANSION=1
```

**Manual usage:**
```python
from core.query_processing import MultiQueryExpander

expander = MultiQueryExpander(model_adapter, num_variants=3)

# Generate variants
variants = await expander.expand_query("What is machine learning?")
# Returns: [
#   "What is machine learning?",
#   "Explain the concept of machine learning in simple terms",
#   "How does machine learning work?"
# ]

# Retrieve with expansion and fusion
results = await expander.retrieve_with_expansion(
    query,
    retriever,
    k=10,
    fusion_k=60
)
```

**Benefits:**
- Improves recall for ambiguous queries
- Captures different query phrasings
- Robust across diverse user intents

### 8. HyDE (Hypothetical Document Embeddings)

**What it does:** Generates a hypothetical answer to the query, then retrieves using that answer's embedding instead of the query.

**Use case:** When queries are short or vague, HyDE improves semantic matching.

**Manual usage:**
```python
from core.query_processing import HyDERetriever

hyde = HyDERetriever(model_adapter, embedder)

# Retrieve with HyDE
results = await hyde.retrieve_with_hyde(
    query="Capital of France?",
    retriever=vector_retriever,
    k=10,
    use_both=True  # Merge query + hypothetical doc results
)
```

**Hypothetical document example:**
```
Query: "Capital of France?"
HyDE generates: "The capital of France is Paris, a major European city 
known for the Eiffel Tower and serving as the political and cultural center."
```

**Benefits:**
- Better matches when query and documents have different vocabularies
- Improves retrieval for questions vs. declarative text

## API Usage

### Query with Advanced Features

**Request:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "session_id": "session_abc",
    "query": "What are the benefits of RAG?",
    "mode": "parrag",
    "context_ids": [],
    "prefer_low_cost": false
  }'
```

**Response includes trace:**
```json
{
  "answer": "Based on the evidence...",
  "evidence": [...],
  "trace": [
    {"agent": "memory", "action": "load_context", ...},
    {"agent": "planner", "action": "plan", ...},
    {"agent": "hybrid_retriever", "action": "retrieve", "result": {"method": "rrf"}},
    {"agent": "reranker", "action": "rerank", ...},
    {"agent": "corrective_rag", "action": "correct", "result": {"strategy": "high_confidence"}},
    {"agent": "self_rag", "action": "check_hallucination", "result": {"is_supported": true}},
    ...
  ],
  "confidence": 0.85
}
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_ADVANCED_RAG` | `0` | Enable all advanced features |
| `ENABLE_FAISS` | `0` | Use FAISS for vector search |
| `OLLAMA_EMBED_MODEL` | `qwen3-embedding:latest` | Embedding model |
| `USE_OLLAMA` | `0` | Use Ollama for LLM generation |
| `OLLAMA_MODEL` | `llama2` | LLM model name |
| `USE_QUERY_EXPANSION` | `0` | Enable multi-query expansion |
| `VERIFIER_THRESHOLD` | Dynamic | Static threshold for verifier |

## Performance Tips

1. **Use FAISS + GPU embeddings** for large-scale retrieval (millions of docs)
2. **Tune RRF k parameter** (default 60): higher k = more weight on lower ranks
3. **Adjust reranker top_k** based on latency budget (10-20 recommended)
4. **Set conversation max_turns** based on context window (10 for 4K, 20 for 8K+)
5. **Enable query expansion** for ambiguous/exploratory queries only (adds latency)
6. **Use CPU cross-encoder** for <1K docs, GPU for >10K docs

## Troubleshooting

**"Advanced features not available"**
- Install optional dependencies: `pip install sentence-transformers torch transformers`

**Cross-encoder reranking slow**
- Use smaller model: `cross-encoder/ms-marco-MiniLM-L-2-v2`
- Or reduce candidate pool before reranking

**Memory context too long**
- Reduce `max_turns` or `max_tokens` in ConversationMemoryManager
- Check summary threshold (default 7)

**Self-RAG grading inconsistent**
- Use stronger LLM (GPT-4, Claude)
- Or fine-tune grading prompts in `core/self_rag/reflection.py`

## Architecture Diagram

```
User Query
    ↓
Conversation Memory (load context)
    ↓
Query Expansion (optional) → [Query variants]
    ↓
Hybrid Retrieval (RRF)
    ├─ BM25 Retriever
    └─ Vector Retriever (FAISS)
    ↓
Cross-Encoder Reranking
    ↓
Self-RAG Grading (filter irrelevant)
    ↓
Corrective RAG (quality check)
    ├─ High confidence → Use
    ├─ Medium → Refine
    └─ Low → Web search fallback
    ↓
LLM Synthesis (with conversation context)
    ↓
Hallucination Check
    ↓
Save to Conversation Memory
    ↓
Response to User
```

## Migration from Basic to Advanced

**Step 1:** Set `USE_ADVANCED_RAG=1` (no code changes required)

**Step 2:** Test a query - verify trace shows new agents:
```python
# Look for these in trace:
- hybrid_retriever
- reranker
- corrective_rag
- self_rag
- memory
```

**Step 3:** Gradually enable optional features:
```bash
export USE_QUERY_EXPANSION=1  # If needed
```

**Step 4:** Tune thresholds based on your evaluation dataset

## Next Steps

- **Fine-tune** cross-encoder on your domain data
- **Integrate web search** tool for CRAG fallback
- **Add caching** for embeddings and LLM calls
- **Scale with** Weaviate/Pinecone for production vector DB
- **Implement** feedback collection and active learning

---

For implementation details, see source code in:
- `core/chunking/` - Semantic chunking
- `core/retrieval/` - Hybrid search and reranking
- `core/memory/` - Conversation memory
- `core/self_rag/` - Reflection and CRAG
- `core/query_processing/` - Query expansion and HyDE
- `core/orchestrator.py` - Integration layer
