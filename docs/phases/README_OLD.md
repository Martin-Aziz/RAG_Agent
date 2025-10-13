Agentic RAG - MVP Reference Implementation

Overview
--------
This repository contains a minimal Agentic RAG reference implementation in Python using FastAPI. It includes:

- Core orchestrator with PAR-RAG and HopRAG modes
- Router, Vector and BM25 retrievers
- HopRAG passage graph and traversal
- Tool registry with JSON-schema validated tools (calculator, web_search_stub)
- Memory agent (semantic + episodic)
- Simple deterministic SLM stub for tests
- Evaluation harness and example dataset

Running locally
-------------
1. Install dependencies (recommend using a venv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Seed example data:

```bash
# Default (TF-IDF):
python seeds/seed_data.py

# Enable FAISS + Ollama embedder (default model: qwen3-embedding:latest)
export ENABLE_FAISS=1
export OLLAMA_EMBED_MODEL=qwen3-embedding:latest
python seeds/seed_data.py
```

3. Run API:

```bash
uvicorn api.main:app --reload
```

4. Run tests:

```bash
pytest -q
```

Run evaluation harness:

```bash
python -m core.eval --dataset examples/dataset_multi_hop.json
```

Notes
-----
This is an MVP and many components are stubs intended to be replaced with production adapters (real LLMs, FAISS, real BM25).
# RAG_Agent