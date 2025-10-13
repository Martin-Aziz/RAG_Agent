# Agentic RAG - Architecture (MVP)

This document summarizes the implemented components in the MVP.

- API: FastAPI endpoint `/query` that accepts QueryRequest and returns structured QueryResponse.
- Orchestrator: planner (deterministic SLM stub), execution loop for PAR-RAG and HopRAG.
- Router: heuristic router that dispatches to vector or BM25 retrievers.
- Retrievers: Vector (TF-IDF based) and BM25 (simple lexical scoring).
- HopRAG: graph built from shared-word edges and BFS traversal.
- Tools: registry with pydantic schemas for validation and sandboxed execution for calculator and web search stub.
- Memory: episodic list + semantic VectorRetriever-backed store.
- Verifier: simple heuristic grader with pass/fail and corrective rewrite.

Tradeoffs: MVP uses deterministic stubs and heuristics for reproducibility. Replace TF-IDF, Vector store, and stubs with FAISS, real LLMs for production.
