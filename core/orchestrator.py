from typing import List, Dict, Any
from api.schemas import QueryRequest, QueryResponse, EvidenceItem, AgentStep
from core.model_adapters import SLMStub, OllamaAdapter
import os
from core.router import Router
from core.agents.retriever_vector import VectorRetriever
from core.agents.retriever_bm25 import BM25Retriever
from core.agents.hoprag_graph import HopRAG
from core.agents.verifier import Verifier
from core.agents.verifier import EmbeddingVerifier
from core.agents.tool_agent import ToolRegistry
from core.agents.memory_agent import MemoryAgent
from core.agents.retriever_faiss import FAISSRetriever
from core.embedders.ollama_embedder import OllamaEmbedder
import os
import asyncio
import time


class Orchestrator:
    def __init__(self):
        # allow selecting Ollama via env var
        use_ollama = os.getenv("USE_OLLAMA", "0") == "1" or os.getenv("OLLAMA_MODEL") is not None
        if use_ollama:
            self.model = OllamaAdapter()
        else:
            self.model = SLMStub()
        self.router = Router()
        # optionally use FAISS with Ollama embedder
        enable_faiss = os.getenv("ENABLE_FAISS", "0") == "1"
        if enable_faiss:
            embed_model = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:latest")
            if embed_model:
                embedder = OllamaEmbedder(model=embed_model)
                self.vector = FAISSRetriever(embedder=embedder)
                # attempt to load persisted index if available
                try:
                    index_path = os.path.join("data", "faiss_index.iv")
                    mapping_path = os.path.join("data", "faiss_mapping.json")
                    if os.path.exists(index_path) and os.path.exists(mapping_path):
                        self.vector.load(index_path, mapping_path)
                        print(f"Loaded FAISS index from {index_path}")
                except Exception:
                    pass
            else:
                self.vector = FAISSRetriever()
        else:
            self.vector = VectorRetriever()
        self.bm25 = BM25Retriever()
        self.hoprag = HopRAG(self.vector, self.bm25)
        self.verifier = Verifier()
        self.tools = ToolRegistry()
        self.memory = MemoryAgent()

        # if vector is FAISS with embedder, use EmbeddingVerifier
        try:
            embedder = getattr(self.vector, "embedder", None)
            if embedder is not None:
                # dynamic threshold: if environment sets it use that, else base on expected instruction length
                env_th = os.getenv("VERIFIER_THRESHOLD")
                if env_th is not None:
                    threshold = float(env_th)
                else:
                    # allow EmbeddingVerifier to compute dynamic threshold per-instruction
                    threshold = None
                self.verifier = EmbeddingVerifier(embedder=embedder, threshold=threshold)
        except Exception:
            pass

    async def plan(self, query: str, mode: str) -> List[Dict[str, Any]]:
        # deterministic stub planner
        if mode == "parrag":
            # simple heuristic: split on ' and ' or ';' to make substeps
            parts = [p.strip() for p in query.replace(';', '.').split('.') if p.strip()]
            if len(parts) < 2:
                # create a 2-step plan: entity lookup, relation lookup
                return [
                    {"id": "step1", "type": "retrieval", "instruction": f"Find entity related to: {query}", "expected_tool": "vector_or_bm25"},
                    {"id": "step2", "type": "retrieval", "instruction": f"Find facts about the entity from step1", "expected_tool": "vector_or_bm25"}
                ]
            steps = []
            for i, p in enumerate(parts[:4]):
                steps.append({"id": f"step{i+1}", "type": "retrieval", "instruction": p, "expected_tool": "vector_or_bm25"})
            return steps
        elif mode == "hoprag":
            # return seed queries
            seeds = [{"seed": query}]
            return seeds
        else:
            return [{"id": "step1", "type": "retrieval", "instruction": query, "expected_tool": "vector_or_bm25"}]

    async def handle_query(self, req: QueryRequest) -> QueryResponse:
        start = time.time()
        trace: List[AgentStep] = []
        plan = await self.plan(req.query, req.mode)
        trace.append(AgentStep(step_id="planner-1", agent="planner", action="plan", result={"plan": plan}))

        evidence = []
        # HopRAG mode short-circuit
        if req.mode == "hoprag":
            seeds = [s.get("seed") for s in plan]
            passages, t = await self.hoprag.traverse(seeds[0], max_hops=2)
            trace.extend(t)
            for p in passages:
                evidence.append(EvidenceItem(doc_id=p["doc_id"], passage_id=p["passage_id"], score=p.get("score", 1.0), text=p.get("text", "")))
            # prefer async interface if available
            # convert to plain dicts for models expecting simple structures
            evidence_dicts = [ {"doc_id": e.doc_id, "passage_id": e.passage_id, "score": e.score, "text": e.text} for e in evidence ]
            if hasattr(self.model, "generate_answer_async"):
                answer = await self.model.generate_answer_async(req.query, evidence_dicts)
            else:
                answer = self.model.generate_answer(req.query, evidence_dicts)
            return QueryResponse(answer=answer, evidence=evidence, trace=trace, confidence=0.6)

        # PAR-RAG execution
        for step in plan:
            step_id = step.get("id", f"step-{len(trace)+1}")
            instr = step.get("instruction")
            # route
            route = self.router.route(instr)
            trace.append(AgentStep(step_id=step_id, agent="router", action="route", result={"route": route}))
            # retrieve
            items = []
            if route == "vector":
                items = self.vector.retrieve(instr, k=3)
            elif route == "bm25":
                items = self.bm25.retrieve(instr, k=3)
            else:
                # default vector
                items = self.vector.retrieve(instr, k=3)

            trace.append(AgentStep(step_id=step_id, agent="retriever", action="retrieve", result={"count": len(items)}))

            # verifier
            ok, verdict = self.verifier.grade(instr, items)
            # attach computed threshold if available
            trace.append(AgentStep(step_id=step_id, agent="verifier", action="grade", result={"ok": ok, "verdict": verdict}))

            if not ok:
                # self-correction: rewrite using model stub
                if hasattr(self.model, "rewrite_query_async"):
                    rewritten = await self.model.rewrite_query_async(instr)
                else:
                    rewritten = self.model.rewrite_query(instr)
                trace.append(AgentStep(step_id=step_id, agent="transformer", action="rewrite", result={"rewritten": rewritten}))
                # retry vector
                items = self.vector.retrieve(rewritten, k=3)
                trace.append(AgentStep(step_id=step_id, agent="retriever", action="retrieve_retry", result={"count": len(items)}))

            for it in items:
                evidence.append(EvidenceItem(doc_id=it.get("doc_id", "doc"), passage_id=it.get("passage_id", "p"), score=it.get("score", 1.0), text=it.get("text", "")))

        # final synthesis
        # convert evidence to plain dicts for model consumption
        evidence_dicts = [ {"doc_id": e.doc_id, "passage_id": e.passage_id, "score": e.score, "text": e.text} for e in evidence ]
        if hasattr(self.model, "generate_answer_async"):
            answer = await self.model.generate_answer_async(req.query, evidence_dicts)
        else:
            answer = self.model.generate_answer(req.query, evidence_dicts)
        trace.append(AgentStep(step_id="synth-1", agent="synthesizer", action="synthesize", result={"answer": answer}))
        elapsed = time.time() - start
        trace.append(AgentStep(step_id="meta-1", agent="orchestrator", action="timing", result={"elapsed": elapsed}))
        return QueryResponse(answer=answer, evidence=evidence, trace=trace, confidence=0.7)
