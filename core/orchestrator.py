from typing import List, Dict, Any, Optional
import os
import json
import asyncio
import time

from api.schemas import QueryRequest, QueryResponse, EvidenceItem, AgentStep
from core.prompts import build_generation_prompt
from core.model_adapters import SLMStub, OllamaAdapter
from core.router import Router
from core.agents.retriever_vector import VectorRetriever
from core.agents.retriever_bm25 import BM25Retriever
from core.agents.hoprag_graph import HopRAG
from core.agents.verifier import Verifier, EmbeddingVerifier
from core.agents.tool_agent import ToolRegistry
from core.agents.memory_agent import MemoryAgent
from core.agents.retriever_faiss import FAISSRetriever
from core.embedders.ollama_embedder import OllamaEmbedder

# Import new advanced capabilities
try:
    from core.retrieval.hybrid import HybridRetriever, CrossEncoderReranker
    from core.memory.conversation_memory import ConversationMemoryManager
    from core.self_rag.reflection import SelfRAGVerifier, CorrectiveRAGEngine
    from core.query_processing.advanced import MultiQueryExpander, HyDERetriever
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Advanced features not available: {e}")
    ADVANCED_FEATURES_AVAILABLE = False


GREETINGS = ["hello", "hi", "hey", "howdy", "greetings", "good morning", "good afternoon", "good evening"]

# Meta questions about the system itself
META_QUESTIONS = ["who are you", "what are you", "what is your name", "what can you do", 
                  "which model", "what model", "what llm", "which llm"]


def _is_meta_question(query: str) -> bool:
    """Check if the query is asking about the system itself."""
    query_lower = query.lower().strip()
    return any(meta in query_lower for meta in META_QUESTIONS)


def _calculate_relevance_score(query: str, evidence: List[Any]) -> float:
    """Calculate a simple relevance score based on keyword overlap."""
    if not evidence:
        return 0.0
    
    # Extract key terms from query (basic tokenization)
    query_terms = set(query.lower().split())
    # Remove common stop words
    stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'do', 'you', 'know', 'who', 'what', 'when', 'where', 'why', 'how'}
    query_terms = query_terms - stop_words
    
    if not query_terms:
        return 0.5  # Neutral if we can't extract meaningful terms
    
    # Check how many query terms appear in evidence
    max_overlap = 0
    for item in evidence:
        text = ""
        if isinstance(item, dict):
            text = item.get("text", "")
        elif hasattr(item, "text"):
            text = item.text
        else:
            text = str(item)
        
        text_terms = set(text.lower().split())
        overlap = len(query_terms.intersection(text_terms))
        max_overlap = max(max_overlap, overlap)
    
    # Return ratio of overlapping terms
    return min(1.0, max_overlap / len(query_terms))


class Orchestrator:
    def __init__(self):
        # allow selecting Ollama via env var
        use_ollama = os.getenv("USE_OLLAMA", "0") == "1" or os.getenv("OLLAMA_MODEL") is not None
        if use_ollama:
            self.model = OllamaAdapter()
        else:
            self.model = SLMStub()
        self.router = Router()
        
        # Initialize embedder if available
        self.embedder = None
        enable_faiss = os.getenv("ENABLE_FAISS", "0") == "1"
        if enable_faiss:
            embed_model = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:latest")
            if embed_model:
                self.embedder = OllamaEmbedder(model=embed_model)
                self.vector = FAISSRetriever(embedder=self.embedder)
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
        
        # Load persisted docs.json to populate retrievers
        try:
            docs_path = os.path.join("data", "docs.json")
            if os.path.exists(docs_path):
                with open(docs_path) as df:
                    docs = json.load(df)
                # index into retrievers (FAISS retriever uses index() method)
                try:
                    self.vector.index(docs)
                except Exception:
                    pass
                try:
                    self.bm25.index(docs)
                except Exception:
                    pass
        except Exception:
            pass
        
        self.hoprag = HopRAG(self.vector, self.bm25)
        self.verifier = Verifier()
        self.tools = ToolRegistry()
        self.memory = MemoryAgent()
        self.session_states = {}

        # Initialize advanced features if available
        self.use_advanced = os.getenv("USE_ADVANCED_RAG", "0") == "1" and ADVANCED_FEATURES_AVAILABLE
        
        if self.use_advanced:
            # Hybrid retriever with RRF
            self.hybrid_retriever = HybridRetriever(self.vector, self.bm25)
            
            # Cross-encoder reranker
            self.reranker = CrossEncoderReranker()
            
            # Conversation memory
            self.conversation_memory = ConversationMemoryManager(
                max_turns=10,
                max_tokens=4000,
                storage_path="data/memory"
            )
            
            # Self-RAG verifier
            self.self_rag_verifier = SelfRAGVerifier(self.model)
            
            # Corrective RAG
            self.corrective_rag = CorrectiveRAGEngine(
                verifier=self.self_rag_verifier,
                web_search_tool=None  # TODO: integrate web search
            )
            
            # Multi-query expander
            self.query_expander = MultiQueryExpander(self.model, num_variants=3)
            
            # HyDE retriever
            if self.embedder:
                self.hyde_retriever = HyDERetriever(self.model, self.embedder)
            else:
                self.hyde_retriever = None
            
            print("Advanced RAG features enabled: hybrid search, reranking, self-RAG, memory")
        else:
            self.hybrid_retriever = None
            self.reranker = None
            self.conversation_memory = None
            self.self_rag_verifier = None
            self.corrective_rag = None
            self.query_expander = None
            self.hyde_retriever = None

        # Setup traditional verifier
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

    def _prepare_context_pairs(self, evidence: List[EvidenceItem]) -> List[tuple]:
        """Convert evidence items into (doc_id, text) tuples for the prompt builder."""
        pairs = []
        for item in evidence:
            doc_label = item.doc_id or "Document"
            text = (item.text or "").strip()
            if text:
                pairs.append((doc_label, text))
        return pairs

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
        state = self._ensure_session_state(req.session_id, req.user_id)
        self._update_state_from_query(state, req.query)

        # Handle greetings and trivial queries
        if req.query.lower().strip() in GREETINGS:
            state["current_step"] = "greeting"
            state["next_suggestion"] = "invite question"
            if state.get("history"):
                state["history"][-1]["answer"] = "Greeting acknowledged"
            return QueryResponse(
                answer="Hello! I'm here to help you explore the knowledge base. Feel free to ask me anything about your documents.",
                evidence=[],
                trace=[AgentStep(step_id="greet-1", agent="orchestrator", action="greeting", result={"decision": "trivial"})],
                confidence=1.0
            )
        
        # Handle meta questions about the system
        if _is_meta_question(req.query):
            state["current_step"] = "answering meta question"
            if state.get("history"):
                state["history"][-1]["answer"] = "Provided system description"
            return QueryResponse(
                answer="I'm a document-grounded assistant that can only answer questions based on the provided documents in the knowledge base. I don't have information about myself or the models I use - I can only help you explore the documents you've uploaded.",
                evidence=[],
                trace=[AgentStep(step_id="meta-1", agent="orchestrator", action="meta_question", result={"decision": "out_of_scope"})],
                confidence=1.0
            )
        
        # Build conversation context if memory enabled
        context_prefix = ""
        if self.use_advanced and self.conversation_memory:
            context_prefix = self.conversation_memory.build_context(
                req.session_id,
                req.user_id,
                include_user_profile=True
            )
            if context_prefix:
                trace.append(AgentStep(
                    step_id="memory-1",
                    agent="memory",
                    action="load_context",
                    result={"context_length": len(context_prefix)}
                ))
        
        # Enhanced query with conversation context
        enhanced_query = f"{context_prefix}\n\nCurrent query: {req.query}" if context_prefix else req.query
        
        plan = await self.plan(req.query, req.mode)
        trace.append(AgentStep(step_id="planner-1", agent="planner", action="plan", result={"plan": plan}))
        state["current_step"] = f"retrieval planning ({len(plan)} step(s))"

        evidence = []
        
        # HopRAG mode short-circuit
        if req.mode == "hoprag":
            seeds = [s.get("seed") for s in plan]
            passages, t = await self.hoprag.traverse(seeds[0], max_hops=2)
            trace.extend(t)
            for p in passages:
                evidence.append(EvidenceItem(doc_id=p["doc_id"], passage_id=p["passage_id"], score=p.get("score", 1.0), text=p.get("text", "")))
            
            # Generate answer
            context_pairs = self._prepare_context_pairs(evidence)
            prompt_state = self._build_state_for_prompt(state)
            if hasattr(self.model, "generate_answer_async"):
                raw_answer = await self.model.generate_answer_async(enhanced_query, evidence, teaching_state=prompt_state)
            else:
                raw_answer = self.model.generate_answer(enhanced_query, evidence, teaching_state=prompt_state)

            answer = raw_answer
            follow_up = None
            try:
                response_data = json.loads(raw_answer)
                answer = response_data.get("structured_response") or response_data.get("summary", "Could not parse summary from model response.")
                follow_up = response_data.get("follow_up")
                self._merge_teaching_state(state, response_data.get("teaching_state", {}))
            except (json.JSONDecodeError, TypeError):
                pass
            state["current_step"] = "awaiting user feedback"
            if state.get("history"):
                state["history"][-1]["answer"] = answer
            if follow_up:
                state["next_suggestion"] = follow_up

            # Save to memory
            if self.use_advanced and self.conversation_memory:
                self.conversation_memory.add_turn(
                    req.session_id,
                    req.user_id,
                    req.query,
                    answer
                )
            
            return QueryResponse(answer=answer, evidence=evidence, trace=trace, confidence=0.6)

        # Advanced PAR-RAG execution with new features
        if self.use_advanced and self.hybrid_retriever and req.mode == "parrag":
            return await self._handle_advanced_parrag(req, enhanced_query, plan, trace, start, state)
        
        # Traditional PAR-RAG execution (fallback)
        return await self._handle_traditional_parrag(req, enhanced_query, plan, trace, start, state)
    
    async def _handle_advanced_parrag(
        self,
        req: QueryRequest,
        enhanced_query: str,
        plan: List[Dict[str, Any]],
        trace: List[AgentStep],
        start: float,
        state: Dict[str, Any],
    ) -> QueryResponse:
        """Advanced PAR-RAG with hybrid search, reranking, and self-RAG."""
        evidence = []
        state["current_step"] = f"retrieving evidence ({len(plan)} planned step(s))"
        
        for step in plan:
            step_id = step.get("id", f"step-{len(trace)+1}")
            instr = step.get("instruction")
            
            # Use multi-query expansion if enabled
            use_expansion = os.getenv("USE_QUERY_EXPANSION", "0") == "1"
            if use_expansion and self.query_expander:
                items = await self.query_expander.retrieve_with_expansion(
                    instr,
                    self.hybrid_retriever,
                    k=20  # Retrieve more for reranking
                )
                trace.append(AgentStep(
                    step_id=step_id,
                    agent="query_expander",
                    action="expand_retrieve",
                    result={"count": len(items)}
                ))
            else:
                # Hybrid retrieval with RRF
                items = self.hybrid_retriever.retrieve(instr, top_k=20, method="rrf")
                trace.append(AgentStep(
                    step_id=step_id,
                    agent="hybrid_retriever",
                    action="retrieve",
                    result={"count": len(items), "method": "rrf"}
                ))
            
            # Cross-encoder reranking
            if self.reranker and len(items) > 0:
                items = self.reranker.rerank(instr, items, top_k=10)
                trace.append(AgentStep(
                    step_id=step_id,
                    agent="reranker",
                    action="rerank",
                    result={"count": len(items)}
                ))
            
            # Corrective RAG: evaluate retrieval quality and apply corrections
            if self.corrective_rag and len(items) > 0:
                corrected_items, strategy = await self.corrective_rag.correct_retrieval(instr, items)
                items = corrected_items
                trace.append(AgentStep(
                    step_id=step_id,
                    agent="corrective_rag",
                    action="correct",
                    result={"strategy": strategy, "count": len(items)}
                ))
            
            # Add to evidence
            for it in items:
                evidence.append(EvidenceItem(
                    doc_id=it.get("doc_id", "doc"),
                    passage_id=it.get("passage_id", "p"),
                    score=it.get("score", 1.0),
                    text=it.get("text", "")
                ))
        
        # Check relevance of retrieved evidence to the original query
        relevance_score = _calculate_relevance_score(req.query, evidence)
        trace.append(AgentStep(
            step_id="relevance-1",
            agent="relevance_checker",
            action="check_relevance",
            result={"score": relevance_score}
        ))
        
        # If relevance is too low, refuse to answer
        if relevance_score < 0.3:
            state["current_step"] = "awaiting more context"
            state["next_suggestion"] = "provide additional details"
            if state.get("history"):
                state["history"][-1]["answer"] = "Insufficient relevant evidence"
            return QueryResponse(
                answer="I'm sorry — the provided documents don't contain information about that topic. I can only answer questions based on the documents in the knowledge base.",
                evidence=[],
                trace=trace,
                confidence=0.1
            )
        
        # Generate answer with conversation context
        context_pairs = self._prepare_context_pairs(evidence)
        if not context_pairs:
            state["current_step"] = "awaiting more context"
            state["next_suggestion"] = "share related documents"
            if state.get("history"):
                state["history"][-1]["answer"] = "No supporting passages retrieved"
            return QueryResponse(
                answer="I'm sorry — the provided documents don't contain enough information to answer that.",
                evidence=[],
                trace=trace,
                confidence=0.5
            )

        prompt_state = self._build_state_for_prompt(state)
        if hasattr(self.model, "generate_answer_async"):
            raw_answer = await self.model.generate_answer_async(enhanced_query, evidence, teaching_state=prompt_state)
        else:
            raw_answer = self.model.generate_answer(enhanced_query, evidence, teaching_state=prompt_state)

        answer = raw_answer
        follow_up = None
        try:
            response_data = json.loads(raw_answer)
            answer = response_data.get("structured_response") or response_data.get("summary", "Could not parse summary from model response.")
            follow_up = response_data.get("follow_up")
            self._merge_teaching_state(state, response_data.get("teaching_state", {}))
        except (json.JSONDecodeError, TypeError):
            pass
        state["current_step"] = "awaiting user feedback"
        if state.get("history"):
            state["history"][-1]["answer"] = answer
        if follow_up:
            state["next_suggestion"] = follow_up

        trace.append(AgentStep(
            step_id="synth-1",
            agent="synthesizer",
            action="synthesize",
            result={"answer_length": len(answer)}
        ))
        
        # Hallucination check
        if self.self_rag_verifier:
            is_supported, confidence, reasoning = await self.self_rag_verifier.check_hallucination(
                req.query,
                answer,
                evidence
            )
            trace.append(AgentStep(
                step_id="hallucination-check",
                agent="self_rag",
                action="check_hallucination",
                result={
                    "is_supported": is_supported,
                    "confidence": confidence,
                    "reasoning": reasoning
                }
            ))
            
            # If not supported, trigger correction
            if not is_supported and confidence < 0.6:
                answer = "I cannot provide a confident answer based on the available evidence. " + answer
        
        # Save to conversation memory
        if self.conversation_memory:
            self.conversation_memory.add_turn(
                req.session_id,
                req.user_id,
                req.query,
                answer
            )
        
        elapsed = time.time() - start
        trace.append(AgentStep(
            step_id="meta-1",
            agent="orchestrator",
            action="timing",
            result={"elapsed": elapsed}
        ))
        
        return QueryResponse(
            answer=answer,
            evidence=evidence,
            trace=trace,
            confidence=0.85 if len(evidence) > 0 else 0.3
        )
    
    async def _handle_traditional_parrag(
        self,
        req: QueryRequest,
        enhanced_query: str,
        plan: List[Dict[str, Any]],
        trace: List[AgentStep],
        start: float,
        state: Dict[str, Any],
    ) -> QueryResponse:
        """Traditional PAR-RAG execution (original implementation)."""
        evidence = []
        state["current_step"] = f"retrieving evidence ({len(plan)} planned step(s))"
        
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

        # Check relevance of retrieved evidence to the original query
        relevance_score = _calculate_relevance_score(req.query, evidence)
        trace.append(AgentStep(
            step_id="relevance-1",
            agent="relevance_checker",
            action="check_relevance",
            result={"score": relevance_score}
        ))
        
        # If relevance is too low, refuse to answer
        if relevance_score < 0.3:
            state["current_step"] = "awaiting more context"
            state["next_suggestion"] = "provide additional details"
            if state.get("history"):
                state["history"][-1]["answer"] = "Insufficient relevant evidence"
            return QueryResponse(
                answer="I'm sorry — the provided documents don't contain information about that topic. I can only answer questions based on the documents in the knowledge base.",
                evidence=[],
                trace=trace,
                confidence=0.1
            )

        # final synthesis
        context_pairs = self._prepare_context_pairs(evidence)
        if not context_pairs:
            state["current_step"] = "awaiting more context"
            state["next_suggestion"] = "share related documents"
            if state.get("history"):
                state["history"][-1]["answer"] = "No supporting passages retrieved"
            return QueryResponse(
                answer="I'm sorry — the provided documents don't contain enough information to answer that.",
                evidence=[],
                trace=trace,
                confidence=0.5
            )

        prompt_state = self._build_state_for_prompt(state)
        if hasattr(self.model, "generate_answer_async"):
            raw_answer = await self.model.generate_answer_async(enhanced_query, evidence, teaching_state=prompt_state)
        else:
            raw_answer = self.model.generate_answer(enhanced_query, evidence, teaching_state=prompt_state)

        answer = raw_answer
        follow_up = None
        try:
            response_data = json.loads(raw_answer)
            answer = response_data.get("structured_response") or response_data.get("summary", "Could not parse summary from model response.")
            follow_up = response_data.get("follow_up")
            self._merge_teaching_state(state, response_data.get("teaching_state", {}))
        except (json.JSONDecodeError, TypeError):
            pass
        state["current_step"] = "awaiting user feedback"
        if state.get("history"):
            state["history"][-1]["answer"] = answer
        if follow_up:
            state["next_suggestion"] = follow_up

        trace.append(AgentStep(step_id="synth-1", agent="synthesizer", action="synthesize", result={"answer": answer[:100]}))
        
        # Save to memory if advanced features enabled
        if self.use_advanced and self.conversation_memory:
            self.conversation_memory.add_turn(
                req.session_id,
                req.user_id,
                req.query,
                answer
            )
        
        elapsed = time.time() - start
        trace.append(AgentStep(step_id="meta-1", agent="orchestrator", action="timing", result={"elapsed": elapsed}))
        return QueryResponse(answer=answer, evidence=evidence, trace=trace, confidence=0.7 if relevance_score > 0.5 else 0.4)

    def _ensure_session_state(self, session_id: str, user_id: str) -> Dict[str, Any]:
        key = (session_id, user_id)
        if key not in self.session_states:
            self.session_states[key] = {
                "user_goal": None,
                "current_step": "initializing",
                "prerequisites_met": "unknown",
                "next_suggestion": "offer follow-up",
                "expertise": "intermediate",
                "history": [],
            }
        return self.session_states[key]

    @staticmethod
    def _infer_prerequisites_flag(query: str, state: Dict[str, Any]) -> str:
        lowered = query.lower()
        if any(token in lowered for token in ["install", "setup", "set up", "configure", "deploy"]):
            return "needs_confirmation"
        if any(token in lowered for token in ["already", "have", "installed", "configured", "done"]):
            return "likely_met"
        return state.get("prerequisites_met", "unknown")

    @staticmethod
    def _infer_expertise_level(query: str, state: Dict[str, Any]) -> str:
        lowered = query.lower()
        if any(phrase in lowered for phrase in ["i'm new", "beginner", "step by step", "explain like", "eli5", "walk me through"]):
            return "beginner"
        if any(phrase in lowered for phrase in ["advanced", "production", "optimize", "deep dive", "architecture"]):
            return "advanced"
        return state.get("expertise", "intermediate")

    def _update_state_from_query(self, state: Dict[str, Any], query: str) -> None:
        state.setdefault("history", []).append({
            "query": query,
            "timestamp": time.time(),
        })
        if not state.get("user_goal"):
            state["user_goal"] = query.strip()
        state["expertise"] = self._infer_expertise_level(query, state)
        state["prerequisites_met"] = self._infer_prerequisites_flag(query, state)
        state.setdefault("next_suggestion", "offer follow-up")
        state["current_step"] = "planning response"

    @staticmethod
    def _build_state_for_prompt(state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "user_goal": state.get("user_goal") or "(pending)",
            "current_step": state.get("current_step") or "planning response",
            "prerequisites_met": state.get("prerequisites_met", "unknown"),
            "next_suggestion": state.get("next_suggestion") or "offer follow-up",
            "expertise": state.get("expertise", "intermediate"),
        }

    def _merge_teaching_state(self, state: Dict[str, Any], update: Dict[str, Any]) -> None:
        if not update:
            return
        for key in ["user_goal", "current_step", "prerequisites_met", "next_suggestion", "expertise"]:
            value = update.get(key)
            if value:
                state[key] = value
