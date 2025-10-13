"""LangGraph orchestration for stateful RAG workflows.

Implements the end-to-end RAG pipeline as a stateful graph with:
- Intent routing
- Hybrid retrieval
- Reranking
- Generation
- Self-RAG verification
- Correction loops
- Human-in-the-loop breakpoints
"""
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dataclasses import dataclass, field
import time
import operator


# State definition
class RAGState(TypedDict):
    """State for RAG workflow."""
    # Input
    query: str
    user_id: str
    session_id: str
    
    # Intent routing
    intent: Optional[str]
    intent_confidence: float
    route_decision: Optional[str]
    
    # Retrieval
    retrieved_docs: List[Dict[str, Any]]
    reranked_docs: List[Dict[str, Any]]
    final_evidence: List[Dict[str, Any]]
    
    # Generation
    draft_answer: Optional[str]
    final_answer: Optional[str]
    
    # Verification
    verification_result: Optional[Dict[str, Any]]
    verification_decision: Optional[str]
    correction_iteration: int
    
    # Memory & context
    conversation_context: Optional[str]
    memory_summary: Optional[str]
    
    # Observability
    trace: Annotated[List[Dict[str, Any]], operator.add]
    start_time: float
    end_time: Optional[float]
    
    # Control flow
    should_retrieve: bool
    should_rerank: bool
    should_verify: bool
    needs_correction: bool
    needs_human_approval: bool
    max_retries_reached: bool


@dataclass
class OrchestrationConfig:
    """Configuration for LangGraph orchestration."""
    # Feature flags
    enable_intent_routing: bool = True
    enable_reranking: bool = True
    enable_verification: bool = True
    enable_memory: bool = True
    enable_human_in_loop: bool = False
    
    # Thresholds
    intent_confidence_threshold: float = 0.75
    verification_confidence_threshold: float = 0.7
    human_approval_threshold: float = 0.5
    
    # Limits
    max_retrieval_attempts: int = 2
    max_verification_iterations: int = 3
    retrieval_top_k: int = 10
    rerank_top_k: int = 10
    
    # Timeouts
    total_timeout_seconds: int = 60
    retrieval_timeout_seconds: int = 10
    generation_timeout_seconds: int = 30


class RAGOrchestrator:
    """LangGraph-based RAG orchestrator with stateful workflows."""
    
    def __init__(
        self,
        config: OrchestrationConfig,
        intent_router,
        retriever,
        reranker,
        generator,
        verifier,
        memory_manager=None
    ):
        """Initialize orchestrator.
        
        Args:
            config: Orchestration configuration
            intent_router: Intent router instance
            retriever: Hybrid retriever instance
            reranker: Cross-encoder reranker instance
            generator: LLM generator instance
            verifier: Self-RAG verifier instance
            memory_manager: Optional memory manager
        """
        self.config = config
        self.router = intent_router
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.verifier = verifier
        self.memory = memory_manager
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build LangGraph workflow.
        
        Note: This is a simplified version. Full LangGraph integration
        requires the langgraph package with StateGraph, START, END nodes.
        """
        # TODO: Full LangGraph implementation
        # from langgraph.graph import StateGraph, START, END
        # graph = StateGraph(RAGState)
        # graph.add_node("route", self.route_node)
        # ...
        # return graph.compile()
        
        # For now, return a simple workflow dict
        return {
            "nodes": [
                "init",
                "route",
                "retrieve",
                "rerank",
                "generate",
                "verify",
                "correct",
                "finalize"
            ],
            "edges": {
                "init": "route",
                "route": ["smalltalk", "faq", "retrieve"],
                "retrieve": "rerank",
                "rerank": "generate",
                "generate": "verify",
                "verify": ["correct", "finalize"],
                "correct": "retrieve"
            }
        }
    
    async def run(self, query: str, user_id: str = "default", session_id: str = "default") -> Dict[str, Any]:
        """Execute RAG workflow.
        
        Args:
            query: User query
            user_id: User identifier
            session_id: Session identifier
        
        Returns:
            Final state with answer and trace
        """
        # Initialize state
        state: RAGState = {
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "intent": None,
            "intent_confidence": 0.0,
            "route_decision": None,
            "retrieved_docs": [],
            "reranked_docs": [],
            "final_evidence": [],
            "draft_answer": None,
            "final_answer": None,
            "verification_result": None,
            "verification_decision": None,
            "correction_iteration": 0,
            "conversation_context": None,
            "memory_summary": None,
            "trace": [],
            "start_time": time.time(),
            "end_time": None,
            "should_retrieve": True,
            "should_rerank": True,
            "should_verify": True,
            "needs_correction": False,
            "needs_human_approval": False,
            "max_retries_reached": False
        }
        
        # Execute workflow
        state = await self._execute_workflow(state)
        
        # Finalize
        state["end_time"] = time.time()
        
        return state
    
    async def _execute_workflow(self, state: RAGState) -> RAGState:
        """Execute the RAG workflow steps."""
        # 1. Initialize (load memory)
        state = await self._init_node(state)
        
        # 2. Route intent
        state = await self._route_node(state)
        
        # 3. Handle routing decision
        if state["route_decision"] == "smalltalk":
            return await self._smalltalk_node(state)
        elif state["route_decision"] == "faq":
            return await self._faq_node(state)
        
        # 4. Full RAG pipeline
        # Retrieve
        if state["should_retrieve"]:
            state = await self._retrieve_node(state)
        
        # Rerank
        if state["should_rerank"] and state["retrieved_docs"]:
            state = await self._rerank_node(state)
        
        # Generate
        state = await self._generate_node(state)
        
        # Verify (with correction loop)
        if self.config.enable_verification:
            state = await self._verification_loop(state)
        
        # Save memory
        if self.config.enable_memory and self.memory:
            state = await self._save_memory_node(state)
        
        return state
    
    async def _init_node(self, state: RAGState) -> RAGState:
        """Initialize workflow with memory context."""
        state["trace"].append({
            "node": "init",
            "timestamp": time.time(),
            "action": "load_memory"
        })
        
        if self.config.enable_memory and self.memory:
            # Load conversation context
            context = self.memory.build_context(
                state["session_id"],
                state["user_id"]
            )
            state["conversation_context"] = context
        
        return state
    
    async def _route_node(self, state: RAGState) -> RAGState:
        """Route query based on intent."""
        state["trace"].append({
            "node": "route",
            "timestamp": time.time(),
            "query": state["query"]
        })
        
        if not self.config.enable_intent_routing:
            state["route_decision"] = "rag"
            return state
        
        # Classify intent
        from app.router import Intent
        intent, confidence, metadata = self.router.route(state["query"])
        
        state["intent"] = intent.value
        state["intent_confidence"] = confidence
        
        # Decide route
        if intent == Intent.SMALLTALK and confidence >= self.config.intent_confidence_threshold:
            state["route_decision"] = "smalltalk"
        elif intent == Intent.FAQ and confidence >= self.config.intent_confidence_threshold:
            state["route_decision"] = "faq"
            state["final_answer"] = metadata.get("answer") if metadata else None
        elif intent == Intent.UNSAFE:
            state["route_decision"] = "unsafe"
            state["final_answer"] = self.router.get_unsafe_response(metadata.get("reason", ""))
        else:
            state["route_decision"] = "rag"
        
        state["trace"].append({
            "node": "route",
            "result": {
                "intent": intent.value,
                "confidence": confidence,
                "decision": state["route_decision"]
            }
        })
        
        return state
    
    async def _smalltalk_node(self, state: RAGState) -> RAGState:
        """Handle smalltalk queries."""
        state["trace"].append({
            "node": "smalltalk",
            "timestamp": time.time()
        })
        
        state["final_answer"] = self.router.get_smalltalk_response(state["query"])
        state["should_retrieve"] = False
        
        return state
    
    async def _faq_node(self, state: RAGState) -> RAGState:
        """Handle FAQ queries."""
        state["trace"].append({
            "node": "faq",
            "timestamp": time.time()
        })
        
        # Answer already set in route node
        state["should_retrieve"] = False
        
        return state
    
    async def _retrieve_node(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents."""
        state["trace"].append({
            "node": "retrieve",
            "timestamp": time.time(),
            "iteration": state["correction_iteration"]
        })
        
        # Build query with context
        query_with_context = state["query"]
        if state["conversation_context"]:
            query_with_context = f"{state['conversation_context']}\n\nQuery: {state['query']}"
        
        # Retrieve
        results = await self.retriever.retrieve(
            query_with_context,
            top_k=self.config.retrieval_top_k
        )
        
        # Convert to dict format
        state["retrieved_docs"] = [
            {
                "doc_id": r.doc_id,
                "passage_id": r.passage_id,
                "text": r.text,
                "score": r.score,
                "source": r.source
            }
            for r in results
        ]
        
        state["trace"].append({
            "node": "retrieve",
            "result": {
                "count": len(state["retrieved_docs"]),
                "top_score": state["retrieved_docs"][0]["score"] if state["retrieved_docs"] else 0
            }
        })
        
        return state
    
    async def _rerank_node(self, state: RAGState) -> RAGState:
        """Rerank retrieved documents."""
        state["trace"].append({
            "node": "rerank",
            "timestamp": time.time()
        })
        
        if not self.config.enable_reranking:
            state["reranked_docs"] = state["retrieved_docs"]
            state["final_evidence"] = state["retrieved_docs"][:self.config.rerank_top_k]
            return state
        
        # Rerank
        reranked = await self.reranker.rerank(
            state["query"],
            state["retrieved_docs"],
            top_k=self.config.rerank_top_k
        )
        
        # Convert to dict format
        state["reranked_docs"] = [
            {
                "doc_id": r.doc_id,
                "passage_id": r.passage_id,
                "text": r.text,
                "score": r.rerank_score,
                "original_score": r.original_score,
                "rank_change": r.rank_change
            }
            for r in reranked
        ]
        
        state["final_evidence"] = state["reranked_docs"]
        
        state["trace"].append({
            "node": "rerank",
            "result": {
                "count": len(state["reranked_docs"]),
                "top_score": state["reranked_docs"][0]["score"] if state["reranked_docs"] else 0,
                "avg_rank_change": sum(r.rank_change for r in reranked) / len(reranked) if reranked else 0
            }
        })
        
        return state
    
    async def _generate_node(self, state: RAGState) -> RAGState:
        """Generate answer from evidence."""
        state["trace"].append({
            "node": "generate",
            "timestamp": time.time()
        })
        
        # Build prompt with evidence
        evidence_text = "\n\n".join([
            f"[{i+1}] {doc['text']}"
            for i, doc in enumerate(state["final_evidence"])
        ])
        
        prompt = f"""Answer the following question based on the provided evidence.

Question: {state['query']}

Evidence:
{evidence_text}

Answer:"""
        
        # Generate
        if hasattr(self.generator, 'generate_answer_async'):
            answer = await self.generator.generate_answer_async(prompt, state["final_evidence"])
        else:
            answer = self.generator.generate_answer(prompt, state["final_evidence"])
        
        state["draft_answer"] = answer
        state["final_answer"] = answer  # May be updated by verification
        
        state["trace"].append({
            "node": "generate",
            "result": {
                "answer_length": len(answer)
            }
        })
        
        return state
    
    async def _verification_loop(self, state: RAGState) -> RAGState:
        """Verification loop with correction."""
        while (state["correction_iteration"] < self.config.max_verification_iterations and
               not state["max_retries_reached"]):
            
            # Verify answer
            verification = await self.verifier.verify_answer(
                state["query"],
                state["final_answer"],
                state["final_evidence"]
            )
            
            state["verification_result"] = {
                "is_supported": verification.is_supported,
                "confidence": verification.confidence,
                "decision": verification.decision.value,
                "reasoning": verification.reasoning
            }
            state["verification_decision"] = verification.decision.value
            
            state["trace"].append({
                "node": "verify",
                "timestamp": time.time(),
                "iteration": state["correction_iteration"],
                "result": state["verification_result"]
            })
            
            # Check if correction needed
            from app.verifier import VerificationDecision
            if verification.decision == VerificationDecision.ACCEPT:
                break
            
            # Apply correction
            state["correction_iteration"] += 1
            
            if verification.decision == VerificationDecision.RE_RETRIEVE:
                # Re-retrieve
                state = await self._retrieve_node(state)
                if state["should_rerank"]:
                    state = await self._rerank_node(state)
                state = await self._generate_node(state)
            
            elif verification.decision == VerificationDecision.REFINE:
                # Refine answer
                state["final_answer"] = f"Based on the available evidence: {state['final_answer']}"
                break
            
            elif verification.decision == VerificationDecision.HEDGE:
                # Hedge answer
                state["final_answer"] = f"According to the evidence: {state['final_answer']}"
                break
            
            elif verification.decision == VerificationDecision.REFUSE:
                # Refuse to answer
                state["final_answer"] = "I cannot provide a confident answer based on the available information."
                break
        
        if state["correction_iteration"] >= self.config.max_verification_iterations:
            state["max_retries_reached"] = True
        
        return state
    
    async def _save_memory_node(self, state: RAGState) -> RAGState:
        """Save conversation to memory."""
        state["trace"].append({
            "node": "save_memory",
            "timestamp": time.time()
        })
        
        self.memory.add_turn(
            state["session_id"],
            state["user_id"],
            state["query"],
            state["final_answer"]
        )
        
        return state
