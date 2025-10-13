"""Self-RAG verification with retrieve-generate-critique loops.

Implements answer-evidence alignment, hallucination detection, and
correction loops following the Self-RAG paper (arXiv:2310.11511).
"""
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio


class VerificationDecision(str, Enum):
    """Verification decision outcomes."""
    ACCEPT = "accept"              # High confidence, answer is supported
    REFINE = "refine"              # Medium confidence, refine answer
    RE_RETRIEVE = "re_retrieve"    # Low relevance, need more evidence
    REFUSE = "refuse"              # Insufficient support, refuse to answer
    HEDGE = "hedge"                # Low confidence, hedge with caveats


@dataclass
class RetrievalGrade:
    """Grade for retrieved document relevance."""
    doc_id: str
    is_relevant: bool
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AnswerVerification:
    """Verification result for generated answer."""
    is_supported: bool
    confidence: float
    decision: VerificationDecision
    reasoning: str
    contradictions: List[str]
    unsupported_claims: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SelfRAGVerifier:
    """Self-RAG verifier with retrieval grading and answer verification.
    
    Implements the Self-RAG pattern:
    1. Grade retrieved documents for relevance
    2. Generate answer from relevant documents
    3. Verify answer support against evidence
    4. Trigger correction loops if needed
    """
    
    def __init__(
        self,
        model_adapter,
        min_relevance_score: float = 0.6,
        min_support_score: float = 0.7,
        hallucination_threshold: float = 0.3,
        max_retries: int = 2
    ):
        """Initialize Self-RAG verifier.
        
        Args:
            model_adapter: LLM adapter for grading and verification
            min_relevance_score: Minimum relevance to keep document
            min_support_score: Minimum support to accept answer
            hallucination_threshold: Threshold for hallucination detection
            max_retries: Maximum re-retrieval attempts
        """
        self.model = model_adapter
        self.min_relevance = min_relevance_score
        self.min_support = min_support_score
        self.hallucination_threshold = hallucination_threshold
        self.max_retries = max_retries
    
    async def grade_retrieval(
        self,
        query: str,
        document: Dict[str, Any]
    ) -> RetrievalGrade:
        """Grade whether a retrieved document is relevant to the query.
        
        Args:
            query: User query
            document: Retrieved document
        
        Returns:
            RetrievalGrade with relevance assessment
        """
        doc_text = document.get("text", "")[:500]  # Limit for efficiency
        doc_id = document.get("doc_id", "unknown")
        
        prompt = f"""You are a relevance grader. Assess whether the following document contains information relevant to answering the query.

Query: {query}

Document excerpt: {doc_text}

Does this document contain relevant information to answer the query?

Respond in this exact format:
VERDICT: [Yes/No]
CONFIDENCE: [0.0-1.0]
REASONING: [Brief explanation]
"""
        
        try:
            if hasattr(self.model, 'generate_answer_async'):
                response = await self.model.generate_answer_async(prompt, [])
            else:
                response = self.model.generate_answer(prompt, [])
            
            # Parse response
            is_relevant, confidence, reasoning = self._parse_grade_response(response)
            
            return RetrievalGrade(
                doc_id=doc_id,
                is_relevant=is_relevant and confidence >= self.min_relevance,
                confidence=confidence,
                reasoning=reasoning,
                metadata={"raw_response": response}
            )
        
        except Exception as e:
            print(f"⚠️  Retrieval grading failed: {e}")
            # Fallback: assume relevant with medium confidence
            return RetrievalGrade(
                doc_id=doc_id,
                is_relevant=True,
                confidence=0.5,
                reasoning=f"Grading failed: {str(e)}",
                metadata={"error": str(e)}
            )
    
    async def verify_answer(
        self,
        query: str,
        answer: str,
        evidence: List[Dict[str, Any]]
    ) -> AnswerVerification:
        """Verify if generated answer is supported by evidence.
        
        Args:
            query: User query
            answer: Generated answer
            evidence: List of evidence documents
        
        Returns:
            AnswerVerification with support assessment and decision
        """
        # Build evidence context
        evidence_text = "\n".join([
            f"[{i+1}] {doc.get('text', '')[:300]}"
            for i, doc in enumerate(evidence[:5])
        ])
        
        prompt = f"""You are a fact-checker. Determine if the answer is fully supported by the provided evidence.

Query: {query}

Evidence:
{evidence_text}

Answer: {answer}

Analyze:
1. Is each claim in the answer supported by the evidence?
2. Are there any contradictions between the answer and evidence?
3. Are there any unsupported claims?

Respond in this exact format:
SUPPORTED: [Yes/No]
CONFIDENCE: [0.0-1.0]
CONTRADICTIONS: [List any contradictions, or "None"]
UNSUPPORTED_CLAIMS: [List any unsupported claims, or "None"]
REASONING: [Brief explanation]
"""
        
        try:
            if hasattr(self.model, 'generate_answer_async'):
                response = await self.model.generate_answer_async(prompt, [])
            else:
                response = self.model.generate_answer(prompt, [])
            
            # Parse response
            is_supported, confidence, contradictions, unsupported, reasoning = \
                self._parse_verification_response(response)
            
            # Determine decision
            decision = self._determine_decision(
                is_supported,
                confidence,
                contradictions,
                unsupported
            )
            
            return AnswerVerification(
                is_supported=is_supported,
                confidence=confidence,
                decision=decision,
                reasoning=reasoning,
                contradictions=contradictions,
                unsupported_claims=unsupported,
                metadata={"raw_response": response}
            )
        
        except Exception as e:
            print(f"⚠️  Answer verification failed: {e}")
            # Fallback: accept with low confidence
            return AnswerVerification(
                is_supported=True,
                confidence=0.5,
                decision=VerificationDecision.HEDGE,
                reasoning=f"Verification failed: {str(e)}",
                contradictions=[],
                unsupported_claims=[],
                metadata={"error": str(e)}
            )
    
    def _determine_decision(
        self,
        is_supported: bool,
        confidence: float,
        contradictions: List[str],
        unsupported_claims: List[str]
    ) -> VerificationDecision:
        """Determine verification decision based on support analysis."""
        # Refuse if contradictions found
        if contradictions:
            return VerificationDecision.REFUSE
        
        # Refuse if confidence very low
        if confidence < self.hallucination_threshold:
            return VerificationDecision.REFUSE
        
        # Re-retrieve if multiple unsupported claims
        if len(unsupported_claims) > 2:
            return VerificationDecision.RE_RETRIEVE
        
        # Refine if some unsupported claims but mostly supported
        if unsupported_claims and is_supported:
            return VerificationDecision.REFINE
        
        # Hedge if low confidence but supported
        if confidence < self.min_support:
            return VerificationDecision.HEDGE
        
        # Accept if high confidence and fully supported
        return VerificationDecision.ACCEPT
    
    def _parse_grade_response(self, response: str) -> Tuple[bool, float, str]:
        """Parse retrieval grading response."""
        import re
        
        # Extract verdict
        verdict_match = re.search(r'VERDICT:\s*(Yes|No)', response, re.IGNORECASE)
        is_relevant = verdict_match.group(1).lower() == "yes" if verdict_match else False
        
        # Extract confidence
        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response)
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        
        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
        return is_relevant, confidence, reasoning
    
    def _parse_verification_response(
        self, response: str
    ) -> Tuple[bool, float, List[str], List[str], str]:
        """Parse answer verification response."""
        import re
        
        # Extract supported
        supported_match = re.search(r'SUPPORTED:\s*(Yes|No)', response, re.IGNORECASE)
        is_supported = supported_match.group(1).lower() == "yes" if supported_match else False
        
        # Extract confidence
        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response)
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        confidence = max(0.0, min(1.0, confidence))
        
        # Extract contradictions
        contra_match = re.search(r'CONTRADICTIONS:\s*(.+?)(?=UNSUPPORTED|REASONING|$)', response, re.DOTALL)
        contradictions_text = contra_match.group(1).strip() if contra_match else ""
        contradictions = [] if "none" in contradictions_text.lower() else [
            c.strip() for c in contradictions_text.split('\n') if c.strip()
        ]
        
        # Extract unsupported claims
        unsup_match = re.search(r'UNSUPPORTED_CLAIMS:\s*(.+?)(?=REASONING|$)', response, re.DOTALL)
        unsupported_text = unsup_match.group(1).strip() if unsup_match else ""
        unsupported = [] if "none" in unsupported_text.lower() else [
            c.strip() for c in unsupported_text.split('\n') if c.strip()
        ]
        
        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
        return is_supported, confidence, contradictions, unsupported, reasoning


class CorrectionEngine:
    """Correction engine for Self-RAG loops.
    
    Implements correction strategies based on verification decisions.
    """
    
    def __init__(self, verifier: SelfRAGVerifier, retriever):
        """Initialize correction engine.
        
        Args:
            verifier: SelfRAGVerifier instance
            retriever: Retriever instance for re-retrieval
        """
        self.verifier = verifier
        self.retriever = retriever
        self.correction_history = []
    
    async def apply_correction(
        self,
        query: str,
        answer: str,
        evidence: List[Dict[str, Any]],
        verification: AnswerVerification,
        iteration: int = 0
    ) -> Tuple[str, List[Dict[str, Any]], VerificationDecision]:
        """Apply correction strategy based on verification decision.
        
        Args:
            query: User query
            answer: Generated answer
            evidence: Current evidence
            verification: Verification result
            iteration: Current iteration count
        
        Returns:
            Tuple of (corrected_answer, updated_evidence, final_decision)
        """
        self.correction_history.append({
            "iteration": iteration,
            "decision": verification.decision,
            "confidence": verification.confidence
        })
        
        # Max iterations check
        if iteration >= self.verifier.max_retries:
            return self._apply_hedge(answer), evidence, VerificationDecision.HEDGE
        
        # Apply strategy based on decision
        if verification.decision == VerificationDecision.ACCEPT:
            return answer, evidence, VerificationDecision.ACCEPT
        
        elif verification.decision == VerificationDecision.HEDGE:
            return self._apply_hedge(answer), evidence, VerificationDecision.HEDGE
        
        elif verification.decision == VerificationDecision.REFINE:
            refined = self._refine_answer(answer, verification)
            return refined, evidence, VerificationDecision.REFINE
        
        elif verification.decision == VerificationDecision.RE_RETRIEVE:
            # Re-retrieve with refined query
            new_evidence = await self._re_retrieve(query, verification)
            return answer, new_evidence, VerificationDecision.RE_RETRIEVE
        
        elif verification.decision == VerificationDecision.REFUSE:
            return self._generate_refusal(verification), evidence, VerificationDecision.REFUSE
        
        return answer, evidence, verification.decision
    
    def _apply_hedge(self, answer: str) -> str:
        """Add hedging language to answer."""
        hedges = [
            "Based on the available information, ",
            "According to the evidence, ",
            "It appears that "
        ]
        
        # Don't double-hedge
        if any(h in answer for h in hedges):
            return answer
        
        return f"{hedges[0]}{answer}"
    
    def _refine_answer(self, answer: str, verification: AnswerVerification) -> str:
        """Refine answer by removing unsupported claims."""
        # Simple refinement: add caveat about unsupported claims
        if verification.unsupported_claims:
            caveat = "\n\nNote: Some aspects of this answer could not be fully verified with the available evidence."
            return answer + caveat
        return answer
    
    async def _re_retrieve(
        self,
        query: str,
        verification: AnswerVerification
    ) -> List[Dict[str, Any]]:
        """Re-retrieve with refined query."""
        # Refine query based on unsupported claims
        refined_query = query
        if verification.unsupported_claims:
            # Add missing aspects to query
            refined_query = f"{query} (specifically: {verification.unsupported_claims[0]})"
        
        # Re-retrieve
        try:
            if hasattr(self.retriever, 'retrieve'):
                new_docs = await asyncio.to_thread(
                    self.retriever.retrieve,
                    refined_query,
                    k=10
                )
                return new_docs
        except Exception as e:
            print(f"⚠️  Re-retrieval failed: {e}")
        
        return []
    
    def _generate_refusal(self, verification: AnswerVerification) -> str:
        """Generate refusal message."""
        base = "I cannot provide a confident answer to this question with the available information."
        
        if verification.contradictions:
            base += f" The evidence contains contradictions: {verification.contradictions[0]}"
        elif verification.unsupported_claims:
            base += " Some aspects of the answer cannot be verified with the available evidence."
        
        return base
    
    def get_correction_summary(self) -> Dict[str, Any]:
        """Get summary of correction history."""
        if not self.correction_history:
            return {"iterations": 0}
        
        return {
            "iterations": len(self.correction_history),
            "decisions": [h["decision"] for h in self.correction_history],
            "final_confidence": self.correction_history[-1]["confidence"],
            "converged": self.correction_history[-1]["decision"] == VerificationDecision.ACCEPT
        }
