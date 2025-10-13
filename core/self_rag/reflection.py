"""Self-RAG implementation with retrieval grading and hallucination checking.

Implements reflection loops for self-correction: grades retrieved documents,
checks for hallucinations, and triggers corrective actions.
"""
from typing import List, Dict, Any, Tuple, Optional
import asyncio


class SelfRAGVerifier:
    """Self-RAG verifier with LLM-based grading and hallucination detection."""
    
    def __init__(self, model_adapter, confidence_threshold: float = 0.7):
        """
        Args:
            model_adapter: LLM adapter with generate_answer_async method
            confidence_threshold: Minimum confidence for accepting results
        """
        self.model = model_adapter
        self.confidence_threshold = confidence_threshold
    
    async def grade_retrieval(self, query: str, document: Dict[str, Any]) -> Tuple[bool, float, str]:
        """Grade whether a retrieved document is relevant to the query.
        
        Returns:
            (is_relevant: bool, confidence: float, reasoning: str)
        """
        doc_text = document.get("text", "")[:500]  # Limit for efficiency
        
        prompt = f"""You are a relevance grader. Assess whether the following document contains information relevant to answering the query.

Query: {query}

Document: {doc_text}

Does this document contain relevant information to answer the query? 
Respond with ONLY: "Yes" or "No"
Then on a new line provide a confidence score from 0.0 to 1.0.
Then briefly explain why.

Format:
<verdict>Yes</verdict>
<confidence>0.85</confidence>
<reasoning>The document discusses...</reasoning>
"""
        
        try:
            if hasattr(self.model, 'generate_answer_async'):
                response = await self.model.generate_answer_async(prompt, [])
            else:
                response = self.model.generate_answer(prompt, [])
            
            # Parse response
            is_relevant, confidence, reasoning = self._parse_grade_response(response)
            return is_relevant, confidence, reasoning
        
        except Exception as e:
            # Fallback: assume relevant
            return True, 0.5, f"Grading failed: {e}"
    
    async def check_hallucination(
        self,
        query: str,
        answer: str,
        evidence: List[Dict[str, Any]]
    ) -> Tuple[bool, float, str]:
        """Check if generated answer is supported by evidence.
        
        Returns:
            (is_supported: bool, confidence: float, reasoning: str)
        """
        evidence_text = "\n".join([
            f"[{i+1}] {doc.get('text', '')[:200]}"
            for i, doc in enumerate(evidence[:5])
        ])
        
        prompt = f"""You are a fact-checker. Determine if the answer is fully supported by the provided evidence.

Query: {query}

Evidence:
{evidence_text}

Answer: {answer}

Is this answer fully supported by the evidence? Can you verify all claims?
Respond with ONLY: "Yes" or "No"
Then on a new line provide a confidence score from 0.0 to 1.0.
Then briefly explain which claims are supported or unsupported.

Format:
<verdict>Yes</verdict>
<confidence>0.90</confidence>
<reasoning>All claims are verified...</reasoning>
"""
        
        try:
            if hasattr(self.model, 'generate_answer_async'):
                response = await self.model.generate_answer_async(prompt, [])
            else:
                response = self.model.generate_answer(prompt, [])
            
            is_supported, confidence, reasoning = self._parse_grade_response(response)
            return is_supported, confidence, reasoning
        
        except Exception as e:
            # Fallback: assume supported
            return True, 0.5, f"Hallucination check failed: {e}"
    
    def _parse_grade_response(self, response: str) -> Tuple[bool, float, str]:
        """Parse LLM grading response."""
        import re
        
        # Extract verdict
        verdict_match = re.search(r'<verdict>(Yes|No)</verdict>', response, re.IGNORECASE)
        if verdict_match:
            verdict = verdict_match.group(1).lower() == "yes"
        else:
            # Fallback: check for Yes/No in first line
            first_line = response.split('\n')[0].strip().lower()
            verdict = "yes" in first_line
        
        # Extract confidence
        conf_match = re.search(r'<confidence>([\d.]+)</confidence>', response)
        if conf_match:
            confidence = float(conf_match.group(1))
        else:
            # Fallback: parse numbers from response
            numbers = re.findall(r'\b0\.\d+\b|\b1\.0\b', response)
            confidence = float(numbers[0]) if numbers else 0.6
        
        # Extract reasoning
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            # Use full response
            reasoning = response[:200]
        
        return verdict, confidence, reasoning


class CorrectiveRAGEngine:
    """Corrective RAG (CRAG) with dynamic correction strategies.
    
    Evaluates retrieval confidence and triggers:
    - High confidence (>0.8): Use as-is
    - Medium confidence (0.5-0.8): Knowledge refinement (extract key sentences)
    - Low confidence (<0.5): Web search fallback
    """
    
    def __init__(
        self,
        verifier: SelfRAGVerifier,
        web_search_tool: Optional[Any] = None,
        high_threshold: float = 0.8,
        low_threshold: float = 0.5
    ):
        """
        Args:
            verifier: SelfRAGVerifier for grading
            web_search_tool: Tool for web search fallback
            high_threshold: Confidence threshold for high-quality retrieval
            low_threshold: Confidence threshold below which to trigger fallback
        """
        self.verifier = verifier
        self.web_search_tool = web_search_tool
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
    
    async def correct_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Apply corrective strategy based on retrieval quality.
        
        Returns:
            (corrected_docs: List, strategy: str)
        """
        if not retrieved_docs:
            return await self._fallback_web_search(query)
        
        # Grade all documents
        graded = []
        for doc in retrieved_docs:
            is_relevant, confidence, reasoning = await self.verifier.grade_retrieval(query, doc)
            graded.append({
                "doc": doc,
                "is_relevant": is_relevant,
                "confidence": confidence,
                "reasoning": reasoning
            })
        
        # Compute average confidence
        relevant_grades = [g for g in graded if g["is_relevant"]]
        if not relevant_grades:
            # No relevant docs → web search
            return await self._fallback_web_search(query)
        
        avg_confidence = sum(g["confidence"] for g in relevant_grades) / len(relevant_grades)
        
        # Apply correction strategy
        if avg_confidence >= self.high_threshold:
            # High confidence: use as-is
            corrected = [g["doc"] for g in relevant_grades]
            return corrected, "high_confidence"
        
        elif avg_confidence >= self.low_threshold:
            # Medium confidence: refine knowledge (extract key sentences)
            corrected = await self._refine_knowledge(query, relevant_grades)
            return corrected, "knowledge_refinement"
        
        else:
            # Low confidence: web search fallback
            return await self._fallback_web_search(query)
    
    async def _refine_knowledge(
        self,
        query: str,
        graded_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract key sentences from documents for refinement."""
        refined = []
        for g in graded_docs:
            doc = g["doc"]
            text = doc.get("text", "")
            
            # Simple sentence extraction: split and rank by keyword overlap
            sentences = text.split(". ")
            query_words = set(query.lower().split())
            
            scored_sentences = []
            for sent in sentences:
                sent_words = set(sent.lower().split())
                overlap = len(query_words.intersection(sent_words))
                if overlap > 0:
                    scored_sentences.append((overlap, sent))
            
            # Keep top 2-3 sentences
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            key_sentences = [s for _, s in scored_sentences[:3]]
            
            if key_sentences:
                refined_doc = doc.copy()
                refined_doc["text"] = ". ".join(key_sentences)
                refined_doc["refined"] = True
                refined.append(refined_doc)
        
        return refined
    
    async def _fallback_web_search(self, query: str) -> Tuple[List[Dict[str, Any]], str]:
        """Fallback to web search when retrieval fails."""
        if self.web_search_tool is None:
            # No web search available
            return [], "no_fallback"
        
        try:
            # Call web search tool
            results = await self.web_search_tool.search(query, k=3)
            web_docs = [
                {
                    "doc_id": f"web_{i}",
                    "text": result.get("snippet", ""),
                    "source": result.get("url", ""),
                    "from_web": True
                }
                for i, result in enumerate(results)
            ]
            return web_docs, "web_search_fallback"
        
        except Exception as e:
            print(f"Web search fallback failed: {e}")
            return [], "fallback_failed"
