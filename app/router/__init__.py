"""Intent routing and safety module."""
from typing import Dict, Tuple, Optional, List
from enum import Enum
import re
from app.config import get_config


class Intent(str, Enum):
    """Query intent types."""
    SMALLTALK = "smalltalk"
    FAQ = "faq"
    RAG = "rag"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


class IntentRouter:
    """Routes queries based on intent classification and safety checks."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize intent router.
        
        Args:
            config: Router configuration override
        """
        self.config = config or get_config().router
        
        # Compile smalltalk patterns
        self.smalltalk_patterns = [
            re.compile(rf'\b{pattern}\b', re.IGNORECASE)
            for pattern in self.config.intent_classifier.smalltalk_patterns
        ]
        
        # Jailbreak detection patterns
        self.jailbreak_patterns = [
            re.compile(r'ignore (previous|above|prior) instructions?', re.IGNORECASE),
            re.compile(r'disregard.*system', re.IGNORECASE),
            re.compile(r'you (are|is) (now|a) (?!assistant|helpful)', re.IGNORECASE),
            re.compile(r'act as if', re.IGNORECASE),
            re.compile(r'pretend (you|to be)', re.IGNORECASE)
        ]
        
        # FAQ templates (can be loaded from database/file in production)
        self.faq_templates = {
            "what is rag": {
                "answer": "RAG (Retrieval-Augmented Generation) is a technique that enhances LLMs by retrieving relevant context before generation.",
                "confidence": 1.0,
                "citations": ["docs/rag_overview.md"]
            },
            "how does this work": {
                "answer": "This system uses hybrid retrieval (BM25 + vector search), reranking, and Self-RAG verification to provide accurate answers.",
                "confidence": 1.0,
                "citations": ["docs/architecture.md"]
            }
        }
    
    def route(self, query: str) -> Tuple[Intent, float, Optional[Dict]]:
        """Route query to appropriate handler.
        
        Args:
            query: User query
        
        Returns:
            Tuple of (intent, confidence, metadata)
        """
        # Safety check first
        if self.config.safety.enabled:
            is_safe, reason = self._check_safety(query)
            if not is_safe:
                return Intent.UNSAFE, 1.0, {"reason": reason}
        
        # Check for smalltalk
        if self.config.intent_classifier.enabled:
            is_smalltalk, confidence = self._is_smalltalk(query)
            if is_smalltalk and confidence >= self.config.intent_classifier.confidence_threshold:
                return Intent.SMALLTALK, confidence, None
        
        # Check FAQ match
        if self.config.faq_matcher.enabled:
            faq_match, confidence = self._match_faq(query)
            if faq_match and confidence >= self.config.faq_matcher.similarity_threshold:
                return Intent.FAQ, confidence, faq_match
        
        # Default to RAG
        return Intent.RAG, 0.5, None
    
    def _check_safety(self, query: str) -> Tuple[bool, Optional[str]]:
        """Check if query is safe.
        
        Returns:
            Tuple of (is_safe, reason)
        """
        # Length check
        if len(query) > self.config.safety.max_query_length:
            return False, f"Query too long (max {self.config.safety.max_query_length} chars)"
        
        # Jailbreak detection
        if self.config.safety.jailbreak_detection:
            for pattern in self.jailbreak_patterns:
                if pattern.search(query):
                    return False, "Potential jailbreak attempt detected"
        
        return True, None
    
    def _is_smalltalk(self, query: str) -> Tuple[bool, float]:
        """Check if query is smalltalk.
        
        Returns:
            Tuple of (is_smalltalk, confidence)
        """
        query_lower = query.lower().strip()
        
        # Exact matches get high confidence
        for pattern in self.smalltalk_patterns:
            if pattern.search(query_lower):
                # Simple heuristic: short queries with pattern matches are likely smalltalk
                if len(query_lower.split()) <= 3:
                    return True, 0.95
                else:
                    return True, 0.75
        
        return False, 0.0
    
    def _match_faq(self, query: str) -> Tuple[Optional[Dict], float]:
        """Match query to FAQ templates.
        
        Returns:
            Tuple of (faq_data, confidence)
        """
        query_lower = query.lower().strip()
        
        # Simple keyword matching (can be replaced with embedding similarity)
        best_match = None
        best_score = 0.0
        
        for template_key, template_data in self.faq_templates.items():
            # Simple word overlap score
            template_words = set(template_key.split())
            query_words = set(query_lower.split())
            overlap = len(template_words & query_words)
            score = overlap / max(len(template_words), len(query_words))
            
            if score > best_score:
                best_score = score
                best_match = template_data
        
        return best_match, best_score
    
    def get_smalltalk_response(self, query: str) -> str:
        """Generate smalltalk response.
        
        Args:
            query: User query
        
        Returns:
            Response string
        """
        query_lower = query.lower().strip()
        
        # Simple template-based responses
        if any(p in query_lower for p in ["hello", "hi", "hey"]):
            return "Hello! I'm an AI assistant here to help you find information. What would you like to know?"
        
        if any(p in query_lower for p in ["goodbye", "bye"]):
            return "Goodbye! Feel free to return if you have more questions."
        
        if any(p in query_lower for p in ["thanks", "thank you"]):
            return "You're welcome! Let me know if you need anything else."
        
        return "I'm here to help! Please ask me a question."
    
    def get_unsafe_response(self, reason: str) -> str:
        """Generate response for unsafe queries.
        
        Args:
            reason: Reason for marking unsafe
        
        Returns:
            Response string
        """
        return f"I'm sorry, but I can't process this request. Reason: {reason}"
    
    def add_faq(self, key: str, answer: str, confidence: float = 1.0, citations: Optional[List[str]] = None):
        """Add FAQ template dynamically.
        
        Args:
            key: FAQ key (normalized query)
            answer: FAQ answer
            confidence: Confidence score
            citations: List of citation sources
        """
        self.faq_templates[key.lower()] = {
            "answer": answer,
            "confidence": confidence,
            "citations": citations or []
        }


class FastIntentClassifier:
    """Lightweight intent classifier for production use.
    
    Uses simple heuristics + optional embedding similarity for fast routing.
    For production, can be replaced with fine-tuned model.
    """
    
    def __init__(self, embedder=None):
        """Initialize classifier.
        
        Args:
            embedder: Optional embedder for similarity-based classification
        """
        self.embedder = embedder
        
        # Intent examples for similarity matching
        self.intent_examples = {
            Intent.SMALLTALK: [
                "hello",
                "hi there",
                "goodbye",
                "thanks",
                "how are you"
            ],
            Intent.FAQ: [
                "what is rag",
                "how does this work",
                "what can you do"
            ]
        }
        
        self.intent_embeddings = {}
        if self.embedder:
            self._build_intent_embeddings()
    
    def _build_intent_embeddings(self):
        """Build embeddings for intent examples."""
        for intent, examples in self.intent_examples.items():
            self.intent_embeddings[intent] = self.embedder.embed(examples)
    
    def classify(self, query: str, threshold: float = 0.75) -> Tuple[Intent, float]:
        """Classify query intent.
        
        Args:
            query: User query
            threshold: Confidence threshold
        
        Returns:
            Tuple of (intent, confidence)
        """
        if not self.embedder:
            # Fallback to rule-based
            return Intent.UNKNOWN, 0.0
        
        # Get query embedding
        query_embedding = self.embedder.embed([query])[0]
        
        # Compare with intent examples
        best_intent = Intent.UNKNOWN
        best_score = 0.0
        
        for intent, examples_embeddings in self.intent_embeddings.items():
            # Compute max similarity to any example
            scores = [
                self._cosine_similarity(query_embedding, ex_emb)
                for ex_emb in examples_embeddings
            ]
            max_score = max(scores)
            
            if max_score > best_score:
                best_score = max_score
                best_intent = intent
        
        return best_intent, best_score
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity."""
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
