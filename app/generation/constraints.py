"""Constrained generation for refusal and hedging behaviors.

Implements constraints for:
- Safety refusal (harmful content)
- Low confidence hedging
- Clarification requests
- Topic boundaries
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BehaviorType(str, Enum):
    """Generation behavior types."""
    NORMAL = "normal"  # Normal generation
    REFUSE = "refuse"  # Refuse to answer
    HEDGE = "hedge"  # Hedge with low confidence
    CLARIFY = "clarify"  # Request clarification
    REDIRECT = "redirect"  # Redirect to appropriate topic


@dataclass
class GenerationConstraints:
    """Constraints for generation."""
    behavior: BehaviorType = BehaviorType.NORMAL
    
    # Safety constraints
    allow_harmful: bool = False
    allow_personal_info: bool = False
    allow_medical_advice: bool = False
    allow_legal_advice: bool = False
    
    # Quality constraints
    min_confidence: float = 0.5
    min_support_score: float = 0.7
    require_citations: bool = True
    
    # Length constraints
    max_length: int = 500
    min_length: int = 50
    
    # Content constraints
    allowed_topics: Optional[List[str]] = None
    blocked_topics: Optional[List[str]] = None
    
    def should_refuse(self, context: Dict[str, Any]) -> bool:
        """Check if generation should be refused.
        
        Args:
            context: Generation context
            
        Returns:
            True if should refuse
        """
        # Check confidence
        if context.get("confidence", 1.0) < self.min_confidence:
            return True
        
        # Check support score
        if context.get("support_score", 1.0) < self.min_support_score:
            return True
        
        # Check safety flags
        if context.get("is_harmful", False) and not self.allow_harmful:
            return True
        
        if context.get("requests_medical_advice", False) and not self.allow_medical_advice:
            return True
        
        if context.get("requests_legal_advice", False) and not self.allow_legal_advice:
            return True
        
        return False
    
    def should_hedge(self, context: Dict[str, Any]) -> bool:
        """Check if generation should include hedging.
        
        Args:
            context: Generation context
            
        Returns:
            True if should hedge
        """
        confidence = context.get("confidence", 1.0)
        support = context.get("support_score", 1.0)
        
        # Hedge if confidence or support is low but above refusal threshold
        if self.min_confidence <= confidence < 0.8:
            return True
        
        if self.min_support_score <= support < 0.8:
            return True
        
        return False


class ConstrainedGenerator:
    """Generates text with behavioral constraints."""
    
    def __init__(
        self,
        model_adapter,
        constraints: Optional[GenerationConstraints] = None,
    ):
        """Initialize constrained generator.
        
        Args:
            model_adapter: LLM adapter
            constraints: Generation constraints
        """
        self.model = model_adapter
        self.constraints = constraints or GenerationConstraints()
    
    async def generate(
        self,
        prompt: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate with constraints.
        
        Args:
            prompt: Input prompt
            context: Generation context (confidence, support, etc.)
            
        Returns:
            Dictionary with generated text and metadata
        """
        # Determine behavior
        if self.constraints.should_refuse(context):
            return await self._generate_refusal(context)
        
        if self.constraints.should_hedge(context):
            return await self._generate_hedged(prompt, context)
        
        # Normal generation
        return await self._generate_normal(prompt, context)
    
    async def _generate_normal(
        self,
        prompt: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normal unconstrained generation.
        
        Args:
            prompt: Input prompt
            context: Context dictionary
            
        Returns:
            Generation result
        """
        try:
            response = await self.model.generate(
                prompt,
                max_tokens=self.constraints.max_length,
                temperature=0.7,
            )
            
            return {
                "text": response,
                "behavior": BehaviorType.NORMAL,
                "confidence": context.get("confidence", 1.0),
                "metadata": context,
            }
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {
                "text": "I apologize, but I encountered an error generating a response.",
                "behavior": BehaviorType.REFUSE,
                "error": str(e),
            }
    
    async def _generate_refusal(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate refusal message.
        
        Args:
            context: Context with refusal reason
            
        Returns:
            Refusal result
        """
        # Determine refusal reason
        reasons = []
        
        if context.get("confidence", 1.0) < self.constraints.min_confidence:
            reasons.append("low confidence in the answer")
        
        if context.get("support_score", 1.0) < self.constraints.min_support_score:
            reasons.append("insufficient supporting evidence")
        
        if context.get("is_harmful", False):
            reasons.append("the request may be harmful")
        
        if context.get("requests_medical_advice", False):
            reasons.append("I cannot provide medical advice")
        
        if context.get("requests_legal_advice", False):
            reasons.append("I cannot provide legal advice")
        
        reason = ", ".join(reasons) if reasons else "it doesn't meet safety guidelines"
        
        # Generate refusal message
        refusal_text = f"""I cannot provide a confident answer to this question because {reason}.

{self._get_refusal_explanation(context)}

{self._get_alternatives(context)}"""
        
        return {
            "text": refusal_text,
            "behavior": BehaviorType.REFUSE,
            "reasons": reasons,
            "metadata": context,
        }
    
    async def _generate_hedged(
        self,
        prompt: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate with hedging language.
        
        Args:
            prompt: Input prompt
            context: Context dictionary
            
        Returns:
            Hedged generation result
        """
        # Generate normal response first
        try:
            response = await self.model.generate(
                prompt,
                max_tokens=self.constraints.max_length,
                temperature=0.7,
            )
            
            # Add hedging language
            confidence = context.get("confidence", 0.7)
            support = context.get("support_score", 0.7)
            
            hedge_prefix = self._get_hedge_prefix(confidence, support)
            hedge_suffix = self._get_hedge_suffix(confidence, support)
            
            hedged_text = f"{hedge_prefix}\n\n{response}\n\n{hedge_suffix}"
            
            return {
                "text": hedged_text,
                "behavior": BehaviorType.HEDGE,
                "confidence": confidence,
                "support_score": support,
                "metadata": context,
            }
            
        except Exception as e:
            logger.error(f"Hedged generation error: {e}")
            return await self._generate_refusal(context)
    
    def _get_hedge_prefix(self, confidence: float, support: float) -> str:
        """Get hedging prefix based on confidence/support.
        
        Args:
            confidence: Confidence score
            support: Support score
            
        Returns:
            Hedge prefix string
        """
        if confidence < 0.6 or support < 0.6:
            return "Based on limited information, here's what I can share:"
        elif confidence < 0.7 or support < 0.7:
            return "While I'm not completely certain, here's my understanding:"
        else:
            return "Based on the available information:"
    
    def _get_hedge_suffix(self, confidence: float, support: float) -> str:
        """Get hedging suffix with caveats.
        
        Args:
            confidence: Confidence score
            support: Support score
            
        Returns:
            Hedge suffix string
        """
        caveats = []
        
        if confidence < 0.7:
            caveats.append("Please note that my confidence in this answer is moderate")
        
        if support < 0.7:
            caveats.append("the supporting evidence is limited")
        
        if caveats:
            return "⚠️ " + " and ".join(caveats) + ". Please verify this information from authoritative sources."
        
        return ""
    
    def _get_refusal_explanation(self, context: Dict[str, Any]) -> str:
        """Get detailed refusal explanation.
        
        Args:
            context: Context dictionary
            
        Returns:
            Explanation string
        """
        if context.get("is_harmful", False):
            return "This request appears to involve harmful content. I'm designed to be helpful, harmless, and honest."
        
        if context.get("requests_medical_advice", False):
            return "For medical concerns, please consult with a qualified healthcare professional."
        
        if context.get("requests_legal_advice", False):
            return "For legal matters, please consult with a licensed attorney."
        
        if context.get("confidence", 1.0) < self.constraints.min_confidence:
            return "The retrieved information doesn't provide sufficient confidence for a reliable answer."
        
        return "I don't have enough reliable information to answer this question confidently."
    
    def _get_alternatives(self, context: Dict[str, Any]) -> str:
        """Get alternative actions for user.
        
        Args:
            context: Context dictionary
            
        Returns:
            Alternatives string
        """
        alternatives = []
        
        if context.get("confidence", 1.0) < self.constraints.min_confidence:
            alternatives.append("Rephrase your question to be more specific")
            alternatives.append("Provide more context about what you're looking for")
        
        if context.get("is_harmful", False):
            alternatives.append("Ask a different question")
        
        if context.get("requests_medical_advice", False):
            alternatives.append("Consult a healthcare professional")
            alternatives.append("Visit an emergency room if urgent")
        
        if context.get("requests_legal_advice", False):
            alternatives.append("Consult a licensed attorney")
            alternatives.append("Contact your local legal aid office")
        
        if alternatives:
            alt_str = "\n".join(f"- {alt}" for alt in alternatives)
            return f"You might want to:\n{alt_str}"
        
        return ""
    
    def validate_output(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate generated output against constraints.
        
        Args:
            text: Generated text
            context: Generation context
            
        Returns:
            Validation results
        """
        results = {
            "valid": True,
            "violations": [],
        }
        
        # Check length constraints
        if len(text) < self.constraints.min_length:
            results["violations"].append(f"Too short: {len(text)} < {self.constraints.min_length}")
            results["valid"] = False
        
        if len(text) > self.constraints.max_length:
            results["violations"].append(f"Too long: {len(text)} > {self.constraints.max_length}")
            results["valid"] = False
        
        # Check citation requirement
        if self.constraints.require_citations:
            import re
            citations = re.findall(r'\[Doc \d+\]', text)
            if not citations:
                results["violations"].append("No citations found")
                results["valid"] = False
        
        return results
