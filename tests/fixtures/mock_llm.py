"""Mock LLM for deterministic testing."""
import hashlib
from typing import Dict, List, Optional, Any


class MockLLM:
    """Mock LLM client that returns deterministic responses."""
    
    def __init__(self):
        self.call_count = 0
        self.call_history = []
        self.custom_responses = {}
        self.default_behavior = "echo"  # echo, grounded, refuse
        self.failure_mode = None  # None, timeout, error, rate_limit
        
    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.call_history = []
        self.custom_responses = {}
        self.failure_mode = None
        
    def set_response(self, prompt_substring: str, response: str):
        """Set custom response for prompts containing substring."""
        self.custom_responses[prompt_substring] = response
        
    def set_failure_mode(self, mode: str):
        """Set failure mode: timeout, error, rate_limit."""
        self.failure_mode = mode
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response (synchronous)."""
        self.call_count += 1
        self.call_history.append({"prompt": prompt, "kwargs": kwargs})
        
        # Handle failure modes
        if self.failure_mode == "timeout":
            raise TimeoutError("LLM request timed out")
        elif self.failure_mode == "error":
            raise RuntimeError("LLM service error")
        elif self.failure_mode == "rate_limit":
            raise Exception("Rate limit exceeded")
            
        # Check custom responses
        for substring, response in self.custom_responses.items():
            if substring in prompt:
                return response
                
        # Default behaviors
        if self.default_behavior == "echo":
            return f"Mock response to: {prompt[:100]}..."
        elif self.default_behavior == "grounded":
            return self._generate_grounded_response(prompt)
        elif self.default_behavior == "refuse":
            return "I cannot assist with that request."
            
        return "Default mock response"
        
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Generate response (async)."""
        return self.generate(prompt, **kwargs)
        
    def _generate_grounded_response(self, prompt: str) -> str:
        """Generate a grounded response that cites sources."""
        # Extract context from prompt
        if "[Context]" in prompt:
            context_start = prompt.find("[Context]")
            context_end = prompt.find("[/Context]") if "[/Context]" in prompt else len(prompt)
            context = prompt[context_start:context_end]
            
            # Extract doc IDs from context
            doc_ids = []
            for line in context.split("\n"):
                if "doc" in line.lower() or "source" in line.lower():
                    # Simple extraction - look for doc1, doc2, etc.
                    words = line.split()
                    for word in words:
                        if word.startswith("doc") and any(c.isdigit() for c in word):
                            doc_ids.append(word.rstrip(".,;:"))
                            
            # Generate answer with citations
            if doc_ids:
                return f"According to the provided sources, the answer is found in {', '.join(doc_ids)}. [sources: {', '.join(doc_ids)}]"
                
        return "Based on the provided information, I can provide an answer. [sources: unknown]"
        
    def __call__(self, prompt: str, **kwargs) -> str:
        """Make instance callable."""
        return self.generate(prompt, **kwargs)
