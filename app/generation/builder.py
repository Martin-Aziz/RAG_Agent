"""Prompt builder for constructing generation prompts.

Builds prompts from:
- Templates
- Retrieved documents
- Memory context
- User preferences
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .templates import PromptLibrary, TemplateType, get_template

logger = logging.getLogger(__name__)


@dataclass
class PromptContext:
    """Context for prompt building."""
    query: str
    user_id: str
    session_id: str
    
    # Retrieved content
    documents: List[Dict[str, Any]] = field(default_factory=list)
    graph_results: Optional[Dict[str, Any]] = None
    
    # Memory context
    short_term_turns: List[Dict[str, Any]] = field(default_factory=list)
    session_summaries: List[Dict[str, Any]] = field(default_factory=list)
    user_profile: Optional[Dict[str, Any]] = None
    
    # Metadata
    intent: str = "rag"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptBuilder:
    """Builds prompts from templates and context."""
    
    def __init__(
        self,
        library: Optional[PromptLibrary] = None,
        max_context_docs: int = 5,
        max_doc_length: int = 500,
        include_citations: bool = True,
    ):
        """Initialize prompt builder.
        
        Args:
            library: PromptLibrary or None for default
            max_context_docs: Maximum documents to include
            max_doc_length: Maximum length per document
            include_citations: Include citation markers
        """
        self.library = library or PromptLibrary()
        self.max_context_docs = max_context_docs
        self.max_doc_length = max_doc_length
        self.include_citations = include_citations
    
    def build_rag_prompt(
        self,
        context: PromptContext,
        template_name: str = "rag_generation",
    ) -> str:
        """Build RAG generation prompt.
        
        Args:
            context: PromptContext with query and documents
            template_name: Template name to use
            
        Returns:
            Formatted prompt string
        """
        template = self.library.get(template_name)
        if not template:
            logger.error(f"Template not found: {template_name}")
            return f"Question: {context.query}\n\nAnswer:"
        
        # Format context documents
        context_docs = self._format_documents(context.documents)
        
        # Build variable dict
        variables = {
            "query": context.query,
            "context_docs": context_docs,
            "current_date": datetime.now().strftime("%Y-%m-%d"),
        }
        
        # Add optional memory context
        if context.short_term_turns:
            variables["conversation_history"] = self._format_conversation_history(
                context.short_term_turns
            )
        
        if context.session_summaries:
            variables["session_summaries"] = self._format_summaries(
                context.session_summaries
            )
        
        if context.user_profile:
            variables["user_context"] = self._format_user_profile(
                context.user_profile
            )
        
        # Format template
        return template.format(**variables)
    
    def build_graph_rag_prompt(
        self,
        context: PromptContext,
    ) -> str:
        """Build GraphRAG generation prompt with graph reasoning.
        
        Args:
            context: PromptContext with documents and graph results
            
        Returns:
            Formatted prompt string
        """
        template = self.library.get("graph_rag_generation")
        if not template:
            # Fallback to regular RAG
            return self.build_rag_prompt(context)
        
        # Format documents
        context_docs = self._format_documents(context.documents)
        
        # Format graph results
        graph_path = ""
        graph_entities = ""
        
        if context.graph_results:
            graph_path = self._format_graph_paths(
                context.graph_results.get("paths", [])
            )
            graph_entities = self._format_graph_entities(
                context.graph_results.get("nodes", [])
            )
        
        variables = {
            "query": context.query,
            "context_docs": context_docs,
            "graph_path": graph_path or "No graph paths found",
            "graph_entities": graph_entities or "No entities found",
        }
        
        return template.format(**variables)
    
    def build_refusal_prompt(
        self,
        reason: str,
        explanation: Optional[str] = None,
        alternatives: Optional[List[str]] = None,
    ) -> str:
        """Build refusal prompt.
        
        Args:
            reason: Reason for refusal
            explanation: Optional detailed explanation
            alternatives: Optional alternative actions
            
        Returns:
            Formatted refusal message
        """
        template = self.library.get("refusal")
        if not template:
            return f"I cannot answer this question because {reason}."
        
        variables = {
            "reason": reason,
            "explanation": explanation or "",
            "alternatives": self._format_alternatives(alternatives) if alternatives else "",
        }
        
        return template.format(**variables)
    
    def build_hedging_prompt(
        self,
        partial_answer: str,
        caveat: str,
        confidence: Optional[float] = None,
    ) -> str:
        """Build hedging prompt with caveats.
        
        Args:
            partial_answer: Partial answer to provide
            caveat: Caveat or limitation
            confidence: Optional confidence score
            
        Returns:
            Formatted hedging message
        """
        template = self.library.get("hedging")
        if not template:
            return f"{partial_answer}\n\nHowever, {caveat}"
        
        confidence_statement = ""
        if confidence is not None:
            confidence_statement = f"Confidence: {confidence:.0%}"
        
        variables = {
            "partial_answer": partial_answer,
            "caveat": caveat,
            "confidence_statement": confidence_statement,
        }
        
        return template.format(**variables)
    
    def build_system_prompt(
        self,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build system prompt.
        
        Args:
            user_profile: Optional user profile for personalization
            
        Returns:
            System prompt string
        """
        template = self.library.get("default_system")
        if not template:
            return "You are a helpful AI assistant."
        
        variables = {
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "user_context": self._format_user_profile(user_profile) if user_profile else "New user",
        }
        
        return template.format(**variables)
    
    def _format_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents for context.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Formatted document string
        """
        if not documents:
            return "No documents provided."
        
        # Limit number of documents
        docs_to_include = documents[:self.max_context_docs]
        
        lines = []
        for idx, doc in enumerate(docs_to_include, 1):
            text = doc.get("text", "")
            
            # Truncate long documents
            if len(text) > self.max_doc_length:
                text = text[:self.max_doc_length] + "..."
            
            if self.include_citations:
                doc_id = doc.get("doc_id", f"doc{idx}")
                score = doc.get("score", 0.0)
                lines.append(f"[Doc {idx}] ({doc_id}, score: {score:.3f})")
                lines.append(text)
            else:
                lines.append(f"Document {idx}:")
                lines.append(text)
            
            lines.append("")  # Blank line between documents
        
        return "\n".join(lines)
    
    def _format_conversation_history(self, turns: List[Dict[str, Any]]) -> str:
        """Format conversation history.
        
        Args:
            turns: List of turn dictionaries
            
        Returns:
            Formatted conversation string
        """
        if not turns:
            return "No previous conversation."
        
        lines = []
        for turn in turns[-5:]:  # Last 5 turns
            user_msg = turn.get("user_message", "")
            assistant_msg = turn.get("assistant_message", "")
            
            lines.append(f"User: {user_msg}")
            lines.append(f"Assistant: {assistant_msg}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_summaries(self, summaries: List[Dict[str, Any]]) -> str:
        """Format session summaries.
        
        Args:
            summaries: List of summary dictionaries
            
        Returns:
            Formatted summary string
        """
        if not summaries:
            return "No previous summaries."
        
        lines = []
        for idx, summary in enumerate(summaries, 1):
            summary_text = summary.get("summary_text", "")
            topics = summary.get("topics", [])
            
            lines.append(f"Summary {idx}:")
            lines.append(summary_text)
            if topics:
                lines.append(f"Topics: {', '.join(topics)}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_user_profile(self, profile: Dict[str, Any]) -> str:
        """Format user profile.
        
        Args:
            profile: User profile dictionary
            
        Returns:
            Formatted profile string
        """
        interests = profile.get("interests", [])
        preferences = profile.get("preferences", {})
        
        parts = []
        
        if interests:
            parts.append(f"Interests: {', '.join(interests[:5])}")
        
        if preferences:
            pref_strs = [f"{k}={v}" for k, v in list(preferences.items())[:3]]
            parts.append(f"Preferences: {', '.join(pref_strs)}")
        
        return "; ".join(parts) if parts else "No profile information"
    
    def _format_graph_paths(self, paths: List[List[Any]]) -> str:
        """Format graph paths.
        
        Args:
            paths: List of paths (lists of nodes)
            
        Returns:
            Formatted path string
        """
        if not paths:
            return "No paths found"
        
        lines = []
        for idx, path in enumerate(paths[:3], 1):  # Top 3 paths
            if isinstance(path, list):
                node_names = [
                    node.get("name", str(node)) if isinstance(node, dict) else str(node)
                    for node in path
                ]
                path_str = " -> ".join(node_names)
                lines.append(f"Path {idx}: {path_str}")
        
        return "\n".join(lines) if lines else "No valid paths"
    
    def _format_graph_entities(self, nodes: List[Any]) -> str:
        """Format graph entities.
        
        Args:
            nodes: List of graph nodes
            
        Returns:
            Formatted entity string
        """
        if not nodes:
            return "No entities found"
        
        lines = []
        for node in nodes[:10]:  # Top 10 entities
            if isinstance(node, dict):
                name = node.get("name", "")
                node_type = node.get("type", "")
                lines.append(f"- {name} ({node_type})")
        
        return "\n".join(lines) if lines else "No valid entities"
    
    def _format_alternatives(self, alternatives: List[str]) -> str:
        """Format alternative actions.
        
        Args:
            alternatives: List of alternatives
            
        Returns:
            Formatted alternatives string
        """
        if not alternatives:
            return ""
        
        lines = [f"- {alt}" for alt in alternatives]
        return "\n".join(lines)
