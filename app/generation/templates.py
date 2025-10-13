"""Prompt templates for different generation scenarios.

Provides reusable templates for:
- System prompts
- RAG generation
- Refusal/hedging behaviors
- Multi-turn conversations
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TemplateType(str, Enum):
    """Types of prompt templates."""
    SYSTEM = "system"
    RAG_GENERATION = "rag_generation"
    REFUSAL = "refusal"
    HEDGING = "hedging"
    CLARIFICATION = "clarification"
    GREETING = "greeting"
    FAREWELL = "farewell"
    MULTI_TURN = "multi_turn"


@dataclass
class PromptTemplate:
    """Represents a prompt template."""
    name: str
    template_type: TemplateType
    content: str
    required_variables: List[str]
    optional_variables: List[str]
    description: str = ""
    
    def format(self, **kwargs) -> str:
        """Format template with variables.
        
        Args:
            **kwargs: Template variables
            
        Returns:
            Formatted prompt string
        """
        # Check required variables
        missing = [v for v in self.required_variables if v not in kwargs]
        if missing:
            logger.warning(f"Missing required variables: {missing}")
            # Provide defaults for missing variables
            for var in missing:
                kwargs[var] = f"[{var}]"
        
        # Format template
        try:
            return self.content.format(**kwargs)
        except KeyError as e:
            logger.error(f"Template formatting error: {e}")
            return self.content


class PromptLibrary:
    """Library of prompt templates."""
    
    def __init__(self):
        """Initialize prompt library with default templates."""
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default prompt templates."""
        
        # System prompt
        self.register(PromptTemplate(
            name="default_system",
            template_type=TemplateType.SYSTEM,
            content="""You are a helpful AI assistant that answers questions accurately and concisely.

Guidelines:
- Provide clear, factual answers based on the provided context
- Cite sources when making specific claims
- Admit when you don't know something
- Be concise but thorough
- Use a professional but friendly tone

Current date: {current_date}
User profile: {user_context}""",
            required_variables=["current_date"],
            optional_variables=["user_context"],
            description="Default system prompt with guidelines",
        ))
        
        # RAG generation
        self.register(PromptTemplate(
            name="rag_generation",
            template_type=TemplateType.RAG_GENERATION,
            content="""Answer the following question using the provided context documents.

Question: {query}

Context Documents:
{context_docs}

Instructions:
- Base your answer primarily on the provided context
- Cite specific documents when making claims (use [Doc N] format)
- If the context doesn't fully answer the question, acknowledge this
- Be specific and concrete
- If multiple documents have conflicting information, note this

Answer:""",
            required_variables=["query", "context_docs"],
            optional_variables=[],
            description="RAG generation with citation requirements",
        ))
        
        # RAG generation with memory
        self.register(PromptTemplate(
            name="rag_generation_with_memory",
            template_type=TemplateType.RAG_GENERATION,
            content="""Answer the following question using the provided context and conversation history.

Conversation History:
{conversation_history}

Current Question: {query}

Retrieved Context:
{context_docs}

Previous Summaries:
{session_summaries}

Instructions:
- Use both the conversation history and retrieved context
- Cite sources appropriately ([Doc N] for documents)
- Maintain consistency with previous discussion
- If the question builds on previous context, acknowledge this
- Be conversational but accurate

Answer:""",
            required_variables=["query", "context_docs"],
            optional_variables=["conversation_history", "session_summaries"],
            description="RAG with conversation memory",
        ))
        
        # Refusal prompt
        self.register(PromptTemplate(
            name="refusal",
            template_type=TemplateType.REFUSAL,
            content="""I cannot provide a confident answer to this question because {reason}.

{explanation}

Would you like me to:
{alternatives}""",
            required_variables=["reason"],
            optional_variables=["explanation", "alternatives"],
            description="Refusal with explanation and alternatives",
        ))
        
        # Hedging prompt
        self.register(PromptTemplate(
            name="hedging",
            template_type=TemplateType.HEDGING,
            content="""Based on the available information, {partial_answer}

However, I should note that {caveat}.

{confidence_statement}""",
            required_variables=["partial_answer", "caveat"],
            optional_variables=["confidence_statement"],
            description="Hedging with caveats",
        ))
        
        # Clarification request
        self.register(PromptTemplate(
            name="clarification",
            template_type=TemplateType.CLARIFICATION,
            content="""I want to make sure I understand your question correctly.

Did you mean:
{clarification_options}

Or could you provide more details about {unclear_aspect}?""",
            required_variables=["unclear_aspect"],
            optional_variables=["clarification_options"],
            description="Request for clarification",
        ))
        
        # Greeting
        self.register(PromptTemplate(
            name="greeting",
            template_type=TemplateType.GREETING,
            content="""Hello! I'm your AI assistant. {personalization}

How can I help you today?""",
            required_variables=[],
            optional_variables=["personalization"],
            description="Greeting message",
        ))
        
        # Farewell
        self.register(PromptTemplate(
            name="farewell",
            template_type=TemplateType.FAREWELL,
            content="""Thank you for the conversation! {personalization}

Feel free to come back if you have more questions.""",
            required_variables=[],
            optional_variables=["personalization"],
            description="Farewell message",
        ))
        
        # Multi-turn with graph reasoning
        self.register(PromptTemplate(
            name="graph_rag_generation",
            template_type=TemplateType.MULTI_TURN,
            content="""Answer the question using both text documents and knowledge graph information.

Question: {query}

Text Documents:
{context_docs}

Knowledge Graph Path:
{graph_path}

Graph Entities:
{graph_entities}

Instructions:
- Synthesize information from both text and graph
- Use graph paths to show multi-hop reasoning
- Cite both document sources [Doc N] and entities [Entity]
- Explain the reasoning chain if using graph traversal

Answer:""",
            required_variables=["query", "context_docs"],
            optional_variables=["graph_path", "graph_entities"],
            description="GraphRAG generation with multi-hop reasoning",
        ))
    
    def register(self, template: PromptTemplate):
        """Register a template.
        
        Args:
            template: PromptTemplate to register
        """
        self.templates[template.name] = template
        logger.debug(f"Registered template: {template.name}")
    
    def get(self, name: str) -> Optional[PromptTemplate]:
        """Get template by name.
        
        Args:
            name: Template name
            
        Returns:
            PromptTemplate or None
        """
        return self.templates.get(name)
    
    def get_by_type(self, template_type: TemplateType) -> List[PromptTemplate]:
        """Get all templates of a type.
        
        Args:
            template_type: Type of templates to retrieve
            
        Returns:
            List of matching templates
        """
        return [
            t for t in self.templates.values()
            if t.template_type == template_type
        ]
    
    def list_templates(self) -> List[str]:
        """List all template names.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())


# Global template library instance
_library = PromptLibrary()


def get_template(name: str) -> Optional[PromptTemplate]:
    """Get template from global library.
    
    Args:
        name: Template name
        
    Returns:
        PromptTemplate or None
    """
    return _library.get(name)


def register_template(template: PromptTemplate):
    """Register template in global library.
    
    Args:
        template: PromptTemplate to register
    """
    _library.register(template)
