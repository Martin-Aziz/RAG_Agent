"""Generation module for structured answer generation.

Provides:
- Prompt templates and builders
- Citation formatting
- Constrained decoding for refusal/hedging
- Multi-turn conversation handling
"""

from .templates import PromptTemplate, PromptLibrary, TemplateType
from .builder import PromptBuilder, PromptContext
from .citations import CitationFormatter, CitationStyle
from .constraints import ConstrainedGenerator, GenerationConstraints, BehaviorType

__all__ = [
    "PromptTemplate",
    "PromptLibrary",
    "TemplateType",
    "PromptBuilder",
    "PromptContext",
    "CitationFormatter",
    "CitationStyle",
    "ConstrainedGenerator",
    "GenerationConstraints",
    "BehaviorType",
]
