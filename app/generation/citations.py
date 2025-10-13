"""Citation formatting for RAG responses.

Supports multiple citation styles:
- Inline: [Doc 1], [Doc 2]
- Footnote: [1], [2] with references at end
- APA-style: (Author, Year)
- Numeric: [1], [2], [3]
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CitationStyle(str, Enum):
    """Citation formatting styles."""
    INLINE = "inline"  # [Doc 1], [Doc 2]
    FOOTNOTE = "footnote"  # [1], [2] with references
    APA = "apa"  # (Author, Year)
    NUMERIC = "numeric"  # [1], [2], [3]
    NONE = "none"  # No citations


@dataclass
class Citation:
    """Represents a citation."""
    doc_id: str
    index: int
    text_snippet: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CitationFormatter:
    """Formats citations in generated text."""
    
    def __init__(
        self,
        style: CitationStyle = CitationStyle.INLINE,
        include_snippets: bool = False,
        max_snippet_length: int = 100,
    ):
        """Initialize citation formatter.
        
        Args:
            style: Citation style to use
            include_snippets: Include text snippets in references
            max_snippet_length: Maximum snippet length
        """
        self.style = style
        self.include_snippets = include_snippets
        self.max_snippet_length = max_snippet_length
    
    def format_answer_with_citations(
        self,
        answer: str,
        documents: List[Dict[str, Any]],
    ) -> Tuple[str, List[Citation]]:
        """Format answer with proper citations.
        
        Args:
            answer: Generated answer text
            documents: Source documents
            
        Returns:
            (formatted_answer, citations)
        """
        # Extract citation markers from answer
        citations = self._extract_citations(answer, documents)
        
        # Format based on style
        if self.style == CitationStyle.INLINE:
            formatted = self._format_inline(answer, citations)
        elif self.style == CitationStyle.FOOTNOTE:
            formatted = self._format_footnote(answer, citations)
        elif self.style == CitationStyle.APA:
            formatted = self._format_apa(answer, citations)
        elif self.style == CitationStyle.NUMERIC:
            formatted = self._format_numeric(answer, citations)
        else:
            formatted = self._remove_citations(answer)
        
        return formatted, citations
    
    def _extract_citations(
        self,
        answer: str,
        documents: List[Dict[str, Any]],
    ) -> List[Citation]:
        """Extract citations from answer text.
        
        Args:
            answer: Answer text with citation markers
            documents: Source documents
            
        Returns:
            List of citations
        """
        citations = []
        
        # Find patterns like [Doc 1], [Doc 2], etc.
        pattern = r'\[Doc (\d+)\]'
        matches = re.finditer(pattern, answer)
        
        for match in matches:
            doc_num = int(match.group(1))
            
            # Get corresponding document
            if 0 < doc_num <= len(documents):
                doc = documents[doc_num - 1]
                
                citation = Citation(
                    doc_id=doc.get("doc_id", f"doc{doc_num}"),
                    index=doc_num,
                    text_snippet=doc.get("text", "")[:self.max_snippet_length],
                    metadata=doc.get("metadata", {}),
                )
                
                if citation.doc_id not in [c.doc_id for c in citations]:
                    citations.append(citation)
        
        return citations
    
    def _format_inline(self, answer: str, citations: List[Citation]) -> str:
        """Format with inline citations [Doc 1].
        
        Args:
            answer: Answer text
            citations: List of citations
            
        Returns:
            Formatted text
        """
        # Inline citations are already in the text
        return answer
    
    def _format_footnote(self, answer: str, citations: List[Citation]) -> str:
        """Format with footnote-style citations.
        
        Args:
            answer: Answer text
            citations: List of citations
            
        Returns:
            Formatted text with references section
        """
        # Replace [Doc N] with [N]
        formatted = re.sub(r'\[Doc (\d+)\]', r'[\1]', answer)
        
        # Add references section
        if citations:
            formatted += "\n\nReferences:\n"
            for citation in citations:
                ref = f"[{citation.index}] {citation.doc_id}"
                
                if self.include_snippets and citation.text_snippet:
                    ref += f": {citation.text_snippet}..."
                
                formatted += ref + "\n"
        
        return formatted
    
    def _format_apa(self, answer: str, citations: List[Citation]) -> str:
        """Format with APA-style citations.
        
        Args:
            answer: Answer text
            citations: List of citations
            
        Returns:
            Formatted text with APA citations
        """
        # Convert [Doc N] to (Source, Year) if metadata available
        formatted = answer
        
        for citation in citations:
            author = citation.metadata.get("author", "Unknown")
            year = citation.metadata.get("year", "n.d.")
            
            # Replace first occurrence of [Doc N]
            pattern = f'\\[Doc {citation.index}\\]'
            replacement = f'({author}, {year})'
            formatted = re.sub(pattern, replacement, formatted, count=1)
        
        # Add references section
        if citations:
            formatted += "\n\nReferences:\n"
            for citation in citations:
                author = citation.metadata.get("author", "Unknown")
                year = citation.metadata.get("year", "n.d.")
                title = citation.metadata.get("title", citation.doc_id)
                
                ref = f"{author} ({year}). {title}."
                formatted += ref + "\n"
        
        return formatted
    
    def _format_numeric(self, answer: str, citations: List[Citation]) -> str:
        """Format with numeric citations [1], [2].
        
        Args:
            answer: Answer text
            citations: List of citations
            
        Returns:
            Formatted text with numeric citations
        """
        # Create citation map
        citation_map = {c.index: i + 1 for i, c in enumerate(citations)}
        
        # Replace [Doc N] with [numeric]
        formatted = answer
        for old_idx, new_idx in citation_map.items():
            pattern = f'\\[Doc {old_idx}\\]'
            formatted = re.sub(pattern, f'[{new_idx}]', formatted)
        
        # Add references section
        if citations:
            formatted += "\n\nReferences:\n"
            for i, citation in enumerate(citations, 1):
                ref = f"[{i}] {citation.doc_id}"
                
                if self.include_snippets and citation.text_snippet:
                    ref += f": {citation.text_snippet}..."
                
                formatted += ref + "\n"
        
        return formatted
    
    def _remove_citations(self, answer: str) -> str:
        """Remove all citation markers.
        
        Args:
            answer: Answer text
            
        Returns:
            Text with citations removed
        """
        # Remove [Doc N] patterns
        formatted = re.sub(r'\[Doc \d+\]', '', answer)
        
        # Clean up extra spaces
        formatted = re.sub(r'\s+', ' ', formatted)
        formatted = formatted.strip()
        
        return formatted
    
    def extract_cited_documents(
        self,
        answer: str,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract documents that were actually cited.
        
        Args:
            answer: Answer with citations
            documents: All source documents
            
        Returns:
            List of cited documents
        """
        citations = self._extract_citations(answer, documents)
        
        cited_docs = []
        for citation in citations:
            # Find matching document
            for doc in documents:
                if doc.get("doc_id") == citation.doc_id:
                    cited_docs.append(doc)
                    break
        
        return cited_docs
    
    def validate_citations(
        self,
        answer: str,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate that all citations are valid.
        
        Args:
            answer: Answer with citations
            documents: Source documents
            
        Returns:
            Validation results
        """
        # Find all citation markers
        pattern = r'\[Doc (\d+)\]'
        matches = re.findall(pattern, answer)
        
        valid_citations = []
        invalid_citations = []
        
        for match in matches:
            doc_num = int(match)
            if 1 <= doc_num <= len(documents):
                valid_citations.append(doc_num)
            else:
                invalid_citations.append(doc_num)
        
        return {
            "total_citations": len(matches),
            "valid_citations": len(set(valid_citations)),
            "invalid_citations": invalid_citations,
            "citation_coverage": len(set(valid_citations)) / len(documents) if documents else 0,
        }
    
    def add_citation_markers(
        self,
        text: str,
        relevant_doc_indices: List[int],
    ) -> str:
        """Add citation markers to text.
        
        Args:
            text: Text without citations
            relevant_doc_indices: Indices of relevant documents
            
        Returns:
            Text with citation markers added
        """
        # Simple approach: add citations at end of sentences
        sentences = text.split('. ')
        
        # Distribute citations across sentences
        citations_per_sentence = len(relevant_doc_indices) // len(sentences) + 1
        
        formatted_sentences = []
        doc_idx = 0
        
        for sentence in sentences:
            formatted_sentence = sentence
            
            # Add 1-2 citations to this sentence
            citations_to_add = min(citations_per_sentence, len(relevant_doc_indices) - doc_idx)
            
            if citations_to_add > 0:
                citation_strs = [
                    f"[Doc {relevant_doc_indices[doc_idx + i]}]"
                    for i in range(citations_to_add)
                ]
                formatted_sentence += " " + " ".join(citation_strs)
                doc_idx += citations_to_add
            
            formatted_sentences.append(formatted_sentence)
        
        return '. '.join(formatted_sentences)
