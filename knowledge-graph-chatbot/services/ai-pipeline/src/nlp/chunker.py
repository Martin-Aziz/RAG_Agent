"""
nlp/chunker.py — Semantic document chunker.

Splits documents into overlapping chunks suitable for NLP processing.
Uses sentence boundaries to avoid splitting in the middle of sentences.

Design decisions:
- Chunk by sentences rather than characters to preserve semantic coherence
- Overlap of 2 sentences between chunks to maintain cross-chunk context
- Max chunk size of 512 tokens to fit within REBEL model's context window
- Each chunk carries metadata about its position in the source document
"""

from __future__ import annotations

import re
import uuid
from typing import List

from loguru import logger

from src.models.schemas import Document, TextChunk


# Maximum characters per chunk. 512 tokens ≈ ~2000 chars for English text.
DEFAULT_MAX_CHUNK_SIZE = 2000
# Number of sentences to overlap between consecutive chunks
DEFAULT_SENTENCE_OVERLAP = 2
# Minimum chunk size to avoid creating tiny fragments
MIN_CHUNK_SIZE = 100


class SemanticChunker:
    """Splits documents into semantically coherent chunks with overlap.

    Uses sentence-boundary detection rather than arbitrary character splits
    to ensure each chunk contains complete sentences for better NER and
    relation extraction accuracy.
    """

    def __init__(
        self,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        sentence_overlap: int = DEFAULT_SENTENCE_OVERLAP,
    ):
        self.max_chunk_size = max_chunk_size
        self.sentence_overlap = sentence_overlap

        # Sentence-splitting regex: split on period/question/exclamation
        # followed by whitespace or end of string. Handles abbreviations
        # like "U.S." and "Dr." by requiring uppercase after the split.
        self._sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$'
        )

        logger.info(
            f"SemanticChunker initialized: max_size={max_chunk_size}, "
            f"overlap={sentence_overlap}"
        )

    def chunk_document(self, document: Document) -> List[TextChunk]:
        """Split a document into overlapping text chunks.

        Args:
            document: The source document to chunk.

        Returns:
            List of TextChunk objects with position metadata.
        """
        text = document.content.strip()
        if not text:
            logger.warning(f"Empty document: {document.id}")
            return []

        # Step 1: Split into sentences
        sentences = self._split_sentences(text)

        if not sentences:
            logger.warning(f"No sentences found in document: {document.id}")
            return []

        # Step 2: Group sentences into chunks respecting max size
        chunks = self._group_into_chunks(sentences, document)

        logger.info(
            f"Document {document.id}: {len(text)} chars → "
            f"{len(sentences)} sentences → {len(chunks)} chunks"
        )

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into individual sentences.

        Uses regex-based sentence boundary detection.
        Falls back to newline splitting for structured text (e.g., bullet points).
        """
        # First try regex sentence splitting
        sentences = self._sentence_pattern.split(text)

        # Filter out empty strings and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]

        # If we got very few sentences, the text might be structured (lists, etc.)
        # Fall back to paragraph/newline splitting
        if len(sentences) <= 1 and len(text) > self.max_chunk_size:
            sentences = [s.strip() for s in text.split('\n') if s.strip()]

        return sentences

    def _group_into_chunks(
        self,
        sentences: List[str],
        document: Document,
    ) -> List[TextChunk]:
        """Group sentences into chunks with overlap.

        Greedily adds sentences to the current chunk until max_chunk_size
        is reached, then starts a new chunk with sentence_overlap sentences
        carried over from the previous chunk.
        """
        chunks: List[TextChunk] = []
        current_sentences: List[str] = []
        current_size = 0
        char_offset = 0  # Track position in original document

        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence)

            # If adding this sentence would exceed max size, finalize current chunk
            if current_size + sentence_len > self.max_chunk_size and current_sentences:
                chunk_text = ' '.join(current_sentences)

                # Only create chunk if it meets minimum size
                if len(chunk_text) >= MIN_CHUNK_SIZE:
                    chunks.append(TextChunk(
                        id=str(uuid.uuid4()),
                        text=chunk_text,
                        document_id=document.id,
                        chunk_index=len(chunks),
                        start_char=char_offset,
                        end_char=char_offset + len(chunk_text),
                        metadata={
                            "sentence_count": str(len(current_sentences)),
                            "document_title": document.title,
                        },
                    ))

                # Start new chunk with overlap: carry over last N sentences
                overlap_start = max(0, len(current_sentences) - self.sentence_overlap)
                overlap_sentences = current_sentences[overlap_start:]

                # Update char offset
                non_overlap_text = ' '.join(current_sentences[:overlap_start])
                char_offset += len(non_overlap_text) + 1  # +1 for space

                current_sentences = overlap_sentences
                current_size = sum(len(s) for s in current_sentences)

            current_sentences.append(sentence)
            current_size += sentence_len

        # Don't forget the last chunk
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            if len(chunk_text) >= MIN_CHUNK_SIZE or not chunks:
                chunks.append(TextChunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    document_id=document.id,
                    chunk_index=len(chunks),
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text),
                    metadata={
                        "sentence_count": str(len(current_sentences)),
                        "document_title": document.title,
                    },
                ))

        return chunks
