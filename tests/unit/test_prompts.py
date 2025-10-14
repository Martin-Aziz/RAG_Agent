"""
Tests for core/prompts.py — prompt templates and context formatting.
"""
import pytest
from core.prompts import (
    SYSTEM_PROMPT,
    RETRIEVAL_PROMPT,
    format_context_documents,
    build_generation_prompt,
)


def test_system_prompt_has_required_rules():
    """The system prompt must include critical grounding rules."""
    assert "document-grounded" in SYSTEM_PROMPT.lower()
    assert "context documents" in SYSTEM_PROMPT.lower() or "given `context documents`" in SYSTEM_PROMPT.lower()
    assert "do not rely on memory" in SYSTEM_PROMPT.lower() or "no external knowledge" in SYSTEM_PROMPT.lower()


def test_retrieval_prompt_template():
    """The retrieval prompt should be a template with question placeholder."""
    assert "{question}" in RETRIEVAL_PROMPT
    filled = RETRIEVAL_PROMPT.format(question="What is AI?")
    assert "What is AI?" in filled


def test_format_context_documents_empty():
    """Should handle empty context gracefully."""
    result = format_context_documents([])
    assert result == ""


def test_format_context_documents_single():
    """Should format a single document correctly."""
    docs = [("Doc1", "This is a test document.")]
    result = format_context_documents(docs)
    assert "Doc1" in result
    assert "This is a test document." in result


def test_format_context_documents_multiple():
    """Should format multiple documents with newlines."""
    docs = [
        ("Doc1", "First document."),
        ("Doc2", "Second document."),
    ]
    result = format_context_documents(docs)
    lines = result.strip().split("\n")
    assert len(lines) == 2
    assert "Doc1" in lines[0]
    assert "Doc2" in lines[1]


def test_format_context_documents_truncates_long_chunks():
    """Should truncate overly long chunks to prevent bloat."""
    long_text = "A" * 2000
    docs = [("LongDoc", long_text)]
    result = format_context_documents(docs)
    # Should be truncated and contain ellipsis
    assert len(result) < len(long_text)
    assert "…" in result or "..." in result


def test_build_generation_prompt_includes_system():
    """The generation prompt should include the system prompt."""
    context = [("DocA", "Sample text.")]
    prompt = build_generation_prompt("What is this?", context)
    assert "document-grounded" in prompt.lower()
    assert "CONTEXT DOCUMENTS" in prompt
    assert "USER QUESTION" in prompt
    assert "What is this?" in prompt


def test_build_generation_prompt_no_context():
    """Should handle missing context gracefully with a fallback message."""
    prompt = build_generation_prompt("Any question?", [])
    assert "no relevant context retrieved" in prompt.lower()


def test_build_generation_prompt_full():
    """End-to-end test with realistic input."""
    context = [
        ("ReportA", "The project was completed on time."),
        ("ReportB", "Stakeholders approved the final deliverable."),
    ]
    prompt = build_generation_prompt("Was the project successful?", context)
    assert "ReportA" in prompt
    assert "ReportB" in prompt
    assert "Was the project successful?" in prompt
    assert "CONTEXT DOCUMENTS" in prompt
    assert len(prompt) > 200  # Should be substantial
