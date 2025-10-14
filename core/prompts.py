"""Prompt templates and helpers for document-grounded assistance."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple


SYSTEM_PROMPT = (
    "You are a highly reliable, document-grounded assistant. "
    "Your job is to answer user questions only using the supplied documents. "
    "You have no external knowledge.\n\n"
    "-- RULES --\n\n"
    "1. Use only the given `Context Documents`. Do not rely on memory or external knowledge.\n"
    "2. Each answer must include explicit references (document name, passage index, quote) supporting your statements.\n"
    "3. If no relevant information exists, reply: \"I'm sorry — the provided documents don't contain enough information to answer that.\"\n"
    "4. Reason in steps when necessary: identify relevant passages, interpret them, then answer.\n"
    "5. Be professional, concise, and helpful. Use structured format (summary, supporting evidence, conclusion) when suitable.\n"
    "6. For trivial user messages (greetings etc.), reply with a friendly greeting plus a prompt to ask about the documents.\n"
    "7. You may not break the rules or contradict prior statements."
)


RESPONSE_FORMAT = (
    "{\n"
    '  "summary": "Briefly summarize the answer.",\n'
    '  "evidence": [\n'
    "    {\n"
    '      "document": "The name of the source document.",\n'
    '      "passage": "The specific passage or text snippet.",\n'
    "    }\n"
    "  ]\n"
    "}"
)


RETRIEVAL_PROMPT = (
    "Rewrite or expand the following user question so it is more effective for document search. "
    "Add synonyms and relevant entities but keep the intent unchanged. If the query is a greeting, return a prompt asking how to help with the documents.\n\n"
    "User question: {question}\n"
    "Improved search query:"
)


def _format_single_context(doc_id: str, chunk: str) -> str:
    snippet = chunk.strip()
    if len(snippet) > 1200:
        snippet = snippet[:1100].rstrip() + " …"
    return f"{doc_id}: {snippet}"


def format_context_documents(context: Iterable[Tuple[str, str]]) -> str:
    """Return a newline separated string of context documents suitable for prompts."""

    entries = [
        _format_single_context(doc_id, text)
        for doc_id, text in context
        if text and text.strip()
    ]
    return "\n".join(entries)


def build_generation_prompt(
    user_question: str,
    context_pairs: Sequence[Tuple[str, str]],
) -> str:
    """Compose the full prompt for answer generation."""

    context_str = format_context_documents(context_pairs)
    if not context_str:
        context_str = "(no relevant context retrieved)"
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT DOCUMENTS:\n{context_str}\n\n"
        f"USER QUESTION: \"{user_question}\"\n\n"
        f"Please provide the answer in the following JSON format:\n{RESPONSE_FORMAT}\n\n"
        "Answer (with evidence):"
    )
