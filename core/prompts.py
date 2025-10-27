"""Prompt templates and helpers for document-grounded assistance."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple


SYSTEM_PROMPT = (
    "You are GitHub Copilot, a highly knowledgeable, context-aware, document-grounded technical teacher AI built on a Retrieval-Augmented Generation (RAG) system. "
    "Your job is to mentor users through technical tasks using only the supplied knowledge base content.\n\n"
    "-- CORE PRINCIPLES --\n"
    "1. Act like a human instructor: be friendly, patient, and adaptive. Gauge the user's expertise and adjust depth accordingly.\n"
    "2. Maintain conversational state. You will receive the current `Teaching State`; update it as you guide the user.\n"
    "3. Use only the provided `Context Documents`. Do not rely on memory or external knowledge. If information is missing, say so clearly.\n"
    "4. Cite every factual claim using inline references like [doc_id] or [doc_id:passage_id].\n"
    "5. Encourage understanding: explain the \"why\" behind steps, offer optional theory, and propose mini check-ins when helpful.\n"
    "6. Be proactive with follow-ups (e.g., offer configuration samples, verification steps, or deeper dives).\n"
    "7. Never fabricate sources, and keep a professional yet supportive tone."
)


RESPONSE_FORMAT = (
    "{\n"
    '  "summary": "One-sentence overview of the guidance.",\n'
    '  "structured_response": "Markdown text that MUST include the sections: ✅ Requirements, 🧩 Steps, 💡 Tips or common pitfalls, 📘 Optional theory, 🤖 Follow-up, 📍 State. Cite sources inline.",\n'
    '  "citations": [\n'
    "    {\n"
    '      "document": "Source document identifier.",\n'
    '      "passage": "Quoted or paraphrased supporting text.",\n'
    "    }\n"
    "  ],\n"
    '  "teaching_state": {\n'
    '    "user_goal": "Updated understanding of the user goal.",\n'
    '    "current_step": "What stage of the guidance we are in.",\n'
    '    "prerequisites_met": "true/false/unknown with optional notes.",\n'
    '    "next_suggestion": "Optional suggestion for the next interaction."\n'
    "  },\n"
    '  "follow_up": "Question or offer inviting the user to continue."\n'
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


def _format_teaching_state(teaching_state: Optional[Dict[str, object]]) -> str:
    if not teaching_state:
        return (
            "user_goal: (not yet established)\n"
            "current_step: assessing request\n"
            "prerequisites_met: unknown\n"
            "next_suggestion: ask user which depth they prefer"
        )

    def _fmt_value(key: str) -> str:
        value = teaching_state.get(key)
        if value is None or value == "":
            return "(not set)"
        if isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value)
        if isinstance(value, dict):
            inner = "; ".join(f"{k}: {v}" for k, v in value.items())
            return inner or "(empty)"
        return str(value)

    lines = [
        f"user_goal: {_fmt_value('user_goal')}",
        f"current_step: {_fmt_value('current_step')}",
        f"prerequisites_met: {_fmt_value('prerequisites_met')}",
        f"next_suggestion: {_fmt_value('next_suggestion')}",
    ]
    if 'expertise' in teaching_state:
        lines.append(f"expertise: {_fmt_value('expertise')}")
    return "\n".join(lines)


def build_generation_prompt(
    user_question: str,
    context_pairs: Sequence[Tuple[str, str]],
    teaching_state: Optional[Dict[str, object]] = None,
) -> str:
    """Compose the full prompt for answer generation."""

    context_str = format_context_documents(context_pairs)
    if not context_str:
        context_str = "(no relevant context retrieved)"
    state_block = _format_teaching_state(teaching_state)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Current Teaching State:\n{state_block}\n\n"
        f"CONTEXT DOCUMENTS:\n{context_str}\n\n"
        f"USER QUESTION: \"{user_question}\"\n\n"
        f"Please provide the answer in the following JSON format:\n{RESPONSE_FORMAT}\n\n"
        "Answer (with evidence):"
    )
