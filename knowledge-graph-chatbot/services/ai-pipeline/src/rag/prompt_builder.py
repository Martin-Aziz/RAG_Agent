"""
rag/prompt_builder.py — Grounded prompt construction for graph-based RAG.

Builds structured prompts that instruct the LLM to answer using ONLY
information from the knowledge graph context, with mandatory citations.

Design decisions:
- System prompt enforces grounded answering with [NODE:id] citations
- Graph context is serialized as structured text (entities + relationships)
- Conversation history is included for multi-turn coherence
- Topic domain is configurable via environment variable
"""

from __future__ import annotations

import os
from typing import List, Optional

from loguru import logger

from src.models.schemas import ChatMessage, SubgraphResult


# The topic domain is injected into the system prompt for domain-specific answers
TOPIC_DOMAIN = os.getenv(
    "TOPIC_DOMAIN",
    "Cybersecurity threat intelligence and CVE vulnerabilities"
)


# ============================================================================
# System prompt template — enforces grounded, citation-backed answers
# ============================================================================

GROUNDED_SYSTEM_PROMPT = """You are an expert assistant with access to a \
structured knowledge graph about {topic_domain}.

INSTRUCTIONS:
- Answer ONLY using information from the KNOWLEDGE GRAPH CONTEXT below
- Cite every fact with the node ID in brackets like [NODE:abc123]
- If the graph context does not contain enough information, explicitly say \
"The knowledge graph does not contain sufficient information to answer this question."
- Structure complex answers with logical reasoning steps
- Never hallucinate facts not present in the graph context
- When discussing relationships, mention the relationship type explicitly
- For cybersecurity topics, include severity levels and mitigation steps when available

KNOWLEDGE GRAPH CONTEXT:
{graph_context}

CONVERSATION HISTORY:
{history}

USER QUESTION: {query}

GROUNDED ANSWER (with citations):"""


class PromptBuilder:
    """Constructs grounded prompts from knowledge graph subgraphs.

    Serializes graph data into a structured text format that LLMs can
    reason over, and wraps it in a citation-enforcing system prompt.
    """

    def __init__(self, topic_domain: Optional[str] = None):
        self.topic_domain = topic_domain or TOPIC_DOMAIN
        logger.info(f"PromptBuilder initialized: domain='{self.topic_domain}'")

    def build(
        self,
        query: str,
        context: str,
        history: List[ChatMessage],
    ) -> str:
        """Build a complete prompt from query, graph context, and history.

        Args:
            query: The user's question.
            context: Serialized graph context from build_graph_context().
            history: Previous conversation turns for multi-turn coherence.

        Returns:
            Complete formatted prompt string ready for LLM inference.
        """
        # Format conversation history as a readable block
        history_text = self._format_history(history)

        # Fill in the template
        prompt = GROUNDED_SYSTEM_PROMPT.format(
            topic_domain=self.topic_domain,
            graph_context=context if context else "No relevant entities or relationships found.",
            history=history_text if history_text else "No previous conversation.",
            query=query,
        )

        logger.debug(
            f"Prompt built: query_len={len(query)}, "
            f"context_len={len(context)}, "
            f"history_turns={len(history)}, "
            f"total_len={len(prompt)}"
        )

        return prompt

    def build_graph_context(self, subgraph: SubgraphResult) -> str:
        """Serialize a subgraph into structured text for LLM consumption.

        Format:
        ENTITIES:
        - [NODE:id] EntityName (EntityType): property1=value1, property2=value2
        
        RELATIONSHIPS:
        - [NODE:src] SourceName --[RELATION_TYPE]--> [NODE:tgt] TargetName
          (source: document_id, confidence: 0.95)

        This structured format gives the LLM clear entity references to cite.
        """
        if not subgraph.nodes and not subgraph.edges:
            return "No relevant entities or relationships found in the knowledge graph."

        sections = []

        # Section 1: Entities with their properties
        if subgraph.nodes:
            entity_lines = ["ENTITIES:"]
            for node in subgraph.nodes:
                # Format properties as key=value pairs
                props = ", ".join(
                    f"{k}={v}" for k, v in node.properties.items()
                ) if node.properties else "no additional properties"

                entity_lines.append(
                    f"- [NODE:{node.id}] {node.name} ({node.label}): {props}"
                )

            sections.append("\n".join(entity_lines))

        # Section 2: Relationships with provenance
        if subgraph.edges:
            relation_lines = ["RELATIONSHIPS:"]

            # Build a lookup map for node names by ID
            node_names = {n.id: n.name for n in subgraph.nodes}

            for edge in subgraph.edges:
                src_name = node_names.get(edge.source_id, edge.source_id)
                tgt_name = node_names.get(edge.target_id, edge.target_id)

                provenance = ""
                if edge.source_document:
                    provenance = f" (source: {edge.source_document}"
                    if edge.weight > 0:
                        provenance += f", confidence: {edge.weight:.2f}"
                    provenance += ")"

                relation_lines.append(
                    f"- [NODE:{edge.source_id}] {src_name} "
                    f"--[{edge.relation_type}]--> "
                    f"[NODE:{edge.target_id}] {tgt_name}{provenance}"
                )

            sections.append("\n".join(relation_lines))

        return "\n\n".join(sections)

    def _format_history(self, history: List[ChatMessage]) -> str:
        """Format conversation history into a readable block.

        Only includes the last 10 turns to avoid exceeding context limits.
        """
        if not history:
            return ""

        # Keep only last 10 turns to avoid context overflow
        recent = history[-10:]

        lines = []
        for msg in recent:
            role = "User" if msg.role == "user" else "Assistant"
            # Truncate very long messages in history
            content = msg.content[:500]
            if len(msg.content) > 500:
                content += "..."
            lines.append(f"{role}: {content}")

        return "\n".join(lines)
