"""
tests/test_rag.py — Unit tests for the RAG pipeline with mocked gRPC.
Tests prompt construction, context building, and pipeline orchestration.
"""

import pytest
from unittest.mock import AsyncMock

from src.models.schemas import (
    ChatMessage, ChatOptions, GraphNode, GraphEdge,
    SubgraphResult, TokenType,
)
from src.rag.prompt_builder import PromptBuilder
from src.rag.retriever import GraphAwareRetriever


class TestPromptBuilder:
    """Test the grounded prompt builder."""

    def setup_method(self):
        self.builder = PromptBuilder(topic_domain="Cybersecurity")

    def test_build_graph_context_with_entities(self):
        """Graph context should include formatted entity list."""
        subgraph = SubgraphResult(
            nodes=[
                GraphNode(id="n1", label="CVE", name="CVE-2021-44228",
                         properties={"severity": "CRITICAL"}, confidence=0.98),
            ],
            edges=[],
        )

        context = self.builder.build_graph_context(subgraph)
        assert "CVE-2021-44228" in context
        assert "[NODE:n1]" in context
        assert "CRITICAL" in context

    def test_build_graph_context_with_relationships(self):
        """Graph context should include formatted relationships."""
        subgraph = SubgraphResult(
            nodes=[
                GraphNode(id="n1", label="CVE", name="CVE-2021-44228"),
                GraphNode(id="n2", label="SOFTWARE", name="Log4j"),
            ],
            edges=[
                GraphEdge(id="e1", source_id="n1", target_id="n2",
                         relation_type="AFFECTS", weight=1.0,
                         source_document="NVD"),
            ],
        )

        context = self.builder.build_graph_context(subgraph)
        assert "--[AFFECTS]-->" in context
        assert "NVD" in context

    def test_build_empty_context(self):
        """Empty subgraph should produce a clear 'no info' message."""
        context = self.builder.build_graph_context(SubgraphResult())
        assert "not contain" in context.lower() or "no relevant" in context.lower()

    def test_build_prompt_includes_query(self):
        """Final prompt should include the user's query."""
        prompt = self.builder.build(
            query="What is CVE-2021-44228?",
            context="ENTITIES:\n- CVE-2021-44228",
            history=[],
        )
        assert "What is CVE-2021-44228?" in prompt

    def test_build_prompt_includes_history(self):
        """Prompt should include formatted conversation history."""
        history = [
            ChatMessage(role="user", content="Tell me about Log4j"),
            ChatMessage(role="assistant", content="Log4j is a Java logging library."),
        ]

        prompt = self.builder.build(
            query="What CVEs affect it?",
            context="",
            history=history,
        )
        assert "Tell me about Log4j" in prompt
        assert "Log4j is a Java logging library" in prompt


class TestGraphAwareRetriever:
    """Test the graph-aware retriever with mock data."""

    def setup_method(self):
        self.mock_embedder = AsyncMock()
        self.mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)

        self.retriever = GraphAwareRetriever(
            embedder=self.mock_embedder,
            graph_client=None,  # No real gRPC connection
        )

    @pytest.mark.asyncio
    async def test_retrieve_returns_subgraph(self):
        """Retriever should return a non-empty subgraph for valid queries."""
        result = await self.retriever.retrieve("CVE-2021-44228")

        assert len(result.nodes) > 0
        assert any(n.label == "CVE" for n in result.nodes)

    @pytest.mark.asyncio
    async def test_retrieve_embeds_query(self):
        """Retriever should embed the query before searching."""
        await self.retriever.retrieve("What are the latest threats?")
        self.mock_embedder.embed.assert_called_once()
