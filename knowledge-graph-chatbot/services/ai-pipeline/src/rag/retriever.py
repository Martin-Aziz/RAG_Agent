"""
rag/retriever.py — Graph-aware retriever combining vector search + subgraph expansion.

Implements the retrieval stage of the RAG pipeline:
1. Embed the query → 384-dim vector
2. Vector search → top-K candidate nodes by semantic similarity
3. Subgraph expansion → 2-hop neighborhood from candidate nodes
4. Return enriched context for prompt construction

This two-stage approach (vector → graph) ensures both:
- Semantic relevance (via embeddings): find nodes about the query topic
- Structural context (via graph traversal): include related entities and paths
"""

from __future__ import annotations

from typing import List, Optional

from loguru import logger

from src.models.schemas import (
    GraphNode, GraphEdge, SubgraphResult, VectorSearchResult
)
from src.nlp.embedder import SentenceEmbedder


class GraphAwareRetriever:
    """Combines vector similarity search with graph traversal for retrieval.

    Retrieves relevant context from the knowledge graph by:
    1. Finding semantically similar nodes via vector search
    2. Expanding to include structurally connected nodes via BFS
    3. Merging and scoring results for prompt construction
    """

    def __init__(
        self,
        embedder: SentenceEmbedder,
        graph_client,  # gRPC client for graph-engine
        vector_top_k: int = 10,
        expansion_hops: int = 2,
        max_context_nodes: int = 40,
    ):
        """Initialize the retriever.

        Args:
            embedder: Sentence embedding model for query vectorization.
            graph_client: gRPC client to the graph-engine service.
            vector_top_k: Number of nearest neighbors to retrieve.
            expansion_hops: BFS depth for subgraph expansion.
            max_context_nodes: Maximum nodes in the returned context.
        """
        self.embedder = embedder
        self.graph_client = graph_client
        self.vector_top_k = vector_top_k
        self.expansion_hops = expansion_hops
        self.max_context_nodes = max_context_nodes

        logger.info(
            f"GraphAwareRetriever: top_k={vector_top_k}, "
            f"hops={expansion_hops}, max_nodes={max_context_nodes}"
        )

    async def retrieve(
        self,
        query: str,
        filter_labels: Optional[List[str]] = None,
    ) -> SubgraphResult:
        """Retrieve relevant subgraph context for a query.

        Steps:
        1. Embed the query into a vector
        2. Vector search for top-K relevant nodes
        3. Expand to 2-hop subgraph from candidate nodes
        4. Return the merged subgraph

        Args:
            query: Natural language query from the user.
            filter_labels: Optional filter to restrict to specific entity types.

        Returns:
            SubgraphResult with nodes and edges forming the query context.
        """
        logger.info(f"Retrieving context for: '{query[:100]}...'")

        # Step 1: Embed the query
        query_embedding = await self.embedder.embed(query)
        logger.debug(f"Query embedded: dim={len(query_embedding)}")

        # Step 2: Vector search for top-K relevant nodes
        try:
            vector_results = await self._vector_search(
                query_embedding, filter_labels
            )
            logger.info(f"Vector search: {len(vector_results)} candidates")
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            vector_results = []

        if not vector_results:
            logger.warning("No vector search results — returning empty context")
            return SubgraphResult()

        # Step 3: Expand to subgraph via BFS from candidate nodes
        seed_ids = [r.node_id for r in vector_results]

        try:
            subgraph = await self._expand_subgraph(seed_ids)
            logger.info(
                f"Subgraph expanded: {len(subgraph.nodes)} nodes, "
                f"{len(subgraph.edges)} edges"
            )
        except Exception as e:
            logger.error(f"Subgraph expansion failed: {e}")
            # Fallback: return just the vector search results as nodes
            subgraph = SubgraphResult(
                nodes=[r.node for r in vector_results if r.node],
            )

        return subgraph

    async def _vector_search(
        self,
        query_embedding: List[float],
        filter_labels: Optional[List[str]] = None,
    ) -> List[VectorSearchResult]:
        """Perform vector nearest neighbor search via graph-engine gRPC.

        In production, this calls graph_client.VectorSearch().
        For now, returns mock results for standalone testing.
        """
        if self.graph_client is None:
            logger.warning("No graph client — returning mock vector results")
            return self._mock_vector_results()

        # Production gRPC call would be:
        # response = await self.graph_client.VectorSearch(
        #     VectorSearchRequest(
        #         embedding=query_embedding,
        #         k=self.vector_top_k,
        #         label_filter=filter_labels or [],
        #     )
        # )
        # return [VectorSearchResult(...) for r in response.results]

        return self._mock_vector_results()

    async def _expand_subgraph(
        self, seed_ids: List[str]
    ) -> SubgraphResult:
        """Expand seed nodes into a multi-hop subgraph via graph-engine gRPC.

        In production, this calls graph_client.QuerySubgraph().
        """
        if self.graph_client is None:
            logger.warning("No graph client — returning mock subgraph")
            return self._mock_subgraph()

        # Production gRPC call would be:
        # response = await self.graph_client.QuerySubgraph(
        #     SubgraphQuery(
        #         seed_node_ids=seed_ids,
        #         max_hops=self.expansion_hops,
        #         max_nodes=self.max_context_nodes,
        #     )
        # )
        # return SubgraphResult(
        #     nodes=[...], edges=[...],
        #     total_nodes_visited=response.total_nodes_visited,
        #     traversal_time_ms=response.traversal_time_ms,
        # )

        return self._mock_subgraph()

    def _mock_vector_results(self) -> List[VectorSearchResult]:
        """Mock vector results for testing without graph-engine."""
        return [
            VectorSearchResult(
                node_id="cve-2021-44228",
                distance=0.15,
                node=GraphNode(
                    id="cve-2021-44228",
                    label="CVE",
                    name="CVE-2021-44228 (Log4Shell)",
                    properties={
                        "severity": "CRITICAL",
                        "cvss": "10.0",
                        "description": "Apache Log4j2 Remote Code Execution",
                    },
                    confidence=0.98,
                ),
            ),
            VectorSearchResult(
                node_id="sw-log4j",
                distance=0.22,
                node=GraphNode(
                    id="sw-log4j",
                    label="SOFTWARE",
                    name="Apache Log4j 2",
                    properties={"version": "2.0-2.14.1", "vendor": "Apache"},
                    confidence=0.95,
                ),
            ),
        ]

    def _mock_subgraph(self) -> SubgraphResult:
        """Mock subgraph for testing without graph-engine."""
        return SubgraphResult(
            nodes=[
                GraphNode(id="cve-2021-44228", label="CVE", name="CVE-2021-44228 (Log4Shell)",
                          properties={"severity": "CRITICAL", "cvss": "10.0"}, confidence=0.98),
                GraphNode(id="sw-log4j", label="SOFTWARE", name="Apache Log4j 2",
                          properties={"version": "2.0-2.14.1"}, confidence=0.95),
                GraphNode(id="ta-apt41", label="THREAT_ACTOR", name="APT41",
                          properties={"origin": "China"}, confidence=0.88),
                GraphNode(id="mit-upgrade", label="MITIGATION", name="Upgrade to Log4j 2.17.0",
                          properties={"status": "recommended"}, confidence=0.92),
            ],
            edges=[
                GraphEdge(id="e1", source_id="cve-2021-44228", target_id="sw-log4j",
                          relation_type="AFFECTS", weight=1.0, source_document="NVD"),
                GraphEdge(id="e2", source_id="ta-apt41", target_id="cve-2021-44228",
                          relation_type="EXPLOITS", weight=0.95, source_document="CISA Alert"),
                GraphEdge(id="e3", source_id="mit-upgrade", target_id="cve-2021-44228",
                          relation_type="MITIGATES", weight=1.0, source_document="Apache Advisory"),
            ],
            total_nodes_visited=4,
            traversal_time_ms=1.5,
        )
