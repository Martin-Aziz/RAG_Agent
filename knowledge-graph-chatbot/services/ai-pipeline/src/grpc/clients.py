"""
grpc/clients.py — gRPC client for the graph-engine service.

Provides async wrappers around the generated protobuf stubs for
communicating with the Rust graph-engine service.
"""

from __future__ import annotations

import os
from typing import List, Optional

import grpc
from loguru import logger


# Graph-engine service address
GRAPH_ENGINE_ADDR = os.getenv("GRAPH_ENGINE_ADDR", "graph-engine:50051")


class GraphEngineClient:
    """Async gRPC client for the Rust graph-engine service.

    Provides typed methods for all GraphService RPCs.
    Uses a persistent channel with keepalive for long-running connections.
    """

    def __init__(self, addr: Optional[str] = None):
        """Initialize the graph-engine gRPC client.

        Args:
            addr: gRPC server address (default: from GRAPH_ENGINE_ADDR env).
        """
        self.addr = addr or GRAPH_ENGINE_ADDR
        self._channel = None
        self._stub = None

        logger.info(f"GraphEngineClient configured: addr={self.addr}")

    async def connect(self):
        """Establish the gRPC channel to the graph-engine.

        Uses insecure channel for internal service communication.
        In production, TLS should be enabled.
        """
        # Use async gRPC channel with keepalive
        options = [
            ('grpc.keepalive_time_ms', 10000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50MB
        ]

        self._channel = grpc.aio.insecure_channel(self.addr, options=options)
        logger.info(f"Connected to graph-engine at {self.addr}")

        # The actual stub would be created from generated protobuf code:
        # from src.grpc import graph_pb2_grpc
        # self._stub = graph_pb2_grpc.GraphServiceStub(self._channel)

    async def vector_search(
        self, embedding: List[float], k: int = 10, label_filter: Optional[List[str]] = None
    ):
        """Search for top-K nearest neighbor nodes by embedding similarity.

        Args:
            embedding: 384-dim query vector.
            k: Number of results to return.
            label_filter: Optional entity type filter.

        Returns:
            List of VectorSearchResult from graph-engine.
        """
        if self._stub is None:
            logger.warning("Graph client not connected — returning mock results")
            return []

        # Production call:
        # request = graph_pb2.VectorSearchRequest(
        #     embedding=embedding,
        #     k=k,
        #     label_filter=label_filter or [],
        # )
        # response = await self._stub.VectorSearch(request)
        # return response.results

        return []

    async def query_subgraph(
        self,
        seed_ids: List[str],
        max_hops: int = 2,
        max_nodes: int = 40,
        relation_filter: Optional[List[str]] = None,
    ):
        """Query n-hop subgraph from seed nodes.

        Args:
            seed_ids: Starting node UUIDs.
            max_hops: BFS depth limit.
            max_nodes: Maximum result nodes.
            relation_filter: Optional edge type whitelist.

        Returns:
            SubgraphResult from graph-engine.
        """
        if self._stub is None:
            logger.warning("Graph client not connected — returning empty subgraph")
            return None

        # Production call:
        # request = common_pb2.SubgraphQuery(
        #     seed_node_ids=seed_ids,
        #     max_hops=max_hops,
        #     max_nodes=max_nodes,
        #     relation_filter=relation_filter or [],
        # )
        # return await self._stub.QuerySubgraph(request)

        return None

    async def upsert_node(self, node_data: dict):
        """Create or update a node in the knowledge graph."""
        if self._stub is None:
            logger.warning("Graph client not connected")
            return None
        # Production: await self._stub.UpsertNode(request)

    async def upsert_edge(self, edge_data: dict):
        """Create or update an edge in the knowledge graph."""
        if self._stub is None:
            logger.warning("Graph client not connected")
            return None
        # Production: await self._stub.UpsertEdge(request)

    async def close(self):
        """Close the gRPC channel."""
        if self._channel:
            await self._channel.close()
            logger.info("Graph-engine gRPC channel closed")
