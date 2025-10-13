"""Graph traversal utilities for multi-hop reasoning.

Provides algorithms for:
- Breadth-first and depth-first traversal
- Path finding and ranking
- Subgraph extraction
- Relevance scoring for graph paths
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class TraversalStrategy(str, Enum):
    """Graph traversal strategies."""
    BFS = "bfs"  # Breadth-first search
    DFS = "dfs"  # Depth-first search
    BIDIRECTIONAL = "bidirectional"  # Bidirectional search
    BEST_FIRST = "best_first"  # Heuristic-guided search


@dataclass
class TraversalPath:
    """Represents a path through the graph."""
    nodes: List[str]  # Node IDs in order
    edges: List[Tuple[str, str, str]]  # (source, target, rel_type)
    length: int
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """String representation of path."""
        path_str = " -> ".join(self.nodes)
        return f"Path(length={self.length}, score={self.score:.3f}): {path_str}"


@dataclass
class TraversalResult:
    """Result of graph traversal."""
    paths: List[TraversalPath]
    visited_nodes: Set[str]
    strategy: TraversalStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphTraversal:
    """Graph traversal algorithms for multi-hop reasoning."""
    
    def __init__(self, graph_store):
        """Initialize graph traversal.
        
        Args:
            graph_store: Neo4j graph store
        """
        self.graph_store = graph_store
    
    async def find_all_paths(
        self,
        source: str,
        target: str,
        max_length: int = 3,
        strategy: TraversalStrategy = TraversalStrategy.BFS,
    ) -> TraversalResult:
        """Find all paths between source and target.
        
        Args:
            source: Source entity name
            target: Target entity name
            max_length: Maximum path length
            strategy: Traversal strategy
            
        Returns:
            TraversalResult with all paths
        """
        if strategy == TraversalStrategy.BFS:
            return await self._bfs_all_paths(source, target, max_length)
        elif strategy == TraversalStrategy.DFS:
            return await self._dfs_all_paths(source, target, max_length)
        elif strategy == TraversalStrategy.BIDIRECTIONAL:
            return await self._bidirectional_search(source, target, max_length)
        else:
            logger.warning(f"Strategy {strategy} not implemented, using BFS")
            return await self._bfs_all_paths(source, target, max_length)
    
    async def _bfs_all_paths(
        self,
        source: str,
        target: str,
        max_length: int,
    ) -> TraversalResult:
        """BFS to find all paths within max_length.
        
        Args:
            source: Source entity name
            target: Target entity name
            max_length: Maximum path length
            
        Returns:
            TraversalResult with paths
        """
        paths = []
        visited = set()
        
        # Queue: (current_node, path_nodes, path_edges)
        queue = deque([(source, [source], [])])
        visited.add(source)
        
        try:
            while queue:
                current, path_nodes, path_edges = queue.popleft()
                
                # Check if we reached target
                if current == target and len(path_nodes) > 1:
                    path = TraversalPath(
                        nodes=path_nodes,
                        edges=path_edges,
                        length=len(path_nodes) - 1,
                    )
                    paths.append(path)
                    continue
                
                # Check max length
                if len(path_nodes) >= max_length + 1:
                    continue
                
                # Get neighbors
                neighbors_result = await self.graph_store.find_neighbors(
                    current,
                    max_hops=1,
                )
                
                for node in neighbors_result.nodes:
                    if node.properties["name"] not in visited:
                        neighbor_name = node.properties["name"]
                        new_path_nodes = path_nodes + [neighbor_name]
                        
                        # Find edge between current and neighbor
                        edge_type = self._find_edge_type(
                            current,
                            neighbor_name,
                            neighbors_result.edges,
                        )
                        new_path_edges = path_edges + [(current, neighbor_name, edge_type)]
                        
                        queue.append((neighbor_name, new_path_nodes, new_path_edges))
                        visited.add(neighbor_name)
            
            logger.info(f"BFS found {len(paths)} paths from {source} to {target}")
            
            return TraversalResult(
                paths=paths,
                visited_nodes=visited,
                strategy=TraversalStrategy.BFS,
                metadata={"max_length": max_length},
            )
            
        except Exception as e:
            logger.error(f"BFS error: {e}")
            return TraversalResult(
                paths=[],
                visited_nodes=visited,
                strategy=TraversalStrategy.BFS,
                metadata={"error": str(e)},
            )
    
    async def _dfs_all_paths(
        self,
        source: str,
        target: str,
        max_length: int,
    ) -> TraversalResult:
        """DFS to find all paths within max_length.
        
        Args:
            source: Source entity name
            target: Target entity name
            max_length: Maximum path length
            
        Returns:
            TraversalResult with paths
        """
        paths = []
        visited = set()
        
        async def dfs(current: str, path_nodes: List[str], path_edges: List[Tuple]):
            """Recursive DFS helper."""
            if current == target and len(path_nodes) > 1:
                path = TraversalPath(
                    nodes=path_nodes,
                    edges=path_edges,
                    length=len(path_nodes) - 1,
                )
                paths.append(path)
                return
            
            if len(path_nodes) >= max_length + 1:
                return
            
            visited.add(current)
            
            # Get neighbors
            neighbors_result = await self.graph_store.find_neighbors(
                current,
                max_hops=1,
            )
            
            for node in neighbors_result.nodes:
                neighbor_name = node.properties["name"]
                if neighbor_name not in visited:
                    edge_type = self._find_edge_type(
                        current,
                        neighbor_name,
                        neighbors_result.edges,
                    )
                    
                    await dfs(
                        neighbor_name,
                        path_nodes + [neighbor_name],
                        path_edges + [(current, neighbor_name, edge_type)],
                    )
            
            visited.remove(current)
        
        try:
            await dfs(source, [source], [])
            
            logger.info(f"DFS found {len(paths)} paths from {source} to {target}")
            
            return TraversalResult(
                paths=paths,
                visited_nodes=visited,
                strategy=TraversalStrategy.DFS,
                metadata={"max_length": max_length},
            )
            
        except Exception as e:
            logger.error(f"DFS error: {e}")
            return TraversalResult(
                paths=[],
                visited_nodes=set(),
                strategy=TraversalStrategy.DFS,
                metadata={"error": str(e)},
            )
    
    async def _bidirectional_search(
        self,
        source: str,
        target: str,
        max_length: int,
    ) -> TraversalResult:
        """Bidirectional search (search from both ends).
        
        Args:
            source: Source entity name
            target: Target entity name
            max_length: Maximum path length
            
        Returns:
            TraversalResult with paths
        """
        # TODO: Implement bidirectional search
        # For now, fallback to BFS
        return await self._bfs_all_paths(source, target, max_length)
    
    def _find_edge_type(
        self,
        source: str,
        target: str,
        edges: List,
    ) -> str:
        """Find edge type between source and target.
        
        Args:
            source: Source node name
            target: Target node name
            edges: List of edges
            
        Returns:
            Edge type string
        """
        for edge in edges:
            if (edge.properties.get("source") == source and 
                edge.properties.get("target") == target):
                return edge.relationship_type
        return "RELATED_TO"
    
    async def rank_paths(
        self,
        paths: List[TraversalPath],
        query: str,
        scoring_method: str = "length_and_relevance",
    ) -> List[TraversalPath]:
        """Rank paths by relevance to query.
        
        Args:
            paths: List of paths to rank
            query: Query string
            scoring_method: Scoring method to use
            
        Returns:
            Sorted list of paths with scores
        """
        if scoring_method == "length":
            # Prefer shorter paths
            for path in paths:
                path.score = 1.0 / (path.length + 1)
        
        elif scoring_method == "length_and_relevance":
            # Combine path length and node relevance
            for path in paths:
                # Base score inversely proportional to length
                length_score = 1.0 / (path.length + 1)
                
                # Node relevance (check if query terms in node names)
                query_terms = set(query.lower().split())
                node_names = set(n.lower() for n in path.nodes)
                overlap = len(query_terms & node_names)
                relevance_score = overlap / len(query_terms) if query_terms else 0
                
                path.score = 0.6 * length_score + 0.4 * relevance_score
        
        # Sort by score descending
        ranked_paths = sorted(paths, key=lambda p: p.score, reverse=True)
        
        return ranked_paths
    
    async def extract_subgraph(
        self,
        entity_names: List[str],
        max_hops: int = 2,
        min_relevance: float = 0.3,
    ) -> Dict[str, Any]:
        """Extract relevant subgraph around entities.
        
        Args:
            entity_names: List of entity names to center on
            max_hops: Maximum distance from entities
            min_relevance: Minimum relevance score
            
        Returns:
            Dictionary with nodes, edges, and metadata
        """
        try:
            # Get subgraph from graph store
            result = await self.graph_store.get_subgraph(
                entity_names,
                max_hops,
            )
            
            # Score nodes by relevance
            scored_nodes = []
            for node in result.nodes:
                # Simple relevance: distance from query entities
                if node.properties["name"] in entity_names:
                    relevance = 1.0
                else:
                    relevance = 0.5  # Default for neighbors
                
                if relevance >= min_relevance:
                    scored_nodes.append({
                        "node": node,
                        "relevance": relevance,
                    })
            
            return {
                "nodes": [sn["node"] for sn in scored_nodes],
                "edges": result.edges,
                "relevance_scores": {
                    sn["node"].properties["name"]: sn["relevance"]
                    for sn in scored_nodes
                },
                "metadata": {
                    "query_entities": entity_names,
                    "max_hops": max_hops,
                    "node_count": len(scored_nodes),
                    "edge_count": len(result.edges),
                },
            }
            
        except Exception as e:
            logger.error(f"Subgraph extraction error: {e}")
            return {
                "nodes": [],
                "edges": [],
                "relevance_scores": {},
                "metadata": {"error": str(e)},
            }
    
    async def find_connecting_entities(
        self,
        entity1: str,
        entity2: str,
        max_intermediate: int = 3,
    ) -> List[str]:
        """Find entities that connect two entities.
        
        Args:
            entity1: First entity name
            entity2: Second entity name
            max_intermediate: Maximum number of intermediate entities
            
        Returns:
            List of connecting entity names
        """
        try:
            # Find paths between entities
            result = await self.find_all_paths(
                entity1,
                entity2,
                max_length=max_intermediate + 2,
            )
            
            # Extract unique intermediate entities
            intermediate_entities = set()
            for path in result.paths:
                # Skip first and last (those are entity1 and entity2)
                for node in path.nodes[1:-1]:
                    intermediate_entities.add(node)
            
            return list(intermediate_entities)
            
        except Exception as e:
            logger.error(f"Connecting entities error: {e}")
            return []
    
    async def compute_centrality(
        self,
        entity_names: List[str],
        max_hops: int = 2,
    ) -> Dict[str, float]:
        """Compute centrality scores for entities in subgraph.
        
        Args:
            entity_names: List of entity names
            max_hops: Maximum hops for subgraph
            
        Returns:
            Dictionary mapping entity names to centrality scores
        """
        try:
            # Get subgraph
            subgraph = await self.extract_subgraph(entity_names, max_hops)
            
            # Compute degree centrality (simple version)
            degree_map = {}
            for edge in subgraph["edges"]:
                source = edge.properties.get("source", "")
                target = edge.properties.get("target", "")
                
                degree_map[source] = degree_map.get(source, 0) + 1
                degree_map[target] = degree_map.get(target, 0) + 1
            
            # Normalize by max degree
            max_degree = max(degree_map.values()) if degree_map else 1
            centrality = {
                entity: degree / max_degree
                for entity, degree in degree_map.items()
            }
            
            return centrality
            
        except Exception as e:
            logger.error(f"Centrality computation error: {e}")
            return {}
