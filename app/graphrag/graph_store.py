"""Neo4j graph store for GraphRAG.

Provides integration with Neo4j for storing and querying knowledge graphs.
Supports:
- Entity and relation storage
- Cypher query execution
- Graph traversal and pattern matching
- Multi-hop reasoning
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import json

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the graph."""
    id: str
    labels: List[str]
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Represents an edge in the graph."""
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphQuery:
    """Represents a graph query."""
    cypher: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    limit: int = 100
    timeout: float = 30.0


@dataclass
class GraphResult:
    """Result of a graph query."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    paths: List[List[GraphNode]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Neo4jGraphStore:
    """Neo4j graph store client."""
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        max_connection_lifetime: int = 3600,
        max_connection_pool_size: int = 50,
    ):
        """Initialize Neo4j connection.
        
        Args:
            uri: Neo4j URI
            username: Neo4j username
            password: Neo4j password
            database: Database name
            max_connection_lifetime: Max connection lifetime in seconds
            max_connection_pool_size: Max connection pool size
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Connect to Neo4j."""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
            )
            logger.info(f"Connected to Neo4j at {self.uri}")
        except ImportError:
            logger.warning("neo4j driver not installed, graph store unavailable")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")
    
    async def execute_query(self, query: GraphQuery) -> GraphResult:
        """Execute a Cypher query.
        
        Args:
            query: GraphQuery with Cypher and parameters
            
        Returns:
            GraphResult with nodes and edges
        """
        if not self.driver:
            logger.warning("Neo4j driver not available")
            return GraphResult(nodes=[], edges=[])
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query.cypher, query.parameters)
                
                nodes = []
                edges = []
                paths = []
                
                for record in result:
                    # Extract nodes and edges from record
                    for key in record.keys():
                        value = record[key]
                        
                        if hasattr(value, 'labels'):  # Node
                            node = self._node_to_graph_node(value)
                            if node not in nodes:
                                nodes.append(node)
                        
                        elif hasattr(value, 'type'):  # Relationship
                            edge = self._relationship_to_graph_edge(value)
                            if edge not in edges:
                                edges.append(edge)
                        
                        elif isinstance(value, list):  # Path
                            path_nodes = [
                                self._node_to_graph_node(n)
                                for n in value
                                if hasattr(n, 'labels')
                            ]
                            if path_nodes:
                                paths.append(path_nodes)
                
                return GraphResult(
                    nodes=nodes,
                    edges=edges,
                    paths=paths,
                    metadata={"query": query.cypher, "record_count": len(nodes)},
                )
                
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return GraphResult(nodes=[], edges=[], metadata={"error": str(e)})
    
    def _node_to_graph_node(self, neo4j_node) -> GraphNode:
        """Convert Neo4j node to GraphNode."""
        return GraphNode(
            id=str(neo4j_node.id),
            labels=list(neo4j_node.labels),
            properties=dict(neo4j_node),
        )
    
    def _relationship_to_graph_edge(self, neo4j_rel) -> GraphEdge:
        """Convert Neo4j relationship to GraphEdge."""
        return GraphEdge(
            source_id=str(neo4j_rel.start_node.id),
            target_id=str(neo4j_rel.end_node.id),
            relationship_type=neo4j_rel.type,
            properties=dict(neo4j_rel),
        )
    
    async def add_entity(self, entity) -> bool:
        """Add entity to graph.
        
        Args:
            entity: Entity object to add
            
        Returns:
            True if successful
        """
        if not self.driver:
            return False
        
        try:
            query = GraphQuery(
                cypher="""
                MERGE (e:Entity {name: $name})
                SET e.type = $type,
                    e.confidence = $confidence,
                    e.attributes = $attributes
                RETURN e
                """,
                parameters={
                    "name": entity.name,
                    "type": entity.entity_type.value,
                    "confidence": entity.confidence,
                    "attributes": json.dumps(entity.attributes),
                },
            )
            
            result = await self.execute_query(query)
            return len(result.nodes) > 0
            
        except Exception as e:
            logger.error(f"Failed to add entity: {e}")
            return False
    
    async def add_relation(self, relation) -> bool:
        """Add relation to graph.
        
        Args:
            relation: Relation object to add
            
        Returns:
            True if successful
        """
        if not self.driver:
            return False
        
        try:
            query = GraphQuery(
                cypher="""
                MATCH (source:Entity {name: $source_name})
                MATCH (target:Entity {name: $target_name})
                MERGE (source)-[r:RELATION {type: $rel_type}]->(target)
                SET r.confidence = $confidence,
                    r.evidence = $evidence,
                    r.attributes = $attributes
                RETURN r
                """,
                parameters={
                    "source_name": relation.source,
                    "target_name": relation.target,
                    "rel_type": relation.relation_type.value,
                    "confidence": relation.confidence,
                    "evidence": relation.evidence,
                    "attributes": json.dumps(relation.attributes),
                },
            )
            
            result = await self.execute_query(query)
            return len(result.edges) > 0
            
        except Exception as e:
            logger.error(f"Failed to add relation: {e}")
            return False
    
    async def find_entity(self, name: str) -> Optional[GraphNode]:
        """Find entity by name.
        
        Args:
            name: Entity name
            
        Returns:
            GraphNode if found
        """
        query = GraphQuery(
            cypher="MATCH (e:Entity {name: $name}) RETURN e",
            parameters={"name": name},
        )
        
        result = await self.execute_query(query)
        return result.nodes[0] if result.nodes else None
    
    async def find_neighbors(
        self,
        entity_name: str,
        max_hops: int = 1,
        relationship_types: Optional[List[str]] = None,
    ) -> GraphResult:
        """Find neighbors of an entity.
        
        Args:
            entity_name: Entity name
            max_hops: Maximum hops to traverse
            relationship_types: Optional filter for relationship types
            
        Returns:
            GraphResult with neighbors
        """
        rel_filter = ""
        if relationship_types:
            rel_types = "|".join(relationship_types)
            rel_filter = f":{rel_types}"
        
        query = GraphQuery(
            cypher=f"""
            MATCH path = (start:Entity {{name: $name}})-[r{rel_filter}*1..{max_hops}]-(neighbor:Entity)
            RETURN start, neighbor, r, path
            """,
            parameters={"name": entity_name},
        )
        
        return await self.execute_query(query)
    
    async def find_path(
        self,
        source_name: str,
        target_name: str,
        max_hops: int = 3,
    ) -> GraphResult:
        """Find shortest path between two entities.
        
        Args:
            source_name: Source entity name
            target_name: Target entity name
            max_hops: Maximum path length
            
        Returns:
            GraphResult with path
        """
        query = GraphQuery(
            cypher="""
            MATCH path = shortestPath(
                (source:Entity {name: $source_name})-[*1..%d]-(target:Entity {name: $target_name})
            )
            RETURN path
            """ % max_hops,
            parameters={
                "source_name": source_name,
                "target_name": target_name,
            },
        )
        
        return await self.execute_query(query)
    
    async def search_entities(
        self,
        query_text: str,
        entity_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[GraphNode]:
        """Search entities by text.
        
        Args:
            query_text: Search query
            entity_types: Optional filter for entity types
            limit: Maximum results
            
        Returns:
            List of matching entities
        """
        type_filter = ""
        if entity_types:
            type_filter = f"AND e.type IN {entity_types}"
        
        query = GraphQuery(
            cypher=f"""
            MATCH (e:Entity)
            WHERE e.name CONTAINS $query_text {type_filter}
            RETURN e
            LIMIT {limit}
            """,
            parameters={"query_text": query_text},
        )
        
        result = await self.execute_query(query)
        return result.nodes
    
    async def get_subgraph(
        self,
        entity_names: List[str],
        max_hops: int = 2,
    ) -> GraphResult:
        """Get subgraph containing specified entities.
        
        Args:
            entity_names: List of entity names
            max_hops: Maximum hops between entities
            
        Returns:
            GraphResult with subgraph
        """
        query = GraphQuery(
            cypher="""
            MATCH (e:Entity)
            WHERE e.name IN $entity_names
            MATCH path = (e)-[*0..%d]-(neighbor:Entity)
            WHERE neighbor.name IN $entity_names
            RETURN path
            """ % max_hops,
            parameters={"entity_names": entity_names},
        )
        
        return await self.execute_query(query)
    
    async def clear_graph(self) -> bool:
        """Clear all nodes and relationships (use with caution!)."""
        if not self.driver:
            return False
        
        try:
            query = GraphQuery(cypher="MATCH (n) DETACH DELETE n")
            await self.execute_query(query)
            logger.warning("Graph cleared!")
            return True
        except Exception as e:
            logger.error(f"Failed to clear graph: {e}")
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics.
        
        Returns:
            Dictionary with node/edge counts and other stats
        """
        query = GraphQuery(
            cypher="""
            MATCH (n)
            OPTIONAL MATCH ()-[r]->()
            RETURN count(DISTINCT n) as node_count,
                   count(DISTINCT r) as edge_count
            """
        )
        
        result = await self.execute_query(query)
        
        if result.metadata.get("error"):
            return {"error": result.metadata["error"]}
        
        return {
            "nodes": result.metadata.get("node_count", 0),
            "edges": result.metadata.get("edge_count", 0),
            "database": self.database,
        }
