"""Query planner for mixed graph+text retrieval.

Plans and executes hybrid queries that combine:
- Graph traversal (multi-hop reasoning)
- Text retrieval (vector/BM25)
- Result fusion and ranking
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Types of queries supported."""
    SIMPLE_LOOKUP = "simple_lookup"           # Single entity lookup
    MULTI_HOP = "multi_hop"                   # Multi-hop graph traversal
    NEIGHBORHOOD = "neighborhood"              # Entity neighborhood
    PATH_FINDING = "path_finding"             # Find path between entities
    HYBRID = "hybrid"                          # Mix graph + text
    TEXT_ONLY = "text_only"                   # Pure text retrieval


@dataclass
class QueryPlan:
    """Represents an execution plan for a query."""
    query_type: QueryType
    steps: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    graph_queries: List[str] = field(default_factory=list)
    text_queries: List[str] = field(default_factory=list)
    fusion_strategy: str = "weighted"  # weighted, rrf, cascade
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanStep:
    """Individual step in query execution."""
    step_type: str  # graph_lookup, text_retrieval, fusion, rerank
    operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[int] = field(default_factory=list)  # Step indices this depends on


class GraphQueryPlanner:
    """Plans and executes mixed graph+text queries."""
    
    def __init__(
        self,
        graph_store,
        text_retriever=None,
        entity_extractor=None,
    ):
        """Initialize query planner.
        
        Args:
            graph_store: Neo4j graph store
            text_retriever: Text retriever (hybrid/vector/BM25)
            entity_extractor: Entity extractor for query analysis
        """
        self.graph_store = graph_store
        self.text_retriever = text_retriever
        self.entity_extractor = entity_extractor
    
    async def plan_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryPlan:
        """Generate execution plan for a query.
        
        Args:
            query: Natural language query
            context: Optional context (conversation history, user profile)
            
        Returns:
            QueryPlan with execution steps
        """
        # Extract entities from query
        entities = await self._extract_query_entities(query)
        
        # Classify query type
        query_type = self._classify_query(query, entities)
        
        # Generate plan based on type
        if query_type == QueryType.SIMPLE_LOOKUP:
            plan = self._plan_simple_lookup(query, entities)
        elif query_type == QueryType.MULTI_HOP:
            plan = self._plan_multi_hop(query, entities)
        elif query_type == QueryType.NEIGHBORHOOD:
            plan = self._plan_neighborhood(query, entities)
        elif query_type == QueryType.PATH_FINDING:
            plan = self._plan_path_finding(query, entities)
        elif query_type == QueryType.HYBRID:
            plan = self._plan_hybrid(query, entities)
        else:  # TEXT_ONLY
            plan = self._plan_text_only(query)
        
        logger.info(f"Query plan: {query_type.value} with {len(plan.steps)} steps")
        
        return plan
    
    async def execute_plan(self, plan: QueryPlan) -> Dict[str, Any]:
        """Execute a query plan.
        
        Args:
            plan: QueryPlan to execute
            
        Returns:
            Dictionary with results from all steps
        """
        results = {
            "query_type": plan.query_type.value,
            "entities": plan.entities,
            "graph_results": [],
            "text_results": [],
            "fused_results": [],
            "metadata": plan.metadata,
        }
        
        step_outputs = []
        
        try:
            for idx, step in enumerate(plan.steps):
                step_type = step.get("type")
                
                if step_type == "graph_lookup":
                    output = await self._execute_graph_step(step, step_outputs)
                    results["graph_results"].append(output)
                
                elif step_type == "text_retrieval":
                    output = await self._execute_text_step(step, step_outputs)
                    results["text_results"].append(output)
                
                elif step_type == "fusion":
                    output = await self._execute_fusion_step(step, step_outputs)
                    results["fused_results"] = output
                
                elif step_type == "rerank":
                    output = await self._execute_rerank_step(step, step_outputs)
                    results["fused_results"] = output
                
                step_outputs.append(output)
            
            logger.info(f"Plan executed: {len(step_outputs)} steps completed")
            
        except Exception as e:
            logger.error(f"Plan execution error: {e}")
            results["error"] = str(e)
        
        return results
    
    async def _extract_query_entities(self, query: str) -> List[str]:
        """Extract entities mentioned in query.
        
        Args:
            query: Query text
            
        Returns:
            List of entity names
        """
        if not self.entity_extractor:
            # Simple extraction: look for capitalized words
            words = query.split()
            entities = [w for w in words if w[0].isupper() and len(w) > 2]
            return entities
        
        try:
            result = await self.entity_extractor.extract(query)
            return [e.name for e in result.entities]
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return []
    
    def _classify_query(self, query: str, entities: List[str]) -> QueryType:
        """Classify query type based on patterns.
        
        Args:
            query: Query text
            entities: Extracted entities
            
        Returns:
            QueryType
        """
        query_lower = query.lower()
        
        # Multi-hop patterns
        multi_hop_patterns = [
            "who founded the company that",
            "what did the person who",
            "where is the company that acquired",
            "relationship between",
            "how are",
            "connected",
        ]
        if any(p in query_lower for p in multi_hop_patterns):
            return QueryType.MULTI_HOP
        
        # Path finding patterns
        path_patterns = ["path between", "connection between", "link between"]
        if any(p in query_lower for p in path_patterns):
            return QueryType.PATH_FINDING
        
        # Neighborhood patterns
        neighborhood_patterns = ["related to", "associated with", "works with", "partners of"]
        if any(p in query_lower for p in neighborhood_patterns):
            return QueryType.NEIGHBORHOOD
        
        # Simple lookup patterns
        if len(entities) == 1 and any(p in query_lower for p in ["what is", "who is", "where is"]):
            return QueryType.SIMPLE_LOOKUP
        
        # Hybrid if we have entities and text
        if entities and len(query.split()) > 5:
            return QueryType.HYBRID
        
        # Default to text only
        return QueryType.TEXT_ONLY
    
    def _plan_simple_lookup(self, query: str, entities: List[str]) -> QueryPlan:
        """Plan for simple entity lookup."""
        plan = QueryPlan(
            query_type=QueryType.SIMPLE_LOOKUP,
            entities=entities,
        )
        
        # Step 1: Graph lookup
        plan.steps.append({
            "type": "graph_lookup",
            "operation": "find_entity",
            "parameters": {"entity_name": entities[0] if entities else ""},
        })
        
        # Step 2: Text retrieval as backup
        plan.steps.append({
            "type": "text_retrieval",
            "operation": "retrieve",
            "parameters": {"query": query, "top_k": 5},
        })
        
        # Step 3: Fusion
        plan.steps.append({
            "type": "fusion",
            "operation": "cascade",  # Prefer graph, fallback to text
            "parameters": {"dependencies": [0, 1]},
        })
        
        return plan
    
    def _plan_multi_hop(self, query: str, entities: List[str]) -> QueryPlan:
        """Plan for multi-hop reasoning."""
        plan = QueryPlan(
            query_type=QueryType.MULTI_HOP,
            entities=entities,
            fusion_strategy="weighted",
        )
        
        # Step 1: Find starting entities
        plan.steps.append({
            "type": "graph_lookup",
            "operation": "find_neighbors",
            "parameters": {
                "entity_names": entities,
                "max_hops": 3,
            },
        })
        
        # Step 2: Text retrieval for context
        plan.steps.append({
            "type": "text_retrieval",
            "operation": "retrieve",
            "parameters": {"query": query, "top_k": 10},
        })
        
        # Step 3: Fusion with graph priority
        plan.steps.append({
            "type": "fusion",
            "operation": "weighted",
            "parameters": {
                "dependencies": [0, 1],
                "weights": {"graph": 0.7, "text": 0.3},
            },
        })
        
        return plan
    
    def _plan_neighborhood(self, query: str, entities: List[str]) -> QueryPlan:
        """Plan for neighborhood exploration."""
        plan = QueryPlan(
            query_type=QueryType.NEIGHBORHOOD,
            entities=entities,
        )
        
        # Step 1: Get 1-hop neighbors
        plan.steps.append({
            "type": "graph_lookup",
            "operation": "find_neighbors",
            "parameters": {
                "entity_name": entities[0] if entities else "",
                "max_hops": 1,
            },
        })
        
        # Step 2: Get subgraph
        plan.steps.append({
            "type": "graph_lookup",
            "operation": "get_subgraph",
            "parameters": {
                "entity_names": entities,
                "max_hops": 2,
            },
        })
        
        return plan
    
    def _plan_path_finding(self, query: str, entities: List[str]) -> QueryPlan:
        """Plan for path finding between entities."""
        plan = QueryPlan(
            query_type=QueryType.PATH_FINDING,
            entities=entities,
        )
        
        if len(entities) >= 2:
            # Step 1: Find shortest path
            plan.steps.append({
                "type": "graph_lookup",
                "operation": "find_path",
                "parameters": {
                    "source_name": entities[0],
                    "target_name": entities[1],
                    "max_hops": 5,
                },
            })
        
        return plan
    
    def _plan_hybrid(self, query: str, entities: List[str]) -> QueryPlan:
        """Plan for hybrid graph+text query."""
        plan = QueryPlan(
            query_type=QueryType.HYBRID,
            entities=entities,
            fusion_strategy="rrf",
        )
        
        # Step 1: Graph retrieval
        plan.steps.append({
            "type": "graph_lookup",
            "operation": "search_and_expand",
            "parameters": {
                "entity_names": entities,
                "max_hops": 2,
            },
        })
        
        # Step 2: Text retrieval
        plan.steps.append({
            "type": "text_retrieval",
            "operation": "retrieve",
            "parameters": {"query": query, "top_k": 10},
        })
        
        # Step 3: RRF fusion
        plan.steps.append({
            "type": "fusion",
            "operation": "rrf",
            "parameters": {
                "dependencies": [0, 1],
                "k": 60,
            },
        })
        
        # Step 4: Rerank
        plan.steps.append({
            "type": "rerank",
            "operation": "cross_encoder",
            "parameters": {"top_k": 5},
        })
        
        return plan
    
    def _plan_text_only(self, query: str) -> QueryPlan:
        """Plan for text-only retrieval."""
        plan = QueryPlan(
            query_type=QueryType.TEXT_ONLY,
        )
        
        # Step 1: Text retrieval
        plan.steps.append({
            "type": "text_retrieval",
            "operation": "retrieve",
            "parameters": {"query": query, "top_k": 10},
        })
        
        return plan
    
    async def _execute_graph_step(self, step: Dict[str, Any], previous_outputs: List[Any]) -> Any:
        """Execute a graph lookup step.
        
        Args:
            step: Step definition
            previous_outputs: Outputs from previous steps
            
        Returns:
            Step output
        """
        operation = step.get("operation")
        params = step.get("parameters", {})
        
        try:
            if operation == "find_entity":
                return await self.graph_store.find_entity(params["entity_name"])
            
            elif operation == "find_neighbors":
                if "entity_names" in params:
                    # Multiple entities - get subgraph
                    return await self.graph_store.get_subgraph(
                        params["entity_names"],
                        params.get("max_hops", 2),
                    )
                else:
                    return await self.graph_store.find_neighbors(
                        params["entity_name"],
                        params.get("max_hops", 1),
                    )
            
            elif operation == "get_subgraph":
                return await self.graph_store.get_subgraph(
                    params["entity_names"],
                    params.get("max_hops", 2),
                )
            
            elif operation == "find_path":
                return await self.graph_store.find_path(
                    params["source_name"],
                    params["target_name"],
                    params.get("max_hops", 3),
                )
            
            elif operation == "search_and_expand":
                # Search entities and expand neighborhood
                entities = params["entity_names"]
                max_hops = params.get("max_hops", 2)
                
                # Get subgraph for all entities
                subgraph = await self.graph_store.get_subgraph(entities, max_hops)
                return subgraph
            
            else:
                logger.warning(f"Unknown graph operation: {operation}")
                return None
                
        except Exception as e:
            logger.error(f"Graph step error: {e}")
            return None
    
    async def _execute_text_step(self, step: Dict[str, Any], previous_outputs: List[Any]) -> Any:
        """Execute a text retrieval step.
        
        Args:
            step: Step definition
            previous_outputs: Outputs from previous steps
            
        Returns:
            Step output
        """
        if not self.text_retriever:
            logger.warning("Text retriever not available")
            return []
        
        params = step.get("parameters", {})
        
        try:
            results = await self.text_retriever.retrieve(
                params["query"],
                top_k=params.get("top_k", 10),
            )
            return results
            
        except Exception as e:
            logger.error(f"Text step error: {e}")
            return []
    
    async def _execute_fusion_step(self, step: Dict[str, Any], previous_outputs: List[Any]) -> Any:
        """Execute a fusion step.
        
        Args:
            step: Step definition
            previous_outputs: Outputs from previous steps
            
        Returns:
            Fused results
        """
        operation = step.get("operation", "weighted")
        params = step.get("parameters", {})
        dependencies = params.get("dependencies", [])
        
        # Gather results from dependencies
        results = [previous_outputs[i] for i in dependencies if i < len(previous_outputs)]
        
        if operation == "cascade":
            # Prefer first non-empty result
            for result in results:
                if result:
                    return result
            return []
        
        elif operation == "weighted":
            # Weighted fusion (simplified)
            weights = params.get("weights", {"graph": 0.5, "text": 0.5})
            # TODO: Implement proper weighted fusion
            return results
        
        elif operation == "rrf":
            # RRF fusion (simplified)
            k = params.get("k", 60)
            # TODO: Implement RRF fusion
            return results
        
        return results
    
    async def _execute_rerank_step(self, step: Dict[str, Any], previous_outputs: List[Any]) -> Any:
        """Execute a reranking step.
        
        Args:
            step: Step definition
            previous_outputs: Outputs from previous steps
            
        Returns:
            Reranked results
        """
        # TODO: Integrate with reranker from Phase 2
        return previous_outputs[-1] if previous_outputs else []
