"""Agent orchestrator and registry.

Manages agent lifecycle and provides unified interface.
"""

import asyncio
import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import Agent, AgentSystem, AgentRole
from .autogen_agents import AutoGenAgentSystem
from .crewai_agents import CrewAIAgentSystem
from .custom_agents import CustomAgentSystem

logger = logging.getLogger(__name__)


class AgentFramework(str, Enum):
    """Supported agent frameworks."""
    AUTOGEN = "autogen"
    CREWAI = "crewai"
    CUSTOM = "custom"


class AgentRegistry:
    """Registry for agent systems."""
    
    def __init__(self):
        """Initialize agent registry."""
        self.systems: Dict[str, AgentSystem] = {}
        self.default_framework = AgentFramework.CUSTOM
    
    def register(self, name: str, system: AgentSystem):
        """Register an agent system.
        
        Args:
            name: System name
            system: AgentSystem instance
        """
        self.systems[name] = system
        logger.info(f"Registered agent system: {name}")
    
    def get(self, name: str) -> Optional[AgentSystem]:
        """Get agent system by name.
        
        Args:
            name: System name
            
        Returns:
            AgentSystem or None
        """
        return self.systems.get(name)
    
    def list_systems(self) -> List[str]:
        """List registered systems.
        
        Returns:
            List of system names
        """
        return list(self.systems.keys())


class AgentOrchestrator:
    """Orchestrates multi-agent workflows."""
    
    def __init__(
        self,
        model_adapter=None,
        framework: AgentFramework = AgentFramework.CUSTOM,
        enable_agents: bool = True,
    ):
        """Initialize agent orchestrator.
        
        Args:
            model_adapter: LLM adapter for agents
            framework: Agent framework to use
            enable_agents: Enable multi-agent mode
        """
        self.model = model_adapter
        self.framework = framework
        self.enable_agents = enable_agents
        
        # Initialize registry
        self.registry = AgentRegistry()
        
        # Create systems
        if enable_agents:
            self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize agent systems."""
        
        # Custom system (always available)
        custom_system = CustomAgentSystem(self.model)
        self.registry.register("custom", custom_system)
        
        # AutoGen system
        try:
            autogen_system = AutoGenAgentSystem(self.model)
            self.registry.register("autogen", autogen_system)
        except Exception as e:
            logger.warning(f"Could not initialize AutoGen system: {e}")
        
        # CrewAI system
        try:
            crewai_system = CrewAIAgentSystem(self.model)
            self.registry.register("crewai", crewai_system)
        except Exception as e:
            logger.warning(f"Could not initialize CrewAI system: {e}")
        
        logger.info(f"Initialized {len(self.registry.list_systems())} agent systems")
    
    async def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        framework: Optional[AgentFramework] = None,
    ) -> Dict[str, Any]:
        """Run multi-agent workflow.
        
        Args:
            query: Query string
            context: Optional context (documents, etc.)
            framework: Optional framework override
            
        Returns:
            Result dictionary
        """
        if not self.enable_agents:
            return {
                "query": query,
                "final_answer": query,
                "metadata": {"agents_disabled": True},
            }
        
        # Select framework
        fw = framework or self.framework
        system = self.registry.get(fw.value)
        
        if not system:
            logger.warning(f"Framework {fw.value} not available, using custom")
            system = self.registry.get("custom")
        
        if not system:
            logger.error("No agent systems available")
            return {
                "query": query,
                "final_answer": query,
                "error": "No agent systems available",
            }
        
        # Prepare task
        task = {
            "query": query,
            "context": context or {},
        }
        
        # Run system
        try:
            result = await system.run(task)
            result["framework"] = fw.value
            return result
        except Exception as e:
            logger.error(f"Agent system error: {e}")
            return {
                "query": query,
                "final_answer": query,
                "error": str(e),
            }
    
    async def run_with_planning(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run with explicit planning step.
        
        Args:
            query: Query string
            context: Optional context
            
        Returns:
            Result with plan
        """
        system = self.registry.get(self.framework.value) or self.registry.get("custom")
        
        if not system:
            return {"error": "No system available"}
        
        # Get planner agent
        planner = system.get_agent("planner")
        if not planner:
            logger.warning("No planner agent available")
            return await self.run(query, context)
        
        # Create plan
        from .base import AgentMessage
        plan_msg = AgentMessage(
            sender="orchestrator",
            recipient="planner",
            content=f"Create detailed execution plan for: {query}",
        )
        
        plan_response = await planner.process(plan_msg)
        
        # Execute based on plan
        result = await self.run(query, context)
        result["plan"] = plan_response.content
        
        return result
    
    async def run_with_critique(
        self,
        query: str,
        answer: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run with critique/refinement loop.
        
        Args:
            query: Original query
            answer: Answer to critique
            context: Optional context
            
        Returns:
            Result with critique and refined answer
        """
        system = self.registry.get(self.framework.value) or self.registry.get("custom")
        
        if not system:
            return {"error": "No system available"}
        
        # Get judge/critic agent
        judge = system.get_agent("judge") or system.get_agent("critic")
        if not judge:
            logger.warning("No judge/critic agent available")
            return {
                "query": query,
                "answer": answer,
                "critique": "No critique available",
            }
        
        # Critique answer
        from .base import AgentMessage
        critique_msg = AgentMessage(
            sender="orchestrator",
            recipient=judge.name,
            content=f"Critique this answer:\nQuery: {query}\nAnswer: {answer}",
        )
        
        critique_response = await judge.process(critique_msg)
        
        return {
            "query": query,
            "original_answer": answer,
            "critique": critique_response.content,
            "metadata": context or {},
        }
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics from all agent systems.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "enabled": self.enable_agents,
            "framework": self.framework.value,
            "systems": {},
        }
        
        for name, system in self.registry.systems.items():
            stats["systems"][name] = {
                "agents": len(system.agents),
                "agent_roles": [a.role.value for a in system.agents.values()],
            }
        
        return stats
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], model_adapter=None) -> "AgentOrchestrator":
        """Create orchestrator from configuration.
        
        Args:
            config: Configuration dictionary
            model_adapter: LLM adapter
            
        Returns:
            AgentOrchestrator instance
        """
        framework_str = config.get("framework", "custom")
        framework = AgentFramework(framework_str)
        
        return cls(
            model_adapter=model_adapter,
            framework=framework,
            enable_agents=config.get("enable_agents", True),
        )
