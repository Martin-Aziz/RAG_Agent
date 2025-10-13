"""Multi-agent system for specialized task handling.

Supports multiple agent frameworks:
- AutoGen: Microsoft's multi-agent conversation framework
- CrewAI: Specialized role-based agents
- Custom: Simple agent orchestration

Agent roles:
- Planner: Query decomposition
- Extractor: Entity/relation extraction
- QA: Question answering
- Judge: Quality assessment
- Finalizer: Result synthesis
"""

from .base import Agent, AgentRole, AgentMessage, AgentConfig
from .autogen_agents import AutoGenAgentSystem
from .crewai_agents import CrewAIAgentSystem
from .custom_agents import CustomAgentSystem
from .orchestrator import AgentOrchestrator, AgentRegistry

__all__ = [
    "Agent",
    "AgentRole",
    "AgentMessage",
    "AgentConfig",
    "AutoGenAgentSystem",
    "CrewAIAgentSystem",
    "CustomAgentSystem",
    "AgentOrchestrator",
    "AgentRegistry",
]
