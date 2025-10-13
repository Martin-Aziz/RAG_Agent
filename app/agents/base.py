"""Base agent classes and interfaces.

Defines common agent abstractions used across frameworks.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Agent roles in the system."""
    PLANNER = "planner"  # Query decomposition and planning
    EXTRACTOR = "extractor"  # Entity/relation extraction
    QA = "qa"  # Question answering
    JUDGE = "judge"  # Quality assessment
    FINALIZER = "finalizer"  # Result synthesis
    RESEARCHER = "researcher"  # Information gathering
    CRITIC = "critic"  # Critical evaluation


@dataclass
class AgentMessage:
    """Message between agents."""
    sender: str
    recipient: str
    content: str
    message_type: str = "text"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "message_type": self.message_type,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    role: AgentRole
    system_prompt: str
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    tools: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Agent:
    """Base agent interface."""
    
    def __init__(self, config: AgentConfig):
        """Initialize agent.
        
        Args:
            config: AgentConfig
        """
        self.config = config
        self.name = config.name
        self.role = config.role
        self.message_history: List[AgentMessage] = []
    
    async def process(self, message: AgentMessage) -> AgentMessage:
        """Process a message and return response.
        
        Args:
            message: Input message
            
        Returns:
            Response message
        """
        raise NotImplementedError("Subclasses must implement process()")
    
    def add_message(self, message: AgentMessage):
        """Add message to history.
        
        Args:
            message: Message to add
        """
        self.message_history.append(message)
    
    def get_history(self) -> List[AgentMessage]:
        """Get message history.
        
        Returns:
            List of messages
        """
        return self.message_history
    
    def clear_history(self):
        """Clear message history."""
        self.message_history = []


class AgentSystem:
    """Base class for multi-agent systems."""
    
    def __init__(self, name: str = "agent_system"):
        """Initialize agent system.
        
        Args:
            name: System name
        """
        self.name = name
        self.agents: Dict[str, Agent] = {}
    
    def register_agent(self, agent: Agent):
        """Register an agent.
        
        Args:
            agent: Agent to register
        """
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name} ({agent.role.value})")
    
    def get_agent(self, name: str) -> Optional[Agent]:
        """Get agent by name.
        
        Args:
            name: Agent name
            
        Returns:
            Agent or None
        """
        return self.agents.get(name)
    
    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent system on a task.
        
        Args:
            task: Task specification
            
        Returns:
            Result dictionary
        """
        raise NotImplementedError("Subclasses must implement run()")
