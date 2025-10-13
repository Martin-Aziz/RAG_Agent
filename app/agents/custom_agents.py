"""Custom lightweight agent system.

Simple agent orchestration without external dependencies.
Useful when AutoGen/CrewAI are not available.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .base import Agent, AgentRole, AgentMessage, AgentConfig, AgentSystem

logger = logging.getLogger(__name__)


class CustomAgent(Agent):
    """Simple custom agent."""
    
    def __init__(self, config: AgentConfig, model_adapter=None):
        """Initialize custom agent.
        
        Args:
            config: AgentConfig
            model_adapter: LLM adapter
        """
        super().__init__(config)
        self.model = model_adapter
    
    async def process(self, message: AgentMessage) -> AgentMessage:
        """Process message.
        
        Args:
            message: Input message
            
        Returns:
            Response message
        """
        self.add_message(message)
        
        if not self.model:
            response = f"[{self.role.value}] Processed: {message.content[:100]}..."
        else:
            # Build prompt with role-specific instructions
            prompt = self._build_prompt(message)
            response = await self.model.generate(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        
        response_msg = AgentMessage(
            sender=self.name,
            recipient=message.sender,
            content=response,
        )
        
        self.add_message(response_msg)
        return response_msg
    
    def _build_prompt(self, message: AgentMessage) -> str:
        """Build prompt for agent.
        
        Args:
            message: Input message
            
        Returns:
            Formatted prompt
        """
        return f"""{self.config.system_prompt}

Task: {message.content}

Response:"""


class CustomAgentSystem(AgentSystem):
    """Simple custom multi-agent system."""
    
    def __init__(self, model_adapter=None):
        """Initialize custom agent system.
        
        Args:
            model_adapter: LLM adapter for agents
        """
        super().__init__(name="custom_system")
        self.model = model_adapter
        
        # Initialize default agents
        self._create_default_agents()
    
    def _create_default_agents(self):
        """Create default agent roles."""
        
        # Planner
        planner_config = AgentConfig(
            name="planner",
            role=AgentRole.PLANNER,
            system_prompt="You are a planning agent. Break down queries into steps.",
            temperature=0.3,
        )
        self.register_agent(CustomAgent(planner_config, self.model))
        
        # Extractor
        extractor_config = AgentConfig(
            name="extractor",
            role=AgentRole.EXTRACTOR,
            system_prompt="You are an extraction agent. Extract entities and relations.",
            temperature=0.2,
        )
        self.register_agent(CustomAgent(extractor_config, self.model))
        
        # QA
        qa_config = AgentConfig(
            name="qa",
            role=AgentRole.QA,
            system_prompt="You are a QA agent. Answer questions using provided context.",
            temperature=0.5,
        )
        self.register_agent(CustomAgent(qa_config, self.model))
        
        # Judge
        judge_config = AgentConfig(
            name="judge",
            role=AgentRole.JUDGE,
            system_prompt="You are a judge agent. Assess answer quality.",
            temperature=0.2,
        )
        self.register_agent(CustomAgent(judge_config, self.model))
    
    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run custom agent workflow.
        
        Args:
            task: Task with query and context
            
        Returns:
            Result with agent outputs
        """
        query = task.get("query", "")
        context = task.get("context", {})
        
        result = {
            "query": query,
            "agent_outputs": {},
            "final_answer": "",
            "metadata": {},
        }
        
        # Sequential processing through agents
        
        # 1. Planning (optional)
        if "planner" in self.agents:
            planner = self.agents["planner"]
            plan_msg = AgentMessage(
                sender="system",
                recipient="planner",
                content=f"Create execution plan for: {query}",
            )
            plan_response = await planner.process(plan_msg)
            result["agent_outputs"]["plan"] = plan_response.content
        
        # 2. Extraction (if documents provided)
        documents = context.get("documents", [])
        if documents and "extractor" in self.agents:
            extractor = self.agents["extractor"]
            extract_msg = AgentMessage(
                sender="system",
                recipient="extractor",
                content=f"Extract entities from documents about: {query}",
            )
            extract_response = await extractor.process(extract_msg)
            result["agent_outputs"]["extraction"] = extract_response.content
        
        # 3. QA
        if "qa" in self.agents:
            qa = self.agents["qa"]
            qa_msg = AgentMessage(
                sender="system",
                recipient="qa",
                content=f"Answer: {query}\n\nContext: {self._format_context(context)}",
            )
            qa_response = await qa.process(qa_msg)
            result["agent_outputs"]["answer"] = qa_response.content
            result["final_answer"] = qa_response.content
        
        # 4. Quality check
        if "judge" in self.agents and result.get("final_answer"):
            judge = self.agents["judge"]
            judge_msg = AgentMessage(
                sender="system",
                recipient="judge",
                content=f"Assess:\nQuery: {query}\nAnswer: {result['final_answer']}",
            )
            judge_response = await judge.process(judge_msg)
            result["agent_outputs"]["assessment"] = judge_response.content
        
        return result
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for agents.
        
        Args:
            context: Context dictionary
            
        Returns:
            Formatted string
        """
        parts = []
        
        if "documents" in context:
            docs = context["documents"][:3]
            doc_strs = [f"[Doc {i+1}] {d.get('text', '')[:200]}..." for i, d in enumerate(docs)]
            parts.append("Documents:\n" + "\n".join(doc_strs))
        
        if "graph_results" in context:
            parts.append("Graph results available")
        
        return "\n\n".join(parts) if parts else "No context"
