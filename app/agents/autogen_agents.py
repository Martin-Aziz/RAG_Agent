"""AutoGen agent system integration.

Integrates Microsoft's AutoGen framework for multi-agent conversations.
Provides agents for:
- Query planning and decomposition
- Quality assessment and critique
- Result synthesis
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .base import Agent, AgentRole, AgentMessage, AgentConfig, AgentSystem

logger = logging.getLogger(__name__)


class AutoGenAgent(Agent):
    """AutoGen-based agent."""
    
    def __init__(self, config: AgentConfig, model_adapter=None):
        """Initialize AutoGen agent.
        
        Args:
            config: AgentConfig
            model_adapter: LLM adapter
        """
        super().__init__(config)
        self.model = model_adapter
        self.autogen_agent = None
        
        self._initialize_autogen()
    
    def _initialize_autogen(self):
        """Initialize AutoGen agent (if available)."""
        try:
            # Try to import autogen
            import autogen
            
            # Create AutoGen config
            llm_config = {
                "model": self.config.model_name,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            
            # Create AutoGen assistant agent
            self.autogen_agent = autogen.AssistantAgent(
                name=self.name,
                system_message=self.config.system_prompt,
                llm_config=llm_config,
            )
            
            logger.info(f"Initialized AutoGen agent: {self.name}")
            
        except ImportError:
            logger.warning("AutoGen not installed, using fallback implementation")
            self.autogen_agent = None
        except Exception as e:
            logger.error(f"Failed to initialize AutoGen agent: {e}")
            self.autogen_agent = None
    
    async def process(self, message: AgentMessage) -> AgentMessage:
        """Process message using AutoGen.
        
        Args:
            message: Input message
            
        Returns:
            Response message
        """
        self.add_message(message)
        
        if self.autogen_agent and self.model:
            try:
                # Use AutoGen for processing
                response = await self._process_with_autogen(message)
            except Exception as e:
                logger.error(f"AutoGen processing error: {e}")
                response = await self._process_fallback(message)
        else:
            # Fallback to simple processing
            response = await self._process_fallback(message)
        
        response_msg = AgentMessage(
            sender=self.name,
            recipient=message.sender,
            content=response,
        )
        
        self.add_message(response_msg)
        return response_msg
    
    async def _process_with_autogen(self, message: AgentMessage) -> str:
        """Process using AutoGen agent.
        
        Args:
            message: Input message
            
        Returns:
            Response string
        """
        # Note: This is a simplified interface
        # Real AutoGen requires conversation setup
        prompt = f"{self.config.system_prompt}\n\nUser: {message.content}\n\nAssistant:"
        response = await self.model.generate(prompt)
        return response
    
    async def _process_fallback(self, message: AgentMessage) -> str:
        """Fallback processing without AutoGen.
        
        Args:
            message: Input message
            
        Returns:
            Response string
        """
        if not self.model:
            return f"[{self.role.value}] Received: {message.content}"
        
        prompt = f"{self.config.system_prompt}\n\nTask: {message.content}\n\nResponse:"
        response = await self.model.generate(prompt, temperature=self.config.temperature)
        return response


class AutoGenAgentSystem(AgentSystem):
    """Multi-agent system using AutoGen."""
    
    def __init__(self, model_adapter=None, enable_autogen: bool = True):
        """Initialize AutoGen system.
        
        Args:
            model_adapter: LLM adapter for agents
            enable_autogen: Enable AutoGen framework
        """
        super().__init__(name="autogen_system")
        self.model = model_adapter
        self.enable_autogen = enable_autogen
        
        # Initialize default agents
        self._create_default_agents()
    
    def _create_default_agents(self):
        """Create default agent roles."""
        
        # Planner agent
        planner_config = AgentConfig(
            name="planner",
            role=AgentRole.PLANNER,
            system_prompt="""You are a query planning agent. Your job is to:
1. Analyze complex queries
2. Break them down into subtasks
3. Determine the execution order
4. Identify required information

Output a structured plan with numbered steps.""",
            temperature=0.3,
        )
        self.register_agent(AutoGenAgent(planner_config, self.model))
        
        # Judge agent
        judge_config = AgentConfig(
            name="judge",
            role=AgentRole.JUDGE,
            system_prompt="""You are a quality assessment agent. Your job is to:
1. Evaluate answer quality
2. Check for factual accuracy
3. Assess completeness
4. Verify citations

Output a quality score (0-10) and feedback.""",
            temperature=0.2,
        )
        self.register_agent(AutoGenAgent(judge_config, self.model))
        
        # Finalizer agent
        finalizer_config = AgentConfig(
            name="finalizer",
            role=AgentRole.FINALIZER,
            system_prompt="""You are a result synthesis agent. Your job is to:
1. Combine results from multiple agents
2. Resolve conflicts
3. Format the final answer
4. Ensure clarity and coherence

Output a polished final answer.""",
            temperature=0.5,
        )
        self.register_agent(AutoGenAgent(finalizer_config, self.model))
    
    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run multi-agent workflow.
        
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
        
        # Step 1: Planning
        planner = self.get_agent("planner")
        if planner:
            plan_msg = AgentMessage(
                sender="system",
                recipient="planner",
                content=f"Create a plan for: {query}",
            )
            plan_response = await planner.process(plan_msg)
            result["agent_outputs"]["plan"] = plan_response.content
        
        # Step 2: Execute main task (QA)
        # (In real implementation, this would follow the plan)
        qa_output = context.get("answer", query)
        result["agent_outputs"]["qa"] = qa_output
        
        # Step 3: Quality assessment
        judge = self.get_agent("judge")
        if judge:
            judge_msg = AgentMessage(
                sender="system",
                recipient="judge",
                content=f"Assess this answer:\nQuery: {query}\nAnswer: {qa_output}",
            )
            judge_response = await judge.process(judge_msg)
            result["agent_outputs"]["assessment"] = judge_response.content
        
        # Step 4: Finalization
        finalizer = self.get_agent("finalizer")
        if finalizer:
            final_msg = AgentMessage(
                sender="system",
                recipient="finalizer",
                content=f"Synthesize final answer from:\nPlan: {result['agent_outputs'].get('plan', '')}\nQA: {qa_output}\nAssessment: {result['agent_outputs'].get('assessment', '')}",
            )
            final_response = await finalizer.process(final_msg)
            result["final_answer"] = final_response.content
        else:
            result["final_answer"] = qa_output
        
        return result
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get full conversation history from all agents.
        
        Returns:
            List of message dictionaries
        """
        all_messages = []
        
        for agent in self.agents.values():
            for msg in agent.get_history():
                all_messages.append(msg.to_dict())
        
        # Sort by timestamp
        all_messages.sort(key=lambda m: m["timestamp"])
        
        return all_messages
