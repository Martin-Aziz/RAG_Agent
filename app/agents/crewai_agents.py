"""CrewAI agent system integration.

Integrates CrewAI framework for role-based agent collaboration.
Provides specialized agents for RAG tasks.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

try:  # Optional dependency
    from crewai import Agent as CrewAgent, Crew, Task  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    CrewAgent = None  # type: ignore
    Crew = None  # type: ignore
    Task = None  # type: ignore

from .base import Agent, AgentRole, AgentMessage, AgentConfig, AgentSystem

logger = logging.getLogger(__name__)


class CrewAIAgent(Agent):
    """CrewAI-based agent."""
    
    def __init__(self, config: AgentConfig, model_adapter=None):
        """Initialize CrewAI agent.
        
        Args:
            config: AgentConfig
            model_adapter: LLM adapter
        """
        super().__init__(config)
        self.model = model_adapter
        self.crew_agent = None
        
        self._initialize_crewai()
    
    def _initialize_crewai(self):
        """Initialize CrewAI agent (if available)."""
        if CrewAgent is None:
            logger.warning("CrewAI not installed, using fallback implementation")
            self.crew_agent = None
            return

        try:
            # Create CrewAI agent
            self.crew_agent = CrewAgent(
                role=self.config.role.value,
                goal=self._get_goal_for_role(),
                backstory=self.config.system_prompt,
                verbose=True,
                allow_delegation=False,
            )
            logger.info(f"Initialized CrewAI agent: {self.name}")
        except Exception as e:
            logger.error(f"Failed to initialize CrewAI agent: {e}")
            self.crew_agent = None
    
    def _get_goal_for_role(self) -> str:
        """Get goal description for agent role.
        
        Returns:
            Goal string
        """
        goals = {
            AgentRole.PLANNER: "Break down complex queries into actionable steps",
            AgentRole.EXTRACTOR: "Extract entities and relations from text",
            AgentRole.QA: "Answer questions accurately using provided context",
            AgentRole.JUDGE: "Assess answer quality and accuracy",
            AgentRole.FINALIZER: "Synthesize information into coherent answers",
            AgentRole.RESEARCHER: "Gather relevant information from sources",
            AgentRole.CRITIC: "Provide constructive feedback on outputs",
        }
        return goals.get(self.config.role, "Complete assigned task")
    
    async def process(self, message: AgentMessage) -> AgentMessage:
        """Process message using CrewAI.
        
        Args:
            message: Input message
            
        Returns:
            Response message
        """
        self.add_message(message)
        
        # Use fallback processing (CrewAI is task-based, not message-based)
        response = await self._process_fallback(message)
        
        response_msg = AgentMessage(
            sender=self.name,
            recipient=message.sender,
            content=response,
        )
        
        self.add_message(response_msg)
        return response_msg
    
    async def _process_fallback(self, message: AgentMessage) -> str:
        """Fallback processing.
        
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


class CrewAIAgentSystem(AgentSystem):
    """Multi-agent system using CrewAI."""
    
    def __init__(self, model_adapter=None, enable_crewai: bool = True):
        """Initialize CrewAI system.
        
        Args:
            model_adapter: LLM adapter for agents
            enable_crewai: Enable CrewAI framework
        """
        super().__init__(name="crewai_system")
        self.model = model_adapter
        self.enable_crewai = enable_crewai
        self.crew = None
        
        # Initialize default agents
        self._create_default_agents()
    
    def _create_default_agents(self):
        """Create default agent roles."""
        
        # Researcher agent
        researcher_config = AgentConfig(
            name="researcher",
            role=AgentRole.RESEARCHER,
            system_prompt="""You are a research specialist. Your expertise is in:
- Gathering relevant information from documents
- Identifying key facts and evidence
- Organizing research findings
- Noting gaps in available information

Provide thorough research summaries.""",
            temperature=0.3,
        )
        self.register_agent(CrewAIAgent(researcher_config, self.model))
        
        # QA agent
        qa_config = AgentConfig(
            name="qa_specialist",
            role=AgentRole.QA,
            system_prompt="""You are a question answering specialist. Your expertise is in:
- Understanding complex questions
- Synthesizing information from research
- Providing clear, accurate answers
- Citing sources appropriately

Provide comprehensive answers with citations.""",
            temperature=0.5,
        )
        self.register_agent(CrewAIAgent(qa_config, self.model))
        
        # Critic agent
        critic_config = AgentConfig(
            name="critic",
            role=AgentRole.CRITIC,
            system_prompt="""You are a critical evaluator. Your expertise is in:
- Identifying weaknesses in answers
- Checking factual accuracy
- Spotting logical inconsistencies
- Suggesting improvements

Provide constructive critique.""",
            temperature=0.2,
        )
        self.register_agent(CrewAIAgent(critic_config, self.model))
    
    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run CrewAI workflow.
        
        Args:
            task: Task with query and context
            
        Returns:
            Result with agent outputs
        """
        query = task.get("query", "")
        context = task.get("context", {})
        documents = context.get("documents", [])
        
        result = {
            "query": query,
            "agent_outputs": {},
            "final_answer": "",
            "metadata": {},
        }
        
        # Step 1: Research
        researcher = self.get_agent("researcher")
        if researcher:
            research_msg = AgentMessage(
                sender="system",
                recipient="researcher",
                content=f"Research information for: {query}\n\nDocuments: {self._format_docs(documents)}",
            )
            research_response = await researcher.process(research_msg)
            result["agent_outputs"]["research"] = research_response.content
        
        # Step 2: Answer
        qa = self.get_agent("qa_specialist")
        if qa:
            qa_msg = AgentMessage(
                sender="system",
                recipient="qa_specialist",
                content=f"Answer: {query}\n\nResearch: {result['agent_outputs'].get('research', '')}",
            )
            qa_response = await qa.process(qa_msg)
            result["agent_outputs"]["answer"] = qa_response.content
        
        # Step 3: Critique
        critic = self.get_agent("critic")
        if critic:
            critic_msg = AgentMessage(
                sender="system",
                recipient="critic",
                content=f"Evaluate this answer:\nQuery: {query}\nAnswer: {result['agent_outputs'].get('answer', '')}",
            )
            critic_response = await critic.process(critic_msg)
            result["agent_outputs"]["critique"] = critic_response.content
        
        # Final answer is the QA output (potentially refined based on critique)
        result["final_answer"] = result["agent_outputs"].get("answer", query)
        
        return result
    
    def _format_docs(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents for agents.
        
        Args:
            documents: List of documents
            
        Returns:
            Formatted string
        """
        if not documents:
            return "No documents provided"
        
        lines = []
        for idx, doc in enumerate(documents[:5], 1):
            text = doc.get("text", "")[:300]
            lines.append(f"[Doc {idx}] {text}...")
        
        return "\n".join(lines)
    
    def create_crew(self):
        """Create CrewAI crew (if CrewAI is available)."""
        if Crew is None or Task is None:
            logger.warning("CrewAI not available")
            return

        try:
            # Create tasks
            tasks = [
                Task(
                    description="Research the topic",
                    agent=self.agents["researcher"].crew_agent,
                ),
                Task(
                    description="Answer the question",
                    agent=self.agents["qa_specialist"].crew_agent,
                ),
                Task(
                    description="Critique the answer",
                    agent=self.agents["critic"].crew_agent,
                ),
            ]

            # Create crew
            self.crew = Crew(
                agents=[a.crew_agent for a in self.agents.values() if a.crew_agent],
                tasks=tasks,
                verbose=True,
            )

            logger.info("Created CrewAI crew")
        except Exception as e:
            logger.error(f"Failed to create crew: {e}")
