"""Memory manager integrating all memory tiers.

Coordinates:
- Short-term (working context)
- Session (conversation summaries)
- Long-term (user profiles)

Provides unified interface for memory operations.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime

from .short_term import ShortTermMemory, Turn, WorkingContext
from .session import SessionMemory, ConversationSummary
from .long_term import LongTermMemory, UserProfile

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory manager."""
    # Short-term settings
    short_term_max_turns: int = 5
    short_term_max_tokens: int = 2000
    relevance_decay: float = 0.9
    
    # Session settings
    summary_trigger_turns: int = 10
    session_timeout: int = 3600
    max_summaries: int = 10
    
    # Long-term settings
    storage_path: str = "./data/user_profiles"
    auto_save: bool = True
    
    # Feature flags
    enable_summarization: bool = True
    enable_personalization: bool = True
    enable_topic_tracking: bool = True


class MemoryManager:
    """Unified memory manager for all memory tiers."""
    
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        model_adapter=None,
    ):
        """Initialize memory manager.
        
        Args:
            config: MemoryConfig or None for defaults
            model_adapter: LLM for summarization
        """
        self.config = config or MemoryConfig()
        self.model = model_adapter
        
        # Initialize memory tiers
        self.short_term = ShortTermMemory(
            max_turns=self.config.short_term_max_turns,
            max_tokens=self.config.short_term_max_tokens,
            relevance_decay=self.config.relevance_decay,
        )
        
        self.session = SessionMemory(
            model_adapter=model_adapter,
            summary_trigger_turns=self.config.summary_trigger_turns,
            session_timeout=self.config.session_timeout,
            max_summaries=self.config.max_summaries,
        )
        
        self.long_term = LongTermMemory(
            storage_path=self.config.storage_path,
            auto_save=self.config.auto_save,
        )
        
        logger.info("Memory manager initialized with all tiers")
    
    async def add_interaction(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add interaction to all memory tiers.
        
        Args:
            user_id: User ID
            session_id: Session ID
            user_message: User's message
            assistant_message: Assistant's response
            metadata: Optional metadata (topics, entities, etc.)
            
        Returns:
            Dictionary with memory updates
        """
        result = {
            "turn_added": False,
            "summary_created": False,
            "profile_updated": False,
        }
        
        # Add to short-term memory
        turn = self.short_term.add_turn(
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
            metadata=metadata,
        )
        result["turn_added"] = True
        
        # Get all turns for session
        context = self.short_term.get_context(session_id)
        
        # Check if we should summarize
        if self.config.enable_summarization and len(context.turns) >= self.config.summary_trigger_turns:
            summary = await self.session.add_turns_and_summarize(
                session_id=session_id,
                turns=context.turns,
            )
            if summary:
                result["summary_created"] = True
                result["summary"] = summary.to_dict()
        
        # Update long-term profile
        if self.config.enable_personalization:
            topics = metadata.get("topics", []) if metadata else []
            entities = metadata.get("entities", []) if metadata else []
            
            self.long_term.update_interaction(
                user_id=user_id,
                query=user_message,
                topics=topics,
                entities=entities,
            )
            result["profile_updated"] = True
        
        return result
    
    async def get_context(
        self,
        user_id: str,
        session_id: str,
        current_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get full context across all memory tiers.
        
        Args:
            user_id: User ID
            session_id: Session ID
            current_query: Optional current query for relevance
            
        Returns:
            Dictionary with context from all tiers
        """
        context = {
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Short-term context (working memory)
        working_context = self.short_term.get_context(session_id, current_query)
        context["short_term"] = {
            "turns": [t.to_dict() for t in working_context.turns],
            "token_count": working_context.current_tokens,
        }
        
        # Session context (summaries)
        if current_query:
            relevant_summaries = self.session.get_relevant_summaries(
                session_id=session_id,
                query=current_query,
                max_summaries=3,
            )
        else:
            session_state = self.session.get_or_create_session(session_id)
            relevant_summaries = session_state.summaries[-3:]  # Last 3 summaries
        
        context["session"] = {
            "summaries": [s.to_dict() for s in relevant_summaries],
            "current_topic": self.session.sessions.get(session_id, None),
        }
        
        # Long-term context (user profile)
        if self.config.enable_personalization:
            user_context = self.long_term.get_user_context(user_id)
            context["long_term"] = user_context
        
        return context
    
    async def detect_topic_switch(
        self,
        user_id: str,
        session_id: str,
        new_query: str,
    ) -> Dict[str, Any]:
        """Detect topic switch and update memory accordingly.
        
        Args:
            user_id: User ID
            session_id: Session ID
            new_query: New query
            
        Returns:
            Dictionary with topic switch info
        """
        if not self.config.enable_topic_tracking:
            return {"topic_switched": False}
        
        is_switch, new_topic = self.session.detect_topic_switch(
            session_id=session_id,
            new_query=new_query,
        )
        
        result = {
            "topic_switched": is_switch,
            "new_topic": new_topic,
        }
        
        if is_switch:
            # Clear short-term memory on topic switch
            self.short_term.clear_session(session_id)
            logger.info(f"Topic switch: cleared short-term memory for {session_id}")
            
            result["action"] = "cleared_short_term_memory"
        
        return result
    
    def set_user_preference(
        self,
        user_id: str,
        key: str,
        value: Any,
    ):
        """Set user preference.
        
        Args:
            user_id: User ID
            key: Preference key
            value: Preference value
        """
        self.long_term.set_preference(user_id, key, value)
    
    def get_user_preference(
        self,
        user_id: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """Get user preference.
        
        Args:
            user_id: User ID
            key: Preference key
            default: Default value
            
        Returns:
            Preference value
        """
        return self.long_term.get_preference(user_id, key, default)
    
    async def end_session(
        self,
        user_id: str,
        session_id: str,
    ):
        """End session and finalize memory.
        
        Args:
            user_id: User ID
            session_id: Session ID
        """
        # Get final turn count
        context = self.short_term.get_context(session_id)
        turn_count = len(context.turns)
        
        # Update long-term with session stats
        self.long_term.update_session(user_id, turn_count)
        
        # Force summarization if enabled
        if self.config.enable_summarization and turn_count > 0:
            await self.session.add_turns_and_summarize(
                session_id=session_id,
                turns=context.turns,
                force_summarize=True,
            )
        
        # Clear short-term memory
        self.short_term.clear_session(session_id)
        
        logger.info(f"Session {session_id} ended: {turn_count} turns")
    
    def record_feedback(
        self,
        user_id: str,
        is_positive: bool,
    ):
        """Record user feedback.
        
        Args:
            user_id: User ID
            is_positive: True for positive, False for negative
        """
        self.long_term.record_feedback(user_id, is_positive)
    
    def cleanup(self):
        """Cleanup expired sessions and save state."""
        # Cleanup expired sessions
        self.session.cleanup_expired_sessions()
        
        # Save all profiles
        self.long_term.save_all_profiles()
        
        logger.info("Memory cleanup complete")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from all memory tiers.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "short_term": self.short_term.get_statistics(),
            "session": self.session.get_statistics(),
            "long_term": self.long_term.get_statistics(),
        }
    
    @classmethod
    def from_config(cls, config_dict: Dict[str, Any], model_adapter=None) -> "MemoryManager":
        """Create memory manager from configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary
            model_adapter: LLM adapter
            
        Returns:
            MemoryManager instance
        """
        config = MemoryConfig(
            short_term_max_turns=config_dict.get("short_term_max_turns", 5),
            short_term_max_tokens=config_dict.get("short_term_max_tokens", 2000),
            relevance_decay=config_dict.get("relevance_decay", 0.9),
            summary_trigger_turns=config_dict.get("summary_trigger_turns", 10),
            session_timeout=config_dict.get("session_timeout", 3600),
            max_summaries=config_dict.get("max_summaries", 10),
            storage_path=config_dict.get("storage_path", "./data/user_profiles"),
            auto_save=config_dict.get("auto_save", True),
            enable_summarization=config_dict.get("enable_summarization", True),
            enable_personalization=config_dict.get("enable_personalization", True),
            enable_topic_tracking=config_dict.get("enable_topic_tracking", True),
        )
        
        return cls(config=config, model_adapter=model_adapter)
