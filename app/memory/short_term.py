"""Short-term memory for working context.

Maintains immediate conversation context (last N turns).
Implements:
- Turn management (add/remove/prune)
- Relevance scoring
- Token budget management
- Context window optimization
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """Represents a single conversation turn."""
    turn_id: str
    user_message: str
    assistant_message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Computed attributes
    token_count: int = 0
    relevance_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "turn_id": self.turn_id,
            "user_message": self.user_message,
            "assistant_message": self.assistant_message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "token_count": self.token_count,
            "relevance_score": self.relevance_score,
        }


@dataclass
class WorkingContext:
    """Current working context (active turns)."""
    turns: List[Turn] = field(default_factory=list)
    max_turns: int = 5
    max_tokens: int = 2000
    current_tokens: int = 0
    
    def add_turn(self, turn: Turn):
        """Add turn to working context."""
        self.turns.append(turn)
        self.current_tokens += turn.token_count
        
        # Prune if needed
        self._prune_if_needed()
    
    def _prune_if_needed(self):
        """Prune old turns if limits exceeded."""
        # Prune by turn count
        while len(self.turns) > self.max_turns:
            removed = self.turns.pop(0)
            self.current_tokens -= removed.token_count
        
        # Prune by token count
        while self.current_tokens > self.max_tokens and self.turns:
            removed = self.turns.pop(0)
            self.current_tokens -= removed.token_count
    
    def get_context_string(self) -> str:
        """Get formatted context string."""
        lines = []
        for turn in self.turns:
            lines.append(f"User: {turn.user_message}")
            lines.append(f"Assistant: {turn.assistant_message}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "turns": [t.to_dict() for t in self.turns],
            "max_turns": self.max_turns,
            "max_tokens": self.max_tokens,
            "current_tokens": self.current_tokens,
        }


class ShortTermMemory:
    """Manages short-term working context."""
    
    def __init__(
        self,
        max_turns: int = 5,
        max_tokens: int = 2000,
        relevance_decay: float = 0.9,
    ):
        """Initialize short-term memory.
        
        Args:
            max_turns: Maximum number of turns to keep
            max_tokens: Maximum token budget
            relevance_decay: Decay factor for older turns
        """
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.relevance_decay = relevance_decay
        
        # Session ID -> WorkingContext
        self.contexts: Dict[str, WorkingContext] = {}
    
    def get_or_create_context(self, session_id: str) -> WorkingContext:
        """Get or create working context for session.
        
        Args:
            session_id: Session ID
            
        Returns:
            WorkingContext
        """
        if session_id not in self.contexts:
            self.contexts[session_id] = WorkingContext(
                max_turns=self.max_turns,
                max_tokens=self.max_tokens,
            )
        return self.contexts[session_id]
    
    def add_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Turn:
        """Add a turn to short-term memory.
        
        Args:
            session_id: Session ID
            user_message: User's message
            assistant_message: Assistant's response
            metadata: Optional metadata
            
        Returns:
            Created Turn object
        """
        context = self.get_or_create_context(session_id)
        
        turn = Turn(
            turn_id=f"{session_id}_{len(context.turns)}",
            user_message=user_message,
            assistant_message=assistant_message,
            timestamp=datetime.now(),
            metadata=metadata or {},
            token_count=self._estimate_tokens(user_message, assistant_message),
            relevance_score=1.0,
        )
        
        context.add_turn(turn)
        
        logger.info(
            f"Added turn to session {session_id}: "
            f"{len(context.turns)} turns, {context.current_tokens} tokens"
        )
        
        return turn
    
    def get_context(
        self,
        session_id: str,
        current_query: Optional[str] = None,
    ) -> WorkingContext:
        """Get working context for session.
        
        Args:
            session_id: Session ID
            current_query: Optional current query for relevance scoring
            
        Returns:
            WorkingContext
        """
        context = self.get_or_create_context(session_id)
        
        # Update relevance scores if we have current query
        if current_query:
            self._update_relevance(context, current_query)
        
        # Apply decay to older turns
        self._apply_decay(context)
        
        return context
    
    def _estimate_tokens(self, user_msg: str, assistant_msg: str) -> int:
        """Estimate token count (rough approximation).
        
        Args:
            user_msg: User message
            assistant_msg: Assistant message
            
        Returns:
            Estimated token count
        """
        # Rough estimate: 1 token ~= 4 characters
        total_chars = len(user_msg) + len(assistant_msg)
        return total_chars // 4
    
    def _update_relevance(self, context: WorkingContext, query: str):
        """Update relevance scores based on current query.
        
        Args:
            context: Working context
            query: Current query
        """
        query_terms = set(query.lower().split())
        
        for turn in context.turns:
            # Check overlap with user message
            turn_terms = set(turn.user_message.lower().split())
            overlap = len(query_terms & turn_terms)
            
            # Score based on overlap
            if query_terms:
                turn.relevance_score = overlap / len(query_terms)
            else:
                turn.relevance_score = 0.5
    
    def _apply_decay(self, context: WorkingContext):
        """Apply temporal decay to relevance scores.
        
        Args:
            context: Working context
        """
        for i, turn in enumerate(context.turns):
            # More recent turns get higher scores
            position_factor = (i + 1) / len(context.turns)
            turn.relevance_score *= position_factor * self.relevance_decay
    
    def get_relevant_turns(
        self,
        session_id: str,
        query: str,
        min_relevance: float = 0.3,
        max_turns: Optional[int] = None,
    ) -> List[Turn]:
        """Get relevant turns for a query.
        
        Args:
            session_id: Session ID
            query: Query string
            min_relevance: Minimum relevance threshold
            max_turns: Maximum turns to return
            
        Returns:
            List of relevant turns
        """
        context = self.get_context(session_id, query)
        
        # Filter by relevance
        relevant = [t for t in context.turns if t.relevance_score >= min_relevance]
        
        # Sort by relevance
        relevant.sort(key=lambda t: t.relevance_score, reverse=True)
        
        # Limit count
        if max_turns:
            relevant = relevant[:max_turns]
        
        return relevant
    
    def clear_session(self, session_id: str):
        """Clear short-term memory for session.
        
        Args:
            session_id: Session ID
        """
        if session_id in self.contexts:
            del self.contexts[session_id]
            logger.info(f"Cleared short-term memory for session {session_id}")
    
    def get_all_sessions(self) -> List[str]:
        """Get all active session IDs.
        
        Returns:
            List of session IDs
        """
        return list(self.contexts.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_turns = sum(len(ctx.turns) for ctx in self.contexts.values())
        total_tokens = sum(ctx.current_tokens for ctx in self.contexts.values())
        
        return {
            "active_sessions": len(self.contexts),
            "total_turns": total_turns,
            "total_tokens": total_tokens,
            "avg_turns_per_session": total_turns / len(self.contexts) if self.contexts else 0,
        }
