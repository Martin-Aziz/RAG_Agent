"""Session memory for conversation summarization.

Maintains summarized conversation history within a session (1-hour window).
Implements:
- Automatic summarization of old turns
- Topic tracking and switching
- Semantic chunking of conversations
- Session persistence
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
import json

logger = logging.getLogger(__name__)


@dataclass
class ConversationSummary:
    """Summary of conversation segment."""
    summary_id: str
    session_id: str
    start_time: datetime
    end_time: datetime
    turn_count: int
    summary_text: str
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary_id": self.summary_id,
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "turn_count": self.turn_count,
            "summary_text": self.summary_text,
            "topics": self.topics,
            "entities": self.entities,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSummary":
        """Create from dictionary."""
        return cls(
            summary_id=data["summary_id"],
            session_id=data["session_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            turn_count=data["turn_count"],
            summary_text=data["summary_text"],
            topics=data.get("topics", []),
            entities=data.get("entities", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SessionState:
    """State of a conversation session."""
    session_id: str
    start_time: datetime
    last_activity: datetime
    summaries: List[ConversationSummary] = field(default_factory=list)
    current_topic: Optional[str] = None
    active_entities: Set[str] = field(default_factory=set)
    
    def is_expired(self, timeout: int = 3600) -> bool:
        """Check if session is expired.
        
        Args:
            timeout: Timeout in seconds (default 1 hour)
            
        Returns:
            True if expired
        """
        elapsed = (datetime.now() - self.last_activity).total_seconds()
        return elapsed > timeout


class SessionMemory:
    """Manages session-level conversation memory."""
    
    def __init__(
        self,
        model_adapter=None,
        summary_trigger_turns: int = 10,
        session_timeout: int = 3600,
        max_summaries: int = 10,
    ):
        """Initialize session memory.
        
        Args:
            model_adapter: LLM for summarization
            summary_trigger_turns: Trigger summarization after N turns
            session_timeout: Session timeout in seconds
            max_summaries: Maximum summaries to keep per session
        """
        self.model = model_adapter
        self.summary_trigger_turns = summary_trigger_turns
        self.session_timeout = session_timeout
        self.max_summaries = max_summaries
        
        # Session ID -> SessionState
        self.sessions: Dict[str, SessionState] = {}
    
    def get_or_create_session(self, session_id: str) -> SessionState:
        """Get or create session state.
        
        Args:
            session_id: Session ID
            
        Returns:
            SessionState
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(
                session_id=session_id,
                start_time=datetime.now(),
                last_activity=datetime.now(),
            )
        else:
            # Update last activity
            self.sessions[session_id].last_activity = datetime.now()
        
        return self.sessions[session_id]
    
    async def add_turns_and_summarize(
        self,
        session_id: str,
        turns: List[Any],
        force_summarize: bool = False,
    ) -> Optional[ConversationSummary]:
        """Add turns and trigger summarization if needed.
        
        Args:
            session_id: Session ID
            turns: List of Turn objects
            force_summarize: Force summarization even if threshold not met
            
        Returns:
            ConversationSummary if summarization occurred
        """
        session = self.get_or_create_session(session_id)
        
        # Check if we should summarize
        should_summarize = (
            force_summarize or
            len(turns) >= self.summary_trigger_turns
        )
        
        if not should_summarize:
            return None
        
        # Generate summary
        summary = await self._generate_summary(session_id, turns)
        
        if summary:
            session.summaries.append(summary)
            
            # Prune old summaries
            if len(session.summaries) > self.max_summaries:
                session.summaries = session.summaries[-self.max_summaries:]
            
            logger.info(
                f"Created summary for session {session_id}: "
                f"{summary.turn_count} turns, {len(session.summaries)} total summaries"
            )
        
        return summary
    
    async def _generate_summary(
        self,
        session_id: str,
        turns: List[Any],
    ) -> Optional[ConversationSummary]:
        """Generate summary of conversation turns.
        
        Args:
            session_id: Session ID
            turns: List of turns to summarize
            
        Returns:
            ConversationSummary
        """
        if not self.model or not turns:
            return None
        
        try:
            # Prepare conversation text
            conv_text = "\n".join([
                f"User: {t.user_message}\nAssistant: {t.assistant_message}"
                for t in turns
            ])
            
            # Generate summary prompt
            prompt = f"""Summarize the following conversation. Include:
1. Main topics discussed
2. Key entities mentioned (people, places, organizations)
3. Important conclusions or decisions

Conversation:
{conv_text}

Summary (JSON format):
{{
  "summary": "...",
  "topics": [...],
  "entities": [...]
}}
"""
            
            # Generate summary
            response = await self.model.generate(prompt, temperature=0.3, max_tokens=300)
            
            # Parse response
            summary_data = self._parse_summary_response(response)
            
            # Create summary object
            summary = ConversationSummary(
                summary_id=f"{session_id}_{len(self.sessions[session_id].summaries)}",
                session_id=session_id,
                start_time=turns[0].timestamp,
                end_time=turns[-1].timestamp,
                turn_count=len(turns),
                summary_text=summary_data.get("summary", ""),
                topics=summary_data.get("topics", []),
                entities=summary_data.get("entities", []),
            )
            
            # Update session state
            if summary.topics:
                self.sessions[session_id].current_topic = summary.topics[0]
            self.sessions[session_id].active_entities.update(summary.entities)
            
            return summary
            
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return None
    
    def _parse_summary_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM summary response.
        
        Args:
            response: LLM response
            
        Returns:
            Parsed summary data
        """
        try:
            # Try to extract JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data
        except Exception as e:
            logger.error(f"Summary parse error: {e}")
        
        # Fallback: use raw response as summary
        return {
            "summary": response[:500],
            "topics": [],
            "entities": [],
        }
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get full session context (all summaries).
        
        Args:
            session_id: Session ID
            
        Returns:
            Dictionary with session context
        """
        session = self.get_or_create_session(session_id)
        
        return {
            "session_id": session_id,
            "start_time": session.start_time.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "summaries": [s.to_dict() for s in session.summaries],
            "current_topic": session.current_topic,
            "active_entities": list(session.active_entities),
            "is_expired": session.is_expired(self.session_timeout),
        }
    
    def detect_topic_switch(
        self,
        session_id: str,
        new_query: str,
        threshold: float = 0.3,
    ) -> Tuple[bool, Optional[str]]:
        """Detect if conversation topic has switched.
        
        Args:
            session_id: Session ID
            new_query: New query to check
            threshold: Similarity threshold for topic switch
            
        Returns:
            (is_switch, new_topic)
        """
        session = self.get_or_create_session(session_id)
        
        if not session.current_topic:
            # No previous topic
            return False, None
        
        # Simple topic switch detection: check term overlap
        current_terms = set(session.current_topic.lower().split())
        query_terms = set(new_query.lower().split())
        
        if not current_terms or not query_terms:
            return False, None
        
        overlap = len(current_terms & query_terms)
        similarity = overlap / len(query_terms)
        
        if similarity < threshold:
            # Topic switched
            new_topic = " ".join(list(query_terms)[:5])  # First 5 terms
            logger.info(
                f"Topic switch detected in session {session_id}: "
                f"'{session.current_topic}' -> '{new_topic}'"
            )
            return True, new_topic
        
        return False, None
    
    def get_relevant_summaries(
        self,
        session_id: str,
        query: str,
        max_summaries: int = 3,
    ) -> List[ConversationSummary]:
        """Get relevant summaries for a query.
        
        Args:
            session_id: Session ID
            query: Query string
            max_summaries: Maximum summaries to return
            
        Returns:
            List of relevant summaries
        """
        session = self.get_or_create_session(session_id)
        
        if not session.summaries:
            return []
        
        # Score summaries by relevance
        query_terms = set(query.lower().split())
        scored_summaries = []
        
        for summary in session.summaries:
            # Check overlap with summary text and topics
            summary_terms = set(summary.summary_text.lower().split())
            topic_terms = set(" ".join(summary.topics).lower().split())
            all_terms = summary_terms | topic_terms
            
            overlap = len(query_terms & all_terms)
            score = overlap / len(query_terms) if query_terms else 0
            
            scored_summaries.append((summary, score))
        
        # Sort by score
        scored_summaries.sort(key=lambda x: x[1], reverse=True)
        
        # Return top summaries
        return [s for s, _ in scored_summaries[:max_summaries]]
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        expired = [
            sid for sid, session in self.sessions.items()
            if session.is_expired(self.session_timeout)
        ]
        
        for sid in expired:
            del self.sessions[sid]
            logger.info(f"Removed expired session: {sid}")
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get session memory statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_summaries = sum(len(s.summaries) for s in self.sessions.values())
        
        return {
            "active_sessions": len(self.sessions),
            "total_summaries": total_summaries,
            "avg_summaries_per_session": (
                total_summaries / len(self.sessions) if self.sessions else 0
            ),
        }
