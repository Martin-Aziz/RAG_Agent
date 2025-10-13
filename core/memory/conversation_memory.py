"""Conversation memory management for multi-turn dialogue.

Implements short-term (session) and long-term (user) memory with
automatic summarization and token budget management.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import time
import json
import os


class ConversationTurn(BaseModel):
    """Single turn in a conversation."""
    user_message: str
    assistant_message: str
    timestamp: float
    metadata: Dict[str, Any] = {}


class ConversationState(BaseModel):
    """State for a single conversation session."""
    session_id: str
    user_id: str
    turns: List[ConversationTurn] = []
    summary: str = ""
    current_topic: str = ""
    created_at: float
    updated_at: float
    
    class Config:
        arbitrary_types_allowed = True


class UserProfile(BaseModel):
    """Long-term user profile across sessions."""
    user_id: str
    preferences: Dict[str, Any] = {}
    learned_facts: List[str] = []
    interaction_history: List[str] = []  # Session IDs
    total_queries: int = 0
    created_at: float
    updated_at: float


class ConversationMemoryManager:
    """Manages short-term and long-term conversation memory.
    
    Short-term: Last N turns per session with automatic summarization
    Long-term: User preferences and facts learned across sessions
    """
    
    def __init__(
        self,
        max_turns: int = 10,
        max_tokens: int = 4000,
        summarize_threshold: int = 7,
        storage_path: str = "data/memory"
    ):
        """
        Args:
            max_turns: Maximum turns to keep in short-term memory
            max_tokens: Token budget for conversation context
            summarize_threshold: Summarize when turns exceed this
            storage_path: Directory for persistent storage
        """
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.summarize_threshold = summarize_threshold
        self.storage_path = storage_path
        
        # In-memory caches
        self._sessions: Dict[str, ConversationState] = {}
        self._users: Dict[str, UserProfile] = {}
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
    
    def get_session(self, session_id: str, user_id: str = "default") -> ConversationState:
        """Get or create conversation session state."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        
        # Try to load from disk
        session_path = os.path.join(self.storage_path, f"session_{session_id}.json")
        if os.path.exists(session_path):
            try:
                with open(session_path, 'r') as f:
                    data = json.load(f)
                    state = ConversationState(**data)
                    self._sessions[session_id] = state
                    return state
            except Exception:
                pass
        
        # Create new session
        now = time.time()
        state = ConversationState(
            session_id=session_id,
            user_id=user_id,
            turns=[],
            summary="",
            created_at=now,
            updated_at=now
        )
        self._sessions[session_id] = state
        return state
    
    def get_user_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile."""
        if user_id in self._users:
            return self._users[user_id]
        
        # Try to load from disk
        user_path = os.path.join(self.storage_path, f"user_{user_id}.json")
        if os.path.exists(user_path):
            try:
                with open(user_path, 'r') as f:
                    data = json.load(f)
                    profile = UserProfile(**data)
                    self._users[user_id] = profile
                    return profile
            except Exception:
                pass
        
        # Create new profile
        now = time.time()
        profile = UserProfile(
            user_id=user_id,
            preferences={},
            learned_facts=[],
            interaction_history=[],
            total_queries=0,
            created_at=now,
            updated_at=now
        )
        self._users[user_id] = profile
        return profile
    
    def add_turn(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a conversation turn to session memory."""
        state = self.get_session(session_id, user_id)
        
        turn = ConversationTurn(
            user_message=user_message,
            assistant_message=assistant_message,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        state.turns.append(turn)
        state.updated_at = time.time()
        
        # Check if summarization needed
        if len(state.turns) > self.summarize_threshold:
            self._summarize_session(state)
        
        # Update user profile
        profile = self.get_user_profile(user_id)
        profile.total_queries += 1
        profile.updated_at = time.time()
        if session_id not in profile.interaction_history:
            profile.interaction_history.append(session_id)
        
        # Persist
        self.save_session(state)
        self.save_user_profile(profile)
    
    def build_context(
        self,
        session_id: str,
        user_id: str,
        include_user_profile: bool = True
    ) -> str:
        """Build conversation context for prompt.
        
        Returns formatted string with summary + recent turns + user context.
        """
        state = self.get_session(session_id, user_id)
        context_parts = []
        
        # Add user profile context
        if include_user_profile:
            profile = self.get_user_profile(user_id)
            if profile.preferences or profile.learned_facts:
                context_parts.append("User Profile:")
                if profile.preferences:
                    context_parts.append(f"Preferences: {json.dumps(profile.preferences)}")
                if profile.learned_facts:
                    context_parts.append(f"Known facts: {'; '.join(profile.learned_facts[:5])}")
                context_parts.append("")
        
        # Add conversation summary if exists
        if state.summary:
            context_parts.append("Conversation Summary:")
            context_parts.append(state.summary)
            context_parts.append("")
        
        # Add recent turns
        if state.turns:
            context_parts.append("Recent Conversation:")
            # Keep last N turns that fit in token budget
            recent_turns = state.turns[-self.max_turns:]
            for turn in recent_turns:
                context_parts.append(f"User: {turn.user_message}")
                context_parts.append(f"Assistant: {turn.assistant_message}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _summarize_session(self, state: ConversationState):
        """Summarize older conversation turns to free memory.
        
        In production, call LLM to generate summary. Here we use simple concatenation.
        """
        # Keep recent turns, summarize older ones
        if len(state.turns) <= self.summarize_threshold:
            return
        
        # Turns to summarize
        turns_to_summarize = state.turns[:self.summarize_threshold]
        state.turns = state.turns[self.summarize_threshold:]
        
        # Simple summary (in production, use LLM)
        summary_parts = []
        if state.summary:
            summary_parts.append(state.summary)
        
        for turn in turns_to_summarize:
            summary_parts.append(f"User asked about: {turn.user_message[:100]}")
        
        state.summary = " | ".join(summary_parts[-5:])  # Keep last 5 summaries
    
    def save_session(self, state: ConversationState):
        """Persist session state to disk."""
        session_path = os.path.join(self.storage_path, f"session_{state.session_id}.json")
        try:
            with open(session_path, 'w') as f:
                # Convert to dict for JSON serialization
                data = state.dict()
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save session {state.session_id}: {e}")
    
    def save_user_profile(self, profile: UserProfile):
        """Persist user profile to disk."""
        user_path = os.path.join(self.storage_path, f"user_{profile.user_id}.json")
        try:
            with open(user_path, 'w') as f:
                data = profile.dict()
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save user profile {profile.user_id}: {e}")
    
    def update_user_preference(self, user_id: str, key: str, value: Any):
        """Update user preference."""
        profile = self.get_user_profile(user_id)
        profile.preferences[key] = value
        profile.updated_at = time.time()
        self.save_user_profile(profile)
    
    def add_learned_fact(self, user_id: str, fact: str):
        """Add a learned fact about the user."""
        profile = self.get_user_profile(user_id)
        if fact not in profile.learned_facts:
            profile.learned_facts.append(fact)
            profile.updated_at = time.time()
            self.save_user_profile(profile)
    
    def clear_session(self, session_id: str):
        """Clear a session from memory and disk."""
        if session_id in self._sessions:
            del self._sessions[session_id]
        
        session_path = os.path.join(self.storage_path, f"session_{session_id}.json")
        if os.path.exists(session_path):
            try:
                os.remove(session_path)
            except Exception:
                pass
