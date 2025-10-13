"""Long-term memory for user profiles and preferences.

Maintains persistent user information across sessions:
- User preferences and settings
- Historical interaction patterns
- Learned user interests
- Personalization data
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """User profile with preferences and history."""
    user_id: str
    created_at: datetime
    last_active: datetime
    
    # Preferences
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    # Interests and topics
    interests: List[str] = field(default_factory=list)
    frequent_topics: Dict[str, int] = field(default_factory=dict)
    
    # Interaction history
    total_queries: int = 0
    total_sessions: int = 0
    avg_session_length: float = 0.0
    
    # Entities user frequently asks about
    frequent_entities: Dict[str, int] = field(default_factory=dict)
    
    # Feedback and corrections
    positive_feedback_count: int = 0
    negative_feedback_count: int = 0
    
    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "preferences": self.preferences,
            "interests": self.interests,
            "frequent_topics": self.frequent_topics,
            "total_queries": self.total_queries,
            "total_sessions": self.total_sessions,
            "avg_session_length": self.avg_session_length,
            "frequent_entities": self.frequent_entities,
            "positive_feedback_count": self.positive_feedback_count,
            "negative_feedback_count": self.negative_feedback_count,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active=datetime.fromisoformat(data["last_active"]),
            preferences=data.get("preferences", {}),
            interests=data.get("interests", []),
            frequent_topics=data.get("frequent_topics", {}),
            total_queries=data.get("total_queries", 0),
            total_sessions=data.get("total_sessions", 0),
            avg_session_length=data.get("avg_session_length", 0.0),
            frequent_entities=data.get("frequent_entities", {}),
            positive_feedback_count=data.get("positive_feedback_count", 0),
            negative_feedback_count=data.get("negative_feedback_count", 0),
            metadata=data.get("metadata", {}),
        )


class LongTermMemory:
    """Manages long-term user profiles and preferences."""
    
    def __init__(
        self,
        storage_path: str = "./data/user_profiles",
        auto_save: bool = True,
    ):
        """Initialize long-term memory.
        
        Args:
            storage_path: Path to store user profiles
            auto_save: Automatically save profiles after updates
        """
        self.storage_path = storage_path
        self.auto_save = auto_save
        
        # User ID -> UserProfile
        self.profiles: Dict[str, UserProfile] = {}
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        
        # Load existing profiles
        self._load_profiles()
    
    def _load_profiles(self):
        """Load profiles from disk."""
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.storage_path, filename)
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        profile = UserProfile.from_dict(data)
                        self.profiles[profile.user_id] = profile
            
            logger.info(f"Loaded {len(self.profiles)} user profiles")
            
        except Exception as e:
            logger.error(f"Failed to load profiles: {e}")
    
    def _save_profile(self, profile: UserProfile):
        """Save profile to disk.
        
        Args:
            profile: UserProfile to save
        """
        try:
            filepath = os.path.join(self.storage_path, f"{profile.user_id}.json")
            with open(filepath, "w") as f:
                json.dump(profile.to_dict(), f, indent=2)
            
            logger.debug(f"Saved profile for user {profile.user_id}")
            
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
    
    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile.
        
        Args:
            user_id: User ID
            
        Returns:
            UserProfile
        """
        if user_id not in self.profiles:
            self.profiles[user_id] = UserProfile(
                user_id=user_id,
                created_at=datetime.now(),
                last_active=datetime.now(),
            )
            
            if self.auto_save:
                self._save_profile(self.profiles[user_id])
            
            logger.info(f"Created new profile for user {user_id}")
        else:
            # Update last active
            self.profiles[user_id].last_active = datetime.now()
        
        return self.profiles[user_id]
    
    def update_interaction(
        self,
        user_id: str,
        query: str,
        topics: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
    ):
        """Update profile with interaction data.
        
        Args:
            user_id: User ID
            query: Query string
            topics: Extracted topics
            entities: Extracted entities
        """
        profile = self.get_or_create_profile(user_id)
        
        # Update query count
        profile.total_queries += 1
        
        # Update topics
        if topics:
            for topic in topics:
                topic_lower = topic.lower()
                profile.frequent_topics[topic_lower] = (
                    profile.frequent_topics.get(topic_lower, 0) + 1
                )
        
        # Update entities
        if entities:
            for entity in entities:
                entity_lower = entity.lower()
                profile.frequent_entities[entity_lower] = (
                    profile.frequent_entities.get(entity_lower, 0) + 1
                )
        
        # Update interests (top topics)
        if profile.frequent_topics:
            top_topics = sorted(
                profile.frequent_topics.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            profile.interests = [topic for topic, _ in top_topics]
        
        if self.auto_save:
            self._save_profile(profile)
    
    def update_session(self, user_id: str, session_length: int):
        """Update profile with session data.
        
        Args:
            user_id: User ID
            session_length: Session length in turns
        """
        profile = self.get_or_create_profile(user_id)
        
        # Update session count
        profile.total_sessions += 1
        
        # Update average session length
        total_length = profile.avg_session_length * (profile.total_sessions - 1)
        profile.avg_session_length = (total_length + session_length) / profile.total_sessions
        
        if self.auto_save:
            self._save_profile(profile)
    
    def record_feedback(
        self,
        user_id: str,
        is_positive: bool,
    ):
        """Record user feedback.
        
        Args:
            user_id: User ID
            is_positive: True for positive feedback, False for negative
        """
        profile = self.get_or_create_profile(user_id)
        
        if is_positive:
            profile.positive_feedback_count += 1
        else:
            profile.negative_feedback_count += 1
        
        if self.auto_save:
            self._save_profile(profile)
    
    def set_preference(
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
        profile = self.get_or_create_profile(user_id)
        profile.preferences[key] = value
        
        if self.auto_save:
            self._save_profile(profile)
        
        logger.info(f"Set preference {key}={value} for user {user_id}")
    
    def get_preference(
        self,
        user_id: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """Get user preference.
        
        Args:
            user_id: User ID
            key: Preference key
            default: Default value if not set
            
        Returns:
            Preference value
        """
        profile = self.get_or_create_profile(user_id)
        return profile.preferences.get(key, default)
    
    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get full user context for personalization.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with user context
        """
        profile = self.get_or_create_profile(user_id)
        
        return {
            "user_id": user_id,
            "interests": profile.interests[:5],  # Top 5 interests
            "frequent_topics": dict(
                sorted(
                    profile.frequent_topics.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ),
            "frequent_entities": dict(
                sorted(
                    profile.frequent_entities.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ),
            "preferences": profile.preferences,
            "interaction_stats": {
                "total_queries": profile.total_queries,
                "total_sessions": profile.total_sessions,
                "avg_session_length": profile.avg_session_length,
                "positive_feedback": profile.positive_feedback_count,
                "negative_feedback": profile.negative_feedback_count,
            },
        }
    
    def get_similar_users(
        self,
        user_id: str,
        max_users: int = 5,
    ) -> List[str]:
        """Find users with similar interests.
        
        Args:
            user_id: User ID
            max_users: Maximum similar users to return
            
        Returns:
            List of similar user IDs
        """
        profile = self.get_or_create_profile(user_id)
        
        if not profile.interests:
            return []
        
        user_interests = set(profile.interests)
        
        # Score other users by interest overlap
        scored_users = []
        for other_id, other_profile in self.profiles.items():
            if other_id == user_id:
                continue
            
            other_interests = set(other_profile.interests)
            overlap = len(user_interests & other_interests)
            
            if overlap > 0:
                score = overlap / len(user_interests)
                scored_users.append((other_id, score))
        
        # Sort by score
        scored_users.sort(key=lambda x: x[1], reverse=True)
        
        return [uid for uid, _ in scored_users[:max_users]]
    
    def save_all_profiles(self):
        """Save all profiles to disk."""
        for profile in self.profiles.values():
            self._save_profile(profile)
        
        logger.info(f"Saved {len(self.profiles)} profiles")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get long-term memory statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.profiles:
            return {
                "total_users": 0,
                "total_queries": 0,
                "total_sessions": 0,
            }
        
        total_queries = sum(p.total_queries for p in self.profiles.values())
        total_sessions = sum(p.total_sessions for p in self.profiles.values())
        
        return {
            "total_users": len(self.profiles),
            "total_queries": total_queries,
            "total_sessions": total_sessions,
            "avg_queries_per_user": total_queries / len(self.profiles),
            "avg_sessions_per_user": total_sessions / len(self.profiles),
        }
